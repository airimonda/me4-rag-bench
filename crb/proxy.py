"""Local OpenAI-compatible token proxy (see SPEC "Token proxy").

Students point their LLM client at http://127.0.0.1:8790/v1. Requests are forwarded to CRB_UPSTREAM
with the Authorization header untouched, `"usage": {"include": true}` is added so OpenRouter reports
cost, streamed (SSE) replies are passed through chunk by chunk, and one line per call goes to
proxy.jsonl with the question that was active. Keys and prompts are never logged.
"""
from __future__ import annotations

import codecs
import http.client
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import DEFAULT_PROXY_PORT, PROXY_HOST, upstream as default_upstream

_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding", "content-encoding"}


class _UsageTap:
    """Pulls model + usage out of a JSON body or the SSE lines of a stream, keeping almost nothing."""

    def __init__(self) -> None:
        self.model: str | None = None
        self.usage: dict | None = None
        self._tail = ""

    def feed_sse(self, text: str) -> None:
        self._tail += text
        *lines, self._tail = self._tail.split("\n")
        for line in lines:
            self._line(line)

    def finish_sse(self) -> None:
        self._line(self._tail)
        self._tail = ""

    def _line(self, line: str) -> None:
        line = line.strip()
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if not data or data == "[DONE]" or ('"usage"' not in data and '"model"' not in data):
            return
        try:
            self.feed_obj(json.loads(data))
        except ValueError:
            pass

    def feed_obj(self, obj) -> None:
        if not isinstance(obj, dict):
            return
        if isinstance(obj.get("model"), str):
            self.model = obj["model"]
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class TokenProxy:
    """Start with .start(); read `.port` (use port=0 for a free one); stop with .stop()."""

    def __init__(self, state, log_path: Path, port: int = DEFAULT_PROXY_PORT, upstream_url: str | None = None,
                 host: str = PROXY_HOST) -> None:
        self.state = state
        self.log_path = Path(log_path)
        self.upstream = (upstream_url or default_upstream()).rstrip("/")
        self._log_lock = threading.Lock()
        proxy = self

        class Handler(_Handler):
            owner = proxy

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> "TokenProxy":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2},
                                        daemon=True, name="crb-proxy")
        self._thread.start()
        return self

    def stop(self) -> None:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:  # noqa: BLE001
            pass

    def log(self, rec: dict) -> None:
        try:
            with self._log_lock, open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


class _Handler(BaseHTTPRequestHandler):
    owner: TokenProxy
    protocol_version = "HTTP/1.1"
    server_version = "crb-proxy"

    def log_message(self, *args) -> None:  # silence default stderr logging
        pass

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = lambda self: self._forward()

    # -- helpers ---------------------------------------------------------------------------------
    def _read_body(self) -> bytes:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = b""
            while True:
                size = int((self.rfile.readline() or b"0").split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return out
                out += self.rfile.read(size)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send_json(self, status: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _prepare(self, body: bytes) -> tuple[bytes, str | None, bool]:
        """Add usage accounting to JSON request bodies. Returns (body, model, wants_stream)."""
        if not body or "json" not in (self.headers.get("Content-Type") or "json").lower():
            return body, None, False
        try:
            obj = json.loads(body)
        except ValueError:
            return body, None, False
        if not isinstance(obj, dict):
            return body, None, False
        usage = obj.get("usage")
        obj["usage"] = {**usage, "include": True} if isinstance(usage, dict) else {"include": True}
        stream = bool(obj.get("stream"))
        if stream and "stream_options" not in obj:
            obj["stream_options"] = {"include_usage": True}
        model = obj.get("model") if isinstance(obj.get("model"), str) else None
        return json.dumps(obj).encode("utf-8"), model, stream

    # -- the forwarding itself -------------------------------------------------------------------
    def _forward(self) -> None:
        own = self.owner
        route = urlparse(self.path)
        if route.path in ("/", "/health"):
            self._send_json(200, {"ok": True, "crb_proxy": True, "upstream": own.upstream})
            return
        try:
            body = self._read_body()
        except (ValueError, OSError):
            self._send_json(400, {"error": "crb proxy could not read the request body"})
            return
        body, req_model, _ = self._prepare(body)
        rest = route.path[3:] if route.path.startswith("/v1") else route.path
        up = urlparse(own.upstream)
        target = up.path.rstrip("/") + rest + (f"?{route.query}" if route.query else "")
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
        headers["Accept-Encoding"] = "identity"
        headers["Host"] = up.netloc
        if body:
            headers["Content-Length"] = str(len(body))

        rec = {"t": round(own.state.t(), 2), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "qid": own.state.qid, "axis": own.state.axis, "phase": own.state.phase,
               "method": self.command, "path": route.path, "model": req_model, "status": None,
               "stream": False, "prompt_tokens": None, "completion_tokens": None, "cost": None,
               "latency_ms": None, "error": None}
        t0 = time.perf_counter()
        tap = _UsageTap()
        conn = None
        try:
            port = up.port or (443 if up.scheme == "https" else 80)
            if up.scheme == "https":
                conn = http.client.HTTPSConnection(up.hostname, port, timeout=600, context=ssl.create_default_context())
            else:
                conn = http.client.HTTPConnection(up.hostname, port, timeout=600)
            conn.request(self.command, target, body=body or None, headers=headers)
            resp = conn.getresponse()
            rec["status"] = resp.status
            out_headers = [(k, v) for k, v in resp.getheaders() if k.lower() not in _HOP]
            ctype = (resp.getheader("Content-Type") or "").lower()
            if "event-stream" in ctype:
                rec["stream"] = True
                self.send_response(resp.status)
                for k, v in out_headers:
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    tap.feed_sse(dec.decode(chunk))
                tap.finish_sse()
            else:
                data = resp.read()
                self.send_response(resp.status)
                for k, v in out_headers:
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                if "json" in ctype:
                    try:
                        tap.feed_obj(json.loads(data))
                    except ValueError:
                        pass
        except (BrokenPipeError, ConnectionResetError):
            rec["error"] = "the client closed the connection early"
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"could not reach the upstream LLM ({type(exc).__name__}: {exc})"[:200]
            if rec["status"] is None:
                rec["status"] = 502
                try:
                    self._send_json(502, {"error": {"message": rec["error"], "type": "crb_proxy_error"}})
                except OSError:
                    pass
            self.close_connection = True
        finally:
            if conn is not None:
                conn.close()
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if tap.model:
            rec["model"] = tap.model
        if tap.usage:
            u = tap.usage
            rec["prompt_tokens"] = _num(u.get("prompt_tokens", u.get("input_tokens")))
            rec["completion_tokens"] = _num(u.get("completion_tokens", u.get("output_tokens")))
            rec["cost"] = _num(u.get("cost"))
        own.log(rec)

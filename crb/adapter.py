"""Client for the student's adapter (see SPEC "The adapter contract").

Uses only http.client so that timing is exact: `first_token` is when the first non-blank piece of
answer text arrives (SSE delta or plain-text chunk), `total` when the answer is complete, both on
`time.perf_counter()`. Nothing here raises on adapter problems; errors come back as plain-English
strings in the result objects. The timeout is a real total deadline, not just a per-read timeout.
"""
from __future__ import annotations

import codecs
import http.client
import json
import socket
import ssl
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .config import ANSWER_TIMEOUT, HEALTH_TIMEOUT, INGEST_TIMEOUT, RETRIEVE_TIMEOUT

_ACCEPT = "text/event-stream, text/plain;q=0.9, application/json;q=0.8"


@dataclass
class AnswerResult:
    answer: str | None = None
    cited: list = field(default_factory=list)
    error: str | None = None
    first_token_ms: float | None = None
    total_ms: float = 0.0
    status: int | None = None


@dataclass
class RetrieveResult:
    pages: list | None = None      # None = adapter has no retriever (or the call failed)
    error: str | None = None
    ms: float = 0.0
    status: int | None = None


@dataclass
class SimpleResult:
    ok: bool = False
    data: dict = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0
    status: int | None = None


class _Deadline(Exception):
    pass


def _connect(base_url: str, path: str, timeout: float):
    u = urlparse(base_url if "://" in base_url else "http://" + base_url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"'{base_url}' is not a valid adapter URL (use http://host:port)")
    port = u.port or (443 if u.scheme == "https" else 80)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(u.hostname, port, timeout=timeout, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(u.hostname, port, timeout=timeout)
    return conn, u.path.rstrip("/") + path


def _explain(exc: BaseException, base_url: str, timeout: float) -> str:
    """Plain-English reason for a network failure."""
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return f"the adapter did not finish within {timeout:g} s"
    if isinstance(exc, ConnectionRefusedError):
        return f"cannot connect to the adapter at {base_url} (is it running?)"
    if isinstance(exc, socket.gaierror):
        return f"cannot find the host in {base_url}"
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, (http.client.HTTPException, OSError)):
        return f"connection to the adapter at {base_url} failed ({type(exc).__name__}: {exc})"
    return f"unexpected adapter error ({type(exc).__name__}: {exc})"


def _send(base_url: str, path: str, payload: dict | None, timeout: float, method: str = "POST"):
    """Send a JSON request; return (conn, response). The caller closes conn."""
    conn, full = _connect(base_url, path, timeout)
    try:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": _ACCEPT, "Connection": "close"}
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        conn.request(method, full, body=body, headers=headers)
        return conn, conn.getresponse()
    except BaseException:
        conn.close()
        raise


def _short(data: bytes, n: int = 200) -> str:
    return data.decode("utf-8", "replace").strip().replace("\n", " ")[:n]


def _iter_chunks(conn, resp, deadline: float):
    """Yield body pieces as soon as they arrive; raise _Deadline when the total time is used up."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise _Deadline()
        if conn.sock is not None:
            conn.sock.settimeout(remaining)
        try:
            chunk = resp.read1(65536)
        except (socket.timeout, TimeoutError):
            raise _Deadline() from None
        if not chunk:
            return
        yield chunk


def _json_call(base_url: str, path: str, payload: dict | None, timeout: float, method: str = "POST") -> SimpleResult:
    t0 = time.perf_counter()
    res = SimpleResult()
    try:
        conn, resp = _send(base_url, path, payload, timeout, method)
        try:
            res.status = resp.status
            raw = b"".join(_iter_chunks(conn, resp, t0 + timeout))
        finally:
            conn.close()
        if res.status >= 400:
            res.error = f"the adapter answered {path} with HTTP {res.status}: {_short(raw)}"
        else:
            try:
                data = json.loads(raw.decode("utf-8"))
                res.data = data if isinstance(data, dict) else {"value": data}
                res.ok = True
            except ValueError:
                res.error = f"the adapter's reply to {path} is not JSON: {_short(raw)}"
    except _Deadline:
        res.error = f"the adapter did not finish {path} within {timeout:g} s"
    except BaseException as exc:  # noqa: BLE001 - never raise on adapter errors
        if isinstance(exc, KeyboardInterrupt):
            raise
        res.error = _explain(exc, base_url, timeout)
    res.seconds = time.perf_counter() - t0
    return res


def health(base_url: str, timeout: float = HEALTH_TIMEOUT) -> SimpleResult:
    """GET /health. `.data` holds the adapter's reply (name, version)."""
    return _json_call(base_url, "/health", None, timeout, method="GET")


def ingest(base_url: str, collection: str, directory: str, timeout: float = INGEST_TIMEOUT) -> SimpleResult:
    """POST /ingest and wait until the adapter says it is ready. `.seconds` is the ingest time."""
    res = _json_call(base_url, "/ingest", {"collection": collection, "dir": str(directory)}, timeout)
    if res.ok and res.data.get("ok") is False:
        res.ok = False
        res.error = f"the adapter could not ingest '{collection}': {res.data.get('error') or res.data}"
    return res


def retrieve(base_url: str, collection: str, question: str, k: int = 10,
             timeout: float = RETRIEVE_TIMEOUT) -> RetrieveResult:
    """POST /retrieve. `pages` is a list of page ids, best first, or None if there is no retriever."""
    res = _json_call(base_url, "/retrieve", {"collection": collection, "question": question, "k": k}, timeout)
    out = RetrieveResult(ms=round(res.seconds * 1000, 1), status=res.status)
    if not res.ok:
        out.error = res.error
        return out
    pages = res.data.get("pages")
    if pages is None:
        out.error = "the adapter returned no pages from /retrieve (no retriever?)"
    elif not isinstance(pages, list):
        out.error = "the adapter's /retrieve reply has no 'pages' list"
    else:
        out.pages = [str(p) for p in pages]
    return out


def answer(base_url: str, payload: dict, timeout: float = ANSWER_TIMEOUT) -> AnswerResult:
    """POST /answer and time it. Handles SSE, chunked text/plain and plain JSON replies."""
    t0 = time.perf_counter()
    deadline = t0 + timeout
    res = AnswerResult()
    parts: list[str] = []
    first: list[float] = []   # one-element box: perf_counter of the first non-blank text

    def add(text: str) -> None:
        if text:
            parts.append(text)
            if not first and text.strip():
                first.append(time.perf_counter())

    try:
        conn, resp = _send(base_url, "/answer", payload, timeout)
        try:
            res.status = resp.status
            ctype = (resp.getheader("Content-Type") or "").lower()
            if resp.status >= 400:
                body = b"".join(_iter_chunks(conn, resp, deadline))
                res.error = f"the adapter answered /answer with HTTP {resp.status}: {_short(body)}"
            elif "json" in ctype:
                body = b"".join(_iter_chunks(conn, resp, deadline))
                _read_json_answer(body, res, add)   # JSON: first token == total
            else:
                _read_stream(conn, resp, deadline, "event-stream" in ctype, res, add)
        finally:
            conn.close()
    except _Deadline:
        res.error = f"the adapter did not finish /answer within {timeout:g} s"
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        res.error = _explain(exc, base_url, timeout)
    end = time.perf_counter()
    res.total_ms = round((end - t0) * 1000, 1)
    res.first_token_ms = round((first[0] - t0) * 1000, 1) if first else None
    res.answer = "".join(parts)
    return res


def _read_json_answer(body: bytes, res: AnswerResult, add) -> None:
    try:
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        res.error = f"the adapter's /answer reply is not valid JSON: {_short(body)}"
        return
    if data.get("error"):
        res.error = f"the adapter reported an error: {str(data['error'])[:200]}"
    text = data.get("answer", data.get("text"))
    if text is None and not data.get("error"):
        res.error = "the adapter's /answer JSON has no 'answer' field"
    add(str(text) if text is not None else "")
    res.cited = [str(c) for c in (data.get("cited") or [])]


def _read_stream(conn, resp, deadline, sse_hint: bool, res: AnswerResult, add) -> None:
    """Read a streamed reply; text/event-stream (or a body starting with 'data:') is parsed as SSE."""
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    sse = sse_hint
    decided = sse_hint
    buf = ""
    for chunk in _iter_chunks(conn, resp, deadline):
        text = dec.decode(chunk)
        if not decided:
            buf += text
            if not buf.strip():
                continue
            sse = buf.lstrip().startswith("data:")
            decided = True
            text, buf = buf, ""
        if not sse:
            add(text)
            continue
        buf += text
        *lines, buf = buf.split("\n")
        for line in lines:
            _sse_line(line, res, add)
    tail = dec.decode(b"", final=True)
    if sse:
        buf += tail
        for line in buf.split("\n"):
            _sse_line(line, res, add)
    elif not decided:
        add(buf + tail)   # body was only whitespace
    else:
        add(tail)


def _sse_line(line: str, res: AnswerResult, add) -> None:
    line = line.rstrip("\r")
    if not line.startswith("data:"):
        return
    data = line[5:]
    if data.startswith(" "):
        data = data[1:]
    if not data.strip() or data.strip() == "[DONE]":
        return
    try:
        obj = json.loads(data)
    except ValueError:
        add(data)
        return
    if not isinstance(obj, dict):
        add(str(obj))
        return
    if obj.get("error"):
        res.error = f"the adapter reported an error: {str(obj['error'])[:200]}"
    delta = obj.get("delta")
    if isinstance(delta, str):
        add(delta)
    if obj.get("done"):
        res.cited = [str(c) for c in (obj.get("cited") or [])]
        if obj.get("answer") and not isinstance(delta, str):
            add(str(obj["answer"]))

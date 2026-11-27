"""crb.adapter against a fake adapter server started by the test (SSE, chunked text, JSON, errors)."""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from crb import adapter


class FakeAdapter:
    """A student's adapter in miniature. `calls` records (path, body). The question text picks the behaviour."""

    def __init__(self, protocol: str = "HTTP/1.1", answer_fn=None):
        self.calls: list[tuple[str, dict]] = []
        self.ingests = 0
        outer = self
        self.answer_fn = answer_fn

        class H(BaseHTTPRequestHandler):
            protocol_version = protocol

            def log_message(self, *a):
                pass

            def _send(self, status, obj):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"ok": True, "name": "fake", "version": "1"})
                else:
                    self._send(404, {"error": "no"})

            def _chunk(self, text: str):
                b = text.encode()
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.calls.append((self.path, body))
                if self.path == "/ingest":
                    outer.ingests += 1
                    self._send(200, {"ok": True, "pages": 12})
                elif self.path == "/retrieve":
                    q = body["question"]
                    if "noretriever" in q:
                        self._send(200, {"pages": None})
                    else:
                        self._send(200, {"pages": ["p1", "p2", "p3"]})
                elif self.path == "/answer":
                    self.answer(body)
                else:
                    self._send(404, {"error": "no"})

            def answer(self, body):
                if outer.answer_fn:
                    return outer.answer_fn(self, body)
                q = body["question"]
                if "sse" in q:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    time.sleep(0.2)
                    self.wfile.write(b'data: {"delta": "Hel"}\n\n')
                    self.wfile.flush()
                    time.sleep(0.2)
                    self.wfile.write('data: {"delta": "lo é"}\n\n'.encode())
                    self.wfile.write(b'data: {"done": true, "cited": ["p1", "p2"]}\n\n')
                    self.wfile.flush()
                elif "chunked" in q:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self._chunk("\n")            # blank lead-in must not count as the first token
                    time.sleep(0.3)
                    self._chunk("Plain ")
                    time.sleep(0.2)
                    self._chunk("text")
                    self.wfile.write(b"0\r\n\r\n")
                elif "json" in q:
                    time.sleep(0.2)
                    self._send(200, {"answer": "From JSON", "cited": ["p9"]})
                elif "garbage" in q:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "5")
                    self.end_headers()
                    self.wfile.write(b"nope!")
                elif "boom" in q:
                    self._send(500, {"error": "kaput"})
                elif "errorstream" in q:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b'data: {"delta": "partial"}\n\ndata: {"error": "model crashed"}\n\n')
                elif "slow" in q:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b'data: {"delta": "start"}\n\n')
                    self.wfile.flush()
                    time.sleep(3)
                else:
                    self._send(200, {"answer": f"echo {body.get('mode')}", "cited": []})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake():
    f = FakeAdapter()
    yield f
    f.close()


def ask(url, q, **kw):
    return adapter.answer(url, {"collection": "c", "question": q, "mode": "e2e"}, **kw)


def test_health_and_retrieve(fake):
    h = adapter.health(fake.url)
    assert h.ok and h.data["name"] == "fake"
    r = adapter.retrieve(fake.url, "c", "anything", 10)
    assert r.pages == ["p1", "p2", "p3"] and r.error is None and r.ms > 0
    assert fake.calls[-1][1]["k"] == 10
    n = adapter.retrieve(fake.url, "c", "noretriever")
    assert n.pages is None and "no pages" in n.error


def test_ingest_timed(fake):
    res = adapter.ingest(fake.url, "vidore_v3_hr", "/tmp/pages")
    assert res.ok and res.data["pages"] == 12 and res.seconds >= 0
    assert fake.calls[-1][1] == {"collection": "vidore_v3_hr", "dir": "/tmp/pages"}


def test_sse_timing_and_cited(fake):
    r = ask(fake.url, "sse please")
    assert r.error is None
    assert r.answer == "Hello é" and r.cited == ["p1", "p2"]
    assert 150 <= r.first_token_ms < 380          # first delta after ~200 ms
    assert r.total_ms >= 380 and r.total_ms > r.first_token_ms + 100


def test_sse_without_chunking_http10_streams():
    """Plain http.server (HTTP/1.0, close-delimited body) must still stream, not wait for the end."""
    f = FakeAdapter(protocol="HTTP/1.0")
    try:
        r = ask(f.url, "sse")
        assert r.answer == "Hello é" and r.first_token_ms < r.total_ms - 100
    finally:
        f.close()


def test_chunked_text_first_token_ignores_blank_chunk(fake):
    r = ask(fake.url, "chunked")
    assert r.error is None and r.answer == "\nPlain text" and r.cited == []
    assert r.first_token_ms >= 270                # not the blank first chunk
    assert r.total_ms > r.first_token_ms + 150


def test_json_first_token_equals_total(fake):
    r = ask(fake.url, "json")
    assert r.answer == "From JSON" and r.cited == ["p9"] and r.error is None
    assert r.first_token_ms is not None and abs(r.total_ms - r.first_token_ms) < 30
    assert r.total_ms >= 180


def test_payload_is_sent_as_given(fake):
    payload = {"collection": "c", "question": "gold?", "mode": "gold",
               "context": [{"page_id": "a", "markdown": "m", "image_path": "/x.png"}]}
    r = adapter.answer(fake.url, payload)
    assert r.answer == "echo gold"
    assert fake.calls[-1] == ("/answer", payload)


def test_http_error_is_a_string_not_an_exception(fake):
    r = ask(fake.url, "boom")
    assert r.status == 500 and "HTTP 500" in r.error and r.answer == ""


def test_bad_json_reply(fake):
    r = ask(fake.url, "garbage")
    assert "not valid JSON" in r.error


def test_sse_error_event_keeps_partial_answer(fake):
    r = ask(fake.url, "errorstream")
    assert "model crashed" in r.error and r.answer == "partial"


def test_timeout_is_a_total_deadline(fake):
    t0 = time.perf_counter()
    r = ask(fake.url, "slow", timeout=0.6)
    assert time.perf_counter() - t0 < 2.0
    assert "within 0.6 s" in r.error
    assert r.answer == "start" and r.first_token_ms is not None


def test_connection_refused_is_reported_not_raised():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    url = f"http://127.0.0.1:{port}"
    r = ask(url, "hello")
    assert "cannot connect" in r.error and r.first_token_ms is None
    assert not adapter.health(url).ok
    assert adapter.retrieve(url, "c", "q").error
    assert adapter.ingest(url, "c", "/x").error


def test_bad_url_is_reported():
    r = adapter.answer("ftp://nope", {"question": "q"})
    assert "not a valid adapter URL" in r.error

"""crb.proxy: passthrough to a fake OpenAI-compatible upstream, streaming, usage capture, safe logging."""
from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from crb.proxy import TokenProxy
from crb.state import State

SECRET = "sk-test-secret-123"
PROMPT = "tell me something very private about page 7"


class FakeUpstream:
    def __init__(self):
        self.seen: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            # HTTP/1.0 close-delimited on purpose: the proxy must still stream it chunk by chunk.
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b'data: {"model":"m/x","choices":[{"delta":{"content":"Hi"}}]}\n\n')
                    self.wfile.flush()
                    time.sleep(0.4)
                    self.wfile.write(b'data: {"model":"m/x","choices":[{"delta":{"content":" there"}}]}\n\n')
                    self.wfile.write(b'data: {"model":"m/x","choices":[],"usage":{"prompt_tokens":11,'
                                     b'"completion_tokens":5,"cost":0.0007}}\n\n')
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    data = json.dumps({"model": "m/x", "choices": [{"message": {"content": "ok"}}],
                                       "usage": {"prompt_tokens": 20, "completion_tokens": 3, "cost": 0.0012}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            def do_GET(self):
                data = b'{"data": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def rig(tmp_path):
    up = FakeUpstream()
    state = State()
    state.set(phase="e2e", qid="hr-12", axis=3)
    proxy = TokenProxy(state, tmp_path / "proxy.jsonl", port=0, upstream_url=f"http://127.0.0.1:{up.port}/api/v1").start()
    yield up, proxy, tmp_path / "proxy.jsonl"
    proxy.stop()
    up.close()


def post(proxy, body: dict, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "Authorization": f"Bearer {SECRET}"})
    return c, c.getresponse()


def log_lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_json_passthrough_usage_and_cost(rig):
    up, proxy, log = rig
    c, r = post(proxy, {"model": "m/x", "messages": [{"role": "user", "content": PROMPT}]})
    body = json.loads(r.read())
    c.close()
    assert r.status == 200 and body["choices"][0]["message"]["content"] == "ok"
    seen = up.seen[0]
    assert seen["path"] == "/api/v1/chat/completions"          # /v1 replaced by the upstream base path
    assert seen["auth"] == f"Bearer {SECRET}"                  # key passes through untouched
    assert seen["body"]["usage"] == {"include": True}          # cost accounting requested
    time.sleep(0.1)
    (rec,) = log_lines(log)
    assert (rec["qid"], rec["axis"], rec["phase"]) == ("hr-12", 3, "e2e")
    assert (rec["prompt_tokens"], rec["completion_tokens"], rec["cost"]) == (20, 3, 0.0012)
    assert rec["model"] == "m/x" and rec["status"] == 200 and rec["stream"] is False


def test_streaming_is_passed_through_as_it_arrives(rig):
    up, proxy, log = rig
    t0 = time.perf_counter()
    c, r = post(proxy, {"model": "m/x", "stream": True, "messages": [{"role": "user", "content": PROMPT}]})
    assert r.status == 200 and "event-stream" in r.getheader("Content-Type")
    first = r.read1(4096)
    t_first = time.perf_counter() - t0
    rest = b""
    while True:
        part = r.read1(4096)
        if not part:
            break
        rest += part
    t_all = time.perf_counter() - t0
    c.close()
    assert b'"Hi"' in first
    assert t_first < 0.3 and t_all >= 0.4                       # the first chunk did not wait for the end
    assert b'" there"' in rest and rest.rstrip().endswith(b"[DONE]")
    assert up.seen[0]["body"]["stream_options"] == {"include_usage": True}
    time.sleep(0.1)
    (rec,) = log_lines(log)
    assert rec["stream"] is True
    assert (rec["prompt_tokens"], rec["completion_tokens"], rec["cost"]) == (11, 5, 0.0007)


def test_log_never_holds_keys_or_prompts(rig):
    _, proxy, log = rig
    for stream in (False, True):
        c, r = post(proxy, {"model": "m/x", "stream": stream, "messages": [{"role": "user", "content": PROMPT}]})
        r.read()
        c.close()
    time.sleep(0.1)
    text = log.read_text()
    assert SECRET not in text and "private" not in text and "Bearer" not in text
    assert len(text.splitlines()) == 2


def test_current_question_follows_state(rig):
    _, proxy, log = rig
    proxy.state.set(phase="crag", qid="crag-9", axis=4)
    c, r = post(proxy, {"model": "m/x", "messages": []})
    r.read()
    c.close()
    time.sleep(0.1)
    assert log_lines(log)[-1]["qid"] == "crag-9"


def test_get_models_and_health(rig):
    _, proxy, _ = rig
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
    c.request("GET", "/v1/models")
    assert json.loads(c.getresponse().read()) == {"data": []}
    c.close()
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
    c.request("GET", "/health")
    assert json.loads(c.getresponse().read())["crb_proxy"] is True
    c.close()


def test_upstream_down_gives_502_and_is_logged(tmp_path):
    state = State()
    # port 9 (discard) on localhost: nothing listens, connection refused
    proxy = TokenProxy(state, tmp_path / "p.jsonl", port=0, upstream_url="http://127.0.0.1:9/v1").start()
    try:
        c, r = post(proxy, {"model": "m", "messages": []})
        body = json.loads(r.read())
        c.close()
        assert r.status == 502 and "could not reach" in body["error"]["message"]
        time.sleep(0.1)
        rec = json.loads((tmp_path / "p.jsonl").read_text().splitlines()[0])
        assert rec["status"] == 502 and rec["error"]
    finally:
        proxy.stop()

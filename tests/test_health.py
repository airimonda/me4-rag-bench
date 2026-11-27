"""Tests for the Pi health sampler and the example adapters. Offline."""
from __future__ import annotations

import csv
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from crb import health
from crb.state import State

ROOT = Path(__file__).resolve().parent.parent


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def serve(server):
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return th


# ------------------------------------------------------------------ sampler

def test_sampler_writes_csv(tmp_path):
    state = State()
    state.set(phase="ingest")
    out = tmp_path / "health.csv"
    s = health.HealthSampler(tmp_path, None, out, state)
    s.start()
    time.sleep(2.3)
    s.stop()
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == health.COLUMNS
    assert len(rows) >= 2
    assert rows[0]["phase"] == "ingest"
    assert rows[0]["disk_free_mb"] != ""
    assert rows[0]["iso"].endswith("+00:00")
    summ = s.summary()
    assert summ["samples"] == len(rows)
    assert "temp_max" in summ and "available" in summ


def test_decode_throttled():
    d = health.decode_throttled(0x50005)
    assert d["under_voltage_now"] and d["throttled_now"]
    assert not d["freq_capped_now"] and not d["soft_temp_now"]
    assert d["under_voltage_boot"] and d["throttled_boot"]
    assert not d["freq_capped_boot"]
    d = health.decode_throttled(0x8)
    assert d["soft_temp_now"] and not d["under_voltage_now"]
    assert health.decode_throttled(0) == {k: False for k in d}
    assert health.parse_throttled("throttled=0x50005\n") == 0x50005
    assert health.parse_throttled("garbage") is None


def test_vcgencmd_parsers():
    assert health.parse_temp("temp=48.3'C\n") == 48.3
    assert health.parse_clock_mhz("frequency(48)=1500398464") == 1500.4
    assert health.parse_volts("volt=0.8438V") == 0.8438
    assert health.parse_temp("") is None


PROC_STAT_A = """cpu  100 0 100 800 0 0 0 0 0 0
cpu0 50 0 50 400 0 0 0 0 0 0
cpu1 50 0 50 400 0 0 0 0 0 0
intr 1
"""
PROC_STAT_B = """cpu  200 0 200 1000 0 0 0 0 0 0
cpu0 150 0 100 450 0 0 0 0 0 0
cpu1 50 0 100 550 0 0 0 0 0 0
"""


def test_cpu_percent():
    a, b = health.parse_proc_stat(PROC_STAT_A), health.parse_proc_stat(PROC_STAT_B)
    assert a["cpu"] == (200, 1000)
    overall, busiest = health.cpu_percent(a, b)
    assert overall == 50.0           # 200 busy of 400
    assert busiest == 75.0           # cpu0: 150 busy of 200


def test_meminfo_loadavg():
    m = health.parse_meminfo("MemTotal: 4000000 kB\nMemAvailable: 2048000 kB\nSwapTotal: 1048576 kB\nSwapFree: 524288 kB\n")
    assert m == {"mem_avail_mb": 2000.0, "swap_used_mb": 512.0}
    assert health.parse_loadavg("0.52 0.40 0.30 1/200 999") == 0.52
    assert health.parse_status_rss_mb("Name: x\nVmRSS:\t  20480 kB\n") == 20.0


NET_TCP = """  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 12345 1 0000 100 0 0 10 0
   1: 0100007F:1F41 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 22222 1 0000 100 0 0 10 0
   2: 0100007F:C000 0100007F:1F40 01 00000000:00000000 00:00000000 00000000  1000        0 33333 1 0000 100 0 0 10 0
"""


def test_listen_inodes():
    assert health.parse_listen_inodes(NET_TCP, 8000) == {"12345"}      # 0x1F40 == 8000, state 0A only
    assert health.parse_listen_inodes(NET_TCP, 9999) == set()


def test_find_pid_missing_proc(tmp_path):
    assert health.find_pid_for_port(8000, proc=str(tmp_path / "nope")) is None


def test_summary_from_rows(tmp_path):
    s = health.HealthSampler(tmp_path, None, tmp_path / "h.csv", State())
    s._rows = [
        {"temp_c": "50", "throttled_hex": "0x0", "freq_capped_now": 0, "throttled_now": 0, "soft_temp_now": 0,
         "under_voltage_now": 0, "cpu_pct": "10", "mem_avail_mb": "900", "arm_mhz": "1500", "net_rtt_ms": "40"},
        {"temp_c": "81", "throttled_hex": "0x8", "freq_capped_now": 0, "throttled_now": 0, "soft_temp_now": 1,
         "under_voltage_now": 0, "cpu_pct": "90", "mem_avail_mb": "700", "arm_mhz": "1000", "net_rtt_ms": "60"},
    ]
    r = s.summary()
    assert r["temp_max"] == 81.0 and r["throttled_seconds"] == 1 and r["throttled_share"] == 0.5
    assert r["under_voltage"] is False and r["mem_avail_min_mb"] == 700.0 and r["arm_mhz_min"] == 1000.0
    assert r["net_rtt_p50"] == 50.0 and r["available"] is True


# ------------------------------------------------------------------ baseline adapter

def test_bm25_ranking():
    base = load_example("baseline_adapter")
    ids = ["cats", "dogs", "stocks"]
    texts = ["cats purr and chase mice", "dogs bark and fetch the ball", "the stock market fell sharply on bonds"]
    bm = base.BM25(ids, texts)
    assert bm.top("why do cats purr", 3)[0] == "cats"
    assert bm.top("stock market bonds", 1) == ["stocks"]
    assert bm.top("zzzz", 3) == []


def test_html_chunking():
    base = load_example("baseline_adapter")
    text = base.html_to_text("<html><head><title>x</title><style>p{}</style></head><body><script>bad()</script><p>Hello</p><p>World</p></body></html>")
    assert "bad" not in text and "Hello" in text and "World" in text
    chunks = base.chunk_text("\n".join(["line " * 50] * 30), 1500)
    assert all(len(c) <= 1500 for c in chunks) and len(chunks) > 1


class FakeLLM(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        FakeLLM.seen.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(n))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for piece in ("Par", "is"):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    llm = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    serve(llm)
    monkeypatch.setenv("CRB_LLM_BASE", f"http://127.0.0.1:{llm.server_address[1]}/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("CRB_BASELINE_MODEL", "fake/model")
    base = load_example("baseline_adapter")
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path / "cache")
    FakeLLM.seen.clear()
    srv = base.make_server("127.0.0.1", 0)
    serve(srv)
    yield base, f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()
    llm.shutdown()


def sse(url, body):
    r = requests.post(url + "/answer", json=body, stream=True, timeout=10)
    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return r, events


def test_baseline_flow(baseline):
    base, url, tmp = baseline
    pages = tmp / "pages"
    pages.mkdir()
    (pages / "a__p1.md").write_text("The capital of France is Paris.")
    (pages / "b__p2.md").write_text("Bananas are yellow fruit.")
    assert requests.get(url + "/health").json()["ok"] is True
    assert requests.post(url + "/ingest", json={"collection": "c", "dir": str(pages)}).json() == {"ok": True, "pages": 2}
    assert (tmp / "cache" / "c.pkl").exists()
    got = requests.post(url + "/retrieve", json={"collection": "c", "question": "capital of France", "k": 2}).json()
    assert got["pages"][0] == "a__p1"

    # e2e
    r, ev = sse(url, {"collection": "c", "question": "What is the capital of France?", "mode": "e2e"})
    assert r.headers["Content-Type"].startswith("text/event-stream")
    assert "".join(e["delta"] for e in ev if "delta" in e) == "Paris"
    assert ev[-1]["done"] is True and ev[-1]["cited"][0] == "a__p1"
    path, auth, body = FakeLLM.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test"
    assert body["stream"] is True and body["model"] == "fake/model"
    assert "Paris" in body["messages"][1]["content"]

    # gold
    r, ev = sse(url, {"collection": "c", "question": "q?", "mode": "gold",
                      "context": [{"page_id": "x", "markdown": "GOLDTEXT", "image_path": None}]})
    assert ev[-1] == {"done": True, "cited": ["x"]}
    assert "GOLDTEXT" in FakeLLM.seen[-1][2]["messages"][1]["content"]

    # web
    html = tmp / "p.html"
    html.write_text("<html><body><script>x()</script><p>Mount Everest is 8849 metres tall.</p></body></html>")
    r, ev = sse(url, {"collection": "crag_task1", "question": "How tall is Everest?", "mode": "web",
                      "query_time": "03/05/2024, 23:18:31 PT", "pages": [{"path": str(html), "page_name": "Everest"}]})
    assert ev[-1] == {"done": True, "cited": []}
    sent = FakeLLM.seen[-1][2]["messages"][1]["content"]
    assert "8849" in sent and "x()" not in sent and "03/05/2024" in sent


def test_baseline_bad_requests(baseline):
    base, url, tmp = baseline
    r = requests.post(url + "/retrieve", json={"collection": "nope", "question": "q"})
    assert r.status_code == 400 and "ingest" in r.json()["error"]
    r = requests.post(url + "/answer", data="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    r = requests.post(url + "/answer", json={"question": "q", "mode": "weird"})
    assert r.status_code == 400
    assert requests.get(url + "/health").status_code == 200    # still alive


# ------------------------------------------------------------------ template adapter

def test_template_adapter(tmp_path):
    tpl = load_example("adapter_template")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), tpl.Handler)
    serve(srv)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        h = requests.get(url + "/health").json()
        assert h["ok"] is True and h["name"]
        (tmp_path / "p1.md").write_text("hello")
        assert requests.post(url + "/ingest", json={"collection": "c", "dir": str(tmp_path)}).json() == {"ok": True, "pages": 1}
        assert requests.post(url + "/retrieve", json={"collection": "c", "question": "q", "k": 10}).json() == {"pages": ["p1"]}
        r = requests.post(url + "/answer", json={"collection": "c", "question": "q", "mode": "e2e"}, stream=True)
        events = [json.loads(ln[6:]) for ln in r.iter_lines(decode_unicode=True) if ln.startswith("data: ")]
        assert events[-1]["done"] is True
        assert "".join(e.get("delta", "") for e in events) == "This is a placeholder answer."
    finally:
        srv.shutdown()

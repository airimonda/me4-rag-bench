"""Scoring tests: all offline. The judge is a fake OpenRouter server served from this file."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from crb import literature
from crb.score import crag_eval, diagnosis, judge as J, latency, results as R, retrieval, stats

HR, CS, CRAG = "vidore_v3_hr", "vidore_v3_computer_science", "crag_task1"


# ---------------------------------------------------------------- fake judge server

class FakeJudge:
    """OpenRouter look-alike. Verdicts are driven by marker words in the prompt.

    Vidore: WRONG -> Incorrect, PARTIAL -> Partially Correct, else Correct.
    CRAG: the gpt-3.5 judge accepts BOTH or ONLY35; the llama judge accepts BOTH.
    Anything else is incorrect. Haiku (smoke) accepts BOTH.
    """

    def __init__(self, fail_first: int = 0, fence: bool = False, cost: float = 0.001):
        self.requests: list[dict] = []
        self.auth: list[str] = []
        self.fail_first = fail_first
        self.fence = fence
        self.cost = cost
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer._lock:
                    outer.requests.append(body)
                    outer.auth.append(self.headers.get("Authorization", ""))
                    n = len(outer.requests)
                if n <= outer.fail_first:
                    self.send_response(500 if n % 2 else 429)
                    self.end_headers()
                    return
                text = outer.reply(body)
                data = json.dumps({"choices": [{"message": {"content": text}}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": outer.cost}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def reply(self, body: dict) -> str:
        content = body["messages"][0]["content"]
        model = body["model"]
        if '"Partially Correct"' in content:  # vidore prompt
            label = "Incorrect" if "WRONG" in content else "Partially Correct" if "PARTIAL" in content else "Correct"
            out = json.dumps({"explanation": "fake", "judgment": label})
        else:
            if model.startswith("openai/gpt-3.5"):
                ok = "BOTH" in content or "ONLY35" in content
            else:
                ok = "BOTH" in content
            out = json.dumps({"explanation": "fake", "judgment": "accurate" if ok else "incorrect"})
        return "```json\n" + out + "\n```" if self.fence else out

    def models(self) -> list[str]:
        return [r["model"] for r in self.requests]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_judge(monkeypatch, tmp_path):
    fj = FakeJudge()
    monkeypatch.setenv("CRB_JUDGE_BASE", fj.url)
    monkeypatch.setenv("CRB_JUDGE_KEY", "test-key-123")
    monkeypatch.setenv("CRB_JUDGE_CACHE", str(tmp_path / "cache" / "judge.sqlite"))
    monkeypatch.setenv("CRB_JUDGE_BACKOFF", "0")
    yield fj
    fj.stop()


# ---------------------------------------------------------------- fake packs and synthetic run

class FakePacks:
    def __init__(self, questions: dict[str, list[dict]], qrels: dict[str, dict]):
        self.q, self.r = questions, qrels

    def load_questions(self, pack):
        if pack not in self.q:
            raise FileNotFoundError(pack)
        return self.q[pack]

    def load_qrels(self, pack):
        return self.r.get(pack, {})


def vidore_questions(pack: str, short: str, n: int) -> tuple[list[dict], dict]:
    qs, qrels = [], {}
    for i in range(1, n + 1):
        qid = f"{short}-{i}"
        qs.append({"qid": qid, "pack": pack, "question": f"What does item {i} say?", "answer": f"gold text {i}",
                   "alt_answers": [], "qtype": "extractive" if i % 2 else "multi-hop",
                   "content_type": ["Text"] if i % 2 else ["Table", "Chart"], "query_time": None,
                   "gold_pages": [f"d{i}__p1", f"d{i}__p2"]})
        qrels[qid] = {f"d{i}__p1": 2, f"d{i}__p2": 1}
    return qs, qrels


def crag_questions() -> list[dict]:
    rows = [("crag-1", "Who wrote it?", "Ada", "simple"), ("crag-2", "When did it start?", "1999", "simple"),
            ("crag-3", "Where is it?", "Paris", "comparison"), ("crag-4", "How many are there?", "12", "set"),
            ("crag-5", "Which is bigger?", "the red one", "comparison"),
            ("crag-6", "When did the king land on Mars?", "invalid question", "false_premise"),
            ("crag-7", "What colour is the sky?", "blue", "simple")]
    qs = [{"qid": q, "pack": CRAG, "question": t, "answer": a, "alt_answers": [], "qtype": ty,
           "content_type": [], "query_time": "03/05/2024, 23:18:31 PT", "gold_pages": []} for q, t, a, ty in rows]
    for i in range(8, 11):  # pack has 10 questions, 7 are in the sample
        qs.append({"qid": f"crag-{i}", "pack": CRAG, "question": f"Filler {i}?", "answer": "x", "alt_answers": [],
                   "qtype": "simple", "content_type": [], "query_time": None, "gold_pages": []})
    return qs


def line(qid, pack, axis, t, *, answer=None, retrieved=None, error=None, warmup=False, first=None, total=None,
         retrieve=None, cited=None):
    return {"qid": qid, "pack": pack, "axis": axis, "warmup": warmup, "t": t, "retrieved": retrieved,
            "answer": answer, "cited": cited or [], "error": error,
            "latency_ms": {"first_token": first, "total": total, "retrieve": retrieve}}


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def health_rows(n: int = 100, throttle=(40, 50)) -> list[dict]:
    rows = []
    for i in range(n):
        ph = "ingest" if i < 10 else "retrieval" if i < 25 else "gold" if i < 45 else "e2e" if i < 70 else "crag"
        thr = throttle[0] <= i < throttle[1]
        rows.append({"t": i, "iso": f"2026-12-04T09:{i // 60:02d}:{i % 60:02d}+00:00", "temp_c": 50 + i * 0.25,
                     "throttled_hex": "0x4" if thr else "0x0", "under_voltage_now": 0,
                     "freq_capped_now": 0, "throttled_now": int(thr), "soft_temp_now": 0,
                     "arm_mhz": 1000 if thr else 1800, "core_v": 0.9, "cpu_pct": 30 + (i % 7) * 5,
                     "cpu_max_core_pct": 90 + (i % 5), "load1": 1.5, "mem_avail_mb": 900 - i, "swap_used_mb": 0,
                     "adapter_rss_mb": 400, "disk_free_mb": 10000, "net_rtt_ms": 30 + i % 10, "phase": ph})
    return rows


def write_health(path: Path, rows):
    cols = ["t", "iso", "temp_c", "throttled_hex", "under_voltage_now", "freq_capped_now", "throttled_now",
            "soft_temp_now", "arm_mhz", "core_v", "cpu_pct", "cpu_max_core_pct", "load1", "mem_avail_mb",
            "swap_used_mb", "adapter_rss_mb", "disk_free_mb", "net_rtt_ms", "phase"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def make_run(tmp_path: Path, monkeypatch, *, mode: str = "official", health: bool = True,
             proxy: bool = True) -> Path:
    """Small run: HR pack of 20 questions (sample of 10 answered), CRAG pack of 10 (7 answered)."""
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    hr_q, hr_qrels = vidore_questions(HR, "hr", 20)
    monkeypatch.setattr(R, "_pack_api", lambda: FakePacks({HR: hr_q, CRAG: crag_questions()}, {HR: hr_qrels}))
    rows = []
    t = 0.0
    for i in range(1, 21):  # axis 1: all 20 questions; 1-14 found, 15-20 missed
        t += 1
        got = [f"d{i}__p1", f"d{i}__p2", "z1"] if i <= 14 else ["zz1", "zz2"]
        rows.append(line(f"hr-{i}", HR, 1, t, retrieved=got, retrieve=100 + i * 10, warmup=i <= 3))
    sample = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16]
    gold_ans = {i: "OK gold" for i in sample}
    gold_ans[7] = "WRONG gold"
    e2e_ans = {i: "OK e2e" for i in sample}
    e2e_ans.update({7: "WRONG e2e", 8: "WRONG e2e", 15: "WRONG e2e", 16: "PARTIAL e2e"})
    for k, i in enumerate(sample):
        t += 1
        rows.append(line(f"hr-{i}", HR, 2, t, answer=gold_ans[i], retrieved=[], first=800 + 100 * k, total=2000 + 100 * k,
                         warmup=k < 3))
    for k, i in enumerate(sample):
        t += 1
        rows.append(line(f"hr-{i}", HR, 3, t, answer=e2e_ans[i], retrieved=[], first=1000 + 200 * k,
                         total=3000 + 300 * k, warmup=k < 3))
    crag = [("crag-1", "Ada."), ("crag-2", "Sorry, I don't know."), ("crag-3", "BOTH agree"),
            ("crag-4", "ONLY35 says so"), ("crag-5", "NEITHER likes it"), ("crag-6", "invalid question"),
            ("crag-7", "Invalid question.")]
    for k, (qid, text) in enumerate(crag):
        t += 1
        rows.append(line(qid, CRAG, 4, t, answer=text, retrieved=[], first=1500 + 100 * k, total=4000 + 100 * k,
                         warmup=k < 3))
    write_jsonl(run / "answers.jsonl", rows)
    session = {"student": "2024-00001", "mode": mode, "sample": "class-v1", "status": "done",
               "pi": {"model": "Raspberry Pi 4 Model B Rev 1.4", "ram_gb": 4},
               "adapter": {"url": "http://127.0.0.1:8000", "health": {"name": "my-rag", "version": "1.0"}},
               "start": "2026-12-04T09:00:00Z", "end": "2026-12-04T10:27:00Z",
               "ingest": {HR: {"ok": True, "seconds": 120.0, "pages": 200, "error": None}},
               "notes": ["run note from the harness"]}
    (run / "run.json").write_text(json.dumps(session))
    if health:
        write_health(run / "health.csv", health_rows())
    if proxy:
        write_jsonl(run / "proxy.jsonl", [
            {"t": 5.0, "model": "m", "prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.01, "qid": "hr-1"},
            {"t": 6.0, "model": "m", "prompt_tokens": 2000, "completion_tokens": 200, "cost": 0.02, "qid": "hr-2"},
            {"t": 7.0, "model": "m", "prompt_tokens": 3000, "completion_tokens": 300, "cost": 0.03, "qid": "crag-3"},
            {"t": 8.0, "model": "m", "prompt_tokens": 500, "completion_tokens": 50, "cost": 0.04,
             "qid": "crag-4", "finish_reason": "length"}])
    return run


# ---------------------------------------------------------------- statistics

def test_percentile_and_median():
    assert stats.percentile([1, 2, 3, 4], 50) == 2.5
    assert stats.percentile([10], 90) == 10
    assert stats.percentile([], 50) is None
    assert stats.percentile(range(1, 11), 90) == pytest.approx(9.1)
    assert stats.median([3, 1, 2]) == 2


def test_bootstrap_ci_and_fpc():
    vals = [100.0] * 68 + [0.0] * 32            # 68% correct, n = 100
    full = stats.bootstrap_ci(vals, None)
    assert full == pytest.approx(1.96 * math.sqrt(0.68 * 0.32 / 100) * 100, rel=0.15)
    corrected = stats.bootstrap_ci(vals, 318)
    assert corrected == pytest.approx(full * math.sqrt((318 - 100) / 317))   # same seed, only the FPC differs
    assert corrected < full
    assert stats.bootstrap_ci(vals, 100) == 0.0                              # n == N: exact
    assert stats.bootstrap_ci(vals, 318) == stats.bootstrap_ci(vals, 318)     # seeded
    assert stats.bootstrap_ci([], 10) is None


# ---------------------------------------------------------------- retrieval

def test_ndcg_recall_hand_computed():
    qrel = {"a": 2, "b": 1, "c": 1, "n": 0}
    got = retrieval.ndcg_at_k(["b", "x", "a"], qrel)
    dcg = 1 / math.log2(2) + 2 / math.log2(4)
    ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert got == pytest.approx(dcg / ideal)
    assert retrieval.recall_at_k(["b", "x", "a"], qrel) == pytest.approx(2 / 3)
    assert retrieval.ndcg_at_k(["a", "b", "c"], qrel) == pytest.approx(1.0)
    assert retrieval.ndcg_at_k([], qrel) == 0.0
    assert retrieval.ndcg_at_k(["a"], {"a": 0}) is None


def test_ndcg_cutoff_and_duplicates():
    qrel = {"gold": 1}
    late = [f"x{i}" for i in range(10)] + ["gold"]
    assert retrieval.ndcg_at_k(late, qrel) == 0.0
    assert retrieval.recall_at_k(late, qrel) == 0.0
    # a repeated page counts once
    assert retrieval.ndcg_at_k(["gold", "gold"], qrel) == pytest.approx(1.0)
    rep = retrieval.score_run({"q1": ["gold"], "q2": ["no"], "q3": ["gold"]},
                              {"q1": qrel, "q2": qrel})  # q3 has no qrels: not scored
    assert rep["n"] == 2 and rep["ndcg10"] == pytest.approx(50.0)


# ---------------------------------------------------------------- literature and ranks

def test_literature_values_verbatim():
    p = literature.published()
    assert [len(p[k]["rows"]) for k in ("retrieval", "gold", "e2e", "crag")] == [21, 3, 8, 8]
    by = {r["name"]: r for r in p["retrieval"]["rows"]}
    assert by["ColEmbed-3B-v2"][literature.HR] == 65.4 and by["ColEmbed-3B-v2"][literature.CS] == 78.6
    assert by["Qwen3-8B"][literature.CS] == 73.5 and by["BGE-M3"][literature.HR] == 45.3
    e2e = {r["name"]: r for r in p["e2e"]["rows"]}
    assert e2e["GPT-5.2 · ColEmbed, image"][literature.CS] == 86.5
    assert "note" in e2e["Qwen3-VL-235B · ColEmbed, image"] and "note" not in e2e["Kimi K2 · ColEmbed, text"]
    assert {r["name"]: r[literature.CRAG] for r in p["crag"]["rows"]}["Llama 2 7B"] == -66.7
    assert json.dumps(p)  # JSON-ready


def test_rank_and_median():
    pub = [r["value"] for r in literature.rows("retrieval", HR)]
    r = R.rank_in(55.0, pub)
    higher = sum(1 for v in pub if v > 55.0)
    assert r["rank"] == higher + 1 == 10 and r["of"] == 22
    assert r["median"] == round(statistics.median(pub), 2)
    assert r["side"] == ("above" if 55.0 > statistics.median(pub) else "below")
    assert R.rank_in(100.0, pub)["rank"] == 1
    assert R.rank_in(0.0, pub)["rank"] == 22
    med = statistics.median(pub)
    assert R.rank_in(med, pub)["side"] == "at"
    crag = R.rank_in(21.5, [r["value"] for r in literature.rows("crag", CRAG)])
    assert crag["rank"] == 1 and crag["of"] == 9 and crag["side"] == "above"
    assert literature.rows("retrieval", "dev_vidore") == []


# ---------------------------------------------------------------- diagnosis

def test_diagnosis_tags():
    gold = ["g1", "g2"]
    assert diagnosis.tag(True, gold, ["x"], False) == "correct"
    assert diagnosis.tag(False, gold, ["x", "y"], True) == "retrieval_miss"
    assert diagnosis.tag(False, gold, ["x", "g2"], False) == "generation_miss"
    assert diagnosis.tag(False, gold, ["g1"], True) == "integration_miss"
    assert diagnosis.tag(False, gold, None, True) == "integration_miss"      # no axis-1 line: cannot call a miss
    assert diagnosis.tag(False, gold, ["g1"], None) == "integration_miss"    # no axis-2 line
    late = [f"x{i}" for i in range(10)] + ["g1"]
    assert diagnosis.tag(False, gold, late, True) == "retrieval_miss"        # outside the top 10
    assert diagnosis.count(["correct", "correct", "retrieval_miss"]) == {
        "correct": 2, "retrieval_miss": 1, "generation_miss": 0, "integration_miss": 0}


# ---------------------------------------------------------------- CRAG rule step

def test_crag_rule_step():
    rs = crag_eval.rule_step
    assert rs("The answer is Ada!", "ada.", [], "simple") is None        # not an exact match: needs the judges
    assert rs(" ADA ", "Ada", [], "simple")[0] == "accurate"
    assert rs("Dr. Ada", "Ada", ["dr ada"], "simple")[0] == "accurate"    # alt answer, punctuation ignored
    assert rs("I don’t know.", "Ada", [], "simple")[0] == "missing"
    assert rs("", "Ada", [], "simple")[0] == "missing"
    assert rs(None, "Ada", [], "simple")[0] == "missing"
    assert rs("Ada", "Ada", [], "simple", error="timeout")[0] == "missing"
    assert rs("invalid question", "x", [], "false_premise")[0] == "accurate"
    assert rs("This is an invalid question.", "x", [], "false_premise")[0] == "accurate"
    assert rs("Invalid question", "Ada", [], "simple")[0] == "incorrect"
    assert rs("Invalid question", "invalid question", [], "simple")[0] == "accurate"
    assert rs("Probably the moon", "invalid question", [], "false_premise") is None


# ---------------------------------------------------------------- judge client

def test_judge_parsing():
    assert J.parse_vidore('```json\n{"explanation": "x", "judgment": "Partially Correct"}\n```') == "Partially Correct"
    assert J.parse_vidore('Sure. {"judgment": "Incorrect", "explanation": "no"} done') == "Incorrect"
    assert J.parse_vidore('judgment: Correct') == "Correct"
    assert J.parse_vidore("I cannot decide") is None
    assert J.parse_crag('{"judgment": "accurate"}') == "accurate"
    assert J.parse_crag('{"accuracy": false}') == "incorrect"
    assert J.parse_crag('{"judgment": "Incorrect"}') == "incorrect"
    assert J.parse_crag("garbage") is None


def test_judge_retries_cache_and_request_shape(monkeypatch, tmp_path):
    fj = FakeJudge(fail_first=2, fence=True)
    try:
        monkeypatch.setenv("CRB_JUDGE_BASE", fj.url)
        monkeypatch.setenv("CRB_JUDGE_KEY", "secret-key-xyz")
        monkeypatch.setenv("CRB_JUDGE_BACKOFF", "0")
        jd = J.Judge(cache=J.Cache(tmp_path / "c.sqlite"))
        v = jd.vidore("hr-1", "Q?", "gold", "answer OK")
        assert v.label == "Correct" and v.cost == 0.001 and not v.cached and v.error is None
        assert len(fj.requests) == 3                      # two failures, then success
        body = fj.requests[-1]
        assert body["model"] == "openai/gpt-5.2" and body["reasoning"] == {"effort": "medium"}
        assert body["temperature"] == 0 and body["usage"] == {"include": True}
        assert fj.auth[-1] == "Bearer secret-key-xyz"
        again = jd.vidore("hr-1", "Q?", "gold", "answer OK")
        assert again.cached and again.label == "Correct" and len(fj.requests) == 3
        jd.vidore("hr-1", "Q?", "gold", "different answer")   # other answer hash: a new call
        assert len(fj.requests) == 4
        c = jd.crag("crag-1", "Q?", "gold", "BOTH", "meta-llama/llama-3.1-70b-instruct")
        assert c.label == "accurate" and "reasoning" not in fj.requests[-1]
    finally:
        fj.stop()


def test_judge_without_key_fails_soft(monkeypatch, tmp_path):
    monkeypatch.delenv("CRB_JUDGE_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    jd = J.Judge(cache=J.Cache(tmp_path / "c.sqlite"))
    v = jd.vidore("q", "Q?", "g", "a")
    assert v.label is None and "key" in v.error.lower()


# ---------------------------------------------------------------- CRAG two-judge averaging

def test_crag_grade_two_judges(fake_judge):
    jd = J.Judge()
    item = lambda qid, text: {"qid": qid, "question": "Q?", "gold": "Ada", "alt_answers": [], "qtype": "simple",
                              "answer": text, "error": None}
    both, fails = crag_eval.grade(item("c1", "BOTH"), jd)
    assert not fails and [ln["score"] for ln in both] == [1.0, 1.0]
    assert crag_eval.outcome(both) == {"score": 1.0, "accurate": 1.0, "incorrect": 0.0, "missing": 0.0}
    half, _ = crag_eval.grade(item("c2", "ONLY35"), jd)
    o = crag_eval.outcome(half)
    assert o["score"] == 0.0 and o["accurate"] == 0.5 and o["incorrect"] == 0.5
    none, _ = crag_eval.grade(item("c3", "NEITHER"), jd)
    assert crag_eval.outcome(none)["score"] == -1.0
    calls = len(fake_judge.requests)
    rule, _ = crag_eval.grade(item("c4", "ada"), jd)           # exact match: no model call
    assert rule[0]["judge"] == "rule" and len(fake_judge.requests) == calls
    assert set(fake_judge.models()) == set(J.CRAG_MODELS)


# ---------------------------------------------------------------- latency / health

def test_latency_summary():
    ans = [line("a", "p", 3, i, first=1000 * (i + 1), total=2000 * (i + 1), warmup=i < 2) for i in range(10)]
    ans.append(line("b", "p", 3, 11, first=99999, total=99999, error="timeout"))
    ans.append(line("c", "p", 1, 12, retrieve=300, retrieved=["x"]))
    s = latency.summarize(ans, {"ingest": {"p": {"ok": True, "seconds": 12.34, "pages": 50}}})
    assert s["first_token"]["n"] == 8 and s["first_token"]["p50"] == 6500 and s["first_token"]["max"] == 10000
    assert s["retrieve"]["p50"] == 300 and s["errors"] == 1
    assert s["cold_start_ms"] == 1000
    assert sum(s["hist_first_token_500ms"]) == 8 and len(s["hist_first_token_500ms"]) == 20
    assert s["hist_first_token_500ms"][-1] == 1          # only the 10 s answer lands in the 9.5 s+ bin
    assert s["ingest"] == {"p": {"seconds": 12.3, "pages": 50}}


def test_health_summary_and_downsample():
    rows = [dict(r, _t=float(r["t"])) for r in health_rows(100)]
    h = latency.summarize_health(rows)
    assert h["available"] and h["temp_max"] == pytest.approx(74.8, abs=0.1)
    assert h["throttled_seconds"] == 10 and h["throttled_share"] == pytest.approx(0.1)
    assert h["throttled_by_phase"] == {"gold": 5, "e2e": 5}
    assert h["under_voltage"] is False and h["arm_mhz_min"] == 1000 and h["arm_mhz_max"] == 1800
    assert h["throttle_bands"] == [[40.0, 50.0]]
    assert [p["phase"] for p in h["phases"]] == ["ingest", "retrieval", "gold", "e2e", "crag"]
    big = [dict(r, _t=float(i)) for i, r in enumerate(health_rows(100) * 52)]  # 5200 samples
    hb = latency.summarize_health(big)
    assert len(hb["timeline"]["t"]) <= 2000 and len(hb["timeline"]["temp"]) == len(hb["timeline"]["t"])
    assert hb["samples"] == 5200
    assert latency.summarize_health([]) == {"available": False, "samples": 0}
    # a laptop: only cpu columns filled
    lap = [{"_t": float(i), "t": i, "cpu_pct": "20", "phase": "e2e"} for i in range(5)]
    assert latency.summarize_health(lap)["available"] is False


# ---------------------------------------------------------------- results.build

def test_build_full_run(tmp_path, monkeypatch, fake_judge):
    run = make_run(tmp_path, monkeypatch)
    res = R.build(run, judge=True, mode="official")
    m = res["metrics"]
    assert res["mode"] == "official" and res["comparable"] is True and res["status"] == "done"
    # retrieval: 14 of 20 perfect
    ret = m["retrieval"][HR]
    assert ret == {"ndcg10": 70.0, "recall10": 70.0, "n": 20, "pack_size": 20}
    # gold: 9 of 10 correct, sampled from a pack of 20
    g = m["gold"][HR]
    assert g["acc"] == 90.0 and g["n"] == 10 and g["pack_size"] == 20
    e = m["e2e"][HR]
    assert e["acc"] == 60.0 and e["n"] == 10
    values = [100.0] * 6 + [0.0] * 4
    assert e["ci95"] == pytest.approx(stats.bootstrap_ci(values, 20), abs=0.01) and e["ci95"] > 0
    assert e["partial"] == 10.0
    bc = e["by_content_type"]
    assert bc["Text"]["n"] == 5 and bc["Table"]["n"] == 5 and bc["Text"]["correct"] + bc["Table"]["correct"] == 6
    # diagnosis: retrieval miss x2 (15, 16), generation miss (7), integration miss (8), correct x6
    assert m["diagnosis"][HR] == {"correct": 6, "retrieval_miss": 2, "generation_miss": 1, "integration_miss": 1}
    # CRAG: scores +1, 0, +1, 0, -1, +1, -1
    c = m["crag"][CRAG]
    assert c["n"] == 7 and c["truthfulness"] == pytest.approx(100 / 7, abs=0.01)
    assert c["accurate"] == pytest.approx(50.0) and c["hallucinated"] == pytest.approx(35.71, abs=0.01)
    assert c["missing"] == pytest.approx(14.29, abs=0.01)
    assert c["by_qtype"]["comparison"]["n"] == 2
    scores = [100, 0, 100, 0, -100, 100, -100]
    assert c["ci95"] == pytest.approx(stats.bootstrap_ci([float(s) for s in scores], 10), abs=0.01)
    # latency, health, cost
    lat = m["latency"]
    assert lat["first_token"]["n"] == 7 + 7 + 4          # axes 2, 3, 4 minus 3 warm-up answers each
    assert lat["ingest"][HR] == {"seconds": 120.0, "pages": 200}
    assert m["health"]["available"] and m["health"]["throttled_seconds"] == 10
    cost = m["cost"]
    assert cost["measured"] and cost["system_usd"] == pytest.approx(0.10) and cost["tokens_in"] == 6500
    assert cost["context_cap_hits"] == 1 and cost["answers"] == 27
    assert cost["per_answer_usd"] == pytest.approx(0.10 / 27, abs=1e-6)
    # judges: 20 vidore + 6 crag calls, 0.001 each, rule lines are free
    assert len(fake_judge.requests) == 26
    assert cost["judge_usd"] == pytest.approx(0.026)
    # ranks
    rk = res["ranks"]
    assert rk[f"e2e.{HR}"]["rank"] == 8 and rk[f"e2e.{HR}"]["of"] == 9 and rk[f"e2e.{HR}"]["side"] == "below"
    assert rk[f"gold.{HR}"]["rank"] == 1 and rk[f"gold.{HR}"]["side"] == "above"
    assert rk[f"crag.{CRAG}"]["rank"] == 1
    assert f"retrieval.{HR}" in rk and f"gold.{CS}" not in rk
    # worst answers
    tags = {w["tag"] for w in res["worst"]}
    assert tags == {"retrieval_miss", "generation_miss", "integration_miss", "hallucinated"}
    assert {w["tag"] for w in res["worst"][:4]} == {"retrieval_miss", "hallucinated"}   # the two most common first
    assert all(set(w) == {"qid", "pack", "question", "gold", "answer", "tag"} for w in res["worst"])
    assert "run note from the harness" in res["notes"]
    # files
    on_disk = json.loads((run / "results.json").read_text())
    assert on_disk["metrics"]["e2e"][HR]["acc"] == 60.0
    jl = [json.loads(x) for x in (run / "judgments.jsonl").read_text().splitlines()]
    assert {"qid", "axis", "judge", "label", "score", "raw"} <= set(jl[0])
    assert {ln["judge"] for ln in jl if ln["axis"] == 4} == {"rule", *J.CRAG_MODELS}
    assert "test-key-123" not in (run / "results.json").read_text()
    assert "test-key-123" not in (run / "judgments.jsonl").read_text()


def test_rejudge_is_free_and_judge_false_reuses(tmp_path, monkeypatch, fake_judge):
    run = make_run(tmp_path, monkeypatch)
    first = R.build(run, judge=True)
    n = len(fake_judge.requests)
    again = R.build(run, judge=True)
    assert len(fake_judge.requests) == n                       # all from the cache
    assert again["metrics"]["e2e"][HR]["acc"] == first["metrics"]["e2e"][HR]["acc"]
    assert again["metrics"]["cost"]["judge_usd"] == pytest.approx(0.026)
    reuse = R.build(run, judge=False)
    assert len(fake_judge.requests) == n
    assert reuse["metrics"]["crag"][CRAG]["truthfulness"] == first["metrics"]["crag"][CRAG]["truthfulness"]


def test_build_without_judge_says_not_judged(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, health=False, proxy=False)
    res = R.build(run, judge=False)
    m = res["metrics"]
    assert m["retrieval"][HR]["ndcg10"] == 70.0                  # needs no judge
    assert m["e2e"][HR]["acc"] is None and m["e2e"][HR]["n"] == 0 and m["e2e"][HR]["answered"] == 10
    assert m["crag"][CRAG]["truthfulness"] is None
    assert m["diagnosis"] == {} and res["worst"] == []
    assert [k for k in res["ranks"]] == [f"retrieval.{HR}"]
    assert m["cost"]["measured"] is False and m["health"]["available"] is False
    assert any("not judged" in n.lower() for n in res["notes"])
    assert any("cost not measured" in n.lower() for n in res["notes"])
    assert not (run / "judgments.jsonl").exists()


def test_build_without_key_keeps_cached_only(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    monkeypatch.delenv("CRB_JUDGE_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("CRB_JUDGE_CACHE", str(tmp_path / "empty.sqlite"))
    res = R.build(run, judge=True)
    assert res["metrics"]["e2e"][HR]["acc"] is None
    assert res["judge"]["failed"] > 0 and any("could not be judged" in n or "failed" in n for n in res["notes"])


def test_smoke_mode_uses_cheap_judge(tmp_path, monkeypatch, fake_judge):
    run = make_run(tmp_path, monkeypatch, mode="smoke")
    res = R.build(run, judge=True, mode="official")            # run.json says smoke: it wins over the default
    assert res["mode"] == "smoke" and res["comparable"] is False
    assert set(fake_judge.models()) == {"anthropic/claude-haiku-5.5"}
    assert any("Practice run" in n for n in res["notes"])
    res2 = R.build(run, judge=True, mode="official", judge_model="some/other-judge")
    assert res2["comparable"] is False and "some/other-judge" in fake_judge.models()


def test_dev_packs_are_not_comparable(tmp_path, monkeypatch, fake_judge):
    run = make_run(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text().splitlines()]
    for r in rows:
        r["pack"] = {HR: "dev_vidore", CRAG: "dev_crag"}[r["pack"]]
    write_jsonl(run / "answers.jsonl", rows)
    hr_q, hr_qrels = vidore_questions("dev_vidore", "hr", 20)
    cq = [dict(q, pack="dev_crag") for q in crag_questions()]
    monkeypatch.setattr(R, "_pack_api", lambda: FakePacks({"dev_vidore": hr_q, "dev_crag": cq}, {"dev_vidore": hr_qrels}))
    res = R.build(run, judge=True)
    assert res["comparable"] is False and res["ranks"] == {}
    assert res["metrics"]["e2e"]["dev_vidore"]["acc"] == 60.0 and res["metrics"]["crag"]["dev_crag"]["n"] == 7


def test_missing_pack_and_failed_ingest_are_graceful(tmp_path, monkeypatch, fake_judge):
    run = make_run(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "_pack_api", lambda: FakePacks({CRAG: crag_questions()}, {}))   # HR pack missing
    sess = json.loads((run / "run.json").read_text())
    sess["status"] = "incomplete"
    sess["ingest"][HR] = {"ok": False, "seconds": 3.0, "pages": 0, "error": "adapter said no"}
    (run / "run.json").write_text(json.dumps(sess))
    res = R.build(run, judge=True)
    assert res["metrics"]["retrieval"] == {} and res["metrics"]["e2e"][HR]["acc"] is None
    assert res["metrics"]["crag"][CRAG]["n"] == 7
    assert any("could not be read" in n for n in res["notes"])
    assert any("incomplete" in n for n in res["notes"]) and any("adapter said no" in n for n in res["notes"])
    assert res["metrics"]["latency"]["ingest"][HR]["ok"] is False


def test_build_on_empty_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_pack_api", lambda: FakePacks({}, {}))
    run = tmp_path / "empty"
    run.mkdir()
    res = R.build(run, judge=False)
    assert res["ranks"] == {} and res["worst"] == [] and (run / "results.json").exists()

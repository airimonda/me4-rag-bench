"""crb.runner end to end against a fake adapter, with fake data packs / health / scoring / report modules."""
from __future__ import annotations

import json
import socket
import sys
import types
from pathlib import Path

import pytest

from crb import runner
from crb.config import RunConfig
from test_adapter import FakeAdapter

PACKS = {"dev_vidore": "vidore", "dev_crag": "crag"}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_fake_modules(data: Path, calls: dict) -> dict:
    """Stand-ins for crb.data.packs, crb.health, crb.score.results, crb.report.* (owned by other people)."""
    pages = data / "packs" / "dev_vidore" / "pages"
    pages.mkdir(parents=True)
    for pid in ("d__p1", "d__p2"):
        (pages / f"{pid}.png").write_bytes(b"png")
        (pages / f"{pid}.md").write_text(f"# markdown of {pid}\nbody\n")
    (data / "packs" / "dev_crag").mkdir(parents=True)
    (data / "samples").mkdir()
    questions = {
        "dev_vidore": [{"qid": f"dv-{i}", "pack": "dev_vidore", "question": f"vidore question {i}", "gold_pages": ["d__p1", "d__p2"],
                        "answer": "a", "qtype": "x", "query_time": None} for i in range(4)],
        "dev_crag": [{"qid": f"dc-{i}", "pack": "dev_crag", "question": f"crag question {i}", "gold_pages": [],
                      "answer": "a", "qtype": "x", "query_time": "03/05/2024, 23:18:31 PT"} for i in range(3)],
    }
    sample = {"seed": 7, "packs": {
        "dev_vidore": {"retrieval": [q["qid"] for q in questions["dev_vidore"]],
                       "answer": [q["qid"] for q in questions["dev_vidore"]][:3]},
        "dev_crag": {"answer": [q["qid"] for q in questions["dev_crag"]]}}}
    (data / "samples" / "dev.json").write_text(json.dumps(sample))

    packs = types.ModuleType("crb.data.packs")
    packs.data_dir = lambda: data
    packs.load_questions = lambda pack: questions[pack]
    packs.load_sample = lambda name: sample
    packs.load_qrels = lambda pack: {}
    packs.page_paths = lambda pack, pid: (pages / f"{pid}.png", pages / f"{pid}.md")
    packs.crag_pages = lambda pack, qid: [{"path": f"/w/{qid}/{k}.html", "page_name": f"n{k}", "page_url": "u",
                                           "page_snippet": "s", "page_last_modified": "m"} for k in range(5)]
    packs.verify = lambda pack, quick=False: []

    class Sampler:
        def __init__(self, data_dir, adapter_port, out_csv, state):
            self.out, self.state = Path(out_csv), state
            calls["sampler"].append(self)

        def start(self):
            self.out.write_text("t,phase\n")

        def stop(self):
            with open(self.out, "a") as f:
                f.write(f"{self.state.t():.1f},{self.state.phase}\n")

    health = types.ModuleType("crb.health")
    health.HealthSampler = Sampler

    def build(run_dir, judge=True, mode="official"):
        calls["build"].append((Path(run_dir), judge, mode))
        res = {"mode": mode, "ranks": {"retrieval.dev_vidore": {"rank": 3, "of": 9, "median": 50.0, "side": "above"}},
               "metrics": {"latency": {"first_token": {"p90": 123}}, "cost": {"measured": False}}}
        (Path(run_dir) / "results.json").write_text(json.dumps(res))
        return res

    score = types.ModuleType("crb.score.results")
    score.build = build
    html = types.ModuleType("crb.report.html")
    html.write = lambda run_dir, results: (Path(run_dir) / "report.html").write_text("<html>") or Path(run_dir) / "report.html"
    summary = types.ModuleType("crb.report.summary")
    summary.write = lambda run_dir, results: (Path(run_dir) / "summary.md").write_text("ok") or Path(run_dir) / "summary.md"
    return {"crb.data.packs": packs, "crb.health": health, "crb.score.results": score,
            "crb.report.html": html, "crb.report.summary": summary}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    calls = {"sampler": [], "build": []}
    data = tmp_path / "data"
    for name, mod in make_fake_modules(data, calls).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("CRB_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("CRB_DATA", str(data))
    monkeypatch.setenv("CRB_PROXY_PORT", str(free_port()))
    monkeypatch.setenv("CRB_JUDGE_KEY", "sk-not-real")
    monkeypatch.setenv("CRB_CONFIG_DIR", str(tmp_path / "cfg"))
    fake = FakeAdapter()
    yield types.SimpleNamespace(fake=fake, calls=calls, tmp=tmp_path)
    fake.close()


def answers(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "answers.jsonl").read_text().splitlines()]


def only_run_dir(tmp: Path) -> Path:
    (d,) = list((tmp / "runs" / "s123").iterdir())
    return d


def test_smoke_run_all_four_axes(rig, capsys):
    cfg = RunConfig(student="s123", adapter=rig.fake.url, mode="smoke")
    assert runner.run(cfg) == 0                       # includes the real pre-flight against the fake adapter
    d = only_run_dir(rig.tmp)
    rows = answers(d)
    by_axis = {a: [r for r in rows if r["axis"] == a] for a in (1, 2, 3, 4)}
    assert {a: len(v) for a, v in by_axis.items()} == {1: 4, 2: 3, 3: 3, 4: 3}
    assert all(r["answer"] is None and r["latency_ms"]["retrieve"] > 0 and r["retrieved"] == ["p1", "p2", "p3"] for r in by_axis[1])
    for a in (2, 3, 4):
        assert all(r["answer"] and r["latency_ms"]["first_token"] is not None for r in by_axis[a])
    # warmup: first 3 per (pack, axis)
    assert [r["warmup"] for r in by_axis[1]] == [True, True, True, False]
    assert all(r["warmup"] for r in by_axis[2])
    assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)
    # request shapes
    posts = [(p, b) for p, b in rig.fake.calls if p == "/answer" and b["collection"] != "probe"]
    gold = next(b for p, b in posts if b["mode"] == "gold")
    assert [c["page_id"] for c in gold["context"]] == ["d__p1", "d__p2"]
    assert gold["context"][0]["markdown"].startswith("# markdown of d__p1") and gold["context"][0]["image_path"].endswith("d__p1.png")
    web = next(b for p, b in posts if b["mode"] == "web")
    assert len(web["pages"]) == 5 and web["pages"][0]["page_name"] == "n0" and web["query_time"].startswith("03/05/2024")
    assert "context" not in next(b for p, b in posts if b["mode"] == "e2e")
    assert rig.fake.ingests == 1 and rig.fake.calls[[p for p, _ in rig.fake.calls].index("/ingest")][1]["collection"] == "dev_vidore"
    # run directory files
    sess = json.loads((d / "run.json").read_text())
    assert sess["status"] == "done" and sess["student"] == "s123" and sess["mode"] == "smoke" and sess["sample"] == "dev"
    assert sess["adapter_health"]["name"] == "fake" and sess["ingest"]["dev_vidore"]["ok"] is True and sess["end"]
    assert (d / "run.log").read_text().strip()
    assert rig.calls["build"] == [(d, True, "smoke")]
    assert (d / "report.html").exists() and (d / "summary.md").exists()
    assert (d / "health.csv").read_text().splitlines()[0] == "t,phase"
    out = capsys.readouterr().out
    assert "rank 3 of 9" in out and "p90: 123 ms" in out and "not measured" in out and "report.html" in out


def test_interrupt_then_resume_skips_done_work(rig, monkeypatch):
    cfg = RunConfig(student="s123", adapter=rig.fake.url, mode="smoke")
    real = runner.Runner._ask_one
    n = {"calls": 0}

    def flaky(self, pack, qid, axis):
        n["calls"] += 1
        if n["calls"] == 7:
            raise KeyboardInterrupt
        return real(self, pack, qid, axis)

    monkeypatch.setattr(runner.Runner, "_ask_one", flaky)
    assert runner.run(cfg, skip_preflight=True) == 1
    d = only_run_dir(rig.tmp)
    assert json.loads((d / "run.json").read_text())["status"] == "interrupted"
    assert len(answers(d)) == 6 and not rig.calls["build"]      # kept, not scored
    first_t = answers(d)[-1]["t"]

    monkeypatch.setattr(runner.Runner, "_ask_one", real)
    asked_before = len([1 for p, _ in rig.fake.calls if p in ("/retrieve", "/answer")])
    assert runner.run(RunConfig(student="x", adapter=None, mode="official", resume=d), skip_preflight=True) == 0
    rows = answers(d)
    keys = [(r["qid"], r["axis"]) for r in rows]
    assert len(rows) == 13 and len(set(keys)) == 13              # all 4 axes, no duplicates
    assert {r["axis"] for r in rows} == {1, 2, 3, 4}
    asked_after = len([1 for p, _ in rig.fake.calls if p in ("/retrieve", "/answer")])
    assert asked_after - asked_before == 13 - 6                 # only the missing questions were asked
    assert rig.fake.ingests == 1                                 # not re-ingested
    assert rows[6]["t"] >= first_t                               # clock continues across the resume
    sess = json.loads((d / "run.json").read_text())
    assert sess["status"] == "done" and sess["mode"] == "smoke"  # taken from the first session
    health = (d / "health.csv").read_text().splitlines()
    assert health[0] == "t,phase" and len(health) == 3 and not (d / ".health.resume.csv").exists()
    assert rig.calls["build"][-1][0] == d


def test_adapter_errors_never_stop_the_run(rig):
    def answer_fn(handler, body):
        handler._send(500, {"error": "kaput"})

    rig.fake.answer_fn = answer_fn
    assert runner.run(RunConfig(student="s123", adapter=rig.fake.url, mode="smoke"), skip_preflight=True) == 0
    rows = answers(only_run_dir(rig.tmp))
    assert len(rows) == 13
    assert all(r["error"] and "HTTP 500" in r["error"] for r in rows if r["axis"] in (2, 3, 4))
    assert all(r["error"] is None for r in rows if r["axis"] == 1)


def test_scoring_failure_keeps_answers_and_explains(rig, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(sys.modules["crb.score.results"], "build", boom)
    rc = runner.run(RunConfig(student="s123", adapter=rig.fake.url, mode="smoke"), skip_preflight=True)
    d = only_run_dir(rig.tmp)
    assert rc == 1 and len(answers(d)) == 13
    out = capsys.readouterr().out
    assert "judge exploded" in out and f"crb score {d}" in out


def test_missing_health_module_is_noted_not_fatal(rig, monkeypatch):
    monkeypatch.setitem(sys.modules, "crb.health", None)         # import raises ImportError
    assert runner.run(RunConfig(student="s123", adapter=rig.fake.url, mode="smoke"), skip_preflight=True) == 0
    d = only_run_dir(rig.tmp)
    sess = json.loads((d / "run.json").read_text())
    assert any("health" in n for n in sess["notes"])


def test_preflight_fails_plainly_when_adapter_is_down(rig, capsys):
    rig.fake.close()
    rc = runner.run(RunConfig(student="s123", adapter=rig.fake.url, mode="smoke", no_judge=True))
    out = capsys.readouterr().out
    assert rc == 2 and "FAIL" in out and "fix:" in out and "cannot connect" in out
    assert not (rig.tmp / "runs").exists()                       # nothing created


def test_preflight_reports_missing_judge_key(rig, monkeypatch, capsys):
    monkeypatch.delenv("CRB_JUDGE_KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert runner.run(RunConfig(student="s123", adapter=rig.fake.url, mode="smoke")) == 2
    assert "no judge API key" in capsys.readouterr().out
    assert runner.run(RunConfig(student="s123", adapter=rig.fake.url, mode="smoke", no_judge=True)) == 0


def test_plan_modes(rig, monkeypatch):
    packs = sys.modules["crb.data.packs"]
    assert runner.build_plan("smoke").sample_name == "dev"
    # without dev samples on disk: first 10 per axis of class-v1
    big = {"packs": {"vidore_v3_hr": {"retrieval": [f"hr-{i}" for i in range(30)], "answer": [f"hr-{i}" for i in range(12)]},
                     "crag_task1": {"answer": [f"crag-{i}" for i in range(40)]}}}
    monkeypatch.setattr(packs, "load_sample", lambda name: big)
    (rig.tmp / "data" / "samples" / "dev.json").unlink()
    p = runner.build_plan("smoke")
    assert [len(v) for v in (p.packs["vidore_v3_hr"]["retrieval"], p.packs["vidore_v3_hr"]["answer"], p.packs["crag_task1"]["answer"])] == [10, 10, 10]
    assert p.count(1) == 10 and p.count(4) == 10
    off = runner.build_plan("official")
    assert off.sample_name == "class-v1" and len(off.packs["crag_task1"]["answer"]) == 40
    monkeypatch.setattr(packs, "load_questions", lambda pack: [{"qid": f"{pack}-{i}"} for i in range(55)])
    assert len(runner.build_plan("full").packs["vidore_v3_hr"]["answer"]) == 55

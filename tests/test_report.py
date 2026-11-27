"""Report tests: results.json -> report.html / summary.md. Offline; the judge is the fake server from test_score."""
from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from crb.report import html as report_html
from crb.report import summary as report_summary
from crb.score import results as R
from test_score import (CRAG, CS, HR, FakeJudge, FakePacks, fake_judge, health_rows,  # noqa: F401
                        line, make_run, vidore_questions, write_health, write_jsonl)

TEMPLATE = Path(report_html.__file__).with_name("template.html")


def data_block(page: str) -> str:
    m = re.search(r'<script id="crb-data" type="application/json">(.*?)</script>', page, flags=re.S)
    assert m, "data block missing"
    return m.group(1)


@pytest.fixture
def built(tmp_path, monkeypatch, fake_judge):
    run = make_run(tmp_path, monkeypatch)
    return run, R.build(run, judge=True)


def test_html_contains_data_and_is_clean(built):
    run, res = built
    path = report_html.write(run, res)
    assert path == run / "report.html" and path.exists()
    page = path.read_text(encoding="utf-8")
    raw = data_block(page)
    assert "</" not in raw and "<!--" not in raw
    assert json.loads(raw) == json.loads(json.dumps(res, default=str))
    assert "example" not in page.lower()
    assert "__CRB_DATA__" not in page
    # offline: nothing is fetched
    assert "<link" not in page and not re.search(r"<script[^>]*\ssrc=", page)
    assert "@import" not in page and "url(http" not in page and "fetch(" not in page and "XMLHttpRequest" not in page
    assert page.count("</script>") == 2 and page.count("<script") == 2
    for needle in ("Results", "Why: which layer broke", "Speed", "Cost", "Pi health", "Answers to read first", "Files",
                   "Show as a table", "Above the median of published systems on"):
        assert needle in page


def test_hostile_text_cannot_break_out(built):
    run, res = built
    res["worst"][0]["answer"] = '</script><img src=x onerror=alert(1)> <!--  '
    res["session"]["student"] = "</script><b>x"
    page = report_html.write(run, res).read_text(encoding="utf-8")
    assert page.count("</script>") == 2 and "<img src=x" not in page
    assert json.loads(data_block(page))["worst"][0]["answer"].startswith("</script><img")


def test_template_script_is_valid_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    page = TEMPLATE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", page, flags=re.S)
    assert len(scripts) == 1
    js = Path(__file__).parent / "_template_check.js"
    try:
        js.write_text(scripts[0], encoding="utf-8")
        r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=20)
        assert r.returncode == 0, r.stderr
    finally:
        js.unlink(missing_ok=True)


def test_template_has_no_hardcoded_numbers_or_banner():
    page = TEMPLATE.read_text(encoding="utf-8")
    low = page.lower()
    assert "example" not in low and "mockup" not in low
    for stale in ("2024-00000", "minimax", "78.4", "4 Dec 2026", "Pi 4 · 4 GB"):
        assert stale not in page
    assert page.count("__CRB_DATA__") == 1
    assert "Practice run — not comparable to published results" in page


def test_missing_pieces_still_render(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, health=False, proxy=False)
    res = R.build(run, judge=False)
    page = report_html.write(run, res).read_text(encoding="utf-8")
    data = json.loads(data_block(page))
    assert data["metrics"]["health"]["available"] is False
    assert data["metrics"]["cost"]["measured"] is False
    assert data["metrics"]["e2e"][HR]["acc"] is None
    for s in ("not judged", "no Pi stats", "not measured"):   # strings the page shows for these cases
        assert s in page
    md = report_summary.write(run, res).read_text(encoding="utf-8")
    assert "not judged" in md and "Cost: not measured" in md and "no Pi stats" in md


def test_smoke_marks_not_comparable(tmp_path, monkeypatch, fake_judge):
    run = make_run(tmp_path, monkeypatch, mode="smoke")
    res = R.build(run, judge=True)
    assert res["comparable"] is False
    page = report_html.write(run, res).read_text(encoding="utf-8")
    assert json.loads(data_block(page))["comparable"] is False
    md = report_summary.write(run, res).read_text(encoding="utf-8")
    assert "Practice run — not comparable to published results" in md


def test_summary_md(built):
    run, res = built
    path = report_summary.write(run, res)
    md = path.read_text(encoding="utf-8")
    assert path == run / "summary.md"
    assert "Above the median of published systems on" in md and f"{HR}" in md
    assert re.search(r"\| .*e2e|Layer 3 · whole system · vidore_v3_hr \| % correct \| 60\.0 ", md)
    assert "Speed: first token" in md and "Pi health: peak" in md and "Cost:" in md
    n_above = sum(1 for r in res["ranks"].values() if r["side"] == "above")
    assert f"on {n_above} of {len(res['ranks'])} tests" in md


# ---------------------------------------------------------------- bigger synthetic run (also used for the sample report)

def make_big_run(run: Path, scale: float = 1.0, seed: int = 11) -> dict:
    """A realistic synthetic run. Sets R._pack_api; callers restore it (tests use monkeypatch)."""
    rng = random.Random(seed)
    run.mkdir(parents=True, exist_ok=True)
    n_hr, n_cs, n_crag = max(8, int(318 * scale)), max(8, int(215 * scale)), max(12, int(1335 * scale))
    ans_v, ans_c = max(5, int(100 * scale)), max(8, int(400 * scale))
    hr_q, hr_r = vidore_questions(HR, "hr", n_hr)
    cs_q, cs_r = vidore_questions(CS, "cs", n_cs)
    for q in hr_q + cs_q:
        q["content_type"] = rng.sample(["Text", "Table", "Chart", "Image", "Infographic"], k=rng.choice([1, 1, 2]))
    types = ["simple", "simple_w_condition", "comparison", "aggregation", "set", "multi-hop", "post-processing",
             "false_premise"]
    crag_q = [{"qid": f"crag-{i}", "pack": CRAG, "question": f"Question number {i} about the web page?",
               "answer": f"answer {i}" if i % 11 else "invalid question", "alt_answers": [],
               "qtype": types[i % len(types)] if i % 11 else "false_premise", "content_type": [],
               "query_time": "03/05/2024, 23:18:31 PT", "gold_pages": []} for i in range(1, n_crag + 1)]
    R._pack_api = lambda: FakePacks({HR: hr_q, CS: cs_q, CRAG: crag_q}, {HR: hr_r, CS: cs_r})

    rows, t = [], 0.0
    hit: dict[str, bool] = {}
    for pack, short, n, p_hit in ((HR, "hr", n_hr, 0.62), (CS, "cs", n_cs, 0.8)):
        for i in range(1, n + 1):
            t += 0.4
            qid = f"{short}-{i}"
            hit[qid] = rng.random() < p_hit
            if hit[qid]:
                pos = rng.choice([0, 0, 0, 1, 2, 4])
                got = [f"x{k}" for k in range(pos)] + [f"d{i}__p1", f"d{i}__p2"] + ["y1", "y2"]
            else:
                got = ["zz1", "zz2", f"d{i}__p2" if rng.random() < 0.15 else "zz3"]
            rows.append(line(qid, pack, 1, t, retrieved=got[:10], retrieve=int(rng.lognormvariate(5.3, 0.4)),
                             warmup=i <= 3))
    def pick(weights):
        return rng.choices(["OK", "PARTIAL", "WRONG"], weights=weights)[0]
    for pack, short, n, na in ((HR, "hr", n_hr, ans_v), (CS, "cs", n_cs, ans_v)):
        sample = sorted(rng.sample(range(1, n + 1), na))
        for axis in (2, 3):
            for k, i in enumerate(sample):
                t += 3
                qid = f"{short}-{i}"
                w = (72, 10, 18) if axis == 2 else ((52, 14, 34) if hit[qid] else (8, 12, 80))
                if short == "cs":
                    w = (82, 8, 10) if axis == 2 else ((70, 10, 20) if hit[qid] else (10, 10, 80))
                ft = int(rng.lognormvariate(7.5, 0.45))
                rows.append(line(qid, pack, axis, t, answer=f"{pick(w)} answer for {qid}", retrieved=[],
                                 first=ft, total=ft + int(rng.lognormvariate(8.0, 0.5)), warmup=k < 3,
                                 error="timeout after 120 s" if (axis == 3 and k == 7) else None))
    for k, i in enumerate(sorted(rng.sample(range(1, n_crag + 1), ans_c))):
        t += 4
        gold = crag_q[i - 1]["answer"]
        r = rng.random()
        text = (gold if r < 0.08 else "I don't know" if r < 0.30 else "BOTH say so" if r < 0.55
                else "ONLY35 says so" if r < 0.65 else "NEITHER fits")
        ft = int(rng.lognormvariate(7.9, 0.55))
        rows.append(line(f"crag-{i}", CRAG, 4, t, answer=text, retrieved=[], first=ft,
                         total=ft + int(rng.lognormvariate(8.3, 0.5)), warmup=k < 3))
    write_jsonl(run / "answers.jsonl", rows)

    n = int(5220 * scale) or 60
    hrows = []
    cuts = [0.0, 0.14, 0.18, 0.38, 0.63, 1.0]
    names = ["ingest", "retrieval", "gold", "e2e", "crag"]
    bands = [(0.72, 0.76), (0.83, 0.855), (0.92, 0.945)]
    for i in range(n):
        f = i / n
        ph = names[max(j for j in range(5) if f >= cuts[j])]
        thr = any(a <= f < b for a, b in bands)
        base = 52 + 12 * min(f / 0.14, 1) + 8 * max(0, f - 0.14) + (6 * (f - 0.63) / 0.37 if f > 0.63 else 0)
        temp = round(base + rng.gauss(0, 0.8), 1)
        cpu = max(3, min(100, (82 if ph == "ingest" else 55 if ph == "retrieval" else 38 if ph != "crag" else 47) + rng.gauss(0, 12)))
        hrows.append({"t": i, "iso": "", "temp_c": temp, "throttled_hex": "0x4" if thr else "0x0",
                      "under_voltage_now": 0, "freq_capped_now": int(thr), "throttled_now": int(thr),
                      "soft_temp_now": 0, "arm_mhz": int(rng.uniform(1000, 1300)) if thr else 1800, "core_v": 0.9,
                      "cpu_pct": round(cpu, 1), "cpu_max_core_pct": round(min(100, cpu + 30), 1), "load1": 1.2,
                      "mem_avail_mb": int(1100 - 500 * f + rng.gauss(0, 20)), "swap_used_mb": 0,
                      "adapter_rss_mb": 420, "disk_free_mb": 20000, "net_rtt_ms": int(rng.gauss(38, 8)), "phase": ph})
    write_health(run / "health.csv", hrows)
    proxy = [{"t": float(i), "model": "minimax/minimax-m3.1-flash", "prompt_tokens": 9000 + i, "completion_tokens": 1300,
              "cost": 0.0024, "qid": "x"} for i in range(int(600 * scale) or 10)]
    write_jsonl(run / "proxy.jsonl", proxy)
    (run / "run.json").write_text(json.dumps({
        "student": "2024-00001", "mode": "official", "sample": "class-v1", "status": "done", "crb_version": "0.1.0",
        "pi": {"model": "Raspberry Pi 4 Model B Rev 1.4", "ram_gb": 4, "cooler": "cooler"},
        "adapter": {"url": "http://127.0.0.1:8000", "health": {"name": "hybrid text RAG", "version": "0.3"}},
        "start": "2026-12-04T08:00:00Z", "end": "2026-12-04T09:27:00Z",
        "ingest": {HR: {"ok": True, "seconds": 330.0, "pages": 1110, "error": None},
                   CS: {"ok": True, "seconds": 360.0, "pages": 1360, "error": None}},
        "notes": []}))
    return {"hit": hit}


def test_big_run_report(tmp_path, monkeypatch, fake_judge):
    monkeypatch.setattr(R, "_pack_api", R._pack_api)   # restore after make_big_run replaces it
    run = tmp_path / "big"
    make_big_run(run, scale=0.12)
    res = R.build(run, judge=True)
    m = res["metrics"]
    assert m["retrieval"][HR]["n"] == int(318 * 0.12) and 0 < m["retrieval"][HR]["ndcg10"] < 100
    assert m["latency"]["errors"] >= 1
    assert len(m["health"]["timeline"]["t"]) <= 2000
    page = report_html.write(run, res).read_text(encoding="utf-8")
    assert len(page) < 2_000_000
    report_summary.write(run, res)
    assert (run / "summary.md").exists()

"""summary.md: the verdict as text, for slides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

LAYERS = (("retrieval", "Layer 1 · finds the right pages", "ndcg10", "nDCG@10"),
          ("gold", "Layer 2 · answers with the right pages", "acc", "% correct"),
          ("e2e", "Layer 3 · whole system", "acc", "% correct"),
          ("crag", "Layer 4 · avoids making things up", "truthfulness", "truthfulness"))


def _f(v: Any, d: int = 1) -> str:
    return "–" if not isinstance(v, (int, float)) or isinstance(v, bool) else f"{v:.{d}f}"


def _sec(ms: Any) -> str:
    return "–" if not isinstance(ms, (int, float)) else f"{ms / 1000:.1f} s"


def render(results: dict) -> str:
    m = results.get("metrics") or {}
    ranks = results.get("ranks") or {}
    sess = results.get("session") or {}
    mode = results.get("mode", "official")
    out: list[str] = ["# Class RAG Benchmark results", ""]
    meta = [str(sess.get("student") or "unknown student"), f"mode {mode}", f"sample {sess.get('sample') or 'class-v1'}"]
    out.append(" · ".join(meta))
    out.append("")
    if results.get("comparable") is False or mode == "smoke":
        out += ["> Practice run — not comparable to published results", ""]
    if results.get("status") in ("interrupted", "incomplete", "running"):
        out += [f"> Run {results['status']}: scores cover only the questions answered so far.", ""]

    sides = [r.get("side") for r in ranks.values()]
    if sides:
        out.append(f"**Above the median of published systems on {sides.count('above')} of {len(sides)} tests, "
                   f"below on {sides.count('below')}.**")
    else:
        out.append("**No test could be compared with the published systems yet.**")
    out.append("")
    out += ["| Test | Metric | You | Rank | vs median |", "|---|---|---|---|---|"]
    for layer, title, field, unit in LAYERS:
        for pack, mm in sorted((m.get(layer) or {}).items()):
            v = mm.get(field)
            if v is None:
                out.append(f"| {title} · {pack} | {unit} | not judged | – | – |")
                continue
            ci = mm.get("ci95")
            you = _f(v) + (f" ± {_f(ci)}" if ci else "")
            rk = ranks.get(f"{layer}.{pack}")
            rank = f"{rk['rank']} of {rk['of']}" if rk else "–"
            side = f"{rk['side']} ({_f(rk['median'])})" if rk else "–"
            out.append(f"| {title} · {pack} | {unit} | {you} | {rank} | {side} |")
    out.append("")

    ft = (m.get("latency") or {}).get("first_token") or {}
    tot = (m.get("latency") or {}).get("total") or {}
    out.append(f"- Speed: first token p50 {_sec(ft.get('p50'))}, p90 {_sec(ft.get('p90'))}; "
               f"total answer p90 {_sec(tot.get('p90'))}.")
    c = m.get("cost") or {}
    if c.get("measured") or isinstance(c.get("judge_usd"), (int, float)):
        bits = []
        if isinstance(c.get("system_usd"), (int, float)):
            bits.append(f"${c['system_usd']:.2f} your system")
        if isinstance(c.get("judge_usd"), (int, float)):
            bits.append(f"${c['judge_usd']:.2f} judges")
        out.append("- Cost: " + ", ".join(bits) + ".")
    else:
        out.append("- Cost: not measured.")
    h = m.get("health") or {}
    if h.get("available"):
        thr = h.get("throttled_share")
        out.append(f"- Pi health: peak {_f(h.get('temp_max'))} °C, "
                   + ("throttle flags not read" if thr is None else f"throttled {thr * 100:.1f}% of the run")
                   + (", under-voltage seen" if h.get("under_voltage") else "") + ".")
    else:
        out.append("- Pi health: no Pi stats.")
    notes = results.get("notes") or []
    if notes:
        out += ["", "Notes:"] + [f"- {n}" for n in notes]
    out.append("")
    return "\n".join(out)


def write(run_dir: Path, results: dict) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "summary.md"
    tmp = run_dir / "summary.md.tmp"
    tmp.write_text(render(results), encoding="utf-8")
    os.replace(tmp, out)
    return out

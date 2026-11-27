"""Build results.json (and judgments.jsonl) from a run directory.

    build(run_dir, judge=True, mode="official") -> dict

Reads run.json, answers.jsonl, health.csv, proxy.jsonl, judgments.jsonl (when not judging) and the
packs through crb.data.packs (imported lazily). Writes judgments.jsonl (when judging) and results.json.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import literature
from . import cost as cost_mod
from . import crag_eval, diagnosis, judge as J, latency, retrieval, stats

MODES = ("smoke", "official", "full")
AXIS_KEY = {1: "retrieval", 2: "gold", 3: "e2e", 4: "crag"}
CONTENT_TYPES_NOTE = "A question with several evidence types counts once in each."


# ------------------------------------------------------------------ helpers

def _pack_api():
    """crb.data.packs, imported lazily (tests monkeypatch this)."""
    from ..data import packs  # noqa: WPS433
    return packs


def is_crag(pack: str) -> bool:
    return "crag" in pack.lower()


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def resolve_mode(mode: str, session: Mapping[str, Any]) -> str:
    """The run's own mode wins when the caller left the default ("official")."""
    m = (mode or "official").lower()
    sm = str(session.get("mode") or "").lower()
    if m == "official" and sm in MODES:
        return sm
    return m if m in MODES else "official"


def _r2(x: float | None) -> float | None:
    return None if x is None else round(x, 2)


def _last_by_key(answers: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], dict] = {}
    for a in answers:
        try:
            out[(int(a["axis"]), str(a["qid"]))] = dict(a)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _short(text: Any, n: int = 400) -> str:
    s = "" if text is None else str(text)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _gold_pages(q: Mapping[str, Any], qrel: Mapping[str, float] | None) -> set[str]:
    if qrel:
        pos = {p for p, s in qrel.items() if s > 0}
        if pos:
            return pos
    return {str(p) for p in (q.get("gold_pages") or [])}


# ------------------------------------------------------------------ judging

def _judge_jobs(ans: Mapping[tuple[int, str], dict], questions: Mapping[str, Mapping[str, dict]]):
    jobs = []
    for (axis, qid), a in sorted(ans.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if axis not in (2, 3, 4):
            continue
        pack = a.get("pack")
        q = (questions.get(pack) or {}).get(qid)
        if q is None:
            continue
        if axis == 4 and is_crag(pack) or axis in (2, 3) and not is_crag(pack):
            jobs.append((axis, qid, pack, q, a))
    return jobs


def _run_judging(jobs, judge: J.Judge, vidore_model: str, crag_models: Sequence[str]):
    def one(job):
        axis, qid, pack, q, a = job
        if axis == 4:
            item = {"qid": qid, "question": q.get("question", ""), "gold": q.get("answer"),
                    "alt_answers": q.get("alt_answers") or [], "qtype": q.get("qtype"),
                    "answer": a.get("answer"), "error": a.get("error")}
            lines, fails = crag_eval.grade(item, judge, crag_models)
            return [dict(ln, axis=4) for ln in lines], fails
        text = a.get("answer")
        if a.get("error") or text is None or not str(text).strip():
            return [{"qid": qid, "axis": axis, "judge": "rule", "label": "Incorrect", "score": 0.0,
                     "raw": "no answer", "cost_usd": 0.0}], []
        v = judge.vidore(qid, q.get("question", ""), str(q.get("answer", "")), str(text), vidore_model)
        if v.label is None:
            return [], [f"{qid} {vidore_model}: {v.error}"]
        return [{"qid": qid, "axis": axis, "judge": vidore_model, "label": v.label,
                 "score": J.VIDORE_SCORE[v.label], "raw": v.raw, "cost_usd": v.cost}], []

    lines: list[dict] = []
    fails: list[str] = []
    for ls, fs in judge.map(one, jobs):
        lines.extend(ls)
        fails.extend(fs)
    return lines, fails


def _write_judgments(path: Path, lines: Sequence[Mapping[str, Any]]) -> None:
    order = sorted(lines, key=lambda ln: (ln.get("axis", 0), ln.get("qid", ""), ln.get("judge", "")))
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in order:
            f.write(json.dumps({"qid": ln["qid"], "axis": ln["axis"], "judge": ln["judge"],
                                "label": ln["label"], "score": ln["score"], "raw": ln.get("raw", ""),
                                "cost_usd": ln.get("cost_usd")}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------ ranks

def compute_ranks(metrics: Mapping[str, Any]) -> dict:
    spec = (("retrieval", "ndcg10"), ("gold", "acc"), ("e2e", "acc"), ("crag", "truthfulness"))
    out: dict[str, dict] = {}
    for layer, field in spec:
        for pack, m in (metrics.get(layer) or {}).items():
            v = m.get(field) if isinstance(m, Mapping) else None
            pub = literature.rows(layer, pack)
            if v is None or not pub:
                continue
            out[f"{layer}.{pack}"] = rank_in(v, [r["value"] for r in pub])
    return out


def rank_in(value: float, published: Sequence[float]) -> dict:
    """Rank 1 = best. Compared at the one-decimal precision the paper tables use."""
    v = round(value, 1)
    higher = sum(1 for p in published if p > v)
    med = stats.median(published)
    side = "above" if v > round(med, 1) else "below" if v < round(med, 1) else "at"
    return {"rank": higher + 1, "of": len(published) + 1, "median": round(med, 2), "side": side,
            "value": round(value, 2)}


# ------------------------------------------------------------------ build

def build(run_dir: Path, judge: bool = True, mode: str = "official", judge_model: str | None = None) -> dict:
    run_dir = Path(run_dir)
    session = _read_json(run_dir / "run.json")
    mode = resolve_mode(mode, session)
    notes: list[str] = []

    answers_raw = cost_mod.read_jsonl(run_dir / "answers.jsonl")
    ans = _last_by_key(answers_raw)
    answers = list(ans.values())

    # packs
    pack_names = sorted({a.get("pack") for a in answers if a.get("pack")})
    questions: dict[str, dict[str, dict]] = {}
    qrels: dict[str, dict[str, dict[str, float]]] = {}
    try:
        api = _pack_api()
    except ImportError:
        api = None
        notes.append("crb.data.packs is not available, so gold answers and relevance labels could not be read.")
    for p in pack_names:
        if api is None:
            questions[p], qrels[p] = {}, {}
            continue
        try:
            questions[p] = {str(q["qid"]): q for q in api.load_questions(p)}
        except Exception as e:  # missing or unreadable pack
            questions[p] = {}
            notes.append(f"Pack {p} could not be read ({type(e).__name__}); its scores are skipped. "
                         f"Run `crb fetch --pack {p}`.")
        try:
            qrels[p] = retrieval.normalize_qrels(api.load_qrels(p)) if not is_crag(p) else {}
        except Exception:
            qrels[p] = {}

    # judge settings
    if judge_model:
        vidore_model, crag_models = judge_model, (judge_model,)
    elif mode == "smoke":
        vidore_model, crag_models = J.SMOKE_MODEL, (J.SMOKE_MODEL,)
    else:
        vidore_model, crag_models = J.VIDORE_MODEL, J.CRAG_MODELS
    uses_dev = any(str(p).startswith("dev_") for p in pack_names)
    comparable = mode != "smoke" and not judge_model and not uses_dev
    if uses_dev and mode != "smoke":
        notes.append("Practice packs (dev_*) were used: not comparable to published results.")

    # judgments
    jpath = run_dir / "judgments.jsonl"
    fails: list[str] = []
    judge_paid = None
    if judge:
        jobs = _judge_jobs(ans, questions)
        jd = J.Judge()
        try:
            jlines, fails = _run_judging(jobs, jd, vidore_model, crag_models)
        finally:
            jd.close()
        _write_judgments(jpath, jlines)
        judge_paid = jd.cost_usd
    else:
        jlines = cost_mod.read_jsonl(jpath)
    vlabel: dict[tuple[int, str], str] = {}
    crag_lines: dict[str, list[dict]] = {}
    for ln in jlines:
        try:
            axis, qid = int(ln["axis"]), str(ln["qid"])
        except (KeyError, TypeError, ValueError):
            continue
        if axis in (2, 3):
            vlabel[(axis, qid)] = ln.get("label")
        elif axis == 4:
            crag_lines.setdefault(qid, []).append(ln)
    judged_any = bool(jlines)
    if judge and fails:
        reasons = sorted({f.split(": ", 1)[-1] for f in fails})
        notes.append(f"{len(fails)} judge call(s) failed and those answers are left out of the scores: "
                     + "; ".join(reasons[:3]))
    if not judged_any:
        notes.append("Answers were not judged: layers 2 to 4 show \"not judged\". Run `crb score` with a judge key.")

    metrics: dict[str, Any] = {"retrieval": {}, "gold": {}, "e2e": {}, "crag": {}, "diagnosis": {}}

    # axis 1: retrieval
    by_pack_ret: dict[str, dict[str, Sequence[str] | None]] = {}
    top10_by_q: dict[str, list[str]] = {}
    for (axis, qid), a in ans.items():
        if axis != 1:
            continue
        retrieved = a.get("retrieved")
        if retrieved is None and a.get("error"):
            retrieved = []
        if retrieved is None:
            continue
        by_pack_ret.setdefault(a.get("pack"), {})[qid] = retrieved
        top10_by_q[qid] = list(retrieved)[:10]
    for pack, rows in sorted(by_pack_ret.items()):
        if is_crag(pack) or not qrels.get(pack):
            continue
        s = retrieval.score_run(rows, qrels[pack])
        if not s["n"]:
            continue
        total = len(questions.get(pack) or {}) or None
        metrics["retrieval"][pack] = {"ndcg10": _r2(s["ndcg10"]), "recall10": _r2(s["recall10"]),
                                      "n": s["n"], "pack_size": total}
        if mode != "smoke" and total and s["n"] < total:
            notes.append(f"Retrieval on {pack}: {s['n']} of {total} questions scored.")

    # axes 2, 3: judged accuracy
    for axis in (2, 3):
        key = AXIS_KEY[axis]
        for pack in pack_names:
            if is_crag(pack):
                continue
            answered = [qid for (ax, qid), a in ans.items() if ax == axis and a.get("pack") == pack]
            if not answered:
                continue
            total = len(questions.get(pack) or {}) or None
            judged = [qid for qid in answered if vlabel.get((axis, qid))]
            m: dict[str, Any] = {"acc": None, "ci95": None, "n": len(judged), "answered": len(answered),
                                 "pack_size": total}
            if judged:
                hits = [100.0 if vlabel[(axis, q)] == "Correct" else 0.0 for q in judged]
                m["acc"] = _r2(sum(hits) / len(hits))
                m["ci95"] = _r2(stats.bootstrap_ci(hits, total))
                m["partial"] = _r2(100.0 * sum(1 for q in judged if vlabel[(axis, q)] == "Partially Correct")
                                   / len(judged))
            if axis == 3 and judged:
                by_ct: dict[str, list[float]] = {}
                for q in judged:
                    cts = (questions.get(pack, {}).get(q, {}).get("content_type")) or ["Unknown"]
                    if isinstance(cts, str):
                        cts = [cts]
                    for ct in cts:
                        by_ct.setdefault(str(ct), []).append(1.0 if vlabel[(3, q)] == "Correct" else 0.0)
                m["by_content_type"] = {ct: {"acc": _r2(100 * sum(v) / len(v)), "n": len(v), "correct": int(sum(v))}
                                        for ct, v in sorted(by_ct.items())}
            metrics[key][pack] = m

    # axis 4: CRAG
    for pack in pack_names:
        if not is_crag(pack):
            continue
        answered = [qid for (ax, qid), a in ans.items() if ax == 4 and a.get("pack") == pack]
        if not answered:
            continue
        total = len(questions.get(pack) or {}) or None
        outs = {qid: crag_eval.outcome(crag_lines.get(qid, [])) for qid in answered}
        outs = {k: v for k, v in outs.items() if v}
        m = {"truthfulness": None, "ci95": None, "accurate": None, "hallucinated": None, "missing": None,
             "n": len(outs), "answered": len(answered), "pack_size": total}
        if outs:
            scores = [100.0 * o["score"] for o in outs.values()]
            n = len(outs)
            m.update({"truthfulness": _r2(sum(scores) / n),
                      "ci95": _r2(stats.bootstrap_ci(scores, total)),
                      "accurate": _r2(100 * sum(o["accurate"] for o in outs.values()) / n),
                      "hallucinated": _r2(100 * sum(o["incorrect"] for o in outs.values()) / n),
                      "missing": _r2(100 * sum(o["missing"] for o in outs.values()) / n)})
            by_qt: dict[str, list[dict]] = {}
            for qid, o in outs.items():
                by_qt.setdefault(str(questions.get(pack, {}).get(qid, {}).get("qtype") or "unknown"), []).append(o)
            m["by_qtype"] = {qt: {"truthfulness": _r2(100 * sum(o["score"] for o in v) / len(v)),
                                  "accurate": _r2(100 * sum(o["accurate"] for o in v) / len(v)),
                                  "hallucinated": _r2(100 * sum(o["incorrect"] for o in v) / len(v)),
                                  "missing": _r2(100 * sum(o["missing"] for o in v) / len(v)), "n": len(v)}
                             for qt, v in sorted(by_qt.items())}
        metrics["crag"][pack] = m

    # diagnosis (needs judged end-to-end answers)
    tags_by_q: dict[tuple[str, str], str] = {}
    for pack in pack_names:
        if is_crag(pack):
            continue
        tags = []
        for (ax, qid), a in sorted(ans.items()):
            if ax != 3 or a.get("pack") != pack or not vlabel.get((3, qid)):
                continue
            q = questions.get(pack, {}).get(qid, {})
            gold = _gold_pages(q, qrels.get(pack, {}).get(qid))
            top10 = top10_by_q.get(qid)  # axis-1 line of the same qid; axes 2-4 carry no retrieved list
            if top10 is None and a.get("retrieved"):
                top10 = list(a["retrieved"])[:10]
            g = vlabel.get((2, qid))
            t = diagnosis.tag(vlabel[(3, qid)] == "Correct", gold, top10, None if g is None else g == "Correct")
            tags.append(t)
            tags_by_q[(pack, qid)] = t
        if tags:
            metrics["diagnosis"][pack] = diagnosis.count(tags)

    # latency, health, cost
    hrows = latency.read_health(run_dir / "health.csv")
    metrics["latency"] = latency.summarize(answers, session, hrows)
    metrics["health"] = latency.summarize_health(hrows)
    n_ans = sum(1 for a in answers if a.get("axis") in (2, 3, 4))
    judge_usd = None
    if judged_any:
        known = [ln.get("cost_usd") for ln in jlines if isinstance(ln.get("cost_usd"), (int, float))]
        judge_usd = round(sum(known), 6) if known else None
    metrics["cost"] = cost_mod.summarize(cost_mod.read_jsonl(run_dir / "proxy.jsonl"), n_ans, judge_usd)
    if judge and judge_paid is not None:
        metrics["cost"]["judge_paid_now_usd"] = round(judge_paid, 6)
    if not metrics["cost"]["measured"]:
        notes.append("Cost not measured: no call went through the token proxy.")
    if not metrics["health"].get("available"):
        notes.append("No Pi stats: health.csv has no temperature or throttle samples.")
    if not metrics["latency"]["ingest"] and "ingest" not in session:
        notes.append("Ingest time was not recorded in run.json.")
    if mode == "smoke":
        notes.append("Practice run: cheap judge and small sample. Not comparable to published results.")
    if judge_model:
        notes.append(f"Scored with the judge {judge_model}, not the papers' judges: not comparable.")

    status = str(session.get("status") or "").lower()
    if status in ("interrupted", "incomplete", "running"):
        notes.append(f"The run is {status}: some questions were not answered, so scores cover only part of the sample.")
    for n in session.get("notes") or []:
        if isinstance(n, str) and n not in notes:
            notes.append(n)
    for p_, v in (session.get("ingest") or {}).items():
        if isinstance(v, Mapping) and v.get("ok") is False:
            notes.append(f"Ingest of {p_} failed" + (f": {v['error']}" if v.get("error") else "") + ".")

    ranks = compute_ranks(metrics)
    results = {
        "session": session, "mode": mode, "comparable": comparable, "status": status or None,
        "judge": {"ran": bool(judge), "has_judgments": judged_any, "vidore_model": vidore_model,
                  "crag_models": list(crag_models), "failed": len(fails)},
        "metrics": metrics, "published": literature.published(), "ranks": ranks,
        "worst": _worst(ans, questions, vlabel, crag_lines, tags_by_q),
        "notes": notes,
    }
    with open(run_dir / "results.json.tmp", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    os.replace(run_dir / "results.json.tmp", run_dir / "results.json")
    return results


def _worst(ans, questions, vlabel, crag_lines, tags_by_q, limit: int = 10, per_tag: int = 5) -> list[dict]:
    """Wrong answers, most common failure type first, packs interleaved."""
    groups: dict[str, list[dict]] = {}
    for (pack, qid), t in sorted(tags_by_q.items()):
        if t == "correct":
            continue
        a = ans.get((3, qid), {})
        groups.setdefault(t, []).append(_worst_row(qid, pack, questions, a, t))
    for (axis, qid), a in sorted(ans.items()):
        if axis != 4:
            continue
        o = crag_eval.outcome(crag_lines.get(qid, []))
        if o and o["score"] < 0:
            groups.setdefault("hallucinated", []).append(_worst_row(qid, a.get("pack"), questions, a, "hallucinated"))
    out: list[dict] = []
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    picked: dict[str, list[dict]] = {}
    for t, rows in ordered:
        by_pack: dict[str, list[dict]] = {}
        for r in rows:
            by_pack.setdefault(r["pack"], []).append(r)
        mixed = [r for grp in itertools.zip_longest(*by_pack.values()) for r in grp if r]
        picked[t] = mixed
        out.extend(mixed[:per_tag])
    if len(out) < limit:
        for t, _rows in ordered:
            out.extend(picked[t][per_tag:])
    return out[:limit]


def _worst_row(qid, pack, questions, a, tag) -> dict:
    q = questions.get(pack, {}).get(qid, {})
    text = a.get("answer")
    if a.get("error"):
        text = f"(no answer: {_short(a['error'], 80)})"
    elif text is None or not str(text).strip():
        text = "(no answer)"
    return {"qid": qid, "pack": pack, "question": _short(q.get("question")), "gold": _short(q.get("answer")),
            "answer": _short(text), "tag": tag}

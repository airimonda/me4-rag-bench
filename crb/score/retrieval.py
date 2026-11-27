"""nDCG@10 and Recall@10 from graded relevance (linear gain, as pytrec_eval / MTEB)."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def normalize_qrels(obj: Any) -> dict[str, dict[str, float]]:
    """Accept {qid: {page: score}} or an iterable of {"qid","page_id","score"} rows."""
    out: dict[str, dict[str, float]] = {}
    if not obj:
        return out
    if isinstance(obj, Mapping):
        for qid, rel in obj.items():
            if isinstance(rel, Mapping):
                out[str(qid)] = {str(p): float(s) for p, s in rel.items()}
            else:
                out[str(qid)] = {str(p): float(s) for p, s in rel}
        return out
    for row in obj:
        out.setdefault(str(row["qid"]), {})[str(row["page_id"])] = float(row.get("score", 1))
    return out


def _dedupe(retrieved: Sequence[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in retrieved or []:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str] | None, qrel: Mapping[str, float], k: int = 10) -> float | None:
    """None when the question has no relevant page (undefined)."""
    pos = {p: s for p, s in qrel.items() if s > 0}
    if not pos:
        return None
    top = _dedupe(retrieved)[:k]
    got = dcg(pos.get(p, 0.0) for p in top)
    ideal = dcg(sorted(pos.values(), reverse=True)[:k])
    return got / ideal if ideal > 0 else None


def recall_at_k(retrieved: Sequence[str] | None, qrel: Mapping[str, float], k: int = 10) -> float | None:
    pos = {p for p, s in qrel.items() if s > 0}
    if not pos:
        return None
    top = set(_dedupe(retrieved)[:k])
    return len(pos & top) / len(pos)


def score_run(rows: Mapping[str, Sequence[str] | None], qrels: Mapping[str, Mapping[str, float]],
              k: int = 10) -> dict:
    """rows: qid -> retrieved list. Returns {"ndcg10","recall10","n"} x100, averaged over scorable questions."""
    nd: list[float] = []
    rc: list[float] = []
    for qid, retrieved in rows.items():
        q = qrels.get(qid)
        if not q:
            continue
        a = ndcg_at_k(retrieved, q, k)
        b = recall_at_k(retrieved, q, k)
        if a is None or b is None:
            continue
        nd.append(a)
        rc.append(b)
    n = len(nd)
    return {"ndcg10": 100 * sum(nd) / n if n else None,
            "recall10": 100 * sum(rc) / n if n else None, "n": n}

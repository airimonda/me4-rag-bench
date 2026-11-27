"""CRAG auto-evaluation: a rule step first, then two LLM judges whose scores are averaged.

accurate = +1, missing ("I don't know", empty, failed) = 0, incorrect (hallucinated) = -1.
Truthfulness = mean per-question score x 100.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Sequence

from . import judge as J

LABEL_SCORE = {"accurate": 1.0, "missing": 0.0, "incorrect": -1.0}
INVALID = "invalid question"


def normalize(text: Any) -> str:
    t = unicodedata.normalize("NFKC", str(text if text is not None else "")).lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^\w\s']", " ", t)
    t = t.replace("'", "")
    return re.sub(r"\s+", " ", t).strip()


def rule_step(answer: str | None, gold: Any, alt: Sequence[Any] | None, qtype: str | None,
              error: str | None = None) -> tuple[str, str] | None:
    """Return (label, reason) when the rule step decides, else None (needs the LLM judges)."""
    if error or answer is None or not str(answer).strip():
        return "missing", "no answer"
    raw = str(answer).replace("’", "'").lower()
    na = normalize(answer)
    golds = [normalize(g) for g in [gold, *(alt or [])] if g is not None and str(g).strip()]
    if na and na in golds:
        return "accurate", "exact match"
    if "i don't know" in raw or "i dont know" in raw:
        return "missing", "says i don't know"
    if INVALID in na:
        gold_is_invalid = INVALID in golds
        if qtype == "false_premise" or gold_is_invalid:
            return "accurate", "invalid question on a false-premise question"
        return "incorrect", "invalid question on a valid question"
    return None


def _gold_text(gold: Any, alt: Sequence[Any] | None) -> str:
    g = str(gold if gold is not None else "")
    extra = [str(a) for a in (alt or []) if a is not None and str(a).strip()]
    return g + (" (also accepted: " + "; ".join(extra) + ")" if extra else "")


def grade(item: Mapping[str, Any], judge: "J.Judge", models: Sequence[str] = J.CRAG_MODELS) -> tuple[list[dict], list[str]]:
    """Judgment lines for one CRAG answer item (no axis field) plus a list of failure messages.

    item: qid, question, gold, alt_answers, qtype, answer, error.
    """
    qid = item["qid"]
    r = rule_step(item.get("answer"), item.get("gold"), item.get("alt_answers"), item.get("qtype"),
                  item.get("error"))
    if r:
        label, reason = r
        return [{"qid": qid, "judge": "rule", "label": label, "score": LABEL_SCORE[label],
                 "raw": reason, "cost_usd": 0.0}], []
    lines: list[dict] = []
    fails: list[str] = []
    gold = _gold_text(item.get("gold"), item.get("alt_answers"))
    for m in models:
        v = judge.crag(qid, item.get("question", ""), gold, str(item.get("answer")), m)
        if v.label is None:
            fails.append(f"{qid} {m}: {v.error}")
            continue
        lines.append({"qid": qid, "judge": m, "label": v.label, "score": LABEL_SCORE[v.label],
                      "raw": v.raw, "cost_usd": v.cost})
    return lines, fails


def outcome(lines: Sequence[Mapping[str, Any]]) -> dict | None:
    """Per-question outcome from its judgment lines. Shares sum to 1; score is their accurate - incorrect."""
    if not lines:
        return None
    rule = [ln for ln in lines if ln.get("judge") == "rule"]
    use = rule[:1] if rule else [ln for ln in lines if ln.get("label") in LABEL_SCORE]
    if not use:
        return None
    n = len(use)
    acc = sum(1 for ln in use if ln["label"] == "accurate") / n
    inc = sum(1 for ln in use if ln["label"] == "incorrect") / n
    mis = sum(1 for ln in use if ln["label"] == "missing") / n
    return {"score": acc - inc, "accurate": acc, "incorrect": inc, "missing": mis}

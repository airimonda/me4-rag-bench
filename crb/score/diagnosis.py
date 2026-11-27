"""Diagnosis tags for ViDoRe end-to-end answers: which layer broke first."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

TAGS = ("correct", "retrieval_miss", "generation_miss", "integration_miss")
TAG_LABELS = {"correct": "Correct", "retrieval_miss": "Retrieval miss",
              "generation_miss": "Generation miss", "integration_miss": "Integration miss",
              "hallucinated": "Hallucinated"}


def tag(e2e_correct: bool, gold_pages: Iterable[str], top10: Sequence[str] | None,
        gold_correct: bool | None) -> str:
    """correct / retrieval miss / generation miss / integration miss.

    - correct: the end-to-end answer was judged Correct.
    - retrieval miss: no gold page in the axis-1 top 10.
    - generation miss: the gold-pages answer (axis 2) was not Correct either.
    - integration miss: otherwise (right pages found, model can read them, whole pipeline still failed).
    `gold_correct` is None when axis 2 is unavailable; the generation test is then skipped.
    `top10` is None when no axis-1 line exists for the question; the retrieval test is then skipped.
    """
    if e2e_correct:
        return "correct"
    gold = set(gold_pages)
    if gold and top10 is not None and not (gold & set(top10[:10])):
        return "retrieval_miss"
    if gold_correct is False:
        return "generation_miss"
    return "integration_miss"


def count(tags: Iterable[str]) -> dict:
    out = {t: 0 for t in TAGS}
    for t in tags:
        out[t] = out.get(t, 0) + 1
    return out

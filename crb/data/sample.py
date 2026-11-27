"""The class-v1 sample: which questions every student's official run uses.

Rule (SPEC.md): stratified by `qtype`, proportional allocation, largest-remainder rounding,
`random.Random(7)`, sorted output. ViDoRe: 100 answer questions per pack (retrieval = all English
questions). CRAG: 400 answer questions.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from crb.data import packs

SEED = 7
SAMPLE_NAME = "class-v1"
ANSWER_SIZES = {"vidore_v3_hr": 100, "vidore_v3_computer_science": 100, "crag_task1": 400}


def stratified_sample(items: list[tuple[str, str]], n: int, seed: int = SEED) -> list[str]:
    """Pick `n` ids from `items` = [(qid, stratum)], proportional per stratum.

    Quotas: floor(n * size / total) per stratum, then the leftover seats go to the strata with the
    biggest fractional remainders (ties: stratum name order). Inside a stratum the ids are drawn
    with `random.Random(seed)`. Strata are visited in sorted order and ids sorted before drawing, so
    the result depends only on (set of items, n, seed), never on input order. Output is sorted.
    """
    total = len(items)
    if n >= total:
        return sorted(q for q, _ in items)
    groups: dict[str, list[str]] = {}
    for qid, stratum in items:
        groups.setdefault(stratum, []).append(qid)
    strata = sorted(groups)
    for s in strata:
        groups[s].sort()
    # Integer arithmetic only, so there are no float ties: exact = n * size / total.
    quota = {s: (n * len(groups[s])) // total for s in strata}
    rem = {s: (n * len(groups[s])) % total for s in strata}
    left = n - sum(quota.values())
    for s in sorted(strata, key=lambda s: (-rem[s], s))[:left]:
        quota[s] += 1
    rng = random.Random(seed)
    chosen: list[str] = []
    for s in strata:
        chosen.extend(rng.sample(groups[s], quota[s]))
    return sorted(chosen)


def pack_sample(pack: str, questions: list[dict]) -> dict:
    """The sample entry for one pack, from its question dicts."""
    items = [(q["qid"], q.get("qtype") or "unknown") for q in questions]
    n = ANSWER_SIZES.get(pack, 100)
    entry: dict = {}
    if pack != "crag_task1":
        entry["retrieval"] = sorted(q for q, _ in items)
    entry["answer"] = stratified_sample(items, n)
    return entry


def build(data: Path | None = None, pack_ids: tuple[str, ...] = packs.PACK_IDS) -> Path:
    """(Re)write samples/class-v1.json for every pack whose questions.jsonl is on disk.
    Entries of packs that are not downloaded yet are kept from an earlier file if there was one."""
    data = Path(data) if data else packs.data_dir()
    out = data / "samples" / f"{SAMPLE_NAME}.json"
    current: dict = {"seed": SEED, "packs": {}}
    if out.exists():
        try:
            current = json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            pass
    current["seed"] = SEED
    entries = current.setdefault("packs", {})
    for pack in pack_ids:
        qpath = data / "packs" / pack / "questions.jsonl"
        if not qpath.is_file():
            continue
        with open(qpath, encoding="utf-8") as f:
            questions = [json.loads(line) for line in f if line.strip()]
        entries[pack] = pack_sample(pack, questions)
    current["packs"] = {k: entries[k] for k in sorted(entries)}
    write_json(out, current)
    return out


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

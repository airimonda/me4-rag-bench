"""Build the two ViDoRe V3 packs (vidore_v3_hr, vidore_v3_computer_science) from Hugging Face.

Download the original parquet files (queries, qrels, corpus) into `<data>/_cache/<pack>/`, then write
page PNG + markdown files, `questions.jsonl` (English only), `qrels.jsonl` and `manifest.json`.
The corpus is read in small row batches so peak memory stays well under 1 GB on a Pi 4.
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

from crb.data import fetch, packs

PACKS = {
    "vidore_v3_hr": {"repo": "vidore/vidore_v3_hr", "short": "hr"},
    "vidore_v3_computer_science": {"repo": "vidore/vidore_v3_computer_science", "short": "cs"},
}
LICENSE = "CC-BY-4.0"
VERSION = "class-v1"
BATCH_ROWS = 8


def _as_list(v) -> list:
    """Parquet list columns come back as lists; tolerate the string form "['a', 'b']" too."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return list(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                pass
        return [s] if s else []
    return [v]


def _write(path: Path, data: bytes, files: dict[str, str], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    files[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()


def english_questions(queries_pq: Path, pack: str) -> list[dict]:
    """Questions of one pack from the queries parquet: English rows only, schema of SPEC.md
    (gold_pages are filled in later from the qrels)."""
    import pyarrow.parquet as pq
    short = PACKS[pack]["short"]
    cols = ["query_id", "query", "language", "query_types", "content_type", "answer"]
    out = []
    for row in pq.read_table(queries_pq, columns=cols).to_pylist():
        if (row["language"] or "").strip().lower() != "english":
            continue
        types = _as_list(row["query_types"])
        out.append({
            "qid": f"{short}-{row['query_id']}",
            "pack": pack,
            "question": row["query"],
            "answer": row["answer"] or "",
            "alt_answers": [],
            "qtype": str(types[0]) if types else "unknown",
            "content_type": [str(c) for c in _as_list(row["content_type"])],
            "query_time": None,
            "gold_pages": [],
            "_query_id": row["query_id"],
        })
    out.sort(key=lambda q: int(q["_query_id"]))
    return out


def build_from_files(pack: str, data: Path, queries_pq: Path, qrels_pq: Path,
                     corpus_pqs: list[Path], say: Callable[[str], None] = print) -> dict:
    """Write the whole pack from local parquet files. Returns the manifest dict."""
    import pyarrow.parquet as pq
    short = PACKS[pack]["short"]
    root = Path(data) / "packs" / pack
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)  # a half-built pack must not look complete
    files: dict[str, str] = {}

    questions = english_questions(queries_pq, pack)
    say(f"{pack}: {len(questions)} English questions found.")
    by_qid = {q["_query_id"]: q for q in questions}

    # --- pages
    id_to_page: dict[int, str] = {}
    seen: set[str] = set()
    n_pages = 0
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in corpus_pqs)
    prog = fetch.Progress(say)
    pages_dir = root / "pages"
    for cp in corpus_pqs:
        pf = pq.ParquetFile(cp, memory_map=False)
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            for row in batch.to_pylist():
                pid = packs.page_id_for(row["doc_id"], row["page_number_in_doc"])
                if pid in seen:
                    raise ValueError(f"{pack}: two corpus rows map to the same page id {pid}")
                seen.add(pid)
                img = (row.get("image") or {}).get("bytes")
                if not img:
                    raise ValueError(f"{pack}: corpus row {row['corpus_id']} has no image bytes")
                _write(pages_dir / f"{pid}.png", img, files, root)
                _write(pages_dir / f"{pid}.md", (row.get("markdown") or "").encode("utf-8"), files, root)
                id_to_page[int(row["corpus_id"])] = pid
                n_pages += 1
            prog.tick(f"{pack}: wrote {n_pages} of {total} pages ...")
    say(f"{pack}: wrote {n_pages} pages.")

    # --- qrels (all languages are in the file; keep rows of English questions)
    qrel_rows: list[dict] = []
    gold: dict[int, list[tuple[int, int, str]]] = {}
    missing_pages = 0
    for i, r in enumerate(pq.read_table(qrels_pq, columns=["query_id", "corpus_id", "score"]).to_pylist()):
        q = by_qid.get(r["query_id"])
        if q is None:
            continue
        pid = id_to_page.get(int(r["corpus_id"]))
        if pid is None:
            missing_pages += 1
            continue
        qrel_rows.append({"qid": q["qid"], "page_id": pid, "score": int(r["score"])})
        if r["score"] > 0:
            gold.setdefault(r["query_id"], []).append((-int(r["score"]), i, pid))
    for qid_num, lst in gold.items():
        by_qid[qid_num]["gold_pages"] = [pid for _, _, pid in sorted(lst)]
    if missing_pages:
        say(f"{pack}: warning, {missing_pages} qrels rows point at pages not in the corpus (skipped).")
    no_gold = sum(1 for q in questions if not q["gold_pages"])
    if no_gold:
        say(f"{pack}: warning, {no_gold} questions have no relevant page in the qrels.")

    # --- questions + qrels files
    qlines = []
    for q in questions:
        q = {k: v for k, v in q.items() if not k.startswith("_")}
        qlines.append(json.dumps(q, ensure_ascii=False))
    _write(root / "questions.jsonl", ("\n".join(qlines) + "\n").encode("utf-8"), files, root)
    rlines = [json.dumps(r) for r in qrel_rows]
    _write(root / "qrels.jsonl", ("\n".join(rlines) + "\n").encode("utf-8"), files, root)

    manifest = {
        "pack": pack,
        "source": f"https://huggingface.co/datasets/{PACKS[pack]['repo']}",
        "license": LICENSE,
        "version": VERSION,
        "files": dict(sorted(files.items())),
        "counts": {"pages": n_pages, "questions": len(questions), "qrels": len(qrel_rows)},
    }
    write_manifest(root, manifest)
    say(f"{pack}: ready ({len(questions)} questions, {n_pages} pages, {len(qrel_rows)} relevance judgments).")
    return manifest


def write_manifest(root: Path, manifest: dict) -> None:
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    tmp.replace(root / "manifest.json")


def build(pack: str, data: Path, say: Callable[[str], None] = print, keep_cache: bool = False) -> dict:
    """Download the parquet files of `pack` (resumable) and build the pack."""
    if pack not in PACKS:
        raise ValueError(f"unknown ViDoRe pack {pack}")
    repo = PACKS[pack]["repo"]
    cache = Path(data) / "_cache" / pack
    say(f"{pack}: looking up files on Hugging Face ...")
    got: dict[str, list[Path]] = {}
    for config in ("queries", "qrels", "corpus"):
        paths = []
        for f in fetch.hf_files(repo, config):
            dest = cache / f["path"]
            say(f"{pack}: downloading {f['path']} ({fetch._mb(f['size'])}) ...")
            fetch.download(f["url"], dest, expected_size=f["size"], say=say, label=f"{pack}/{f['path']}",
                           headers=fetch._auth())
            paths.append(dest)
        got[config] = paths
    say(f"{pack}: download complete, building pages (this takes a few minutes) ...")
    manifest = build_from_files(pack, Path(data), got["queries"][0], got["qrels"][0], got["corpus"], say=say)
    if not keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
        try:
            (Path(data) / "_cache").rmdir()
        except OSError:
            pass
    return manifest

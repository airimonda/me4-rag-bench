"""Read API for the benchmark data packs (everything other modules need to touch data on disk).

Layout and schemas are fixed in SPEC.md. Nothing here needs the network or pyarrow.

    data_dir()                  -> Path          $CRB_DATA or ~/crb-data
    pack_dir(pack)              -> Path
    load_questions(pack)        -> list[dict]    questions.jsonl, file order
    load_qrels(pack)            -> {qid: {page_id: score}}     (ViDoRe packs only; {} otherwise)
    load_sample(name)           -> dict          samples/<name>.json  ({"seed": 7, "packs": {...}})
    page_paths(pack, page_id)   -> PagePaths(png, md)          absolute Paths (ViDoRe packs)
    crag_pages(pack, qid)       -> list[dict]    the pages of one CRAG question, ready for /answer
    verify(pack)                -> list[str]     problems found (empty list = pack is fine)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import NamedTuple

PACK_IDS = ("vidore_v3_hr", "vidore_v3_computer_science", "crag_task1")
DEV_PACK_IDS = ("dev_vidore", "dev_crag")

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


class PagePaths(NamedTuple):
    png: Path
    md: Path


class DataError(Exception):
    """A pack, file or id that should exist does not. The message says what to do about it."""


def data_dir() -> Path:
    """Root folder of all downloaded data: $CRB_DATA, default ~/crb-data."""
    env = os.environ.get("CRB_DATA")
    return Path(env).expanduser() if env else Path.home() / "crb-data"


def pack_dir(pack: str) -> Path:
    return data_dir() / "packs" / pack


def safe_name(text: str) -> str:
    """Make `text` filesystem-safe: only [A-Za-z0-9._-] survive, everything else becomes '_'."""
    return _SAFE.sub("_", str(text))


def page_id_for(doc_id: str, page_number: int) -> str:
    """ViDoRe page id: <doc_id>__p<page_number_in_doc>, filesystem-safe."""
    return safe_name(f"{doc_id}__p{page_number}")


def _require(path: Path, what: str, pack: str) -> Path:
    if not path.exists():
        raise DataError(
            f"{what} for pack '{pack}' not found at {path}. Run `crb fetch` "
            f"(or `crb fetch --pack {pack}`) first."
        )
    return path


def load_questions(pack: str) -> list[dict]:
    path = _require(pack_dir(pack) / "questions.jsonl", "questions.jsonl", pack)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_qrels(pack: str) -> dict[str, dict[str, int]]:
    """{qid: {page_id: graded relevance}}. Packs without qrels (CRAG) give {}."""
    path = pack_dir(pack) / "qrels.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.setdefault(r["qid"], {})[r["page_id"]] = r["score"]
    return out


def load_sample(name: str) -> dict:
    path = data_dir() / "samples" / f"{name}.json"
    if not path.exists():
        raise DataError(f"Sample '{name}' not found at {path}. Run `crb fetch` first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def page_paths(pack: str, page_id: str) -> PagePaths:
    """Absolute paths of the page image and its markdown (the files may not exist for odd ids)."""
    base = pack_dir(pack).resolve() / "pages"
    return PagePaths(base / f"{page_id}.png", base / f"{page_id}.md")


def crag_pages(pack: str, qid: str) -> list[dict]:
    """The pages of one CRAG question (k = 0..), shaped like the `pages` field of /answer mode web:
    {"path", "page_name", "page_url", "page_snippet", "page_last_modified"}."""
    d = pack_dir(pack).resolve() / "web" / qid
    if not d.is_dir():
        raise DataError(f"No web pages for question '{qid}' in pack '{pack}' (looked in {d}).")
    ks = sorted(int(p.stem) for p in d.glob("*.html") if p.stem.isdigit())
    pages = []
    for k in ks:
        meta: dict = {}
        mp = d / f"{k}.json"
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        pages.append({
            "path": str(d / f"{k}.html"),
            "page_name": meta.get("page_name", ""),
            "page_url": meta.get("page_url", ""),
            "page_snippet": meta.get("page_snippet", ""),
            "page_last_modified": meta.get("page_last_modified", ""),
        })
    return pages


# ---------------------------------------------------------------- verification

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(pack: str, quick: bool = False) -> list[str]:
    """Check a pack on disk; returns plain-English problems (empty = fine).

    Checks: manifest present, every listed file exists and (unless quick=True) has the right sha256,
    question/qrels consistency (gold pages exist, qrels point at known pages and questions), and
    that the web pages of every CRAG question exist. `quick=True` skips hashing (existence only).
    """
    problems: list[str] = []
    root = pack_dir(pack)
    mpath = root / "manifest.json"
    if not mpath.exists():
        return [f"Pack '{pack}' is not downloaded (no manifest at {mpath}). Run `crb fetch --pack {pack}`."]
    try:
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        return [f"Pack '{pack}': manifest.json is unreadable ({e}). Run `crb fetch --pack {pack}` again."]
    files = manifest.get("files") or {}
    if not files:
        problems.append(f"Pack '{pack}': manifest lists no files.")
    for rel, want in sorted(files.items()):
        p = root / rel
        if not p.is_file():
            problems.append(f"Pack '{pack}': file missing: {rel}")
        elif not quick and sha256_file(p) != want:
            problems.append(f"Pack '{pack}': file is damaged (checksum differs): {rel}")
    # Consistency checks only make sense when the question file is there.
    qpath = root / "questions.jsonl"
    if not qpath.is_file():
        problems.append(f"Pack '{pack}': questions.jsonl missing.")
        return problems
    try:
        questions = load_questions(pack)
    except ValueError as e:
        problems.append(f"Pack '{pack}': questions.jsonl is not valid JSON lines ({e}).")
        return problems
    qids = {q["qid"] for q in questions}
    if len(qids) != len(questions):
        problems.append(f"Pack '{pack}': duplicate question ids in questions.jsonl.")
    counts = manifest.get("counts") or {}
    if "questions" in counts and counts["questions"] != len(questions):
        problems.append(
            f"Pack '{pack}': manifest says {counts['questions']} questions, file has {len(questions)}.")
    is_crag = (root / "web").is_dir() or "crag" in pack
    if is_crag:
        for q in questions:
            d = root / "web" / q["qid"]
            if not (d / "0.html").is_file():
                problems.append(f"Pack '{pack}': no web pages for question {q['qid']}.")
                if len(problems) > 50:
                    break
    else:
        pages_dir = root / "pages"
        page_ids = {p.stem for p in pages_dir.glob("*.png")} if pages_dir.is_dir() else set()
        if "pages" in counts and counts["pages"] != len(page_ids):
            problems.append(f"Pack '{pack}': manifest says {counts['pages']} pages, found {len(page_ids)} images.")
        for pid in sorted(page_ids):
            if not (pages_dir / f"{pid}.md").is_file():
                problems.append(f"Pack '{pack}': page {pid} has no markdown file.")
                if len(problems) > 50:
                    break
        bad = 0
        for q in questions:
            for g in q.get("gold_pages") or []:
                if g not in page_ids:
                    bad += 1
                    if bad <= 5:
                        problems.append(f"Pack '{pack}': question {q['qid']} points at unknown page {g}.")
        qrels = load_qrels(pack)
        for qid, rel in qrels.items():
            if qid not in qids:
                problems.append(f"Pack '{pack}': qrels mention unknown question {qid}.")
                break
            unknown = [p for p in rel if p not in page_ids]
            if unknown:
                problems.append(f"Pack '{pack}': qrels for {qid} mention unknown page {unknown[0]}.")
                break
    return problems

"""Deterministic, offline generator for the tiny dev packs used by `crb fetch --dev` and the tests.

    packs/dev_vidore/   12 pages (tiny valid PNGs + markdown), 8 questions with gold pages, qrels
    packs/dev_crag/     6 questions x 5 small HTML pages
    samples/dev.json    which questions the dev runs use

Same bytes every time (no timestamps, no randomness), so checksums are stable. No PIL: the PNGs are
written by hand with zlib. Question ids: `dv-1..8` (dev_vidore) and `dc-1..6` (dev_crag).
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

from crb.data import packs, sample

LICENSE = "synthetic test data, public domain (CC0-1.0)"
VERSION = "dev-1"

DOCS = {
    "dev_doc_a": "Starfall Harbor Annual Report",
    "dev_doc_b": "Guide to Growing Lemons",
}
PAGES_PER_DOC = 6


def make_png(width: int, height: int, seed: int) -> bytes:
    """A valid RGB PNG with a seed-dependent colour gradient (deterministic)."""
    rows = []
    for y in range(height):
        row = bytearray([0])  # filter type 0
        for x in range(width):
            row += bytes(((seed * 37 + x * 8) % 256, (seed * 91 + y * 8) % 256, (seed * 53 + x * y) % 256))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _page_text(doc: str, n: int) -> str:
    facts = {
        ("dev_doc_a", 0): "Starfall Harbor Annual Report. Contents: harbor traffic, finances, staff.",
        ("dev_doc_a", 1): "Harbor traffic: in 2023 Starfall Harbor handled 4,200 ships, up from 3,800 in 2022.",
        ("dev_doc_a", 2): "Finances: the harbor earned 12 million credits in 2023 and spent 9 million.",
        ("dev_doc_a", 3): "Staff: the harbor employs 310 people. The harbor master is Mira Okonkwo.",
        ("dev_doc_a", 4): "Table of docks: Dock A holds 40 ships, Dock B holds 25 ships, Dock C holds 60 ships.",
        ("dev_doc_a", 5): "Outlook: a new Dock D with room for 80 ships opens in 2025.",
        ("dev_doc_b", 0): "Guide to Growing Lemons. Contents: soil, water, pests.",
        ("dev_doc_b", 1): "Soil: lemon trees need well-drained soil with a pH between 6.0 and 7.0.",
        ("dev_doc_b", 2): "Water: water young lemon trees twice a week, mature trees once a week.",
        ("dev_doc_b", 3): "Pests: aphids and scale insects are the most common lemon pests.",
        ("dev_doc_b", 4): "Harvest: lemons are ready when they are bright yellow and slightly soft.",
        ("dev_doc_b", 5): "Frost: cover trees when the temperature falls below 2 degrees Celsius.",
    }
    return f"# {DOCS[doc]} - page {n}\n\n{facts[(doc, n)]}\n"


# (question, answer, qtype, content_type, [(doc, page, score)])
VIDORE_QUESTIONS = [
    ("How many ships did Starfall Harbor handle in 2023?", "4,200 ships.", "extractive", ["Text"],
     [("dev_doc_a", 1, 2)]),
    ("Who is the harbor master of Starfall Harbor?", "Mira Okonkwo.", "extractive", ["Text"],
     [("dev_doc_a", 3, 2)]),
    ("What was the harbor's profit in 2023, in million credits?", "3 million credits (12 earned minus 9 spent).",
     "numerical", ["Text"], [("dev_doc_a", 2, 2)]),
    ("Which dock holds the most ships?", "Dock C, with 60 ships.", "extractive", ["Table", "Text"],
     [("dev_doc_a", 4, 2)]),
    ("Will the harbor get bigger, and when?", "Yes, Dock D with room for 80 ships opens in 2025.",
     "open-ended", ["Text"], [("dev_doc_a", 5, 2), ("dev_doc_a", 4, 1)]),
    ("What soil pH do lemon trees need?", "A pH between 6.0 and 7.0.", "extractive", ["Text"],
     [("dev_doc_b", 1, 2)]),
    ("Do young and mature lemon trees need the same watering?", "No: young trees twice a week, mature once a week.",
     "boolean", ["Text"], [("dev_doc_b", 2, 2)]),
    ("Which pests and which weather threaten lemons?",
     "Aphids and scale insects; frost below 2 degrees Celsius.", "multi-hop", ["Text"],
     [("dev_doc_b", 3, 2), ("dev_doc_b", 5, 2)]),
]

CRAG_DOMAINS = ["finance", "music", "movie", "sports", "open", "finance"]
CRAG_QTYPES = ["simple", "comparison", "simple", "aggregation", "false_premise", "set"]
CRAG_QUESTIONS = [
    ("What does the dev company Acme Widgets sell?", "widgets"),
    ("Which dev band released the album Blue Echo?", "The Quiet Foxes"),
    ("In what year was the dev film Harbor Lights released?", "1999"),
    ("How many points did the dev team Falcons score in their last game?", "88"),
    ("When did the dev city of Zorbia land on the moon?", "invalid question"),
    ("Name the three founders of the dev firm Tri Corp.", "Ana, Bo, and Cy"),
]


def _web_page(qi: int, k: int, question: str, answer: str) -> tuple[str, dict]:
    relevant = k < 2  # first two pages carry the answer, the rest are distractors
    name = f"Dev page {qi}-{k}"
    body = (f"<html><head><title>{name}</title></head><body><h1>{name}</h1>"
            f"<p>Background text number {k} for question {qi}.</p>")
    if relevant and answer != "invalid question":
        body += f"<p>Fact: {answer}.</p>"
    body += "<table><tr><td>row</td><td>%d</td></tr></table></body></html>" % (qi * 10 + k)
    meta = {
        "page_name": name,
        "page_url": f"https://dev.example.org/q{qi}/p{k}",
        "page_snippet": f"Background text number {k} for question {qi}.",
        "page_last_modified": f"Mon, 0{k + 1} Jan 2024 00:00:00 GMT",
    }
    return body + "\n", meta


class _Writer:
    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, str] = {}

    def put(self, rel: str, data: bytes) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self.files[rel] = hashlib.sha256(data).hexdigest()

    def manifest(self, pack: str, source: str, counts: dict) -> None:
        m = {"pack": pack, "source": source, "license": LICENSE, "version": VERSION,
             "files": dict(sorted(self.files.items())), "counts": counts}
        (self.root / "manifest.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


def generate_vidore(data: Path) -> dict:
    root = Path(data) / "packs" / "dev_vidore"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)
    w = _Writer(root)
    n_pages = 0
    for di, doc in enumerate(sorted(DOCS)):
        for n in range(PAGES_PER_DOC):
            pid = packs.page_id_for(doc, n)
            w.put(f"pages/{pid}.png", make_png(24, 32, di * 10 + n))
            w.put(f"pages/{pid}.md", _page_text(doc, n).encode("utf-8"))
            n_pages += 1
    questions, qrels = [], []
    for i, (q, a, qtype, ctype, gold) in enumerate(VIDORE_QUESTIONS, 1):
        qid = f"dv-{i}"
        gp = [packs.page_id_for(d, p) for d, p, _ in sorted(gold, key=lambda g: -g[2])]
        questions.append({"qid": qid, "pack": "dev_vidore", "question": q, "answer": a, "alt_answers": [],
                          "qtype": qtype, "content_type": ctype, "query_time": None, "gold_pages": gp})
        for d, p, s in gold:
            qrels.append({"qid": qid, "page_id": packs.page_id_for(d, p), "score": s})
    w.put("questions.jsonl", _jsonl(questions))
    w.put("qrels.jsonl", _jsonl(qrels))
    w.manifest("dev_vidore", "generated offline by crb.data.devpack",
               {"pages": n_pages, "questions": len(questions), "qrels": len(qrels)})
    return {"retrieval": [q["qid"] for q in questions], "answer": [q["qid"] for q in questions]}


def generate_crag(data: Path) -> dict:
    root = Path(data) / "packs" / "dev_crag"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)
    w = _Writer(root)
    questions = []
    for i, (q, a) in enumerate(CRAG_QUESTIONS, 1):
        qid = f"dc-{i}"
        for k in range(5):
            html_text, meta = _web_page(i, k, q, a)
            w.put(f"web/{qid}/{k}.html", html_text.encode("utf-8"))
            w.put(f"web/{qid}/{k}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        questions.append({"qid": qid, "pack": "dev_crag", "question": q, "answer": a, "alt_answers": [],
                          "qtype": CRAG_QTYPES[i - 1], "content_type": [],
                          "query_time": f"03/0{i}/2024, 12:00:00 PT", "gold_pages": []})
    w.put("questions.jsonl", _jsonl(questions))
    w.manifest("dev_crag", "generated offline by crb.data.devpack",
               {"questions": len(questions), "pages": len(questions) * 5})
    return {"answer": [q["qid"] for q in questions]}


def generate(data: Path | None = None) -> Path:
    """Write both dev packs and samples/dev.json under `data` (default: packs.data_dir()).
    Returns the data directory. Safe to call repeatedly: identical output."""
    data = Path(data) if data else packs.data_dir()
    entries = {"dev_crag": generate_crag(data), "dev_vidore": generate_vidore(data)}
    sample.write_json(data / "samples" / "dev.json", {"seed": sample.SEED, "packs": entries})
    return data

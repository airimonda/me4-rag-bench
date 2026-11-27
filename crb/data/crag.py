"""Build the crag_task1 pack from the CRAG task 1&2 dev file (v4, bz2 jsonl, Git LFS, ~740 MB).

Keeps `split == 1` (the public test questions). For each question writes `web/<qid>/<k>.html`
(cleaned page text, see `strip_html`), `web/<qid>/<k>.json` (page metadata), plus
`questions.jsonl` and `manifest.json`.
"""
from __future__ import annotations

import ast
import bz2
import hashlib
import html
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterator

from crb.data import fetch, packs

PACK = "crag_task1"
URL = ("https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/"
       "crag_task_1_and_2_dev_v4.jsonl.bz2?download=")
SOURCE = "https://github.com/facebookresearch/CRAG"
LICENSE = "CC BY-NC 4.0"
VERSION = "class-v1"
FILENAME = "crag_task_1_and_2_dev_v4.jsonl.bz2"
PUBLIC_SPLIT = 1

# Subtrees removed completely (tag and everything inside).
DROP = frozenset({
    "script", "style", "noscript", "svg", "template", "iframe", "canvas", "object", "embed",
    "audio", "video", "select", "button",
})
# Tags kept in the output (everything else is "unwrapped": the tag goes, its text stays).
KEEP = frozenset({
    "html", "head", "title", "body", "main", "article", "section", "header", "footer", "aside", "nav",
    "div", "p", "br", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "dl", "dt", "dd", "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption",
    "b", "strong", "i", "em", "u", "sub", "sup", "pre", "code", "blockquote", "figure", "figcaption",
})
VOID = frozenset({"br"})
KEEP_ATTRS = {"td": ("colspan", "rowspan"), "th": ("colspan", "rowspan")}  # links, ids, classes, styles all go
_WS = re.compile(r"\s+")


class _Stripper(HTMLParser):
    """Rebuilds a page keeping only text-bearing structure.

    Dropped: scripts, styles, noscript, svg, comments, form controls, and every attribute except table
    spans (so inline `style=`, classes, ids, data-*, link targets all go). Inline wrappers such as
    <span> and <a> are unwrapped: their text stays, the tag goes. Tags that are empty
    after cleaning are removed again (cascading), whitespace is collapsed.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.kinds: list[str] = []      # parallel to out: "open:<tag>", "text", "br", "close"
        self.stack: list[str] = []      # open KEEP tags
        self.skip_tag: str | None = None
        self.skip_depth = 0

    # -- helpers
    def _emit(self, s: str, kind: str) -> None:
        self.out.append(s)
        self.kinds.append(kind)

    def _pop(self) -> None:
        self.out.pop()
        self.kinds.pop()

    def handle_starttag(self, tag, attrs):
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_depth += 1
            return
        if tag in DROP:
            self.skip_tag, self.skip_depth = tag, 1
            return
        if tag not in KEEP:
            return
        keep = KEEP_ATTRS.get(tag, ())
        at = "".join(f' {k}="{html.escape(v, quote=True)}"' for k, v in attrs if k in keep and v)
        if tag in VOID:
            if self.kinds and self.kinds[-1] != "br":
                self._emit("<br>", "br")
            return
        self.stack.append(tag)
        self._emit(f"<{tag}{at}>", f"open:{tag}")

    def handle_endtag(self, tag):
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_depth -= 1
                if self.skip_depth == 0:
                    self.skip_tag = None
            return
        if tag not in KEEP or tag in VOID or tag not in self.stack:
            return
        # close any tags opened after `tag` first (malformed nesting)
        while self.stack:
            top = self.stack.pop()
            if self.kinds and self.kinds[-1] == f"open:{top}":
                self._pop()                  # empty element: drop it
            else:
                self._emit(f"</{top}>", "close")
            if top == tag:
                break

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if self.skip_tag is not None:
            return
        text = _WS.sub(" ", data)
        last = self.kinds[-1] if self.kinds else None
        if last is None or last == "br" or last.startswith("open:"):
            text = text.lstrip()          # a leading space at the start of an element is noise
        if not text:
            return
        if last == "text":
            if self.out[-1].endswith(" ") and text.startswith(" "):
                text = text[1:]
            self.out[-1] += html.escape(text, quote=False)
        else:
            self._emit(html.escape(text, quote=False), "text")

    def handle_comment(self, data):
        pass

    def handle_decl(self, decl):
        pass

    def handle_pi(self, data):
        pass

    def unknown_decl(self, data):
        pass

    def result(self) -> str:
        while self.stack:
            top = self.stack.pop()
            if self.kinds and self.kinds[-1] == f"open:{top}":
                self._pop()
            else:
                self._emit(f"</{top}>", "close")
        return "".join(self.out).strip()


def strip_html(page: str | None) -> str:
    """Return a lean version of an HTML page: text-bearing tags only, no scripts / styles /
    noscript / svg / comments / inline styles / ids / classes. Empty input gives an empty string."""
    if not page:
        return ""
    p = _Stripper()
    try:
        p.feed(page)
        p.close()
    except Exception:  # html.parser is lenient, but never let one bad page kill the build
        pass
    return p.result()


# ------------------------------------------------------------------ parsing

def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return list(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                pass
        return [s] if s else []
    return [v]


def iter_records(path: Path) -> Iterator[dict]:
    """Yield the records of a CRAG .jsonl.bz2 one at a time (bz2 stream-decompressed line by line)."""
    try:
        with bz2.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except EOFError as e:
        raise fetch.DownloadError(
            f"The CRAG download is incomplete ({path}). Delete it and run `crb fetch --pack crag_task1` again."
        ) from e


def question_from_record(rec: dict) -> dict:
    return {
        "qid": f"crag-{rec['interaction_id']}",
        "pack": PACK,
        "question": rec.get("query", ""),
        "answer": "" if rec.get("answer") is None else str(rec["answer"]),
        "alt_answers": [str(a) for a in _as_list(rec.get("alt_ans"))],
        "qtype": rec.get("question_type") or "unknown",
        "content_type": [],
        "query_time": rec.get("query_time"),
        "gold_pages": [],
    }


def _write(path: Path, data: bytes, files: dict[str, str], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    files[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()


def build_from_file(src: Path, data: Path, say: Callable[[str], None] = print, expected: int | None = None) -> dict:
    """Write the pack from a local .jsonl.bz2. `expected` = number of split==1 questions we hope
    to see (only used for the progress line)."""
    root = Path(data) / "packs" / PACK
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)
    files: dict[str, str] = {}
    questions: list[dict] = []
    n_pages = 0
    raw_bytes = 0
    kept_bytes = 0
    seen = 0
    prog = fetch.Progress(say)
    for rec in iter_records(Path(src)):
        seen += 1
        try:
            split = int(rec.get("split"))
        except (TypeError, ValueError):
            split = -1
        if split != PUBLIC_SPLIT:
            continue
        q = question_from_record(rec)
        qdir = root / "web" / q["qid"]
        for k, sr in enumerate(rec.get("search_results") or []):
            raw = sr.get("page_result") or ""
            cleaned = strip_html(raw)
            raw_bytes += len(raw)
            body = cleaned.encode("utf-8")
            kept_bytes += len(body)
            _write(qdir / f"{k}.html", body, files, root)
            meta = {key: sr.get(key) or "" for key in
                    ("page_name", "page_url", "page_snippet", "page_last_modified")}
            _write(qdir / f"{k}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"), files, root)
            n_pages += 1
        questions.append(q)
        of = f" of about {expected}" if expected else ""
        prog.tick(f"crag_task1: {len(questions)}{of} questions written (read {seen} records) ...")
    if not questions:
        raise ValueError("No split==1 questions found in the CRAG file; is this the right file?")
    questions.sort(key=lambda q: q["qid"])
    lines = [json.dumps(q, ensure_ascii=False) for q in questions]
    _write(root / "questions.jsonl", ("\n".join(lines) + "\n").encode("utf-8"), files, root)
    manifest = {
        "pack": PACK,
        "source": SOURCE,
        "source_file": URL.split("?")[0],
        "license": LICENSE,
        "version": VERSION,
        "files": dict(sorted(files.items())),
        "counts": {"questions": len(questions), "pages": n_pages},
    }
    from crb.data.vidore import write_manifest
    write_manifest(root, manifest)
    say(f"crag_task1: ready ({len(questions)} questions, {n_pages} pages; "
        f"page text {raw_bytes / 1e6:.0f} MB -> {kept_bytes / 1e6:.0f} MB after cleaning).")
    return manifest


def build(data: Path, say: Callable[[str], None] = print, keep_cache: bool = False) -> dict:
    cache = Path(data) / "_cache" / PACK
    dest = cache / FILENAME
    say("crag_task1: downloading the CRAG file (about 740 MB) ...")
    fetch.download(URL, dest, say=say, label="crag_task1")
    say("crag_task1: download complete, cleaning the web pages (several minutes) ...")
    manifest = build_from_file(dest, Path(data), say=say, expected=1335)
    if not keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
        try:
            (Path(data) / "_cache").rmdir()
        except OSError:
            pass
    return manifest

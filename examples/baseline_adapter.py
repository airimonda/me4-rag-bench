#!/usr/bin/env python3
"""Reference system for the class RAG benchmark: BM25 over page text + one cheap LLM call.

This is the floor every student can compare against, and a working example of the adapter
contract (docs/ADAPTER.md). Start it with `crb baseline` or:

    python3 examples/baseline_adapter.py --port 8000

How it works
    /ingest    reads every pages/*.md into an in-memory BM25 index (one per collection) and
               saves a copy to ~/.cache/crb/baseline/<collection>.pkl
    /retrieve  top-k page ids by BM25
    /answer    gold: uses the context the harness gives.   e2e: top 5 pages by BM25.
               web: cuts the 5 HTML pages into ~1,500 character chunks, keeps the 6 best.
               Then one streamed chat call to an OpenAI-compatible endpoint.

Environment
    CRB_LLM_BASE        default http://127.0.0.1:8790/v1 (the crb token proxy)
    OPENROUTER_API_KEY  your key (sent in the Authorization header)
    CRB_BASELINE_MODEL  default anthropic/claude-haiku-5.5

Standard library + requests only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import threading
from collections import Counter
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

NAME = "crb-baseline"
VERSION = "1.0"
CACHE_DIR = Path(os.environ.get("CRB_BASELINE_CACHE", "~/.cache/crb/baseline")).expanduser()
CONTEXT_CAP = 12_000          # characters of context sent to the model
E2E_PAGES = 5
WEB_CHUNK = 1_500
WEB_TOP = 6

SYSTEM_PROMPT = (
    "You answer questions from the context you are given. "
    "Answer in as few words as possible. "
    "If the context does not contain the answer and you are not sure, say: I don't know. "
    "If the question rests on a false premise, say: invalid question."
)

STOP = frozenset(
    "a an the of to in on for and or is are was were be been it its this that with as at by from "
    "what which who whom how when where why do does did".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


# ------------------------------------------------------------------ BM25

class BM25:
    """Okapi BM25 with an inverted index. Documents are identified by position; `ids` maps back."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, text in enumerate(texts):
            toks = tokenize(text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((i, tf))
        n = len(self.ids)
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0

    def to_state(self) -> dict:
        """Plain data (no classes), so the pickle loads from any script."""
        return {"ids": self.ids, "k1": self.k1, "b": self.b, "doc_len": self.doc_len,
                "postings": self.postings, "avg_len": self.avg_len}

    @classmethod
    def from_state(cls, st: dict) -> "BM25":
        self = cls([], [], st["k1"], st["b"])
        self.ids, self.doc_len, self.postings, self.avg_len = st["ids"], st["doc_len"], st["postings"], st["avg_len"]
        return self

    def scores(self, query: str) -> dict[int, float]:
        n = len(self.ids)
        out: dict[int, float] = {}
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0))
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top(self, query: str, k: int) -> list[str]:
        sc = self.scores(query)
        order = sorted(sc, key=lambda i: (-sc[i], i))
        return [self.ids[i] for i in order[:k]]

    def top_index(self, query: str, k: int) -> list[int]:
        sc = self.scores(query)
        return sorted(sc, key=lambda i: (-sc[i], i))[:k]


# ------------------------------------------------------------------ store (per collection)

class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.index: dict[str, BM25] = {}
        self.pages: dict[str, dict[str, str]] = {}   # collection -> page_id -> markdown

    def ingest(self, collection: str, directory: Path) -> int:
        files = sorted(Path(directory).glob("*.md"))
        if not files:
            raise ValueError(f"no .md pages found in {directory}")
        ids = [p.stem for p in files]
        texts = [p.read_text(encoding="utf-8", errors="replace") for p in files]
        bm = BM25(ids, texts)
        with self.lock:
            self.index[collection] = bm
            self.pages[collection] = dict(zip(ids, texts))
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_DIR / f"{_safe(collection)}.pkl", "wb") as f:
                pickle.dump({"bm25": bm.to_state(), "pages": self.pages[collection]}, f)
        except OSError as e:
            print(f"warning: could not save index cache: {e}", file=sys.stderr)
        return len(ids)

    def get(self, collection: str) -> BM25:
        with self.lock:
            bm = self.index.get(collection)
        if bm is not None:
            return bm
        path = CACHE_DIR / f"{_safe(collection)}.pkl"
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            bm = BM25.from_state(data["bm25"])
            with self.lock:
                self.index[collection] = bm
                self.pages[collection] = data["pages"]
            return bm
        except (OSError, pickle.PickleError, KeyError, EOFError, AttributeError):
            raise LookupError(f"collection '{collection}' is not ingested yet: call /ingest first")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "collection"


STORE = Store()


# ------------------------------------------------------------------ web pages (CRAG)

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def chunk_text(text: str, size: int = WEB_CHUNK) -> list[str]:
    """Cut at line breaks where possible so chunks stay near `size` characters."""
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > size:          # one very long line
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:size])
            line = line[size:]
        if cur and len(cur) + len(line) + 1 > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def web_context(pages: list[dict], question: str) -> str:
    chunks: list[str] = []
    for pg in pages:
        try:
            html = Path(pg["path"]).read_text(encoding="utf-8", errors="replace")
        except (OSError, KeyError):
            continue
        title = pg.get("page_name") or ""
        for c in chunk_text(html_to_text(html)):
            chunks.append(f"[{title}] {c}" if title else c)
    if not chunks:
        return ""
    bm = BM25([str(i) for i in range(len(chunks))], chunks)
    keep = sorted(bm.top_index(question, WEB_TOP))     # original order reads better
    return "\n\n".join(chunks[i] for i in keep)


# ------------------------------------------------------------------ answer building

def _cap(parts: list[str], cap: int = CONTEXT_CAP) -> str:
    out, used = [], 0
    for p in parts:
        room = cap - used
        if room <= 0:
            break
        out.append(p[:room])
        used += min(len(p), room) + 2
    return "\n\n".join(out)


def build_prompt(req: dict) -> tuple[str, list[str]]:
    """Return (user prompt, cited page ids) for an /answer request."""
    question = req.get("question")
    if not question:
        raise ValueError("missing 'question'")
    mode = req.get("mode", "e2e")
    cited: list[str] = []
    if mode == "gold":
        ctx = req.get("context") or []
        cited = [c.get("page_id") for c in ctx if c.get("page_id")]
        context = _cap([f"[{c.get('page_id')}]\n{c.get('markdown') or ''}" for c in ctx])
    elif mode == "web":
        pages = req.get("pages") or []
        context = _cap([web_context(pages, question)])
    elif mode == "e2e":
        bm = STORE.get(req.get("collection", ""))
        cited = bm.top(question, E2E_PAGES)
        texts = STORE.pages[req["collection"]]
        context = _cap([f"[{pid}]\n{texts.get(pid, '')}" for pid in cited])
    else:
        raise ValueError(f"unknown mode '{mode}' (use e2e, gold or web)")
    when = f"The question was asked at: {req['query_time']}\n" if req.get("query_time") else ""
    return f"Context:\n{context}\n\n{when}Question: {question}\nAnswer:", cited


def open_llm_stream(prompt: str):
    """Start the streamed chat call. Returns the requests.Response (already checked)."""
    base = os.environ.get("CRB_LLM_BASE", "http://127.0.0.1:8790/v1").rstrip("/")
    model = os.environ.get("CRB_BASELINE_MODEL", "anthropic/claude-haiku-5.5")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "max_tokens": 400,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(f"{base}/chat/completions", headers=headers, json=body, stream=True, timeout=(10, 90))
    except requests.RequestException as e:
        raise RuntimeError(f"could not reach the LLM at {base}: {e.__class__.__name__}. "
                           "Is `crb` running (it starts the proxy)? Is CRB_LLM_BASE right?")
    if r.status_code != 200:
        detail = r.text[:200].replace("\n", " ")
        r.close()
        hint = " Check OPENROUTER_API_KEY." if r.status_code in (401, 403) else ""
        raise RuntimeError(f"LLM returned HTTP {r.status_code}: {detail}.{hint}")
    return r


def iter_deltas(resp):
    """Yield text pieces from an OpenAI-style SSE stream."""
    resp.encoding = "utf-8"
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        for ch in obj.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                yield piece


# ------------------------------------------------------------------ HTTP server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            obj = json.loads(self.rfile.read(n).decode("utf-8"))
        except ValueError:
            raise ValueError("request body is not valid JSON")
        if not isinstance(obj, dict):
            raise ValueError("request body must be a JSON object")
        return obj

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            self._json({"ok": True, "name": NAME, "version": VERSION})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            req = self._body()
            if path == "/ingest":
                for key in ("collection", "dir"):
                    if key not in req:
                        raise ValueError(f"missing '{key}'")
                n = STORE.ingest(req["collection"], Path(req["dir"]))
                self._json({"ok": True, "pages": n})
            elif path == "/retrieve":
                for key in ("collection", "question"):
                    if key not in req:
                        raise ValueError(f"missing '{key}'")
                k = int(req.get("k", 10))
                self._json({"pages": STORE.get(req["collection"]).top(req["question"], k)})
            elif path == "/answer":
                self._answer(req)
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, LookupError) as e:
            self._safe_error(400, str(e))
        except Exception as e:
            print(f"error on {path}: {e!r}", file=sys.stderr)
            self._safe_error(502 if isinstance(e, RuntimeError) else 500, str(e))

    def _safe_error(self, status: int, msg: str) -> None:
        try:
            self._json({"error": msg}, status)
        except OSError:
            pass

    def _answer(self, req: dict) -> None:
        prompt, cited = build_prompt(req)
        resp = open_llm_stream(prompt)       # fail before any byte is sent: the harness sees an error
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
                self.wfile.flush()

            try:
                for piece in iter_deltas(resp):
                    send({"delta": piece})
            except requests.RequestException as e:
                print(f"LLM stream broke: {e!r}", file=sys.stderr)
            send({"done": True, "cited": cited})
        finally:
            resp.close()


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    ap = argparse.ArgumentParser(description="crb baseline adapter: BM25 + one LLM call")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    try:
        srv = make_server(a.host, a.port)
    except OSError as e:
        sys.exit(f"Cannot start on port {a.port}: {e.strerror}. Is another adapter running? Try --port {a.port + 1}.")
    print(f"{NAME} listening on http://{a.host}:{a.port}  (Ctrl-C to stop)")
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("note: OPENROUTER_API_KEY is not set; /answer will fail until you set it.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()

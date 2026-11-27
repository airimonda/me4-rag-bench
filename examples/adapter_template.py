#!/usr/bin/env python3
"""Adapter template for the class RAG benchmark (crb).

Copy this file, then replace every TODO with a call to YOUR system.
It runs as it is: it answers with placeholder text, so `crb check` can pass against it.

    python3 adapter_template.py --port 8000

Contract (full text: docs/ADAPTER.md). All bodies are JSON, UTF-8.
    GET  /health     -> {"ok": true, "name": "...", "version": "..."}
    POST /ingest     -> {"ok": true, "pages": N}            index the pages in "dir"
    POST /retrieve   -> {"pages": ["page_id", ...]}         best first, or null if you have no retriever
    POST /answer     -> stream (SSE) or {"answer": "...", "cited": [...]}

Standard library only. Use any language you like instead; only the HTTP contract matters.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NAME = "my-rag-system"      # shown in the report
VERSION = "0.1"

# Your index lives here, one entry per collection. TODO: replace with your own store.
INDEX: dict[str, dict[str, str]] = {}   # collection -> {page_id: markdown}
LOCK = threading.Lock()


# ------------------------------------------------------------------ your system goes here

def ingest(collection: str, directory: Path) -> int:
    """Read the pages in `directory` and build your index. Return the number of pages.

    ViDoRe collections: directory holds <page_id>.png (image) and <page_id>.md (markdown text).
    The page id is the file name without the extension. Reply only when you are ready:
    the harness waits (up to 60 minutes) for the answer.
    """
    pages = {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in sorted(directory.glob("*.md"))}
    # TODO: chunk, embed, build your vector store / BM25 / graph here.
    with LOCK:
        INDEX[collection] = pages
    return len(pages)


def retrieve(collection: str, question: str, k: int) -> list[str] | None:
    """Return up to k page ids, best first. Return None if your system has no retriever.

    Hits are counted per PAGE. If you search chunks, map each chunk back to its page id and
    remove duplicates before returning.
    """
    with LOCK:
        pages = list(INDEX.get(collection, {}))
    # TODO: search your index. Placeholder: first k pages.
    return pages[:k]


def answer(req: dict):
    """Yield the answer text in pieces, then return the list of cited page ids.

    This is a generator: `yield "text"` for every piece, and `return [ids]` at the end.
    req["mode"] tells you what to do:
      "e2e":  search your own index (req["collection"]) and answer.
      "gold": do NOT search. Use req["context"], a list of
              {"page_id", "markdown", "image_path"} the harness picked for you.
      "web":  CRAG. Answer from req["pages"], 5 local HTML files
              {"path", "page_name", "page_url", "page_snippet", "page_last_modified"},
              and req["query_time"] (when the question was asked). Do not search the internet.
    If you call an LLM through an OpenAI-compatible client, set its base URL to
    http://127.0.0.1:8790/v1 so the harness can count your tokens (docs/ADAPTER.md).
    """
    mode = req.get("mode", "e2e")
    question = req.get("question", "")
    if mode == "gold":
        cited = [c.get("page_id") for c in req.get("context") or []]
    elif mode == "web":
        cited = []          # CRAG has no page ids to cite
    else:
        cited = retrieve(req.get("collection", ""), question, 5) or []
    # TODO: build your prompt from the context and call your LLM. Placeholder answer below.
    for piece in ("This is a ", "placeholder ", "answer."):
        yield piece
    return cited


# ------------------------------------------------------------------ plumbing (no need to edit)

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
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
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}

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
                n = ingest(req["collection"], Path(req["dir"]))
                self._json({"ok": True, "pages": n})
            elif path == "/retrieve":
                self._json({"pages": retrieve(req["collection"], req["question"], int(req.get("k", 10)))})
            elif path == "/answer":
                self._stream_answer(req)
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # one bad request must never stop the server
            print(f"error on {path}: {e!r}", file=sys.stderr)
            try:
                self._json({"error": str(e)}, 500)
            except OSError:
                pass

    def _stream_answer(self, req: dict) -> None:
        """Server-sent events: data: {"delta": ...} lines, then data: {"done": true, "cited": [...]}.

        The harness stamps the first byte as time-to-first-token. If you cannot stream, send one
        JSON reply instead: {"answer": "...", "cited": [...]} (then first token = total time).
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        gen = answer(req)
        cited: list = []
        try:
            while True:
                try:
                    send({"delta": next(gen)})
                except StopIteration as stop:
                    cited = list(stop.value or [])
                    break
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            print(f"error in answer: {e!r}", file=sys.stderr)
            send({"delta": ""})
        send({"done": True, "cited": cited})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"{NAME} adapter listening on http://{a.host}:{a.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()

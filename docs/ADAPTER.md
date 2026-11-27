# Adapter contract

Your system is a small HTTP server on the Pi. Default address: `http://127.0.0.1:8000`. Any language works. All bodies are JSON, UTF-8.

Working examples:

- `examples/adapter_template.py`: empty skeleton. Copy it and fill in the TODOs.
- `examples/baseline_adapter.py`: the reference system (BM25 + one LLM call).

Check yours with `crb check`.

## The four calls

| Call | What it does |
|---|---|
| `GET /health` | Is the server up? |
| `POST /ingest` | Read the documents and build your index. |
| `POST /retrieve` | Return the best page ids for a question. |
| `POST /answer` | Answer a question. |

### GET /health

Reply: `{"ok": true, "name": "my-system", "version": "1.0"}`. The name and version appear in the report.

```bash
curl http://127.0.0.1:8000/health
```

### POST /ingest

Request: `{"collection": "vidore_v3_hr", "dir": "/home/pi/crb-data/packs/vidore_v3_hr/pages"}`

The folder has two files per page: `<page_id>.md` (text) and `<page_id>.png` (image). The page id is the file name without the extension.

Reply when ready: `{"ok": true, "pages": 1110}`. It may take minutes. The harness waits up to 60 minutes. The time is measured and reported.

```bash
curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' \
  -d '{"collection":"vidore_v3_hr","dir":"/home/pi/crb-data/packs/vidore_v3_hr/pages"}'
```

`/ingest` is called once per ViDoRe pack. CRAG is never ingested: its pages come with each question.

### POST /retrieve

Request: `{"collection": "vidore_v3_hr", "question": "...", "k": 10}`

Reply: `{"pages": ["page_id_1", "page_id_2", ...]}`, best first, at most `k`. Reply `{"pages": null}` if your system has no retriever. Then test 1 is skipped. Timeout: 30 seconds.

Pages are counted, not chunks. If you search chunks, map each chunk back to its page and remove duplicates.

```bash
curl -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"collection":"vidore_v3_hr","question":"What is the unemployment rate?","k":10}'
```

### POST /answer

Request:

```json
{"collection": "vidore_v3_hr", "question": "...",
 "mode": "e2e",
 "context": [{"page_id": "...", "markdown": "...", "image_path": "/abs/path.png"}],
 "pages":   [{"path": "/abs/web/crag-x/0.html", "page_name": "...", "page_url": "...",
              "page_snippet": "...", "page_last_modified": "..."}],
 "query_time": "03/05/2024, 23:18:31 PT"}
```

`mode` says what to do:

| mode | Test | What you get | What you do |
|---|---|---|---|
| `e2e` | 3 | `collection`, `question` | Search your own index, then answer. |
| `gold` | 2 | `context`: the right pages | Do not search. Answer from `context` only. |
| `web` | 4 | `pages`: 5 local HTML files, `query_time` | Answer from those 5 pages. Do not use the internet. |

`context` is only sent in `gold` mode. `pages` and `query_time` are only sent in `web` mode.

Reply, streaming (preferred): `Content-Type: text/event-stream`.

```
data: {"delta": "The rate "}

data: {"delta": "is 6.1%."}

data: {"done": true, "cited": ["page_id_1"]}
```

The first byte of answer text is the time to first token. Plain chunked `text/plain` also works.

Reply, not streaming: `{"answer": "...", "cited": ["page_id_1"]}`. Then first token = total time.

`cited` is the list of page ids you used. For `web` mode use `[]`. Timeout: 120 seconds. A timeout or an error is recorded as a failed answer. It never stops the run.

Short answers score best. For CRAG, "I don't know" counts as better than a wrong answer, and "invalid question" is right when the question has a false premise.

```bash
curl -N -X POST http://127.0.0.1:8000/answer -H 'Content-Type: application/json' \
  -d '{"collection":"vidore_v3_hr","question":"What is the unemployment rate?","mode":"e2e"}'
```

## A minimal server in Python

```python
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ok": True, "name": "mine", "version": "1"}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
```

The full version with all four calls and streaming is `examples/adapter_template.py`.

## The token proxy

While `crb` runs, it serves an OpenAI-compatible proxy at `http://127.0.0.1:8790/v1`. Point your LLM client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8790/v1", api_key=os.environ["OPENROUTER_API_KEY"])
```

or with `requests`:

```python
requests.post("http://127.0.0.1:8790/v1/chat/completions",
              headers={"Authorization": f"Bearer {key}"}, json={...})
```

The proxy forwards to OpenRouter. Your key passes through untouched. It supports streaming. For each call it records the time, model, tokens, cost and which question was active. Without the proxy your report shows cost as "not measured". That is fine, just not as informative.

Calls that skip the proxy (a local model on the Pi, for example) cost nothing in tokens, but show up in the speed and Pi-health numbers.

## Common architectures

Pick the one closest to yours.

### Text chunks (BM25, vectors, hybrid)

- `/ingest`: read `<page_id>.md`, cut into chunks, embed or index them. Remember each chunk's page id.
- `/retrieve`: search chunks, map to pages, remove duplicates, return the top `k` page ids.
- `/answer` `e2e`: retrieve, put the top chunks in the prompt, answer. `gold`: use the `markdown` of `context` directly. `web`: extract text from the 5 HTML files, chunk, keep the best chunks.

### Page images (ColPali style)

- `/ingest`: embed every `.png` page. This is slow on a Pi; plan for it. The time is reported.
- `/retrieve`: return the best pages by image score.
- `/answer` `e2e`: send the top page images to a vision model. `gold`: use `image_path` from `context` (and `markdown` if you want both). `web`: HTML only, so use text.

### Agent with tools

- `/ingest`: build whatever the tools need (an index, a file list).
- `/retrieve`: run the agent's search tool alone and return page ids. If the agent has no separate search step, reply `{"pages": null}`.
- `/answer` `e2e`: run the full agent loop. Stream the final answer. Tool calls before it count as time to first token, so stream early if you can. `gold`: switch the search tool off and use `context`. `web`: give the 5 pages as the only source.

### Long context, no retriever

- `/ingest`: just load and remember the pages.
- `/retrieve`: reply `{"pages": null}`. Test 1 will be skipped.
- `/answer` `e2e`: put as many pages as fit into the prompt. Cap the size so cost stays sane. The report counts how many times you hit your cap if you use the proxy. `gold`: put `context` in the prompt. `web`: put the 5 pages in.

### Graph RAG

- `/ingest`: extract entities and links from the pages and build the graph. Slow; the time is reported.
- `/retrieve`: walk the graph, return the pages that support the best nodes.
- `/answer` `e2e`: retrieve through the graph, then answer from those pages. `gold`: skip the graph and use `context`. `web`: build a tiny graph from the 5 pages, or just read them.

In every design: return page ids (not chunk ids), stream if you can, and keep answers short.

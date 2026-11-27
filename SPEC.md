# me4-rag-bench — build spec

This file is the contract between the modules. Every module owner builds to it. If something here
is wrong or missing, the module owner stops and reports; nobody silently changes an interface.

## What it is

A plug-and-play benchmark that each AI 231 student runs **on their own Raspberry Pi 4/5** against
their own document-QA (RAG) system. It feeds open datasets with published results to the student's
system over HTTP, times everything on the Pi, samples the Pi's health every second, scores answers
with the papers' own judges, and writes an offline `report.html` that places the student's run
inside the published results tables.

Results are compared with published systems only.

Design references (read them, they are the target look and wording):
- `docs/design/report_mockup.html` — the report. `report.html` must look and behave like it.
- `docs/design/explainer.html` — the explainer page for the class (axes, datasets, costs).
- Ease of use target: `airimonda/vcm-benchmark` (one command, wizard questions, pre-flight,
  plain error messages, done notification). Report target: `trueshotjames/ai231-me3-full-duplex-benchmarking`.

## Fixed decisions

- Python >= 3.10. Runs on 64-bit Raspberry Pi OS (Pi 4 4 GB is the floor) and on macOS/Linux laptops
  for development. Windows is not a target.
- Runtime dependencies kept small: `requests`, `pyarrow` (only for building packs), standard library
  otherwise. No numpy/pandas at run time. No heavy web framework: use `http.server` / `socketserver`.
- Command: `crb` (installed by `pip install -e .`, also `python -m crb`).
- All network calls to LLMs go through OpenRouter's OpenAI-compatible API
  (`https://openrouter.ai/api/v1`). Judge key: env `CRB_JUDGE_KEY`, falling back to
  `OPENROUTER_API_KEY`. Keys are never written to any output file or log.
- Time: all latencies measured with `time.perf_counter()` on the Pi; timestamps in results use
  ISO 8601 UTC plus `time.monotonic()` seconds since run start (`t`), so health samples and
  questions line up.
- Seed for every sample: 7. Sample name: `class-v1`.

## The datasets ("packs")

| Pack id | Source | Questions | Notes |
|---|---|---|---|
| `vidore_v3_hr` | HF `vidore/vidore_v3_hr` (CC-BY-4.0) | 318 English | 1,110 pages, image + markdown per page |
| `vidore_v3_computer_science` | HF `vidore/vidore_v3_computer_science` (CC-BY-4.0) | 215 English | 1,360 pages |
| `crag_task1` | `facebookresearch/CRAG` `data/crag_task_1_and_2_dev_v4.jsonl.bz2` (Git LFS; CC BY-NC 4.0) | 1,335 public test (`split == 1`) | 5 HTML pages per question |

ViDoRe HF configs: `corpus` (corpus_id, image, doc_id, markdown, page_number_in_doc), `queries`
(query_id, query, language, query_types, content_type, answer, ...), `qrels` (query_id, corpus_id,
score, content_type, bounding_boxes), `documents_metadata`. Keep only `language == "english"`.

### On-disk layout (written by `crb fetch`, under `$CRB_DATA`, default `~/crb-data`)

```
packs/<pack_id>/
  manifest.json            # {"pack": id, "source": url, "license": "...", "version": "class-v1", "files": {relpath: sha256}, "counts": {...}}
  pages/<page_id>.png      # ViDoRe only: page image
  pages/<page_id>.md       # ViDoRe: markdown from the dataset. CRAG: not used
  web/<qid>/<k>.html       # CRAG only: page k (0..4) of question qid, scripts/styles/comments removed
  web/<qid>/<k>.json       # CRAG only: {"page_name","page_url","page_snippet","page_last_modified"}
  questions.jsonl          # one question per line, schema below
  qrels.jsonl              # ViDoRe only: {"qid","page_id","score"}  (score = graded relevance from the dataset)
samples/class-v1.json      # {"seed":7, "packs": {"vidore_v3_hr": {"retrieval": [all qids], "answer": [100 qids]}, "vidore_v3_computer_science": {...}, "crag_task1": {"answer": [400 qids]}}}
```

`page_id` for ViDoRe = `<doc_id>__p<page_number_in_doc>` made filesystem-safe (only `[A-Za-z0-9._-]`).
`qid` = `<pack short>-<original id>`: `hr-<query_id>`, `cs-<query_id>`, `crag-<interaction_id>`.

`questions.jsonl` schema:
```json
{"qid": "hr-12", "pack": "vidore_v3_hr", "question": "...", "answer": "gold answer",
 "alt_answers": [], "qtype": "multi-hop", "content_type": ["Chart","Table"],
 "query_time": null, "gold_pages": ["doc__p14", "doc__p15"]}
```
CRAG: `qtype` = `question_type`, `query_time` set, `alt_answers` = `alt_ans`, `gold_pages` = [] .

Sampling (`samples/class-v1.json`): stratified by `qtype`, proportional allocation, largest-remainder
rounding, seeded `random.Random(7)`, sorted output. ViDoRe answer sample 100 per pack, CRAG 400.
Retrieval always uses every English ViDoRe question.

Dev pack (`crb fetch --dev`, also used by tests): `packs/dev_vidore` (12 pages, 8 questions, tiny PNGs,
markdown pages written by the generator) and `packs/dev_crag` (6 questions × 5 small HTML pages) with
the same schemas, plus `samples/dev.json`. Generated locally, no network. Deterministic.

## The adapter contract (the student's system)

The student runs an HTTP server on the Pi (default `http://127.0.0.1:8000`). Any language.
All bodies are JSON, UTF-8.

| Method & path | Request | Response |
|---|---|---|
| `GET /health` | – | `{"ok": true, "name": "...", "version": "..."}` |
| `POST /ingest` | `{"collection": "vidore_v3_hr", "dir": "/abs/path/packs/vidore_v3_hr/pages"}` | `{"ok": true, "pages": 1110}` when ready (may take minutes; harness waits up to 60 min) |
| `POST /retrieve` | `{"collection": "...", "question": "...", "k": 10}` | `{"pages": ["page_id", ...]}` best first; may be `null` if the system has no retriever |
| `POST /answer` | see below | streamed text (preferred) or JSON |

`/answer` request:
```json
{"collection": "vidore_v3_hr", "question": "...",
 "mode": "e2e" | "gold" | "web",
 "context": [{"page_id": "...", "markdown": "...", "image_path": "/abs/...png"}],  // only mode "gold": search must be skipped
 "pages":   [{"path": "/abs/.../web/crag-x/0.html", "page_name": "...", "page_url": "...", "page_snippet": "...", "page_last_modified": "..."}], // only mode "web" (CRAG): answer from these 5 pages
 "query_time": "03/05/2024, 23:18:31 PT"}  // only mode "web"
```
`/answer` response, either:
- **Streaming** (preferred): `Content-Type: text/event-stream` with lines `data: {"delta": "text"}` and a
  final `data: {"done": true, "cited": ["page_id", ...]}`, or `text/plain` chunked raw text.
  First byte of answer text = time to first token.
- **JSON**: `{"answer": "...", "cited": ["page_id", ...]}`. Then first token = total time.

Timeouts: `/answer` 120 s, `/retrieve` 30 s. A timeout or error is recorded as a failed answer, never
crashes the run.

## Token proxy

`crb` starts a local OpenAI-compatible proxy at `http://127.0.0.1:8790/v1` for the whole run. Students
point their LLM client's base URL there (their own API key passes through untouched in the
`Authorization` header). The proxy forwards to `CRB_UPSTREAM` (default `https://openrouter.ai/api/v1`),
supports streaming (SSE) passthrough, and records per call: time, model, prompt/completion tokens,
cost (OpenRouter `usage.cost` when present; request `usage: {"include": true}` is added), and which
question was active (the runner sets the current qid in shared state). Written to `proxy.jsonl`.
If a student does not use the proxy, cost shows as "not measured" — never an error.

## Run phases and output

`crb run` phases, in order: preflight → ingest (each ViDoRe pack, timed) → axis 1 retrieval (all English
ViDoRe questions, `/retrieve`) → axis 2 gold pages (`/answer` mode gold, answer sample) → axis 3
end-to-end (`/answer` mode e2e, same sample) → axis 4 CRAG (`/answer` mode web, CRAG sample) → judging →
scoring → report. One question at a time. The first 3 questions after each ingest are marked
`warmup: true` (still scored, excluded from latency percentiles). The run is resumable: a re-run with
`--resume RUN_DIR` skips qids already in `answers.jsonl`.

Modes: `--smoke` (dev packs or 10 questions per axis, cheap judge `anthropic/claude-haiku-5.5`, report
marked "not comparable"), `--official` (class-v1 sample, papers' judges), `--full` (every question).

Run directory: `runs/<student>/<YYYYmmdd-HHMMSS>/`:
```
run.json        # session: student, mode, crb version, git commit of crb, adapter url + /health reply, pi info, start/end, sample name
answers.jsonl   # one line per question per axis (schema below)
health.csv      # 1 Hz samples (columns below)
proxy.jsonl     # token proxy log
judgments.jsonl # one line per judged answer: {"qid","axis","judge","label","score","raw"}
results.json    # every number in the report (schema below)
report.html     # offline, single file
summary.md      # verdict as text for slides
run.log         # human-readable log
```

`answers.jsonl` line:
```json
{"qid": "hr-12", "pack": "vidore_v3_hr", "axis": 3, "warmup": false, "t": 812.4,
 "retrieved": ["p1","p2"], "answer": "...", "cited": ["p1"], "error": null,
 "latency_ms": {"first_token": 2100, "total": 6400, "retrieve": null}}
```
Axis 1 lines have `answer: null`, `latency_ms.retrieve` set.

`health.csv` columns: `t,iso,temp_c,throttled_hex,under_voltage_now,freq_capped_now,throttled_now,soft_temp_now,arm_mhz,core_v,cpu_pct,cpu_max_core_pct,load1,mem_avail_mb,swap_used_mb,adapter_rss_mb,disk_free_mb,net_rtt_ms,phase`.
Missing values (e.g. no `vcgencmd` on a laptop) are empty. `phase` ∈ preflight, ingest, retrieval, gold,
e2e, crag, judging.

`results.json` top level:
```json
{"session": {...run.json...}, "mode": "official", "comparable": true,
 "metrics": {
   "retrieval": {"vidore_v3_hr": {"ndcg10": 48.2, "recall10": 61.0, "n": 318}, "vidore_v3_computer_science": {...}},
   "gold":  {"vidore_v3_hr": {"acc": 68.0, "ci95": 7.7, "n": 100}, ...},
   "e2e":   {"vidore_v3_hr": {"acc": 52.0, "ci95": 7.7, "n": 100, "by_content_type": {"Text": {"acc":71,"n":84}, ...}}, ...},
   "crag":  {"crag_task1": {"truthfulness": 21.5, "ci95": 6.5, "accurate": 47.0, "hallucinated": 25.5, "missing": 27.5, "n": 400, "by_qtype": {...}}},
   "diagnosis": {"vidore_v3_hr": {"correct": 52, "retrieval_miss": 21, "generation_miss": 18, "integration_miss": 9}, ...},
   "latency": {"first_token": {"p50": 2100, "p90": 4800, "max": 11200, "n": 600}, "total": {...}, "retrieve": {...}, "ingest": {"vidore_v3_hr": {"seconds": 330, "pages": 1110}}, "cold_start_ms": 9800, "hist_first_token_500ms": [4,38,...]},
   "cost": {"measured": true, "system_usd": 1.46, "judge_usd": 1.90, "tokens_in": 6100000, "tokens_out": 900000, "per_answer_usd": 0.0024, "context_cap_hits": 0},
   "health": {"temp_max": 78.4, "throttled_share": 0.031, "throttled_seconds": 161, "under_voltage": false, "cpu_p95": 88, "busiest_core_p95": 99, "mem_avail_min_mb": 612, "arm_mhz_min": 1000, "net_rtt_p50": 38, "net_rtt_p90": 120, "available": true}
 },
 "published": {...the literature tables, see crb/literature.py...},
 "ranks": {"retrieval.vidore_v3_hr": {"rank": 17, "of": 22, "median": 53.3, "side": "below"}, ...},
 "worst": [ {"qid","pack","question","gold","answer","tag"} x up to 10 ],
 "notes": ["..."]}
```

## Scoring rules

- nDCG@10 and Recall@10 per question from `qrels.jsonl` graded scores, averaged ×100. Exact (no CI).
- ViDoRe judge (axes 2–3): `openai/gpt-5.2`, `reasoning: {"effort": "medium"}`, temperature 0, prompt in
  `crb/prompts/vidore_judge.txt` modelled on ViDoRe V3 Fig. 24: expert judge compares the answer with the gold
  answer, returns JSON `{"explanation": "...", "judgment": "Correct" | "Partially Correct" | "Incorrect"}`.
  Accuracy = share judged "Correct".
- CRAG judge (axis 4): CRAG auto-eval. Rule step first: normalized exact match with gold/alt → accurate; reply
  containing "i don't know" (case-insensitive) or empty → missing; "invalid question" counts accurate only when
  `qtype == false_premise`. Otherwise two LLM judges, `openai/gpt-3.5-turbo-0613` and
  `meta-llama/llama-3.1-70b-instruct`, each returns accurate/incorrect; per question score = mean of the two
  (+1 accurate, −1 incorrect). Truthfulness = mean score × 100. Report accuracy, hallucination, missing rates.
- 95% intervals: percentile bootstrap, 2,000 resamples, seed 7, with finite-population correction
  `sqrt((N-n)/(N-1))` applied to the half-width when sampling from a pack of size N. Exact when n == N.
- Diagnosis per ViDoRe answer-sample question: correct (axis 3 Correct) / retrieval miss (no gold page in the
  axis-1 top 10) / generation miss (gold-pages axis 2 not Correct) / integration miss (otherwise).
- Ranks: insert the student value into each published table (`crb/literature.py`), rank 1 = best; median of
  the published rows only; side = above/below/at.
- Judge results are cached by `(judge model, prompt hash, qid, answer hash)` in `~/.cache/crb/judge.sqlite`
  so `crb score` can be re-run for free.

## Pi health sampler

Thread at 1 Hz. Sources: `vcgencmd measure_temp`, `vcgencmd get_throttled` (bits 0–3 now, 16–19 since boot),
`vcgencmd measure_clock arm`, `vcgencmd measure_volts core`, `/proc/stat` (all cores + per core),
`/proc/loadavg`, `/proc/meminfo`, `os.statvfs` on the data dir, RSS of the adapter process (found from the
adapter port via `/proc/net/tcp` inode → `/proc/<pid>/fd`; or `--adapter-pid`), TCP connect time to
`openrouter.ai:443` every 10 s. Each source fails soft (empty value). Sampler overhead must stay under 1% of one core.

## CLI

```
crb                 # wizard: student number, adapter URL, mode; then check + run + report
crb fetch [--dev] [--pack ID]   # download/build packs and samples into $CRB_DATA
crb check           # pre-flight only: data present + checksums, adapter /health, endpoints answer a probe, disk, vcgencmd, judge key, proxy port free
crb run [--smoke|--official|--full] [--student S] [--adapter URL] [--resume DIR] [--no-judge]
crb score RUN_DIR [--judge-model M]   # (re)judge and (re)score
crb report RUN_DIR  # rebuild report.html + summary.md from results.json
crb serve [RUN_DIR] [--port 8302]     # serve runs/ read-only on 0.0.0.0 and print the http://<pi-ip>:8302 link
crb baseline        # start the reference adapter (examples/baseline_adapter.py) on :8000
```
Messages are plain English, one line per problem with the fix. Ctrl-C stops cleanly; everything done is kept.
At the end: a terminal summary (rank and above/below median per test, latency p90, cost, peak temp) and the
report path. On a laptop, a desktop notification when done (macOS `osascript`, Linux `notify-send`), ignored
if unavailable.

## Module ownership

| Owner | Files |
|---|---|
| A — harness | `crb/__init__.py`, `crb/__main__.py`, `crb/cli.py`, `crb/wizard.py`, `crb/config.py`, `crb/adapter.py` (HTTP client + streaming timing), `crb/preflight.py`, `crb/runner.py`, `crb/proxy.py`, `crb/state.py` (shared current-qid/phase), `crb/serve.py`, `pyproject.toml`, `tests/test_adapter.py`, `tests/test_runner.py`, `tests/test_proxy.py` |
| B — data | `crb/data/__init__.py`, `crb/data/fetch.py`, `crb/data/vidore.py`, `crb/data/crag.py`, `crb/data/sample.py`, `crb/data/devpack.py`, `crb/data/packs.py` (read API: `load_questions(pack)`, `load_qrels(pack)`, `load_sample(name)`, `page_paths(pack, page_id)`, `crag_pages(pack, qid)`, `data_dir()`), `tests/test_data.py` |
| C — scoring + report | `crb/score/__init__.py`, `crb/score/retrieval.py`, `crb/score/judge.py`, `crb/score/crag_eval.py`, `crb/score/stats.py`, `crb/score/diagnosis.py`, `crb/score/latency.py`, `crb/score/cost.py`, `crb/score/results.py` (builds results.json from a run dir), `crb/literature.py`, `crb/prompts/*.txt`, `crb/report/__init__.py`, `crb/report/html.py`, `crb/report/template.html`, `crb/report/summary.py`, `tests/test_score.py`, `tests/test_report.py` |
| D — Pi + examples + docs | `crb/health.py` (sampler class: `HealthSampler(data_dir, adapter_port, out_csv, state).start()/stop()/summary()`), `examples/baseline_adapter.py`, `examples/adapter_template.py`, `scripts/install_pi.sh`, `README.md`, `docs/GUIDE.md`, `docs/METHOD.md`, `docs/ADAPTER.md`, `docs/TROUBLESHOOTING.md`, `datasets/LICENSES.md`, `tests/test_health.py` |

Cross-module calls, fixed signatures:
- `crb.data.packs`: as listed above; `data_dir() -> Path` honours `$CRB_DATA`.
- `crb.state.State`: thread-safe object with `.phase`, `.qid`, `.axis` (A owns; B, C, D only read).
- `crb.health.HealthSampler` (D) used by `crb.runner` (A).
- `crb.score.results.build(run_dir: Path, judge: bool = True, mode: str = "official") -> dict` (C) writes
  `judgments.jsonl` + `results.json` and returns the dict; used by runner and `crb score`.
- `crb.report.html.write(run_dir: Path, results: dict) -> Path` and `crb.report.summary.write(run_dir, results) -> Path` (C).
- `crb.data.fetch.main(args)` (B) used by `crb fetch`.

Tests: `pytest`, offline, under 60 s total, using the dev pack (generate it in a tmp `$CRB_DATA`) and a fake
adapter / fake upstream LLM served from the test itself. No test calls a real API.

Do not commit. Do not touch files you do not own (except reading). Do not `git stash`.

# me4-rag-bench

`crb` is a benchmark for document question-answering (RAG) systems running on a Raspberry Pi. It evaluates a system through a small HTTP interface, scores it on open datasets that have published results, and produces an offline report that places the run within those published results.

Each run reports four layers of answer quality, latency measured on the device, token cost, and device health (temperature, throttling, power, CPU and memory) sampled once per second. Results are expressed as a rank and as a position relative to the median of the published systems on the same data, metric and judge.

> **Required before use:** your system must be wrapped in an *adapter*, a small HTTP server on the Pi that exposes four endpoints. The benchmark cannot run without it. See [Before you start](#before-you-start-an-adapter-is-required).

## Features

- **Architecture-agnostic.** Any system that implements four HTTP endpoints can be evaluated, whether it uses text chunks, page images, tool-using agents, long context or a knowledge graph.
- **Comparable with the literature.** Datasets, metrics and judges follow ViDoRe V3 and CRAG, so scores can be read against the published tables.
- **Layered diagnosis.** Retrieval, generation with the correct pages, and the full pipeline are scored separately, and every question is attributed to the layer that failed.
- **Measured on the device.** Latency, ingest time and resource use are recorded on the Pi itself; token use is counted by a local proxy rather than self-reported.
- **Plug and play.** One install script, an interactive wizard, pre-flight checks, resumable runs and a self-contained HTML report.

## Before you start: an adapter is required

`crb` does not know how your system is built and does not import its code. It talks to your system only through an **adapter**: a small HTTP server that runs on the Pi next to your system and translates the benchmark's requests into calls to your own pipeline. **The benchmark cannot run until your adapter is working.** `crb check` fails with `cannot connect to the adapter` until it is.

The adapter is a thin layer. It should call the same ingestion, retrieval and answering code your system already uses, with the same models and prompts you would use on demo day. Do not tune it for the benchmark questions. For an existing pipeline, writing the adapter typically takes one to two hours.

### What the adapter must do

It listens on `http://127.0.0.1:8000` (another address can be passed with `--adapter`) and implements four endpoints. All request and response bodies are JSON (UTF-8).

| Endpoint | When the benchmark calls it | What the adapter receives | What it must return | What is measured |
|---|---|---|---|---|
| `GET /health` | Pre-flight and at the start of a run | – | `{"ok": true, "name": "...", "version": "..."}` | Name and version are shown in the report |
| `POST /ingest` | Once per ViDoRe dataset, before any questions | `collection` and `dir`, a folder holding `<page_id>.md` (page text) and `<page_id>.png` (page image) for every page | `{"ok": true, "pages": N}` once the collection is indexed (up to 60 minutes) | Ingest time |
| `POST /retrieve` | Layer 1, once per ViDoRe question | `collection`, `question`, `k` (10) | `{"pages": ["page_id", ...]}`, best first, or `{"pages": null}` if the system has no separate retrieval step | nDCG@10, Recall@10, retrieval latency |
| `POST /answer` | Layers 2, 3 and 4, once per question | `question` and `mode` (see below) | A streamed answer, or `{"answer": "...", "cited": ["page_id", ...]}` | Accuracy, time to first token, total time, tokens |

`/answer` is called in three modes, and the adapter must respect each one:

| `mode` | Layer | Extra fields sent | Required behaviour |
|---|---|---|---|
| `e2e` | 3 | `collection` | Run your full pipeline: search your own index, then answer. |
| `gold` | 2 | `context`: the correct pages, each with `page_id`, `markdown` and `image_path` | Do **not** search. Answer only from the pages given. |
| `web` | 4 (CRAG) | `pages`: five local HTML files with their metadata; `query_time` | Answer only from those five pages. Do not access the internet. Say "I don't know" when the pages do not support an answer, and "invalid question" when the question has a false premise. |

### Rules that affect your scores

1. **Return page ids, not chunk ids.** A page id is the file name without its extension (for example `report_2023__p14`). If your index stores chunks, map each chunk back to its page and remove duplicates before returning.
2. **Stream the answer** as server-sent events (`data: {"delta": "..."}` lines, then `data: {"done": true, "cited": [...]}`) or chunked plain text. With a non-streaming JSON reply, time to first token equals total time.
3. **Keep answers short.** Judges compare against short reference answers; for CRAG a wrong answer scores −1 and "I don't know" scores 0.
4. **Send LLM calls through the token proxy** (`http://127.0.0.1:8790/v1`, OpenAI-compatible) so that cost is measured. Your API key passes through unchanged. Without the proxy, cost is reported as "not measured".
5. **Stay within the time limits:** 120 seconds per answer and 30 seconds per retrieval. A timeout or error counts as a failed answer; the run continues.
6. **Keep the adapter running for the whole run** (for example in `tmux`). Questions are sent one at a time.

### How to build it

1. Copy the template: `cp examples/adapter_template.py my_adapter.py`. It runs as it is and returns placeholder answers, so the wiring can be tested before your code is connected.
2. Fill in its three functions with calls to your system:
   - `ingest(collection, directory)`: read the `.md` and/or `.png` files in `directory`, build your index, and return the number of pages.
   - `retrieve(collection, question, k)`: search your index and return up to `k` page ids, best first (or `None` if you have no retriever).
   - `answer(req)`: build the prompt for `req["mode"]` as described above, call your model, `yield` the answer text piece by piece, and `return` the cited page ids.
3. Start it on the Pi: `python3 my_adapter.py --port 8000`.
4. Test each endpoint by hand:

   ```bash
   curl http://127.0.0.1:8000/health
   curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' \
     -d '{"collection":"vidore_v3_hr","dir":"'"$HOME"'/crb-data/packs/vidore_v3_hr/pages"}'
   curl -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
     -d '{"collection":"vidore_v3_hr","question":"What was the employment rate in 2023?","k":10}'
   curl -N -X POST http://127.0.0.1:8000/answer -H 'Content-Type: application/json' \
     -d '{"collection":"vidore_v3_hr","question":"What was the employment rate in 2023?","mode":"e2e"}'
   ```

5. Run `crb check`, then a practice run (`--smoke`) before the official run.

The adapter can be written in any language; only the HTTP interface matters. `crb baseline` starts the reference adapter (`examples/baseline_adapter.py`, BM25 plus one model call), which is a complete working example. [docs/ADAPTER.md](docs/ADAPTER.md) gives the full request and response formats, a streaming example, and guidance for text-chunk, page-image, agent, long-context and graph-based systems.

## Requirements

- **Raspberry Pi:** Raspberry Pi 4 (4 GB) or Raspberry Pi 5, 64-bit Raspberry Pi OS, Python 3.10 or later, about 2.5 GB of free disk space
- **Laptop (recommended):** macOS, Linux or Windows 10/11 with Python 3.10+ and the OpenSSH client, on a network from which `ssh` to the Pi works
- **API key:** an OpenRouter key for the judges (`OPENROUTER_API_KEY` or `CRB_JUDGE_KEY`)

## Running the benchmark

The benchmark always runs on the Pi. The recommended way is to control it from a laptop: `crb remote` starts the run on the Pi, shows its progress, and, when it finishes, copies the results to the laptop and opens the report in the browser. The run continues on the Pi if the laptop disconnects.

### Step 1. Install on the Pi (once)

```bash
git clone https://github.com/airimonda/me4-rag-bench.git
cd me4-rag-bench
bash scripts/install_pi.sh
source .venv/bin/activate
crb fetch                       # downloads and builds the datasets (about 1.2 GB, 10–20 minutes)
```

Store the API key on the Pi so that runs started over SSH can read it:

```bash
mkdir -p ~/.config/crb
echo 'OPENROUTER_API_KEY=sk-or-...' > ~/.config/crb/env
chmod 600 ~/.config/crb/env
```

### Step 2. Install on the laptop (once)

```bash
git clone https://github.com/airimonda/me4-rag-bench.git
cd me4-rag-bench
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
```

Check that the laptop can reach the Pi: `ssh <user>@<pi-address>` should open a shell. Setting up an SSH key (`ssh-keygen`, then `ssh-copy-id <user>@<pi-address>`) avoids repeated password prompts.

### Step 3. Start your adapter on the Pi

This step requires the adapter described in [Before you start](#before-you-start-an-adapter-is-required). Start it in a way that keeps it running after you log out, for example in `tmux`:

```bash
tmux new -s adapter             # on the Pi
cd ~/me4-rag-bench && source .venv/bin/activate
python3 my_adapter.py --port 8000   # your adapter; or `crb baseline` for the reference system
# detach with Ctrl-b, then d
```

`examples/adapter_template.py` is a minimal skeleton to adapt; `crb baseline` starts the BM25 reference system.

### Step 4. Check the setup

On the Pi:

```bash
crb check
```

Every line should read `ok` (on a Pi, `WARN` lines are informational). Fix any `FAIL` line using the suggestion printed under it. The check makes at most one model call.

### Step 5. Run from the laptop

```bash
crb remote <user>@<pi-address> --smoke --student <student-number>      # practice run, a few minutes, a few cents
crb remote <user>@<pi-address> --official --student <student-number>   # official run, about 30–90 minutes
```

When the run finishes, the run folder is copied to `~/crb-runs/<student>/<date-time>/` on the laptop and `report.html` opens in the default browser.

If you stop following with Ctrl-C, or the laptop sleeps, the run keeps going on the Pi. To collect the report later:

```bash
crb pull <user>@<pi-address> --wait     # waits for the current run to finish, then copies and opens it
crb pull <user>@<pi-address>            # copies and opens the newest finished run
```

### Alternative: run directly on the Pi

```bash
crb                             # interactive wizard
crb run --official --student <student-number>
```

Use `tmux` for long runs so that a dropped SSH session does not stop the run. The report is written to `runs/<student>/<date-time>/report.html` on the Pi; open it on the laptop with `crb pull <user>@<pi-address>`, or serve it with `crb serve`.

### Run modes

| Mode | Questions | Judges | Comparable with the papers |
|---|---|---|---|
| `--smoke` | 10 per test | one inexpensive judge | No; for checking the setup |
| `--official` | fixed `class-v1` sample (533 retrieval, 200 ViDoRe, 400 CRAG) | the papers' judges | Yes, with 95% intervals |
| `--full` | every question | the papers' judges | Yes, with no sampling error |

### Other commands

| Command | Purpose |
|---|---|
| `crb run --resume RUN_DIR` | Continue an interrupted run; answered questions are skipped |
| `crb score RUN_DIR` | Re-judge and re-score a run (cached judgments are reused at no cost) |
| `crb report RUN_DIR` | Rebuild `report.html` and `summary.md` from `results.json` |
| `crb serve` | Serve the runs folder read-only over HTTP |
| `crb fetch --dev` | Build a tiny offline dataset for development |

## Evaluation

| Layer | Question | Metric | Data |
|---|---|---|---|
| 1. Retrieval | Does the system find the right pages? | nDCG@10, Recall@10 | ViDoRe V3, all English questions (533) |
| 2. Grounded generation | Given the correct pages, is the answer correct? | Accuracy, GPT-5.2 judge | ViDoRe V3, 100 questions per dataset |
| 3. End to end | Does the full pipeline answer correctly? | Accuracy, GPT-5.2 judge | Same questions as layer 2 |
| 4. Truthfulness | Does the system avoid unsupported answers? | Accuracy − hallucination (CRAG auto-eval) | CRAG Task 1, 400 questions |
| Latency | How fast is it on the device? | Time to first token and total time (p50, p90), ingest rate | All answers |
| Cost | What does a run cost? | Tokens and USD per answer | Local token proxy |
| Device health | Does the device stay within limits? | Peak temperature, time throttled, under-voltage, CPU, memory | 1 Hz sampler |

The official run uses a fixed stratified sample (`class-v1`, seed 7). Accuracy figures carry 95% bootstrap intervals with a finite-population correction. Formulas, judge configuration, sampling and deviations from the original papers are documented in [docs/METHOD.md](docs/METHOD.md).

## Datasets

| Dataset | Content | Licence |
|---|---|---|
| [ViDoRe V3 HR](https://huggingface.co/datasets/vidore/vidore_v3_hr) | 1,110 pages of EU labour-market reports, 318 English questions | CC BY 4.0 |
| [ViDoRe V3 Computer Science](https://huggingface.co/datasets/vidore/vidore_v3_computer_science) | 1,360 textbook pages, 215 English questions | CC BY 4.0 |
| [CRAG Task 1](https://github.com/facebookresearch/CRAG) | Public test set, five web pages per question | CC BY-NC 4.0 |

See [datasets/LICENSES.md](datasets/LICENSES.md). CRAG data may be used for non-commercial purposes only.

## Output

| File | Contents |
|---|---|
| `report.html` | Self-contained report |
| `summary.md` | Plain-text summary |
| `results.json` | All computed metrics |
| `answers.jsonl` | Every question, answer and timing |
| `judgments.jsonl` | Judge decisions |
| `health.csv` | Device samples, one per second |
| `proxy.jsonl` | Model calls with tokens and cost |
| `run.log` | Run log |

## Runtime and cost

An official run takes about 1.5 hours on a Raspberry Pi 4 and uses about 8 million tokens. With a flash-class model the system under test costs about USD 3.40, and judging costs about USD 1.90. Details are in [docs/GUIDE.md](docs/GUIDE.md).

## Documentation

- [docs/GUIDE.md](docs/GUIDE.md): step-by-step setup and run on a Raspberry Pi
- [docs/ADAPTER.md](docs/ADAPTER.md): HTTP interface and token proxy
- [docs/METHOD.md](docs/METHOD.md): metrics, judges, sampling and sources
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): common problems and fixes
- [SPEC.md](SPEC.md): design specification

## References

- Loison et al. *ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios.* arXiv:2601.08620, 2026.
- Yang et al. *CRAG: Comprehensive RAG Benchmark.* arXiv:2406.04744, 2024.

## Licence

The benchmark code is released under the MIT licence (see [LICENSE](LICENSE)). Datasets are subject to their own licences, listed above.

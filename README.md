# me4-rag-bench

`crb` is a benchmark for document question-answering (RAG) systems running on a Raspberry Pi. It evaluates a system through a small HTTP interface, scores it on open datasets that have published results, and produces an offline report that places the run within those published results.

Each run reports four layers of answer quality, latency measured on the device, token cost, and device health (temperature, throttling, power, CPU and memory) sampled once per second. Results are expressed as a rank and as a position relative to the median of the published systems on the same data, metric and judge.

## Features

- **Architecture-agnostic.** Any system that implements four HTTP endpoints can be evaluated, whether it uses text chunks, page images, tool-using agents, long context or a knowledge graph.
- **Comparable with the literature.** Datasets, metrics and judges follow ViDoRe V3 and CRAG, so scores can be read against the published tables.
- **Layered diagnosis.** Retrieval, generation with the correct pages, and the full pipeline are scored separately, and every question is attributed to the layer that failed.
- **Measured on the device.** Latency, ingest time and resource use are recorded on the Pi itself; token use is counted by a local proxy rather than self-reported.
- **Plug and play.** One install script, an interactive wizard, pre-flight checks, resumable runs and a self-contained HTML report.

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

### Step 3. Start the system under test on the Pi

The system under test is an HTTP server on the Pi (default `http://127.0.0.1:8000`) that implements the interface in [docs/ADAPTER.md](docs/ADAPTER.md). Start it in a way that keeps it running after you log out, for example in `tmux`:

```bash
tmux new -s adapter             # on the Pi
cd ~/me4-rag-bench && source .venv/bin/activate
python3 your_adapter.py         # your system; or `crb baseline` for the reference system
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

## Adapter interface

| Endpoint | Purpose |
|---|---|
| `GET /health` | Identify the system |
| `POST /ingest` | Index a collection of pages |
| `POST /retrieve` | Return the top-k page identifiers for a question |
| `POST /answer` | Answer a question; streaming (server-sent events or chunked text) or JSON |

`/answer` is called in three modes: `e2e` (full pipeline), `gold` (correct pages supplied, retrieval bypassed) and `web` (CRAG, five supplied web pages). Request and response formats, examples and guidance for common architectures are in [docs/ADAPTER.md](docs/ADAPTER.md).

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

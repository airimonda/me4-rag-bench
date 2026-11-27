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

- Raspberry Pi 4 (4 GB) or Raspberry Pi 5, 64-bit Raspberry Pi OS, Python 3.10 or later
- About 2.5 GB of free disk space
- An OpenRouter API key for the judges (`CRB_JUDGE_KEY` or `OPENROUTER_API_KEY`)

macOS and Linux laptops are supported for development; device health metrics are then partly unavailable.

## Installation

```bash
git clone https://github.com/airimonda/me4-rag-bench.git
cd me4-rag-bench
bash scripts/install_pi.sh
source .venv/bin/activate
crb fetch            # downloads and builds the datasets, about 1.2 GB on disk
```

## Usage

1. Start the system under test as an HTTP server (default `http://127.0.0.1:8000`) implementing the interface in [docs/ADAPTER.md](docs/ADAPTER.md). A reference implementation and a template are included:

   ```bash
   crb baseline                            # BM25 reference system
   python3 examples/adapter_template.py    # minimal skeleton to adapt
   ```

2. Verify the setup. Pre-flight checks make at most one model call.

   ```bash
   export OPENROUTER_API_KEY=...
   crb check
   ```

3. Run the benchmark.

   ```bash
   crb                       # interactive wizard
   crb run --official        # class-v1 sample, judged as in the papers
   crb run --smoke           # quick practice run, not comparable
   crb run --full            # every question in every dataset
   ```

4. Open the report. It is written to `runs/<student>/<date-time>/report.html`. To view it from another machine on the network:

   ```bash
   crb serve
   ```

Other commands: `crb score RUN_DIR` re-judges and re-scores a run, `crb report RUN_DIR` rebuilds the report, and `crb run --resume RUN_DIR` continues an interrupted run.

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

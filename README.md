# me4-rag-bench (`crb`)

A benchmark for your document question-answering (RAG) system. It runs on your own Raspberry Pi.

It sends open datasets to your system over HTTP, times every answer, checks the Pi's health once a second, and scores the answers with the judges used in the papers. At the end you get one `report.html` that shows where your system sits among published systems.

You are never compared with other students. You are compared with published results only. For each test the report shows your rank and whether you are above or below the median of the published systems.

## Quick start

You need a Raspberry Pi 4 (4 GB) or Pi 5 with 64-bit Raspberry Pi OS, about 2.5 GB free disk, and an OpenRouter key from the teaching team.

1. Install.

   ```bash
   git clone https://github.com/airimonda/me4-rag-bench.git
   cd me4-rag-bench
   bash scripts/install_pi.sh
   source .venv/bin/activate
   ```

2. Download the data (about 1.2 GB, one time).

   ```bash
   crb fetch
   ```

3. Start your system as an HTTP server on port 8000. No system yet? Try the reference one or the template:

   ```bash
   crb baseline                              # reference system
   python3 examples/adapter_template.py      # empty skeleton to copy
   ```

   The contract your server must follow is in [docs/ADAPTER.md](docs/ADAPTER.md).

4. Check that everything is ready. This costs nothing.

   ```bash
   export OPENROUTER_API_KEY=your-key-here
   crb check
   ```

5. Run it.

   ```bash
   crb
   ```

   `crb` asks a few questions, runs everything, and prints the path of `report.html`. To look at the report from your laptop, run `crb serve` on the Pi and open the link it prints. See [docs/GUIDE.md](docs/GUIDE.md).

Want to try it cheaply first? `crb run --smoke` uses tiny samples and a cheap judge. Its report is marked "not comparable".

## What is measured

Four layers of quality, then speed, cost and Pi health. Each one is scored on its own, so you can see which part of your system to fix.

| # | What | Question it answers | Data |
|---|---|---|---|
| 1 | Retrieval | Did your search find the right pages? (nDCG@10, Recall@10; no LLM) | ViDoRe V3, all English questions |
| 2 | Gold pages | Given the right pages, is the answer right? (judge accuracy) | ViDoRe V3, 100 questions per pack |
| 3 | End to end | Does the whole system answer correctly? (judge accuracy) | the same questions |
| 4 | Truthfulness | Does it say "I don't know" instead of making things up? | CRAG, 400 questions |
| - | Speed | Time to first token and total time, p50 / p90, ingest time | measured by `crb` on the Pi |
| - | Cost | Tokens and dollars per answer, counted by the token proxy | the proxy |
| - | Pi health | Temperature, throttling, under-voltage, CPU, memory, network | sampled every second |

A diagnosis then tells you, for every question, whether the miss was in retrieval, in generation, or in putting the two together. Details and formulas: [docs/METHOD.md](docs/METHOD.md).

## Datasets

| Pack | Content | Licence | Link |
|---|---|---|---|
| ViDoRe V3 HR | 1,110 pages of EU labour-market reports, 318 English questions | CC-BY-4.0 | https://huggingface.co/datasets/vidore/vidore_v3_hr |
| ViDoRe V3 Computer Science | 1,360 textbook pages, 215 English questions | CC-BY-4.0 | https://huggingface.co/datasets/vidore/vidore_v3_computer_science |
| CRAG Task 1 | 5 web pages per question, 400 sampled public-test questions | CC BY-NC 4.0 (non-commercial) | https://github.com/facebookresearch/CRAG |

Licence details: [datasets/LICENSES.md](datasets/LICENSES.md).

## What you get

Everything lands in `runs/<student>/<date-time>/`:

| File | What it is |
|---|---|
| `report.html` | The report. One file, works offline. |
| `summary.md` | The result as plain text, for slides. |
| `results.json` | Every number in the report. |
| `answers.jsonl` | Every question, your answer, timings. |
| `health.csv` | Pi health, one row per second. |
| `proxy.jsonl` | Every LLM call: tokens, cost, time. |
| `judgments.jsonl` | What each judge said. |
| `run.log` | What happened, in order. |

## Time and cost

An official run takes about 1.5 hours on a Pi 4 and uses about 8 million tokens. With a flash-class model that is about $3.40 for your system, plus about $1.90 for the judges. The class key pays the judges. See [docs/GUIDE.md](docs/GUIDE.md).

## More help

- [docs/GUIDE.md](docs/GUIDE.md): full walkthrough on the Pi
- [docs/ADAPTER.md](docs/ADAPTER.md): the HTTP contract and the token proxy
- [docs/METHOD.md](docs/METHOD.md): every metric and its source
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): when something goes wrong

## Credits and licences

- This tool: MIT (see `LICENSE`).
- ViDoRe V3: Loison et al., 2026 (arXiv 2601.08620), CC-BY-4.0.
- CRAG: Yang et al., 2024 (arXiv 2406.04744), CC BY-NC 4.0. Use is non-commercial only.
- The reference adapter is a floor to compare against, not a model answer.

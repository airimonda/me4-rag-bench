# Guide: running the benchmark on your Pi

Take it one step at a time. Nothing here can break your system. If a step fails, the message tells you the fix, and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) has more.

## What you need

- A Raspberry Pi 4 (4 GB) or Pi 5, 64-bit Raspberry Pi OS, on the network.
- About 2.5 GB of free disk.
- An OpenRouter key from the teaching team.
- A good power supply and a fan or heatsink. The run keeps the Pi busy for over an hour, and the report shows if it got too hot.
- Your RAG system, or the reference one (`crb baseline`).

## 1. Install

```bash
git clone https://github.com/airimonda/me4-rag-bench.git
cd me4-rag-bench
bash scripts/install_pi.sh
source .venv/bin/activate
```

The script is safe to run again. It checks Python (3.10 or newer), disk space, and creates `.venv`. Run `source .venv/bin/activate` in every new terminal.

## 2. Download the data

```bash
crb fetch
```

About 1.2 GB. It checks every file after downloading. Data goes to `~/crb-data`. To use another folder, run `export CRB_DATA=/path/to/folder` first. If it stops, run the same command again. It continues.

## 3. Start your adapter

Your system must answer four HTTP calls on `http://127.0.0.1:8000`. See [ADAPTER.md](ADAPTER.md). Start it in a second terminal (or a `tmux` window) and leave it running.

No adapter yet? Two choices:

```bash
crb baseline                          # the reference system
python3 examples/adapter_template.py  # a skeleton that returns placeholder answers
```

Your LLM client should use `http://127.0.0.1:8790/v1` as its base URL. That is the token proxy. It lets `crb` count your tokens and cost. Without it the report says "cost not measured". Nothing else changes.

## 4. Set your key

```bash
export OPENROUTER_API_KEY=your-key-here
```

If you were given a separate judge key, also set `CRB_JUDGE_KEY`. `crb` never writes keys to its output files.

To make the key available to runs started from your laptop over SSH (where shell profiles are often skipped), also store it in `~/.config/crb/env` on the Pi:

```bash
mkdir -p ~/.config/crb
echo 'OPENROUTER_API_KEY=your-key-here' > ~/.config/crb/env
chmod 600 ~/.config/crb/env
```

## 5. Check

```bash
crb check
```

It looks at the data, your adapter, the endpoints, disk, `vcgencmd`, the judge key and the proxy port. Each problem is one line with the fix. It does not spend money on the judges.

## 6. Try a small run

```bash
crb run --smoke
```

Retrieval plus a few questions per test, a cheap judge. It takes a few minutes. The report says "not comparable", which is right. Use it to find bugs before the long run.

## 7. The official run

```bash
crb
```

`crb` asks for your student number, your adapter URL, and the mode. Choose official. Or skip the questions:

```bash
crb run --official --student 12
```

What happens, in order:

1. Pre-flight check.
2. Ingest: your adapter reads each ViDoRe pack. Timed.
3. Test 1, retrieval: all English ViDoRe questions.
4. Test 2, gold pages.
5. Test 3, end to end.
6. Test 4, CRAG.
7. Judging, scoring, report.

Questions go one at a time, so one answer never slows another. The first 3 questions after each ingest are scored but left out of the speed numbers (a system is slow when it is cold).

Time and cost for the official run:

| | Value |
|---|---|
| Time on a Pi 4 | about 1.5 hours |
| Tokens | about 8 million |
| Your system (flash-class model such as minimax-m3.1-flash) | about $3.40 |
| Judges | about $1.90, paid by the class key |

A bigger model costs more. `crb run --full` runs every question and costs about three times more. Use it only if you want smaller error bars.

You can press Ctrl-C at any time. Everything done so far is kept.

## 8. Resume a stopped run

```bash
crb run --resume runs/12/20261127-101500
```

Use the run folder that was printed. Questions already answered are skipped.

## 9. Read the report

The easiest way is to start the run from your laptop. With `crb` installed on the laptop (see the README, step 2):

```bash
crb remote pi@raspberrypi.local --official --student 12
```

This starts the run on the Pi, shows its progress, and when it finishes copies the run to `~/crb-runs/` on the laptop and opens `report.html` in your browser. If you close the laptop, the run continues on the Pi; collect it later with:

```bash
crb pull pi@raspberrypi.local --wait
```

If you ran on the Pi directly, `crb pull pi@raspberrypi.local` on the laptop copies the newest run and opens it.

Another option is `crb serve` on the Pi, which prints a link like `http://192.168.1.42:8302`. Some laptops cannot open addresses on the local network (for example because of macOS's Local Network permission); `crb pull` works in that case.

In the report:

- For each test you see your number, your rank among the published systems, and whether you are above or below their median.
- The bar next to a score is the 95% error bar (we ran a sample, not every question).
- Speed shows p50 and p90. p90 is the time 9 in 10 answers beat.
- The diagnosis tells you if misses came from retrieval, generation, or the two together.
- Pi health shows peak temperature, seconds throttled and under-voltage.
- The worst 10 answers are listed so you know what to fix.

## 10. Redo scoring or the report without re-running

```bash
crb score runs/12/20261127-101500      # judge again (cached judgments are free)
crb report runs/12/20261127-101500     # rebuild the report only
```

## Good habits

- Do not tune your system on the sample questions. Run once, read the report, then improve and run again.
- Run it when the Pi is not doing anything else.
- Keep the Pi cool. If the report shows throttling, speed numbers are lower than your system really is.

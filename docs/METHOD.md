# Method

Every number in the report, where it comes from, and where this benchmark differs from the papers.

## The idea

Your system is split into layers so a bad score can be traced to its cause.

1. Retrieval: question to ranked pages. No LLM.
2. Gold pages: the right pages are handed over, retrieval is off. Errors left are the generator's.
3. End to end: the whole system, as on demo day.
4. Truthfulness (CRAG): does it avoid inventing answers?

Speed, cost and Pi health are measured next to these, on every run. They are never merged into the quality scores.

You are compared with published systems only. The report inserts your number into the published table of each test. It shows your rank (1 = best), and whether you are above or below the median of the published rows (your own row is not in the median).

## Data

| Pack | Source | Used |
|---|---|---|
| `vidore_v3_hr` | ViDoRe V3, Human Resources | 1,110 pages, 318 English questions |
| `vidore_v3_computer_science` | ViDoRe V3, Computer Science | 1,360 pages, 215 English questions |
| `crag_task1` | CRAG Task 1 and 2 dev v4 | public test (`split == 1`), 1,335 questions, 5 web pages each |

Only English questions are used. Pages are counted per page (ViDoRe) or per web page (CRAG).

## Sampling

- Seed 7 for every sample. Sample name: `class-v1`. Every student gets the same questions.
- Retrieval uses every English ViDoRe question. It needs no LLM, so there is nothing to save.
- Gold pages and end to end: 100 questions per ViDoRe pack. CRAG: 400 questions.
- Stratified by question type, proportional allocation, largest-remainder rounding, so the sample keeps the mix of types of the full set.
- Drawn once. Never redraw it and never tune on it.
- `--full` runs every question. `--smoke` runs 10 questions per test with a cheap judge, and the report says "not comparable".

## Retrieval (test 1)

For each question, `/retrieve` returns up to 10 page ids. They are compared with the graded relevance labels (`qrels`) from the dataset.

- nDCG@10: gain `rel` at rank `i` is discounted by `log2(i + 1)`. DCG@10 = sum of `rel_i / log2(i + 1)`. nDCG@10 = DCG@10 divided by the DCG of the ideal ranking. Averaged over questions, times 100.
- Recall@10: share of the question's gold pages that are in your top 10. Averaged, times 100.

Both are exact, with no error bar, since every question is used.

## Judges

### ViDoRe (tests 2 and 3)

- Judge: `openai/gpt-5.2`, reasoning effort medium, temperature 0.
- Prompt: an expert judge compares your answer with the gold answer and returns `Correct`, `Partially Correct` or `Incorrect` with a short explanation. It follows the ViDoRe V3 judge prompt (paper figure 24).
- Accuracy = share judged `Correct`. "Partially correct" counts as not correct.
- Accuracy is also split by content type (text, table, chart, ...).

### CRAG (test 4)

CRAG auto-evaluation, in two steps.

1. Rules first. Normalised exact match with the gold answer or an alternative counts as accurate. An empty reply, or one that contains "i don't know", is missing. "invalid question" counts as accurate only for false-premise questions.
2. Otherwise two LLM judges decide accurate or incorrect: `openai/gpt-3.5-turbo-0613` and `meta-llama/llama-3.1-70b-instruct`. The question score is the mean of the two judgments (+1 accurate, -1 incorrect).

Scores per question: +1 accurate, 0 missing, -1 hallucinated.

- Truthfulness = mean score x 100 = accuracy - hallucination.
- Accuracy, hallucination rate and missing rate are shown too. A made-up answer is worse than "I don't know".

Judge results are cached by judge model, prompt hash, question and answer. `crb score` can run again for free.

## Diagnosis

For each question in the ViDoRe answer sample:

| Result | Rule |
|---|---|
| correct | test 3 judged Correct |
| retrieval miss | no gold page in your top 10 (test 1) |
| generation miss | not correct in test 2 (gold pages given) |
| integration miss | none of the above: retrieval and generation each work, the combination fails |

The rules are applied in this order. They are a guide to where to look first, not a proof.

## Confidence intervals

Accuracy is measured on a sample, so it has an error bar.

- 95% percentile bootstrap, 2,000 resamples, seed 7.
- Finite-population correction: when the sample (`n`) comes from a pack of `N` questions, the half-width is multiplied by `sqrt((N - n) / (N - 1))`. When `n == N` the bar is 0 (exact).
- Read the bar before the rank. If the published number is inside your bar, you are close to it. A close result needs `--full` to settle.

## Ranks and the median

For each test, your number is placed into the published table (`crb/literature.py`). Rank 1 is best. The median uses the published rows only. The side is "above", "below" or "at" the median. There are no pass/fail verdicts: the report shows where you stand, nothing more.

## Speed

All times are taken with `time.perf_counter()` by `crb` on the Pi, not reported by your system.

| Metric | Starts | Stops |
|---|---|---|
| First token | `/answer` sent | first byte of answer text |
| Total | `/answer` sent | last byte |
| Retrieve | `/retrieve` sent | page list received |
| Ingest | `/ingest` sent | reply received (seconds, pages per minute) |
| Cold start | first question after ingest | its last byte |

- One question at a time.
- The first 3 questions after each ingest are marked `warmup` and left out of p50 and p90. Cold start is reported separately.
- p50 is the middle value, p90 is the time 9 in 10 answers beat, max is the slowest. Never the mean, since it hides the slow tail.
- A non-streaming server has first token = total.
- Gold-page answers are not timed for speed, since they skip your search.
- Rating bands for time to first token (p90): up to 1 s instant, up to 3 s good, up to 10 s acceptable, above that too slow for a live demo. These come from classic usability limits. Neither paper publishes latency.

## Cost

The token proxy sits between your system and OpenRouter. For each call it records the model, prompt and completion tokens, and the cost (OpenRouter's `usage.cost`). Cost per answer is the total for your system divided by the answers. Judge cost is shown apart. If you do not use the proxy, cost is "not measured".

## Pi health

Sampled every second for the whole run (`health.csv`). Sources:

| Column | Source |
|---|---|
| `temp_c`, `arm_mhz`, `core_v` | `vcgencmd measure_temp`, `measure_clock arm`, `measure_volts core` (or the kernel's thermal and cpufreq files) |
| `throttled_hex` and four flags | `vcgencmd get_throttled`, bits 0-3 |
| `cpu_pct`, `cpu_max_core_pct` | `/proc/stat` differences between samples (all cores, and the busiest core) |
| `load1`, `mem_avail_mb`, `swap_used_mb` | `/proc/loadavg`, `/proc/meminfo` |
| `adapter_rss_mb` | memory of the process that listens on your adapter port |
| `disk_free_mb` | free space where the data lives |
| `net_rtt_ms` | TCP connect time to `openrouter.ai:443`, every 10 s |
| `phase` | what the run was doing |

The four flags: under-voltage now, frequency capped now, throttled now, soft temperature limit now. "Seconds throttled" counts seconds where any of the last three was set. "Under-voltage" is true if it was seen at any second. Missing sources leave empty cells, and the report says "not available" (normal on a laptop). The sampler uses well under 1% of one core.

## Deviations from the papers

Be careful when you compare. The numbers are close to the papers' numbers, not identical.

- **CRAG judge prompt.** The CRAG paper did not release its judge prompt. We use the rule step plus two LLM judges with a prompt written for this benchmark, in the spirit of the paper.
- **Judge versions.** The exact 2024 versions of the CRAG judges are not available. `gpt-3.5-turbo-0613` and `llama-3.1-70b-instruct` are used as the closest available. Expect small differences.
- **ViDoRe retriever names.** Some retriever names in the published table of the ViDoRe paper (Table 3) were inferred from the text, since the table does not always name the retriever. A name may be wrong. The numbers are copied as printed.
- **English only.** ViDoRe has translated queries too. Only English queries are used, so the numbers match the English rows. Paper Table 3 does not say which languages it used.
- **Sample, not full set.** Tests 2, 3 and 4 use a fixed sample. The error bar shows what this costs.
- **CRAG pages.** Scripts, styles and comments are removed from the HTML to keep the pack near 200 MB. CRAG Task 1 only: no knowledge-graph APIs, so the Task 2 and 3 scores in the paper are not comparable.
- **Public test only.** The KDD Cup 2024 winner (hidden test set) and human-graded industry systems are not comparable.
- **Newer retrievers.** Retrievers newer than the paper may beat the tables. We use the paper so the protocol matches.
- **Dataset quirks.** ViDoRe questions are partly LLM-generated, then human-checked. CRAG is web QA from 2024.

## Sources

- Loison et al., "ViDoRe V3", 2026, arXiv 2601.08620.
- Yang et al., "CRAG - Comprehensive RAG Benchmark", 2024, arXiv 2406.04744.
- Class explainer: `docs/design/explainer.html`.

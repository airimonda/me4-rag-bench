# Dataset licences

`crb fetch` downloads these datasets from their original sources. They are not part of this repository.

| Dataset | Licence | Where from |
|---|---|---|
| ViDoRe V3, Human Resources | CC-BY-4.0 | https://huggingface.co/datasets/vidore/vidore_v3_hr |
| ViDoRe V3, Computer Science | CC-BY-4.0 | https://huggingface.co/datasets/vidore/vidore_v3_computer_science |
| CRAG (Comprehensive RAG Benchmark) | CC BY-NC 4.0 | https://github.com/facebookresearch/CRAG |
| This tool (`crb`) | MIT | see `LICENSE` |

## ViDoRe V3

Licence: Creative Commons Attribution 4.0 (CC-BY-4.0). You may share and adapt it, also commercially, if you give credit.

Credit: Loison et al., "ViDoRe V3", 2026 (arXiv 2601.08620).

The packs keep English questions only, with the page images, the markdown text and the graded relevance labels from the dataset.

## CRAG

Licence: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0). **Non-commercial use only.** Use this benchmark for course work and research, not for a product. Use the official release from `facebookresearch/CRAG`, not the Apache-licensed mirrors.

Credit: Yang et al., "CRAG - Comprehensive RAG Benchmark", 2024 (arXiv 2406.04744).

The packs keep the public test split (`split == 1`) of `crag_task_1_and_2_dev_v4`. The HTML pages are stored with scripts, styles and comments removed to save space. No other change is made.

## This tool

MIT licence. The code in this repository is yours to use and change. The datasets above keep their own licences.

"""Published results the report places a run into. Values are copied verbatim from the papers.

ViDoRe V3: Loison et al., arXiv 2601.08620, 2026. CRAG: Yang et al., arXiv 2406.04744, 2024.
Column order of the two ViDoRe values is (HR = vidore_v3_hr, CS = vidore_v3_computer_science).
"""
from __future__ import annotations

import copy

HR = "vidore_v3_hr"
CS = "vidore_v3_computer_science"
CRAG = "crag_task1"

VIDORE_PAPER = "ViDoRe V3 (Loison et al., arXiv 2601.08620, 2026)"
CRAG_PAPER = "CRAG (Yang et al., arXiv 2406.04744, 2024)"

# (name, kind, HR, CS)
_T9 = [
    ("Jina-v4 (text)", "text", 58.8, 67.3),
    ("Qwen3-8B", "text", 52.3, 73.5),
    ("LFM2-350M", "text", 53.2, 70.6),
    ("Mxbai Edge 32M", "text", 52.7, 68.0),
    ("BM25S", "text", 49.6, 64.7),
    ("Qwen3-0.6B", "text", 46.2, 70.5),
    ("GTE-ModernColBERT", "text", 46.2, 63.6),
    ("BGE-M3", "text", 45.3, 63.6),
    ("ColEmbed-3B-v2", "visual", 65.4, 78.6),
    ("ColEmbed-3B", "visual", 64.9, 77.8),
    ("ColEmbed-1B", "visual", 64.5, 75.5),
    ("Jina-v4 (visual)", "visual", 64.6, 74.2),
    ("ColNomic-7B", "visual", 62.9, 78.2),
    ("ColNomic-3B", "visual", 62.6, 75.5),
    ("ColQwen2.5", "visual", 59.2, 75.2),
    ("Nomic-7B", "visual", 55.9, 70.9),
    ("ColQwen2", "visual", 54.7, 73.5),
    ("ColPali", "visual", 53.3, 72.5),
    ("Nomic-3B", "visual", 51.9, 62.1),
    ("ColModernVBERT", "visual", 47.0, 59.7),
    ("ColSmol-256M", "visual", 46.0, 57.4),
]

# Table 3, gold-page (oracle) rows
_T3_GOLD = [
    ("Gemini 3 Pro · text pages", 72.3, 80.9),
    ("Gemini 3 Pro · page images", 71.1, 86.5),
    ("Gemini 3 Pro · hybrid", 70.4, 86.0),
]

_INFERRED = "Retriever inferred from the table grouping (ColEmbed), not stated on the row."

# Table 3, end-to-end rows: (name, HR, CS, note)
_T3_E2E = [
    ("Gemini 3 Pro · Jina-v4 + rerank, text", 63.2, 80.9, None),
    ("Gemini 3 Pro · hybrid + rerank", 64.8, 85.1, None),
    ("Gemini 3 Pro · ColEmbed, text", 62.9, 82.3, None),
    ("Kimi K2 · ColEmbed, text", 55.7, 81.4, None),
    ("Gemini 3 Pro · ColEmbed, image", 65.4, 83.3, _INFERRED),
    ("Gemini 3 Flash · ColEmbed, image", 63.8, 80.9, _INFERRED),
    ("GPT-5.2 · ColEmbed, image", 66.0, 86.5, _INFERRED),
    ("Qwen3-VL-235B · ColEmbed, image", 60.7, 86.0, _INFERRED),
]

# Table 11, Task 1 truthfulness %
_T11 = [
    ("GPT-4 Turbo", 7.7),
    ("Llama 3 70B", 4.5),
    ("FLAN-T5-XXL 11B", -9.0),
    ("Mixtral-8x7B", -10.8),
    ("Llama 3 8B", -17.1),
    ("Llama 2 70B", -31.7),
    ("Falcon 40B", -33.6),
    ("Llama 2 7B", -66.7),
]


PUBLISHED: dict = {
    "retrieval": {
        "title": "retrieval nDCG@10, English queries",
        "metric": "nDCG@10",
        "source": f"{VIDORE_PAPER}, Table 9",
        "rows": [{"name": n, "kind": k, HR: h, CS: c} for n, k, h, c in _T9],
    },
    "gold": {
        "title": "% correct with the gold pages",
        "metric": "% correct",
        "source": f"{VIDORE_PAPER}, Table 3, gold-page rows",
        "rows": [{"name": n, "kind": "oracle", HR: h, CS: c} for n, h, c in _T3_GOLD],
    },
    "e2e": {
        "title": "end-to-end % correct",
        "metric": "% correct",
        "source": f"{VIDORE_PAPER}, Table 3",
        "rows": [dict({"name": n, "kind": "e2e", HR: h, CS: c}, **({"note": note} if note else {}))
                 for n, h, c, note in _T3_E2E],
    },
    "crag": {
        "title": "truthfulness (accuracy − hallucination)",
        "metric": "truthfulness %",
        "source": f"{CRAG_PAPER}, Table 11, Task 1",
        "rows": [{"name": n, "kind": "llm", CRAG: v} for n, v in _T11],
    },
}

# layer -> list of packs with a published column
LAYERS = ("retrieval", "gold", "e2e", "crag")


def published() -> dict:
    """A deep copy that is safe to put into results.json."""
    return copy.deepcopy(PUBLISHED)


def rows(layer: str, pack: str) -> list[dict]:
    """Published rows of `layer` that have a value for `pack`: [{"name","value","kind","note"?}]."""
    t = PUBLISHED.get(layer)
    if not t:
        return []
    out = []
    for r in t["rows"]:
        if r.get(pack) is not None:
            d = {"name": r["name"], "value": r[pack], "kind": r.get("kind")}
            if r.get("note"):
                d["note"] = r["note"]
            out.append(d)
    return out

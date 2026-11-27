"""Cost and token counts from the token proxy log (proxy.jsonl)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_IN = ("prompt_tokens", "tokens_in", "input_tokens")
_OUT = ("completion_tokens", "tokens_out", "output_tokens")
_COST = ("cost", "cost_usd", "usd")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = json.loads(line)
                except ValueError:
                    continue
                if isinstance(v, dict):
                    rows.append(v)
    except OSError:
        pass
    return rows


def _num(row: Mapping[str, Any], keys) -> float | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    usage = row.get("usage")
    if isinstance(usage, Mapping):
        for k in keys:
            v = usage.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def _cut(row: Mapping[str, Any]) -> bool:
    if row.get("context_cap_hit") or row.get("context_cap"):
        return True
    if str(row.get("finish_reason", "")).lower() == "length":
        return True
    err = str(row.get("error") or "").lower()
    return "context_length" in err or "maximum context" in err


def summarize(proxy_rows: list[dict], n_answers: int, judge_usd: float | None) -> dict:
    """metrics.cost. `measured` is False when the proxy saw no call (student did not use it)."""
    out: dict[str, Any] = {"measured": False, "system_usd": None, "judge_usd": judge_usd,
                           "tokens_in": None, "tokens_out": None, "per_answer_usd": None,
                           "context_cap_hits": None, "calls": 0, "answers": n_answers}
    # records without any usage (non-LLM paths, failed calls) are not model calls
    proxy_rows = [r for r in proxy_rows
                  if any(v is not None for v in (_num(r, _IN), _num(r, _OUT), _num(r, _COST)))]
    if not proxy_rows:
        return out
    tin = tout = 0.0
    cost = 0.0
    have_cost = False
    partial = False
    cuts = 0
    for r in proxy_rows:
        a, b, c = _num(r, _IN), _num(r, _OUT), _num(r, _COST)
        tin += a or 0
        tout += b or 0
        if c is None:
            partial = True
        else:
            have_cost = True
            cost += c
        if _cut(r):
            cuts += 1
    out.update({"measured": True, "calls": len(proxy_rows), "tokens_in": int(tin), "tokens_out": int(tout),
                "system_usd": round(cost, 6) if have_cost else None,
                "context_cap_hits": cuts, "cost_partial": bool(have_cost and partial)})
    if have_cost and n_answers:
        out["per_answer_usd"] = round(cost / n_answers, 6)
    return out

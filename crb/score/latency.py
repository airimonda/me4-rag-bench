"""Latency percentiles from answers.jsonl and Pi health numbers from health.csv."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .stats import median, percentile

HIST_BIN_MS = 500
HIST_BINS = 20
TIMELINE_MAX_POINTS = 2000
ANSWER_AXES = (2, 3, 4)
PHASE_ORDER = ("preflight", "ingest", "retrieval", "gold", "e2e", "crag", "judging")


def _ms(row: Mapping[str, Any], key: str) -> float | None:
    lm = row.get("latency_ms")
    if not isinstance(lm, Mapping):
        return None
    v = lm.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _dist(xs: Sequence[float]) -> dict:
    if not xs:
        return {"p50": None, "p90": None, "max": None, "n": 0}
    return {"p50": round(percentile(xs, 50)), "p90": round(percentile(xs, 90)),
            "max": round(max(xs)), "n": len(xs)}


def summarize(answers: Sequence[Mapping[str, Any]], session: Mapping[str, Any] | None,
              health_rows: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """metrics.latency. Warm-up questions and failed answers are left out of the percentiles."""
    ok = [a for a in answers if not a.get("warmup") and not a.get("error")]
    ans = [a for a in ok if a.get("axis") in ANSWER_AXES]
    ft = [v for v in (_ms(a, "first_token") for a in ans) if v is not None]
    tot = [v for v in (_ms(a, "total") for a in ans) if v is not None]
    ret = [v for v in (_ms(a, "retrieve") for a in ok) if v is not None]

    bins = [0] * HIST_BINS
    for v in ft:
        bins[min(HIST_BINS - 1, int(v // HIST_BIN_MS))] += 1

    cold = None
    warm = [a for a in answers if a.get("warmup") and isinstance(a.get("t"), (int, float))]
    if warm:
        first = min(warm, key=lambda a: a["t"])
        cold = _ms(first, "first_token") or _ms(first, "total") or _ms(first, "retrieve")
        cold = round(cold) if cold is not None else None

    out = {"first_token": _dist(ft), "total": _dist(tot), "retrieve": _dist(ret),
           "ingest": _ingest(session, health_rows), "cold_start_ms": cold,
           "hist_first_token_500ms": bins,
           "errors": sum(1 for a in answers if a.get("error") and a.get("axis") in ANSWER_AXES)}
    return out


def _ingest(session: Mapping[str, Any] | None, health_rows) -> dict:
    out: dict[str, dict] = {}
    ing = (session or {}).get("ingest")
    if isinstance(ing, Mapping):
        for pack, v in ing.items():
            if not isinstance(v, Mapping):
                continue
            sec = v.get("seconds", v.get("elapsed_s", v.get("s")))
            pages = v.get("pages", v.get("n_pages", v.get("count")))
            if v.get("ok") is False:
                out[str(pack)] = {"seconds": None, "pages": None, "ok": False, "error": v.get("error")}
            elif isinstance(sec, (int, float)):
                out[str(pack)] = {"seconds": round(float(sec), 1),
                                  "pages": int(pages) if isinstance(pages, (int, float)) else None}
    return out


# ------------------------------------------------------------------ health

def _f(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _flag(v: Any) -> bool | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == "":
        return None
    return s in ("1", "true", "yes", "y", "t")


def read_health(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = _f(r.get("t"))
                if t is None:
                    continue
                r["_t"] = t
                rows.append(r)
    except OSError:
        pass
    rows.sort(key=lambda r: r["_t"])
    return rows


def _r(x: float | None, nd: int = 1) -> float | None:
    return None if x is None else round(x, nd)


def _downsample(rows: Sequence[Mapping[str, Any]], max_points: int) -> dict:
    step = max(1, math.ceil(len(rows) / max_points))
    t: list[float] = []
    temp: list[float | None] = []
    cpu: list[float | None] = []
    mhz: list[float | None] = []
    for i in range(0, len(rows), step):
        chunk = rows[i:i + step]

        def vals(col: str) -> list[float]:
            return [v for v in (_f(r.get(col)) for r in chunk) if v is not None]
        tv, cv, mv = vals("temp_c"), vals("cpu_pct"), vals("arm_mhz")
        t.append(round(chunk[0]["_t"], 1))
        temp.append(_r(max(tv)) if tv else None)           # keep the peaks
        cpu.append(_r(sum(cv) / len(cv)) if cv else None)
        mhz.append(_r(min(mv), 0) if mv else None)         # keep the dips
    return {"t": t, "temp": temp, "cpu": cpu, "mhz": mhz}


def summarize_health(rows: Sequence[Mapping[str, Any]], max_points: int = TIMELINE_MAX_POINTS) -> dict:
    """metrics.health from parsed health.csv rows. {"available": False} when there is nothing to show."""
    if not rows:
        return {"available": False, "samples": 0}

    def col(name: str) -> list[float]:
        return [v for v in (_f(r.get(name)) for r in rows) if v is not None]

    temps, cpu, core = col("temp_c"), col("cpu_pct"), col("cpu_max_core_pct")
    mem, mhz, rtt = col("mem_avail_mb"), col("arm_mhz"), col("net_rtt_ms")
    swap, rss, disk = col("swap_used_mb"), col("adapter_rss_mb"), col("disk_free_mb")

    ts = [r["_t"] for r in rows]
    dts = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    dt = median(dts) or 1.0
    flagged = [r for r in rows if _flag(r.get("throttled_now")) is not None
               or _flag(r.get("freq_capped_now")) is not None or _flag(r.get("soft_temp_now")) is not None
               or _flag(r.get("under_voltage_now")) is not None]

    def is_thr(r: Mapping[str, Any]) -> bool:
        return any(_flag(r.get(k)) for k in ("throttled_now", "freq_capped_now", "soft_temp_now"))

    thr_rows = [r for r in flagged if is_thr(r)]
    uv = any(_flag(r.get("under_voltage_now")) for r in flagged) if flagged else None

    # throttle bands and per-phase seconds from full-resolution samples
    bands: list[list[float]] = []
    by_phase: dict[str, float] = {}
    cur: list[float] | None = None
    for r in rows:
        if is_thr(r):
            ph = (r.get("phase") or "").strip() or "unknown"
            by_phase[ph] = by_phase.get(ph, 0.0) + dt
            if cur is not None and r["_t"] - cur[1] <= 2.5 * dt:
                cur[1] = r["_t"] + dt
            else:
                cur = [r["_t"], r["_t"] + dt]
                bands.append(cur)
        else:
            cur = None
    phases: list[dict] = []
    for r in rows:
        ph = (r.get("phase") or "").strip()
        if not ph:
            continue
        if phases and phases[-1]["phase"] == ph:
            phases[-1]["end"] = round(r["_t"] + dt, 1)
        else:
            if phases:
                phases[-1]["end"] = round(r["_t"], 1)
            phases.append({"phase": ph, "start": round(r["_t"], 1), "end": round(r["_t"] + dt, 1)})

    duration = (ts[-1] - ts[0] + dt) if ts else 0.0
    has_pi = bool(temps or flagged)
    return {
        "available": has_pi,
        "samples": len(rows),
        "duration_s": round(duration, 1),
        "temp_max": _r(max(temps)) if temps else None,
        "throttled_share": round(len(thr_rows) / len(flagged), 4) if flagged else None,
        "throttled_seconds": round(len(thr_rows) * dt) if flagged else None,
        "throttled_by_phase": {k: round(v) for k, v in by_phase.items()},
        "under_voltage": uv,
        "cpu_p95": _r(percentile(cpu, 95)),
        "busiest_core_p95": _r(percentile(core, 95)),
        "mem_avail_min_mb": _r(min(mem)) if mem else None,
        "arm_mhz_min": _r(min(mhz), 0) if mhz else None,
        "arm_mhz_max": _r(max(mhz), 0) if mhz else None,
        "swap_max_mb": _r(max(swap)) if swap else None,
        "adapter_rss_max_mb": _r(max(rss)) if rss else None,
        "disk_free_min_mb": _r(min(disk)) if disk else None,
        "net_rtt_p50": _r(percentile(rtt, 50)),
        "net_rtt_p90": _r(percentile(rtt, 90)),
        "phases": phases,
        "throttle_bands": [[round(a, 1), round(b, 1)] for a, b in bands],
        "timeline": _downsample(rows, max_points),
    }

"""Pre-flight: everything that can go wrong before a long run, one plain-English line each with the fix.

`run_checks` returns a list of (level, message, fix) where level is "ok", "warn" or "fail".
Only "fail" stops a run. Modules by other owners are imported lazily and a missing one never crashes.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

from . import adapter, config

OK, WARN, FAIL = "ok", "warn", "fail"
Check = tuple  # (level, message, fix)

PROBE_QUESTION = "What colour is the sky?"
PROBE_PAGE = "The sky on a clear day is blue."


def _check_python() -> Check:
    v = sys.version_info
    if v >= (3, 10):
        return (OK, f"Python {v.major}.{v.minor}.{v.micro}", "")
    return (FAIL, f"Python {v.major}.{v.minor} is too old", "Install Python 3.10 or newer")


def _check_data(plan, deep: bool) -> list[Check]:
    out: list[Check] = []
    try:
        packs = importlib.import_module("crb.data.packs")
        root = packs.data_dir()
    except Exception as exc:  # noqa: BLE001
        return [(FAIL, f"the data module is missing ({exc})", "Reinstall crb: pip install -e .")]
    if not Path(root).is_dir():
        return [(FAIL, f"no data folder at {root}", "Run `crb fetch` to download the datasets (or `crb fetch --dev` for a tiny offline test)")]
    verify = getattr(packs, "verify", None)
    for pack in plan.packs:
        if verify is None:
            ok = (Path(root) / "packs" / pack / "manifest.json").exists()
            problems = [] if ok else [f"pack '{pack}' is not downloaded"]
        else:
            try:
                problems = verify(pack, quick=not deep)
            except Exception as exc:  # noqa: BLE001
                problems = [f"pack '{pack}' could not be checked ({exc})"]
        if problems:
            more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
            out.append((FAIL, problems[0] + more, f"Run `crb fetch --pack {pack}` to repair it"))
        else:
            out.append((OK, f"data pack {pack} is present" + (" and its checksums match" if deep else ""), ""))
    return out


def _check_adapter(url: str, plan) -> list[Check]:
    out: list[Check] = []
    h = adapter.health(url)
    if not h.ok:
        return [(FAIL, h.error or f"the adapter at {url} does not answer /health",
                 "Start your adapter first (or `crb baseline` for the reference one), then check the URL")]
    name = h.data.get("name") or "adapter"
    out.append((OK, f"adapter '{name}' {h.data.get('version') or ''} is up at {url}".replace("  ", " "), ""))
    pack = next((p for p in plan.packs if not config_is_crag(p)), None) or next(iter(plan.packs), "probe")
    r = adapter.retrieve(url, pack, PROBE_QUESTION, 3)
    if r.pages is not None:
        out.append((OK, "/retrieve answers a probe question", ""))
    elif r.status is None:
        out.append((FAIL, r.error or "/retrieve did not answer", "Make sure the adapter implements POST /retrieve (see docs/ADAPTER.md)"))
    else:
        out.append((WARN, (r.error or "/retrieve gave no pages") + " (before ingest this can be normal)",
                    "If your system has no retriever, axis 1 will be empty; otherwise check POST /retrieve"))
    probe = {"collection": "probe", "question": PROBE_QUESTION, "mode": "gold",
             "context": [{"page_id": "probe__p1", "markdown": PROBE_PAGE, "image_path": None}]}
    a = adapter.answer(url, probe, timeout=config.ANSWER_TIMEOUT)
    if a.error:
        out.append((FAIL, a.error, "Read the message above (an LLM key problem is the usual cause); POST /answer must work in gold mode with the given context (see docs/ADAPTER.md)"))
    elif not (a.answer or "").strip():
        out.append((WARN, "/answer returned an empty answer to the probe question", "Check that your adapter reads the 'context' field in gold mode"))
    else:
        out.append((OK, f"/answer works (probe answered in {a.total_ms / 1000:.1f} s)", ""))
    return out


def config_is_crag(pack: str) -> bool:
    return "crag" in pack


def _check_disk(plan) -> Check:
    try:
        root = importlib.import_module("crb.data.packs").data_dir()
        probe = Path(root) if Path(root).exists() else Path.home()
    except Exception:  # noqa: BLE001
        probe = Path.home()
    free = [shutil.disk_usage(probe).free / 2**30]
    runs = config.runs_root()
    anchor = runs
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    free.append(shutil.disk_usage(anchor).free / 2**30)
    g = min(free)
    if g < 0.3:
        return (FAIL, f"only {g:.1f} GB of disk space free", "Free at least 1 GB (a run itself needs little, but your adapter's index may need more)")
    if g < 2:
        return (WARN, f"{g:.1f} GB of disk space free", "Free some space; your adapter's index for ~1,100 pages may need 1 GB or more")
    return (OK, f"{g:.0f} GB of disk space free", "")


def _check_vcgencmd() -> Check:
    if shutil.which("vcgencmd"):
        return (OK, "vcgencmd found: Pi temperature and throttling will be recorded", "")
    return (WARN, "vcgencmd not found: temperature and throttling will not be recorded",
            "Fine on a laptop. On the Pi: sudo apt install libraspberrypi-bin (or raspberrypi-utils)")


def _check_judge(no_judge: bool) -> Check:
    if no_judge:
        return (OK, "no judge requested: answers are saved, scoring that needs a judge is skipped", "")
    if config.judge_key():
        return (OK, "judge key found (CRB_JUDGE_KEY / OPENROUTER_API_KEY)", "")
    return (FAIL, "no judge API key found",
            "export CRB_JUDGE_KEY=sk-or-...  (an OpenRouter key), or run with --no-judge and score later with `crb score`")


def _start_probe_proxy():
    """Bring the token proxy up for the probe (a student's adapter usually calls its LLM through it).
    Returns (proxy | None, check). The proxy is stopped again before the real run starts."""
    import tempfile

    from .proxy import TokenProxy
    from .state import State
    port = config.proxy_port()
    try:
        proxy = TokenProxy(State(), Path(tempfile.gettempdir()) / "crb-preflight-proxy.jsonl", port=port).start()
        return proxy, (OK, f"token proxy port {port} is free", "")
    except OSError:
        return None, (FAIL, f"port {port} (the token proxy) is already in use",
                      "Stop the other crb run, or set CRB_PROXY_PORT to a free port")


def run_checks(cfg, plan, deep: bool = False) -> list[Check]:
    """All checks, in the order a student would want to fix them."""
    proxy, port_check = _start_probe_proxy()
    try:
        adapter_checks = _check_adapter(cfg.adapter, plan)
    finally:
        if proxy:
            proxy.stop()
    out: list[Check] = [_check_python()]
    out += _check_data(plan, deep)
    out += adapter_checks
    out += [_check_disk(plan), _check_vcgencmd(), _check_judge(cfg.no_judge), port_check]
    return out


def print_report(results: list[Check], file=None) -> None:
    """One line per check; problems get the fix on the next line."""
    file = file or sys.stdout
    mark = {OK: "ok   ", WARN: "WARN ", FAIL: "FAIL "}
    for level, msg, fix in results:
        print(f"  [{mark[level]}] {msg}", file=file)
        if level != OK and fix:
            print(f"          fix: {fix}", file=file)
    fails = sum(1 for r in results if r[0] == FAIL)
    warns = sum(1 for r in results if r[0] == WARN)
    print(f"\n  {len(results)} checks: {fails} failed, {warns} warnings", file=file)


def to_json(results: list[Check]) -> str:
    return json.dumps([{"level": a, "message": b, "fix": c} for a, b, c in results], indent=2)

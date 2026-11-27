"""The benchmark run: phases, resume, run directory files (see SPEC "Run phases and output").

Phases: preflight -> ingest (ViDoRe packs) -> axis 1 retrieval -> axis 2 gold pages -> axis 3 end-to-end
-> axis 4 CRAG -> judging -> scoring -> report. One question at a time. Every answer is appended to
answers.jsonl the moment it is known, so Ctrl-C (or a crash) loses nothing and `--resume` skips what
is done. Modules owned by other people (crb.data.packs, crb.health, crb.score, crb.report) are
imported lazily inside functions.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from . import __version__, adapter, config
from .config import RunConfig
from .state import State

AXIS_PHASE = {1: "retrieval", 2: "gold", 3: "e2e", 4: "crag"}
AXIS_LABEL = {1: "retrieval", 2: "gold pages", 3: "end-to-end", 4: "CRAG web"}
WARMUP_QUESTIONS = 3


class PlanError(Exception):
    """The questions for this mode cannot be chosen (usually: data not fetched)."""


def is_crag(pack: str) -> bool:
    return "crag" in pack


def _packs():
    return importlib.import_module("crb.data.packs")


# --------------------------------------------------------------------------------------- plan
@dataclass
class Plan:
    sample_name: str
    packs: dict = field(default_factory=dict)   # pack -> {"retrieval": [qids], "answer": [qids]}

    def count(self, axis: int) -> int:
        n = 0
        for pack, s in self.packs.items():
            if axis == 1:
                n += 0 if is_crag(pack) else len(s.get("retrieval", []))
            elif axis in (2, 3):
                n += 0 if is_crag(pack) else len(s.get("answer", []))
            else:
                n += len(s.get("answer", [])) if is_crag(pack) else 0
        return n

    def total(self) -> int:
        return sum(self.count(a) for a in AXIS_PHASE)


def _dev_present() -> bool:
    try:
        d = Path(_packs().data_dir())
    except Exception:  # noqa: BLE001
        return False
    return (d / "samples" / "dev.json").exists() and (d / "packs" / "dev_vidore").is_dir()


def build_plan(mode: str) -> Plan:
    """Choose the questions for a mode. smoke = dev packs if present, else 10 per axis of class-v1."""
    packs = _packs()
    try:
        if mode == "smoke" and _dev_present():
            name = "dev"
        else:
            name = "class-v1"
        sample = packs.load_sample(name)
    except Exception as exc:  # noqa: BLE001
        raise PlanError(f"cannot read the question sample ({type(exc).__name__}: {exc})") from exc
    plan = Plan(sample_name=name)
    for pack, s in (sample.get("packs") or {}).items():
        retr = list(s.get("retrieval") or [])
        ans = list(s.get("answer") or [])
        if mode == "full":
            try:
                ans = [q["qid"] for q in packs.load_questions(pack)]
            except Exception as exc:  # noqa: BLE001
                raise PlanError(f"cannot read the questions of {pack} ({exc})") from exc
        elif mode == "smoke" and name != "dev":
            retr, ans = retr[:10], ans[:10]
        plan.packs[pack] = {"retrieval": retr, "answer": ans}
    if mode == "smoke" and name != "dev":
        plan.sample_name = "class-v1 (first 10 per axis)"
    return plan


# ------------------------------------------------------------------------------ page file helpers
def _flatten(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _flatten(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)
    elif v is not None:
        yield v


def gold_context(pack: str, page_ids: list[str]) -> list[dict]:
    """The `context` list for a gold-mode /answer call. Tolerant of what page_paths returns
    (a dict, a tuple, or one path): files are told apart by their suffix."""
    out = []
    for pid in page_ids:
        image, md_text = None, ""
        try:
            found = _packs().page_paths(pack, pid)
        except Exception:  # noqa: BLE001
            found = None
        if isinstance(found, dict):  # a dict may hold the markdown text itself under a "markdown" key
            txt = found.get("markdown")
            if isinstance(txt, str) and "\n" in txt:
                md_text = txt
        for item in _flatten(found):
            p = Path(str(item))
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and image is None:
                image = str(p)
            elif p.suffix.lower() == ".md" and not md_text:
                try:
                    md_text = p.read_text(encoding="utf-8")
                except OSError:
                    pass
        out.append({"page_id": pid, "markdown": md_text, "image_path": image})
    return out


def web_pages(pack: str, qid: str) -> list[dict]:
    """The `pages` list for a web-mode /answer call: path + the sidecar fields."""
    keys = ("page_name", "page_url", "page_snippet", "page_last_modified")
    out = []
    for item in _packs().crag_pages(pack, qid):
        if isinstance(item, dict):
            d = dict(item)
            if "path" not in d:
                d["path"] = str(d.get("html") or d.get("file") or "")
        else:
            d = {"path": str(item)}
            side = Path(str(item)).with_suffix(".json")
            try:
                d.update({k: v for k, v in json.loads(side.read_text(encoding="utf-8")).items() if k in keys})
            except (OSError, ValueError):
                pass
        d["path"] = str(d["path"])
        for k in keys:
            d.setdefault(k, "")
        out.append(d)
    return out


# ------------------------------------------------------------------------------------- run files
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass  # a half-written last line after a crash
    except OSError:
        pass
    return rows


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def pi_info() -> dict:
    """Where the benchmark ran. Everything optional."""
    info = {"python": platform.python_version(), "machine": platform.machine(), "os": platform.platform(),
            "cpus": os.cpu_count(), "model": None, "mem_total_mb": None}
    try:
        info["model"] = Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["mem_total_mb"] = round(int(line.split()[1]) / 1024)
    except (OSError, ValueError):
        pass
    return info


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RunLog:
    """run.log plus console output."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def line(self, msg: str, show: bool = True) -> None:
        if show:
            print(msg, flush=True)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{_iso()} {msg}\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------------------- Runner
class Runner:
    def __init__(self, cfg: RunConfig, run_dir: Path, plan: Plan, state: State | None = None, k: int = 10,
                 answer_timeout: float = config.ANSWER_TIMEOUT, retrieve_timeout: float = config.RETRIEVE_TIMEOUT) -> None:
        self.cfg, self.dir, self.plan = cfg, Path(run_dir), plan
        self.state = state or State()
        self.k, self.answer_timeout, self.retrieve_timeout = k, answer_timeout, retrieve_timeout
        self.log = RunLog(self.dir / "run.log")
        self.answers_path = self.dir / "answers.jsonl"
        self.session: dict = {}
        self.notes: list[str] = []
        self._questions: dict[str, dict] = {}
        self._warm: dict[tuple, int] = {}
        self.done = {(r.get("qid"), r.get("axis")) for r in read_jsonl(self.answers_path)}
        for r in read_jsonl(self.answers_path):
            key = (r.get("pack"), r.get("axis"))
            self._warm[key] = self._warm.get(key, 0) + 1
        self.errors = 0
        self.asked = 0

    # -- session file -------------------------------------------------------------------------
    def _write_session(self, **extra) -> None:
        self.session.update(extra)
        self.session["notes"] = self.notes
        (self.dir / "run.json").write_text(json.dumps(self.session, indent=2), encoding="utf-8")

    def _questions_of(self, pack: str) -> dict[str, dict]:
        if pack not in self._questions:
            self._questions[pack] = {q["qid"]: q for q in _packs().load_questions(pack)}
        return self._questions[pack]

    # -- phases -------------------------------------------------------------------------------
    def _ingest(self) -> None:
        self.state.set("ingest")
        ing = self.session.setdefault("ingest", {})
        for pack in self.plan.packs:
            if is_crag(pack):
                continue
            if ing.get(pack, {}).get("ok"):
                self.log.line(f"ingest {pack}: already done earlier in this run, skipping")
                continue
            folder = Path(_packs().data_dir()) / "packs" / pack / "pages"
            self.log.line(f"ingest {pack}: sending {folder} to the adapter (can take minutes)...")
            self.state.set("ingest")
            res = adapter.ingest(self.cfg.adapter, pack, str(folder))
            ing[pack] = {"ok": res.ok, "seconds": round(res.seconds, 2), "pages": res.data.get("pages"),
                         "error": res.error}
            if res.ok:
                self.log.line(f"ingest {pack}: done in {res.seconds:.0f} s")
            else:
                self.log.line(f"ingest {pack} FAILED: {res.error}. Skipping the ViDoRe tests for this pack. "
                              "Fix the adapter and use `crb run --resume " + str(self.dir) + "`.")
            self._write_session()

    def _tasks(self, axis: int):
        for pack, s in self.plan.packs.items():
            crag = is_crag(pack)
            if axis == 1 and not crag:
                qids = s.get("retrieval", [])
            elif axis in (2, 3) and not crag:
                qids = s.get("answer", [])
            elif axis == 4 and crag:
                qids = s.get("answer", [])
            else:
                continue
            if not crag and not self.session.get("ingest", {}).get(pack, {}).get("ok"):
                continue
            for qid in qids:
                yield pack, qid

    def _record(self, pack: str, qid: str, axis: int, **fields) -> None:
        key = (pack, axis)
        n = self._warm.get(key, 0)
        self._warm[key] = n + 1
        line = {"qid": qid, "pack": pack, "axis": axis, "warmup": n < WARMUP_QUESTIONS,
                "t": round(self.state.t(), 2), "retrieved": [], "answer": None, "cited": [], "error": None,
                "latency_ms": {"first_token": None, "total": None, "retrieve": None}}
        line.update(fields)
        with open(self.answers_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
        self.done.add((qid, axis))
        self.asked += 1
        if line["error"]:
            self.errors += 1

    def _ask_one(self, pack: str, qid: str, axis: int) -> None:
        q = self._questions_of(pack).get(qid)
        if q is None:
            self._record(pack, qid, axis, error=f"question {qid} not found in the {pack} data")
            return
        text = q["question"]
        url = self.cfg.adapter
        if axis == 1:
            r = adapter.retrieve(url, pack, text, self.k, self.retrieve_timeout)
            self._record(pack, qid, 1, retrieved=r.pages or [], error=r.error,
                         latency_ms={"first_token": None, "total": None, "retrieve": r.ms})
            return
        if axis == 2:
            payload = {"collection": pack, "question": text, "mode": "gold",
                       "context": gold_context(pack, q.get("gold_pages") or [])}
        elif axis == 3:
            payload = {"collection": pack, "question": text, "mode": "e2e"}
        else:
            payload = {"collection": pack, "question": text, "mode": "web",
                       "pages": web_pages(pack, qid), "query_time": q.get("query_time")}
        r = adapter.answer(url, payload, self.answer_timeout)
        self._record(pack, qid, axis, answer=r.answer, cited=r.cited, error=r.error,
                     latency_ms={"first_token": r.first_token_ms, "total": r.total_ms, "retrieve": None})

    def _axis(self, axis: int) -> None:
        todo = [(p, q) for p, q in self._tasks(axis) if (q, axis) not in self.done]
        total = self.plan.count(axis)
        if not todo:
            return
        phase = AXIS_PHASE[axis]
        skipped = total - len(todo)
        self.log.line(f"axis {axis} ({AXIS_LABEL[axis]}): {len(todo)} questions"
                      + (f" ({skipped} already done)" if skipped > 0 else ""))
        tty = sys.stdout.isatty()
        started = time.monotonic()
        for i, (pack, qid) in enumerate(todo, 1):
            self.state.set(phase=phase, qid=qid, axis=axis)
            self._ask_one(pack, qid, axis)
            if tty or i % 25 == 0 or i == len(todo):
                eta = (time.monotonic() - started) / i * (len(todo) - i)
                msg = f"  axis {axis} {i}/{len(todo)}  {pack}  errors so far {self.errors}  ~{eta / 60:.1f} min left"
                print(("\r" + msg + "   ") if tty else msg, end="\n" if (not tty or i == len(todo)) else "", flush=True)
        self.state.set(phase=phase)

    def run_questions(self) -> None:
        """Ingest + all four axes. Raises KeyboardInterrupt to the caller."""
        self._ingest()
        for axis in (1, 2, 3, 4):
            self._axis(axis)
        self.state.set(phase="judging")


# ------------------------------------------------------------------------------------ top level
def _start_health(run_dir: Path, adapter_url: str, state: State, log: RunLog, notes: list):
    """Start the 1 Hz sampler if crb.health exists. Returns (sampler | None, merge_fn)."""
    try:
        health = importlib.import_module("crb.health")
        out = run_dir / "health.csv"
        part = None
        if out.exists() and out.stat().st_size > 0:      # resumed run: keep the old samples
            part = run_dir / ".health.resume.csv"
            out = part
        port = urlparse(adapter_url).port or 80
        sampler = health.HealthSampler(Path(_packs().data_dir()), port, out, state)
        sampler.start()

        def merge() -> None:
            if part is not None and part.exists():
                rows = part.read_text(encoding="utf-8").splitlines()[1:]
                with open(run_dir / "health.csv", "a", encoding="utf-8") as f:
                    f.write("".join(r + "\n" for r in rows))
                part.unlink()
        return sampler, merge
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Pi health sampling was not available ({type(exc).__name__}: {exc}); health numbers are missing.")
        log.line(notes[-1])
        return None, (lambda: None)


def _start_proxy(run_dir: Path, state: State, log: RunLog, notes: list):
    try:
        from .proxy import TokenProxy
        proxy = TokenProxy(state, run_dir / "proxy.jsonl", port=config.proxy_port()).start()
        log.line(f"token proxy listening on {proxy.url} (point your LLM client's base URL there to measure cost)")
        return proxy
    except Exception as exc:  # noqa: BLE001
        notes.append(f"token proxy could not start ({exc}); cost shows as 'not measured'.")
        log.line(notes[-1])
        return None


def score_and_report(run_dir: Path, mode: str, judge: bool, log: RunLog | None = None,
                     judge_model: str | None = None) -> dict | None:
    """Judge, score and write report.html + summary.md. Failures leave answers.jsonl untouched."""
    say = log.line if log else print
    hint = f"Your answers are saved. Run `crb score {run_dir}` to try again."
    try:
        build = importlib.import_module("crb.score.results").build
        kwargs = {"judge": judge, "mode": mode}
        if judge_model:
            import inspect
            if "judge_model" in inspect.signature(build).parameters:
                kwargs["judge_model"] = judge_model
        results = build(Path(run_dir), **kwargs)
    except Exception as exc:  # noqa: BLE001
        say(f"Scoring failed ({type(exc).__name__}: {exc}). {hint}")
        return None
    for modname, what in (("crb.report.html", "report.html"), ("crb.report.summary", "summary.md")):
        try:
            importlib.import_module(modname).write(Path(run_dir), results)
        except Exception as exc:  # noqa: BLE001
            say(f"Could not write {what} ({type(exc).__name__}: {exc}). Run `crb report {run_dir}` to try again.")
    return results


_MAIN_METRIC = {"retrieval": ("ndcg10", "nDCG@10"), "gold": ("acc", "accuracy"), "e2e": ("acc", "accuracy"),
                "crag": ("truthfulness", "truthfulness")}


def print_summary(results: dict, run_dir: Path) -> None:
    """Terminal summary: rank and above/below median per test, latency p90, cost, peak temperature."""
    m = results.get("metrics", {}) or {}
    print("\n=== Results ===")
    ranks = results.get("ranks", {}) or {}
    for key, r in ranks.items():
        axis, _, pack = key.partition(".")
        metric, label = _MAIN_METRIC.get(axis, (None, ""))
        val = (m.get(axis, {}).get(pack, {}) or {}).get(metric)
        val_s = f"{label} {val:g}  " if isinstance(val, (int, float)) else ""
        side = {"above": "above", "below": "below", "at": "at"}.get(r.get("side"), "?")
        print(f"  {key:<34} {val_s}rank {r.get('rank')} of {r.get('of')}  ({side} the published median {r.get('median')})")
    if not ranks:
        print("  (no ranks available)")
    p90 = ((m.get("latency", {}) or {}).get("first_token") or {}).get("p90")
    print(f"  time to first token p90: {p90:g} ms" if isinstance(p90, (int, float)) else "  time to first token p90: not measured")
    cost = m.get("cost", {}) or {}
    if cost.get("measured") and isinstance(cost.get("system_usd"), (int, float)):
        print(f"  system cost: ${cost['system_usd']:.2f}")
    else:
        print("  system cost: not measured (point your LLM client at the token proxy to measure it)")
    h = m.get("health", {}) or {}
    if h.get("available") and isinstance(h.get("temp_max"), (int, float)):
        print(f"  peak temperature: {h['temp_max']:g} C")
    print(f"\nReport: {Path(run_dir) / 'report.html'}")
    print(f"View it from your laptop: crb serve {run_dir}")


def notify(title: str, message: str) -> None:
    """Desktop notification where available; silently nothing otherwise."""
    try:
        if sys.platform == "darwin" and shutil.which("osascript"):
            esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')  # noqa: E731
            subprocess.run(["osascript", "-e", f'display notification "{esc(message)}" with title "{esc(title)}"'],
                           timeout=5, capture_output=True)
        elif shutil.which("notify-send") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            subprocess.run(["notify-send", title, message], timeout=5, capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def run(cfg: RunConfig, *, skip_preflight: bool = False) -> int:
    """`crb run`: returns a process exit code (0 ok, 1 interrupted/incomplete, 2 pre-flight failed)."""
    from . import preflight

    resuming = cfg.resume is not None
    run_dir = Path(cfg.resume).expanduser().resolve() if resuming else None
    old: dict = {}
    if resuming:
        if not (run_dir / "run.json").exists():
            print(f"Cannot resume: {run_dir} has no run.json. Check the path (it looks like runs/<student>/<time>).")
            return 2
        old = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        cfg.student = old.get("student") or cfg.student
        cfg.mode = old.get("mode") or cfg.mode
        cfg.adapter = cfg.adapter or old.get("adapter_url") or config.DEFAULT_ADAPTER
    cfg.adapter = cfg.adapter or config.DEFAULT_ADAPTER

    try:
        plan = build_plan(cfg.mode)
    except PlanError as exc:
        print(f"Problem: {exc}\n  Fix: run `crb fetch` (or `crb fetch --dev` for a quick test) and try again.")
        return 2

    if not skip_preflight:
        results = preflight.run_checks(cfg, plan)
        preflight.print_report(results)
        if any(r[0] == "fail" for r in results):
            print("\nNot starting: fix the problems marked FAIL above, then run again.")
            return 2

    if run_dir is None:
        run_dir = config.new_run_dir(cfg.student)
    last_t = max([float(r.get("t") or 0) for r in read_jsonl(run_dir / "answers.jsonl")] or [0.0])
    state = State(t_offset=max(float(old.get("elapsed_s") or 0), last_t))
    runner = Runner(cfg, run_dir, plan, state)
    log = runner.log
    log.line(f"{'Resuming' if resuming else 'Starting'} run in {run_dir}  (mode {cfg.mode}, sample {plan.sample_name}, "
             f"{plan.total()} questions)")
    h = adapter.health(cfg.adapter)
    runner.session = dict(old) if resuming else {}
    runner.session.update({
        "student": cfg.student, "mode": cfg.mode, "crb_version": __version__, "crb_commit": _git_commit(),
        "adapter_url": cfg.adapter, "adapter_health": h.data if h.ok else None, "pi": pi_info(),
        "sample": plan.sample_name, "start": old.get("start") or _iso(), "end": None, "status": "running",
        "judge": not cfg.no_judge})
    runner.notes = list(old.get("notes") or [])
    runner._write_session()

    sampler, merge_health = _start_health(run_dir, cfg.adapter, state, log, runner.notes)
    proxy = _start_proxy(run_dir, state, log, runner.notes)
    status = "done"
    try:
        runner.run_questions()
    except KeyboardInterrupt:
        status = "interrupted"
        print()
        log.line(f"Stopped. Everything done so far is saved. Continue later with: crb run --resume {run_dir}")
    finally:
        if proxy:
            proxy.stop()
        if sampler:
            try:
                sampler.stop()
                merge_health()
            except Exception as exc:  # noqa: BLE001
                log.line(f"health sampler did not stop cleanly: {exc}")
    elapsed = state.t()
    if status == "done" and any(not v.get("ok") for v in runner.session.get("ingest", {}).values()):
        status = "incomplete"
    runner._write_session(elapsed_s=round(elapsed, 1), status=status, end=_iso() if status != "interrupted" else None)
    if runner.errors:
        log.line(f"{runner.errors} of {runner.asked} questions in this session got an error from your adapter "
                 "(see the 'error' field in answers.jsonl).")
    if status == "interrupted":
        notify("crb stopped", f"Run paused. Resume: crb run --resume {run_dir}")
        return 1

    state.set(phase="judging")
    log.line("Judging and scoring..." if not cfg.no_judge else "Scoring (no judge)...")
    results = score_and_report(run_dir, cfg.mode, not cfg.no_judge, log, cfg.judge_model)
    try:
        runner._write_session(end=_iso(), scored=results is not None)
    except OSError:
        pass
    if results is not None:
        print_summary(results, run_dir)
    notify("crb finished", f"Report: {run_dir / 'report.html'}" if results else f"Answers saved in {run_dir}")
    return 0 if status == "done" and results is not None else 1

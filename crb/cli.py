"""Command line: `crb` (wizard) and its subcommands (see SPEC "CLI")."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__, config


def _add_mode_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="quick test run (not comparable with the papers)")
    g.add_argument("--official", action="store_true", help="the class sample, judged like the papers (default)")
    g.add_argument("--full", action="store_true", help="every question of every dataset")


def _mode_of(args) -> str:
    return "smoke" if args.smoke else "full" if args.full else "official"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crb", description="Benchmark your document-QA (RAG) system on this Raspberry Pi. "
        "Run `crb` with no arguments for the guided version.")
    p.add_argument("--version", action="version", version=f"crb {__version__}")
    sub = p.add_subparsers(dest="cmd")

    f = sub.add_parser("fetch", help="download (or build the tiny dev) datasets into $CRB_DATA")
    f.add_argument("--dev", action="store_true", help="build the tiny offline dev packs (no internet)")
    f.add_argument("--pack", help="only this pack (vidore_v3_hr, vidore_v3_computer_science, crag_task1)")
    f.add_argument("--data-dir", help="where to put the data (default $CRB_DATA or ~/crb-data)")
    f.add_argument("--force", action="store_true", help="download again even if present")

    c = sub.add_parser("check", help="pre-flight only: data, adapter, disk, judge key, ports")
    _add_mode_flags(c)
    c.add_argument("--adapter", help=f"adapter URL (default {config.DEFAULT_ADAPTER})")
    c.add_argument("--no-judge", action="store_true", help="do not require a judge key")
    c.add_argument("--deep", action="store_true", help="also verify every data file's checksum (slow on a Pi)")

    r = sub.add_parser("run", help="run the benchmark")
    _add_mode_flags(r)
    r.add_argument("--student", help="your student number")
    r.add_argument("--adapter", help=f"adapter URL (default {config.DEFAULT_ADAPTER})")
    r.add_argument("--resume", metavar="RUN_DIR", help="continue a stopped run (skips questions already answered)")
    r.add_argument("--no-judge", action="store_true", help="save answers but skip judging (use `crb score` later)")

    s = sub.add_parser("score", help="(re)judge and (re)score a run")
    s.add_argument("run_dir")
    s.add_argument("--judge-model", help="override the judge model")
    s.add_argument("--no-judge", action="store_true", help="score what needs no judge (retrieval) only")

    rp = sub.add_parser("report", help="rebuild report.html and summary.md from results.json")
    rp.add_argument("run_dir")

    sv = sub.add_parser("serve", help="serve the runs folder read-only so you can open the report from your laptop")
    sv.add_argument("run_dir", nargs="?")
    sv.add_argument("--port", type=int, default=config.DEFAULT_SERVE_PORT)

    b = sub.add_parser("baseline", help="start the reference adapter on :8000 (anything after -- goes to it)")
    b.add_argument("extra", nargs=argparse.REMAINDER)

    rm = sub.add_parser("remote", help="(on your laptop) run on the Pi over SSH, then copy the run here and open the report")
    rm.add_argument("host", help="the Pi, as you would type it for ssh (e.g. abnunez@192.168.1.80 or watsonpi)")
    _add_mode_flags(rm)
    rm.add_argument("--student", help="your student number")
    rm.add_argument("--adapter", help="adapter URL as seen from the Pi")
    rm.add_argument("--no-judge", action="store_true")
    _add_remote_flags(rm)

    pl = sub.add_parser("pull", help="(on your laptop) copy a run from the Pi and open its report")
    pl.add_argument("host", help="the Pi, as you would type it for ssh")
    pl.add_argument("--run", help="run folder on the Pi (default: the newest)")
    pl.add_argument("--wait", action="store_true", help="wait for the run that is going on now to finish first")
    _add_remote_flags(pl)
    return p


def _add_remote_flags(p: argparse.ArgumentParser) -> None:
    from .remote import DEFAULT_REMOTE_DIR
    p.add_argument("--dir", default=DEFAULT_REMOTE_DIR, help=f"repo folder on the Pi (default {DEFAULT_REMOTE_DIR})")
    p.add_argument("--to", help="where to put runs on this computer (default ~/crb-runs)")
    p.add_argument("--no-open", action="store_true", help="do not open the report in a browser")


def _cmd_remote(args) -> int:
    from . import remote
    run_args = ["--" + _mode_of(args)]
    if args.student:
        run_args += ["--student", args.student]
    if args.adapter:
        run_args += ["--adapter", args.adapter]
    if args.no_judge:
        run_args.append("--no-judge")
    return remote.remote_run(args.host, run_args, args.dir, Path(args.to).expanduser() if args.to else None,
                             not args.no_open)


def _cmd_pull(args) -> int:
    from . import remote
    return remote.pull(args.host, args.dir, args.run, args.wait,
                       Path(args.to).expanduser() if args.to else None, not args.no_open)


def _cmd_check(args) -> int:
    from . import preflight, runner
    cfg = config.RunConfig(adapter=args.adapter or config.load_last().get("adapter") or config.DEFAULT_ADAPTER,
                           mode=_mode_of(args), no_judge=args.no_judge)
    try:
        plan = runner.build_plan(cfg.mode)
    except runner.PlanError as exc:
        print(f"  [FAIL ] {exc}\n          fix: run `crb fetch` (or `crb fetch --dev`) and try again")
        return 2
    results = preflight.run_checks(cfg, plan, deep=args.deep)
    preflight.print_report(results)
    return 2 if any(r[0] == preflight.FAIL for r in results) else 0


def _cmd_run(args) -> int:
    from . import runner
    last = config.load_last()
    cfg = config.RunConfig(
        student=args.student or last.get("student") or "student",
        adapter=args.adapter or (None if args.resume else last.get("adapter")),
        mode=_mode_of(args), no_judge=args.no_judge,
        resume=Path(args.resume) if args.resume else None)
    if args.student or args.adapter:
        config.save_last(student=args.student, adapter=args.adapter)
    return runner.run(cfg)


def _cmd_wizard() -> int:
    from . import runner, wizard
    try:
        cfg = wizard.run_wizard()
    except KeyboardInterrupt:
        print("\nBye.")
        return 130
    return runner.run(cfg) if cfg else 1


def _load_run(run_dir: str):
    d = Path(run_dir).expanduser().resolve()
    if not (d / "answers.jsonl").exists():
        print(f"{d} has no answers.jsonl. Check the path (it looks like runs/<student>/<time>).")
        return d, None
    try:
        return d, json.loads((d / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return d, {}


def _cmd_score(args) -> int:
    from . import runner
    d, sess = _load_run(args.run_dir)
    if sess is None:
        return 2
    results = runner.score_and_report(d, sess.get("mode") or "official", not args.no_judge, None, args.judge_model)
    if results is None:
        return 1
    runner.print_summary(results, d)
    return 0


def _cmd_report(args) -> int:
    import importlib
    d = Path(args.run_dir).expanduser().resolve()
    try:
        results = json.loads((d / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"{d} has no usable results.json. Run `crb score {d}` first.")
        return 2
    rc = 0
    for mod, what in (("crb.report.html", "report.html"), ("crb.report.summary", "summary.md")):
        try:
            path = importlib.import_module(mod).write(d, results)
            print(f"Wrote {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not write {what} ({type(exc).__name__}: {exc}).")
            rc = 1
    return rc


def _cmd_baseline(args) -> int:
    script = Path(__file__).resolve().parent.parent / "examples" / "baseline_adapter.py"
    if not script.exists():
        print(f"The reference adapter is not here ({script}). Reinstall from the repository checkout.")
        return 2
    extra = [a for a in args.extra if a != "--"]
    print("Starting the reference adapter on http://127.0.0.1:8000 (Ctrl-C to stop)")
    try:
        return subprocess.call([sys.executable, str(script), *extra])
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config.load_env_files()
    try:
        if args.cmd is None:
            return _cmd_wizard()
        if args.cmd == "fetch":
            from crb.data import fetch
            return int(fetch.main(args) or 0)
        if args.cmd == "serve":
            from . import serve
            return serve.serve(args.run_dir, args.port)
        return {"check": _cmd_check, "run": _cmd_run, "score": _cmd_score,
                "report": _cmd_report, "baseline": _cmd_baseline,
                "remote": _cmd_remote, "pull": _cmd_pull}[args.cmd](args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

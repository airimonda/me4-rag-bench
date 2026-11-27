"""The questions `crb` asks when run with no arguments. Answers are remembered in ~/.config/crb/last.json."""
from __future__ import annotations

import getpass
import json
import os
import sys

from . import config
from .config import RunConfig

_MODES = [
    ("smoke", "quick test: the tiny dev packs, or 10 questions per test; cheap judge; the report says 'not comparable'"),
    ("official", "the class sample (about 1,300 questions) judged like the papers: this is the one to hand in"),
    ("full", "every question of every dataset (takes much longer and costs more)"),
]


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        print()
        raise KeyboardInterrupt from None
    return answer or (default or "")


def _unfinished(student: str):
    """The student's latest run if it was stopped before finishing."""
    d = config.latest_run(student)
    if d is None:
        return None
    try:
        status = json.loads((d / "run.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None
    return d if status in ("running", "interrupted", "incomplete") else None


def run_wizard() -> RunConfig | None:
    """Ask the three questions. Returns None if the student backs out."""
    last = config.load_last()
    print("crb: the RAG benchmark. Press Enter to accept the answer in [brackets]; Ctrl-C quits.\n")
    student = ""
    while not student:
        student = _ask("Student number", last.get("student"))
    adapter = _ask("Adapter URL (where your system listens)", last.get("adapter") or config.DEFAULT_ADAPTER)

    old = _unfinished(student)
    if old is not None:
        if _ask(f"An unfinished run exists ({old}). Resume it? (y/n)", "y").lower().startswith("y"):
            config.save_last(student=student, adapter=adapter)
            return _with_judge(RunConfig(student=student, adapter=adapter, mode="official", resume=old))

    print("\nWhich run?")
    for i, (name, text) in enumerate(_MODES, 1):
        print(f"  [{i}] {name:<9} {text}")
    default = str([m for m, _ in _MODES].index(last.get("mode")) + 1) if last.get("mode") in dict(_MODES) else "1"
    pick = _ask("Choose 1-3", default)
    mode = _MODES[int(pick) - 1][0] if pick in ("1", "2", "3") else "smoke"
    config.save_last(student=student, adapter=adapter, mode=mode)
    return _with_judge(RunConfig(student=student, adapter=adapter, mode=mode))


def _with_judge(cfg: RunConfig) -> RunConfig:
    """If there is no judge key, offer to paste one (kept in memory only) or to skip judging."""
    if config.judge_key():
        return cfg
    print("\nNo judge key found (CRB_JUDGE_KEY or OPENROUTER_API_KEY).")
    try:
        key = getpass.getpass("Paste an OpenRouter key to judge answers (not saved anywhere), or Enter to skip judging: ").strip()
    except EOFError:
        key = ""
    if key:
        os.environ["CRB_JUDGE_KEY"] = key
    else:
        cfg.no_judge = True
        print("OK: answers will be saved; score them later with `crb score RUN_DIR` once you have a key.")
    return cfg


def is_interactive() -> bool:
    return sys.stdin.isatty()

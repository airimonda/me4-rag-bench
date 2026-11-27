"""Paths, defaults and remembered wizard answers. Nothing here ever stores an API key."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ADAPTER = "http://127.0.0.1:8000"
DEFAULT_UPSTREAM = "https://openrouter.ai/api/v1"
PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 8790
DEFAULT_SERVE_PORT = 8302
SMOKE_JUDGE_MODEL = "anthropic/claude-haiku-5.5"
MODES = ("smoke", "official", "full")

# Adapter timeouts (seconds), from SPEC.
ANSWER_TIMEOUT = 120.0
RETRIEVE_TIMEOUT = 30.0
HEALTH_TIMEOUT = 10.0
INGEST_TIMEOUT = 3600.0


def runs_root() -> Path:
    """Where run directories live: $CRB_RUNS, else ./runs."""
    return Path(os.environ.get("CRB_RUNS") or "runs").expanduser().resolve()


def config_dir() -> Path:
    return Path(os.environ.get("CRB_CONFIG_DIR") or "~/.config/crb").expanduser()


def proxy_port() -> int:
    try:
        return int(os.environ.get("CRB_PROXY_PORT", DEFAULT_PROXY_PORT))
    except ValueError:
        return DEFAULT_PROXY_PORT


def upstream() -> str:
    return os.environ.get("CRB_UPSTREAM") or DEFAULT_UPSTREAM


def judge_key() -> str | None:
    """The judge's API key from the environment (never written anywhere)."""
    return os.environ.get("CRB_JUDGE_KEY") or os.environ.get("OPENROUTER_API_KEY") or None


def load_env_files() -> list[Path]:
    """Read KEY=VALUE lines from ~/.config/crb/env and ./.env into the environment.

    Variables already set win. This lets a run started over SSH (where shell profiles are often
    skipped) still find OPENROUTER_API_KEY. Returns the files that were read.
    """
    read = []
    for path in (config_dir() / "env", Path(".env")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        read.append(path)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.removeprefix("export ").partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    return read


def load_last() -> dict:
    try:
        data = json.loads((config_dir() / "last.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_last(**updates) -> None:
    """Remember wizard answers. Failing to save is not an error."""
    data = load_last()
    data.update({k: v for k, v in updates.items() if v is not None})
    try:
        d = config_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "last.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def slug(student: str) -> str:
    """Filesystem-safe student id."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(student).strip()).strip("._")
    return s or "student"


def new_run_dir(student: str) -> Path:
    """Create runs/<student>/<YYYYmmdd-HHMMSS>/ (adds a suffix if it already exists)."""
    base = runs_root() / slug(student)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d, n = base / stamp, 1
    while d.exists():
        n += 1
        d = base / f"{stamp}-{n}"
    d.mkdir(parents=True)
    return d


def latest_run(student: str) -> Path | None:
    base = runs_root() / slug(student)
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir() if (p / "run.json").exists())
    return runs[-1] if runs else None


@dataclass
class RunConfig:
    """Everything `crb run` needs, however it was asked for (flags or wizard)."""
    student: str = "student"
    adapter: str = DEFAULT_ADAPTER
    mode: str = "official"
    no_judge: bool = False
    resume: Path | None = None
    judge_model: str | None = None

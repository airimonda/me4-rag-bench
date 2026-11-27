"""LLM judge client (OpenRouter chat completions) with retries, a sqlite cache and parsing.

Key: env CRB_JUDGE_KEY, else OPENROUTER_API_KEY. Base URL: env CRB_JUDGE_BASE (default OpenRouter).
Cache: ~/.cache/crb/judge.sqlite, or the path in env CRB_JUDGE_CACHE. Keys never reach any log.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import requests

DEFAULT_BASE = "https://openrouter.ai/api/v1"
VIDORE_MODEL = "openai/gpt-5.2"
CRAG_MODELS = ("openai/gpt-3.5-turbo-0613", "meta-llama/llama-3.1-70b-instruct")
SMOKE_MODEL = "anthropic/claude-haiku-5.5"
WORKERS = 4
TIMEOUT_S = 120

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

VIDORE_LABELS = ("Correct", "Partially Correct", "Incorrect")
VIDORE_SCORE = {"Correct": 1.0, "Partially Correct": 0.5, "Incorrect": 0.0}


class JudgeError(Exception):
    """A judge call failed for good (after retries). Message never contains the key."""


class NoJudgeKey(JudgeError):
    pass


@dataclass
class Verdict:
    label: str | None
    raw: str = ""
    cost: float | None = None
    cached: bool = False
    error: str | None = None


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def judge_key() -> str | None:
    return os.environ.get("CRB_JUDGE_KEY") or os.environ.get("OPENROUTER_API_KEY") or None


def judge_base() -> str:
    return (os.environ.get("CRB_JUDGE_BASE") or DEFAULT_BASE).rstrip("/")


def cache_path() -> Path:
    env = os.environ.get("CRB_JUDGE_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "crb" / "judge.sqlite"


# ---------------------------------------------------------------- parsing

def _json_obj(text: str) -> dict | None:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    for cand in (t,):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            v = json.loads(t[a:b + 1])
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    return None


def parse_vidore(text: str) -> str | None:
    """Return one of VIDORE_LABELS or None."""
    obj = _json_obj(text)
    val = None
    if obj:
        for k in ("judgment", "judgement", "label", "verdict"):
            if k in obj:
                val = str(obj[k])
                break
    if val is None:
        m = re.search(r'"?judg(?:e)?ment"?\s*[:=]\s*"?([A-Za-z ]+)', text or "", flags=re.I)
        if m:
            val = m.group(1)
    if val is None and (text or "").strip().strip(".").lower() in ("correct", "incorrect", "partially correct"):
        val = text
    if val is None:
        return None
    v = val.strip().lower()
    if v.startswith("partial"):
        return "Partially Correct"
    if v.startswith("incorrect") or v.startswith("wrong"):
        return "Incorrect"
    if v.startswith("correct"):
        return "Correct"
    return None


def parse_crag(text: str) -> str | None:
    """Return 'accurate' or 'incorrect' or None."""
    obj = _json_obj(text)
    val = None
    if obj:
        for k in ("judgment", "judgement", "accuracy", "label", "verdict"):
            if k in obj:
                val = obj[k]
                break
    if val is None:
        m = re.search(r'"?(?:judg(?:e)?ment|accuracy)"?\s*[:=]\s*"?([A-Za-z]+)', text or "", flags=re.I)
        if m:
            val = m.group(1)
    if val is None and (text or "").strip().strip(".").lower() in ("accurate", "incorrect", "true", "false"):
        val = text
    if val is None:
        return None
    if isinstance(val, bool):
        return "accurate" if val else "incorrect"
    v = str(val).strip().lower()
    if v.startswith(("incorrect", "inaccurate", "false", "no", "wrong")):
        return "incorrect"
    if v.startswith(("accurate", "correct", "true", "yes")):
        return "accurate"
    return None


# ---------------------------------------------------------------- cache

class Cache:
    """(judge model, prompt hash, qid, answer hash) -> verdict. Fails soft if the file is unusable."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else cache_path()
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS judge (model TEXT, prompt_hash TEXT, qid TEXT, answer_hash TEXT,"
                " label TEXT, raw TEXT, cost REAL, PRIMARY KEY (model, prompt_hash, qid, answer_hash))")
            self._db.commit()
        except (OSError, sqlite3.Error):
            self._db = None

    def get(self, model: str, prompt_hash: str, qid: str, answer_hash: str):
        if not self._db:
            return None
        with self._lock:
            try:
                return self._db.execute(
                    "SELECT label, raw, cost FROM judge WHERE model=? AND prompt_hash=? AND qid=? AND answer_hash=?",
                    (model, prompt_hash, qid, answer_hash)).fetchone()
            except sqlite3.Error:
                return None

    def put(self, model: str, prompt_hash: str, qid: str, answer_hash: str, label: str, raw: str,
            cost: float | None) -> None:
        if not self._db:
            return
        with self._lock:
            try:
                self._db.execute("INSERT OR REPLACE INTO judge VALUES (?,?,?,?,?,?,?)",
                                 (model, prompt_hash, qid, answer_hash, label, raw, cost))
                self._db.commit()
            except sqlite3.Error:
                pass

    def close(self) -> None:
        if self._db:
            try:
                self._db.close()
            except sqlite3.Error:
                pass
            self._db = None


# ---------------------------------------------------------------- client

def _backoff_scale() -> float:
    try:
        return float(os.environ.get("CRB_JUDGE_BACKOFF", "1"))
    except ValueError:
        return 1.0


class Judge:
    """Runs judge prompts. `vidore()` and `crag()` return a Verdict and never raise."""

    def __init__(self, key: str | None = None, base: str | None = None, cache: Cache | None = None,
                 max_attempts: int = 5, workers: int = WORKERS, sleep: Callable[[float], None] = time.sleep):
        self.key = key if key is not None else judge_key()
        self.base = (base or judge_base()).rstrip("/")
        self.cache = cache if cache is not None else Cache()
        self.max_attempts = max_attempts
        self.workers = workers
        self._sleep = sleep
        self.vidore_template = load_prompt("vidore_judge.txt")
        self.crag_template = load_prompt("crag_judge.txt")
        self._cost_lock = threading.Lock()
        self.cost_usd = 0.0          # paid in this process (cache hits cost nothing)
        self.calls = 0

    # -- low level
    def _post(self, model: str, prompt: str, reasoning: str | None) -> tuple[str, float | None]:
        if not self.key:
            raise NoJudgeKey("No judge key: set CRB_JUDGE_KEY (or OPENROUTER_API_KEY) to judge answers.")
        body: dict = {"model": model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}],
                      "usage": {"include": True}}
        if reasoning:
            body["reasoning"] = {"effort": reasoning}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "crb"}
        last = "no attempt made"
        for attempt in range(self.max_attempts):
            if attempt:
                self._sleep(min(30.0, 2.0 ** attempt) * _backoff_scale() * (0.75 + random.random() / 2))
            try:
                r = requests.post(f"{self.base}/chat/completions", headers=headers, json=body,
                                  timeout=TIMEOUT_S)
            except requests.RequestException as e:
                last = f"network error ({type(e).__name__})"
                continue
            if r.status_code in (401, 403):
                raise JudgeError(f"The judge key was rejected (HTTP {r.status_code}).")
            if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code >= 400:
                raise JudgeError(f"Judge request failed (HTTP {r.status_code}).")
            try:
                data = r.json()
                text = data["choices"][0]["message"].get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                last = "unreadable reply"
                continue
            cost = None
            usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
                cost = float(usage["cost"])
            with self._cost_lock:
                self.calls += 1
                if cost:
                    self.cost_usd += cost
            return text, cost
        raise JudgeError(f"Judge call failed after {self.max_attempts} tries ({last}).")

    def _run(self, model: str, template: str, fields: dict, qid: str, answer: str,
             parse: Callable[[str], str | None], reasoning: str | None) -> Verdict:
        prompt = re.sub(r"\{(question|gold_answer|answer)\}", lambda m: fields.get(m.group(1), ""), template)
        ph = sha(template + "|" + (reasoning or ""))
        ah = sha(answer or "")
        hit = self.cache.get(model, ph, qid, ah)
        if hit:
            return Verdict(label=hit[0], raw=hit[1] or "", cost=hit[2], cached=True)
        # one extra try when the reply cannot be parsed
        last_raw = ""
        for _ in range(2):
            try:
                text, cost = self._post(model, prompt, reasoning)
            except JudgeError as e:
                return Verdict(label=None, error=str(e))
            last_raw = text
            label = parse(text)
            if label:
                self.cache.put(model, ph, qid, ah, label, text, cost)
                return Verdict(label=label, raw=text, cost=cost)
        return Verdict(label=None, raw=last_raw, error="The judge reply could not be read.")

    # -- public
    def vidore(self, qid: str, question: str, gold: str, answer: str, model: str = VIDORE_MODEL) -> Verdict:
        reasoning = "medium" if model.startswith("openai/gpt-5") else None
        return self._run(model, self.vidore_template,
                         {"question": question, "gold_answer": gold, "answer": answer},
                         qid, answer, parse_vidore, reasoning)

    def crag(self, qid: str, question: str, gold: str, answer: str, model: str) -> Verdict:
        return self._run(model, self.crag_template,
                         {"question": question, "gold_answer": gold, "answer": answer},
                         qid, answer, parse_crag, None)

    def map(self, fn: Callable, jobs: Iterable) -> list:
        """Run fn(job) for every job on `workers` threads, results in job order."""
        jobs = list(jobs)
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            return list(ex.map(fn, jobs))

    def close(self) -> None:
        self.cache.close()

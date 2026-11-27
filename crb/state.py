"""Shared run state read by the health sampler and the token proxy (thread-safe)."""
from __future__ import annotations

import threading
import time


class State:
    """Which phase / question / axis the runner is on right now.

    The runner (crb.runner) writes; the health sampler and token proxy only read.
    `t()` is seconds since the run started, on the monotonic clock, so every log lines up.
    """

    def __init__(self, t_offset: float = 0.0) -> None:
        """`t_offset` continues the clock of an earlier session when a run is resumed."""
        self._lock = threading.Lock()
        self._t0 = time.monotonic() - t_offset
        self._phase = "preflight"
        self._qid: str | None = None
        self._axis: int | None = None

    def t(self) -> float:
        return time.monotonic() - self._t0

    def set(self, phase: str | None = None, qid: str | None = None, axis: int | None = None) -> None:
        with self._lock:
            if phase is not None:
                self._phase = phase
            self._qid = qid
            self._axis = axis

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @property
    def qid(self) -> str | None:
        with self._lock:
            return self._qid

    @property
    def axis(self) -> int | None:
        with self._lock:
            return self._axis

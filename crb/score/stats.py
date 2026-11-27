"""Small statistics helpers (no numpy): percentiles, medians, bootstrap interval with FPC."""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

BOOT_RESAMPLES = 2000
BOOT_SEED = 7


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def percentile(xs: Iterable[float], q: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default). q in 0..100."""
    s = sorted(xs)
    if not s:
        return None
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def median(xs: Iterable[float]) -> float | None:
    return percentile(xs, 50)


def fpc(n: int, population: int | None) -> float:
    """Finite-population correction sqrt((N-n)/(N-1)). 1.0 when N unknown, 0.0 when n >= N."""
    if population is None:
        return 1.0
    if n >= population or population <= 1:
        return 0.0
    return math.sqrt((population - n) / (population - 1))


def bootstrap_ci(values: Sequence[float], population: int | None = None,
                 resamples: int = BOOT_RESAMPLES, seed: int = BOOT_SEED) -> float | None:
    """Half-width of the 95% percentile-bootstrap interval of the mean, in the units of `values`.

    `population` is the size N of the pack the sample was drawn from; the half-width is multiplied by
    sqrt((N-n)/(N-1)). When n == N the result is exactly 0. Returns None for an empty input.
    """
    n = len(values)
    if n == 0:
        return None
    if population is not None and n >= population:
        return 0.0
    if n == 1:
        return 0.0
    rng = random.Random(seed)
    vals = list(values)
    means = []
    for _ in range(resamples):
        means.append(sum(rng.choices(vals, k=n)) / n)
    lo = percentile(means, 2.5)
    hi = percentile(means, 97.5)
    return (hi - lo) / 2.0 * fpc(n, population)

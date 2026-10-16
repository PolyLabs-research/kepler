"""
replicates.py - Error bars from the spread between replicates, not from the scatter about a curve.

A replicate is one whole run (a seed): its curve over time, or its value at every setting. The
points of one replicate are correlated (a soup that starts dense stays dense for a while), so a
mean curve's scatter about a law understates the uncertainty of the law's constants, and its
chi-square overstates the evidence between laws. Resampling whole replicates keeps that
correlation (the bootstrap over seeds):

    fit_replicates(law, x, Y)       Y holds one row per replicate. The law is fitted to the mean
                                    curve; its constants' covariance, error bars and percentile
                                    intervals come from refitting resamples of the replicates.
    compare_replicates(laws, x, Y)  how often each law is chosen (lowest BIC within the Occam
                                    window) across the same resamples. A verdict the seeds
                                    support holds in nearly every resample; one that flips is
                                    not settled, whatever a single fit's BIC says.

Seeded: the same call gives the same numbers. The mean curve is fitted with each point weighted
by its standard error over replicates (floored at a tenth of the median, so that points every
replicate agrees on, such as a shared starting density, do not take infinite weight).
"""
from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from kepler.fitting import Fit, _jsonable, fit
from kepler.laws import Law
from kepler.ranking import OCCAM_WINDOW, Ranking, rank


def _matrix(x, Y) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim != 2 or Y_arr.shape[1] != len(x_arr):
        raise ValueError(f"Y must hold one row of {len(x_arr)} values per replicate (got shape {Y_arr.shape})")
    if Y_arr.shape[0] < 2:
        raise ValueError("the spread between replicates needs at least two of them")
    if not np.all(np.isfinite(Y_arr)) or not np.all(np.isfinite(x_arr)):
        raise ValueError("x and Y must be finite")
    return x_arr, Y_arr


def _mean_and_error(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = Y.mean(axis=0)
    sem = Y.std(axis=0, ddof=1) / math.sqrt(Y.shape[0])
    positive = sem[sem > 0]
    floor = 0.1 * float(np.median(positive)) if len(positive) else 1.0
    return mean, np.maximum(sem, floor)


@dataclass(eq=False)
class ReplicateFit:
    """The law fitted to the mean curve, with constants' errors from resampled replicates.
    `fit` carries the bootstrap covariance, so kepler.predict gives seed-to-seed bands."""

    fit: Fit
    replicates: int
    resamples: int  # resamples whose fit succeeded
    naive_stderr: Tuple[float, ...]  # the fit's own error bars, which assume independent points
    intervals: Dict[str, Tuple[float, float]]  # percentile intervals at `level`
    level: float
    samples: np.ndarray = field(repr=False)  # constants from each resample

    def value(self, name: str) -> float:
        return self.fit.value(name)

    def error(self, name: str) -> float:
        return self.fit.error(name)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable({
            "fit": self.fit.to_dict(),
            "replicates": self.replicates,
            "resamples": self.resamples,
            "naive_stderr": dict(zip(self.fit.param_names, self.naive_stderr)),
            "intervals": {k: list(v) for k, v in self.intervals.items()},
            "level": self.level,
        })


def _resample(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return Y[rng.integers(0, Y.shape[0], Y.shape[0])]


def fit_replicates(
    skeleton: Union[str, Law],
    x: Sequence[float],
    Y: Sequence[Sequence[float]],
    space: str = "linear",
    *,
    resamples: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> ReplicateFit:
    """Fit `skeleton` to the mean of the replicates `Y` (one row each), with the covariance of
    its constants over `resamples` bootstrap resamples of whole replicates."""
    x_arr, Y_arr = _matrix(x, Y)
    mean, err = _mean_and_error(Y_arr)
    base = fit(skeleton, x_arr, mean, sigma=err, space=space)
    if not base.success:
        return ReplicateFit(base, Y_arr.shape[0], 0, base.stderr, {}, level, np.zeros((0, base.n_constants)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        m, e = _mean_and_error(_resample(Y_arr, rng))
        try:
            f = fit(skeleton, x_arr, m, sigma=e, space=space, p0=base.params)
        except ValueError:
            continue
        if f.success and np.all(np.isfinite(f.params)):
            draws.append(f.params)
    samples = np.asarray(draws, dtype=np.float64).reshape(-1, base.n_constants)
    k = base.n_constants
    if len(samples) >= max(10, 2 * k):
        cov = np.atleast_2d(np.cov(samples, rowvar=False))
        tail = (1.0 - level) / 2.0 * 100.0
        low, high = np.percentile(samples, [tail, 100.0 - tail], axis=0)
        intervals = {p: (float(lo), float(hi)) for p, lo, hi in zip(base.param_names, low, high)}
        message = f"{base.message}; errors from {len(samples)} resamples of {Y_arr.shape[0]} replicates"
    else:
        cov = np.full((k, k), np.inf)
        intervals = {}
        message = f"{base.message}; too few resamples could be fitted ({len(samples)}) for error bars"
    stderr = tuple(float(math.sqrt(v)) if v >= 0 else math.nan for v in np.diag(cov))
    boot = dataclasses.replace(base, covariance=cov, stderr=stderr, message=message, error_scale=math.nan)
    return ReplicateFit(boot, Y_arr.shape[0], len(samples), base.stderr, intervals, level, samples)


@dataclass
class ReplicateComparison:
    """Each law's share of the resamples in which it was chosen, and the ranking on all the data."""

    shares: Dict[str, float]  # law -> share of resamples it won (None counted as "none")
    best: Optional[str]  # the law chosen most often
    share: float  # its share
    resamples: int
    ranking: Ranking = field(repr=False)  # on the mean of every replicate

    def to_dict(self) -> Dict[str, Any]:
        return {"shares": self.shares, "best": self.best, "share": self.share, "resamples": self.resamples,
                "ranking": self.ranking.to_dict()}


def compare_replicates(
    laws: Sequence[Union[str, Law]],
    x: Sequence[float],
    Y: Sequence[Sequence[float]],
    space: str = "linear",
    *,
    resamples: int = 200,
    seed: int = 0,
    occam_window: float = OCCAM_WINDOW,
) -> ReplicateComparison:
    """How often each law is chosen across bootstrap resamples of the replicates."""
    x_arr, Y_arr = _matrix(x, Y)
    mean, err = _mean_and_error(Y_arr)
    full = [fit(law, x_arr, mean, sigma=err, space=space) for law in laws]
    ranking = rank(full, occam_window=occam_window)
    starts = {f.name: f.params if f.success else None for f in full}
    rng = np.random.default_rng(seed)
    wins: Counter = Counter()
    for _ in range(resamples):
        m, e = _mean_and_error(_resample(Y_arr, rng))
        fits: List[Fit] = []
        for law, f0 in zip(laws, full):
            start = starts[f0.name]
            try:
                fits.append(fit(law, x_arr, m, sigma=e, space=space, p0=start) if start is not None
                            else fit(law, x_arr, m, sigma=e, space=space))
            except ValueError:
                fits.append(fit(law, x_arr, m, sigma=e, space=space))
        r = rank(fits, occam_window=occam_window)
        wins[r.best.name if r.best else "none"] += 1
    shares = {f.name: wins.get(f.name, 0) / resamples for f in full}
    if wins.get("none"):
        shares["none"] = wins["none"] / resamples
    best = max(shares, key=shares.get) if shares else None
    return ReplicateComparison(shares, best, shares.get(best, 0.0) if best else 0.0, resamples, ranking)

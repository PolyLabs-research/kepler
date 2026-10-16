"""
ranking.py - Choose among fitted laws: BIC with an Occam window.

This is the rule Kepler's ranked discovery has always used
(galileo_tools.GalileoTools.discover_best), as plain functions that need no
GalileoTools instance:

    bic(rss, n, k) = n ln(rss / n) + k ln n

    eligible       the fit succeeded, its residuals are finite, it is not
                   ill-conditioned (cancelling terms), and it does no worse
                   than predicting the mean (R^2 >= 0)
    Occam window   among eligible fits within `occam_window` BIC points of
                   the best, the one with the fewest constants wins (ties by
                   BIC): a richer law has to fit clearly better to be chosen
                   (Kass & Raftery: 2-6 points is "positive", above 6 "strong")
    confident      the winner reaches R^2 >= min_r2, or it has no x in it (a
                   constant): R^2 measures the variation a law explains beyond
                   a constant, so it is 0 for the constant itself, which wins
                   only when no richer law earns its extra constants

BIC only compares fits of the same data: the same points, the same weights,
the same space (linear or log) and the same likelihood (Fit.neg2loglik has a
different base for known and unknown errors). rank() refuses anything else.

Overdispersion. Fits with absolute sigmas and `inflate` (errors that are real
estimates, which the data may outrun) are compared on a quasi-likelihood: every
chi-square is divided by one shared factor c = max(1, the smallest chi^2 / dof
among the fits), the reduced chi-square of the law that fits best. If the
points scatter more than their stated errors allow, c > 1 and differences
between laws shrink accordingly; sigmas that are too small cannot make a
verdict look decisive. With c = 1 this is the plain known-error BIC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

OCCAM_WINDOW = 6.0
MIN_R2 = 0.5
# R^2 of a constant is 0 by construction (least squares lands on the weighted mean), so rounding
# can leave it a hair below zero; that is not a fit worse than the mean.
R2_TOLERANCE = 1e-9

T = TypeVar("T")


def bic(rss: float, n: int, k: int) -> float:
    """Bayesian information criterion for a least-squares fit with Gaussian residuals:
    n ln(rss / n) + k ln n, with rss floored so an exact fit stays finite."""
    rss = max(float(rss), 1e-300)
    return float(n * math.log(rss / n) + k * math.log(max(n, 2)))


def occam_order(
    items: Sequence[T],
    window: float,
    bic_of: Callable[[T], float],
    k_of: Callable[[T], int],
) -> List[T]:
    """`items` in selection order: those within `window` BIC points of the lowest, fewest
    constants first (then by BIC), followed by the rest by BIC."""
    by_bic = sorted(items, key=bic_of)
    if not by_bic:
        return []
    floor = bic_of(by_bic[0])
    inside = [it for it in by_bic if bic_of(it) - floor <= window]
    outside = [it for it in by_bic if bic_of(it) - floor > window]
    return sorted(inside, key=lambda it: (k_of(it), bic_of(it))) + outside


@dataclass
class RankEntry:
    """One fit's place in a ranking."""

    name: str
    bic: Optional[float]
    delta_bic: Optional[float]
    n_constants: int
    r2: Optional[float]
    eligible: bool
    reason: str
    fit: Any = field(repr=False, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bic": self.bic,
            "delta_bic": self.delta_bic,
            "n_constants": self.n_constants,
            "r2": self.r2,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass
class Ranking:
    """Every fit in selection order (eligible ones first, in Occam order), the winner, and why.

    `best` is None when no fit is eligible: "none of these laws fits" is an answer."""

    entries: List[RankEntry]
    best: Optional[RankEntry]
    note: str
    confident: bool
    occam_window: float = OCCAM_WINDOW
    min_r2: float = MIN_R2
    overdispersion: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best": self.best.name if self.best else None,
            "confident": self.confident,
            "note": self.note,
            "occam_window": self.occam_window,
            "min_r2": self.min_r2,
            "overdispersion": self.overdispersion,
            "entries": [e.to_dict() for e in self.entries],
        }


def bic_from(neg2loglik: float, n: int, k: int) -> float:
    """BIC from -2 ln L (up to a shared constant), the sample size and the constants."""
    return float(neg2loglik + k * math.log(max(n, 2)))


def _overdispersion(fits: Sequence[Any]) -> float:
    """The shared factor c (module docstring): 1 unless every fit has known, inflatable errors."""
    if not fits or not all(getattr(f, "inflate", False) and getattr(f, "likelihood", "gaussian")
                           == "gaussian" for f in fits):
        return 1.0
    reduced = [f.rss / f.dof for f in fits if f.success and f.dof > 0 and _finite(f.rss)]
    return max(1.0, min(reduced)) if reduced else 1.0


def _free_of_x(fit: Any) -> bool:
    expr = getattr(fit, "param_expression", None)
    try:
        return expr is not None and not any(str(s) == "x" for s in expr.free_symbols)
    except AttributeError:
        return False


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def rank(
    fits: Sequence[Any],
    n: Optional[int] = None,
    occam_window: float = OCCAM_WINDOW,
    min_r2: float = MIN_R2,
) -> Ranking:
    """
    Rank fitted laws (kepler.fitting.Fit) of the same data by BIC with an Occam window.

    `n` is the number of data points; it defaults to the fits' own and must match it.
    Raises ValueError when the fits are not of the same data (different n, space or
    weighting), since their BICs would not be comparable.
    """
    fits = list(fits)
    ns = {f.n for f in fits}
    if len(ns) > 1:
        raise ValueError(f"fits of different data cannot be ranked together (n = {sorted(ns)})")
    if n is not None and ns and ns != {n}:
        raise ValueError(f"n={n} but the fits have n={ns.pop()}")
    if len({f.space for f in fits}) > 1 or len({f.weighted for f in fits}) > 1:
        raise ValueError("fits in different spaces or with different weighting cannot be ranked together")
    bases = {(getattr(f, "likelihood", "gaussian"), bool(f.weighted and f.absolute_sigma),
              tuple(getattr(f, "variables", ("x",)))) for f in fits}
    if len(bases) > 1:
        raise ValueError("fits with different likelihoods, variables or error models (absolute vs "
                         "relative sigma) cannot be ranked together")
    c = _overdispersion(fits)

    entries: List[RankEntry] = []
    for f in fits:
        r2 = f.r2_fit if _finite(f.r2_fit) else None
        score = f.bic
        if c > 1.0 and _finite(f.bic):
            score = bic_from(f.neg2loglik / c, f.n_obs or f.n, f.n_constants)
        entry = RankEntry(name=f.name, bic=score if _finite(score) else None, delta_bic=None,
                          n_constants=f.n_constants, r2=r2, eligible=False, reason="", fit=f)
        if not f.success:
            entry.reason = f.message or "fit failed"
        elif entry.bic is None:
            entry.reason = "no finite fit"
        elif f.ill_conditioned:
            entry.reason = "ill-conditioned fit (cancelling terms or unidentifiable constants)"
        elif r2 is None or r2 < -R2_TOLERANCE:
            entry.reason = "diverged fit (worse than the mean, r2 < 0)"
        else:
            entry.eligible = True
        entries.append(entry)

    scored = [e for e in entries if e.bic is not None]
    if scored:
        floor = min(e.bic for e in scored)
        for e in scored:
            e.delta_bic = float(e.bic - floor)

    eligible = occam_order([e for e in entries if e.eligible], occam_window,
                           lambda e: e.bic, lambda e: e.n_constants)
    rest = sorted((e for e in entries if not e.eligible),
                  key=lambda e: e.bic if e.bic is not None else float("inf"))
    ordered = eligible + rest

    if not eligible:
        reasons = "; ".join(f"{e.name}: {e.reason}" for e in entries) or "no fits"
        return Ranking(ordered, None, f"no law fits ({reasons})", False, occam_window, min_r2, c)

    best = eligible[0]
    lowest = min(e.bic for e in eligible)
    trend_free = _free_of_x(best.fit)
    confident = trend_free or (best.r2 is not None and best.r2 >= min_r2)
    note = f"best fit by BIC: {best.name} (r2={best.r2:.4f})"
    if trend_free:
        note += "; no dependence on x earns its extra constants"
    if best.bic > lowest:
        note += f"; chosen over a richer law within the {occam_window:g}-point Occam window"
    if not confident:
        note = f"low confidence: {note}, below r2 = {min_r2:g}"
    if c > 1.0:
        note += f"; scatter {c:.2g}x the stated errors, so BIC differences are divided by {c:.2g}"
    return Ranking(ordered, best, note, confident, occam_window, min_r2, c)

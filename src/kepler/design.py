"""
design.py - Can an experiment tell two laws apart, before it is run?

    separation(truth, rival, x, sigma=None, relative=None, space="linear") -> Separation

`truth` is the law the data are supposed to follow, with its constants (a Fit,
or a law name and a dict of constants); `rival` is another law. The design is
the x values to measure and each measurement's standard error (`sigma` in y's
units, or `relative`, a fraction of y).

Expected separation ("Asimov" data). The rival is fitted to the truth's own
noise-free curve with those errors. Its chi-square there, lambda, is the
expected excess chi-square the rival pays on real data from the truth, so the
expected BIC difference is

    E[BIC(rival) - BIC(truth)] = lambda + (k_rival - k_truth) ln n

For the settling-time prototype: a power law 27 L^0.35 against a logarithm,
10 sizes from 8 to 128, 10% error per size, gives 1.7; the decisive margin the
discovery loop uses is 6.

Seeds. Halving a standard error takes four times the seeds and multiplies lambda
by four, so the factor by which every point's seed count must grow for the
expected difference to reach `margin` is

    seed_factor = (margin - (k_rival - k_truth) ln n) / lambda

(<= 1: the design already suffices; None: the rival can follow the truth
exactly, and no number of seeds at these x values separates them).

Simulation. With simulations > 0, data are drawn from the truth with Gaussian
errors (in log space when space="log"), both laws are fitted, and the share of
draws in which the truth wins by at least `margin` (p_decisive) or the rival
does (p_wrong) is reported. Seeded: the same call gives the same numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import sympy

from kepler.expression import X
from kepler.fitting import Fit, fit
from kepler.laws import LAWS, Law, get_law


@dataclass
class Separation:
    truth: str
    rival: str
    n: int
    space: str
    expected_delta_chi2: float  # the rival's chi-square on the truth's noise-free curve
    expected_delta_bic: float  # E[BIC(rival) - BIC(truth)]
    margin: float
    seed_factor: Optional[float]  # times the seeds needed for the expectation to reach margin
    p_decisive: Optional[float] = None  # simulated: truth ahead by >= margin
    p_wrong: Optional[float] = None  # simulated: rival ahead by >= margin
    simulations: int = 0
    delta_bic_quantiles: List[float] = field(default_factory=list)  # 10th, 50th, 90th

    @property
    def sufficient(self) -> bool:
        return self.expected_delta_bic >= self.margin

    def to_dict(self) -> Dict[str, Any]:
        def clean(v):
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return v

        return {k: clean(v) for k, v in {
            "truth": self.truth,
            "rival": self.rival,
            "n": self.n,
            "space": self.space,
            "expected_delta_chi2": self.expected_delta_chi2,
            "expected_delta_bic": self.expected_delta_bic,
            "margin": self.margin,
            "seed_factor": self.seed_factor,
            "sufficient": self.sufficient,
            "p_decisive": self.p_decisive,
            "p_wrong": self.p_wrong,
            "simulations": self.simulations,
            "delta_bic_quantiles": self.delta_bic_quantiles,
        }.items()}


def _curve(truth: Union[Fit, str, Law], params: Optional[Mapping[str, float]], x: np.ndarray):
    """The truth's name, number of constants and noise-free values at x."""
    if isinstance(truth, Fit):
        if not truth.success:
            raise ValueError("the truth must be a successful fit")
        values = dict(zip(truth.param_names, truth.params))
        expr, symbols = truth.param_expression, truth.param_symbols
        name = truth.name
    else:
        law = truth if isinstance(truth, Law) else get_law(truth)
        if params is None:
            raise ValueError("give the truth's constants (truth_params) with a law name")
        missing = [p for p in law.params if p not in params]
        if missing:
            raise ValueError(f"truth_params lacks {missing}")
        expr, symbols = law.parametrized(X)
        values = {p: float(params[p]) for p in law.params}
        name = law.name
    f = sympy.lambdify([X, *symbols], expr, "numpy")
    with np.errstate(all="ignore"):
        y = np.broadcast_to(np.asarray(f(x, *[values[str(s)] for s in symbols]), dtype=float),
                            x.shape).copy()
    if not np.all(np.isfinite(y)):
        raise ValueError("the truth is not finite at every x")
    return name, len(symbols), y


def _fit(law: Union[str, Law], x, y, sigma, space) -> Fit:
    return fit(law, x, y, sigma=sigma, space=space, absolute_sigma=True)


def separation(
    truth: Union[Fit, str, Law],
    rival: Union[str, Law],
    x: Sequence[float],
    sigma: Optional[Union[float, Sequence[float]]] = None,
    relative: Optional[Union[float, Sequence[float]]] = None,
    space: str = "linear",
    *,
    truth_params: Optional[Mapping[str, float]] = None,
    margin: float = 6.0,
    simulations: int = 0,
    seed: int = 0,
) -> Separation:
    """How well a design separates `truth` from `rival` (see the module docstring)."""
    x_arr = np.asarray(x, dtype=float).ravel()
    n = len(x_arr)
    if n < 2:
        raise ValueError("a design needs at least two x values")
    if (sigma is None) == (relative is None):
        raise ValueError("give exactly one of sigma (y units) or relative (a fraction of y)")
    name, k_truth, y0 = _curve(truth, truth_params, x_arr)
    if space == "log" and np.any(y0 <= 0):
        raise ValueError("a log-space design needs the truth > 0 at every x")
    if sigma is not None:
        s = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)).copy()
    else:
        s = np.broadcast_to(np.asarray(relative, dtype=float), (n,)) * np.abs(y0)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("every standard error must be positive and finite")

    rival_law = rival if isinstance(rival, Law) else get_law(rival)
    k_rival = len(rival_law.params)
    penalty = (k_rival - k_truth) * math.log(n)
    asimov = _fit(rival_law, x_arr, y0, s, space)
    lam = float(asimov.rss) if asimov.success else math.inf
    expected = lam + penalty
    if lam <= 1e-9:
        factor = None if penalty < margin else 0.0
    else:
        factor = max(0.0, (margin - penalty) / lam)

    result = Separation(name, rival_law.name, n, space, lam, expected, margin, factor)
    if simulations > 0:
        rng = np.random.default_rng(seed)
        truth_law: Union[str, Law] = truth.name if isinstance(truth, Fit) else truth
        if isinstance(truth, Fit) and truth.name not in LAWS:
            raise ValueError("simulation needs the truth to be a named law")
        deltas = []
        for _ in range(simulations):
            if space == "log":
                rel = s / np.abs(y0)
                y = y0 * np.exp(rel * rng.standard_normal(n))
            else:
                y = y0 + s * rng.standard_normal(n)
            a = _fit(truth_law, x_arr, y, s, space)
            b = _fit(rival_law, x_arr, y, s, space)
            if a.success and b.success:
                deltas.append(b.bic - a.bic)
        if deltas:
            d = np.asarray(deltas)
            result.p_decisive = float(np.mean(d >= margin))
            result.p_wrong = float(np.mean(d <= -margin))
            result.simulations = len(d)
            result.delta_bic_quantiles = [float(q) for q in np.percentile(d, [10, 50, 90])]
    return result

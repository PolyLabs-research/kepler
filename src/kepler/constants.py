"""
constants.py - Is a fitted constant a known number? Closed forms and named constants within the
error bar, the simplest first, each with the chance that a match as simple and as close would
turn up anyway.

    identify(value, stderr, k=2.0) -> list[ConstantMatch]
    identify_fit(fit) -> {constant: [ConstantMatch, ...]}

Candidates:
    closed forms   rationals p/q (q <= 24), and p/q times or divided by pi, e, sqrt 2, sqrt 3,
                   sqrt 5, the golden ratio, ln 2 or ln 3, and square roots of rationals. A
                   form's complexity is its length in bits: log2 p + log2 q, plus 4 for a named
                   irrational and 1 for a division or a root.
    named          KNOWN: critical exponents and thresholds of the universality classes a
                   lattice rule might belong to (2D percolation, 2D Ising, directed percolation
                   in 1+1 and 2+1 dimensions, KPZ, coarsening, mean field), percolation
                   thresholds of the square, triangular and honeycomb lattices, and the shape
                   constants of the Gumbel law that the largest of many light-tailed times
                   follows (extreme-value statistics). Numerical ones
                   carry their own uncertainty, which widens the window. Complexity: log2 of
                   the table's size, plus 2.

A candidate matches when it lies within k combined standard errors of the value.

Chance. Whether a match means anything depends on how much was tried. For a match of complexity
C, p_chance is the share of [|v|/2, 2|v|] (the scale the value could have had) covered by the
+- k sigma windows of every candidate no more complex than C: the chance that a value with the
same error bar, falling anywhere at that scale, would have matched something at least as simple.
A small p_chance (below 0.01, say) says the match is unlikely to be luck; one near 1 says the
error bar is too wide for any match to mean anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PHI = (1 + math.sqrt(5)) / 2
IRRATIONALS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "sqrt(2)": math.sqrt(2),
    "sqrt(3)": math.sqrt(3),
    "sqrt(5)": math.sqrt(5),
    "phi": _PHI,
    "log(2)": math.log(2),
    "log(3)": math.log(3),
}
MAX_DENOMINATOR = 24
MAX_NUMERATOR = 10_000

# name -> (value, its own standard error: 0 for exact)
KNOWN: Dict[str, Tuple[float, float]] = {
    # 2D percolation (exact)
    "2D percolation nu = 4/3": (4 / 3, 0.0),
    "2D percolation beta = 5/36": (5 / 36, 0.0),
    "2D percolation gamma = 43/18": (43 / 18, 0.0),
    "2D percolation fractal dimension = 91/48": (91 / 48, 0.0),
    "2D percolation cluster-size exponent tau = 187/91": (187 / 91, 0.0),
    # 2D Ising (exact)
    "2D Ising beta = 1/8": (1 / 8, 0.0),
    "2D Ising gamma = 7/4": (7 / 4, 0.0),
    "2D Ising eta = 1/4": (1 / 4, 0.0),
    "2D Ising delta = 15": (15.0, 0.0),
    # Directed percolation, 1+1 dimensions (Jensen 1999)
    "DP(1+1) beta": (0.276486, 0.000008),
    "DP(1+1) nu_perp": (1.096854, 0.000004),
    "DP(1+1) nu_par": (1.733847, 0.000006),
    "DP(1+1) z": (1.580745, 0.000010),
    # Directed percolation, 2+1 dimensions (numerical; Hinrichsen 2000)
    "DP(2+1) beta": (0.5834, 0.0030),
    "DP(2+1) nu_perp": (0.7333, 0.0075),
    "DP(2+1) nu_par": (1.2950, 0.0060),
    "DP(2+1) z": (1.7660, 0.0016),
    # Mean field
    "mean-field beta = 1/2 (Ising), 1 (DP)": (0.5, 0.0),
    # KPZ in 1+1 dimensions (exact)
    "KPZ growth exponent beta = 1/3": (1 / 3, 0.0),
    "KPZ dynamic exponent z = 3/2": (1.5, 0.0),
    # Domain coarsening: L ~ t^(1/2) non-conserved, t^(1/3) conserved
    "coarsening, non-conserved: 1/2": (0.5, 0.0),
    "coarsening, conserved: 1/3": (1 / 3, 0.0),
    # Percolation thresholds
    "site percolation threshold, square": (0.59274621, 0.00000013),
    "site percolation threshold, triangular = 1/2": (0.5, 0.0),
    "site percolation threshold, honeycomb": (0.6970402, 0.0000001),
    "bond percolation threshold, square = 1/2": (0.5, 0.0),
    "bond percolation threshold, triangular = 2 sin(pi/18)": (2 * math.sin(math.pi / 18), 0.0),
    "bond percolation threshold, honeycomb = 1 - 2 sin(pi/18)": (1 - 2 * math.sin(math.pi / 18), 0.0),
    # Extreme value, Gumbel class: the largest of many independent times with exponential (or
    # lighter) tails. Per unit scale, and shape constants that do not depend on the scale.
    "Gumbel mean = Euler-Mascheroni gamma": (0.5772156649015329, 0.0),
    "Gumbel median = -ln ln 2": (-math.log(math.log(2)), 0.0),
    "Gumbel standard deviation = pi/sqrt(6)": (math.pi / math.sqrt(6), 0.0),
    "Gumbel skewness = 12 sqrt(6) zeta(3)/pi^3": (1.1395470994046488, 0.0),
    "Gumbel interquartile range = ln(ln 4 / ln(4/3))": (math.log(math.log(4) / math.log(4 / 3)), 0.0),
    "Gumbel Kelly skewness (Q90 + Q10 - 2 Q50)/(Q90 - Q10)": (
        (math.log(math.log(2) ** 2 / (math.log(10) * math.log(10 / 9))))
        / math.log(math.log(10) / math.log(10 / 9)),
        0.0,
    ),
    # The largest of ~L^2 such times in 2D: location tau ln(L^2), so per unit of its slope in ln L
    "2D Gumbel interquartile range / (dT/d ln L)": (math.log(math.log(4) / math.log(4 / 3)) / 2, 0.0),
}


@dataclass
class ConstantMatch:
    expression: str  # "1/3", "2/3*pi", or a KNOWN name
    value: float
    z: float  # (value - measured) / combined standard error
    complexity: float  # bits
    p_chance: float  # see the module docstring
    tried: int  # candidates no more complex than this one
    named: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _bits(p: int, q: int) -> float:
    return math.log2(p) + math.log2(q) + 1.0


def _closed_forms(lo: float, hi: float) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Every closed form with a value in [lo, hi]: values, expressions, complexities."""
    values: List[float] = []
    names: List[str] = []
    costs: List[float] = []
    forms = [("", 1.0, 0.0, False)]
    for name, v in IRRATIONALS.items():
        forms.append((f"*{name}", v, 4.0, False))
        forms.append((f"/{name}", 1.0 / v, 5.0, False))
    forms.append(("sqrt", 1.0, 1.0, True))
    for q in range(1, MAX_DENOMINATOR + 1):
        for suffix, scale, cost, root in forms:
            low_r, high_r = (lo ** 2, hi ** 2) if root else (lo / scale, hi / scale)
            p_lo = max(1, math.ceil(low_r * q - 1e-12))
            p_hi = min(MAX_NUMERATOR, math.floor(high_r * q + 1e-12))
            for p in range(p_lo, p_hi + 1):
                if math.gcd(p, q) != 1:
                    continue
                r = p / q
                frac = f"{p}" if q == 1 else f"{p}/{q}"
                if root:
                    if math.isqrt(p) ** 2 == p and math.isqrt(q) ** 2 == q:
                        continue  # a rational, already listed
                    values.append(math.sqrt(r))
                    names.append(f"sqrt({frac})")
                else:
                    values.append(r * scale)
                    if suffix and p == 1 and q == 1:
                        names.append(suffix[1:] if suffix.startswith("*") else f"1{suffix}")
                    elif suffix:
                        names.append(f"({frac}){suffix}" if q != 1 else f"{frac}{suffix}")
                    else:
                        names.append(frac)
                costs.append(_bits(p, q) + cost)
    return np.array(values), names, np.array(costs)


def _coverage(centres: np.ndarray, halves: np.ndarray, lo: float, hi: float) -> float:
    """The share of [lo, hi] inside the union of the windows centre +- half."""
    if len(centres) == 0:
        return 0.0
    a = np.clip(centres - halves, lo, hi)
    b = np.clip(centres + halves, lo, hi)
    order = np.argsort(a)
    a, b = a[order], b[order]
    covered, end = 0.0, lo
    for s, t in zip(a, b):
        if t <= end:
            continue
        covered += t - max(s, end)
        end = t
    return covered / (hi - lo)


def identify(value: float, stderr: float, k: float = 2.0, named: bool = True,
             max_matches: int = 10) -> List[ConstantMatch]:
    """Known numbers within k standard errors of `value`, simplest first (module docstring)."""
    if not (math.isfinite(value) and math.isfinite(stderr)) or stderr < 0:
        raise ValueError("value and stderr must be finite, stderr >= 0")
    size = abs(value)
    sigma = max(stderr, 1e-15 * max(size, 1.0))
    if size <= k * sigma:
        return [ConstantMatch("0", 0.0, -value / sigma, 0.0, 1.0, 1, False)]
    sign = "-" if value < 0 else ""
    lo, hi = size / 2.0, 2.0 * size
    vals, exprs, costs = _closed_forms(lo / 1.5, hi * 1.5)
    halves = np.full(len(vals), k * sigma)
    flags = [False] * len(vals)
    if named:
        bits = math.log2(len(KNOWN)) + 2.0
        known = [(n, v, e) for n, (v, e) in KNOWN.items() if lo / 1.5 <= v <= hi * 1.5]
        vals = np.concatenate([vals, [v for _, v, _ in known]])
        exprs = exprs + [n for n, _, _ in known]
        costs = np.concatenate([costs, [bits] * len(known)])
        halves = np.concatenate([halves, [k * math.hypot(sigma, e) for _, _, e in known]])
        flags = flags + [True] * len(known)

    hits = np.flatnonzero(np.abs(vals - size) <= halves)
    order = sorted(hits, key=lambda i: (costs[i], abs(vals[i] - size)))
    matches: List[ConstantMatch] = []
    for i in order[:max_matches]:
        simpler = costs <= costs[i] + 1e-12
        p = _coverage(vals[simpler], halves[simpler], lo, hi)
        z = (vals[i] - size) / (halves[i] / k)
        expr = exprs[i] if not sign else (f"-{exprs[i]}" if not flags[i] else f"-({exprs[i]})")
        matches.append(ConstantMatch(expr, float(math.copysign(vals[i], value)), float(z if not sign else -z),
                                     float(costs[i]), float(p), int(np.sum(simpler)), flags[i]))
    return matches


def identify_fit(fit, k: float = 2.0, names: Optional[Sequence[str]] = None) -> Dict[str, List[ConstantMatch]]:
    """identify() for each constant of a fit (kepler.Fit or GroupFit), by name."""
    names = list(names) if names is not None else list(fit.param_names)
    out = {}
    for name in names:
        i = list(fit.param_names).index(name)
        out[name] = identify(fit.params[i], fit.stderr[i], k)
    return out


def as_fraction(value: float, max_denominator: int = MAX_DENOMINATOR) -> Fraction:
    """The nearest fraction with a small denominator, for printing."""
    return Fraction(value).limit_denominator(max_denominator)

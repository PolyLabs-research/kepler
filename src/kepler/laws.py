"""
laws.py - A library of named law shapes, each with its parameters and a starting guess.

A law is a formula in x with named parameters (`a*x**b`), not Float literals:
SymPy rewrites literals as it builds an expression, and a zero literal vanishes
(Float(0.0) + x becomes x), silently removing a free parameter. Named symbols
are never rewritten. `Law.skeleton()` gives the Float-literal form for tools
that take one (refine_expression, the fit_skeleton MCP tool); every starting
constant is non-zero so the skeleton keeps all of them.

Each law guesses its starting constants from the data by a linearised fit
(log-log for a power law, log-linear for an exponential, a scan over the
exponent for a shifted power law), which is what makes least squares land on
the right minimum with 10 to 20 points.

    constant          a
    logarithmic       a*log(x) + b                 x > 0
    linear            a*x + b
    power             a*x**b                       x > 0
    exponential       a*exp(b*x)                   exponential in L
    exponential_area  a*exp(b*x**2)                exponential in the area L^2
    logistic          c/(1 + exp(-k*(x - x0)))     survival against density
    shifted_power     c + a*x**b                   x > 0; finite-size scaling,
                                                   rho*(L) = rho_inf + a L^-b
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy

from kepler.expression import X

ArrayPair = Tuple[np.ndarray, np.ndarray]


def _nonzero(value: float, scale: float = 1.0) -> float:
    """`value`, moved off zero (and off non-finite values) so it survives as a constant."""
    scale = abs(float(scale)) if np.isfinite(scale) and abs(scale) > 0 else 1.0
    if not np.isfinite(value):
        return scale
    if abs(value) < 1e-9 * max(scale, 1.0):
        return 1e-3 * scale
    return float(value)


def _line(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """(slope, intercept) of a least-squares line, (0, mean) when u does not vary."""
    if len(u) < 2 or np.ptp(u) == 0:
        return 0.0, float(np.mean(v)) if len(v) else 0.0
    slope, intercept = np.polyfit(u, v, 1)
    return float(slope), float(intercept)


def _scale(y: np.ndarray) -> float:
    s = float(np.max(np.abs(y))) if len(y) else 1.0
    return s if s > 0 else 1.0


def _guess_constant(x, y):
    return [float(np.mean(y))]


def _guess_logarithmic(x, y):
    slope, intercept = _line(np.log(x), y)
    return [slope, intercept]


def _guess_linear(x, y):
    slope, intercept = _line(x, y)
    return [slope, intercept]


def _guess_power(x, y):
    if np.all(y > 0):
        b, log_a = _line(np.log(x), np.log(y))
        return [math.exp(log_a), b]
    return [float(np.mean(y)) or 1.0, 0.5]


def _guess_exponential(x, y, area: bool = False):
    u = x ** 2 if area else x
    if np.all(y > 0):
        b, log_a = _line(u, np.log(y))
        return [math.exp(log_a), b]
    span = float(np.ptp(u)) or 1.0
    return [float(np.mean(y)) or 1.0, 1.0 / span]


def _guess_logistic(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    c = float(np.max(ys)) if np.max(ys) > 0 else 1.0
    span = float(np.ptp(xs)) or 1.0
    trend = np.sign(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 and np.ptp(ys) > 0 else 1.0
    k = (trend or 1.0) * 4.0 / span
    half = c / 2.0
    x0 = float(np.median(xs))
    for i in range(len(xs) - 1):
        a, b = ys[i] - half, ys[i + 1] - half
        if a == 0:
            x0 = float(xs[i])
            break
        if a * b < 0:
            x0 = float(xs[i] + (xs[i + 1] - xs[i]) * a / (a - b))
            break
    return [c, k, x0]


def _guess_shifted_power(x, y):
    best = None
    for b in np.concatenate([-np.geomspace(0.05, 4.0, 40), np.geomspace(0.05, 4.0, 40)]):
        design = np.column_stack([np.ones_like(x), x ** b])
        if not np.all(np.isfinite(design)):
            continue
        sol, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        resid = float(np.sum((design @ sol - y) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(sol[0]), float(sol[1]), float(b))
    if best is None:
        return [float(np.mean(y)) or 1.0, 1.0, -1.0]
    _, c, a, b = best
    return [c, a, b]


@dataclass(frozen=True)
class Law:
    """A named law shape: `formula` in x and `params`, a starting guess, and its domain."""

    name: str
    formula: str
    params: Tuple[str, ...]
    description: str
    guess: Callable[[np.ndarray, np.ndarray], Sequence[float]]
    positive_x: bool = False

    def parametrized(self, x_symbol: sympy.Symbol = X) -> Tuple[sympy.Expr, List[sympy.Symbol]]:
        """The formula as a SymPy expression in `x_symbol` and its parameter symbols."""
        symbols = [sympy.Symbol(p) for p in self.params]
        local: Dict[str, sympy.Basic] = {p: s for p, s in zip(self.params, symbols)}
        local["x"] = x_symbol
        return sympy.sympify(self.formula, locals=local), symbols

    def domain_problem(self, x: np.ndarray) -> Optional[str]:
        """Why this law cannot be fitted at these x values, or None."""
        if self.positive_x and np.any(np.asarray(x) <= 0):
            return f"{self.name} needs every x > 0"
        return None

    def initial(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> List[float]:
        """Starting constants guessed from the data, each finite and non-zero. Without data,
        all ones."""
        if x is None or y is None:
            return [1.0] * len(self.params)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        try:
            with np.errstate(all="ignore"):
                values = [float(v) for v in self.guess(x, y)]
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError, OverflowError):
            values = [1.0] * len(self.params)
        if len(values) != len(self.params):
            raise RuntimeError(f"{self.name}: guess gave {len(values)} values for {len(self.params)} parameters")
        scale = _scale(y)
        return [_nonzero(v, scale if i == 0 else 1.0) for i, v in enumerate(values)]

    def skeleton(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> str:
        """The law with its starting constants as Float literals, e.g. '27.0*x**0.35'."""
        from kepler.refine import parametrize

        expr, symbols = self.parametrized()
        replacements = {s: sympy.Float(v) for s, v in zip(symbols, self.initial(x, y))}
        # SymPy's own form reads best; keep it only if it still has every constant (it can
        # merge or drop literals), else write the formula exactly as given.
        text = str(expr.xreplace(replacements))
        if len(parametrize(sympy.sympify(text))[1]) == len(symbols):
            return text
        with sympy.evaluate(False):
            return str(expr.xreplace(replacements))


LAWS: Dict[str, Law] = {law.name: law for law in (
    Law("constant", "a", ("a",), "y does not depend on x", _guess_constant),
    Law("logarithmic", "a*log(x) + b", ("a", "b"), "grows like log x", _guess_logarithmic, positive_x=True),
    Law("linear", "a*x + b", ("a", "b"), "straight line", _guess_linear),
    Law("power", "a*x**b", ("a", "b"), "power law", _guess_power, positive_x=True),
    Law("exponential", "a*exp(b*x)", ("a", "b"), "exponential in x", _guess_exponential),
    Law("exponential_area", "a*exp(b*x**2)", ("a", "b"), "exponential in x^2 (the area of an x by x torus)",
        lambda x, y: _guess_exponential(x, y, area=True)),
    Law("logistic", "c/(1 + exp(-k*(x - x0)))", ("c", "k", "x0"),
        "logistic step of height c at x0 (a survival curve)", _guess_logistic),
    Law("shifted_power", "c + a*x**b", ("c", "a", "b"),
        "power-law approach to a limit (finite-size scaling)", _guess_shifted_power, positive_x=True),
)}

# The six shapes the settling-time prototype compared (2026-10-16).
GROWTH_LAWS: Tuple[str, ...] = ("constant", "logarithmic", "linear", "power", "exponential", "exponential_area")


def get_law(name: str) -> Law:
    try:
        return LAWS[name]
    except KeyError:
        raise ValueError(f"unknown law {name!r}; known: {', '.join(LAWS)}") from None

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

Settling over time (x the generation; x >= 0):

    relaxation            c + a*exp(-x/tau)
    stretched_exponential c + a*exp(-(x/tau)**beta)
    algebraic_relaxation  c + a*(1 + x/tau)**(-b)

Probabilities, fitted with likelihood="binomial" (survived out of seeds):

    logistic_p        1/(1 + exp(-k*(x - x0)))     a step from 0 to 1 at x0
    window            logistic up at x1 times      too sparse to start, too dense
                      logistic down at x2          to last

And against density rho and torus size L at once (x is a dict of arrays):

    crossover         1/(1 + exp(-w*(rho - r)))                the same at every size
    sharpening        1/(1 + exp(-w*L**s*(rho - r)))           sharpens, fixed threshold
    fss_logistic      1/(1 + exp(-w*L**s*(rho - r - a*L**-b))) and the threshold drifts:
                                                               rho*(L) = r + a L^-b
"""
from __future__ import annotations

import functools
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


def _crossings(x: np.ndarray, y: np.ndarray, level: float) -> List[float]:
    """Where y, sorted by x, crosses `level` (linear interpolation)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    out = []
    for i in range(len(xs) - 1):
        a, b = ys[i] - level, ys[i + 1] - level
        if a == 0:
            out.append(float(xs[i]))
        elif a * b < 0:
            out.append(float(xs[i] + (xs[i + 1] - xs[i]) * a / (a - b)))
    return out


def _guess_logistic_p(x, y):
    span = float(np.ptp(x)) or 1.0
    trend = np.sign(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.ptp(y) > 0 else 1.0
    found = _crossings(x, y, 0.5)
    x0 = found[0] if found else float(np.median(x))
    return [(trend or 1.0) * 8.0 / span, x0]


def _guess_window(x, y):
    span = float(np.ptp(x)) or 1.0
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    peak = int(np.argmax(ys))
    half = max(float(ys[peak]) / 2.0, 1e-3)
    left = _crossings(xs[: peak + 1], ys[: peak + 1], half)
    right = _crossings(xs[peak:], ys[peak:], half)
    x1 = left[-1] if left else float(xs[0] + span / 4)
    x2 = right[0] if right else float(xs[-1] - span / 4)
    if x2 <= x1:
        x1, x2 = float(xs[0] + span / 4), float(xs[-1] - span / 4)
    k = 8.0 / span
    return [k, x1, k, x2]


def _guess_fss_logistic(v, y):
    """Per size, where the probability crosses 1/2; then rho*(L) = r + a L^-1 through them."""
    rho, L = np.asarray(v["rho"], dtype=float), np.asarray(v["L"], dtype=float)
    sizes = np.unique(L)
    centres, slopes = [], []
    for size in sizes:
        m = L == size
        found = _crossings(rho[m], y[m], 0.5)
        if found:
            centres.append((size, found[0]))
        span = float(np.ptp(rho[m])) or 1.0
        slopes.append(8.0 / span)
    trend = np.sign(np.corrcoef(rho, y)[0, 1]) if np.ptp(y) > 0 and np.ptp(rho) > 0 else 1.0
    if len(centres) >= 2:
        sz = np.array([c[0] for c in centres])
        cx = np.array([c[1] for c in centres])
        design = np.column_stack([np.ones_like(sz), 1.0 / sz])
        (r, a), *_ = np.linalg.lstsq(design, cx, rcond=None)
    elif centres:
        r, a = centres[0][1], 0.1
    else:
        r, a = float(np.median(rho)), 0.1
    s = 0.5
    w = (trend or 1.0) * float(np.median(slopes)) / float(np.max(L)) ** s
    return [r, a, 1.0, w, s]


def _decay_guess(x, y, level: float):
    """(c, a, t) for a curve settling from y[0] towards c: c is the mean of the last fifth, a the
    initial gap, t when the gap has shrunk to `level` of it (a third of the span if never)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    tail = max(1, len(ys) // 5)
    c = float(np.mean(ys[-tail:]))
    gap = ys - c
    a = float(gap[0]) if gap[0] != 0 else float(np.max(np.abs(gap))) or 1.0
    span = float(np.ptp(xs)) or 1.0
    t = span / 3.0
    shrunk = np.flatnonzero(np.abs(gap) <= level * abs(a))
    if len(shrunk) and xs[shrunk[0]] > xs[0]:
        t = float(xs[shrunk[0]] - xs[0])
    return c, a, max(t, span * 1e-3), float(xs[0])


def _guess_relaxation(x, y):
    c, a, tau, x0 = _decay_guess(x, y, math.exp(-1.0))
    return [c, a * math.exp(min(x0 / tau, 50.0)), tau]


def _guess_stretched(x, y):
    c, a, tau, _ = _decay_guess(x, y, math.exp(-1.0))
    return [c, a, tau, 1.0]


def _guess_algebraic(x, y):
    c, a, half, _ = _decay_guess(x, y, 0.5)
    return [c, a, half, 1.0]


def _guess_crossover(v, y):
    r, a, _, w, s = _guess_fss_logistic(v, y)
    typical = float(np.median(np.asarray(v["L"], dtype=float)))
    return [r + a / typical, w * typical ** s]


def _guess_sharpening(v, y):
    r, a, _, w, s = _guess_fss_logistic(v, y)
    typical = float(np.median(np.asarray(v["L"], dtype=float)))
    return [r + a / typical, w, s]


@functools.lru_cache(maxsize=256)
def _parse_formula(formula: str, params: Tuple[str, ...], variables: Tuple[str, ...],
                   x_symbol: sympy.Symbol):
    symbols = tuple(sympy.Symbol(p) for p in params)
    local: Dict[str, sympy.Basic] = {p: s for p, s in zip(params, symbols)}
    for v in variables:
        local[v] = x_symbol if variables == ("x",) else sympy.Symbol(v)
    return sympy.sympify(formula, locals=local), symbols


@dataclass(frozen=True)
class Law:
    """A named law shape: `formula` in its `variables` (x unless named) and `params`, a
    starting guess, and its domain."""

    name: str
    formula: str
    params: Tuple[str, ...]
    description: str
    guess: Callable[..., Sequence[float]]
    positive_x: bool = False
    variables: Tuple[str, ...] = ("x",)
    positive_vars: Tuple[str, ...] = ()

    def parametrized(self, x_symbol: sympy.Symbol = X) -> Tuple[sympy.Expr, List[sympy.Symbol]]:
        """The formula as a SymPy expression and its parameter symbols. A one-variable law is
        written in `x_symbol`; a law of several variables in symbols named after them."""
        expr, symbols = _parse_formula(self.formula, self.params, self.variables, x_symbol)
        return expr, list(symbols)

    def variable_symbols(self, x_symbol: sympy.Symbol = X) -> Tuple[sympy.Symbol, ...]:
        if self.variables == ("x",):
            return (x_symbol,)
        return tuple(sympy.Symbol(v) for v in self.variables)

    def domain_problem(self, x) -> Optional[str]:
        """Why this law cannot be fitted at these x values (an array, or a dict of arrays for a
        law of several variables), or None."""
        if isinstance(x, dict):
            for v in self.positive_vars:
                if np.any(np.asarray(x[v]) <= 0):
                    return f"{self.name} needs every {v} > 0"
            return None
        if self.positive_x and np.any(np.asarray(x) <= 0):
            return f"{self.name} needs every x > 0"
        return None

    def initial(self, x=None, y: Optional[np.ndarray] = None) -> List[float]:
        """Starting constants guessed from the data, each finite and non-zero. Without data,
        all ones. `x` is an array, or a dict of arrays for a law of several variables."""
        if x is None or y is None:
            return [1.0] * len(self.params)
        if isinstance(x, dict):
            x = {k: np.asarray(v, dtype=np.float64) for k, v in x.items()}
        elif self.variables != ("x",):
            raise ValueError(f"{self.name}'s variables are {', '.join(self.variables)}: pass x as a dict")
        else:
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
    # Approach to a steady state over time (density, damage): x is the generation, x >= 0.
    Law("relaxation", "c + a*exp(-x/tau)", ("c", "a", "tau"),
        "exponential approach to the level c with time constant tau", _guess_relaxation),
    Law("stretched_exponential", "c + a*exp(-(x/tau)**beta)", ("c", "a", "tau", "beta"),
        "stretched (beta < 1) or compressed exponential approach to c", _guess_stretched),
    Law("algebraic_relaxation", "c + a*(1 + x/tau)**(-b)", ("c", "a", "tau", "b"),
        "power-law approach to c, as in coarsening", _guess_algebraic),
    # Probabilities (fit with likelihood="binomial").
    Law("logistic_p", "1/(1 + exp(-k*(x - x0)))", ("k", "x0"),
        "a probability stepping from 0 to 1 (k > 0) or 1 to 0 (k < 0) at x0", _guess_logistic_p),
    Law("window", "1/((1 + exp(-k1*(x - x1)))*(1 + exp(k2*(x - x2))))", ("k1", "x1", "k2", "x2"),
        "a probability that rises at x1 and falls at x2: too sparse to start, too dense to last",
        _guess_window),
    # Two variables: survival over density rho and torus size L, one fit for every size. From
    # no transition (the same curve at every size) to a threshold that sharpens and drifts.
    Law("crossover", "1/(1 + exp(-w*(rho - r)))", ("r", "w"),
        "the same step at every size: a crossover, not a transition",
        _guess_crossover, variables=("rho", "L"), positive_vars=("L",)),
    Law("sharpening", "1/(1 + exp(-w*L**s*(rho - r)))", ("r", "w", "s"),
        "a step at a fixed threshold r that sharpens as L^s",
        _guess_sharpening, variables=("rho", "L"), positive_vars=("L",)),
    Law("fss_logistic", "1/(1 + exp(-w*L**s*(rho - r - a*L**(-b))))", ("r", "a", "b", "w", "s"),
        "finite-size scaling of a threshold: rho*(L) = r + a L^-b, sharpening as L^s",
        _guess_fss_logistic, variables=("rho", "L"), positive_vars=("L",)),
)}

# The six shapes the settling-time prototype compared (2026-10-16).
GROWTH_LAWS: Tuple[str, ...] = ("constant", "logarithmic", "linear", "power", "exponential", "exponential_area")
# Shapes for a quantity settling over time.
TIME_LAWS: Tuple[str, ...] = ("constant", "relaxation", "stretched_exponential", "algebraic_relaxation")
# Shapes for a probability against one setting.
PROBABILITY_LAWS: Tuple[str, ...] = ("constant", "logistic_p", "window")
# A probability against density rho and size L: is there a transition, and where is it at L -> oo?
FSS_LAWS: Tuple[str, ...] = ("crossover", "sharpening", "fss_logistic")


def get_law(name: str) -> Law:
    try:
        return LAWS[name]
    except KeyError:
        raise ValueError(f"unknown law {name!r}; known: {', '.join(LAWS)}") from None

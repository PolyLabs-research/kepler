"""
refine.py - Fit the constants of a predicted equation to the observed data.

The specialists are good at structure and mediocre at digits. Refinement keeps
the predicted structure, treats every Float literal in it as a free parameter
initialised at the predicted value, and fits those parameters by least squares
against the data. Integer and rational literals (x**2, x**(1/2), the 1 in a
logistic) are structural and stay fixed.

Three entry points, one per data shape:

    refine_expression(expr, x, y)   algebraic curves  -> RefineResult
    refine_ar(y, phi)               AR(p) processes   -> dict
    refine_ode(params, t, y)        linear ODEs       -> dict

Every path reports RMSE before and after, R^2 after, and whether the fitted
constants were accepted (they are only accepted when they reduce RMSE).

Degenerate fits
---------------
A template plus a free offset can imitate another family through cancelling
terms: a*x**b + c with b -> 0 and a ~ -c ~ 1e3 is a logarithm in disguise and
posts a respectable R^2. Such fits are detected by the cancellation ratio (the
largest top-level term relative to the signal) and flagged `ill_conditioned`;
refine_expression then prefers the well-conditioned variant.

Weights and log space
---------------------
refine_expression and refine_parametrized take an optional `sigma` per point
(residuals are divided by it, so a point measured with more seeds counts for
more) and `space="log"`, which fits log(f(x)) to log(y) for data spanning
decades (sigma then becomes sigma/y, the delta-method error of log y). RMSE
and R^2 are always reported unweighted in linear space; `fit_rms` is what the
optimiser minimised.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy
from scipy.integrate import odeint
from scipy.optimize import least_squares

from kepler.expression import X

_PENALTY = 1e6
ILL_CONDITIONED_RATIO = 10.0     # top-level term rms / signal rms (legit sums stay well below)
ILL_CONDITIONED_COND = 1e12      # Jacobian condition number at the solution
AR_STATIONARY_MARGIN = 0.99      # max characteristic-root modulus for a stationary AR


@dataclass
class RefineResult:
    expression: sympy.Expr
    parameters: List[float] = field(default_factory=list)
    rmse_initial: float = float("nan")
    rmse: float = float("nan")
    r2: float = float("nan")
    improved: bool = False
    message: str = ""
    # Diagnostics
    n_constants: int = 0
    max_abs_parameter: float = float("nan")
    condition_number: float = float("nan")
    cancellation_ratio: float = float("nan")
    ill_conditioned: bool = False
    variant: str = "template"
    note: str = ""
    # The fitted model with named parameters: `param_expression` in x and `param_symbols`,
    # whose values are `parameters`, in the same order. kepler.fitting builds error bars and
    # prediction bands from these.
    param_expression: Optional[sympy.Expr] = None
    param_symbols: List[sympy.Symbol] = field(default_factory=list)
    # What least squares minimised: residuals in `space` ("linear" or "log"), each divided by
    # its sigma when `weighted`. `fit_rms` is the root mean square of those residuals; with
    # no sigma in linear space it equals `rmse`.
    space: str = "linear"
    weighted: bool = False
    fit_rms: float = float("nan")

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def fit_dict(self) -> Dict[str, object]:
        return {
            "rmse_initial": _as_float(self.rmse_initial),
            "rmse": _as_float(self.rmse),
            "r2": _as_float(self.r2),
            "n_parameters": self.n_parameters,
            "n_constants": int(self.n_constants),
            "improved": bool(self.improved),
            "ill_conditioned": bool(self.ill_conditioned),
            "cancellation_ratio": _as_float(self.cancellation_ratio),
            "condition_number": _as_float(self.condition_number),
            "max_abs_parameter": _as_float(self.max_abs_parameter),
            "variant": self.variant,
            "message": self.message,
            "note": self.note,
        }


@functools.lru_cache(maxsize=1024)
def compile_numpy(args: Tuple[sympy.Symbol, ...], expr) -> object:
    """sympy.lambdify(args, expr) for numpy, cached: compiling dominates the cost of a small
    fit, and a loop fits the same few templates thousands of times. `expr` may be a tuple of
    expressions (the function then returns a list)."""
    return sympy.lambdify(list(args), list(expr) if isinstance(expr, tuple) else expr, modules="numpy")


def _as_float(v: float) -> Optional[float]:
    v = float(v)
    return v if np.isfinite(v) else None


# =============================================================================
# Metrics and diagnostics
# =============================================================================

def _finite_pred(pred, n: int) -> np.ndarray:
    pred = np.asarray(pred)
    if np.iscomplexobj(pred):
        pred = np.where(np.abs(pred.imag) < 1e-9, pred.real, np.nan)
    pred = np.broadcast_to(pred.astype(np.float64, copy=False), (n,))
    return pred


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    pred = _finite_pred(pred, len(y))
    if not np.all(np.isfinite(pred)):
        return float("inf")
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def r_squared(pred: np.ndarray, y: np.ndarray) -> float:
    pred = _finite_pred(pred, len(y))
    if not np.all(np.isfinite(pred)):
        return float("-inf")
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-300:
        return 1.0 if ss_res <= 1e-300 else 0.0
    return 1.0 - ss_res / ss_tot


def count_constants(expr: sympy.Expr) -> int:
    """Number of distinct Float literals in an expression (its fitted constants)."""
    return len(expr.atoms(sympy.Float))


def cancellation_ratio(expr: sympy.Expr, x: np.ndarray, y: np.ndarray, x_symbol: sympy.Symbol = X) -> float:
    """
    Largest top-level term magnitude relative to the signal magnitude.

    Well-conditioned sums have terms comparable to the signal (ratio ~ 1).
    A ratio far above one means large terms cancel to produce the fit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    signal = float(np.sqrt(np.mean(y ** 2)))
    if not np.isfinite(signal) or signal < 1e-12:
        return float("nan")
    worst = 0.0
    for term in sympy.Add.make_args(expr):
        try:
            with np.errstate(all="ignore"):
                vals = _finite_pred(compile_numpy((x_symbol,), term)(x), len(y))
        except Exception:
            return float("nan")
        if not np.all(np.isfinite(vals)):
            return float("inf")
        worst = max(worst, float(np.sqrt(np.mean(vals ** 2))))
    return worst / signal


def _condition_number(jac) -> float:
    try:
        jac = np.asarray(jac, dtype=np.float64)
        if jac.size == 0 or not np.all(np.isfinite(jac)):
            return float("nan")
        return float(np.linalg.cond(jac))
    except Exception:
        return float("nan")


# =============================================================================
# Algebraic expressions
# =============================================================================

def parametrize(expr: sympy.Expr) -> Tuple[sympy.Expr, List[sympy.Symbol], List[float]]:
    """
    Replace fittable numeric literals with symbols c0, c1, ...

    Returns (parametrised expression, symbols, initial values).
    """
    symbols: List[sympy.Symbol] = []
    values: List[float] = []

    def new_param(value: float) -> sympy.Symbol:
        sym = sympy.Symbol(f"c{len(symbols)}")
        symbols.append(sym)
        values.append(float(value))
        return sym

    def rebuild(e: sympy.Basic) -> sympy.Basic:
        if e.is_Number:
            # Integer and rational literals are structure: x**2, 1/(...), 1 + exp(...).
            # Every coefficient the specialists emit is parsed as a Float, so Floats
            # are the fittable constants (a power-law exponent included).
            if not e.is_Float or not np.isfinite(float(e)):
                return e
            return new_param(float(e))
        if e.is_Atom:
            return e
        return e.func(*[rebuild(a) for a in e.args], evaluate=False)

    return rebuild(expr), symbols, values


def _needs_scale(expr: sympy.Expr) -> bool:
    """True when the template has no free multiplicative constant of its own."""
    if expr.is_Float:
        return False
    if isinstance(expr, sympy.Mul):
        return not any(a.is_Float for a in expr.args)
    if isinstance(expr, sympy.Add):
        return not all(t.as_coeff_Mul()[0].is_Float for t in expr.args)
    return True


def _needs_offset(expr: sympy.Expr) -> bool:
    """True when the template has no free additive constant of its own."""
    if expr.is_Float:
        return False
    if isinstance(expr, sympy.Add):
        return not any(t.is_Float for t in expr.args)
    return True


def _substitute(param_expr: sympy.Expr, symbols: Sequence[sympy.Symbol], values: Sequence[float]) -> sympy.Expr:
    expr = param_expr.xreplace({s: sympy.Float(v) for s, v in zip(symbols, values)})
    # Re-evaluate the tree now that constants are back in place.
    return sympy.sympify(str(expr))


SPACES = ("linear", "log")


def _check_space(space: str, y: np.ndarray) -> None:
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES} (got {space!r})")
    if space == "log" and not np.all(y > 0):
        raise ValueError("a log-space fit needs every y > 0")


def _point_weights(sigma, y: np.ndarray, space: str) -> np.ndarray:
    """1/sigma per point, in the fit space (ones when sigma is None)."""
    n = len(y)
    if sigma is None:
        return np.ones(n)
    s = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n,)).astype(np.float64)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma must be positive and finite for every point")
    if space == "log":
        s = s / y  # delta method: sd(log y) ~ sd(y) / y
    return 1.0 / s


def _fit_parametrized(
    expr: sympy.Expr,
    param_expr: sympy.Expr,
    symbols: Sequence[sympy.Symbol],
    p0: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    x_symbol: sympy.Symbol,
    max_nfev: int,
    bound: Optional[float],
    variant: str,
    sigma=None,
    space: str = "linear",
) -> RefineResult:
    """Least-squares fit of `param_expr`'s `symbols`, starting from `p0`; `expr` is the
    unfitted expression reported when the fit fails or does not improve."""
    n = len(y)
    symbols, p0 = list(symbols), [float(v) for v in p0]
    weights = _point_weights(sigma, y, space)
    with np.errstate(all="ignore"):
        target = np.log(y) if space == "log" else y
    base = dict(variant=variant, param_expression=param_expr, param_symbols=symbols,
                space=space, weighted=sigma is not None)

    try:
        func = compile_numpy((x_symbol, *symbols), param_expr)
    except Exception as exc:  # pragma: no cover - defensive
        return RefineResult(expression=expr, message=f"could not compile: {exc}", **base)

    def predict(params: Sequence[float]) -> np.ndarray:
        with np.errstate(all="ignore"):
            try:
                return _finite_pred(func(x, *params), n)
            except Exception:
                return np.full(n, np.nan)

    def fit_residuals(params: Sequence[float]) -> np.ndarray:
        pred = predict(params)
        if space == "log":
            with np.errstate(all="ignore"):
                pred = np.where(pred > 0, np.log(np.where(pred > 0, pred, 1.0)), np.nan)
        return (pred - target) * weights

    def fit_rms(params: Sequence[float]) -> float:
        res = fit_residuals(params)
        if not np.all(np.isfinite(res)):
            return float("inf")
        return float(np.sqrt(np.mean(res ** 2)))

    def finish(expression, params, rmse0, rmse1, r2, improved, message, jac=None) -> RefineResult:
        res = RefineResult(expression=expression, parameters=[float(v) for v in params],
                           rmse_initial=rmse0, rmse=rmse1, r2=r2, improved=improved,
                           message=message, **base)
        res.fit_rms = fit_rms(params)
        res.n_constants = count_constants(expression)
        res.max_abs_parameter = float(np.max(np.abs(params))) if len(params) else float("nan")
        res.condition_number = _condition_number(jac) if jac is not None else float("nan")
        res.cancellation_ratio = cancellation_ratio(expression, x, y, x_symbol)
        at_bound = bound is not None and len(params) and res.max_abs_parameter >= bound * (1 - 1e-6)
        res.ill_conditioned = bool(
            (np.isfinite(res.cancellation_ratio) and res.cancellation_ratio > ILL_CONDITIONED_RATIO)
            or res.cancellation_ratio == float("inf")
            or at_bound
            or (np.isfinite(res.condition_number) and res.condition_number > ILL_CONDITIONED_COND)
        )
        return res

    pred0 = predict(p0)
    rmse0 = rmse(pred0, y)
    r2_0 = r_squared(pred0, y)
    cost0 = fit_rms(p0)

    if not symbols:
        return finish(expr, p0, rmse0, rmse0, r2_0, False, "no free constants")
    if cost0 == 0.0:
        return finish(expr, p0, rmse0, rmse0, r2_0, False, "already exact")

    def residuals(params: np.ndarray) -> np.ndarray:
        res = fit_residuals(params)
        return np.where(np.isfinite(res), np.clip(res, -_PENALTY, _PENALTY), _PENALTY)

    p_start = np.asarray(p0, dtype=np.float64)
    kwargs = {}
    if bound is not None:
        kwargs["bounds"] = (-bound, bound)
        p_start = np.clip(p_start, -bound * (1 - 1e-9), bound * (1 - 1e-9))

    try:
        sol = least_squares(residuals, p_start, method="trf", x_scale="jac", max_nfev=max_nfev, **kwargs)
    except Exception as exc:
        return finish(expr, p0, rmse0, rmse0, r2_0, False, f"optimiser failed: {exc}")

    cost1 = fit_rms(sol.x)
    if not np.isfinite(cost1) or cost1 >= cost0 * (1 - 1e-9):
        return finish(expr, p0, rmse0, rmse0, r2_0, False, "fit did not improve on predicted constants")

    try:
        refined = _substitute(param_expr, symbols, sol.x)
        if refined.is_polynomial(x_symbol):
            refined = sympy.expand(refined)
    except Exception as exc:
        return finish(expr, p0, rmse0, rmse0, r2_0, False, f"could not substitute fitted constants: {exc}")

    pred1 = predict(sol.x)
    return finish(refined, sol.x, rmse0, rmse(pred1, y), r_squared(pred1, y), True,
                  "constants refitted by least squares", jac=sol.jac)


def _refine_variant(
    expr: sympy.Expr,
    x: np.ndarray,
    y: np.ndarray,
    x_symbol: sympy.Symbol,
    max_nfev: int,
    wrap: bool,
    bound: Optional[float],
    sigma=None,
    space: str = "linear",
) -> Optional[RefineResult]:
    """One least-squares fit of `expr`, optionally wrapped as A*expr + B. None if the wrap adds nothing."""
    try:
        param_expr, symbols, p0 = parametrize(expr)
    except Exception as exc:  # pragma: no cover - defensive
        return RefineResult(expression=expr, message=f"could not parametrise: {exc}", variant="template")

    variant = "template"
    if wrap:
        added = False
        if _needs_scale(expr):
            scale_sym = sympy.Symbol("cA")
            param_expr = scale_sym * param_expr
            symbols, p0, added = [*symbols, scale_sym], [*p0, 1.0], True
        if _needs_offset(expr):
            offset_sym = sympy.Symbol("cB")
            param_expr = param_expr + offset_sym
            symbols, p0, added = [*symbols, offset_sym], [*p0, 0.0], True
        if not added:
            return None
        variant = "affine"

    return _fit_parametrized(expr, param_expr, symbols, p0, x, y, x_symbol, max_nfev, bound,
                             variant, sigma, space)


def refine_expression(
    expr: sympy.Expr,
    x: np.ndarray,
    y: np.ndarray,
    x_symbol: sympy.Symbol = X,
    max_nfev: int = 2000,
    affine_wrapper: bool = False,
    bound: Optional[float] = None,
    sigma=None,
    space: str = "linear",
) -> RefineResult:
    """
    Fit all Float literals of `expr` to (x, y) by least squares.

    With affine_wrapper=True a second fit of A*expr + B is run (A and B are
    only added where the template lacks its own scale or offset) and the
    better well-conditioned fit is kept; an ill-conditioned variant is
    reported in `note` and never returned while a sound alternative exists.

    `bound` limits every fitted constant to [-bound, bound]. Use it only in
    normalised coordinates, where legitimate constants are small; it is a
    guard against runaway values, not the degeneracy check (that is the
    cancellation ratio).

    `sigma` (one value, or one per point) weights the residuals; `space="log"`
    fits log(expr) to log(y). See the module docstring.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _check_space(space, y)

    results: List[RefineResult] = []
    for wrap in ([False, True] if affine_wrapper else [False]):
        res = _refine_variant(expr, x, y, x_symbol, max_nfev, wrap, bound, sigma, space)
        if res is not None:
            results.append(res)

    sound = [r for r in results if not r.ill_conditioned]
    pool = sound or results
    best = min(pool, key=lambda r: r.fit_rms if np.isfinite(r.fit_rms) else float("inf"))
    others = [r for r in results if r is not best]
    if others:
        best.note = "; ".join(
            f"{r.variant} variant {'discarded as ill-conditioned' if r.ill_conditioned else 'not better'}"
            f" (r2={r.r2:.4f}, rmse={r.rmse:.3g})" for r in others
        )
    return best


def refine_parametrized(
    param_expr: sympy.Expr,
    symbols: Sequence[sympy.Symbol],
    p0: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    x_symbol: sympy.Symbol = X,
    sigma=None,
    space: str = "linear",
    max_nfev: int = 2000,
    bound: Optional[float] = None,
) -> RefineResult:
    """
    Fit a model written with named parameters, `param_expr` in `x_symbol` and
    `symbols`, starting from `p0`.

    The same least squares as refine_expression, for templates whose constants
    must keep their names and their places: SymPy rewrites Float literals as it
    builds an expression (2.0*(x - 0.5) becomes 2.0*x - 1.0, and 0.0 + x
    becomes x, losing a free constant), but never a Symbol.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _check_space(space, y)
    symbols = list(symbols)
    if len(symbols) != len(p0):
        raise ValueError(f"{len(symbols)} parameters but {len(p0)} starting values")
    start = param_expr.xreplace({s: sympy.Float(float(v)) for s, v in zip(symbols, p0)})
    return _fit_parametrized(start, param_expr, symbols, p0, x, y, x_symbol, max_nfev, bound,
                             "template", sigma, space)


# =============================================================================
# AR processes
# =============================================================================

def ar_predict(y_centered: np.ndarray, phi: Sequence[float]) -> np.ndarray:
    """One-step-ahead prediction of an AR(p) process from lagged values."""
    p = len(phi)
    n = len(y_centered)
    pred = np.zeros(n - p)
    for lag, coef in enumerate(phi, start=1):
        pred += coef * y_centered[p - lag:n - lag]
    return pred


def ar_stationarity(phi: Sequence[float], margin: float = AR_STATIONARY_MARGIN) -> Tuple[bool, float]:
    """(is_stationary, max root modulus) for y[t] = sum phi_i y[t-i]."""
    if not len(phi):
        return True, 0.0
    roots = np.roots([1.0] + [-float(c) for c in phi])
    modulus = float(np.max(np.abs(roots))) if len(roots) else 0.0
    return bool(modulus < margin), modulus


def refine_ar(y: np.ndarray, phi_initial: Sequence[float]) -> Dict[str, object]:
    """Refit AR coefficients by ordinary least squares, keeping the predicted order."""
    y = np.asarray(y, dtype=np.float64)
    p = len(phi_initial)
    mean = float(np.mean(y))
    yc = y - mean
    n = len(yc)
    stationary, modulus = ar_stationarity(phi_initial)
    out: Dict[str, object] = {
        "coefficients": [float(c) for c in phi_initial],
        "order": p,
        "mean": mean,
        "improved": False,
        "rmse_initial": None, "rmse": None, "r2": None,
        "n_parameters": p, "n_constants": p,
        "stationary": stationary, "max_root_modulus": modulus,
        "ill_conditioned": False,
        "message": "",
    }
    if n <= p + 1:
        out["message"] = "too few samples to refit"
        return out

    target = yc[p:]
    pred0 = ar_predict(yc, phi_initial)
    rmse0 = rmse(pred0, target)

    lags = np.column_stack([yc[p - lag:n - lag] for lag in range(1, p + 1)])
    phi, *_ = np.linalg.lstsq(lags, target, rcond=None)
    pred1 = ar_predict(yc, phi)
    rmse1 = rmse(pred1, target)

    out["rmse_initial"] = _as_float(rmse0)
    if np.isfinite(rmse1) and rmse1 < rmse0 * (1 - 1e-9):
        stationary, modulus = ar_stationarity(phi)
        out.update(coefficients=[float(c) for c in phi], improved=True,
                   rmse=_as_float(rmse1), r2=_as_float(r_squared(pred1, target)),
                   stationary=stationary, max_root_modulus=modulus,
                   message="coefficients refitted by least squares")
    else:
        out.update(rmse=_as_float(rmse0), r2=_as_float(r_squared(pred0, target)),
                   message="fit did not improve on predicted coefficients")
    return out


# =============================================================================
# Linear ODEs
# =============================================================================

def simulate_linear_ode(params: Dict[str, float], t: np.ndarray, y0: float, v0: float = 0.0) -> np.ndarray:
    """Simulate dx/dt = a*x or x'' + C1*x' + C2*x = 0 on the time grid t."""
    t = np.asarray(t, dtype=np.float64)
    if params["order"] == 1:
        with np.errstate(over="ignore"):
            return y0 * np.exp(params["a"] * (t - t[0]))
    C1, C2 = params["C1"], params["C2"]

    def system(state, _t):
        x, v = state
        return [v, -C1 * v - C2 * x]

    with np.errstate(all="ignore"):
        sol = odeint(system, [y0, v0], t)
    return sol[:, 0]


def refine_ode(
    params: Dict[str, float],
    t: np.ndarray,
    y: np.ndarray,
    max_nfev: int = 400,
    bound: Optional[float] = None,
) -> Dict[str, object]:
    """Refit ODE coefficients and initial conditions to the observed trajectory."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    order = params["order"]
    # The trajectory fit spends every parameter, initial conditions included:
    # A + B*exp(-c*t) needs three constants as a closed form but four here
    # (C1, C2, y0, v0), and model comparison must count all of them.
    n_params = 2 if order == 1 else 4
    out: Dict[str, object] = {**params, "improved": False, "rmse_initial": None,
                              "rmse": None, "r2": None, "condition_number": None,
                              "ill_conditioned": False,
                              "n_parameters": n_params, "n_constants": n_params,
                              "message": ""}
    if n < 4 or np.any(np.diff(t) <= 0):
        out["message"] = "time axis must be strictly increasing with at least 4 samples"
        return out

    y0 = float(y[0])
    v0 = float(np.gradient(y, t)[0]) if order == 2 else 0.0
    if order == 1:
        p0 = np.array([params["a"], y0])
        unpack = lambda p: ({"order": 1, "a": float(p[0])}, float(p[1]), 0.0)
    else:
        p0 = np.array([params["C1"], params["C2"], y0, v0])
        unpack = lambda p: ({"order": 2, "C1": float(p[0]), "C2": float(p[1])}, float(p[2]), float(p[3]))

    def predict(p: np.ndarray) -> np.ndarray:
        prm, y_init, v_init = unpack(p)
        try:
            return _finite_pred(simulate_linear_ode(prm, t, y_init, v_init), n)
        except Exception:
            return np.full(n, np.nan)

    def residuals(p: np.ndarray) -> np.ndarray:
        res = predict(p) - y
        return np.where(np.isfinite(res), np.clip(res, -_PENALTY, _PENALTY), _PENALTY)

    pred0 = predict(p0)
    rmse0 = rmse(pred0, y)
    out["rmse_initial"] = _as_float(rmse0)
    out["initial_conditions"] = {"y0": y0, "v0": v0} if order == 2 else {"y0": y0}

    kwargs = {}
    p_start = p0
    if bound is not None:
        kwargs["bounds"] = (-bound, bound)
        p_start = np.clip(p0, -bound * (1 - 1e-9), bound * (1 - 1e-9))

    try:
        sol = least_squares(residuals, p_start, method="trf", x_scale="jac", max_nfev=max_nfev, **kwargs)
    except Exception as exc:
        out.update(rmse=_as_float(rmse0), r2=_as_float(r_squared(pred0, y)),
                   message=f"optimiser failed: {exc}")
        return out

    pred1 = predict(sol.x)
    rmse1 = rmse(pred1, y)
    if np.isfinite(rmse1) and rmse1 < rmse0 * (1 - 1e-9):
        prm, y_init, v_init = unpack(sol.x)
        cond = _condition_number(sol.jac)
        out.update(prm, improved=True, rmse=_as_float(rmse1), r2=_as_float(r_squared(pred1, y)),
                   condition_number=_as_float(cond),
                   ill_conditioned=bool(np.isfinite(cond) and cond > ILL_CONDITIONED_COND),
                   message="coefficients refitted against simulated trajectory")
        out["initial_conditions"] = {"y0": y_init, "v0": v_init} if order == 2 else {"y0": y_init}
    else:
        out.update(rmse=_as_float(rmse0), r2=_as_float(r_squared(pred0, y)),
                   message="fit did not improve on predicted coefficients")
    return out

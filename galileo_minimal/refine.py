"""
refine.py - Fit the constants of a predicted equation to the observed data.

The specialists are good at structure and mediocre at digits. Refinement keeps
the predicted structure, treats every numeric literal in it as a free
parameter initialised at the predicted value, and fits those parameters by
least squares against the original data. Integer and rational exponents
(x**2, x**(1/2)) are structural and stay fixed; float exponents such as a
power-law index are fitted.

Three entry points, one per data shape:

    refine_expression(expr, x, y)   algebraic curves  -> RefineResult
    refine_ar(y, phi)               AR(p) processes   -> dict
    refine_ode(params, t, y)        linear ODEs       -> dict

Every path reports RMSE before and after, R^2 after, and whether the fitted
constants were accepted (they are only accepted when they reduce RMSE).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy
from scipy.integrate import odeint
from scipy.optimize import least_squares

from expression import X

_PENALTY = 1e6


@dataclass
class RefineResult:
    expression: sympy.Expr
    parameters: List[float] = field(default_factory=list)
    rmse_initial: float = float("nan")
    rmse: float = float("nan")
    r2: float = float("nan")
    improved: bool = False
    message: str = ""

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def fit_dict(self) -> Dict[str, object]:
        return {
            "rmse_initial": _as_float(self.rmse_initial),
            "rmse": _as_float(self.rmse),
            "r2": _as_float(self.r2),
            "n_parameters": self.n_parameters,
            "improved": bool(self.improved),
            "message": self.message,
        }


def _as_float(v: float) -> Optional[float]:
    v = float(v)
    return v if np.isfinite(v) else None


# =============================================================================
# Metrics
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

    def rebuild(e: sympy.Basic, in_exponent: bool = False) -> sympy.Basic:
        if e.is_Number:
            # Integer and rational literals are structure: x**2, 1/(...), 1 + exp(...).
            # Every coefficient the specialists emit is parsed as a Float, so Floats
            # are the fittable constants (a power-law exponent included).
            if not e.is_Float or not np.isfinite(float(e)):
                return e
            return new_param(float(e))
        if e.is_Atom:
            return e
        if isinstance(e, sympy.Pow):
            base, exponent = e.args
            return sympy.Pow(rebuild(base), rebuild(exponent, in_exponent=True), evaluate=False)
        return e.func(*[rebuild(a) for a in e.args], evaluate=False)

    return rebuild(expr), symbols, values


def _substitute(param_expr: sympy.Expr, symbols: Sequence[sympy.Symbol], values: Sequence[float]) -> sympy.Expr:
    expr = param_expr.xreplace({s: sympy.Float(v) for s, v in zip(symbols, values)})
    # Re-evaluate the tree now that constants are back in place.
    expr = sympy.sympify(str(expr))
    return expr


def refine_expression(
    expr: sympy.Expr,
    x: np.ndarray,
    y: np.ndarray,
    x_symbol: sympy.Symbol = X,
    max_nfev: int = 2000,
    affine_wrapper: bool = False,
) -> RefineResult:
    """
    Fit all numeric literals of `expr` to (x, y) by least squares.

    With affine_wrapper=True the fitted form is A*expr + B with A, B free
    (initialised to 1 and 0), which lets a structure without an offset or
    scale term still absorb one from the data.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    try:
        param_expr, symbols, p0 = parametrize(expr)
        if affine_wrapper:
            scale_sym, offset_sym = sympy.Symbol("cA"), sympy.Symbol("cB")
            param_expr = scale_sym * param_expr + offset_sym
            symbols = [*symbols, scale_sym, offset_sym]
            p0 = [*p0, 1.0, 0.0]
        func = sympy.lambdify([x_symbol, *symbols], param_expr, modules="numpy")
    except Exception as exc:  # pragma: no cover - defensive
        return RefineResult(expression=expr, message=f"could not parametrise: {exc}")

    def predict(params: Sequence[float]) -> np.ndarray:
        with np.errstate(all="ignore"):
            try:
                return _finite_pred(func(x, *params), n)
            except Exception:
                return np.full(n, np.nan)

    pred0 = predict(p0)
    rmse0 = rmse(pred0, y)
    result = RefineResult(expression=expr, parameters=list(p0), rmse_initial=rmse0,
                          rmse=rmse0, r2=r_squared(pred0, y))

    if not symbols:
        result.message = "no free constants"
        return result
    if rmse0 == 0.0:
        result.message = "already exact"
        return result

    def residuals(params: np.ndarray) -> np.ndarray:
        res = predict(params) - y
        return np.where(np.isfinite(res), np.clip(res, -_PENALTY, _PENALTY), _PENALTY)

    try:
        sol = least_squares(residuals, np.asarray(p0, dtype=np.float64), method="trf",
                            x_scale="jac", max_nfev=max_nfev)
    except Exception as exc:
        result.message = f"optimiser failed: {exc}"
        return result

    pred1 = predict(sol.x)
    rmse1 = rmse(pred1, y)
    if not np.isfinite(rmse1) or rmse1 >= rmse0 * (1 - 1e-9):
        result.message = "fit did not improve on predicted constants"
        return result

    try:
        refined = _substitute(param_expr, symbols, sol.x)
        if refined.is_polynomial(x_symbol):
            refined = sympy.expand(refined)
    except Exception as exc:
        result.message = f"could not substitute fitted constants: {exc}"
        return result

    return RefineResult(expression=refined, parameters=[float(v) for v in sol.x],
                        rmse_initial=rmse0, rmse=rmse1, r2=r_squared(pred1, y),
                        improved=True, message="constants refitted by least squares")


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


def refine_ar(y: np.ndarray, phi_initial: Sequence[float]) -> Dict[str, object]:
    """Refit AR coefficients by ordinary least squares, keeping the predicted order."""
    y = np.asarray(y, dtype=np.float64)
    p = len(phi_initial)
    mean = float(np.mean(y))
    yc = y - mean
    n = len(yc)
    out: Dict[str, object] = {
        "coefficients": [float(c) for c in phi_initial],
        "order": p,
        "mean": mean,
        "improved": False,
        "rmse_initial": None, "rmse": None, "r2": None,
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
        out.update(coefficients=[float(c) for c in phi], improved=True,
                   rmse=_as_float(rmse1), r2=_as_float(r_squared(pred1, target)),
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


def refine_ode(params: Dict[str, float], t: np.ndarray, y: np.ndarray, max_nfev: int = 400) -> Dict[str, object]:
    """Refit ODE coefficients and initial conditions to the observed trajectory."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out: Dict[str, object] = {**params, "improved": False, "rmse_initial": None,
                              "rmse": None, "r2": None, "message": ""}
    if n < 4 or np.any(np.diff(t) <= 0):
        out["message"] = "time axis must be strictly increasing with at least 4 samples"
        return out

    order = params["order"]
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

    try:
        sol = least_squares(residuals, p0, method="trf", x_scale="jac", max_nfev=max_nfev)
    except Exception as exc:
        out.update(rmse=_as_float(rmse0), r2=_as_float(r_squared(pred0, y)),
                   message=f"optimiser failed: {exc}")
        return out

    pred1 = predict(sol.x)
    rmse1 = rmse(pred1, y)
    if np.isfinite(rmse1) and rmse1 < rmse0 * (1 - 1e-9):
        prm, y_init, v_init = unpack(sol.x)
        out.update(prm, improved=True, rmse=_as_float(rmse1), r2=_as_float(r_squared(pred1, y)),
                   message="coefficients refitted against simulated trajectory")
        out["initial_conditions"] = {"y0": y_init, "v0": v_init} if order == 2 else {"y0": y_init}
    else:
        out.update(rmse=_as_float(rmse0), r2=_as_float(r_squared(pred0, y)),
                   message="fit did not improve on predicted coefficients")
    return out

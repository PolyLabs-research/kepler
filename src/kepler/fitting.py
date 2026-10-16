"""
fitting.py - The public fit and predict calls: a law's constants with error bars,
and predictions with an uncertainty band.

    fit(skeleton, x, y, sigma=None, space="linear") -> Fit
    predict(fit, x, level=0.95) -> (y, low, high)

`skeleton` is a law from the library (kepler.laws, by name or Law), or an
equation whose Float literals are the constants to fit, Kepler's usual
convention ('2.0*x**0.5 + 1.0'; integer literals such as the 2 in x**2 are
structure and stay fixed). A Float literal that is zero is refused: SymPy would
drop it (0.0 + x is x) and with it a free constant.

The fit is refine.refine_expression / refine_parametrized (scipy's
least_squares). Standard errors come from the Jacobian at the solution, as
scipy's curve_fit computes them: cov = s^2 (J^T J)^-1 with J the Jacobian of
the weighted residuals and s^2 = RSS / (n - k), the reduced chi-square, unless
absolute_sigma=True says the sigmas are absolute (then s^2 = 1). With inflate=True as well,
s^2 = max(1, RSS / (n - k)): scatter beyond the stated errors widens the error bars (the Birge
ratio), scatter within them never narrows them. That suits sigmas that are real error
estimates on few points, where the reduced chi-square itself is noisy (with 5 degrees of
freedom it falls below 0.2 one time in 30). `Fit.error_scale` is the s^2 used.

The band from predict() is the uncertainty of the fitted curve itself, from
the constants' covariance (the delta method), with a Student-t quantile on
n - k degrees of freedom. It is not a prediction interval for one new
measurement: a caller comparing it with a held-out measurement adds that
measurement's own uncertainty.

With space="log" the fit minimises residuals of log y; its band is computed on
log f and exponentiated, so it is asymmetric and always positive.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy
from scipy import stats
from sympy.parsing.sympy_parser import parse_expr

from kepler.expression import X, expression_to_string
from kepler.laws import LAWS, Law
from kepler.ranking import bic as _bic
from kepler.refine import (
    RefineResult,
    _check_space,
    _point_weights,
    compile_numpy,
    refine_expression,
    refine_parametrized,
)

Skeleton = Union[str, sympy.Expr, Law]


@dataclass(eq=False)
class Fit:
    """A fitted law: its constants with standard errors and covariance, and fit statistics.

    `rmse` and `r2` are unweighted, in linear space, for people. `rss`, `r2_fit` and `bic`
    are in the fit space (weighted residuals, linear or log): what least squares minimised
    and what ranking compares."""

    name: str
    formula: str
    expression: sympy.Expr
    param_names: Tuple[str, ...]
    params: Tuple[float, ...]
    stderr: Tuple[float, ...]
    covariance: np.ndarray
    n: int
    rmse: float
    r2: float
    rss: float
    r2_fit: float
    bic: float
    space: str
    weighted: bool
    absolute_sigma: bool
    ill_conditioned: bool
    success: bool
    message: str
    param_expression: sympy.Expr = field(repr=False)
    param_symbols: Tuple[sympy.Symbol, ...] = field(repr=False)
    error_scale: float = float("nan")  # the s^2 the covariance was scaled by

    @property
    def n_constants(self) -> int:
        return len(self.params)

    @property
    def dof(self) -> int:
        return self.n - self.n_constants

    @property
    def equation(self) -> str:
        return expression_to_string(self.expression)

    def value(self, name: str) -> float:
        return self.params[self.param_names.index(name)]

    def error(self, name: str) -> float:
        return self.stderr[self.param_names.index(name)]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready summary (non-finite numbers become None)."""
        return _jsonable({
            "name": self.name,
            "formula": self.formula,
            "equation": self.equation,
            "params": dict(zip(self.param_names, self.params)),
            "stderr": dict(zip(self.param_names, self.stderr)),
            "covariance": self.covariance.tolist(),
            "n": self.n,
            "n_constants": self.n_constants,
            "rmse": self.rmse,
            "r2": self.r2,
            "rss": self.rss,
            "r2_fit": self.r2_fit,
            "bic": self.bic,
            "space": self.space,
            "weighted": self.weighted,
            "absolute_sigma": self.absolute_sigma,
            "error_scale": self.error_scale,
            "ill_conditioned": self.ill_conditioned,
            "success": self.success,
            "message": self.message,
        })


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# =============================================================================
# Skeletons
# =============================================================================

def parse_skeleton(skeleton: Union[str, sympy.Expr], x_symbol: sympy.Symbol = X) -> sympy.Expr:
    """An equation with Float-literal constants as a SymPy expression in x. Raises ValueError
    for other free symbols and for a zero Float literal, which SymPy would silently drop."""
    if isinstance(skeleton, sympy.Expr):
        expr = skeleton
        raw = skeleton
    else:
        text = str(skeleton)
        try:
            raw = parse_expr(text, local_dict={"x": x_symbol}, evaluate=False)
            expr = sympy.sympify(text, locals={"x": x_symbol})
        except Exception as exc:
            raise ValueError(f"cannot parse skeleton {text!r}: {exc}") from None
    extra = expr.free_symbols - {x_symbol}
    if extra:
        names = ", ".join(sorted(str(s) for s in extra))
        raise ValueError(f"skeleton has symbols other than x ({names}); write constants as numbers, e.g. 1.0")
    if any(float(f) == 0.0 for f in raw.atoms(sympy.Float)):
        raise ValueError("skeleton has a zero constant; SymPy drops it (0.0 + x becomes x), so start it "
                         "at a non-zero value")
    if not expr.atoms(sympy.Float):
        raise ValueError("skeleton has no Float constants to fit (integers such as the 2 in x**2 are structure)")
    return expr


def _resolve_law(skeleton: Skeleton) -> Optional[Law]:
    if isinstance(skeleton, Law):
        return skeleton
    if isinstance(skeleton, str) and skeleton in LAWS:
        return LAWS[skeleton]
    return None


# =============================================================================
# Fit
# =============================================================================

@functools.lru_cache(maxsize=1024)
def _derivatives(param_expr: sympy.Expr, symbols: Tuple[sympy.Symbol, ...]) -> Tuple[sympy.Expr, ...]:
    return tuple(sympy.diff(param_expr, s) for s in symbols)


def _gradient_fn(param_expr: sympy.Expr, symbols: Sequence[sympy.Symbol], x_symbol: sympy.Symbol):
    """f and df/dp as numpy functions of (x, *params), each returning float arrays of len(x)."""
    args = (x_symbol, *symbols)
    f = compile_numpy(args, param_expr)
    g = compile_numpy(args, _derivatives(param_expr, tuple(symbols)))

    def as_array(v, m: int) -> np.ndarray:
        arr = np.asarray(v)
        if np.iscomplexobj(arr):
            arr = np.where(np.abs(arr.imag) < 1e-9, arr.real, np.nan)
        return np.broadcast_to(arr.astype(np.float64), (m,)).copy()

    def value(x: np.ndarray, params: Sequence[float]) -> np.ndarray:
        with np.errstate(all="ignore"):
            return as_array(f(x, *params), len(x))

    def grad(x: np.ndarray, params: Sequence[float]) -> np.ndarray:
        with np.errstate(all="ignore"):
            cols = g(x, *params) if symbols else []
            return np.column_stack([as_array(c, len(x)) for c in cols]) if symbols else np.zeros((len(x), 0))

    return value, grad


def _covariance(jac: np.ndarray, s2: float) -> Tuple[np.ndarray, bool]:
    """s2 (J^T J)^-1 by SVD, as curve_fit does; (inf matrix, True) when J is rank-deficient."""
    k = jac.shape[1]
    if k == 0:
        return np.zeros((0, 0)), False
    if not np.all(np.isfinite(jac)) or not math.isfinite(s2):
        return np.full((k, k), np.inf), True
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * (s[0] if len(s) else 0.0)
    if len(s) < k or np.any(s <= threshold):
        return np.full((k, k), np.inf), True
    cov = (vt.T / s ** 2) @ vt
    return cov * s2, False


def _failed(name: str, formula: str, n: int, space: str, weighted: bool, absolute_sigma: bool,
            message: str, param_expr: Optional[sympy.Expr] = None,
            symbols: Sequence[sympy.Symbol] = (), p0: Sequence[float] = ()) -> Fit:
    k = len(symbols)
    expr = param_expr.xreplace({s: sympy.Float(v) for s, v in zip(symbols, p0)}) if param_expr is not None else sympy.nan
    nan = float("nan")
    return Fit(name=name, formula=formula, expression=expr, param_names=tuple(str(s) for s in symbols),
               params=tuple(float(v) for v in p0), stderr=tuple([math.inf] * k),
               covariance=np.full((k, k), np.inf), n=n, rmse=nan, r2=nan, rss=nan, r2_fit=nan, bic=nan,
               space=space, weighted=weighted, absolute_sigma=absolute_sigma, ill_conditioned=False,
               success=False, message=message,
               param_expression=param_expr if param_expr is not None else sympy.nan,
               param_symbols=tuple(symbols))


def fit(
    skeleton: Skeleton,
    x: Sequence[float],
    y: Sequence[float],
    sigma: Optional[Union[float, Sequence[float]]] = None,
    space: str = "linear",
    *,
    absolute_sigma: bool = False,
    inflate: bool = False,
    max_nfev: int = 2000,
) -> Fit:
    """
    Fit a law's constants to (x, y), with standard errors.

    Args:
        skeleton: a law name ('power'), a kepler.laws.Law, or an equation in x whose Float
            literals are the constants to fit ('27.0*x**0.35').
        sigma: the uncertainty of each y (or one for all). Residuals are divided by it, so a
            point measured with more seeds counts for more.
        space: "linear", or "log" to fit log f(x) to log y (every y must be > 0).
        absolute_sigma: treat sigma as absolute rather than relative (covariance not scaled
            by the reduced chi-square).
        inflate: with absolute_sigma, scale the covariance up by the reduced chi-square when it
            exceeds 1 (see the module docstring).

    A law whose domain excludes the data (log x at x <= 0) gives a Fit with success=False
    and the reason in `message`, so it can be ranked as ineligible. Malformed input (lengths
    differ, non-finite values, a zero constant in a skeleton, an unknown space) raises
    ValueError.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(y_arr)
    if len(x_arr) != n:
        raise ValueError(f"x has {len(x_arr)} values but y has {n}")
    if n == 0:
        raise ValueError("no data to fit")
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("x and y must be finite")
    _check_space(space, y_arr)
    weights = _point_weights(sigma, y_arr, space)  # validates sigma
    weighted = sigma is not None

    law = _resolve_law(skeleton)
    if law is not None:
        param_expr, symbols = law.parametrized(X)
        p0 = law.initial(x_arr, y_arr)
        problem = law.domain_problem(x_arr)
        if problem:
            return _failed(law.name, law.formula, n, space, weighted, absolute_sigma, problem,
                           param_expr, symbols, p0)
        res: RefineResult = refine_parametrized(param_expr, symbols, p0, x_arr, y_arr, X,
                                                sigma=sigma, space=space, max_nfev=max_nfev)
        name, formula = law.name, law.formula
    else:
        expr = parse_skeleton(skeleton)
        res = refine_expression(expr, x_arr, y_arr, sigma=sigma, space=space, max_nfev=max_nfev)
        name, formula = str(skeleton), str(res.param_expression)
        param_expr, symbols = res.param_expression, list(res.param_symbols)

    params = [float(v) for v in res.parameters]
    if param_expr is None or len(params) != len(symbols):
        return _failed(name, formula, n, space, weighted, absolute_sigma, res.message or "fit failed")

    value, grad = _gradient_fn(param_expr, symbols, X)
    f = value(x_arr, params)
    target = np.log(y_arr) if space == "log" else y_arr
    with np.errstate(all="ignore"):
        f_space = np.where(f > 0, np.log(np.where(f > 0, f, 1.0)), np.nan) if space == "log" else f
    resid = (f_space - target) * weights
    if not np.all(np.isfinite(resid)):
        out = _failed(name, formula, n, space, weighted, absolute_sigma,
                      "the fitted law is not finite at every x" + ("" if space == "linear" else " (or not positive)"),
                      param_expr, symbols, params)
        return out
    rss = float(np.sum(resid ** 2))
    w2 = weights ** 2
    centre = float(np.sum(w2 * target) / np.sum(w2))
    ss_tot = float(np.sum(w2 * (target - centre) ** 2))
    r2_fit = 1.0 - rss / ss_tot if ss_tot > 1e-300 else (1.0 if rss <= 1e-300 else 0.0)

    k = len(params)
    jac = grad(x_arr, params)
    if space == "log":
        jac = jac / f[:, None]
    jac = jac * weights[:, None]
    dof = n - k
    if weighted and absolute_sigma:
        s2 = max(1.0, rss / dof) if inflate and dof > 0 else 1.0
    else:
        s2 = rss / dof if dof > 0 else math.inf
    cov, singular = _covariance(jac, s2)
    stderr = tuple(float(math.sqrt(v)) if v >= 0 else math.nan for v in np.diag(cov))

    return Fit(
        name=name, formula=formula, expression=res.expression,
        param_names=tuple(str(s) for s in symbols), params=tuple(params), stderr=stderr,
        covariance=cov, n=n, rmse=float(res.rmse), r2=float(res.r2), rss=rss, r2_fit=float(r2_fit),
        bic=_bic(rss, n, k), space=space, weighted=weighted, absolute_sigma=absolute_sigma,
        ill_conditioned=bool(res.ill_conditioned or singular), success=True,
        message=res.message + ("; constants not identifiable from these data" if singular else ""),
        param_expression=param_expr, param_symbols=tuple(symbols), error_scale=float(s2),
    )


# =============================================================================
# Predict
# =============================================================================

def predict_covariance(fit_result: Fit, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The fitted law at `x` in its fit space (log f for a log-space fit, f otherwise) and the
    covariance matrix of those values from the constants' covariance, G C G^T (the delta
    method). Predictions at several x share the constants, so their errors are correlated;
    a joint test of held-out measurements needs the whole matrix, not only the bands.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    m = len(x_arr)
    if not fit_result.success:
        return np.full(m, np.nan), np.full((m, m), np.nan)
    value, grad = _gradient_fn(fit_result.param_expression, fit_result.param_symbols, X)
    params = list(fit_result.params)
    f = value(x_arr, params)
    g = grad(x_arr, params)
    with np.errstate(all="ignore"):
        if fit_result.space == "log":
            g = g / f[:, None]
            f = np.log(f)
        cov = g @ fit_result.covariance @ g.T if fit_result.covariance.size else np.zeros((m, m))
    return f, cov


def predict(fit_result: Fit, x: Union[float, Sequence[float]], level: float = 0.95):
    """
    The fitted law at `x` with a `level` confidence band from the constants' covariance:
    returns (y, low, high), arrays for an array x and floats for a scalar x.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1 (got {level})")
    scalar = np.ndim(x) == 0
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if not fit_result.success:
        nan = np.full(len(x_arr), np.nan)
        return (float(nan[0]), float(nan[0]), float(nan[0])) if scalar else (nan, nan.copy(), nan.copy())

    value, grad = _gradient_fn(fit_result.param_expression, fit_result.param_symbols, X)
    params = list(fit_result.params)
    f = value(x_arr, params)
    g = grad(x_arr, params)
    log_space = fit_result.space == "log"
    if log_space:
        with np.errstate(all="ignore"):
            g = g / f[:, None]
    cov = fit_result.covariance
    with np.errstate(all="ignore"):
        var = np.einsum("ij,jk,ik->i", g, cov, g) if cov.size else np.zeros(len(x_arr))
        sd = np.sqrt(np.maximum(var, 0.0))
    if fit_result.dof > 0 and not (fit_result.weighted and fit_result.absolute_sigma):
        q = float(stats.t.ppf(0.5 + level / 2.0, fit_result.dof))
    else:
        q = float(stats.norm.ppf(0.5 + level / 2.0))
    with np.errstate(all="ignore"):
        if log_space:
            centre = np.log(f)
            low, high = np.exp(centre - q * sd), np.exp(centre + q * sd)
        else:
            low, high = f - q * sd, f + q * sd
    if scalar:
        return float(f[0]), float(low[0]), float(high[0])
    return f, low, high

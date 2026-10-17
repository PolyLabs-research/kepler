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

Model comparison uses -2 ln L up to a constant shared by every law fitted to the same data
(`Fit.neg2loglik`): with absolute sigmas the errors are known and it is the chi-square; otherwise
the error size is unknown and it is n ln(RSS / n). `bic` adds k ln n and `aicc` the small-sample
AIC penalty 2k + 2k(k+1)/(n-k-1). kepler.rank refuses to compare fits whose -2 ln L have
different bases.

The band from predict() is the uncertainty of the fitted curve itself, from
the constants' covariance (the delta method), with a Student-t quantile on
n - k degrees of freedom. It is not a prediction interval for one new
measurement: a caller comparing it with a held-out measurement adds that
measurement's own uncertainty.

With space="log" the fit minimises residuals of log y; its band is computed on
log f and exponentiated, so it is asymmetric and always positive.

Other likelihoods. likelihood="binomial" (y successes out of `trials`), "poisson" (y counts
over `exposure`) and "censored" (y times, log-normal about the law's median, where `censored`
marks a lower bound) fit by maximum likelihood (kepler.mle) instead of least squares. Their
-2 ln L is exact, so their BIC needs no error estimate; `rss` is the deviance (2 ln L of a
model with a free value per row, minus the law's), a chi-square-like lack of fit on n - k
degrees of freedom; a binomial BIC counts every trial. Bands are built on the logit scale for
a probability and the log scale for a rate or a time.

Several variables. `x` may be a dict of arrays, one per variable of the law ("rho" and "L" for
kepler.laws' fss_logistic), and predict() then takes a dict too. A skeleton string in several
variables names them as in the dict; its Float literals are the constants.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy
from scipy import stats
from scipy.optimize import least_squares
from sympy.parsing.sympy_parser import parse_expr

from kepler.expression import X, expression_to_string
from kepler.laws import LAWS, Law
from kepler.likelihoods import Binomial, Likelihood, make
from kepler.mle import Model, maximise
from kepler.refine import (
    RefineResult,
    _check_space,
    _point_weights,
    compile_numpy,
    parametrize,
    r_squared,
    refine_expression,
    refine_parametrized,
)

Skeleton = Union[str, sympy.Expr, Law]
XData = Union[Sequence[float], np.ndarray, Mapping[str, Sequence[float]]]


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
    inflate: bool = False
    # -2 ln L up to a constant shared by every law fitted to the same data, the sample size its
    # BIC penalty counts, and the small-sample AIC.
    neg2loglik: float = float("nan")
    n_obs: int = 0
    aicc: float = float("nan")
    likelihood: str = "gaussian"
    variables: Tuple[str, ...] = ("x",)
    # The scale prediction bands are built on (identity, log or logit), and the likelihood's
    # own fitted constants (a log-normal's log spread), which BIC counts with the law's.
    link: str = "identity"
    nuisance: Dict[str, float] = field(default_factory=dict)
    nuisance_stderr: Dict[str, float] = field(default_factory=dict)

    @property
    def n_constants(self) -> int:
        return len(self.params)

    @property
    def n_fitted(self) -> int:
        """Every constant the fit estimated, the likelihood's nuisance constants included."""
        return len(self.params) + len(self.nuisance)

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
            "inflate": self.inflate,
            "neg2loglik": self.neg2loglik,
            "n_obs": self.n_obs,
            "aicc": self.aicc,
            "likelihood": self.likelihood,
            "variables": list(self.variables),
            "link": self.link,
            "nuisance": self.nuisance,
            "nuisance_stderr": self.nuisance_stderr,
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

def _variable_symbols(variables: Sequence[str], x_symbol: sympy.Symbol = X) -> Tuple[sympy.Symbol, ...]:
    return tuple(x_symbol if v == "x" else sympy.Symbol(v) for v in variables)


def parse_skeleton(skeleton: Union[str, sympy.Expr], x_symbol: sympy.Symbol = X,
                   variables: Sequence[str] = ("x",)) -> sympy.Expr:
    """An equation with Float-literal constants as a SymPy expression in its `variables` (x
    unless named). Raises ValueError for other free symbols and for a zero Float literal,
    which SymPy would silently drop."""
    symbols = _variable_symbols(variables, x_symbol)
    local = {v: s for v, s in zip(variables, symbols)}
    if isinstance(skeleton, sympy.Expr):
        expr = skeleton
        raw = skeleton
    else:
        text = str(skeleton)
        try:
            raw = parse_expr(text, local_dict=dict(local), evaluate=False)
            expr = sympy.sympify(text, locals=dict(local))
        except Exception as exc:
            raise ValueError(f"cannot parse skeleton {text!r}: {exc}") from None
    extra = expr.free_symbols - set(symbols)
    if extra:
        names = ", ".join(sorted(str(s) for s in extra))
        allowed = " and ".join(variables)
        raise ValueError(f"skeleton has symbols other than {allowed} ({names}); write constants as numbers, "
                         "e.g. 1.0")
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


def _gradient_fn(param_expr: sympy.Expr, symbols: Sequence[sympy.Symbol], variables: Sequence[sympy.Symbol]):
    """f and df/dp as numpy functions of (columns, params), `columns` holding one array per
    variable, all of one length m; each returns float arrays of m rows."""
    args = (*variables, *symbols)
    f = compile_numpy(args, param_expr)
    g = compile_numpy(args, _derivatives(param_expr, tuple(symbols)))

    def as_array(v, m: int) -> np.ndarray:
        arr = np.asarray(v)
        if np.iscomplexobj(arr):
            arr = np.where(np.abs(arr.imag) < 1e-9, arr.real, np.nan)
        return np.broadcast_to(arr.astype(np.float64), (m,)).copy()

    def value(cols: Sequence[np.ndarray], params: Sequence[float]) -> np.ndarray:
        m = len(cols[0])
        with np.errstate(all="ignore"):
            return as_array(f(*cols, *params), m)

    def grad(cols: Sequence[np.ndarray], params: Sequence[float]) -> np.ndarray:
        m = len(cols[0])
        if not symbols:
            return np.zeros((m, 0))
        with np.errstate(all="ignore"):
            out = np.column_stack([as_array(c, m) for c in g(*cols, *params)])
        bad = ~np.isfinite(out)
        if np.any(bad):
            # A derivative such as x**b * log(x) is 0 * -inf at x = 0, though its limit is finite:
            # where the law itself is finite, difference it numerically instead.
            rows = np.flatnonzero(np.any(bad, axis=1))
            sub = [c[rows] if np.ndim(c) else c for c in cols]
            if np.all(np.isfinite(value(sub, params))):
                for j in np.flatnonzero(np.any(bad, axis=0)):
                    h = 1e-6 * max(abs(float(params[j])), 1e-3)
                    up, down = list(params), list(params)
                    up[j] += h
                    down[j] -= h
                    fd = (value(sub, up) - value(sub, down)) / (2.0 * h)
                    col = out[rows, j]
                    out[rows, j] = np.where(np.isfinite(col), col, fd)
        return out

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


def information(neg2loglik: float, n: int, k: int) -> Tuple[float, float]:
    """(BIC, AICc) from -2 ln L, the sample size and the number of fitted constants."""
    bic = float(neg2loglik + k * math.log(max(n, 2)))
    aicc = float(neg2loglik + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf))
    return bic, aicc


def _failed(name: str, formula: str, n: int, space: str, weighted: bool, absolute_sigma: bool,
            message: str, param_expr: Optional[sympy.Expr] = None,
            symbols: Sequence[sympy.Symbol] = (), p0: Sequence[float] = (),
            likelihood: str = "gaussian", variables: Tuple[str, ...] = ("x",), link: str = "identity") -> Fit:
    k = len(symbols)
    expr = _with_values(param_expr, symbols, p0) if param_expr is not None else sympy.nan
    nan = float("nan")
    return Fit(name=name, formula=formula, expression=expr, param_names=tuple(str(s) for s in symbols),
               params=tuple(float(v) for v in p0), stderr=tuple([math.inf] * k),
               covariance=np.full((k, k), np.inf), n=n, rmse=nan, r2=nan, rss=nan, r2_fit=nan, bic=nan,
               space=space, weighted=weighted, absolute_sigma=absolute_sigma, ill_conditioned=False,
               success=False, message=message,
               param_expression=param_expr if param_expr is not None else sympy.nan,
               param_symbols=tuple(symbols), n_obs=n, likelihood=likelihood, variables=variables, link=link)


def _columns(x: XData, variables: Sequence[str], n: int) -> List[np.ndarray]:
    """One float array of n values per variable, from an array (a law in x alone) or a dict."""
    if isinstance(x, Mapping):
        missing = [v for v in variables if v not in x]
        if missing:
            raise ValueError(f"x lacks {', '.join(missing)} (the law's variables are {', '.join(variables)})")
        cols = [np.asarray(x[v], dtype=np.float64).ravel() for v in variables]
    else:
        if tuple(variables) != ("x",):
            raise ValueError(f"this law's variables are {', '.join(variables)}: pass x as a dict of arrays")
        cols = [np.asarray(x, dtype=np.float64).ravel()]
    for v, c in zip(variables, cols):
        if len(c) != n:
            label = "x" if tuple(variables) == ("x",) else f"x[{v!r}]"
            raise ValueError(f"{label} has {len(c)} values but y has {n}")
        if not np.all(np.isfinite(c)):
            raise ValueError("x and y must be finite")
    return cols


def _template(skeleton: Skeleton, law: Optional[Law], variables: Tuple[str, ...], cols: List[np.ndarray],
              y_guess: np.ndarray, p0: Optional[Sequence[float]]):
    """(name, formula, parametrised expression, its constants, starting values, domain problem)."""
    if law is not None:
        param_expr, symbols = law.parametrized(X)
        xdata = cols[0] if variables == ("x",) else dict(zip(variables, cols))
        start = law.initial(xdata, y_guess) if p0 is None else [float(v) for v in p0]
        name, formula, problem = law.name, law.formula, law.domain_problem(xdata)
    else:
        expr = parse_skeleton(skeleton, variables=variables)
        param_expr, symbols, values = parametrize(expr)
        start = values if p0 is None else [float(v) for v in p0]
        name, formula, problem = str(skeleton), str(param_expr), None
    if len(start) != len(symbols):
        raise ValueError(f"p0 has {len(start)} values but {name} has {len(symbols)} constants")
    return name, formula, param_expr, list(symbols), start, problem


def _with_values(param_expr: sympy.Expr, symbols: Sequence[sympy.Symbol], params: Sequence[float]) -> sympy.Expr:
    return param_expr.xreplace({s: sympy.Float(v) for s, v in zip(symbols, params)})


def fit(
    skeleton: Skeleton,
    x: XData,
    y: Sequence[float],
    sigma: Optional[Union[float, Sequence[float]]] = None,
    space: str = "linear",
    *,
    likelihood: str = "gaussian",
    trials: Optional[Union[float, Sequence[float]]] = None,
    exposure: Optional[Union[float, Sequence[float]]] = None,
    censored: Optional[Sequence[bool]] = None,
    absolute_sigma: bool = False,
    inflate: bool = False,
    p0: Optional[Sequence[float]] = None,
    max_nfev: int = 2000,
) -> Fit:
    """
    Fit a law's constants to (x, y), with standard errors.

    Args:
        skeleton: a law name ('power'), a kepler.laws.Law, or an equation whose Float
            literals are the constants to fit ('27.0*x**0.35').
        x: the setting at each row, or a dict of arrays for a law of several variables.
        sigma: the uncertainty of each y (or one for all). Residuals are divided by it, so a
            point measured with more seeds counts for more. Gaussian fits only.
        space: "linear", or "log" to fit log f(x) to log y (every y must be > 0).
        likelihood: "gaussian" (least squares), "binomial" (y successes out of `trials`),
            "poisson" (y counts over `exposure`, default 1) or "censored" (y times, log-normal
            about the law; `censored` marks the rows where y is only a lower bound).
        absolute_sigma: treat sigma as absolute rather than relative (covariance not scaled
            by the reduced chi-square).
        inflate: with absolute_sigma, scale the covariance up by the reduced chi-square when it
            exceeds 1 (see the module docstring); for a binomial or Poisson fit, by the
            deviance per degree of freedom (a quasi-likelihood).
        p0: starting constants, instead of the law's own guess or the skeleton's literals.

    A law whose domain excludes the data (log x at x <= 0) gives a Fit with success=False
    and the reason in `message`, so it can be ranked as ineligible. Malformed input (lengths
    differ, non-finite values, a zero constant in a skeleton, an unknown space) raises
    ValueError.
    """
    law = _resolve_law(skeleton)
    if law is not None:
        variables = law.variables
    elif isinstance(x, Mapping):
        variables = tuple(str(v) for v in x)
    else:
        variables = ("x",)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(y_arr)
    if n == 0:
        raise ValueError("no data to fit")
    cols = _columns(x, variables, n)
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("x and y must be finite")

    if likelihood != "gaussian":
        if sigma is not None:
            raise ValueError(f"sigma is for a Gaussian fit; a {likelihood} likelihood brings its own errors")
        if space != "linear":
            raise ValueError(f"space is for a Gaussian fit (a {likelihood} fit has its own scale)")
        lik = make(likelihood, y_arr, trials=trials, exposure=exposure, censored=censored)
        return _fit_likelihood(skeleton, law, variables, cols, lik, p0, inflate, max_nfev)
    if trials is not None or exposure is not None or censored is not None:
        raise ValueError("trials, exposure and censored need a binomial, poisson or censored likelihood")

    _check_space(space, y_arr)
    weights = _point_weights(sigma, y_arr, space)  # validates sigma
    weighted = sigma is not None
    name, formula, param_expr, symbols, start, problem = _template(skeleton, law, variables, cols, y_arr, p0)
    if problem:
        return _failed(name, formula, n, space, weighted, absolute_sigma, problem, param_expr, symbols, start,
                       variables=variables)

    if variables == ("x",) and p0 is None:
        # One variable: Kepler's refinement, which also screens cancelling terms.
        if law is not None:
            res: RefineResult = refine_parametrized(param_expr, symbols, start, cols[0], y_arr, X,
                                                    sigma=sigma, space=space, max_nfev=max_nfev)
        else:
            res = refine_expression(parse_skeleton(skeleton), cols[0], y_arr, sigma=sigma, space=space,
                                    max_nfev=max_nfev)
            formula = str(res.param_expression)
            param_expr, symbols = res.param_expression, list(res.param_symbols)
        params = [float(v) for v in res.parameters]
        if param_expr is None or len(params) != len(symbols):
            return _failed(name, formula, n, space, weighted, absolute_sigma, res.message or "fit failed")
        expression, ill, message = res.expression, bool(res.ill_conditioned), res.message
    else:
        params, ill, message = _least_squares(param_expr, symbols, variables, cols, y_arr, weights, space,
                                              start, max_nfev)
        expression = _with_values(param_expr, symbols, params)

    return _gaussian_summary(name, formula, expression, param_expr, symbols, params, variables, cols, y_arr,
                             weights, space, weighted, absolute_sigma, inflate, ill, message)


def _least_squares(param_expr, symbols, variables, cols, y, weights, space, start, max_nfev):
    """Weighted least squares for any number of variables: (constants, ill-conditioned, message)."""
    value, grad = _gradient_fn(param_expr, symbols, _variable_symbols(variables))
    target = np.log(y) if space == "log" else y

    def model(p):
        f = value(cols, p)
        if space == "log":
            with np.errstate(all="ignore"):
                f = np.where(f > 0, np.log(np.where(f > 0, f, 1.0)), np.nan)
        return f

    def residuals(p):
        r = (model(p) - target) * weights
        return np.where(np.isfinite(r), np.clip(r, -1e6, 1e6), 1e6)

    def jacobian(p):
        jac = grad(cols, p)
        if space == "log":
            with np.errstate(all="ignore"):
                jac = jac / value(cols, p)[:, None]
        jac = jac * weights[:, None]
        return np.where(np.isfinite(jac), jac, 0.0)

    try:
        sol = least_squares(residuals, np.asarray(start, dtype=np.float64), jac=jacobian, method="trf",
                            x_scale="jac", max_nfev=max_nfev)
    except Exception as exc:  # scipy raises on a non-finite start
        return [float(v) for v in start], True, f"optimiser failed: {exc}"
    return [float(v) for v in sol.x], False, str(sol.message)


def _gaussian_summary(name, formula, expression, param_expr, symbols, params, variables, cols, y, weights,
                      space, weighted, absolute_sigma, inflate, ill, message) -> Fit:
    n = len(y)
    value, grad = _gradient_fn(param_expr, symbols, _variable_symbols(variables))
    f = value(cols, params)
    target = np.log(y) if space == "log" else y
    with np.errstate(all="ignore"):
        f_space = np.where(f > 0, np.log(np.where(f > 0, f, 1.0)), np.nan) if space == "log" else f
    resid = (f_space - target) * weights
    if not np.all(np.isfinite(resid)):
        what = "x" if variables == ("x",) else "row"
        positive = "" if space == "linear" else " (or not positive)"
        return _failed(name, formula, n, space, weighted, absolute_sigma,
                       f"the fitted law is not finite at every {what}{positive}",
                       param_expr, symbols, params, variables=variables)
    rss = float(np.sum(resid ** 2))
    w2 = weights ** 2
    centre = float(np.sum(w2 * target) / np.sum(w2))
    ss_tot = float(np.sum(w2 * (target - centre) ** 2))
    r2_fit = 1.0 - rss / ss_tot if ss_tot > 1e-300 else (1.0 if rss <= 1e-300 else 0.0)

    k = len(params)
    jac = grad(cols, params)
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
    # Known errors: -2 ln L is the chi-square. Unknown error size, profiled out: n ln(RSS / n).
    known = weighted and absolute_sigma
    neg2 = rss if known else float(n * math.log(max(rss, 1e-300) / n))
    bic, aicc = information(neg2, n, k)

    return Fit(
        name=name, formula=formula, expression=expression,
        param_names=tuple(str(s) for s in symbols), params=tuple(params), stderr=stderr,
        covariance=cov, n=n, rmse=float(np.sqrt(np.mean((f - y) ** 2))), r2=float(r_squared(f, y)), rss=rss,
        r2_fit=float(r2_fit), bic=bic, space=space, weighted=weighted, absolute_sigma=absolute_sigma,
        ill_conditioned=bool(ill or singular), success=True,
        message=message + ("; constants not identifiable from these data" if singular else ""),
        param_expression=param_expr, param_symbols=tuple(symbols), error_scale=float(s2),
        inflate=bool(inflate and known), neg2loglik=neg2, n_obs=n, aicc=aicc, variables=tuple(variables),
    )


# =============================================================================
# Fit by maximum likelihood
# =============================================================================

def _null_nll(lik: Likelihood) -> float:
    """-ln L of the best constant mu (and the best nuisance constants)."""
    m = len(lik)
    if lik.name == "binomial":
        return lik.nll(np.full(m, float(np.sum(lik.k) / np.sum(lik.n))))
    if lik.name == "poisson":
        return lik.nll(np.full(m, max(float(np.sum(lik.c) / np.sum(lik.e)), 1e-300)))
    model = Model(lambda p: np.full(m, p[0]), lambda p: np.ones((m, 1)))
    return maximise(model, lik, [float(np.median(lik.observed()))]).nll


def _fit_likelihood(skeleton, law, variables, cols, lik: Likelihood, p0, inflate, max_nfev) -> Fit:
    n = len(lik)
    name, formula, param_expr, symbols, start, problem = _template(skeleton, law, variables, cols,
                                                                   lik.observed(), p0)
    if problem:
        return _failed(name, formula, n, "linear", True, True, problem, param_expr, symbols, start,
                       likelihood=lik.name, variables=variables, link=lik.link)
    value, grad = _gradient_fn(param_expr, symbols, _variable_symbols(variables))
    model = Model(lambda p: value(cols, p), lambda p: grad(cols, p))
    sol = maximise(model, lik, start, max_iter=max_nfev)
    if not math.isfinite(sol.nll):
        return _failed(name, formula, n, "linear", True, True, sol.message, param_expr, symbols, start,
                       likelihood=lik.name, variables=variables, link=lik.link)

    k = len(symbols)
    params = [float(v) for v in sol.params]
    mu = model.value(sol.params)
    nll = sol.nll
    saturated = lik.saturated()
    null = _null_nll(lik)
    if saturated is not None:
        deviance = max(0.0, 2.0 * (nll - saturated))
        null_deviance = max(0.0, 2.0 * (null - saturated))
        if null_deviance > 1e-12:
            r2_fit = 1.0 - deviance / null_deviance
        else:
            r2_fit = 1.0 if deviance <= 1e-12 else 0.0
    else:
        # No saturated model (times with a fitted spread): Cox and Snell's R^2 against the best
        # constant, which is the ordinary R^2 for Gaussian data.
        deviance = math.nan
        r2_fit = 1.0 - math.exp(min(700.0, 2.0 * (nll - null) / max(lik.n_obs, 1)))
    dof = n - k
    scale = max(1.0, deviance / dof) if inflate and math.isfinite(deviance) and dof > 0 else 1.0
    cov_full = sol.covariance * scale
    cov = cov_full[:k, :k]
    stderr = tuple(float(math.sqrt(v)) if v >= 0 else math.nan for v in np.diag(cov))
    extra_err = [float(math.sqrt(v)) if v >= 0 else math.nan for v in np.diag(cov_full)[k:]]
    nuisance = {e: float(v) for e, v in zip(lik.extra_names, sol.extra)}
    nuisance_stderr = dict(zip(lik.extra_names, extra_err))

    notes = [sol.message]
    separated = (isinstance(lik, Binomial) and bool(np.all((lik.k == 0) | (lik.k == lik.n)))
                 and deviance < 1e-4 and k > 0)
    if separated:
        notes.append("every row is all successes or all failures and the law reproduces them: the data "
                     "only bound how sharp it is (perfect separation)")
    if sol.singular:
        notes.append("constants not identifiable from these data")
    ill = bool(sol.singular or separated or not sol.converged)

    bic, aicc = information(2.0 * nll, lik.n_obs, k + len(nuisance))
    observed = lik.observed()
    rows = ~lik.cens if hasattr(lik, "cens") else np.ones(n, dtype=bool)
    rmse = float(np.sqrt(np.mean((mu[rows] - observed[rows]) ** 2))) if np.any(rows) else math.nan
    r2 = float(r_squared(mu[rows], observed[rows])) if np.sum(rows) > 1 else math.nan

    return Fit(
        name=name, formula=formula, expression=_with_values(param_expr, symbols, params),
        param_names=tuple(str(s) for s in symbols), params=tuple(params), stderr=stderr, covariance=cov,
        n=n, rmse=rmse, r2=r2, rss=deviance, r2_fit=float(r2_fit), bic=bic, space="linear", weighted=True,
        absolute_sigma=True, ill_conditioned=ill, success=True, message="; ".join(notes),
        param_expression=param_expr, param_symbols=tuple(symbols), error_scale=float(scale),
        inflate=bool(inflate), neg2loglik=2.0 * nll, n_obs=lik.n_obs, aicc=aicc, likelihood=lik.name,
        variables=tuple(variables), link=lik.link, nuisance=nuisance, nuisance_stderr=nuisance_stderr,
    )


# =============================================================================
# Predict
# =============================================================================

def _prediction_columns(fit_result: Fit, x) -> Tuple[List[np.ndarray], bool]:
    if isinstance(x, Mapping):
        missing = [v for v in fit_result.variables if v not in x]
        if missing:
            raise ValueError(f"x lacks {', '.join(missing)} (the law's variables are "
                             f"{', '.join(fit_result.variables)})")
        arrays = [np.asarray(x[v], dtype=np.float64) for v in fit_result.variables]
        scalar = all(a.ndim == 0 for a in arrays)
        cols = [np.array(c, dtype=np.float64) for c in np.broadcast_arrays(*[np.atleast_1d(a).ravel()
                                                                              for a in arrays])]
        return cols, scalar
    if tuple(fit_result.variables) != ("x",):
        raise ValueError(f"this law's variables are {', '.join(fit_result.variables)}: pass x as a dict")
    return [np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()], np.ndim(x) == 0


def _link(fit_result: Fit) -> str:
    return "log" if fit_result.space == "log" else getattr(fit_result, "link", "identity")


def _on_link(fit_result: Fit, cols: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, the law on its link scale, its gradient in the constants on that scale)."""
    value, grad = _gradient_fn(fit_result.param_expression, fit_result.param_symbols,
                               _variable_symbols(fit_result.variables))
    params = list(fit_result.params)
    f = value(cols, params)
    g = grad(cols, params)
    link = _link(fit_result)
    with np.errstate(all="ignore"):
        if link == "log":
            return f, np.log(f), g / f[:, None]
        if link == "logit":
            return f, np.log(f / (1.0 - f)), g / (f * (1.0 - f))[:, None]
    return f, f, g


def predict_covariance(fit_result: Fit, x: XData) -> Tuple[np.ndarray, np.ndarray]:
    """
    The fitted law at `x` on its link scale (log f for a log-space fit, a rate or a time; the
    log odds for a probability; f otherwise) and the covariance matrix of those values from
    the constants' covariance, G C G^T (the delta method). Predictions at several x share the
    constants, so their errors are correlated; a joint test of held-out measurements needs the
    whole matrix, not only the bands.
    """
    cols, _ = _prediction_columns(fit_result, x)
    m = len(cols[0])
    if not fit_result.success:
        return np.full(m, np.nan), np.full((m, m), np.nan)
    _, centre, g = _on_link(fit_result, cols)
    with np.errstate(all="ignore"):
        cov = g @ fit_result.covariance @ g.T if fit_result.covariance.size else np.zeros((m, m))
    return centre, cov


def predict(fit_result: Fit, x: Union[float, XData], level: float = 0.95):
    """
    The fitted law at `x` with a `level` confidence band from the constants' covariance:
    returns (y, low, high), arrays for an array x and floats for a scalar x. A band is built
    on the law's link scale and mapped back, so a probability's stays between 0 and 1 and a
    positive quantity's stays positive.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1 (got {level})")
    cols, scalar = _prediction_columns(fit_result, x)
    m = len(cols[0])
    if not fit_result.success:
        nan = np.full(m, np.nan)
        return (float(nan[0]), float(nan[0]), float(nan[0])) if scalar else (nan, nan.copy(), nan.copy())

    f, centre, g = _on_link(fit_result, cols)
    cov = fit_result.covariance
    with np.errstate(all="ignore"):
        var = np.einsum("ij,jk,ik->i", g, cov, g) if cov.size else np.zeros(m)
        sd = np.sqrt(np.maximum(var, 0.0))
    if fit_result.dof > 0 and not (fit_result.weighted and fit_result.absolute_sigma):
        q = float(stats.t.ppf(0.5 + level / 2.0, fit_result.dof))
    else:
        q = float(stats.norm.ppf(0.5 + level / 2.0))
    link = _link(fit_result)
    with np.errstate(all="ignore"):
        low, high = centre - q * sd, centre + q * sd
        if link == "log":
            low, high = np.exp(low), np.exp(high)
        elif link == "logit":
            low, high = 1.0 / (1.0 + np.exp(-low)), 1.0 / (1.0 + np.exp(-high))
    if scalar:
        return float(f[0]), float(low[0]), float(high[0])
    return f, low, high

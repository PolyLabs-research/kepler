"""
joint.py - One law across several groups (lattices, rules, densities), some constants shared
and the rest each group's own; and which constants the data say are shared.

    fit_groups(law, x, y, group, shared=("b",), sigma=..., likelihood=...) -> GroupFit
    compare_sharing(law, x, y, group, ...) -> Sharing

`group` labels each row. A shared constant is one number for every group ("the exponent is
universal"); any other constant is fitted per group ("each lattice has its own amplitude"),
named like "a[hexagon]". The fit is one likelihood over every row, so a shared constant's error
bar draws on every group's data, and it takes the same keyword arguments as kepler.fit (sigma,
space, absolute_sigma, inflate, likelihood, trials, exposure, censored).

Comparing sharing patterns. All patterns are fits of the same data, so their BICs compare
directly. A pattern that ties constants another leaves free is nested in it: if the tie is
true, -2 ln L rises by a chi-square on (groups - 1) x (constants tied) degrees of freedom, so
each pattern also gets a likelihood-ratio p-value against the pattern sharing nothing. With
`inflate`, every -2 ln L difference is divided by the sharing-nothing fit's reduced chi-square
(or deviance per degree of freedom) when it exceeds 1, as kepler.rank does. The chosen pattern is
the one with the fewest constants within the Occam window of the lowest BIC.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from kepler.fitting import (
    Fit,
    _columns,
    _covariance,
    _gradient_fn,
    _jsonable,
    _variable_symbols,
    _with_values,
    fit,
    information,
)
from kepler.laws import Law, get_law
from kepler.likelihoods import Binomial, make
from kepler.mle import Model, maximise
from kepler.ranking import OCCAM_WINDOW, occam_order
from kepler.refine import _check_space, _point_weights


@dataclass(eq=False)
class GroupFit:
    """A law fitted across groups: shared constants by name, the others as name[group]."""

    name: str
    law: Law = field(repr=False)
    groups: Tuple[str, ...]
    shared: Tuple[str, ...]
    param_names: Tuple[str, ...]
    params: Tuple[float, ...]
    stderr: Tuple[float, ...]
    covariance: np.ndarray = field(repr=False)
    n: int
    n_obs: int
    rss: float  # chi-square, residual sum of squares, or deviance, as in Fit
    neg2loglik: float
    bic: float
    aicc: float
    likelihood: str
    space: str
    weighted: bool
    absolute_sigma: bool
    error_scale: float
    ill_conditioned: bool
    success: bool
    message: str
    nuisance: Dict[str, float] = field(default_factory=dict)
    nuisance_stderr: Dict[str, float] = field(default_factory=dict)
    inflate: bool = False

    @property
    def n_constants(self) -> int:
        return len(self.params)

    @property
    def dof(self) -> int:
        return self.n - self.n_constants

    def _key(self, name: str, group: Optional[str]) -> str:
        if name in self.shared:
            return name
        if group is None:
            raise ValueError(f"{name} is fitted per group: say which group")
        return f"{name}[{group}]"

    def value(self, name: str, group: Optional[str] = None) -> float:
        return self.params[self.param_names.index(self._key(name, group))]

    def error(self, name: str, group: Optional[str] = None) -> float:
        return self.stderr[self.param_names.index(self._key(name, group))]

    def constants(self, group: str) -> Dict[str, float]:
        """The law's constants for one group."""
        return {p: self.value(p, group) for p in self.law.params}

    def fit_for(self, group: str) -> Fit:
        """The law with one group's constants and their covariance (shared ones included), as a
        Fit, so kepler.predict gives that group's curve and band."""
        if group not in self.groups:
            raise ValueError(f"unknown group {group!r}; groups: {', '.join(self.groups)}")
        idx = [self.param_names.index(self._key(p, group)) for p in self.law.params]
        expr, symbols = self.law.parametrized()
        values = [self.params[i] for i in idx]
        cov = self.covariance[np.ix_(idx, idx)]
        nan = float("nan")
        return Fit(
            name=f"{self.name}[{group}]", formula=self.law.formula, expression=_with_values(expr, symbols, values),
            param_names=tuple(self.law.params), params=tuple(values),
            stderr=tuple(self.stderr[i] for i in idx), covariance=cov, n=self.n, rmse=nan, r2=nan,
            rss=self.rss, r2_fit=nan, bic=self.bic, space=self.space, weighted=self.weighted,
            absolute_sigma=self.absolute_sigma, ill_conditioned=self.ill_conditioned, success=self.success,
            message=self.message, param_expression=expr, param_symbols=tuple(symbols),
            error_scale=self.error_scale, neg2loglik=self.neg2loglik, n_obs=self.n_obs, aicc=self.aicc,
            likelihood=self.likelihood, variables=self.law.variables, link=_link_of(self.likelihood),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable({
            "name": self.name,
            "groups": list(self.groups),
            "shared": list(self.shared),
            "params": dict(zip(self.param_names, self.params)),
            "stderr": dict(zip(self.param_names, self.stderr)),
            "n": self.n,
            "n_obs": self.n_obs,
            "n_constants": self.n_constants,
            "rss": self.rss,
            "neg2loglik": self.neg2loglik,
            "bic": self.bic,
            "aicc": self.aicc,
            "likelihood": self.likelihood,
            "space": self.space,
            "error_scale": self.error_scale,
            "nuisance": self.nuisance,
            "ill_conditioned": self.ill_conditioned,
            "success": self.success,
            "message": self.message,
        })


def _link_of(likelihood: str) -> str:
    return {"binomial": "logit", "poisson": "log", "censored_lognormal": "log"}.get(likelihood, "identity")


def _labels(group: Sequence[Any], n: int) -> Tuple[Tuple[str, ...], np.ndarray]:
    labels = [str(g) for g in np.asarray(group, dtype=object).ravel()]
    if len(labels) != n:
        raise ValueError(f"group has {len(labels)} labels but y has {n}")
    order = list(dict.fromkeys(labels))
    if len(order) < 2:
        raise ValueError("a joint fit needs at least two groups")
    index = {g: i for i, g in enumerate(order)}
    return tuple(order), np.array([index[g] for g in labels])


class _Layout:
    """Where each group's copy of each law constant sits in the joint parameter vector."""

    def __init__(self, law: Law, groups: Sequence[str], shared: Sequence[str]):
        unknown = [p for p in shared if p not in law.params]
        if unknown:
            raise ValueError(f"{law.name} has no constant {', '.join(unknown)} (it has {', '.join(law.params)})")
        self.names: List[str] = []
        self.index = np.zeros((len(groups), len(law.params)), dtype=int)
        for j, p in enumerate(law.params):
            if p in shared:
                self.index[:, j] = len(self.names)
                self.names.append(p)
            else:
                for g, label in enumerate(groups):
                    self.index[g, j] = len(self.names)
                    self.names.append(f"{p}[{label}]")


def _sub(value, rows):
    """One group's rows of a per-row argument (a scalar applies to every row)."""
    if value is None or np.ndim(value) == 0:
        return value
    return np.asarray(value)[rows]


def fit_groups(
    skeleton: Union[str, Law],
    x: Union[Sequence[float], Mapping[str, Sequence[float]]],
    y: Sequence[float],
    group: Sequence[Any],
    shared: Sequence[str] = (),
    sigma: Optional[Union[float, Sequence[float]]] = None,
    space: str = "linear",
    *,
    likelihood: str = "gaussian",
    trials=None,
    exposure=None,
    censored=None,
    absolute_sigma: bool = False,
    inflate: bool = False,
    max_nfev: int = 2000,
) -> GroupFit:
    """Fit `skeleton` (a law name or Law) to every group at once, the `shared` constants common
    to all groups and the rest per group (see the module docstring)."""
    law = skeleton if isinstance(skeleton, Law) else get_law(skeleton)
    shared = tuple(p for p in law.params if p in set(shared)) + tuple(p for p in shared if p not in law.params)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = len(y_arr)
    cols = _columns(x, law.variables, n)
    groups, gidx = _labels(group, n)
    layout = _Layout(law, groups, shared)
    xdata = cols[0] if law.variables == ("x",) else dict(zip(law.variables, cols))
    problem = law.domain_problem(xdata)
    gaussian = likelihood == "gaussian"
    name = f"{law.name} ({'shared ' + ', '.join(shared) if shared else 'nothing shared'})"

    # Start: each group fitted on its own; a shared constant starts at their median.
    kw = dict(likelihood=likelihood)
    if gaussian:
        kw.update(space=space, absolute_sigma=absolute_sigma)
    own = []
    for g in range(len(groups)):
        rows = gidx == g
        xg = cols[0][rows] if law.variables == ("x",) else {v: c[rows] for v, c in zip(law.variables, cols)}
        try:
            f = fit(law, xg, y_arr[rows], sigma=_sub(sigma, rows) if gaussian else None,
                    trials=_sub(trials, rows), exposure=_sub(exposure, rows), censored=_sub(censored, rows), **kw)
            own.append(list(f.params) if f.success and np.all(np.isfinite(f.params)) else None)
        except ValueError:
            own.append(None)
    fallback = law.initial(xdata, y_arr if gaussian else make(likelihood, y_arr, trials=trials, exposure=exposure,
                                                              censored=censored).observed())
    start = np.zeros(len(layout.names))
    for j, p in enumerate(law.params):
        values = [o[j] if o is not None else fallback[j] for o in own]
        if p in shared:
            start[layout.index[0, j]] = float(np.median(values))
        else:
            for g in range(len(groups)):
                start[layout.index[g, j]] = values[g]

    expr, symbols = law.parametrized()
    value, grad = _gradient_fn(expr, symbols, _variable_symbols(law.variables))
    group_rows = [np.flatnonzero(gidx == g) for g in range(len(groups))]
    group_cols = [[c[r] for c in cols] for r in group_rows]

    def mu(theta):
        out = np.empty(n)
        for g, rows in enumerate(group_rows):
            out[rows] = value(group_cols[g], theta[layout.index[g]])
        return out

    def jac(theta):
        out = np.zeros((n, len(theta)))
        for g, rows in enumerate(group_rows):
            local = grad(group_cols[g], theta[layout.index[g]])
            for j in range(len(law.params)):
                out[rows, layout.index[g, j]] += local[:, j]
        return out

    common = dict(name=name, law=law, groups=groups, shared=shared, param_names=tuple(layout.names), n=n)
    if problem:
        return _failed(common, start, problem, likelihood, space, gaussian, sigma, absolute_sigma)
    if gaussian:
        return _gaussian(common, mu, jac, start, y_arr, sigma, space, absolute_sigma, inflate, max_nfev)
    lik = make(likelihood, y_arr, trials=trials, exposure=exposure, censored=censored)
    return _likelihood(common, Model(mu, jac), lik, start, inflate, max_nfev)


def _failed(common, start, message, likelihood, space, gaussian, sigma, absolute_sigma) -> GroupFit:
    k = len(start)
    nan = float("nan")
    return GroupFit(**common, params=tuple(float(v) for v in start), stderr=tuple([math.inf] * k),
                    covariance=np.full((k, k), np.inf), n_obs=common["n"], rss=nan, neg2loglik=nan, bic=nan,
                    aicc=nan, likelihood=likelihood if not gaussian else "gaussian", space=space,
                    weighted=sigma is not None or not gaussian, absolute_sigma=absolute_sigma or not gaussian,
                    error_scale=nan, ill_conditioned=False, success=False, message=message)


def _gaussian(common, mu, jac, start, y, sigma, space, absolute_sigma, inflate, max_nfev) -> GroupFit:
    n = common["n"]
    _check_space(space, y)
    weights = _point_weights(sigma, y, space)
    weighted = sigma is not None
    target = np.log(y) if space == "log" else y

    def on_space(theta):
        f = mu(theta)
        if space == "log":
            with np.errstate(all="ignore"):
                f = np.where(f > 0, np.log(np.where(f > 0, f, 1.0)), np.nan)
        return f

    def residuals(theta):
        r = (on_space(theta) - target) * weights
        return np.where(np.isfinite(r), np.clip(r, -1e6, 1e6), 1e6)

    def jacobian(theta):
        j = jac(theta)
        if space == "log":
            with np.errstate(all="ignore"):
                j = j / mu(theta)[:, None]
        j = j * weights[:, None]
        return np.where(np.isfinite(j), j, 0.0)

    sol = least_squares(residuals, start, jac=jacobian, method="trf", x_scale="jac", max_nfev=max_nfev)
    theta = sol.x
    resid = (on_space(theta) - target) * weights
    if not np.all(np.isfinite(resid)):
        return _failed(common, theta, "the fitted law is not finite at every row", "gaussian", space, True,
                       sigma, absolute_sigma)
    rss = float(np.sum(resid ** 2))
    k = len(theta)
    dof = n - k
    known = weighted and absolute_sigma
    if known:
        s2 = max(1.0, rss / dof) if inflate and dof > 0 else 1.0
    else:
        s2 = rss / dof if dof > 0 else math.inf
    cov, singular = _covariance(jacobian(theta), s2)
    neg2 = rss if known else float(n * math.log(max(rss, 1e-300) / n))
    bic, aicc = information(neg2, n, k)
    return GroupFit(**common, params=tuple(float(v) for v in theta),
                    stderr=tuple(float(math.sqrt(v)) if v >= 0 else math.nan for v in np.diag(cov)),
                    covariance=cov, n_obs=n, rss=rss, neg2loglik=neg2, bic=bic, aicc=aicc, likelihood="gaussian",
                    space=space, weighted=weighted, absolute_sigma=absolute_sigma, error_scale=float(s2),
                    ill_conditioned=bool(singular), success=True,
                    message=str(sol.message) + ("; constants not identifiable from these data" if singular else ""),
                    inflate=bool(inflate and known))


def _likelihood(common, model: Model, lik, start, inflate, max_nfev) -> GroupFit:
    n = common["n"]
    sol = maximise(model, lik, start, max_iter=max_nfev)
    k = len(start)
    if not math.isfinite(sol.nll):
        return _failed(common, start, sol.message, lik.name, "linear", False, None, True)
    saturated = lik.saturated()
    deviance = max(0.0, 2.0 * (sol.nll - saturated)) if saturated is not None else math.nan
    dof = n - k
    scale = max(1.0, deviance / dof) if inflate and math.isfinite(deviance) and dof > 0 else 1.0
    cov_full = sol.covariance * scale
    separated = (isinstance(lik, Binomial) and bool(np.all((lik.k == 0) | (lik.k == lik.n)))
                 and deviance < 1e-4)
    notes = [sol.message]
    if separated:
        notes.append("perfect separation: the data only bound how sharp the law is")
    if sol.singular:
        notes.append("constants not identifiable from these data")
    bic, aicc = information(2.0 * sol.nll, lik.n_obs, k + len(lik.extra_names))
    extra_err = [float(math.sqrt(v)) if v >= 0 else math.nan for v in np.diag(cov_full)[k:]]
    return GroupFit(**common, params=tuple(float(v) for v in sol.params),
                    stderr=tuple(float(math.sqrt(v)) if v >= 0 else math.nan for v in np.diag(cov_full)[:k]),
                    covariance=cov_full[:k, :k], n_obs=lik.n_obs, rss=deviance, neg2loglik=2.0 * sol.nll, bic=bic,
                    aicc=aicc, likelihood=lik.name, space="linear", weighted=True, absolute_sigma=True,
                    error_scale=float(scale), ill_conditioned=bool(sol.singular or separated or not sol.converged),
                    success=True, message="; ".join(notes),
                    nuisance={e: float(v) for e, v in zip(lik.extra_names, sol.extra)},
                    nuisance_stderr=dict(zip(lik.extra_names, extra_err)), inflate=bool(inflate))


# =============================================================================
# Which constants are shared?
# =============================================================================

@dataclass
class SharingEntry:
    shared: Tuple[str, ...]
    n_constants: int
    bic: Optional[float]
    delta_bic: Optional[float]
    lr_stat: Optional[float]  # -2 ln L above the sharing-nothing fit (divided by the overdispersion)
    lr_dof: int
    p_value: Optional[float]  # chance of a rise this large if the shared constants really are equal
    eligible: bool
    reason: str
    fit: GroupFit = field(repr=False, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable({
            "shared": list(self.shared), "n_constants": self.n_constants, "bic": self.bic,
            "delta_bic": self.delta_bic, "lr_stat": self.lr_stat, "lr_dof": self.lr_dof,
            "p_value": self.p_value, "eligible": self.eligible, "reason": self.reason,
        })


@dataclass
class Sharing:
    """Every sharing pattern tried, in selection order, and the one chosen."""

    law: str
    groups: Tuple[str, ...]
    entries: List[SharingEntry]
    best: Optional[SharingEntry]
    overdispersion: float
    note: str

    def entry(self, shared: Sequence[str]) -> SharingEntry:
        wanted = set(shared)
        for e in self.entries:
            if set(e.shared) == wanted:
                return e
        raise KeyError(tuple(shared))

    def to_dict(self) -> Dict[str, Any]:
        return {"law": self.law, "groups": list(self.groups), "best": list(self.best.shared) if self.best else None,
                "overdispersion": self.overdispersion, "note": self.note,
                "entries": [e.to_dict() for e in self.entries]}


def compare_sharing(
    skeleton: Union[str, Law],
    x,
    y,
    group,
    patterns: Optional[Sequence[Sequence[str]]] = None,
    *,
    occam_window: float = OCCAM_WINDOW,
    **kwargs,
) -> Sharing:
    """Fit every sharing pattern (each subset of the law's constants, unless `patterns` names
    some) and say which the data support (see the module docstring). Keyword arguments go to
    fit_groups."""
    law = skeleton if isinstance(skeleton, Law) else get_law(skeleton)
    if patterns is None:
        patterns = [c for r in range(len(law.params) + 1) for c in itertools.combinations(law.params, r)]
    patterns = [tuple(p) for p in patterns]
    if () not in patterns:
        patterns = [(), *patterns]
    fits = {p: fit_groups(law, x, y, group, shared=p, **kwargs) for p in patterns}
    free = fits[()]
    groups = free.groups
    c = 1.0
    if kwargs.get("inflate") and free.success and free.dof > 0 and math.isfinite(free.rss):
        c = max(1.0, free.rss / free.dof)

    entries = []
    for p, f in fits.items():
        bic = None
        if f.success and math.isfinite(f.neg2loglik):
            k = f.n_constants + len(f.nuisance)
            bic = float(f.neg2loglik / c + k * math.log(max(f.n_obs, 2)))
        dof = (len(groups) - 1) * len(p)
        lr = p_value = None
        if bic is not None and free.success:
            lr = max(0.0, float((f.neg2loglik - free.neg2loglik) / c))
            p_value = float(stats.chi2.sf(lr, dof)) if dof > 0 else 1.0
        reason = ""
        if not f.success:
            reason = f.message or "fit failed"
        elif f.ill_conditioned:
            reason = "ill-conditioned fit (constants not identifiable)"
        entries.append(SharingEntry(p, f.n_constants, bic, None, lr, dof, p_value,
                                    f.success and not f.ill_conditioned and bic is not None, reason, f))
    scored = [e for e in entries if e.bic is not None]
    if scored:
        floor = min(e.bic for e in scored)
        for e in scored:
            e.delta_bic = e.bic - floor
    ordered = occam_order([e for e in entries if e.eligible], occam_window, lambda e: e.bic, lambda e: e.n_constants)
    rest = sorted((e for e in entries if not e.eligible), key=lambda e: e.bic if e.bic is not None else math.inf)
    best = ordered[0] if ordered else None
    if best is None:
        note = "no sharing pattern could be fitted"
    else:
        own = [q for q in law.params if q not in best.shared]
        note = (f"shared: {', '.join(best.shared) or 'nothing'}; per group: {', '.join(own) or 'nothing'}"
                f" (likelihood-ratio p = {best.p_value:.3g} against sharing nothing)")
    return Sharing(law.name, groups, ordered + rest, best, c, note)

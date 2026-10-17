"""
distributions.py - Which distribution do sizes or times follow? Maximum likelihood above a
lower cutoff xmin, xmin chosen by the Kolmogorov-Smirnov distance, and tests between families,
after Clauset, Shalizi and Newman (2009), "Power-law distributions in empirical data".

    fit_distribution(samples, family, xmin=None, discrete=False) -> DistributionFit
    find_xmin(samples, family="power_law", discrete=False)       -> DistributionFit
    compare_distributions(a, b)                                  -> DistributionComparison
    goodness_of_fit(fit, resamples=100, seed=0)                  -> (p, ks distances)

Families, each for x >= xmin (on the integers when discrete):

    power_law            p(x) ~ x^-alpha                        alpha > 1
    truncated_power_law  p(x) ~ x^-alpha exp(-lambda x)         lambda > 0
    exponential          p(x) ~ exp(-lambda x)                  lambda > 0
    lognormal            p(x) ~ exp(-(ln x - mu)^2 / (2 s^2)) / x   s > 0
                         (discrete: the mass of [x - 1/2, x + 1/2))

A fit says how well, not whether: a power law can be the best of these families and still be
wrong. `goodness_of_fit` answers whether: data drawn from the fit (below xmin, the observed
values), refitted with xmin chosen afresh, give a KS distance at least the observed one in a
share p of draws; p < 0.1 rules the family out. `compare_distributions` answers which of two:
the log-likelihood ratio R summed over the tail, with Vuong's normalised test when the families
are not nested (p small: the sign of R is significant), and a likelihood-ratio test on the
boundary (half a chi-square with one degree of freedom) when one is the other with lambda = 0
(a power law inside a truncated power law). Seeded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtri, zeta

FAMILIES = ("power_law", "truncated_power_law", "exponential", "lognormal")
_NAMES = {
    "power_law": ("alpha",),
    "truncated_power_law": ("alpha", "lambda"),
    "exponential": ("lambda",),
    "lognormal": ("mu", "sigma"),
}
MIN_TAIL = 10
_SUM_LIMIT = 2_000_000  # terms in a discrete normaliser without a closed form


# --- one family's density, distribution function and sampler -----------------------------------

def _log_tpl_norm(alpha: float, lam: float, xmin: float, discrete: bool) -> float:
    """ln of sum / integral of x^-alpha exp(-lambda x) over x >= xmin."""
    if discrete:
        n = int(min(_SUM_LIMIT, max(1000, math.ceil(60.0 / lam))))
        x = xmin + np.arange(n, dtype=np.float64)
        terms = -alpha * np.log(x) - lam * x
        top = float(np.max(terms))
        return top + math.log(float(np.sum(np.exp(terms - top))))
    # Continuous: integrate x^(1 - alpha) exp(-lambda x) d(ln x) on a log grid.
    upper = math.log(xmin + 80.0 / lam)
    u = np.linspace(math.log(xmin), upper, 4001)
    x = np.exp(u)
    terms = (1.0 - alpha) * u - lam * x
    top = float(np.max(terms))
    weights = np.exp(terms - top)
    return top + math.log(float(np.sum(0.5 * (weights[1:] + weights[:-1]) * np.diff(u))))


def _lognormal_log_survival(x, mu: float, s: float) -> np.ndarray:
    return log_ndtr(-(np.log(x) - mu) / s)


def logpdf(family: str, x: np.ndarray, params: Sequence[float], xmin: float, discrete: bool) -> np.ndarray:
    """ln p(x) for x >= xmin."""
    x = np.asarray(x, dtype=np.float64)
    if family == "power_law":
        (alpha,) = params
        if discrete:
            return -alpha * np.log(x) - math.log(zeta(alpha, xmin))
        return math.log(alpha - 1.0) - math.log(xmin) - alpha * np.log(x / xmin)
    if family == "exponential":
        (lam,) = params
        if discrete:
            return math.log(-math.expm1(-lam)) - lam * (x - xmin)
        return math.log(lam) - lam * (x - xmin)
    if family == "truncated_power_law":
        alpha, lam = params
        return -alpha * np.log(x) - lam * x - _log_tpl_norm(alpha, lam, xmin, discrete)
    if family == "lognormal":
        mu, s = params
        if discrete:
            lo = np.maximum(x - 0.5, 1e-300)
            log_s_lo = _lognormal_log_survival(lo, mu, s)
            log_s_hi = _lognormal_log_survival(x + 0.5, mu, s)
            mass = log_s_lo + np.log(-np.expm1(np.minimum(log_s_hi - log_s_lo, -1e-300)))
            return mass - _lognormal_log_survival(max(xmin - 0.5, 1e-300), mu, s)
        z = (np.log(x) - mu) / s
        return (-np.log(x) - math.log(s) - 0.5 * math.log(2 * math.pi) - 0.5 * z * z
                - _lognormal_log_survival(xmin, mu, s))
    raise ValueError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")


def cdf(family: str, x: np.ndarray, params: Sequence[float], xmin: float, discrete: bool) -> np.ndarray:
    """P(X <= x | X >= xmin)."""
    x = np.asarray(x, dtype=np.float64)
    if family == "power_law":
        (alpha,) = params
        if discrete:
            return 1.0 - zeta(alpha, x + 1.0) / zeta(alpha, xmin)
        return 1.0 - (x / xmin) ** (1.0 - alpha)
    if family == "exponential":
        (lam,) = params
        return 1.0 - np.exp(-lam * (x - xmin + (1.0 if discrete else 0.0)))
    if family == "lognormal":
        mu, s = params
        edge = (x + 0.5) if discrete else x
        start = max(xmin - 0.5, 1e-300) if discrete else xmin
        return 1.0 - np.exp(_lognormal_log_survival(edge, mu, s) - _lognormal_log_survival(start, mu, s))
    if family == "truncated_power_law":
        top = float(np.max(x))
        if discrete:
            grid = xmin + np.arange(int(top - xmin) + 1, dtype=np.float64)
            mass = np.cumsum(np.exp(logpdf(family, grid, params, xmin, True)))
            return mass[(x - xmin).astype(int)]
        alpha, lam = params
        u = np.linspace(math.log(xmin), math.log(max(top, xmin * (1 + 1e-9))), 4001)
        dens = np.exp((1.0 - alpha) * u - lam * np.exp(u) - _log_tpl_norm(alpha, lam, xmin, False))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(u))])
        return np.interp(np.log(x), u, cum)
    raise ValueError(f"unknown family {family!r}")


def sample(family: str, params: Sequence[float], xmin: float, discrete: bool, n: int,
           rng: np.random.Generator) -> np.ndarray:
    """n draws from the family above xmin."""
    u = rng.random(n)
    if family == "power_law":
        (alpha,) = params
        if discrete:  # Clauset et al. (D.6): close, and exact as xmin grows
            return np.floor((xmin - 0.5) * (1.0 - u) ** (-1.0 / (alpha - 1.0)) + 0.5)
        return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
    if family == "exponential":
        (lam,) = params
        if discrete:
            return xmin + np.floor(-np.log1p(-u) / lam)
        return xmin - np.log1p(-u) / lam
    if family == "lognormal":
        mu, s = params
        start = max(xmin - 0.5, 1e-300) if discrete else xmin
        tail = np.exp(_lognormal_log_survival(start, mu, s))
        draws = np.exp(mu + s * ndtri(1.0 - tail * (1.0 - u)))
        return np.maximum(np.floor(draws + 0.5), xmin) if discrete else draws
    if family == "truncated_power_law":
        alpha, lam = params
        out: List[float] = []
        while len(out) < n:  # a power law thinned by exp(-lambda (x - xmin))
            m = 2 * (n - len(out)) + 10
            draws = sample("power_law", (max(alpha, 1.0 + 1e-6),), xmin, discrete, m, rng)
            if alpha <= 1.0:  # a power law with alpha <= 1 cannot be normalised: thin an exponential
                draws = sample("exponential", (lam,), xmin, discrete, m, rng)
                keep = rng.random(m) < (draws / xmin) ** (-max(alpha, 0.0))
            else:
                keep = rng.random(m) < np.exp(-lam * (draws - xmin))
            out.extend(draws[keep].tolist())
        return np.asarray(out[:n])
    raise ValueError(f"unknown family {family!r}")


# --- fitting ---------------------------------------------------------------------------------------

def _to_natural(family: str, u: np.ndarray) -> Tuple[float, ...]:
    if family == "power_law":
        return (1.0 + math.exp(u[0]),)
    if family == "exponential":
        return (math.exp(u[0]),)
    if family == "truncated_power_law":
        return (float(u[0]), math.exp(u[1]))
    return (float(u[0]), math.exp(u[1]))  # lognormal


def _start(family: str, tail: np.ndarray, xmin: float, discrete: bool) -> np.ndarray:
    logs = np.log(tail / (xmin - 0.5 if discrete else xmin))
    alpha = 1.0 + len(tail) / max(float(np.sum(logs)), 1e-9)
    spread = max(float(np.mean(tail) - xmin), 1e-3)
    if family == "power_law":
        return np.array([math.log(max(alpha - 1.0, 1e-3))])
    if family == "exponential":
        return np.array([math.log(1.0 / spread)])
    if family == "truncated_power_law":
        return np.array([min(alpha, 3.0), math.log(1.0 / (10.0 * float(np.max(tail))))])
    lt = np.log(tail)
    return np.array([float(np.mean(lt)), math.log(max(float(np.std(lt)), 0.1))])


def _nll(family, tail, xmin, discrete) -> Callable[[np.ndarray], float]:
    def f(u):
        try:
            with np.errstate(all="ignore"):
                v = -float(np.sum(logpdf(family, tail, _to_natural(family, u), xmin, discrete)))
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e300
        return v if math.isfinite(v) else 1e300
    return f


def _hessian(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    k = len(x)
    h = 1e-4 * np.maximum(np.abs(x), 1e-2)
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                up, down = x.copy(), x.copy()
                up[i] += h[i]
                down[i] -= h[i]
                H[i, i] = (f(up) - 2 * f0 + f(down)) / h[i] ** 2
            else:
                pp, pm, mp, mm = (x.copy() for _ in range(4))
                pp[i] += h[i]; pp[j] += h[j]  # noqa: E702
                pm[i] += h[i]; pm[j] -= h[j]  # noqa: E702
                mp[i] -= h[i]; mp[j] += h[j]  # noqa: E702
                mm[i] -= h[i]; mm[j] -= h[j]  # noqa: E702
                H[i, j] = H[j, i] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h[i] * h[j])
    return H


@dataclass(eq=False)
class DistributionFit:
    family: str
    params: Dict[str, float]
    stderr: Dict[str, float]
    xmin: float
    discrete: bool
    n: int  # every sample
    n_tail: int  # samples >= xmin, the ones fitted
    loglik: float
    aic: float
    bic: float
    ks: float  # largest gap between the tail's empirical and fitted distribution functions
    success: bool
    message: str
    tail: np.ndarray = field(repr=False)
    below: np.ndarray = field(repr=False)  # samples under xmin, for the goodness-of-fit draws
    pointwise: np.ndarray = field(repr=False)  # ln p(x) of each tail sample

    @property
    def values(self) -> Tuple[float, ...]:
        return tuple(self.params[p] for p in _NAMES[self.family])

    def logpdf(self, x) -> np.ndarray:
        return logpdf(self.family, np.asarray(x, dtype=np.float64), self.values, self.xmin, self.discrete)

    def cdf(self, x) -> np.ndarray:
        return cdf(self.family, np.asarray(x, dtype=np.float64), self.values, self.xmin, self.discrete)

    def to_dict(self) -> Dict[str, Any]:
        def clean(v):
            return None if isinstance(v, float) and not math.isfinite(v) else v

        return {
            "family": self.family, "params": self.params, "stderr": {k: clean(v) for k, v in self.stderr.items()},
            "xmin": self.xmin, "discrete": self.discrete, "n": self.n, "n_tail": self.n_tail,
            "loglik": clean(self.loglik), "aic": clean(self.aic), "bic": clean(self.bic), "ks": clean(self.ks),
            "success": self.success, "message": self.message,
        }


def _ks(family, tail, values, xmin, discrete) -> float:
    xs = np.sort(tail)
    n = len(xs)
    if discrete:
        # Compare at each distinct value, the empirical CDF counting every copy of it.
        uniq = np.unique(xs)
        emp = np.searchsorted(xs, uniq, side="right") / n
        return float(np.max(np.abs(emp - cdf(family, uniq, values, xmin, discrete))))
    model = cdf(family, xs, values, xmin, discrete)
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - model)), np.max(np.abs(model - lower))))


def _prepare(samples, xmin, discrete) -> Tuple[np.ndarray, float]:
    data = np.asarray(samples, dtype=np.float64).ravel()
    if not np.all(np.isfinite(data)):
        raise ValueError("samples must be finite")
    if discrete and np.any(data != np.round(data)):
        raise ValueError("discrete samples must be integers")
    if np.any(data <= 0):
        raise ValueError("these families need positive samples (shift or drop zeros first)")
    if xmin is None:
        xmin = float(np.min(data))
    if xmin <= 0:
        raise ValueError("xmin must be positive")
    return data, float(xmin)


def fit_distribution(samples: Sequence[float], family: str, xmin: Optional[float] = None,
                     discrete: bool = False) -> DistributionFit:
    """Maximum-likelihood fit of `family` to the samples >= xmin (the smallest sample if None)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")
    data, xmin = _prepare(samples, xmin, discrete)
    tail = data[data >= xmin]
    below = data[data < xmin]
    names = _NAMES[family]
    k = len(names)
    if len(tail) < max(2, k + 1):
        nan = math.nan
        return DistributionFit(family, {p: nan for p in names}, {p: nan for p in names}, xmin, discrete, len(data),
                               len(tail), nan, nan, nan, nan, False, "too few samples above xmin", tail, below,
                               np.zeros(0))
    nll = _nll(family, tail, xmin, discrete)
    u0 = _start(family, tail, xmin, discrete)
    if family == "power_law" and not discrete:
        u = np.array([math.log(max(len(tail) / float(np.sum(np.log(tail / xmin))), 1e-12))])  # exact MLE
        ok, message = True, "closed form"
    else:
        best = None
        for method in ("Nelder-Mead", "BFGS"):
            res = minimize(nll, u0 if best is None else best.x, method=method,
                           options={"maxiter": 4000, **({"xatol": 1e-10, "fatol": 1e-12} if method == "Nelder-Mead"
                                                        else {"gtol": 1e-8})})
            if best is None or res.fun <= best.fun:
                best = res
        u, ok, message = best.x, bool(math.isfinite(best.fun) and best.fun < 1e299), str(best.message)
    values = _to_natural(family, u)
    pointwise = logpdf(family, tail, values, xmin, discrete)
    loglik = float(np.sum(pointwise))
    # Errors: the inverse Hessian of -ln L in the natural constants.
    natural = np.array(values, dtype=np.float64)

    def nll_natural(v):
        try:
            with np.errstate(all="ignore"):
                r = -float(np.sum(logpdf(family, tail, tuple(v), xmin, discrete)))
        except (OverflowError, ValueError, ZeroDivisionError):
            return math.nan
        return r

    try:
        cov = np.linalg.inv(_hessian(nll_natural, natural))
        err = [float(math.sqrt(c)) if c > 0 else math.nan for c in np.diag(cov)]
    except np.linalg.LinAlgError:
        err = [math.nan] * k
    n_tail = len(tail)
    ks = _ks(family, tail, values, xmin, discrete) if ok else math.nan
    return DistributionFit(family, dict(zip(names, map(float, values))), dict(zip(names, err)), xmin, discrete,
                           len(data), n_tail, loglik, 2 * k - 2 * loglik, k * math.log(n_tail) - 2 * loglik, ks,
                           ok and math.isfinite(loglik), message, tail, below, pointwise)


def find_xmin(samples: Sequence[float], family: str = "power_law", discrete: bool = False,
              candidates: Optional[Sequence[float]] = None, min_tail: int = MIN_TAIL,
              max_candidates: int = 60) -> DistributionFit:
    """The fit whose xmin makes the tail's KS distance smallest (Clauset et al.), trying each
    distinct sample value that leaves at least `min_tail` samples above it (thinned to at most
    `max_candidates`, spread over the quantiles)."""
    data = np.asarray(samples, dtype=np.float64).ravel()
    if candidates is None:
        values = np.unique(data)
        counts = np.array([np.sum(data >= v) for v in values])
        values = values[counts >= min_tail]
        if len(values) > max_candidates:
            values = values[np.unique(np.linspace(0, len(values) - 1, max_candidates).round().astype(int))]
        candidates = values
    best = None
    for xmin in candidates:
        f = fit_distribution(data, family, float(xmin), discrete)
        if f.success and math.isfinite(f.ks) and (best is None or f.ks < best.ks):
            best = f
    if best is None:
        raise ValueError(f"no xmin leaves {min_tail} samples for a {family} fit")
    return best


# --- which family, and is it any good -------------------------------------------------------------

@dataclass
class DistributionComparison:
    a: str
    b: str
    loglik_ratio: float  # ln L(a) - ln L(b) over the same tail: positive favours a
    statistic: float  # Vuong's z, or the likelihood-ratio statistic 2|R| when nested
    p: float  # small: the sign of loglik_ratio is significant
    nested: bool
    favoured: Optional[str]  # a or b; None when not nested and p >= alpha (nested: the simpler then)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


_NESTED = {("power_law", "truncated_power_law"), ("exponential", "truncated_power_law")}


def compare_distributions(a: DistributionFit, b: DistributionFit, alpha: float = 0.1) -> DistributionComparison:
    """Is a or b the better description of the same tail (same xmin, same samples)?"""
    if a.xmin != b.xmin or a.n_tail != b.n_tail or a.discrete != b.discrete:
        raise ValueError("compare fits of the same samples above the same xmin")
    d = a.pointwise - b.pointwise
    R = float(np.sum(d))
    nested = (a.family, b.family) in _NESTED or (b.family, a.family) in _NESTED
    if nested:
        # The richer family gains at least nothing; without a significant gain the simpler suffices.
        richer, simpler = (a, b) if a.family == "truncated_power_law" else (b, a)
        gain = max(0.0, float(np.sum(richer.pointwise - simpler.pointwise)))
        stat = 2.0 * gain
        p = 0.5 * float(stats.chi2.sf(stat, 1)) if stat > 0 else 1.0  # lambda = 0 is on the boundary
        favoured = richer.family if p < alpha else simpler.family
    else:
        sd = float(np.std(d, ddof=0))
        stat = R / (sd * math.sqrt(len(d))) if sd > 0 else 0.0
        p = float(math.erfc(abs(stat) / math.sqrt(2.0)))
        favoured = (a.family if R > 0 else b.family) if p < alpha else None
    return DistributionComparison(a.family, b.family, R, float(stat), p, nested, favoured)


def goodness_of_fit(f: DistributionFit, resamples: int = 100, seed: int = 0, refit_xmin: bool = True,
                    min_tail: int = MIN_TAIL) -> Tuple[float, List[float]]:
    """The share p of synthetic data sets (the fit above xmin, the observed values below it)
    whose refitted KS distance is at least the observed one; p < 0.1 rules the family out."""
    if not f.success:
        raise ValueError("goodness of fit needs a successful fit")
    rng = np.random.default_rng(seed)
    share_tail = f.n_tail / f.n
    distances = []
    for _ in range(resamples):
        n_tail = int(rng.binomial(f.n, share_tail))
        n_tail = max(n_tail, min_tail)
        synthetic_tail = sample(f.family, f.values, f.xmin, f.discrete, n_tail, rng)
        synthetic_below = (rng.choice(f.below, f.n - n_tail) if len(f.below) and f.n > n_tail
                           else np.zeros(0))
        data = np.concatenate([synthetic_tail, synthetic_below])
        try:
            g = (find_xmin(data, f.family, f.discrete, min_tail=min_tail) if refit_xmin
                 else fit_distribution(data, f.family, f.xmin, f.discrete))
        except ValueError:
            continue
        if g.success and math.isfinite(g.ks):
            distances.append(g.ks)
    if not distances:
        return math.nan, []
    return float(np.mean(np.asarray(distances) >= f.ks)), distances

"""
likelihoods.py - What a measurement says about a law's value at each row: the negative log
likelihood of the data given the law's prediction `mu` there, and its derivatives.

    Binomial(successes, trials)         mu = a probability, 0 < mu < 1: survived out of seeds
    Poisson(counts, exposure=1)         mu = a rate per unit of exposure, mu > 0
    CensoredLogNormal(times, censored)  mu = the median time. Each time is log-normal about it,
                                        with one spread s shared by every row and fitted with
                                        the law (a nuisance constant, `log_spread`). A censored
                                        time is a lower bound: the run stopped before the event.

Each gives
    nll(mu, extra)       -ln L, up to a constant that depends on the data alone (so the same
                         for every law fitted to them)
    gradient(mu, extra)  (d nll / d mu per row, d nll / d extra)
    fisher(mu)           the expected information per row, d^2 nll / d mu^2 in expectation,
                         where it has a closed form (None otherwise: the fit then differentiates
                         the gradient)
    saturated()          -ln L of a model with one free value per row, where one exists; the
                         deviance is 2 (nll - saturated), a chi-square-like lack of fit on
                         n - k degrees of freedom
    n_obs                the number of independent observations behind the rows (for counts out
                         of trials, the trials), which a BIC penalty counts

Outside its domain (a probability below 0, a rate or median below 0) a law's -ln L is
continued by the quadratic Taylor expansion of ln at a tiny epsilon, so it stays finite and
smooth and rises steeply: an optimiser that steps outside is pushed back without a crash.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import log_ndtr

_EPS = 1e-12
_LOG_EPS = math.log(_EPS)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _xlog(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(ln mu, 1/mu) for mu >= EPS, continued below EPS by ln's second-order expansion."""
    m = np.asarray(mu, dtype=np.float64)
    inside = m >= _EPS
    safe = np.where(inside, m, _EPS)
    d = m - _EPS
    value = np.where(inside, np.log(safe), _LOG_EPS + d / _EPS - d * d / (2.0 * _EPS ** 2))
    slope = np.where(inside, 1.0 / safe, 1.0 / _EPS - d / _EPS ** 2)
    return value, slope


def _xlogy(k: np.ndarray, p: np.ndarray) -> np.ndarray:
    """k ln p with 0 ln 0 = 0."""
    with np.errstate(all="ignore"):
        return np.where(k > 0, k * np.log(np.where(k > 0, p, 1.0)), 0.0)


def _as_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


class Likelihood:
    """Base class: see the module docstring. `extra_names` are nuisance constants the fit
    estimates along with the law's own."""

    name = "likelihood"
    link = "identity"  # the scale on which prediction bands are built: identity, log or logit
    extra_names: Tuple[str, ...] = ()

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def n_obs(self) -> int:
        return len(self)

    def observed(self) -> np.ndarray:
        """The rows as values of mu (a fraction, a rate, a time), for starting guesses and for
        people's summaries."""
        raise NotImplementedError

    def extra_initial(self) -> List[float]:
        return []

    def nll(self, mu: np.ndarray, extra: Sequence[float] = ()) -> float:
        raise NotImplementedError

    def gradient(self, mu: np.ndarray, extra: Sequence[float] = ()) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fisher(self, mu: np.ndarray) -> Optional[np.ndarray]:
        return None

    def saturated(self) -> Optional[float]:
        return None

    def subset(self, rows: np.ndarray) -> "Likelihood":
        """The same likelihood for some of the rows (a boolean mask or indices)."""
        raise NotImplementedError

    def simulate(self, mu: np.ndarray, rng: np.random.Generator, extra: Sequence[float] = ()) -> "Likelihood":
        """Data drawn from this likelihood at mu, with the same design (trials, exposure, the
        censoring points)."""
        raise NotImplementedError

    def expected(self, mu: np.ndarray, extra: Sequence[float] = ()) -> "Likelihood":
        """The noise-free ("Asimov") data at mu, with the same design."""
        raise NotImplementedError


class Binomial(Likelihood):
    """`successes` out of `trials` at each row; mu is the probability of a success."""

    name = "binomial"
    link = "logit"

    def __init__(self, successes: Sequence[float], trials: Sequence[float]):
        k = _as_array(successes, "successes")
        n = np.broadcast_to(_as_array(trials, "trials"), k.shape).astype(np.float64)
        if len(n) != len(k):
            raise ValueError(f"successes has {len(k)} rows but trials has {len(n)}")
        if np.any(n <= 0):
            raise ValueError("every row needs at least one trial")
        if np.any(k < 0) or np.any(k > n):
            raise ValueError("successes must lie between 0 and the trials")
        self.k, self.n = k, n

    def __len__(self) -> int:
        return len(self.k)

    @property
    def n_obs(self) -> int:
        return int(round(float(np.sum(self.n))))

    def observed(self) -> np.ndarray:
        return self.k / self.n

    def nll(self, mu, extra=()):
        lp, _ = _xlog(mu)
        lq, _ = _xlog(1.0 - np.asarray(mu, dtype=np.float64))
        return float(-np.sum(self.k * lp + (self.n - self.k) * lq))

    def gradient(self, mu, extra=()):
        _, dp = _xlog(mu)
        _, dq = _xlog(1.0 - np.asarray(mu, dtype=np.float64))
        return -(self.k * dp - (self.n - self.k) * dq), np.zeros(0)

    def fisher(self, mu):
        p = np.clip(np.asarray(mu, dtype=np.float64), 1e-9, 1.0 - 1e-9)
        return self.n / (p * (1.0 - p))

    def saturated(self):
        p = self.k / self.n
        return float(-np.sum(_xlogy(self.k, p) + _xlogy(self.n - self.k, 1.0 - p)))

    def subset(self, rows):
        return Binomial(self.k[rows], self.n[rows])

    def simulate(self, mu, rng, extra=()):
        p = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)
        return Binomial(rng.binomial(self.n.astype(np.int64), p).astype(np.float64), self.n)

    def expected(self, mu, extra=()):
        return Binomial(self.n * np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0), self.n)


class Poisson(Likelihood):
    """`counts` at each row over `exposure` (1 by default); mu is the rate per unit exposure."""

    name = "poisson"
    link = "log"

    def __init__(self, counts: Sequence[float], exposure: Optional[Sequence[float]] = None):
        c = _as_array(counts, "counts")
        e = np.ones_like(c) if exposure is None else np.broadcast_to(
            _as_array(exposure, "exposure"), c.shape).astype(np.float64)
        if np.any(c < 0):
            raise ValueError("counts cannot be negative")
        if np.any(e <= 0):
            raise ValueError("every exposure must be positive")
        self.c, self.e = c, e

    def __len__(self) -> int:
        return len(self.c)

    def observed(self) -> np.ndarray:
        return self.c / self.e

    def nll(self, mu, extra=()):
        m = np.asarray(mu, dtype=np.float64) * self.e
        lm, _ = _xlog(m)
        return float(np.sum(m - self.c * lm))

    def gradient(self, mu, extra=()):
        m = np.asarray(mu, dtype=np.float64) * self.e
        _, dm = _xlog(m)
        return (1.0 - self.c * dm) * self.e, np.zeros(0)

    def fisher(self, mu):
        return self.e / np.maximum(np.asarray(mu, dtype=np.float64), 1e-300)

    def saturated(self):
        return float(np.sum(self.c - _xlogy(self.c, self.c)))

    def subset(self, rows):
        return Poisson(self.c[rows], self.e[rows])

    def simulate(self, mu, rng, extra=()):
        m = np.maximum(np.asarray(mu, dtype=np.float64), 0.0) * self.e
        return Poisson(rng.poisson(m).astype(np.float64), self.e)

    def expected(self, mu, extra=()):
        return Poisson(np.maximum(np.asarray(mu, dtype=np.float64), 0.0) * self.e, self.e)


class CensoredLogNormal(Likelihood):
    """`times` at each row, log-normal about the median mu with spread s = exp(log_spread);
    where `censored` is true the time is a lower bound (the event had not happened yet)."""

    name = "censored_lognormal"
    link = "log"
    extra_names = ("log_spread",)

    def __init__(self, times: Sequence[float], censored: Optional[Sequence[bool]] = None,
                 caps: Optional[Sequence[float]] = None):
        t = _as_array(times, "times")
        if np.any(t <= 0):
            raise ValueError("every time (or bound) must be positive")
        cens = np.zeros(len(t), dtype=bool) if censored is None else np.asarray(censored, dtype=bool).ravel()
        if len(cens) != len(t):
            raise ValueError(f"times has {len(t)} rows but censored has {len(cens)}")
        self.t, self.cens = t, cens
        self.log_t = np.log(t)
        # Where each run would have stopped (for simulating the same design): a censored row's
        # own bound, and no limit for an event unless `caps` says otherwise.
        self.caps = np.where(cens, t, np.inf) if caps is None else np.broadcast_to(
            np.asarray(caps, dtype=np.float64), t.shape).copy()

    def __len__(self) -> int:
        return len(self.t)

    @property
    def n_events(self) -> int:
        return int(np.sum(~self.cens))

    def observed(self) -> np.ndarray:
        return self.t

    def extra_initial(self) -> List[float]:
        events = self.log_t[~self.cens]
        spread = float(np.std(events)) if len(events) > 1 else 0.5
        return [math.log(max(spread, 0.05))]

    def _z(self, mu, extra):
        lm, dlm = _xlog(mu)
        s = math.exp(float(extra[0]))
        return (self.log_t - lm) / s, dlm, s

    def nll(self, mu, extra=(0.0,)):
        z, _, s = self._z(mu, extra)
        u = ~self.cens
        events = np.sum(math.log(s) + 0.5 * z[u] ** 2 + self.log_t[u] + _HALF_LOG_2PI)
        bounds = -np.sum(log_ndtr(-z[self.cens]))
        return float(events + bounds)

    def gradient(self, mu, extra=(0.0,)):
        z, dlm, s = self._z(mu, extra)
        dz = np.where(self.cens, 0.0, z)
        c = self.cens
        if np.any(c):
            # d/dz of -ln Phi(-z): the hazard phi(z) / Phi(-z), computed in logs.
            dz[c] = np.exp(-0.5 * z[c] ** 2 - _HALF_LOG_2PI - log_ndtr(-z[c]))
        d_mu = dz * (-dlm / s)
        d_log_s = float(np.sum(np.where(c, 0.0, 1.0)) - np.sum(dz * z))
        return d_mu, np.array([d_log_s])

    def subset(self, rows):
        return CensoredLogNormal(self.t[rows], self.cens[rows], self.caps[rows])

    def simulate(self, mu, rng, extra=(0.0,)):
        s = math.exp(float(extra[0]))
        draw = np.asarray(mu, dtype=np.float64) * np.exp(s * rng.standard_normal(len(self.t)))
        return CensoredLogNormal(np.minimum(draw, self.caps), draw >= self.caps, self.caps)

    def expected(self, mu, extra=(0.0,)):
        raise ValueError("a censored likelihood has no noise-free data set; use simulations")


def make(likelihood: str, y, *, trials=None, exposure=None, censored=None) -> Likelihood:
    """The likelihood named `likelihood` ("binomial", "poisson", "censored") for data y."""
    if likelihood == "binomial":
        if trials is None:
            raise ValueError("a binomial fit needs trials (y is the successes)")
        return Binomial(y, trials)
    if likelihood == "poisson":
        return Poisson(y, exposure)
    if likelihood in ("censored", "censored_lognormal"):
        return CensoredLogNormal(y, censored)
    raise ValueError(f"unknown likelihood {likelihood!r}; known: gaussian, binomial, poisson, censored")

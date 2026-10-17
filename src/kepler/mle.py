"""
mle.py - Maximum likelihood for a law: its constants, their covariance, and -ln L.

A `Model` is a law evaluated over bound data: mu(params) at every row and the Jacobian
d mu / d params. `maximise` minimises a likelihood's -ln L (kepler.likelihoods) over the law's
constants and the likelihood's nuisance constants (a log-normal's spread):

1. BFGS (scipy) from the starting constants, each scaled by its starting value so that a slope
   of 30 and a threshold of 0.3 move alike, with the analytic gradient J^T (d nll / d mu);
2. damped Newton steps (Levenberg-Marquardt) on the information matrix to finish: Fisher's
   expected information J^T W J where the likelihood has one, otherwise the derivative of the
   analytic gradient by central differences;
3. the covariance is the inverse information at the solution. If the information is singular
   (its smallest eigenvalue, after scaling to unit diagonal, below 1e-12 of its largest), some
   constant is not identifiable from these data, and the covariance is infinite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from kepler.likelihoods import Likelihood

SINGULAR = 1e-12


@dataclass
class Model:
    """mu at every row and d mu / d params (rows x constants), as functions of the constants."""

    value: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray]


@dataclass
class Solution:
    params: np.ndarray
    extra: np.ndarray
    nll: float
    covariance: np.ndarray  # over the law's constants, then the nuisance constants
    singular: bool
    converged: bool
    message: str


def _scales(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(values), 1e-3)


class _Objective:
    """-ln L and its gradient over (law constants, nuisance constants)."""

    def __init__(self, model: Model, lik: Likelihood, k: int):
        self.model, self.lik, self.k = model, lik, k

    def __call__(self, theta: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        params, extra = theta[: self.k], theta[self.k:]
        with np.errstate(all="ignore"):
            mu = self.model.value(params)
            if not np.all(np.isfinite(mu)):
                return math.inf, None
            value = self.lik.nll(mu, extra)
            if not math.isfinite(value):
                return math.inf, None
            d_mu, d_extra = self.lik.gradient(mu, extra)
            jac = self.model.jacobian(params)
            grad = np.concatenate([jac.T @ d_mu, np.asarray(d_extra, dtype=np.float64)])
        if not np.all(np.isfinite(grad)):
            return math.inf, None
        return float(value), grad

    def information(self, theta: np.ndarray) -> Optional[np.ndarray]:
        params, extra = theta[: self.k], theta[self.k:]
        with np.errstate(all="ignore"):
            if len(extra) == 0:
                mu = self.model.value(params)
                w = self.lik.fisher(mu)
                if w is not None:
                    jac = self.model.jacobian(params)
                    info = jac.T @ (w[:, None] * jac)
                    return info if np.all(np.isfinite(info)) else None
            m = len(theta)
            info = np.zeros((m, m))
            for j in range(m):
                h = 1e-6 * max(abs(theta[j]), 1e-2)
                up, down = theta.copy(), theta.copy()
                up[j] += h
                down[j] -= h
                _, g_up = self(up)
                _, g_down = self(down)
                if g_up is None or g_down is None:
                    return None
                info[:, j] = (g_up - g_down) / (2.0 * h)
            info = 0.5 * (info + info.T)
        return info if np.all(np.isfinite(info)) else None


def _bfgs(obj: _Objective, theta0: np.ndarray, max_iter: int) -> np.ndarray:
    scale = _scales(theta0)

    def scaled(u):
        value, grad = obj(u * scale)
        if grad is None:
            return 1e300, np.zeros_like(u)
        return value, grad * scale

    with np.errstate(all="ignore"):
        res = minimize(scaled, theta0 / scale, jac=True, method="BFGS",
                       options={"maxiter": max_iter, "gtol": 1e-9})
    theta = res.x * scale
    value, _ = obj(theta)
    start, _ = obj(theta0)
    return theta if value <= start else theta0


def _newton(obj: _Objective, theta: np.ndarray, iterations: int = 200) -> Tuple[np.ndarray, bool]:
    value, grad = obj(theta)
    if grad is None:
        return theta, False
    damping = 1e-6
    for _ in range(iterations):
        info = obj.information(theta)
        if info is None:
            return theta, False
        diag = np.maximum(np.abs(np.diag(info)), 1e-300)
        try:
            step = np.linalg.solve(info + damping * np.diag(diag), -grad)
        except np.linalg.LinAlgError:
            damping *= 10.0
            if damping > 1e12:
                return theta, False
            continue
        decrement = float(-grad @ step)
        if 0 <= decrement < 1e-14 * max(1.0, abs(value)):
            return theta, True
        new = theta + step
        new_value, new_grad = obj(new)
        if new_grad is not None and new_value <= value:
            done = value - new_value <= 1e-13 * max(1.0, abs(value))
            theta, value, grad = new, new_value, new_grad
            damping = max(damping / 10.0, 1e-12)
            if done and np.all(np.abs(step) <= 1e-9 * _scales(theta)):
                return theta, True
        else:
            damping *= 10.0
            if damping > 1e12:
                return theta, bool(np.max(np.abs(grad) * _scales(theta)) < 1e-6 * max(1.0, abs(value)))
    return theta, False


def _invert(info: Optional[np.ndarray], m: int) -> Tuple[np.ndarray, bool]:
    if m == 0:
        return np.zeros((0, 0)), False
    if info is None or not np.all(np.isfinite(info)):
        return np.full((m, m), np.inf), True
    d = np.sqrt(np.maximum(np.abs(np.diag(info)), 1e-300))
    scaled = info / np.outer(d, d)
    vals, vecs = np.linalg.eigh(scaled)
    if vals[0] <= SINGULAR * max(vals[-1], 1e-300):
        return np.full((m, m), np.inf), True
    inv = (vecs / vals) @ vecs.T
    return inv / np.outer(d, d), False


def maximise(model: Model, lik: Likelihood, p0: Sequence[float],
             extra0: Optional[Sequence[float]] = None, max_iter: int = 2000) -> Solution:
    """The constants (and nuisance constants) that maximise the likelihood, from p0."""
    k = len(p0)
    extra0 = list(lik.extra_initial() if extra0 is None else extra0)
    theta0 = np.array([*p0, *extra0], dtype=np.float64)
    obj = _Objective(model, lik, k)
    start, _ = obj(theta0)
    if not math.isfinite(start):
        return Solution(theta0[:k], theta0[k:], math.inf, np.full((len(theta0),) * 2, np.inf), False, False,
                        "the law is not finite at the starting constants")
    theta = _bfgs(obj, theta0, max_iter)
    theta, converged = _newton(obj, theta)
    if not converged:
        # A second start: Nelder-Mead is slow but ignores a misleading gradient.
        scale = _scales(theta0)
        with np.errstate(all="ignore"):
            res = minimize(lambda u: obj(u * scale)[0], theta0 / scale, method="Nelder-Mead",
                           options={"maxiter": 200 * len(theta0), "xatol": 1e-10, "fatol": 1e-12})
        other, other_ok = _newton(obj, _bfgs(obj, res.x * scale, max_iter))
        here, there = obj(theta)[0], obj(other)[0]
        tol = 1e-9 * max(1.0, abs(here))
        if there < here - tol or (other_ok and there <= here + tol):
            theta, converged = other, other_ok
    value, _ = obj(theta)
    cov, singular = _invert(obj.information(theta), len(theta))
    message = "converged" if converged else "stopped before converging"
    return Solution(theta[:k], theta[k:], float(value), cov, singular, converged, message)

"""
normalize.py - Map real-world data into the specialists' training frame and back.

Why this exists
---------------
The specialist encoders see a fixed-length summary of the data, and every
summary function hard-clips its values (to +-10 for algebraic and time-series
features). The generators also draw x from narrow fixed ranges. Real data in
other units therefore saturates the features and the models never see the
shape of the curve. This module applies an affine map per domain before
inference and undoes it on the predicted equation afterwards.

Conventions
-----------
    x_norm = (x - x_shift) / x_scale
    y_norm = (y - y_shift) / y_scale

Per-domain policy (chosen to land inside each model's training distribution):

    polynomial      x -> [-5, 5]        y -> zero mean, unit std
    transcendental  x -> [0.1, 10], or a pure scaling x -> x*10/max(x) when the
                                        data starts at or near the origin
                                        y -> unit std, lifted to >= 0 only if
                                             negatives are present (the family
                                             is trained on positive curves)
    ar              x untouched         y -> zero mean, unit std
                                             (AR coefficients are invariant)
    linear_ode      t -> [0, 10]        y -> divided by max|y|, no shift
                                             (homogeneous ODE: shift-variant,
                                             scale-invariant)
    physics_laws    disabled            (multivariate path takes a table, not
                                             a curve; left for a later change)

Inverse maps:
    algebraic      y = y_scale * f((x - x_shift)/x_scale) + y_shift
    ar             coefficients unchanged
    linear_ode     a -> a / t_scale, C1 -> C1 / t_scale, C2 -> C2 / t_scale**2
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import sympy

from expression import X

TRAINING_X_RANGE: Dict[str, Tuple[float, float]] = {
    "polynomial": (-5.0, 5.0),
    "transcendental": (0.1, 10.0),
}
TRAINING_T_SPAN = 10.0

_EPS = 1e-12


@dataclass
class Normalization:
    """Affine normalisation of (x, y) with its inverse for predicted equations."""

    model_name: str
    domain: str
    enabled: bool = True
    x_shift: float = 0.0
    x_scale: float = 1.0
    y_shift: float = 0.0
    y_scale: float = 1.0
    x_assumed: Optional[str] = None  # set when x was fabricated by the caller
    # Equivalent form used for readable output: x_norm = x_lo + (x - x_origin) / x_scale
    x_origin: float = 0.0
    x_lo: float = 0.0

    # ------------------------------------------------------------------ data
    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not self.enabled:
            return x, y
        return (x - self.x_shift) / self.x_scale, (y - self.y_shift) / self.y_scale

    def inverse_x(self, x_norm: np.ndarray) -> np.ndarray:
        return np.asarray(x_norm) * self.x_scale + self.x_shift

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return np.asarray(y_norm) * self.y_scale + self.y_shift

    # ------------------------------------------------------------ equations
    def inverse_expression(self, expr: sympy.Expr, x: sympy.Symbol = X) -> sympy.Expr:
        """Rewrite an expression predicted in normalised coordinates in original units."""
        if not self.enabled:
            return expr
        x_norm = (x - sympy.Float(self.x_shift)) / sympy.Float(self.x_scale)
        inner = expr.xreplace({x: x_norm}) if x in expr.free_symbols else expr
        return sympy.Float(self.y_scale) * inner + sympy.Float(self.y_shift)

    def readable_inverse(self, expr: sympy.Expr, x: sympy.Symbol = X) -> sympy.Expr:
        """
        Original-units form of a normalised expression, written for people.

        Polynomials are expanded exactly in x. Other families keep the data's
        x origin as a literal offset, e.g. exp(-0.25*(x - 2000)), because
        SymPy would otherwise fold the offset into an enormous leading
        constant. The returned expression is still SymPy-parseable.
        """
        if not self.enabled:
            return expr
        if expr.is_polynomial(x):
            return sympy.expand(self.inverse_expression(expr, x))

        if abs(self.x_origin) < _EPS:
            u: sympy.Expr = x
        else:
            sign = "-" if self.x_origin > 0 else "+"
            u = sympy.Symbol(f"(x {sign} {abs(self.x_origin):.10g})")
        x_norm = u if abs(self.x_scale - 1.0) < _EPS else u / sympy.Float(self.x_scale)
        if abs(self.x_lo) > _EPS:
            x_norm = sympy.Float(self.x_lo) + x_norm
        inner = expr.xreplace({x: x_norm}) if x in expr.free_symbols else expr
        # Fold scale factors through powers and logs: (0.1*x)**1.7 -> c*x**1.7
        inner = sympy.expand_log(sympy.expand_power_base(inner, force=True), force=True)
        if abs(self.y_scale - 1.0) > _EPS:
            inner = sympy.Float(self.y_scale) * inner
        if abs(self.y_shift) > _EPS:
            inner = inner + sympy.Float(self.y_shift)
        return inner

    def inverse_ode(self, params: Dict[str, float]) -> Dict[str, float]:
        """Convert ODE coefficients from normalised time to original time units."""
        if not self.enabled:
            return dict(params)
        out = dict(params)
        tau = self.x_scale
        if params.get("order") == 1:
            out["a"] = params["a"] / tau
        elif params.get("order") == 2:
            out["C1"] = params["C1"] / tau
            out["C2"] = params["C2"] / tau ** 2
        return out

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        for key in ("x_shift", "x_scale", "y_shift", "y_scale", "x_origin", "x_lo"):
            d[key] = float(d[key])
        return d


def _affine_to_range(values: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    """Return (shift, scale) so that (values - shift)/scale spans [lo, hi]."""
    vmin, vmax = float(np.min(values)), float(np.max(values))
    span = vmax - vmin
    if span < _EPS:
        return vmin - lo, 1.0
    scale = span / (hi - lo)
    shift = vmin - lo * scale
    return shift, scale


def _safe_scale(value: float) -> float:
    return float(value) if np.isfinite(value) and value > _EPS else 1.0


def fit_normalization(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    domain: str,
    enabled: bool = True,
    x_assumed: Optional[str] = None,
) -> Normalization:
    """Choose the affine map for this model's domain from the data."""
    norm = Normalization(model_name=model_name, domain=domain, enabled=False, x_assumed=x_assumed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if not enabled or len(y) == 0 or domain == "multivariate":
        return norm

    norm.enabled = True

    if domain == "algebraic":
        lo, hi = TRAINING_X_RANGE.get(model_name, (-5.0, 5.0))
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if model_name == "transcendental" and x_min >= 0 and x_max > _EPS and x_min <= (x_max - x_min):
            # Origin-anchored data: a pure scaling keeps power laws and logarithms
            # in their own family (a*(s*x)**b is still a power law in x).
            norm.x_shift, norm.x_scale = 0.0, x_max / hi
            norm.x_origin, norm.x_lo = 0.0, 0.0
        else:
            norm.x_shift, norm.x_scale = _affine_to_range(x, lo, hi)
            norm.x_origin, norm.x_lo = x_min, lo
        if model_name == "transcendental":
            y_min = float(np.min(y))
            norm.y_shift = y_min if y_min < 0 else 0.0
        else:
            norm.y_shift = float(np.mean(y))
        norm.y_scale = _safe_scale(float(np.std(y)))

    elif domain == "timeseries":
        norm.y_shift = float(np.mean(y))
        norm.y_scale = _safe_scale(float(np.std(y)))

    elif domain == "dynamical":
        norm.x_shift, norm.x_scale = _affine_to_range(x, 0.0, TRAINING_T_SPAN)
        norm.x_origin, norm.x_lo = float(np.min(x)), 0.0
        norm.y_shift = 0.0
        norm.y_scale = _safe_scale(float(np.max(np.abs(y))))

    else:
        norm.enabled = False

    return norm

"""rank(): BIC with an Occam window, eligibility, and refusing incomparable fits."""
import math

import numpy as np
import pytest
import sympy

from kepler import GROWTH_LAWS, LAWS, bic, fit, rank

L = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128], dtype=float)

TRUE = {
    "constant": {"a": 50.0},
    "logarithmic": {"a": 10.0, "b": 5.0},
    "linear": {"a": 2.0, "b": 3.0},
    "power": {"a": 27.0, "b": 0.6},
    "exponential": {"a": 3.0, "b": 0.03},
    "exponential_area": {"a": 2.0, "b": 0.0002},
}


def _data(name, noise=0.01, seed=0):
    law = LAWS[name]
    expr, symbols = law.parametrized()
    f = sympy.lambdify([sympy.Symbol("x"), *symbols], expr, "numpy")
    y_true = np.broadcast_to(f(L, *[TRUE[name][p] for p in law.params]), L.shape).astype(float)
    rng = np.random.default_rng(seed)
    return y_true * (1 + noise * rng.standard_normal(len(L))), noise * y_true


def test_bic_formula():
    assert bic(2.0, 10, 3) == pytest.approx(10 * math.log(0.2) + 3 * math.log(10))
    assert math.isfinite(bic(0.0, 10, 1))


@pytest.mark.parametrize("name", GROWTH_LAWS)
def test_rank_names_the_true_law(name):
    y, sigma = _data(name)
    ranking = rank([fit(law, L, y, sigma=sigma) for law in GROWTH_LAWS])
    assert ranking.best is not None and ranking.best.name == name, ranking.note
    assert ranking.confident
    assert ranking.entries[0] is ranking.best
    assert min(e.delta_bic for e in ranking.entries if e.delta_bic is not None) == 0.0


def test_occam_window_prefers_fewer_constants():
    y, sigma = _data("power", noise=0.02, seed=3)
    fits = [fit("power", L, y, sigma=sigma), fit("shifted_power", L, y, sigma=sigma)]
    ranking = rank(fits)
    by_name = {e.name: e for e in ranking.entries}
    assert abs(by_name["power"].bic - by_name["shifted_power"].bic) <= 6.0
    assert ranking.best.name == "power"


def test_occam_window_can_be_narrowed():
    y, sigma = _data("power", noise=0.02, seed=3)
    fits = [fit("power", L, y, sigma=sigma), fit("shifted_power", L, y, sigma=sigma)]
    lowest = min(fits, key=lambda f: f.bic)
    assert rank(fits, occam_window=0.0).best.name == lowest.name


def test_ineligible_fits_keep_their_reasons_and_none_fits_is_an_answer():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    ranking = rank([fit("logarithmic", x, y), fit("exponential", x, y)])
    reasons = {e.name: e.reason for e in ranking.entries}
    assert "x > 0" in reasons["logarithmic"]
    assert ranking.best is None or not ranking.confident
    if ranking.best is None:
        assert ranking.note.startswith("no law fits")


def test_low_r2_is_reported_as_low_confidence():
    rng = np.random.default_rng(5)
    x = np.arange(1.0, 21.0)
    y = 0.05 * x + rng.standard_normal(len(x))
    ranking = rank([fit("linear", x, y)], min_r2=0.9)
    assert ranking.best is not None and not ranking.confident
    assert ranking.note.startswith("low confidence")


def test_incomparable_fits_are_refused():
    y, sigma = _data("linear")
    with pytest.raises(ValueError, match="different data"):
        rank([fit("linear", L, y), fit("linear", L[:-1], y[:-1])])
    with pytest.raises(ValueError, match="spaces or with different weighting"):
        rank([fit("linear", L, y), fit("linear", L, y, sigma=sigma)])
    with pytest.raises(ValueError, match="spaces or with different weighting"):
        rank([fit("power", L, y), fit("power", L, y, space="log")])
    with pytest.raises(ValueError, match="n="):
        rank([fit("linear", L, y)], n=3)


def test_to_dict():
    import json

    y, sigma = _data("linear")
    data = rank([fit(name, L, y, sigma=sigma) for name in ("linear", "power")]).to_dict()
    json.dumps(data)
    assert data["best"] == "linear" and len(data["entries"]) == 2


def test_a_constant_is_never_ineligible_for_rounding():
    # Its R^2 is 0 by construction; rounding can make it -1e-16, which is no divergence.
    rng = np.random.default_rng(3)
    x = np.array([8, 10, 12, 16, 20, 24, 32], dtype=float)
    for _ in range(50):
        y = 60 * np.exp(0.1 * rng.standard_normal(len(x)))
        sigma = 0.1 * y * (1 + rng.random(len(x)))
        ranking = rank([fit("constant", x, y, sigma=sigma, space="log")])
        assert ranking.entries[0].eligible, ranking.entries[0]

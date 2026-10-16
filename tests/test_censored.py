"""Times that are sometimes only bounds: a run that hit its cap before the event says the time is
longer than the cap, not equal to it. The censored log-normal likelihood uses both kinds of row;
dropping the bounds (or treating them as times) biases a law low. Known answers, seeded."""
import math

import numpy as np
import pytest

from kepler import GROWTH_LAWS, fit, predict, rank
from kepler.likelihoods import CensoredLogNormal

SIZES = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128], dtype=float)
TRUTH = {"a": 27.0, "b": 0.35}
SPREAD = 0.5


def draw(seed, seeds=30, cap=120.0, truth=TRUTH, spread=SPREAD):
    """One time per seed per size, log-normal about a*L^b; a run stops at `cap`."""
    rng = np.random.default_rng(seed)
    L = np.repeat(SIZES, seeds)
    t = truth["a"] * L ** truth["b"] * np.exp(spread * rng.standard_normal(len(L)))
    censored = t > cap
    return L, np.where(censored, cap, np.ceil(t)), censored


def test_a_power_law_through_heavy_censoring():
    L, t, censored = draw(1)
    assert 0.2 < censored.mean() < 0.5 and censored[L == 128].mean() > 0.5  # most large runs hit the cap
    f = fit("power", L, t, likelihood="censored", censored=censored)
    assert f.success and not f.ill_conditioned, f.message
    for p, v in TRUTH.items():
        assert f.value(p) == pytest.approx(v, abs=4 * f.error(p)), p
    # The spread is a fitted nuisance constant, counted by the BIC.
    assert math.exp(f.nuisance["log_spread"]) == pytest.approx(SPREAD, abs=4 * f.nuisance_stderr["log_spread"] * SPREAD)
    assert f.n_fitted == 3 and f.bic == pytest.approx(f.neg2loglik + 3 * math.log(len(t)))
    assert f.link == "log" and f.likelihood == "censored_lognormal"


def test_dropping_the_bounds_biases_the_exponent_low():
    L, t, censored = draw(1)
    honest = fit("power", L, t, likelihood="censored", censored=censored)
    dropped = fit("power", L[~censored], t[~censored], likelihood="censored")
    as_times = fit("power", L, t, likelihood="censored")  # bounds read as event times
    assert dropped.value("b") < TRUTH["b"] - 4 * dropped.error("b")
    assert as_times.value("b") < TRUTH["b"] - 4 * as_times.error("b")
    assert abs(honest.value("b") - TRUTH["b"]) < 3 * honest.error("b")


def test_error_bars_cover_the_truth_as_often_as_they_claim():
    hits = 0
    for seed in range(80):
        L, t, censored = draw(100 + seed, seeds=15)
        f = fit("power", L, t, likelihood="censored", censored=censored)
        hits += abs(f.value("b") - TRUTH["b"]) <= 1.96 * f.error("b")
    assert 0.86 <= hits / 80 <= 0.99


def test_rank_sees_the_law_through_the_censoring():
    L, t, censored = draw(2, truth={"a": 2.0, "b": 1.0}, cap=150.0)
    fits = [fit(name, L, t, likelihood="censored", censored=censored) for name in GROWTH_LAWS]
    ranking = rank(fits)
    assert ranking.best.name in ("linear", "power")
    assert {e.name: e for e in ranking.entries}["constant"].delta_bic > 100


def test_a_band_for_a_time_is_positive():
    L, t, censored = draw(3)
    f = fit("power", L, t, likelihood="censored", censored=censored)
    T, low, high = predict(f, [256.0, 512.0])
    assert np.all((0 < low) & (low < T) & (T < high))


def test_when_every_run_hit_the_cap_nothing_is_measured():
    L = np.repeat(SIZES, 5)
    f = fit("power", L, np.full(len(L), 100.0), likelihood="censored", censored=np.ones(len(L), bool))
    assert not f.success or f.ill_conditioned


def test_simulating_the_same_design_keeps_its_caps():
    lik = CensoredLogNormal([5.0, 50.0, 50.0], [False, True, True], caps=[50.0, 50.0, 50.0])
    rng = np.random.default_rng(0)
    again = lik.simulate(np.array([1000.0, 1000.0, 1000.0]), rng, [-3.0])
    assert np.all(again.cens) and np.all(again.t == 50.0)

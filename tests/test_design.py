"""Information criteria with known errors, overdispersion in rank(), and experiment design."""
import math

import numpy as np
import pytest

from kepler import fit, information, rank, separation

L = np.geomspace(8, 128, 10)
POWER = {"a": 27.0, "b": 0.35}


def _power(rel=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = 27.0 * L ** 0.35 * np.exp(rel * rng.standard_normal(len(L)))
    return y, rel * y


def test_known_errors_give_the_chi_square_bic():
    y, sigma = _power()
    f = fit("power", L, y, sigma=sigma, space="log", absolute_sigma=True)
    assert f.neg2loglik == pytest.approx(f.rss)
    assert f.bic == pytest.approx(f.rss + 2 * math.log(len(L)))
    assert f.aicc == pytest.approx(f.rss + 4 + 12 / (len(L) - 3))


def test_unknown_errors_keep_the_profiled_form():
    y, sigma = _power()
    f = fit("power", L, y, sigma=sigma, space="log")
    n = len(L)
    assert f.neg2loglik == pytest.approx(n * math.log(f.rss / n))
    assert information(f.neg2loglik, n, 2)[0] == pytest.approx(f.bic)


def test_rank_refuses_mixed_error_models():
    y, sigma = _power()
    with pytest.raises(ValueError, match="error models"):
        rank([fit("power", L, y, sigma=sigma), fit("linear", L, y, sigma=sigma, absolute_sigma=True)])


def test_overdispersion_shrinks_differences_when_errors_are_too_small():
    y, _ = _power(rel=0.15, seed=3)
    small = 0.03 * y  # stated errors five times too small
    fits = [fit(name, L, y, sigma=small, space="log", absolute_sigma=True, inflate=True)
            for name in ("power", "logarithmic", "linear")]
    plain = [fit(name, L, y, sigma=small, space="log", absolute_sigma=True)
             for name in ("power", "logarithmic", "linear")]
    r_inflated, r_plain = rank(fits), rank(plain)
    assert r_inflated.overdispersion > 4 and r_plain.overdispersion == 1.0
    best = min(f.rss / f.dof for f in fits)
    assert r_inflated.overdispersion == pytest.approx(best)
    gap = {e.name: e.delta_bic for e in r_inflated.entries}
    raw = {e.name: e.delta_bic for e in r_plain.entries}
    assert max(gap.values()) < max(raw.values()) / 4
    assert "scatter" in r_inflated.note


def test_overdispersion_is_one_when_errors_are_honest():
    y, sigma = _power(rel=0.05, seed=4)
    fits = [fit(name, L, y, sigma=sigma, space="log", absolute_sigma=True, inflate=True)
            for name in ("power", "logarithmic")]
    assert rank(fits).overdispersion == pytest.approx(max(1.0, min(f.rss / f.dof for f in fits)))


def test_separation_of_the_prototypes_case():
    s = separation("power", "logarithmic", L, relative=0.10, space="log", truth_params=POWER)
    assert s.expected_delta_bic == pytest.approx(1.73, abs=0.05)
    assert not s.sufficient and s.seed_factor == pytest.approx(6.0 / s.expected_delta_chi2)
    # With that many times the seeds (errors / sqrt(factor)), the expectation reaches the margin.
    more = separation("power", "logarithmic", L, relative=0.10 / math.sqrt(s.seed_factor),
                      space="log", truth_params=POWER)
    assert more.expected_delta_bic == pytest.approx(6.0, rel=1e-3)


def test_simulation_agrees_with_the_expectation():
    s = separation("power", "logarithmic", L, relative=0.05, space="log", truth_params=POWER,
                   simulations=300, seed=2)
    assert s.simulations == 300
    assert s.delta_bic_quantiles[1] == pytest.approx(s.expected_delta_bic, abs=1.5)
    assert 0.3 < s.p_decisive < 0.75 and s.p_wrong < 0.02
    again = separation("power", "logarithmic", L, relative=0.05, space="log", truth_params=POWER,
                       simulations=300, seed=2)
    assert again.p_decisive == s.p_decisive


def test_a_rival_that_nests_the_truth_cannot_be_separated_by_seeds():
    s = separation("power", "shifted_power", L, relative=0.05, space="log", truth_params=POWER)
    assert s.expected_delta_chi2 < 1e-6 and s.seed_factor is None
    assert s.expected_delta_bic == pytest.approx(math.log(len(L)))


def test_an_easy_design_is_reported_sufficient():
    s = separation("exponential", "power", L, relative=0.05, space="log",
                   truth_params={"a": 20.0, "b": 0.05}, simulations=50)
    assert s.sufficient and s.seed_factor < 1 and s.p_decisive > 0.95


def test_a_fit_can_be_the_truth():
    y, sigma = _power()
    truth = fit("power", L, y, sigma=sigma, space="log", absolute_sigma=True)
    s = separation(truth, "logarithmic", L, relative=0.05, space="log")
    assert s.truth == "power" and s.expected_delta_bic > 0


@pytest.mark.parametrize("kwargs, message", [
    ({"relative": 0.1, "sigma": 1.0}, "exactly one"),
    ({}, "exactly one"),
    ({"relative": 0.0}, "positive"),
])
def test_bad_designs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        separation("power", "logarithmic", L, truth_params=POWER, **kwargs)
    with pytest.raises(ValueError, match="constants"):
        separation("power", "logarithmic", L, relative=0.1)

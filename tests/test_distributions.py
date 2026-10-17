"""Distributions of sizes and times: maximum likelihood above xmin, xmin by the KS distance, and
tests between families. Known answers from seeded draws."""
import math

import numpy as np
import pytest

from kepler import compare_distributions, find_xmin, fit_distribution, goodness_of_fit
from kepler.distributions import cdf, sample


def test_a_continuous_power_law_exponent_and_its_error():
    rng = np.random.default_rng(1)
    x = sample("power_law", (2.5,), 1.0, False, 2000, rng)
    f = fit_distribution(x, "power_law", xmin=1.0)
    assert f.params["alpha"] == pytest.approx(2.5, abs=4 * f.stderr["alpha"])
    assert f.stderr["alpha"] == pytest.approx(1.5 / math.sqrt(2000), rel=0.1)
    assert f.ks < 0.03


def test_xmin_is_found_where_the_tail_starts():
    rng = np.random.default_rng(2)
    body = rng.uniform(1.0, 5.0, 1000)
    tail = sample("power_law", (2.5,), 5.0, False, 1000, rng)
    f = find_xmin(np.concatenate([body, tail]), "power_law")
    assert 4.0 <= f.xmin <= 8.0
    assert f.params["alpha"] == pytest.approx(2.5, abs=4 * f.stderr["alpha"])
    assert f.n == 2000 and f.n_tail < 1300


def test_a_discrete_power_law_with_the_zeta_normaliser():
    x = np.random.default_rng(3).zipf(2.2, 3000).astype(float)
    f = fit_distribution(x, "power_law", xmin=1, discrete=True)
    assert f.params["alpha"] == pytest.approx(2.2, abs=4 * f.stderr["alpha"])
    assert cdf("power_law", np.array([1.0]), (2.2,), 1.0, True)[0] == pytest.approx(1 / 1.4905, rel=1e-3)


def test_a_geometric_and_a_rounded_lognormal():
    rng = np.random.default_rng(4)
    geometric = rng.geometric(0.2, 3000).astype(float)  # p(k) = 0.2 * 0.8^(k-1), k >= 1
    f = fit_distribution(geometric, "exponential", xmin=1, discrete=True)
    assert f.params["lambda"] == pytest.approx(-math.log(0.8), abs=4 * f.stderr["lambda"])
    rounded = np.floor(np.exp(3.0 + 0.6 * rng.standard_normal(3000)) + 0.5)
    rounded = rounded[rounded >= 1]
    g = fit_distribution(rounded, "lognormal", xmin=1, discrete=True)
    assert g.params["mu"] == pytest.approx(3.0, abs=4 * g.stderr["mu"])
    assert g.params["sigma"] == pytest.approx(0.6, abs=4 * g.stderr["sigma"])


def test_vuong_tells_a_lognormal_from_a_power_law():
    x = np.exp(2.0 + 1.0 * np.random.default_rng(5).standard_normal(3000))
    a = fit_distribution(x, "lognormal", xmin=float(np.min(x)))
    b = fit_distribution(x, "power_law", xmin=float(np.min(x)))
    result = compare_distributions(a, b)
    assert not result.nested and result.loglik_ratio > 0
    assert result.favoured == "lognormal" and result.p < 0.01


def test_a_cutoff_is_seen_only_when_it_is_there():
    rng = np.random.default_rng(6)
    cut = sample("truncated_power_law", (1.5, 0.01), 1.0, False, 3000, rng)
    plain = fit_distribution(cut, "power_law", 1.0)
    rich = fit_distribution(cut, "truncated_power_law", 1.0)
    assert rich.params["lambda"] == pytest.approx(0.01, abs=4 * rich.stderr["lambda"])
    assert compare_distributions(plain, rich).favoured == "truncated_power_law"
    pure = sample("power_law", (2.0,), 1.0, False, 3000, rng)
    verdict = compare_distributions(fit_distribution(pure, "power_law", 1.0),
                                    fit_distribution(pure, "truncated_power_law", 1.0))
    assert verdict.nested and verdict.p > 0.05 and verdict.favoured == "power_law"


def test_goodness_of_fit_rules_out_the_wrong_family_only():
    rng = np.random.default_rng(7)
    right = fit_distribution(sample("power_law", (2.5,), 1.0, False, 500, rng), "power_law", 1.0)
    p_right, _ = goodness_of_fit(right, resamples=60, seed=1, refit_xmin=False)
    wrong = fit_distribution(1.0 + rng.exponential(2.0, 500), "power_law", 1.0)
    p_wrong, _ = goodness_of_fit(wrong, resamples=60, seed=1, refit_xmin=False)
    assert p_right > 0.1 and p_wrong < 0.05


def test_bad_samples_are_refused():
    with pytest.raises(ValueError, match="positive"):
        fit_distribution([0.0, 1.0, 2.0], "power_law")
    with pytest.raises(ValueError, match="integers"):
        fit_distribution([1.5, 2.0], "power_law", discrete=True)
    with pytest.raises(ValueError, match="unknown family"):
        fit_distribution([1.0, 2.0], "gamma")
    assert not fit_distribution([1.0, 2.0, 3.0], "power_law", xmin=3.0).success

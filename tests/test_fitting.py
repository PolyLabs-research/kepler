"""fit() and predict() against synthetic data whose true law is known."""
import math

import numpy as np
import pytest
from scipy.optimize import curve_fit

from kepler import fit, predict, predict_covariance

L = np.array([6, 8, 10, 12, 16, 20, 24, 32, 40], dtype=float)


def power_data(seed=1, noise=0.03):
    rng = np.random.default_rng(seed)
    y = 27.0 * L ** 0.35 * (1 + noise * rng.standard_normal(len(L)))
    return y, noise * y


def test_power_law_constants_and_errors_match_curve_fit():
    y, sigma = power_data()
    result = fit("power", L, y, sigma=sigma)
    assert result.success and not result.ill_conditioned
    popt, pcov = curve_fit(lambda x, a, b: a * x ** b, L, y, p0=[20, 0.4], sigma=sigma)
    assert result.params == pytest.approx(tuple(popt), rel=1e-6)
    assert result.stderr == pytest.approx(tuple(np.sqrt(np.diag(pcov))), rel=1e-4)
    assert result.param_names == ("a", "b")
    assert result.value("b") == pytest.approx(0.35, abs=3 * result.error("b"))
    assert result.n == len(L) and result.dof == len(L) - 2


def test_absolute_sigma_matches_curve_fit():
    y, sigma = power_data(seed=2)
    result = fit("power", L, y, sigma=sigma, absolute_sigma=True)
    _, pcov = curve_fit(lambda x, a, b: a * x ** b, L, y, p0=[20, 0.4], sigma=sigma, absolute_sigma=True)
    assert result.stderr == pytest.approx(tuple(np.sqrt(np.diag(pcov))), rel=1e-4)


def test_unweighted_fit_matches_curve_fit():
    y, _ = power_data(seed=3)
    result = fit("power", L, y)
    popt, pcov = curve_fit(lambda x, a, b: a * x ** b, L, y, p0=[20, 0.4])
    assert result.params == pytest.approx(tuple(popt), rel=1e-6)
    assert result.stderr == pytest.approx(tuple(np.sqrt(np.diag(pcov))), rel=1e-4)
    assert not result.weighted and result.rss == pytest.approx(result.rmse ** 2 * len(L))
    assert result.r2_fit == pytest.approx(result.r2)


def test_weights_pull_the_fit_toward_precise_points():
    x = np.arange(1.0, 11.0)
    y = 2.0 * x + 1.0
    y[-1] += 8.0  # one bad point
    loose = np.ones_like(x)
    loose[-1] = 100.0
    plain = fit("linear", x, y)
    weighted = fit("linear", x, y, sigma=loose)
    assert abs(weighted.value("a") - 2.0) < abs(plain.value("a") - 2.0) / 10
    assert weighted.weighted


def test_log_space_fit_of_multiplicative_noise():
    rng = np.random.default_rng(4)
    x = np.geomspace(4, 512, 12)
    y = 3.0 * x ** 1.5 * np.exp(0.05 * rng.standard_normal(len(x)))
    result = fit("power", x, y, space="log")
    assert result.space == "log"
    assert result.value("b") == pytest.approx(1.5, abs=4 * result.error("b"))
    _, low, high = predict(result, x)
    assert np.all(low > 0) and np.all(high > low)


def test_custom_skeleton_fits_float_literals():
    y, _ = power_data(seed=5, noise=0.0)
    result = fit("20.0*x**0.5", L, y)
    assert result.success and result.r2 == pytest.approx(1.0)
    assert result.param_names == ("c0", "c1")
    assert sorted(result.params) == pytest.approx([0.35, 27.0], rel=1e-6)
    assert "x**0.35" in result.equation


def test_zero_constant_in_skeleton_is_refused():
    with pytest.raises(ValueError, match="zero constant"):
        fit("0.0 + 1.0*x", L, L)
    with pytest.raises(ValueError, match="other than x"):
        fit("a*x", L, L)
    with pytest.raises(ValueError, match="no Float constants"):
        fit("x**2", L, L)


@pytest.mark.parametrize("kwargs, message", [
    ({"x": [1.0, 2.0], "y": [1.0]}, "values but y has"),
    ({"x": [1.0, np.nan], "y": [1.0, 2.0]}, "finite"),
    ({"x": [1.0, 2.0], "y": [1.0, 2.0], "sigma": [1.0, 0.0]}, "sigma"),
    ({"x": [1.0, 2.0], "y": [1.0, -2.0], "space": "log"}, "y > 0"),
    ({"x": [1.0, 2.0], "y": [1.0, 2.0], "space": "loglog"}, "space"),
])
def test_malformed_input_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fit("linear", **kwargs)


def test_law_outside_its_domain_is_a_failed_fit_not_an_error():
    result = fit("logarithmic", [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    assert not result.success and "x > 0" in result.message
    assert math.isnan(result.bic)


def test_too_few_points_gives_infinite_errors():
    result = fit("linear", [1.0, 2.0], [3.0, 5.0])
    assert result.success and result.dof == 0
    assert all(math.isinf(e) for e in result.stderr)


def test_unidentifiable_constants_are_flagged():
    # Every x the same: slope and intercept cannot be told apart.
    result = fit("linear", [3.0, 3.0, 3.0, 3.0], [1.0, 1.1, 0.9, 1.0])
    assert result.ill_conditioned
    assert "not identifiable" in result.message


def test_predict_band_covers_the_true_curve_at_the_stated_rate():
    rng = np.random.default_rng(6)
    x_new = np.array([48.0, 64.0, 96.0])
    truth = 27.0 * x_new ** 0.35
    trials, hits = 400, 0
    for _ in range(trials):
        y = 27.0 * L ** 0.35 + 1.5 * rng.standard_normal(len(L))
        result = fit("power", L, y)
        _, low, high = predict(result, x_new, level=0.95)
        hits += int(np.all((low <= truth) & (truth <= high)))
    # Joint coverage of three correlated points is a little under 95% each.
    assert 0.85 <= hits / trials <= 0.99


def test_predict_scalar_and_level():
    y, sigma = power_data(seed=7)
    result = fit("power", L, y, sigma=sigma)
    centre, low, high = predict(result, 64.0)
    assert isinstance(centre, float) and low < centre < high
    _, low99, high99 = predict(result, 64.0, level=0.99)
    assert low99 < low and high99 > high
    with pytest.raises(ValueError):
        predict(result, 64.0, level=1.5)


def test_to_dict_is_json_ready():
    import json

    y, sigma = power_data(seed=8)
    data = fit("power", L, y, sigma=sigma).to_dict()
    json.dumps(data)
    assert set(data["params"]) == {"a", "b"} and data["formula"] == "a*x**b"
    failed = fit("logarithmic", [0.0, 1.0], [1.0, 2.0]).to_dict()
    json.dumps(failed)
    assert failed["bic"] is None and failed["success"] is False


@pytest.mark.parametrize("space", ["linear", "log"])
def test_predict_covariance_matches_the_band(space):
    from scipy import stats

    y, sigma = power_data(seed=9)
    result = fit("power", L, y, sigma=sigma, space=space)
    x_new = np.array([48.0, 64.0, 128.0])
    centre, cov = predict_covariance(result, x_new)
    assert cov.shape == (3, 3) and np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-12)
    f, low, high = predict(result, x_new)
    q = stats.t.ppf(0.975, result.dof)
    if space == "log":
        assert np.allclose(centre, np.log(f))
        assert np.allclose(np.log(high) - centre, q * np.sqrt(np.diag(cov)))
    else:
        assert np.allclose(centre, f)
        assert np.allclose(high - f, q * np.sqrt(np.diag(cov)))
    # Neighbouring predictions share the constants: strongly correlated.
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert corr > 0.9


def test_inflate_never_narrows_the_stated_errors():
    y, sigma = power_data(seed=10, noise=0.0)  # exact data: the reduced chi-square is ~0
    scaled = fit("power", L, y, sigma=0.03 * y)
    absolute = fit("power", L, y, sigma=0.03 * y, absolute_sigma=True)
    inflated = fit("power", L, y, sigma=0.03 * y, absolute_sigma=True, inflate=True)
    assert scaled.error("b") < 1e-6 * absolute.error("b")
    assert inflated.error("b") == pytest.approx(absolute.error("b"))
    assert inflated.error_scale == 1.0

    rng = np.random.default_rng(12)
    noisy = 27.0 * L ** 0.35 * (1 + 0.09 * rng.standard_normal(len(L)))  # 3x the stated error
    absolute = fit("power", L, noisy, sigma=0.03 * noisy, absolute_sigma=True)
    inflated = fit("power", L, noisy, sigma=0.03 * noisy, absolute_sigma=True, inflate=True)
    assert inflated.error_scale > 1.0
    assert inflated.error("b") == pytest.approx(absolute.error("b") * np.sqrt(inflated.error_scale))

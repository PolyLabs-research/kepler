"""Counts and several variables: binomial and Poisson likelihoods, the probability laws, and the
joint finite-size-scaling fit over density and size. Every test has a known answer: data drawn
from a law with stated constants, seeded."""
import math

import numpy as np
import pytest

from kepler import FSS_LAWS, LAWS, PROBABILITY_LAWS, fit, predict, predict_covariance, rank, separation
from kepler.likelihoods import Binomial, CensoredLogNormal, Poisson

RHO = np.linspace(0.05, 0.95, 19)


def logistic(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def window(x, k1, x1, k2, x2):
    return logistic(x, k1, x1) * (1.0 - logistic(x, k2, x2))


def fss(rho, L, r, a, b, w, s):
    return 1.0 / (1.0 + np.exp(-w * L ** s * (rho - r - a * L ** (-b))))


def numeric_gradient(f, v, h=1e-6):
    out = np.zeros_like(v)
    for i in range(len(v)):
        up, down = v.copy(), v.copy()
        up[i] += h
        down[i] -= h
        out[i] = (f(up) - f(down)) / (2 * h)
    return out


@pytest.mark.parametrize("lik, mu, extra", [
    (Binomial([0, 3, 7, 10], [10, 10, 10, 10]), np.array([0.05, 0.3, 0.6, 0.9]), []),
    (Poisson([0, 2, 9], [1.0, 2.0, 3.0]), np.array([0.4, 1.1, 2.5]), []),
    (CensoredLogNormal([3.0, 7.0, 20.0, 20.0], [False, False, True, False]), np.array([4.0, 5.0, 9.0, 30.0]), [-0.3]),
])
def test_gradients_match_finite_differences(lik, mu, extra):
    d_mu, d_extra = lik.gradient(mu, extra)
    assert d_mu == pytest.approx(numeric_gradient(lambda m: lik.nll(m, extra), mu.copy()), rel=1e-5, abs=1e-6)
    if extra:
        assert d_extra == pytest.approx(numeric_gradient(lambda e: lik.nll(mu, e), np.array(extra, float)),
                                        rel=1e-5, abs=1e-6)


def test_a_probability_outside_0_1_costs_a_finite_steep_penalty():
    lik = Binomial([3], [10])
    inside, edge, outside = lik.nll(np.array([0.3])), lik.nll(np.array([1e-12])), lik.nll(np.array([-0.01]))
    assert math.isfinite(outside) and outside > edge > inside
    assert lik.gradient(np.array([-0.01]))[0][0] < 0  # pushes back up, into the domain


def test_binomial_input_is_checked():
    with pytest.raises(ValueError, match="between 0 and the trials"):
        Binomial([11], [10])
    with pytest.raises(ValueError, match="at least one trial"):
        Binomial([0], [0])
    with pytest.raises(ValueError, match="needs trials"):
        fit("logistic_p", RHO, np.zeros(19), likelihood="binomial")
    with pytest.raises(ValueError, match="Gaussian fit"):
        fit("logistic_p", RHO, np.zeros(19), sigma=1.0, likelihood="binomial", trials=10)
    with pytest.raises(ValueError, match="need a binomial"):
        fit("logistic_p", RHO, np.zeros(19), trials=10)


def _draw(p, trials, seed):
    return np.random.default_rng(seed).binomial(trials, p).astype(float)


def test_a_survival_step_recovers_its_threshold_and_steepness():
    k = _draw(logistic(RHO, 20.0, 0.4), 40, 3)
    f = fit("logistic_p", RHO, k, likelihood="binomial", trials=40)
    assert f.success and not f.ill_conditioned, f.message
    assert f.value("x0") == pytest.approx(0.4, abs=4 * f.error("x0"))
    assert f.value("k") == pytest.approx(20.0, abs=4 * f.error("k"))
    # The BIC counts every seed, and -2 ln L is exact: no error estimate enters.
    assert f.n_obs == 19 * 40 and f.likelihood == "binomial"
    assert f.bic == pytest.approx(f.neg2loglik + 2 * math.log(19 * 40))
    # The deviance is a chi-square on n - k degrees of freedom when the law is right.
    assert 0.2 < f.rss / f.dof < 2.5


def test_rank_picks_the_step_and_a_window_on_a_step_is_unidentifiable():
    k = _draw(logistic(RHO, 20.0, 0.4), 40, 3)
    fits = [fit(name, RHO, k, likelihood="binomial", trials=40) for name in PROBABILITY_LAWS]
    ranking = rank(fits)
    assert ranking.best.name == "logistic_p"
    by_name = {e.name: e for e in ranking.entries}
    assert not by_name["window"].eligible  # its fall runs off the data: k2, x2 unbounded
    assert by_name["constant"].delta_bic > 100


def test_a_rise_then_fall_is_a_window():
    truth = {"k1": 30.0, "x1": 0.25, "k2": 30.0, "x2": 0.7}
    k = _draw(window(RHO, **truth), 40, 5)
    fits = [fit(name, RHO, k, likelihood="binomial", trials=40) for name in PROBABILITY_LAWS]
    ranking = rank(fits)
    assert ranking.best.name == "window" and ranking.confident
    best = ranking.best.fit
    for p, v in truth.items():
        assert best.value(p) == pytest.approx(v, abs=4 * best.error(p)), p


def test_error_bars_cover_the_truth_as_often_as_they_claim():
    # 95% intervals for the threshold, over 120 independent data sets.
    hits = 0
    for seed in range(120):
        k = _draw(logistic(RHO, 15.0, 0.45), 20, 100 + seed)
        f = fit("logistic_p", RHO, k, likelihood="binomial", trials=20)
        hits += abs(f.value("x0") - 0.45) <= 1.96 * f.error("x0")
    assert 0.88 <= hits / 120 <= 0.99


def test_perfectly_separated_outcomes_are_flagged_not_trusted():
    k = np.where(RHO > 0.42, 20.0, 0.0)
    f = fit("logistic_p", RHO, k, likelihood="binomial", trials=20)
    assert f.ill_conditioned and "separation" in f.message
    assert 0.4 < f.value("x0") < 0.45  # the threshold is still bracketed by the data


def test_a_probability_band_stays_inside_0_1():
    k = _draw(logistic(RHO, 20.0, 0.4), 10, 8)
    f = fit("logistic_p", RHO, k, likelihood="binomial", trials=10)
    p, low, high = predict(f, np.array([0.0, 0.4, 1.0]))
    assert np.all((0 < low) & (low <= p) & (p <= high) & (high < 1))
    centre, cov = predict_covariance(f, [0.4])
    assert centre[0] == pytest.approx(math.log(p[1] / (1 - p[1])))  # the log odds
    assert cov.shape == (1, 1) and cov[0, 0] > 0


def test_overdispersed_counts_widen_the_error_bars_with_inflate():
    rng = np.random.default_rng(4)
    p = np.clip(logistic(RHO, 20.0, 0.4) + 0.15 * rng.standard_normal(19), 0.01, 0.99)  # extra scatter
    k = rng.binomial(40, p).astype(float)
    plain = fit("logistic_p", RHO, k, likelihood="binomial", trials=40)
    quasi = fit("logistic_p", RHO, k, likelihood="binomial", trials=40, inflate=True)
    assert quasi.error_scale == pytest.approx(plain.rss / plain.dof) and quasi.error_scale > 1.5
    assert quasi.error("x0") == pytest.approx(plain.error("x0") * math.sqrt(quasi.error_scale))


def test_a_poisson_rate_recovers_an_exponential():
    x = np.arange(1.0, 13.0)
    exposure = np.full(12, 50.0)
    rng = np.random.default_rng(6)
    counts = rng.poisson(exposure * 0.2 * np.exp(0.25 * x)).astype(float)
    f = fit("exponential", x, counts, likelihood="poisson", exposure=exposure)
    assert f.success and f.link == "log"
    assert f.value("a") == pytest.approx(0.2, abs=4 * f.error("a"))
    assert f.value("b") == pytest.approx(0.25, abs=4 * f.error("b"))


def test_likelihoods_and_error_models_are_never_ranked_together():
    k = _draw(logistic(RHO, 20.0, 0.4), 40, 3)
    counts = fit("logistic_p", RHO, k, likelihood="binomial", trials=40)
    plain = fit("logistic_p", RHO, k / 40, sigma=0.05, absolute_sigma=True)
    with pytest.raises(ValueError, match="different likelihoods"):
        rank([counts, plain])


# -- Several variables -------------------------------------------------------------------------

TRUTH_FSS = {"r": 0.33, "a": 0.8, "b": 1.0, "w": 4.0, "s": 0.75}


def _fss_design():
    rho, L = np.meshgrid(np.linspace(0.2, 0.5, 13), np.array([16.0, 32.0, 64.0, 128.0]))
    return {"rho": rho.ravel(), "L": L.ravel()}


def test_one_fit_over_density_and_size_recovers_the_infinite_size_threshold():
    x = _fss_design()
    trials = np.full(len(x["rho"]), 50.0)
    k = _draw(fss(x["rho"], x["L"], **TRUTH_FSS), trials.astype(int), 3)
    f = fit("fss_logistic", x, k, likelihood="binomial", trials=trials)
    assert f.success and not f.ill_conditioned, f.message
    assert f.variables == ("rho", "L")
    for p, v in TRUTH_FSS.items():
        assert f.value(p) == pytest.approx(v, abs=4 * f.error(p)), p
    assert f.error("r") < 0.01  # the threshold at infinite size, to within a percent of density
    # Predict at a size never measured: a probability, with a band inside (0, 1).
    p, low, high = predict(f, {"rho": 0.36, "L": 256.0})
    assert isinstance(p, float) and 0 < low < p < high < 1


def test_a_skeleton_in_two_variables_fits_by_least_squares():
    rng = np.random.default_rng(2)
    u, v = rng.uniform(1, 5, 30), rng.uniform(1, 5, 30)
    y = 2.0 * u + 0.5 * v ** 2 + 0.01 * rng.standard_normal(30)
    f = fit("1.0*u + 1.0*v**2", {"u": u, "v": v}, y)
    assert f.success and f.variables == ("u", "v")
    assert sorted(f.params) == pytest.approx([0.5, 2.0], abs=0.01)
    assert predict(f, {"u": 1.0, "v": 2.0})[0] == pytest.approx(4.0, abs=0.05)
    with pytest.raises(ValueError, match="pass x as a dict"):
        predict(f, [1.0, 2.0])
    with pytest.raises(ValueError, match="other than u"):
        fit("1.0*u + 1.0*v**2", {"u": u}, y)


def test_a_law_of_several_variables_refuses_a_plain_array():
    with pytest.raises(ValueError, match="pass x as a dict"):
        fit("fss_logistic", RHO, np.zeros(19), likelihood="binomial", trials=10)


def test_a_binomial_design_says_how_many_seeds_separate_a_window_from_a_step():
    truth = {"k1": 30.0, "x1": 0.25, "k2": 30.0, "x2": 0.85}
    few = separation("window", "logistic_p", RHO, likelihood="binomial", trials=2, truth_params=truth,
                     simulations=40, seed=1)
    many = separation("window", "logistic_p", RHO, likelihood="binomial", trials=20, truth_params=truth)
    # Ten times the seeds, ten times the expected chi-square.
    assert many.expected_delta_chi2 == pytest.approx(10 * few.expected_delta_chi2, rel=0.02)
    assert many.sufficient and many.seed_factor < 1
    assert few.p_decisive is not None and few.p_wrong <= 0.05


def test_the_size_laws_rank_a_drifting_threshold_above_a_fixed_one():
    x = _fss_design()
    trials = np.full(len(x["rho"]), 50.0)
    k = _draw(fss(x["rho"], x["L"], **TRUTH_FSS), trials.astype(int), 3)
    ranking = rank([fit(name, x, k, likelihood="binomial", trials=trials) for name in FSS_LAWS])
    assert ranking.best.name == "fss_logistic"
    by_name = {e.name: e for e in ranking.entries}
    assert by_name["sharpening"].delta_bic > 6 and by_name["crossover"].delta_bic > 100


def test_a_binomial_design_over_density_and_size():
    x = _fss_design()
    drift = separation("fss_logistic", "sharpening", x, likelihood="binomial", trials=20, truth_params=TRUTH_FSS)
    assert drift.n == len(x["rho"]) and drift.sufficient
    # Without the drift (a = 0), is there a transition at all? A crossover cannot sharpen.
    fixed = {"r": 0.33, "w": 4.0, "s": 0.75}
    real = separation("sharpening", "crossover", x, likelihood="binomial", trials=5, truth_params=fixed)
    assert real.expected_delta_bic > 6


def test_the_probability_laws_are_in_the_library():
    assert set(PROBABILITY_LAWS) <= set(LAWS) and set(FSS_LAWS) <= set(LAWS)
    assert all(LAWS[name].variables == ("rho", "L") for name in FSS_LAWS)

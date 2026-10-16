"""One law across groups: a constant shared by every group (a universal exponent) or each
group's own, and which the data support. Known answers, seeded."""
import math

import numpy as np
import pytest

from kepler import compare_sharing, fit, fit_groups, predict

SIZES = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128], dtype=float)
LATTICES = ("square", "hexagon", "triangle")


def growth(amplitudes, exponents, rel=0.05, seed=0):
    """T = a_g L^b_g with 5% errors, one row per size per lattice."""
    rng = np.random.default_rng(seed)
    L = np.tile(SIZES, len(LATTICES))
    group = np.repeat(LATTICES, len(SIZES))
    a = np.repeat(amplitudes, len(SIZES))
    b = np.repeat(exponents, len(SIZES))
    truth = a * L ** b
    T = truth * (1 + rel * rng.standard_normal(len(L)))
    return L, T, group, rel * truth


def test_a_shared_exponent_uses_every_lattice():
    L, T, group, err = growth([20.0, 27.0, 35.0], [0.35] * 3)
    joint = fit_groups("power", L, T, group, shared=("b",), sigma=err, absolute_sigma=True)
    assert joint.success and joint.param_names == ("a[square]", "a[hexagon]", "a[triangle]", "b")
    assert joint.value("b") == pytest.approx(0.35, abs=4 * joint.error("b"))
    for g, a in zip(LATTICES, [20.0, 27.0, 35.0]):
        assert joint.value("a", g) == pytest.approx(a, abs=4 * joint.error("a", g))
    alone = [fit("power", L[group == g], T[group == g], sigma=err[group == g], absolute_sigma=True)
             for g in LATTICES]
    assert joint.error("b") < min(f.error("b") for f in alone)
    with pytest.raises(ValueError, match="say which group"):
        joint.value("a")


def test_a_group_fit_predicts_like_any_fit():
    L, T, group, err = growth([20.0, 27.0, 35.0], [0.35] * 3)
    joint = fit_groups("power", L, T, group, shared=("b",), sigma=err, absolute_sigma=True)
    hexagon = joint.fit_for("hexagon")
    y, low, high = predict(hexagon, 256.0)
    assert low < y < high and y == pytest.approx(27.0 * 256 ** 0.35, rel=0.1)
    assert hexagon.params == (joint.value("a", "hexagon"), joint.value("b"))


def test_the_data_say_the_exponent_is_shared_and_the_amplitude_is_not():
    L, T, group, err = growth([20.0, 27.0, 35.0], [0.35] * 3)
    result = compare_sharing("power", L, T, group, sigma=err, absolute_sigma=True)
    assert result.best.shared == ("b",)
    assert result.entry(("b",)).p_value > 0.01
    assert result.entry(("a",)).p_value < 1e-6 and result.entry(("a", "b")).p_value < 1e-6
    assert len(result.entries) == 4 and result.groups == LATTICES


def test_different_exponents_are_not_shared():
    L, T, group, err = growth([27.0, 27.0, 27.0], [0.25, 0.35, 0.45])
    result = compare_sharing("power", L, T, group, sigma=err, absolute_sigma=True)
    assert "b" not in result.best.shared
    assert result.entry(("b",)).p_value < 1e-6


def test_the_sharing_test_is_calibrated():
    """When the exponent really is shared, p < 0.05 about one time in twenty."""
    rejected = 0
    for seed in range(100):
        L, T, group, err = growth([20.0, 27.0, 35.0], [0.35] * 3, seed=1000 + seed)
        result = compare_sharing("power", L, T, group, patterns=[("b",)], sigma=err, absolute_sigma=True)
        rejected += result.entry(("b",)).p_value < 0.05
    assert 1 <= rejected <= 12


def test_a_threshold_shared_across_sizes_by_binomial_counts():
    rng = np.random.default_rng(5)
    rho = np.tile(np.linspace(0.1, 0.6, 11), 3)
    size = np.repeat(["16", "32", "64"], 11)
    steep = np.repeat([10.0, 20.0, 40.0], 11)  # sharper on larger tori, same threshold
    p = 1 / (1 + np.exp(-steep * (rho - 0.35)))
    k = rng.binomial(40, p).astype(float)
    result = compare_sharing("logistic_p", rho, k, size, likelihood="binomial", trials=40)
    assert result.best.shared == ("x0",)
    joint = result.best.fit
    assert joint.value("x0") == pytest.approx(0.35, abs=4 * joint.error("x0"))
    assert joint.value("k", "16") < joint.value("k", "64")
    assert joint.n_obs == 33 * 40


def test_bad_input_is_refused():
    L, T, group, err = growth([20.0, 27.0, 35.0], [0.35] * 3)
    with pytest.raises(ValueError, match="no constant c"):
        fit_groups("power", L, T, group, shared=("c",))
    with pytest.raises(ValueError, match="at least two groups"):
        fit_groups("power", L, T, ["one"] * len(L))
    with pytest.raises(ValueError, match="labels"):
        fit_groups("power", L, T, group[:-1])


def test_the_bic_of_a_joint_fit_is_its_likelihood():
    L, T, group, err = growth([20.0, 27.0, 35.0], [0.35] * 3)
    joint = fit_groups("power", L, T, group, shared=("b",), sigma=err, absolute_sigma=True)
    assert joint.bic == pytest.approx(joint.rss + 4 * math.log(len(L)))
    assert joint.to_dict()["shared"] == ["b"]

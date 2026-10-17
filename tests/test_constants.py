"""Is a fitted constant a known number, and how likely is such a match by chance?"""
import math

import pytest

from kepler import KNOWN, fit, identify, identify_fit


def test_a_precise_value_matches_a_simple_fraction_and_is_unlikely_luck():
    (first, *_) = identify(0.33335, 0.0001)
    assert first.expression == "1/3" and abs(first.z) < 1
    assert first.p_chance < 0.01


def test_a_wide_error_bar_matches_everything_and_says_so():
    (first, *_) = identify(0.349, 0.02)
    assert first.expression == "1/3"
    assert first.p_chance > 0.2  # a 6% error bar: a simple fraction is close to almost anything


def test_irrationals_and_named_constants():
    assert identify(2 * math.pi / 3, 1e-6)[0].expression == "(2/3)*pi"
    named = [m for m in identify(0.59274, 0.00002) if m.named]
    assert named and named[0].expression == "site percolation threshold, square"
    # A numerical constant's own error widens its window.
    dp = [m for m in identify(0.584, 0.01) if m.named]
    assert any(m.expression == "DP(2+1) beta" for m in dp)


def test_signs_zero_and_bad_input():
    assert identify(-0.125, 0.0005)[0].expression == "-1/8"
    (zero,) = identify(0.01, 0.02)
    assert zero.expression == "0" and zero.p_chance == 1.0
    with pytest.raises(ValueError):
        identify(1.0, -1.0)


def test_every_constant_of_a_fit():
    x = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    y = [3.0 * v ** 0.5 for v in x]
    result = identify_fit(fit("power", x, y, sigma=[0.001 * v for v in y], absolute_sigma=True))
    assert result["a"][0].expression == "3" and result["b"][0].expression == "1/2"


def test_the_known_table_is_sane():
    for name, (value, error) in KNOWN.items():
        assert value > 0 and error >= 0, name


def test_identify_names_the_gumbel_shape_constants():
    """The largest of many light-tailed times is Gumbel: its quantile skewness and its
    interquartile range per unit slope in ln L (in 2D) are in the table, and are found."""
    kelly = identify(0.2215, 0.004)
    assert any(m.named and m.expression.startswith("Gumbel Kelly") for m in kelly)
    ratio = identify(0.786, 0.01)
    assert any(m.named and m.expression.startswith("2D Gumbel") for m in ratio)
    assert KNOWN["Gumbel standard deviation = pi/sqrt(6)"][0] == pytest.approx(1.28254983)

"""The law library: every starting constant non-zero, every skeleton keeps every constant,
and each law recovers itself from its own synthetic data."""
import numpy as np
import pytest
import sympy

from kepler import LAWS, fit, get_law
from kepler.refine import parametrize

ADVERSARIAL = {
    "zeros": (np.arange(1.0, 9.0), np.zeros(8)),
    "constant": (np.arange(1.0, 9.0), np.full(8, 5.0)),
    "negative": (np.arange(1.0, 9.0), -np.arange(1.0, 9.0)),
    "one point": (np.array([3.0]), np.array([2.0])),
    "x with zero": (np.arange(0.0, 8.0), np.arange(0.0, 8.0) ** 2),
    "tiny": (np.arange(1.0, 9.0), np.full(8, 1e-15)),
}

TRUE = {
    "constant": {"a": 50.0},
    "logarithmic": {"a": 10.0, "b": 5.0},
    "linear": {"a": 2.0, "b": 3.0},
    "power": {"a": 27.0, "b": 0.35},
    "exponential": {"a": 3.0, "b": 0.08},
    "exponential_area": {"a": 2.0, "b": 0.002},
    "logistic": {"c": 0.9, "k": -20.0, "x0": 0.4},
    "shifted_power": {"c": 0.3, "a": 0.8, "b": -1.2},
    "logistic_p": {"k": 20.0, "x0": 0.4},
    "window": {"k1": 30.0, "x1": 0.25, "k2": 30.0, "x2": 0.7},
}

# Laws in x alone; laws of several variables have their own tests (test_likelihoods.py).
SINGLE = [name for name, law in LAWS.items() if law.variables == ("x",)]


def test_the_zero_constant_trap_is_real():
    # Why laws use named symbols: SymPy silently drops a zero Float.
    x = sympy.Symbol("x")
    assert sympy.Float(0.0) + x == x
    assert len(parametrize(sympy.sympify("0.0 + 1.0*x"))[1]) == 1


@pytest.mark.parametrize("name", SINGLE)
@pytest.mark.parametrize("case", list(ADVERSARIAL))
def test_starting_constants_are_finite_and_non_zero(name, case):
    x, y = ADVERSARIAL[case]
    values = LAWS[name].initial(x, y)
    assert len(values) == len(LAWS[name].params)
    assert all(np.isfinite(v) and v != 0.0 for v in values), values


@pytest.mark.parametrize("name", SINGLE)
@pytest.mark.parametrize("case", [None, *ADVERSARIAL])
def test_skeleton_keeps_every_constant(name, case):
    law = LAWS[name]
    skeleton = law.skeleton(*(ADVERSARIAL[case] if case else (None, None)))
    _, symbols, values = parametrize(sympy.sympify(skeleton))
    assert len(symbols) == len(law.params), skeleton
    assert all(v != 0.0 for v in values)


def _sample(name):
    if name in ("logistic", "logistic_p", "window"):
        return np.linspace(0.05, 0.95, 19)
    if name == "shifted_power":
        return np.array([8, 12, 16, 24, 32, 48, 64, 96, 128], dtype=float)
    return np.array([4, 6, 8, 12, 16, 24, 32, 48, 64], dtype=float)


@pytest.mark.parametrize("name", SINGLE)
def test_each_law_recovers_its_own_constants(name):
    law = LAWS[name]
    x = _sample(name)
    expr, symbols = law.parametrized()
    f = sympy.lambdify([sympy.Symbol("x"), *symbols], expr, "numpy")
    truth = TRUE[name]
    y_true = np.broadcast_to(f(x, *[truth[p] for p in law.params]), x.shape).astype(float)
    rng = np.random.default_rng(11)
    y = y_true * (1 + 0.005 * rng.standard_normal(len(x)))
    result = fit(name, x, y, sigma=0.005 * np.abs(y_true))
    assert result.success, result.message
    for p in law.params:
        assert result.value(p) == pytest.approx(truth[p], abs=5 * result.error(p) + 1e-9 * abs(truth[p])), p


def test_a_law_of_several_variables_starts_from_its_own_data():
    law = LAWS["fss_logistic"]
    assert law.variables == ("rho", "L")
    rho = np.tile(np.linspace(0.1, 0.5, 9), 3)
    L = np.repeat([16.0, 32.0, 64.0], 9)
    for y in (np.zeros(27), np.full(27, 0.5), (rho > 0.3).astype(float)):
        values = law.initial({"rho": rho, "L": L}, y)
        assert len(values) == 5 and all(np.isfinite(v) and v != 0.0 for v in values)
    _, symbols, _ = parametrize(sympy.sympify(law.skeleton()))
    assert len(symbols) == 5
    assert law.domain_problem({"rho": rho, "L": -L}) == "fss_logistic needs every L > 0"
    with pytest.raises(ValueError, match="dict"):
        law.initial(rho, np.zeros(27))


def test_get_law():
    assert get_law("power").formula == "a*x**b"
    with pytest.raises(ValueError, match="unknown law"):
        get_law("cubic")

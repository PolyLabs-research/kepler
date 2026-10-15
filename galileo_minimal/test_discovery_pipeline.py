"""
test_discovery_pipeline.py - Tests for normalisation and refinement in discovery.

Covers the three post-processing modules (expression, normalize, refine) and
the discover_equation path of GalileoTools with a stubbed specialist, so the
whole pipeline is exercised without torch or trained checkpoints.
"""
import numpy as np
import pytest
import sympy

from expression import (
    ExpressionParseError,
    X,
    expression_to_string,
    format_ar_equation,
    format_ode_equation,
    parse_ar_tokens,
    parse_ode_tokens,
    parse_polynomial,
    parse_transcendental,
    prune_small_terms,
    tokens_to_sympy,
)
from generators_base import float_to_tokens
from normalize import fit_normalization
from refine import (
    ar_stationarity,
    cancellation_ratio,
    count_constants,
    parametrize,
    refine_ar,
    refine_expression,
    refine_ode,
    simulate_linear_ode,
)


def _eval(expr, xs):
    with np.errstate(all="ignore"):
        return np.broadcast_to(np.asarray(sympy.lambdify(X, expr, "numpy")(xs), dtype=float), np.shape(xs))


# =============================================================================
# expression.py
# =============================================================================

def test_parse_polynomial_templates():
    xs = np.linspace(-5, 5, 21)
    cases = [
        ("INT_2 MUL X POW INT_2 ADD INT_1 MUL X SUB INT_3", 2 * xs**2 + xs - 3),
        ("NEG CONST 1 POINT 5 0 MUL X POW INT_3 ADD X", -1.5 * xs**3 + xs),
        ("X", xs),
        ("INT_0", np.zeros_like(xs)),
        ("HALF MUL X POW INT_2", 0.5 * xs**2),
    ]
    for tokens, truth in cases:
        expr = parse_polynomial(tokens.split())
        assert np.allclose(_eval(expr, xs), truth), tokens


def test_parse_transcendental_templates():
    xs = np.linspace(0.5, 9.5, 19)
    exp_tokens = float_to_tokens(1.5) + ["MUL", "EXP"] + float_to_tokens(-0.3) + ["MUL", "X"]
    power_tokens = float_to_tokens(2.0) + ["MUL", "X", "POW"] + float_to_tokens(1.7)
    log_tokens = float_to_tokens(1.88) + ["MUL", "LN", "X", "ADD"] + float_to_tokens(-0.83)
    logistic_tokens = float_to_tokens(8.0) + ["DIV", "INT_1", "ADD", "EXP", "NEG"] + float_to_tokens(1.5) + ["MUL", "X", "SUB"] + float_to_tokens(4.0)

    assert np.allclose(_eval(parse_transcendental(exp_tokens), xs), 1.5 * np.exp(-0.3 * xs))
    assert np.allclose(_eval(parse_transcendental(power_tokens), xs), 2.0 * xs**1.7)
    assert np.allclose(_eval(parse_transcendental(log_tokens), xs), 1.88 * np.log(xs) - 0.83)
    assert np.allclose(_eval(parse_transcendental(logistic_tokens), xs), 8.0 / (1 + np.exp(-1.5 * (xs - 4.0))))


def test_parse_polynomial_accepts_unit_constant_and_rejects_garbage():
    xs = np.linspace(-2, 2, 5)
    assert np.allclose(_eval(parse_polynomial("X ADD CONST 1 POINT 0 0".split()), xs), xs + 1)
    for bad in (["MUL", "MUL", "POW"], ["X", "ADD"], ["INT_2", "MUL", "X", "POW"]):
        with pytest.raises(ExpressionParseError):
            parse_polynomial(bad)
    # Truncated template falls through to the infix fallback instead of raising.
    assert expression_to_string(tokens_to_sympy(["INT_2", "MUL", "LN"], "transcendental")) == "2.0*log(x)"


def test_tokens_to_sympy_falls_back_across_families():
    xs = np.linspace(-3, 3, 7)
    expr = tokens_to_sympy("INT_2 MUL X POW INT_2 ADD INT_1".split(), "transcendental")
    assert np.allclose(_eval(expr, xs), 2 * xs**2 + 1)
    with pytest.raises(ExpressionParseError):
        tokens_to_sympy(["MUL", "MUL", "POW"], "polynomial")


def test_parse_ar_and_ode_tokens():
    assert parse_ar_tokens("X MUL CONST NEG 0 POINT 0 5 MUL CONST 0 POINT 1 3".split()) == pytest.approx([-0.05, 0.13])
    assert parse_ode_tokens("DX_DT EQ CONST NEG 0 POINT 8 9 MUL X".split()) == {"order": 1, "a": pytest.approx(-0.89)}
    second = parse_ode_tokens("D2X_DT2 ADD CONST 3 POINT 4 6 MUL DX_DT ADD CONST 7 POINT 1 2 MUL X EQ INT_0".split())
    assert second == {"order": 2, "C1": pytest.approx(3.46), "C2": pytest.approx(7.12)}
    assert parse_ode_tokens(["X", "MUL", "INT_1"]) is None
    with pytest.raises(ExpressionParseError):
        parse_ar_tokens(["INT_1", "MUL", "X"])


def test_formatting_helpers():
    assert format_ode_equation({"order": 2, "C1": 0.0, "C2": 7.12}) == "x'' + 7.12*x = 0"
    assert format_ode_equation({"order": 2, "C1": -0.5, "C2": 4.0}) == "x'' - 0.5*x' + 4*x = 0"
    assert format_ode_equation({"order": 1, "a": -0.4}) == "dx/dt = -0.4*x"
    assert format_ar_equation([0.7, -0.2]) == "y[t] = 0.7*y[t-1] + -0.2*y[t-2]"
    pruned = prune_small_terms(sympy.Float(2.0) * X + sympy.Float(1e-15), 1e-9)
    assert expression_to_string(pruned) == "2.0*x"
    assert expression_to_string(sympy.Float(1.23456789) * X) == "1.23457*x"


# =============================================================================
# normalize.py
# =============================================================================

def test_polynomial_normalisation_maps_into_training_frame():
    x = np.linspace(-50, 50, 80)
    y = 0.02 * x**2 - 1.5 * x + 300
    norm = fit_normalization(x, y, "polynomial", "algebraic")
    xn, yn = norm.transform(x, y)
    assert xn.min() == pytest.approx(-5) and xn.max() == pytest.approx(5)
    assert np.mean(yn) == pytest.approx(0, abs=1e-9) and np.std(yn) == pytest.approx(1)


def test_transcendental_normalisation_policy():
    years = np.linspace(2000, 2020, 60)
    norm = fit_normalization(years, np.exp(-0.1 * (years - 2000)), "transcendental", "algebraic")
    xn, _ = norm.transform(years, years)
    assert xn.min() == pytest.approx(0.1) and xn.max() == pytest.approx(10)

    x = np.linspace(1, 100, 100)
    norm = fit_normalization(x, 4.5 * np.log(x) - 2, "transcendental", "algebraic")
    assert norm.x_shift == 0 and norm.x_origin == 0 and norm.x_scale == pytest.approx(10)
    xn, _ = norm.transform(x, x)
    assert np.allclose(xn, x / 10)


def test_inverse_and_readable_inverse_agree():
    x = np.linspace(2000, 2020, 60)
    y = 20 + 80 * np.exp(-0.25 * (x - 2000))
    norm = fit_normalization(x, y, "transcendental", "algebraic")
    g = sympy.Float(0.3) * sympy.exp(sympy.Float(-0.5) * X) + sympy.Float(0.2)
    xn, _ = norm.transform(x, y)
    expected = norm.inverse_y(_eval(g, xn))
    exact = _eval(norm.inverse_expression(g), x)
    readable = sympy.sympify(expression_to_string(norm.readable_inverse(g), 10))
    assert np.allclose(exact, expected)
    assert np.allclose(_eval(readable, x), expected, rtol=1e-6)
    assert "(x - 2000)" in expression_to_string(norm.readable_inverse(g))


def test_ode_normalisation_inverse_and_disabled_identity():
    t = np.linspace(0, 25, 300)
    y = np.cos(2 * t) * np.exp(-0.3 * t)
    norm = fit_normalization(t, y, "linear_ode", "dynamical")
    tn, yn = norm.transform(t, y)
    assert tn.max() == pytest.approx(10) and np.max(np.abs(yn)) == pytest.approx(1)
    tau = norm.x_scale
    assert norm.inverse_ode({"order": 2, "C1": 0.6 * tau, "C2": 4.0 * tau**2}) == {"order": 2, "C1": pytest.approx(0.6), "C2": pytest.approx(4.0)}
    assert norm.inverse_ode({"order": 1, "a": -0.5 * tau})["a"] == pytest.approx(-0.5)

    off = fit_normalization(t, y, "linear_ode", "dynamical", enabled=False)
    assert not off.enabled
    assert off.inverse_expression(X**2) == X**2
    assert off.inverse_ode({"order": 1, "a": 3.0})["a"] == 3.0


# =============================================================================
# refine.py
# =============================================================================

def test_parametrize_keeps_structural_literals():
    expr = sympy.Float(2.0) * X**2 + 1 / (1 + sympy.exp(sympy.Float(-0.5) * X))
    _, symbols, values = parametrize(expr)
    assert len(symbols) == 2
    assert sorted(values) == pytest.approx([-0.5, 2.0])


def test_refine_polynomial_recovers_exact_constants():
    x = np.linspace(-5, 5, 50)
    y = 2.5 * x**2 - 1.3 * x + 0.7
    guess = sympy.Float(2.0) * X**2 - sympy.Float(1.0) * X + sympy.Float(1.0)
    res = refine_expression(guess, x, y)
    assert res.improved and res.r2 == pytest.approx(1.0) and res.rmse < 1e-8
    assert np.allclose(_eval(res.expression, x), y)


def test_refine_without_free_constants_is_a_no_op():
    x = np.linspace(1, 5, 10)
    res = refine_expression(X**2, x, x**2 + 1)
    assert not res.improved and res.expression == X**2 and "no free constants" in res.message
    wrapped = refine_expression(X**2, x, x**2 + 1, affine_wrapper=True)
    assert wrapped.improved and wrapped.rmse < 1e-8


def test_refine_power_law_keeps_family_through_normalisation():
    x = np.linspace(1, 50, 80)
    y = 3.2 * x**1.7
    norm = fit_normalization(x, y, "transcendental", "algebraic")
    xn, yn = norm.transform(x, y)
    res = refine_expression(parse_transcendental(float_to_tokens(1.5) + ["MUL", "X", "POW"] + float_to_tokens(2.0)), xn, yn, affine_wrapper=True)
    readable = norm.readable_inverse(res.expression)
    exponents = [float(e) for p in readable.atoms(sympy.Pow) for e in [p.exp] if e.is_Float]
    assert exponents and exponents[0] == pytest.approx(1.7, abs=1e-3)
    assert np.allclose(_eval(readable, x), y, rtol=1e-4)


def test_refine_discards_degenerate_affine_fit():
    """A power law plus a free offset can imitate a logarithm through cancelling terms."""
    rng = np.random.default_rng(0)
    x = np.linspace(2000, 2020, 60)
    y = 20 + 80 * np.exp(-0.25 * (x - 2000)) + rng.normal(0, 0.1, 60)
    norm = fit_normalization(x, y, "transcendental", "algebraic")
    xn, yn = norm.transform(x, y)
    for tokens in ("CONST 1 POINT 5 0 MUL X POW CONST NEG 0 POINT 5 0", "CONST 2 POINT 0 0 MUL X POW HALF"):
        res = refine_expression(parse_transcendental(tokens.split()), xn, yn, affine_wrapper=True, bound=1e6)
        assert res.variant == "template" and not res.ill_conditioned, tokens
        assert "discarded as ill-conditioned" in res.note
        assert res.max_abs_parameter < 100 and res.cancellation_ratio < 10
        assert 0.8 < res.r2 < 0.9  # the honest power-law fit, not the 0.97 of the degenerate one


def test_cancellation_ratio_and_constant_count():
    x = np.linspace(-5, 5, 50)
    good = sympy.Float(2.0) * X + sympy.Float(1.0)
    assert cancellation_ratio(good, x, 2 * x + 1) == pytest.approx(1.0, abs=0.2)
    bad = sympy.Float(1000.0) * X ** sympy.Float(0.001) - sympy.Float(999.0)
    assert cancellation_ratio(bad, np.linspace(1, 10, 50), np.log(np.linspace(1, 10, 50))) > 100
    assert count_constants(sympy.Float(2.0) * X ** 2 + sympy.Float(1.0)) == 2
    assert count_constants(X ** 2) == 0


def test_refine_bound_is_respected_and_flagged():
    x = np.linspace(1, 5, 20)
    res = refine_expression(sympy.Float(1.0) * X, x, 5 * x, bound=2.0)
    assert res.max_abs_parameter <= 2.0 + 1e-9
    assert res.ill_conditioned  # pinned at the bound


def test_ar_stationarity():
    assert ar_stationarity([0.6, -0.25])[0]
    assert not ar_stationarity([1.0])[0]
    assert not ar_stationarity([0.5, 0.6])[0]
    assert refine_ar(np.cumsum(np.ones(50)), [0.9])["stationary"] is False


def test_refine_ar_recovers_coefficients():
    rng = np.random.default_rng(0)
    y = np.zeros(600)
    for t in range(2, 600):
        y[t] = 0.6 * y[t - 1] - 0.25 * y[t - 2] + rng.normal(0, 0.1)
    out = refine_ar(y + 5.0, [0.5, -0.1])
    assert out["improved"] and out["order"] == 2 and out["mean"] == pytest.approx(5.0, abs=0.1)
    assert out["coefficients"] == pytest.approx([0.6, -0.25], abs=0.06)


def test_refine_ode_recovers_coefficients():
    t = np.linspace(0, 25, 300)
    y = simulate_linear_ode({"order": 2, "C1": 0.6, "C2": 4.0}, t, 1.5, -0.2)
    out = refine_ode({"order": 2, "C1": 0.5, "C2": 3.5}, t, y)
    assert out["improved"] and out["C1"] == pytest.approx(0.6, abs=1e-3) and out["C2"] == pytest.approx(4.0, abs=1e-3)
    assert out["initial_conditions"]["y0"] == pytest.approx(1.5, abs=1e-3)

    t1 = np.linspace(0, 8, 100)
    out1 = refine_ode({"order": 1, "a": -0.5}, t1, 3 * np.exp(-0.45 * t1))
    assert out1["improved"] and out1["a"] == pytest.approx(-0.45, abs=1e-4)


# =============================================================================
# GalileoTools.discover_equation with a stubbed specialist
# =============================================================================

class _FakeModel:
    def __init__(self, tokens):
        self.tokens = tokens

    def generate(self, features, vocab):
        return vocab.encode(self.tokens)


@pytest.fixture
def tools():
    from galileo_tools import GalileoTools
    return GalileoTools(use_overseer=False)


def _stub(tools, monkeypatch, tokens, available=None):
    """Stub the specialist: `tokens` is one sequence, or a dict model -> sequence."""
    from vocab import Vocabulary
    vocab = Vocabulary.default()

    def load(name, tier):
        seq = tokens[name] if isinstance(tokens, dict) else tokens
        return _FakeModel(seq.split()), vocab

    monkeypatch.setattr(tools, "_load_model", load)
    monkeypatch.setattr(tools, "_predict_tokens", lambda model, v, features: v.decode(model.generate(features, v)))
    if available is not None:
        monkeypatch.setattr(tools, "list_available_models", lambda: available)


STUB_TOKENS = {
    "polynomial": "INT_1 MUL X POW INT_2 ADD INT_1 MUL X",
    "transcendental": "CONST 1 POINT 0 0 MUL EXP CONST NEG 0 POINT 3 0 MUL X",
    "linear_ode": "D2X_DT2 ADD CONST 1 POINT 4 0 MUL DX_DT ADD CONST 2 3 POINT 0 0 MUL X EQ INT_0",
    "ar": "X MUL CONST 0 POINT 5 0 MUL CONST NEG 0 POINT 1 0",
}
ALL_TIER1 = {"polynomial": [1], "transcendental": [1], "linear_ode": [1], "ar": [1]}


def test_discover_polynomial_in_real_units(tools, monkeypatch):
    x = np.linspace(-50, 50, 80)
    y = 0.02 * x**2 - 1.5 * x + 300
    _stub(tools, monkeypatch, "INT_1 MUL X POW INT_2 ADD INT_1 MUL X")
    result = tools.discover_equation(y=y.tolist(), x=x.tolist(), model_hint="polynomial")
    assert result.get("success"), result
    assert result["model_equation"] == "1.0*x**2 + 1.0*x"
    assert result["refined"] and result["fit"]["r2"] == pytest.approx(1.0)
    assert result["confidence"] == pytest.approx(1.0) and result["confidence_source"] == "fit_r2"
    assert result["normalization"]["enabled"]
    assert np.allclose(_eval(sympy.sympify(result["equation"]), x), y, atol=1e-3)
    verified = tools.verify_equation(result["equation"], x=x.tolist(), y=y.tolist())
    assert verified["match"]


def test_discover_exponential_on_offset_axis(tools, monkeypatch):
    rng = np.random.default_rng(1)
    x = np.linspace(2000, 2020, 60)
    y = 20 + 80 * np.exp(-0.25 * (x - 2000)) + rng.normal(0, 0.3, 60)
    _stub(tools, monkeypatch, "CONST 1 POINT 0 0 MUL EXP CONST NEG 0 POINT 3 0 MUL X")
    result = tools.discover_equation(y=y.tolist(), x=x.tolist(), model_hint="transcendental")
    assert result.get("success"), result
    assert "(x - 2000)" in result["equation"]
    assert result["fit"]["r2"] > 0.999 and result["fit"]["rmse"] < 0.5
    assert tools.verify_equation(result["equation"], x=x.tolist(), y=y.tolist())["match"]


def test_discover_ar_reports_refitted_coefficients(tools, monkeypatch):
    rng = np.random.default_rng(2)
    y = np.zeros(600)
    for t in range(2, 600):
        y[t] = 0.6 * y[t - 1] - 0.25 * y[t - 2] + rng.normal(0, 0.1)
    _stub(tools, monkeypatch, "X MUL CONST 0 POINT 5 0 MUL CONST NEG 0 POINT 1 0")
    result = tools.discover_equation(y=(y + 5).tolist(), model_hint="ar")
    assert result.get("success"), result
    assert result["equation"].startswith("y[t] = ")
    assert result["ar"]["coefficients"] == pytest.approx([0.6, -0.25], abs=0.06)
    assert result["confidence_source"] == "heuristic" and "x_assumed" in result


def test_discover_ode_in_original_time_units(tools, monkeypatch):
    t = np.linspace(0, 25, 300)
    y = simulate_linear_ode({"order": 2, "C1": 0.6, "C2": 4.0}, t, 1.5, -0.2)
    # In normalised time (tau = 2.5) the truth is C1 = 1.5, C2 = 25; the stub is deliberately off.
    _stub(tools, monkeypatch, "D2X_DT2 ADD CONST 1 POINT 4 0 MUL DX_DT ADD CONST 2 3 POINT 0 0 MUL X EQ INT_0")
    result = tools.discover_equation(y=y.tolist(), x=t.tolist(), model_hint="linear_ode")
    assert result.get("success"), result
    assert result["ode"]["C1"] == pytest.approx(0.6, abs=1e-2) and result["ode"]["C2"] == pytest.approx(4.0, abs=1e-2)
    assert result["initial_conditions"]["y0"] == pytest.approx(1.5, abs=1e-2)
    assert result["equation"].startswith("x'' + ") and result["fit"]["r2"] > 0.999


def test_discover_flags_and_errors(tools, monkeypatch):
    x = np.linspace(-5, 5, 40)
    y = 2 * x**2 + 1
    _stub(tools, monkeypatch, "INT_2 MUL X POW INT_2 ADD INT_1")
    raw = tools.discover_equation(y=y.tolist(), x=x.tolist(), model_hint="polynomial", normalize=False, refine=False)
    assert raw.get("success") and not raw["refined"] and not raw["normalization"]["enabled"]
    assert raw["equation"] == "2.0*x**2 + 1.0" and raw["fit"]["rmse"] == pytest.approx(0.0, abs=1e-9)

    bad = tools.discover_equation(y=y.tolist(), x=x[:-1].tolist(), model_hint="polynomial")
    assert "error" in bad


def test_verify_reports_r2(tools):
    x = [1, 2, 3, 4, 5]
    exact = tools.verify_equation("x**2", x=x, y=[1, 4, 9, 16, 25])
    assert exact["r2"] == pytest.approx(1.0)
    poor = tools.verify_equation("x", x=x, y=[1, 4, 9, 16, 25])
    assert poor["r2"] < 0.95 and poor["match"] is False


def test_ranked_discovery_picks_the_right_family(tools, monkeypatch):
    _stub(tools, monkeypatch, STUB_TOKENS, ALL_TIER1)
    rng = np.random.default_rng(3)

    x = np.linspace(-50, 50, 80)
    quad = tools.discover_equation(y=(0.02 * x**2 - 1.5 * x + 300).tolist(), x=x.tolist())
    assert quad.get("success"), quad
    assert quad["strategy"] == "rank" and quad["model_used"] == "polynomial"
    assert len(quad["candidates"]) == 4 and quad["candidates"][0]["model"] == "polynomial"
    assert quad["candidates"][0]["delta_bic"] == 0.0 and quad["candidates"][0]["eligible"]
    assert tools.verify_equation(quad["equation"], x=x.tolist(), y=(0.02 * x**2 - 1.5 * x + 300).tolist())["match"]

    x = np.linspace(2000, 2020, 60)
    y = 20 + 80 * np.exp(-0.25 * (x - 2000)) + rng.normal(0, 0.1, 60)
    cool = tools.discover_equation(y=y.tolist(), x=x.tolist())
    assert cool.get("success") and cool["model_used"] == "transcendental", cool.get("candidates")
    assert "(x - 2000)" in cool["equation"]

    t = np.linspace(0, 25, 300)
    osc = tools.discover_equation(y=simulate_linear_ode({"order": 2, "C1": 0.6, "C2": 4.0}, t, 1.5, -0.2).tolist(), x=t.tolist())
    assert osc.get("success") and osc["model_used"] == "linear_ode", osc.get("candidates")
    assert osc["ode"]["C1"] == pytest.approx(0.6, abs=1e-2)

    series = np.zeros(400)
    for i in range(2, 400):
        series[i] = 0.6 * series[i - 1] - 0.25 * series[i - 2] + rng.normal(0, 0.1)
    ar = tools.discover_equation(y=(series + 5).tolist())
    assert ar.get("success") and ar["model_used"] == "ar", ar.get("candidates")
    assert "stationary AR" in ar["ranking_note"]
    assert all(c["r2"] is None or c["r2"] < 0.5 for c in ar["candidates"] if c["track"] == "deterministic")


def test_ranked_discovery_prefers_closed_form_over_equivalent_ode(tools, monkeypatch):
    """An offset exponential also solves x'' + c*x' = 0; BIC must prefer the 3-constant closed form."""
    stubs = dict(STUB_TOKENS)
    stubs["linear_ode"] = "D2X_DT2 ADD CONST 0 POINT 5 0 MUL DX_DT ADD CONST 0 POINT 0 1 MUL X EQ INT_0"
    _stub(tools, monkeypatch, stubs, {"transcendental": [2], "linear_ode": [2]})
    rng = np.random.default_rng(5)
    x = np.linspace(2000, 2020, 60)
    y = 20 + 80 * np.exp(-0.25 * (x - 2000)) + rng.normal(0, 0.1, 60)
    result = tools.discover_equation(y=y.tolist(), x=x.tolist())
    by_model = {c["model"]: c for c in result["candidates"]}
    assert by_model["linear_ode"]["r2"] > 0.999 and by_model["linear_ode"]["n_constants"] == 4
    assert by_model["transcendental"]["n_constants"] == 3
    assert result["model_used"] == "transcendental", result["candidates"]
    assert by_model["transcendental"]["delta_bic"] <= 6.0


def test_ranked_discovery_tier_hint_and_empty(tools, monkeypatch):
    x = np.linspace(-5, 5, 40)
    y = 2 * x**2 + 1
    _stub(tools, monkeypatch, STUB_TOKENS, {"polynomial": [1, 2], "transcendental": [2]})
    only_tier1 = tools.discover_equation(y=y.tolist(), x=x.tolist(), tier=1)
    assert only_tier1.get("success") and [(c["model"], c["tier"]) for c in only_tier1["candidates"]] == [("polynomial", 1)]

    hinted = tools.discover_equation(y=y.tolist(), x=x.tolist(), model_hint="polynomial")
    assert hinted["strategy"] == "hint" and "candidates" not in hinted

    _stub(tools, monkeypatch, STUB_TOKENS, {})
    assert "error" in tools.discover_equation(y=y.tolist(), x=x.tolist())


def test_malformed_prediction_keeps_result(tools, monkeypatch):
    x = np.linspace(0, 5, 20)
    _stub(tools, monkeypatch, "POW MUL EXP LN")
    result = tools.discover_equation(y=(x**2).tolist(), x=x.tolist(), model_hint="polynomial")
    assert result.get("success") and "postprocess_error" in result
    assert result["fit"] is None and result["confidence_source"] == "heuristic"


def test_schemas_describe_new_flags():
    from galileo_tools import ANTHROPIC_TOOL_SCHEMAS, TOOL_SCHEMAS
    discover = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "discover_equation")
    props = discover["function"]["parameters"]["properties"]
    assert {"normalize", "refine", "strategy"} <= set(props)
    anthropic = next(s for s in ANTHROPIC_TOOL_SCHEMAS if s["name"] == "discover_equation")
    assert {"normalize", "refine", "strategy"} <= set(anthropic["input_schema"]["properties"])

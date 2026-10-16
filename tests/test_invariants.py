"""Conserved quantities: integer combinations of counts that never change. Known answers."""
import numpy as np
import pytest

from kepler import find_invariants
from kepler.invariants import lll


def walk(seed, steps=200, size=3):
    return np.cumsum(np.random.default_rng(seed).integers(-3, 4, size=(steps, size)), axis=0)


def test_a_planted_conservation_law_is_found_in_its_simplest_form():
    a, b = walk(1, size=2).T
    c = a + 2 * b - 7
    result = find_invariants({"a": a, "b": b, "c": c})
    (inv,) = result.exact
    assert inv.weights == [1, 2, -1] and inv.value == 7
    assert str(inv) == "a + 2*b - c = 7"


def test_lll_gives_short_vectors_for_several_invariants():
    x = walk(2, size=2)
    a, b = x.T
    # Two invariants hidden in long combinations: c = a + b, d = a - b + 5.
    c, d = a + b, a - b + 5
    result = find_invariants(np.column_stack([a, b, c, d]), names=["a", "b", "c", "d"])
    assert len(result.exact) == 2
    assert max(sum(abs(w) for w in inv.weights) for inv in result.exact) <= 4
    for inv in result.exact:
        values = np.column_stack([a, b, c, d]) @ np.array(inv.weights)
        assert np.all(values == inv.value)


def test_a_parity_conserved_while_the_count_changes():
    rng = np.random.default_rng(3)
    x = 1 + 2 * np.cumsum(rng.integers(-2, 3, 300))  # always odd, never constant
    y = np.cumsum(rng.integers(-3, 4, 300))
    result = find_invariants({"x": x, "y": y}, modulus=2)
    assert result.exact == []
    (parity,) = result.modular
    assert parity.weights == [1, 0] and parity.value == 1 and str(parity) == "x = 1 (mod 2)"


def test_runs_are_pooled_and_known_identities_accounted_for():
    runs = []
    for seed in (6, 7, 8):
        a, b = walk(seed, size=2).T
        runs.append({"a": a, "b": b, "total": a + b, "c": a + 2 * b + seed})
    result = find_invariants(runs)
    assert result.runs == 3 and result.rank == 2 and len(result.exact) == 2
    assert result.implies({"a": 1, "b": 1, "total": -1})
    assert not result.implies({"a": 1})
    # Bookkeeping (total = a + b) explains one invariant; one more is the "law".
    assert result.beyond([{"a": 1, "b": 1, "total": -1}]) == 1
    law = [inv for inv in result.exact if inv.weights[3]][0]
    assert sorted(law.values) == sorted(-v for v in (6, 7, 8)) or sorted(law.values) == [6, 7, 8]


def test_a_short_cycle_leaves_accidents_with_large_weights():
    # A run of period 2 has one difference direction: almost everything looks conserved.
    x = np.array([[1, 2, 3, 4], [2, 3, 5, 4], [1, 2, 3, 4], [2, 3, 5, 4]])
    result = find_invariants(x)
    assert result.rank == 1 and len(result.exact) == 3
    assert any(inv.trivial for inv in result.exact)  # the fourth quantity never changed


def test_nothing_is_conserved_in_noise():
    assert find_invariants(walk(4)).exact == []


def test_lll_reduces_a_skewed_basis():
    reduced = lll([[1, 1, 1], [-1, 0, 2], [3, 5, 6]])
    assert sorted(sum(x * x for x in v) for v in reduced)[0] <= 3


def test_bad_input():
    with pytest.raises(ValueError, match="integers"):
        find_invariants(np.array([[0.5, 1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="two times"):
        find_invariants(np.array([[1, 2]]))
    with pytest.raises(ValueError, match="prime"):
        find_invariants(walk(5), modulus=4)

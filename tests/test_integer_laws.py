"""Exact laws for integer data, each with the points that determined it and the further points
that confirmed it. Known answers."""
from fractions import Fraction

import pytest

from kepler import find_integer_laws
from kepler.integer_laws import disagreement

N = list(range(1, 13))


def best(n, y):
    laws = find_integer_laws(n, y)
    assert laws, "no law found"
    return laws[0]


def test_a_polynomial_and_how_many_points_confirm_it():
    law = best(N, [n * n - n + 41 for n in N])
    assert law.form == "polynomial" and law.expression == "n^2 - n + 41"
    assert (law.determined_by, law.confirmed_by) == (3, 9)
    assert law(20) == 421


def test_a_law_by_residue_class_or_by_gcd():
    y = [n if n % 2 else n // 2 for n in N]
    laws = find_integer_laws(N, y)
    # One unknown beats a polynomial per class, and a polynomial per class that is the same
    # function (n for odd n, n/2 for even) is the same hypothesis, not a rival.
    assert [law.expression for law in laws] == ["n/gcd(n, 2)"]
    assert (laws[0].determined_by, laws[0].confirmed_by) == (1, 11)
    z = [3 * n + 1 if n % 3 == 0 else n * n for n in N]
    assert best(N, z).form == "residues mod 3"


def test_exponential_recurrence_and_period():
    assert best(N, [2 ** n - 1 for n in N]).expression == "2^n - 1"
    fib = [1, 1]
    while len(fib) < 12:
        fib.append(fib[-1] + fib[-2])
    law = best(N, fib)
    assert law.form == "recurrence" and law.expression.startswith("y(n) = 1*y(n-1) + 1*y(n-2)")
    assert law(15) == 610
    law = best(N, [[4, 1, 7][n % 3] for n in N])
    assert law.form == "periodic" and law.unknowns == 3


def test_number_theoretic_features():
    """Periods of linear rules are built from the multiplicative order of 2."""
    odd = [n for n in range(3, 40, 2)]
    y = [2 ** pow_order(n) - 1 for n in odd]
    law = best(odd, y)
    assert law.form == "number theory" and law.expression == "(2^ord(n) - 1)"
    assert law(41) == 2 ** 20 - 1


def pow_order(m):
    k, v = 1, 2 % m
    while v != 1:
        v, k = v * 2 % m, k + 1
    return k


def test_lcm():
    assert best(N, [n * 6 // __import__("math").gcd(n, 6) for n in N]).expression == "lcm(n, 6)"


def test_noise_has_no_law():
    import random

    rng = random.Random(4)
    assert find_integer_laws(N, [rng.randint(0, 1000) for _ in N]) == []


def test_too_few_points_confirm_nothing():
    # Three points determine a quadratic; nothing is left to confirm it.
    laws = find_integer_laws([1, 2, 3], [5, 1, 9])
    assert all(law.confirmed_by >= 1 for law in laws)
    assert not any(law.form == "polynomial" and law.unknowns == 3 for law in laws)


def test_rival_laws_and_where_they_part():
    # 1, 2, 1, 4, 1, 2, 1 is the largest power of 2 dividing n, and also gcd(n, 4), on n <= 7.
    n = list(range(1, 8))
    laws = find_integer_laws(n, [1, 2, 1, 4, 1, 2, 1])
    names = [law.expression for law in laws[:2]]
    assert set(names) == {"2^v(n)", "gcd(n, 4)"}
    assert disagreement(laws[:2], 8) == 8  # the next measurement decides


def test_rational_values_and_bad_input():
    law = best(N, [Fraction(n, 2) for n in N])
    assert law(5) == Fraction(5, 2) and law.unknowns == 1
    with pytest.raises(ValueError, match="appear once"):
        find_integer_laws([1, 1, 2], [1, 1, 2])

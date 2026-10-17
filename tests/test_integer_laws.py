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


def test_the_suborder_of_two():
    """Rule 90's periods on cycles: 2^sord(n) - 1 for odd n, doubled for each factor of 2."""
    def sord(m):
        k, v = 1, 2 % m
        while v not in (1, m - 1):
            v, k = v * 2 % m, k + 1
        return k

    n = [m for m in range(3, 31) if m & (m - 1)]  # powers of 2 aside
    odd = [m // (m & -m) for m in n]
    y = [(m & -m) * (2 ** sord(o) - 1) for m, o in zip(n, odd)]
    laws = find_integer_laws(n, y)
    # These are rule 90's own periods, so the simplest law names them so; the formula is the
    # rival that agrees with it (powers of 2 aside: rule 90 dies there) up to 36 and parts at
    # 37, where rule 90 falls short of it.
    assert laws[0].expression == "P90(n)" and laws[0].confirmed_by == len(n) - 1
    formula = next(law for law in laws if law.expression == "2^v(n)*(2^sord(n) - 1)")
    assert formula.confirmed_by == len(n) - 1
    assert disagreement([laws[0], formula], 33) == 37


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


def test_periods_of_the_one_dimensional_linear_rules():
    """Exact, against a direct simulation from a single cell on every ring up to 24 cells."""
    from kepler.integer_laws import rule_period

    def simulate(rule, m):
        taps = {60: (0, -1), 90: (-1, 1), 150: (-1, 0, 1)}[rule]
        state, seen, t = tuple(int(i == 0) for i in range(m)), {}, 0
        while state not in seen:
            seen[state] = t
            state = tuple(sum(state[(i + k) % m] for k in taps) % 2 for i in range(m))
            t += 1
        return t - seen[state]

    for rule in (60, 90, 150):
        assert [rule_period(rule, m) for m in range(1, 25)] == [simulate(rule, m) for m in range(1, 25)]
    assert rule_period(90, 37) == 87381  # a third of 2^18 - 1 (Martin, Odlyzko and Wolfram)
    assert rule_period(90, 64) == 1  # rule 90 dies on a ring of 2^k cells


def test_a_lattice_rule_tied_to_a_one_dimensional_one():
    """Twice rule 90's period, on sizes where that differs from 2 * 2^v(n) * (2^sord(n) - 1):
    found, and the formula is not."""
    from kepler.integer_laws import rule_period

    n = [m for m in range(10, 121, 2) if (m // (m & -m)) >= 5]
    laws = find_integer_laws(n, [2 * rule_period(90, m) for m in n])
    assert laws[0].expression == "2*P90(n)"
    assert not any(law.expression == "2*2^v(n)*(2^sord(n) - 1)" for law in laws)

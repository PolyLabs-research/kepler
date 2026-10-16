"""
integer_laws.py - Exact laws for integer data: y(n) known exactly, as a deterministic simulation
gives it (a period, a count of distinct objects, a transient on small tori). A law must reproduce
every point exactly. Each is reported with how many points determined it and how many more
confirmed it: a law with no confirmations is an interpolation, not a finding.

    find_integer_laws(n, y) -> list[IntegerLaw], simplest first

Forms, each with its unknowns determined exactly (rational arithmetic) from the fewest points
and checked against all the others:

    polynomial        y = c_0 + c_1 n + ... + c_d n^d, rational c, d <= 3
    residue classes   a polynomial of degree <= 2 for each class of n mod m, m = 2..6
    gcd / lcm         y = k gcd(n, c), k lcm(n, c) or k n / gcd(n, c), c = 2..60
    exponential       y = a b^n + c, b = 2..10
    recurrence        y_n = c_1 y_(n-1) + ... + c_r y_(n-r) + c_0, r <= 3, on consecutive n
    periodic          y_(n+p) = y_n on consecutive n
    number theory     y = k f(n) or k f(n) g(n), f and g among n, the odd part of n, 2^v (the
                      power of 2 in n), ord (the multiplicative order of 2 modulo the odd
                      part) and 2^ord - 1: the quantities periods of linear (XOR) rules on
                      tori are built from

Complexity orders the results: the number of unknowns, then their size in bits.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class IntegerLaw:
    form: str
    expression: str
    unknowns: int
    determined_by: int  # points used to find the unknowns
    confirmed_by: int  # further points it reproduces exactly
    bits: float
    evaluate: Callable[[int], Fraction] = field(repr=False, default=None)

    @property
    def complexity(self) -> Tuple[int, float]:
        return (self.unknowns, self.bits)

    def __call__(self, n: int) -> Fraction:
        return self.evaluate(n)

    def to_dict(self) -> Dict[str, Any]:
        return {"form": self.form, "expression": self.expression, "unknowns": self.unknowns,
                "determined_by": self.determined_by, "confirmed_by": self.confirmed_by, "bits": self.bits}


def _bits(values: Sequence[Fraction]) -> float:
    return float(sum(math.log2(abs(v.numerator) + 1) + math.log2(v.denominator) for v in values))


def _solve(rows: List[List[Fraction]], rhs: List[Fraction]) -> Optional[List[Fraction]]:
    """Exact solution of a square system, or None when it is singular."""
    m = len(rows)
    a = [list(r) + [b] for r, b in zip(rows, rhs)]
    for col in range(m):
        pivot = next((r for r in range(col, m) if a[r][col] != 0), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(m):
            if r != col and a[r][col] != 0:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [a[i][m] / a[i][i] for i in range(m)]


def _fmt(c: Fraction) -> str:
    return str(c.numerator) if c.denominator == 1 else f"{c.numerator}/{c.denominator}"


def _poly_text(coeffs: Sequence[Fraction], var: str = "n") -> str:
    terms = []
    for power in range(len(coeffs) - 1, -1, -1):
        c = coeffs[power]
        if c == 0:
            continue
        mono = "" if power == 0 else (var if power == 1 else f"{var}^{power}")
        if mono and abs(c) == 1:
            body = mono
        elif mono:
            body = f"{_fmt(abs(c))}*{mono}" if abs(c).denominator == 1 else f"({_fmt(abs(c))})*{mono}"
        else:
            body = _fmt(abs(c))
        terms.append(("-" if c < 0 else "+", body))
    if not terms:
        return "0"
    text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, body in terms[1:]:
        text += f" {sign} {body}"
    return text


def _poly_eval(coeffs: Sequence[Fraction], n: int) -> Fraction:
    return sum((c * Fraction(n) ** i for i, c in enumerate(coeffs)), Fraction(0))


def _fit_poly(points: List[Tuple[int, Fraction]], degree: int) -> Optional[List[Fraction]]:
    if len(points) < degree + 1:
        return None
    use = points[: degree + 1]
    return _solve([[Fraction(n) ** i for i in range(degree + 1)] for n, _ in use], [y for _, y in use])


# --- forms --------------------------------------------------------------------------------------

def _polynomials(points) -> List[IntegerLaw]:
    out = []
    for d in range(4):
        coeffs = _fit_poly(points, d)
        if coeffs is None or len(points) < d + 2:
            break
        if coeffs[-1] == 0 and d > 0:
            continue
        if all(_poly_eval(coeffs, n) == y for n, y in points):
            out.append(IntegerLaw("polynomial", _poly_text(coeffs), d + 1, d + 1, len(points) - d - 1,
                                  _bits(coeffs), lambda n, c=tuple(coeffs): _poly_eval(c, n)))
            break  # a higher degree adds only zero coefficients
    return out


def _residues(points) -> List[IntegerLaw]:
    out = []
    consecutive = _consecutive(points)
    for m in range(2, 7):
        for d in range(0 if not consecutive else 1, 3):  # constant per class on consecutive n: periodic
            classes = {r: [(n, y) for n, y in points if n % m == r] for r in range(m)}
            present = {r: c for r, c in classes.items() if c}
            if len(present) < 2 or any(len(c) < d + 2 for c in present.values()):
                continue
            laws = {}
            for r, c in present.items():
                coeffs = _fit_poly(c, d)
                if coeffs is None or not all(_poly_eval(coeffs, n) == y for n, y in c):
                    break
                laws[r] = coeffs
            else:
                if len({tuple(v) for v in laws.values()}) == 1:
                    continue  # one polynomial for every class: not a residue law
                text = "; ".join(f"n = {r} mod {m}: {_poly_text(c)}" for r, c in sorted(laws.items()))
                used = (d + 1) * len(laws)
                out.append(IntegerLaw(f"residues mod {m}", text, used, used, len(points) - used,
                                      _bits([x for c in laws.values() for x in c]),
                                      lambda n, L=laws, m=m: _poly_eval(L[n % m], n) if n % m in L else None))
                break
    return out


_ARITH: Dict[str, Callable[[int, int], int]] = {
    "gcd(n, {c})": lambda n, c: math.gcd(n, c),
    "lcm(n, {c})": lambda n, c: n * c // math.gcd(n, c),
    "n/gcd(n, {c})": lambda n, c: n // math.gcd(n, c),
}


def _scaled(points, name: str, f: Callable[[int], int], bits_extra: float) -> Optional[IntegerLaw]:
    n0, y0 = points[0]
    f0 = f(n0)
    if f0 == 0:
        return None
    k = y0 / f0
    if k == 0 or not all(k * f(n) == y for n, y in points):
        return None
    text = name if k == 1 else f"{_fmt(k)}*{name}"
    return IntegerLaw("scaled", text, 1, 1, len(points) - 1, _bits([k]) + bits_extra,
                      lambda n, k=k: k * f(n))


def _gcd_lcm(points) -> List[IntegerLaw]:
    out = []
    if any(n <= 0 for n, _ in points):
        return out
    # A constant whose factors the data never reach (26 = 2 * 13 on n <= 12) cannot be told
    # from a smaller one; propose only constants up to the largest n.
    for c in range(2, min(60, points[-1][0]) + 1):
        for name, g in _ARITH.items():
            law = _scaled(points, name.format(c=c), lambda n, g=g, c=c: g(n, c), math.log2(c))
            if law is not None:
                law.form = "gcd/lcm"
                out.append(law)
    return out


def _exponentials(points) -> List[IntegerLaw]:
    out = []
    if len(points) < 3:
        return out
    (n0, y0), (n1, y1) = points[0], points[1]
    for b in range(2, 11):
        sol = _solve([[Fraction(b) ** n0, Fraction(1)], [Fraction(b) ** n1, Fraction(1)]], [y0, y1])
        if sol is None or sol[0] == 0:
            continue
        a, c = sol
        if all(a * Fraction(b) ** n + c == y for n, y in points):
            text = (f"{_fmt(a)}*" if a != 1 else "") + f"{b}^n" + (f" + {_fmt(c)}" if c > 0 else
                                                                  f" - {_fmt(-c)}" if c < 0 else "")
            out.append(IntegerLaw("exponential", text, 2, 2, len(points) - 2, _bits([a, c]) + math.log2(b),
                                  lambda n, a=a, b=b, c=c: a * Fraction(b) ** n + c))
    return out


def _consecutive(points) -> bool:
    return all(points[i + 1][0] - points[i][0] == 1 for i in range(len(points) - 1))


def _recurrences(points) -> List[IntegerLaw]:
    out = []
    if not _consecutive(points):
        return out
    ys = [y for _, y in points]
    for r in range(1, 4):
        unknowns = r + 1
        rows = [[ys[i - j] for j in range(1, r + 1)] + [Fraction(1)] for i in range(r, len(ys))]
        targets = ys[r:]
        if len(rows) < unknowns + 1:
            break
        sol = _solve(rows[:unknowns], targets[:unknowns])
        if sol is None:
            continue
        if all(sum(c * v for c, v in zip(sol, row)) == t for row, t in zip(rows, targets)):
            parts = [f"{_fmt(c)}*y(n-{j})" for j, c in zip(range(1, r + 1), sol[:r]) if c != 0]
            if sol[r] != 0:
                parts.append(_fmt(sol[r]))
            start = tuple(ys[:r])
            text = "y(n) = " + " + ".join(parts or ["0"]) + "; from " + ", ".join(_fmt(v) for v in start)
            # The description needs the r starting values as well as the r + 1 coefficients.
            out.append(IntegerLaw("recurrence", text, unknowns + r, unknowns + r, len(ys) - unknowns - r,
                                  _bits(list(sol) + list(start)),
                                  lambda n, s=tuple(sol), y0=start, n0=points[0][0]: _run(s, y0, n0, n)))
            break
    return out


def _run(coeffs: Sequence[Fraction], start: Sequence[Fraction], n0: int, n: int) -> Optional[Fraction]:
    """A recurrence run forward from its starting values to n."""
    r = len(start)
    if n < n0:
        return None
    values = list(start)
    while n0 + len(values) - 1 < n:
        values.append(sum(c * values[-j] for j, c in zip(range(1, r + 1), coeffs[:r])) + coeffs[r])
    return values[n - n0]


def _periodic(points) -> List[IntegerLaw]:
    if not _consecutive(points):
        return []
    ys = [y for _, y in points]
    for p in range(1, len(ys) // 3 + 1):
        if all(ys[i] == ys[i + p] for i in range(len(ys) - p)):
            n0 = points[0][0]
            cycle = ys[:p]
            text = f"period {p}: " + ", ".join(_fmt(v) for v in cycle)
            return [IntegerLaw("periodic", text, p, p, len(ys) - p, _bits(cycle),
                               lambda n, c=tuple(cycle), n0=n0, p=p: c[(n - n0) % p])]
    return []


def _odd_part(n: int) -> int:
    while n % 2 == 0:
        n //= 2
    return n


def _order_of_two(m: int) -> int:
    if m == 1:
        return 1
    k, v = 1, 2 % m
    while v != 1:
        v = (v * 2) % m
        k += 1
    return k


_FEATURES: Dict[str, Callable[[int], int]] = {
    "n": lambda n: n,
    "odd(n)": _odd_part,
    "2^v(n)": lambda n: n // _odd_part(n),
    "ord(n)": lambda n: _order_of_two(_odd_part(n)),
    "(2^ord(n) - 1)": lambda n: 2 ** _order_of_two(_odd_part(n)) - 1,
}


def _number_theory(points) -> List[IntegerLaw]:
    if any(n <= 0 for n, _ in points):
        return []
    out = []
    names = list(_FEATURES)
    combos = [(a,) for a in names] + list(itertools.combinations(names, 2))
    for combo in combos:
        def f(n, combo=combo):
            return math.prod(_FEATURES[c](n) for c in combo)
        law = _scaled(points, "*".join(combo), f, 2.0 * len(combo))
        if law is not None:
            law.form = "number theory"
            out.append(law)
    return out


def find_integer_laws(n: Sequence[int], y: Sequence[int], min_confirmations: int = 1,
                      max_results: int = 10) -> List[IntegerLaw]:
    """Exact laws y(n), simplest first, each confirmed by at least `min_confirmations` points it
    was not determined from (see the module docstring)."""
    if len(n) != len(y):
        raise ValueError(f"n has {len(n)} values but y has {len(y)}")
    points = sorted((int(a), Fraction(b)) for a, b in zip(n, y))
    if len({a for a, _ in points}) != len(points):
        raise ValueError("each n must appear once (an exact law has one value per n)")
    found: List[IntegerLaw] = []
    for search in (_polynomials, _residues, _gcd_lcm, _exponentials, _recurrences, _periodic, _number_theory):
        found.extend(law for law in search(points) if law.confirmed_by >= min_confirmations)
    # Every law here reproduces the data; two that also agree on the next 24 n are one hypothesis
    # (lcm(n, 2)/2 is n/gcd(n, 2)): keep the simpler. Ones that disagree there are rivals.
    beyond = range(points[-1][0] + 1, points[-1][0] + 25)
    unique: Dict[Any, IntegerLaw] = {}
    for law in sorted(found, key=lambda law: law.complexity):
        try:
            key = tuple(law(m) for m in beyond)
        except (ZeroDivisionError, TypeError, KeyError):
            key = law.expression
        unique.setdefault(key, law)
    return list(unique.values())[:max_results]


def disagreement(laws: Sequence[IntegerLaw], start: int, stop: Optional[int] = None) -> Optional[int]:
    """The first n from `start` at which the given laws do not all predict the same value: the
    next measurement that would tell them apart (None if they agree up to `stop`)."""
    stop = start + 1000 if stop is None else stop
    for m in range(start, stop):
        values = {law(m) for law in laws}
        if len(values) > 1:
            return m
    return None

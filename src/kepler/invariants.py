"""
invariants.py - Conserved quantities: integer combinations of counts that never change.

    find_invariants(series, names=None, modulus=None) -> Invariants

`series` holds integer counts over time, one row per time step and one column per quantity
(live cells, cells with k live neighbours, cumulative births and deaths, objects of each kind).
An invariant is an integer vector w for which sum_i w_i c_i(t) is the same at every t, so w is
in the null space of the matrix D of differences between consecutive rows.

The rational null space is found exactly, from the small square matrix D^T D (it has the same
null space as D), then reduced by LLL to short integer vectors: the simplest statements of the
same invariants. An identity of the bookkeeping (live cells = the sum over k of live cells with
k neighbours) is found as readily as a law of the dynamics; telling them apart is the reader's
job, and rediscovering the identities is a check that the search works.

With `modulus=p` (a prime), quantities conserved only mod p are found too (the null space of D
over GF(p)), keeping those not already implied by an exact invariant: a parity a rule conserves
although the counts themselves change.

Too little data. A run that falls into a short cycle visits few distinct states, so D has few
independent rows and its null space is large: most of what it holds is an accident of that run,
and LLL can only shorten it so far (large coefficients are the sign). Pass several runs (a list
of series): their differences are pooled, and only combinations constant in every run survive,
each with its own value per run. `Invariants.rank` is the rank of the pooled D;
`Invariants.implies(w)` says whether a known identity is in the span of what was found, and
`Invariants.beyond(known)` how many invariants the known identities do not account for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sympy


@dataclass
class Invariant:
    weights: List[int]
    names: List[str] = field(repr=False)
    value: int  # sum_i w_i c_i, the same at every time (in the first run)
    modulus: Optional[int] = None  # None: exact; p: conserved mod p
    values: List[int] = field(default_factory=list)  # its value in each run

    @property
    def size(self) -> int:
        """The sum of |w|: small for a law, large for an accident of too little data."""
        return sum(abs(w) for w in self.weights)

    @property
    def trivial(self) -> bool:
        """One quantity that never changed."""
        return sum(1 for w in self.weights if w) == 1

    @property
    def expression(self) -> str:
        terms = []
        for w, name in zip(self.weights, self.names):
            if w == 0:
                continue
            coef = "" if abs(w) == 1 else f"{abs(w)}*"
            terms.append(("-" if w < 0 else "+", f"{coef}{name}"))
        if not terms:
            return "0"
        text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
        for sign, body in terms[1:]:
            text += f" {sign} {body}"
        return text

    def __str__(self) -> str:
        tail = f" (mod {self.modulus})" if self.modulus else ""
        return f"{self.expression} = {self.value}{tail}"

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": dict(zip(self.names, self.weights)), "value": self.value, "values": self.values,
                "modulus": self.modulus, "statement": str(self), "size": self.size, "trivial": self.trivial}


@dataclass
class Invariants:
    exact: List[Invariant]
    modular: List[Invariant]
    names: List[str]
    steps: int  # differences examined, over every run
    runs: int
    rank: int  # rank of the pooled differences: quantities - rank = exact invariants

    def _vector(self, weights) -> List[int]:
        if isinstance(weights, dict):
            unknown = set(weights) - set(self.names)
            if unknown:
                raise ValueError(f"unknown quantities {sorted(unknown)}")
            return [int(weights.get(n, 0)) for n in self.names]
        return [int(w) for w in weights]

    def implies(self, weights) -> bool:
        """Whether a combination (a dict of name -> weight, or a vector) is constant by virtue of
        the exact invariants found: whether it lies in their span."""
        basis = [inv.weights for inv in self.exact]
        if not basis:
            return not any(self._vector(weights))
        return sympy.Matrix(basis + [self._vector(weights)]).rank() == sympy.Matrix(basis).rank()

    def beyond(self, known: Sequence) -> int:
        """How many exact invariants found are not accounted for by the `known` combinations."""
        vectors = [self._vector(k) for k in known]
        found = [inv.weights for inv in self.exact]
        if not found:
            return 0
        both = sympy.Matrix(found + vectors).rank() if vectors else len(found)
        return both - (sympy.Matrix(vectors).rank() if vectors else 0)

    def to_dict(self) -> Dict[str, Any]:
        return {"exact": [i.to_dict() for i in self.exact], "modular": [i.to_dict() for i in self.modular],
                "names": self.names, "steps": self.steps, "runs": self.runs, "rank": self.rank}


# --- LLL ---------------------------------------------------------------------------------------

def lll(basis: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4)) -> List[List[int]]:
    """Lenstra-Lenstra-Lovasz reduction of integer row vectors (exact rational arithmetic)."""
    b = [list(map(int, v)) for v in basis]
    n = len(b)
    if n == 0:
        return b

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v))

    def gram_schmidt():
        ortho: List[List[Fraction]] = []
        mu = [[Fraction(0)] * n for _ in range(n)]
        norms: List[Fraction] = []
        for i in range(n):
            v = [Fraction(x) for x in b[i]]
            for j in range(i):
                mu[i][j] = dot(b[i], ortho[j]) / norms[j] if norms[j] else Fraction(0)
                v = [x - mu[i][j] * y for x, y in zip(v, ortho[j])]
            ortho.append(v)
            norms.append(dot(v, v))
        return mu, norms

    mu, norms = gram_schmidt()
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                b[k] = [x - q * y for x, y in zip(b[k], b[j])]
                mu, norms = gram_schmidt()
        if norms[k] >= (delta - mu[k][k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            mu, norms = gram_schmidt()
            k = max(k - 1, 1)
    return b


def _primitive(v: Sequence[Fraction]) -> List[int]:
    """A rational vector scaled to coprime integers, its first non-zero entry positive."""
    den = 1
    for x in v:
        den = den * Fraction(x).denominator // math.gcd(den, Fraction(x).denominator)
    ints = [int(Fraction(x) * den) for x in v]
    g = 0
    for x in ints:
        g = math.gcd(g, abs(x))
    ints = [x // g for x in ints] if g else ints
    first = next((x for x in ints if x), 0)
    return [-x for x in ints] if first < 0 else ints


def _nullspace_mod(D: np.ndarray, p: int) -> List[List[int]]:
    """A basis of {w : D w = 0 mod p} (p prime)."""
    A = np.mod(D.astype(object), p)
    rows, cols = A.shape
    pivots: List[int] = []
    r = 0
    A = [list(map(int, row)) for row in A]
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if A[i][c] % p), None)
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        inv = pow(A[r][c], -1, p)
        A[r] = [(x * inv) % p for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] % p:
                f = A[i][c]
                A[i] = [(x - f * y) % p for x, y in zip(A[i], A[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        w = [0] * cols
        w[f] = 1
        for i, c in enumerate(pivots):
            w[c] = (-A[i][f]) % p
        basis.append(w)
    return basis


def _rank_mod(vectors: List[List[int]], p: int) -> int:
    if not vectors:
        return 0
    return len(vectors) - len(_nullspace_mod(np.array(vectors, dtype=object).T, p)) if vectors else 0


def _as_matrix(series, names: Optional[Sequence[str]]):
    if isinstance(series, dict):
        names = list(series) if names is None else list(names)
        C = np.column_stack([np.asarray(series[n]) for n in names])
    else:
        C = np.asarray(series)
        if C.ndim != 2:
            raise ValueError("series must be time x quantity")
        names = [f"c{i}" for i in range(C.shape[1])] if names is None else list(names)
    if len(names) != C.shape[1]:
        raise ValueError(f"{len(names)} names for {C.shape[1]} quantities")
    if C.shape[0] < 2:
        raise ValueError("an invariant needs at least two times")
    if not np.all(np.equal(np.mod(C, 1), 0)):
        raise ValueError("counts must be integers")
    return C.astype(np.int64).astype(object), names


def find_invariants(series, names: Optional[Sequence[str]] = None, modulus: Optional[int] = None) -> Invariants:
    """Integer combinations of the quantities that stay constant (module docstring). `series` is
    a 2-D array (time x quantity), a dict of equal-length columns, or a list of either (several
    runs, pooled)."""
    runs = list(series) if isinstance(series, list) else [series]
    if not runs:
        raise ValueError("no series")
    matrices = []
    for run in runs:
        C, run_names = _as_matrix(run, names)
        if names is None:
            names = run_names
        elif list(run_names) != list(names):
            raise ValueError("every run needs the same quantities, in the same order")
        matrices.append(C)
    names = list(names)
    D = np.vstack([np.diff(C, axis=0) for C in matrices])
    starts = [C[0] for C in matrices]
    gram = sympy.Matrix((D.T.dot(D)).tolist())
    rank = int(gram.rank())
    basis = [_primitive(list(v)) for v in gram.nullspace()]
    reduced = lll(basis) if basis else []

    def values(w, mod=None):
        out = [int(sum(int(x) * int(c) for x, c in zip(w, start))) for start in starts]
        return [v % mod for v in out] if mod else out

    exact = []
    for w in sorted((_primitive(v) for v in reduced), key=lambda v: (sum(abs(x) for x in v), v)):
        vals = values(w)
        exact.append(Invariant(w, names, vals[0], None, vals))

    modular: List[Invariant] = []
    if modulus is not None:
        if modulus < 2 or not sympy.isprime(modulus):
            raise ValueError("modulus must be a prime")
        implied = [[x % modulus for x in inv.weights] for inv in exact]
        current = _rank_mod(implied, modulus)
        for w in _nullspace_mod(D, modulus):
            trial = implied + [w]
            new_rank = _rank_mod(trial, modulus)
            if new_rank > current:
                implied, current = trial, new_rank
                centred = [x if x <= modulus // 2 else x - modulus for x in w]
                vals = values(centred, modulus)
                modular.append(Invariant(centred, names, vals[0], modulus, vals))
    return Invariants(exact, modular, names, int(D.shape[0]), len(matrices), rank)

"""
expression.py - Convert specialist token sequences into SymPy expressions.

The decoders emit family-specific token templates rather than a general
infix grammar, so each family is parsed by its own small template parser:

    polynomial      [NEG] c0 {ADD|SUB c_i [MUL] X [POW INT_k]}
    exp             a MUL EXP b MUL X                      -> a*exp(b*x)
    power           a MUL X POW b                          -> a*x**b
    log             a MUL LN X ADD b                       -> a*log(x) + b
    logistic        L DIV INT_1 ADD EXP NEG k MUL X SUB x0 -> L/(1 + exp(-k*(x - x0)))
    ar              X MUL phi_1 MUL phi_2 ...              -> y[t] = sum phi_i*y[t-i]
    first-order ODE DX_DT EQ a MUL X                       -> dx/dt = a*x
    second-order    D2X_DT2 ADD C1 MUL DX_DT ADD C2 MUL X EQ INT_0

Coefficients are either a single snapped constant token (INT_3, HALF, PI, ...)
or the digit encoding CONST [NEG] d d POINT d d.

A generic infix fallback (shared with test.py's historical behaviour) is kept
for sequences that do not match any template, so a malformed prediction still
produces a best-effort expression instead of an exception.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import sympy
from sympy import Symbol

X = Symbol("x")

SPECIAL_TOKENS = {"PAD", "BOS", "EOS", "UNK"}

SNAPPED_CONSTANTS: Dict[str, float] = {
    "HALF": 0.5,
    "THIRD": 1.0 / 3.0,
    "QUARTER": 0.25,
    "NEG_1": -1.0,
    "NEG_2": -2.0,
    "PI": math.pi,
    "E": math.e,
    "PHI": (1 + math.sqrt(5)) / 2,
    "TAU": 2 * math.pi,
    "SQRT2": math.sqrt(2),
    "SQRT3": math.sqrt(3),
    "G_EARTH": 9.80665,
    "G_NEWTON": 6.67430e-11,
    "C_LIGHT": 299792458.0,
    "R_GAS": 8.31446,
}
SNAPPED_CONSTANTS.update({f"INT_{i}": float(i) for i in range(11)})


class ExpressionParseError(ValueError):
    """Raised when a token sequence cannot be turned into an expression."""


# =============================================================================
# Coefficient parsing
# =============================================================================

def strip_special(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if t not in SPECIAL_TOKENS]


def _is_coefficient_start(token: str) -> bool:
    return token == "CONST" or token in SNAPPED_CONSTANTS


def _take_coefficient(tokens: Sequence[str], i: int) -> Tuple[float, int]:
    """Read one coefficient starting at index i. Returns (value, next_index)."""
    if i >= len(tokens):
        raise ExpressionParseError("Expected a coefficient but reached end of sequence")

    tok = tokens[i]
    if tok in SNAPPED_CONSTANTS:
        return SNAPPED_CONSTANTS[tok], i + 1

    if tok != "CONST":
        raise ExpressionParseError(f"Expected a coefficient at position {i}, got {tok!r}")

    i += 1
    sign = 1.0
    if i < len(tokens) and tokens[i] == "NEG":
        sign = -1.0
        i += 1

    digits: List[str] = []
    while i < len(tokens) and (tokens[i].isdigit() or tokens[i] == "POINT"):
        digits.append("." if tokens[i] == "POINT" else tokens[i])
        i += 1

    text = "".join(digits)
    if text in ("", "."):
        raise ExpressionParseError("CONST token without digits")
    try:
        return sign * float(text), i
    except ValueError as exc:
        raise ExpressionParseError(f"Malformed CONST digits {text!r}") from exc


def parse_coefficient(tokens: Sequence[str]) -> Optional[float]:
    """Parse a token sub-sequence that holds exactly one coefficient."""
    tokens = list(tokens)
    if not tokens:
        return None
    try:
        value, nxt = _take_coefficient(tokens, 0)
    except ExpressionParseError:
        return None
    return value if nxt == len(tokens) else None


# =============================================================================
# Algebraic families
# =============================================================================

def _expect(tokens: Sequence[str], i: int, expected: str) -> int:
    if i >= len(tokens) or tokens[i] != expected:
        got = tokens[i] if i < len(tokens) else "<end>"
        raise ExpressionParseError(f"Expected {expected} at position {i}, got {got}")
    return i + 1


def parse_polynomial(tokens: Sequence[str], x: Symbol = X) -> sympy.Expr:
    """Parse the polynomial template into a SymPy expression."""
    tokens = strip_special(tokens)
    if not tokens:
        raise ExpressionParseError("Empty polynomial token sequence")

    i = 0
    sign = 1.0
    if tokens[0] == "NEG":
        sign = -1.0
        i = 1

    expr = sympy.Integer(0)
    while True:
        coef, have_coef = 1.0, False
        if i < len(tokens) and _is_coefficient_start(tokens[i]):
            coef, i = _take_coefficient(tokens, i)
            have_coef = True
            if i < len(tokens) and tokens[i] == "MUL":
                i += 1
        power = 0
        if i < len(tokens) and tokens[i] == "X":
            power = 1
            i += 1
            if i < len(tokens) and tokens[i] == "POW":
                p, i = _take_coefficient(tokens, i + 1)
                if abs(p - round(p)) > 1e-9:
                    raise ExpressionParseError("Non-integer polynomial power")
                power = int(round(p))
        elif not have_coef:
            raise ExpressionParseError(f"Unexpected token {tokens[i] if i < len(tokens) else '<end>'} in polynomial")

        term = sympy.Float(sign * coef) * x ** power if power else sympy.Float(sign * coef)
        expr = expr + term

        if i >= len(tokens):
            break
        if tokens[i] == "ADD":
            sign = 1.0
        elif tokens[i] == "SUB":
            sign = -1.0
        else:
            raise ExpressionParseError(f"Unexpected token {tokens[i]!r} at position {i} in polynomial")
        i += 1
        if i >= len(tokens):
            raise ExpressionParseError("Dangling operator at end of polynomial")

    return expr


def parse_transcendental(tokens: Sequence[str], x: Symbol = X) -> sympy.Expr:
    """Parse one of the four transcendental templates."""
    tokens = strip_special(tokens)
    if not tokens:
        raise ExpressionParseError("Empty transcendental token sequence")

    if "DIV" in tokens and "EXP" in tokens:
        # L DIV INT_1 ADD EXP NEG k MUL X SUB x0
        L, i = _take_coefficient(tokens, 0)
        i = _expect(tokens, i, "DIV")
        i = _expect(tokens, i, "INT_1")
        i = _expect(tokens, i, "ADD")
        i = _expect(tokens, i, "EXP")
        i = _expect(tokens, i, "NEG")
        k, i = _take_coefficient(tokens, i)
        i = _expect(tokens, i, "MUL")
        i = _expect(tokens, i, "X")
        i = _expect(tokens, i, "SUB")
        x0, i = _take_coefficient(tokens, i)
        if i != len(tokens):
            raise ExpressionParseError("Trailing tokens after logistic template")
        return sympy.Float(L) / (1 + sympy.exp(-sympy.Float(k) * (x - sympy.Float(x0))))

    if "EXP" in tokens:
        # a MUL EXP b MUL X
        a, i = _take_coefficient(tokens, 0)
        i = _expect(tokens, i, "MUL")
        i = _expect(tokens, i, "EXP")
        b, i = _take_coefficient(tokens, i)
        i = _expect(tokens, i, "MUL")
        i = _expect(tokens, i, "X")
        if i != len(tokens):
            raise ExpressionParseError("Trailing tokens after exponential template")
        return sympy.Float(a) * sympy.exp(sympy.Float(b) * x)

    if "LN" in tokens or "LOG" in tokens:
        # a MUL LN X ADD b
        a, i = _take_coefficient(tokens, 0)
        i = _expect(tokens, i, "MUL")
        if i >= len(tokens) or tokens[i] not in ("LN", "LOG"):
            raise ExpressionParseError("Expected LN in logarithmic template")
        i += 1
        i = _expect(tokens, i, "X")
        if i == len(tokens):
            return sympy.Float(a) * sympy.log(x)
        if tokens[i] == "ADD":
            b, i = _take_coefficient(tokens, i + 1)
        elif tokens[i] == "SUB":
            b, i = _take_coefficient(tokens, i + 1)
            b = -b
        else:
            raise ExpressionParseError("Expected ADD/SUB in logarithmic template")
        if i != len(tokens):
            raise ExpressionParseError("Trailing tokens after logarithmic template")
        return sympy.Float(a) * sympy.log(x) + sympy.Float(b)

    if "POW" in tokens:
        # a MUL X POW b
        a, i = _take_coefficient(tokens, 0)
        i = _expect(tokens, i, "MUL")
        i = _expect(tokens, i, "X")
        i = _expect(tokens, i, "POW")
        b, i = _take_coefficient(tokens, i)
        if i != len(tokens):
            raise ExpressionParseError("Trailing tokens after power-law template")
        return sympy.Float(a) * x ** sympy.Float(b)

    raise ExpressionParseError("Tokens do not match a transcendental template")


def tokens_to_infix(tokens: Sequence[str]) -> str:
    """
    Best-effort infix rendering of an arbitrary token sequence.

    This mirrors the historical string conversion used by the test harness and
    is only used as a fallback when no template matches.
    """
    tokens = strip_special(tokens)
    parts: List[str] = []
    i = 0
    mapping = {"MUL": "*", "DIV": "/", "ADD": "+", "SUB": "-", "POW": "**", "NEG": "-"}

    def read_const(j: int) -> Tuple[str, int]:
        j += 1
        val = ""
        if j < len(tokens) and tokens[j] == "NEG":
            val += "-"
            j += 1
        while j < len(tokens) and (tokens[j].isdigit() or tokens[j] == "POINT"):
            val += "." if tokens[j] == "POINT" else tokens[j]
            j += 1
        return (val if val not in ("", "-", ".") else "0"), j

    while i < len(tokens):
        t = tokens[i]
        if t == "CONST":
            val, i = read_const(i)
            parts.append(val)
            continue
        if t in mapping:
            parts.append(mapping[t])
        elif t in SNAPPED_CONSTANTS:
            parts.append(repr(SNAPPED_CONSTANTS[t]))
        elif t in ("SIN", "COS", "TAN", "SQRT"):
            parts.append(t.lower() + "(x)")
            if i + 1 < len(tokens) and tokens[i + 1] == "X":
                i += 1
        elif t in ("LN", "LOG"):
            parts.append("log(x)")
            if i + 1 < len(tokens) and tokens[i + 1] == "X":
                i += 1
        elif t == "EXP":
            # Consume the argument up to the next ADD/SUB at this level.
            i += 1
            arg: List[str] = []
            while i < len(tokens) and tokens[i] not in ("ADD", "SUB"):
                tok = tokens[i]
                if tok == "CONST":
                    val, i = read_const(i)
                    arg.append(val)
                    continue
                if tok in mapping:
                    arg.append(mapping[tok])
                elif tok in SNAPPED_CONSTANTS:
                    arg.append(repr(SNAPPED_CONSTANTS[tok]))
                elif tok == "X":
                    arg.append("x")
                else:
                    arg.append(tok.lower())
                i += 1
            parts.append("exp(" + "".join(arg) + ")")
            continue
        elif t == "X":
            parts.append("x")
        else:
            parts.append(t.lower())
        i += 1

    return " ".join(parts)


def tokens_to_sympy(tokens: Sequence[str], model_name: str, x: Symbol = X) -> sympy.Expr:
    """
    Convert a predicted token sequence for an algebraic model into SymPy.

    Tries the template parser for the named family first, then the other
    algebraic family, then the generic infix fallback.
    """
    tokens = strip_special(tokens)
    if not tokens:
        raise ExpressionParseError("Empty token sequence")

    parsers = [parse_polynomial, parse_transcendental]
    if model_name == "transcendental":
        parsers.reverse()

    errors: List[str] = []
    for parser in parsers:
        try:
            return parser(tokens, x)
        except (ExpressionParseError, IndexError, ValueError, TypeError) as exc:
            errors.append(f"{parser.__name__}: {exc}")

    try:
        expr = sympy.sympify(tokens_to_infix(tokens), locals={"x": x})
        if expr.free_symbols - {x}:
            raise ExpressionParseError(f"Unexpected symbols {expr.free_symbols - {x}}")
        return expr
    except Exception as exc:  # sympify raises many different error types
        errors.append(f"infix: {exc}")

    raise ExpressionParseError("; ".join(errors))


# =============================================================================
# Time-series and ODE families
# =============================================================================

def parse_ar_tokens(tokens: Sequence[str]) -> List[float]:
    """Parse the AR template X MUL phi_1 MUL phi_2 ... into a coefficient list."""
    tokens = strip_special(tokens)
    if not tokens or tokens[0] != "X":
        raise ExpressionParseError("AR sequence must start with X")
    i = 1
    phi: List[float] = []
    while i < len(tokens):
        i = _expect(tokens, i, "MUL")
        value, i = _take_coefficient(tokens, i)
        phi.append(value)
    if not phi:
        raise ExpressionParseError("AR sequence has no coefficients")
    return phi


def parse_ode_tokens(tokens: Sequence[str]) -> Optional[Dict[str, float]]:
    """
    Parse ODE coefficients from a token sequence.

    First-order:  DX_DT EQ <a> MUL X                      -> {"order": 1, "a": a}
    Second-order: D2X_DT2 ADD <C1> MUL DX_DT ADD <C2> MUL X EQ INT_0
                                                          -> {"order": 2, "C1": C1, "C2": C2}
    Returns None when the sequence does not match either template.
    """
    tokens = strip_special(tokens)
    try:
        if "D2X_DT2" in tokens:
            d2_idx = tokens.index("D2X_DT2")
            first_add = tokens.index("ADD", d2_idx + 1)
            first_mul = tokens.index("MUL", first_add + 1)
            C1 = parse_coefficient(tokens[first_add + 1:first_mul])
            second_add = tokens.index("ADD", first_mul + 1)
            second_mul = tokens.index("MUL", second_add + 1)
            C2 = parse_coefficient(tokens[second_add + 1:second_mul])
            if C1 is None or C2 is None:
                return None
            return {"order": 2, "C1": C1, "C2": C2}
        if "DX_DT" in tokens:
            eq_idx = tokens.index("EQ")
            mul_idx = tokens.index("MUL", eq_idx + 1)
            a = parse_coefficient(tokens[eq_idx + 1:mul_idx])
            if a is None:
                return None
            return {"order": 1, "a": a}
    except ValueError:
        return None
    return None


# =============================================================================
# Formatting
# =============================================================================

def round_floats(expr: sympy.Expr, digits: int = 6) -> sympy.Expr:
    """Round every Float atom in an expression to `digits` significant figures."""
    replacements = {}
    for atom in expr.atoms(sympy.Float):
        value = float(atom)
        if value == 0.0 or not math.isfinite(value):
            continue
        replacements[atom] = sympy.Float(f"{value:.{digits}g}")
    return expr.xreplace(replacements) if replacements else expr


def prune_small_terms(expr: sympy.Expr, tol: float) -> sympy.Expr:
    """
    Drop top-level additive terms whose coefficient magnitude is below `tol`.

    Least-squares refits leave behind terms like 1.6e-17*x or 3.7e-16; `tol`
    should be tiny relative to the scale of y. Only the outermost sum is
    touched so coefficients inside function arguments are never altered.
    """
    if tol <= 0 or not isinstance(expr, sympy.Add):
        return expr
    kept = [term for term in expr.args if abs(float(term.as_coeff_Mul()[0])) >= tol]
    if len(kept) == len(expr.args):
        return expr
    return sympy.Add(*kept) if kept else sympy.Float(0.0)


def expression_to_string(expr: sympy.Expr, digits: int = 6) -> str:
    """Render an expression as a SymPy-parseable string with rounded floats."""
    return str(round_floats(expr, digits))


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def format_ar_equation(phi: Sequence[float], digits: int = 6) -> str:
    terms = []
    for lag, coef in enumerate(phi, start=1):
        terms.append(f"{_fmt(coef, digits)}*y[t-{lag}]")
    return "y[t] = " + " + ".join(terms)


def _signed_term(coef: float, name: str, digits: int) -> str:
    sign = "-" if coef < 0 else "+"
    return f" {sign} {_fmt(abs(coef), digits)}*{name}"


def format_ode_equation(params: Dict[str, float], digits: int = 6) -> str:
    if params.get("order") == 1:
        return f"dx/dt = {_fmt(params['a'], digits)}*x"
    text = "x''"
    if abs(params["C1"]) > 1e-12:
        text += _signed_term(params["C1"], "x'", digits)
    text += _signed_term(params["C2"], "x", digits)
    return text + " = 0"

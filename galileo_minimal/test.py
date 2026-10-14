#!/usr/bin/env python3
"""
test.py - Model evaluation with tier support, seeded tests, and Phase 4 ODE comparator

Usage:
    python test.py --model polynomial --tier 1
    python test.py --model ar --tier 2
    python test.py --model linear_ode --tier 1
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import sympy
from sympy import Symbol, sympify, expand

from config import get_device, CONFIGS, DOMAINS
from vocab import Vocabulary
from model import SymbolicTransformer
from generators import get_generator


# =============================================================================
# Checkpoint / Vocab loading
# =============================================================================

def infer_vocab_size_from_ckpt(ckpt: dict) -> int:
    sd = ckpt["model"]
    if "token_embed.weight" in sd:
        return sd["token_embed.weight"].shape[0]
    if "output_proj.weight" in sd:
        return sd["output_proj.weight"].shape[0]
    raise KeyError("Could not infer vocab size from checkpoint state_dict.")


def load_vocab_for_checkpoint(checkpoint_dir: Path, ckpt: dict) -> Vocabulary:
    """Load vocab from checkpoint directory or infer from checkpoint."""
    vocab_json = checkpoint_dir / "vocab.json"
    if vocab_json.exists():
        return Vocabulary.load(vocab_json)

    if isinstance(ckpt, dict) and "vocab_tokens" in ckpt:
        return Vocabulary(ckpt["vocab_tokens"])

    # Fallback: infer size and slice default
    target = infer_vocab_size_from_ckpt(ckpt)
    default = Vocabulary.default()
    if target > default.vocab_size:
        raise RuntimeError(
            f"Checkpoint expects vocab_size={target}, but current default vocab_size={default.vocab_size}. "
            "Restore old vocab or store vocab with checkpoints."
        )
    return Vocabulary(default.tokens[:target])


# =============================================================================
# Expression Cleanup (post-processing)
# =============================================================================

COMMON_FRACTIONS = {
    0.5: sympy.Rational(1, 2),
    0.25: sympy.Rational(1, 4),
    0.75: sympy.Rational(3, 4),
    0.333: sympy.Rational(1, 3),
    0.667: sympy.Rational(2, 3),
    0.2: sympy.Rational(1, 5),
    0.4: sympy.Rational(2, 5),
    0.6: sympy.Rational(3, 5),
    0.8: sympy.Rational(4, 5),
    1.5: sympy.Rational(3, 2),
}


def snap_coefficient(coef: float, thresh: float = 0.05):
    """Snap to integer or common fraction if close."""
    rounded = round(coef)
    if abs(coef - rounded) < thresh:
        return int(rounded)
    for val, frac in COMMON_FRACTIONS.items():
        if abs(coef - val) < thresh:
            return frac
        if abs(coef + val) < thresh:
            return -frac
    return coef


def clean_expression(expr, prune_thresh: float = 0.1, snap_thresh: float = 0.05):
    """Prune tiny terms and snap coefficients."""
    if expr.is_Atom:
        if expr.is_Number:
            return snap_coefficient(float(expr), snap_thresh)
        return expr

    try:
        expanded_expr = sympy.expand(expr)
        terms = sympy.Add.make_args(expanded_expr)

        max_coef = 0.0
        term_data = []
        for term in terms:
            coef, rest = term.as_coeff_Mul()
            coef_f = float(coef)
            term_data.append((coef_f, rest))
            max_coef = max(max_coef, abs(coef_f))

        effective_thresh = max(prune_thresh, 0.05 * max_coef)

        cleaned = []
        for coef_f, rest in term_data:
            if abs(coef_f) < effective_thresh:
                continue
            snapped = snap_coefficient(coef_f, snap_thresh)
            if snapped == 0:
                continue
            if rest == 1:
                cleaned.append(sympy.sympify(snapped))
            else:
                cleaned.append(snapped * rest)

        return sympy.Add(*cleaned) if cleaned else sympy.Integer(0)
    except Exception:
        return expr


def tokens_to_string(tokens: List[str]) -> str:
    """Convert token list to expression string (valid SymPy syntax) where possible."""
    parts: List[str] = []
    i = 0
    mapping = {"MUL": "*", "DIV": "/", "ADD": "+", "SUB": "-", "POW": "**", "NEG": "-"}

    while i < len(tokens):
        t = tokens[i]

        if t == "CONST":
            i += 1
            val = ""
            if i < len(tokens) and tokens[i] == "NEG":
                val += "-"
                i += 1
            while i < len(tokens) and (tokens[i].isdigit() or tokens[i] == "POINT"):
                val += "." if tokens[i] == "POINT" else tokens[i]
                i += 1
            parts.append(val if val else "C")
            continue

        if t in mapping:
            parts.append(mapping[t])
        elif t.startswith("INT_"):
            parts.append(t.split("_")[1])
        elif t in ("SIN", "COS", "TAN", "SQRT"):
            parts.append(t.lower() + "(")
            i += 1
            if i < len(tokens):
                if tokens[i] == "X":
                    parts.append("X)")
                else:
                    parts.append(tokens[i] + ")")
            else:
                parts.append("X)")
            i += 1
            continue
        elif t == "EXP":
            parts.append("exp(")
            i += 1
            arg_parts: List[str] = []
            while i < len(tokens):
                tok = tokens[i]
                if tok in ("ADD", "SUB"):
                    break
                if tok == "CONST":
                    i += 1
                    val = ""
                    if i < len(tokens) and tokens[i] == "NEG":
                        val += "-"
                        i += 1
                    while i < len(tokens) and (tokens[i].isdigit() or tokens[i] == "POINT"):
                        val += "." if tokens[i] == "POINT" else tokens[i]
                        i += 1
                    arg_parts.append(val if val else "0")
                    continue
                if tok == "MUL":
                    arg_parts.append("*")
                elif tok == "X":
                    arg_parts.append("X")
                elif tok.startswith("INT_"):
                    arg_parts.append(tok.split("_")[1])
                else:
                    arg_parts.append(tok)
                i += 1
            parts.append("".join(arg_parts) + ")")
            continue
        elif t in ("LN", "LOG"):
            parts.append("log(")
            i += 1
            if i < len(tokens) and tokens[i] == "X":
                parts.append("X)")
                i += 1
            else:
                parts.append("X)")
            continue
        elif t == "HALF":
            parts.append("0.5")
        else:
            parts.append(t)

        i += 1

    return " ".join(parts)


def cleanup_prediction(pred_str: str) -> str:
    """Apply expression cleanup to prediction string."""
    try:
        expr = sympy.sympify(pred_str)
        cleaned = clean_expression(expr)
        return str(cleaned)
    except Exception:
        return pred_str


# =============================================================================
# PHASE 4: ODE-aware comparison for linear_ode
# =============================================================================

# Shared with the tool layer so both paths parse ODE tokens identically.
from expression import parse_ode_tokens as extract_ode_coefficients  # noqa: E402
from expression import parse_coefficient as _parse_coefficient_tokens  # noqa: E402


def compare_linear_ode(truth_tokens: List[str], pred_tokens: List[str], tolerance: float = 0.25) -> Dict[str, Any]:
    """
    PHASE 4: Compare linear ODEs by coefficient recovery.
    
    First-order: check coefficient 'a' with relative error
    Second-order: check coefficients C1 and C2 with relative error
    """
    truth_coefs = extract_ode_coefficients(truth_tokens)
    pred_coefs = extract_ode_coefficients(pred_tokens)
    
    if truth_coefs is None or pred_coefs is None:
        return {
            "match": False, 
            "relative_error": float("inf"), 
            "error": "Could not parse ODE coefficients"
        }
    
    if truth_coefs["order"] != pred_coefs["order"]:
        return {
            "match": False,
            "relative_error": float("inf"),
            "error": f"Order mismatch: truth={truth_coefs['order']}, pred={pred_coefs['order']}"
        }
    
    if truth_coefs["order"] == 1:
        a_t = truth_coefs["a"]
        a_p = pred_coefs["a"]
        
        rel = abs(a_t - a_p) / max(1e-8, abs(a_t))
        sign_ok = (a_t > 0) == (a_p > 0)
        
        return {
            "match": bool(sign_ok and (rel < tolerance)),
            "relative_error": float(rel),
            "truth_a": float(a_t),
            "pred_a": float(a_p),
            "sign_ok": bool(sign_ok),
            "mode": "first_order",
        }
    
    else:  # order == 2
        C1_t, C2_t = truth_coefs["C1"], truth_coefs["C2"]
        C1_p, C2_p = pred_coefs["C1"], pred_coefs["C2"]
        
        # Relative error for both coefficients
        rel_C1 = abs(C1_t - C1_p) / max(1e-8, abs(C1_t))
        rel_C2 = abs(C2_t - C2_p) / max(1e-8, abs(C2_t))
        
        # Overall match if both coefficients are within tolerance
        match = (rel_C1 < tolerance) and (rel_C2 < tolerance)
        avg_rel_error = (rel_C1 + rel_C2) / 2
        
        return {
            "match": bool(match),
            "relative_error": float(avg_rel_error),
            "truth_C1": float(C1_t),
            "truth_C2": float(C2_t),
            "pred_C1": float(C1_p),
            "pred_C2": float(C2_p),
            "rel_C1": float(rel_C1),
            "rel_C2": float(rel_C2),
            "mode": "second_order",
        }


# =============================================================================
# Numerical Comparison (non-ODE)
# =============================================================================

def compare_multivariate(truth_expr, pred_expr, tolerance: float = 0.1) -> Dict[str, Any]:
    """Compare multivariate expressions by structure and coefficient ratio."""
    try:
        truth_expanded = expand(truth_expr)
        pred_expanded = expand(pred_expr)

        try:
            ratio = sympy.simplify(truth_expanded / pred_expanded)
            if ratio.is_number:
                ratio_val = float(ratio)
                coef_error = abs(ratio_val - 1.0)
                return {
                    "match": coef_error < tolerance,
                    "rmse": coef_error,
                    "max_error": coef_error,
                    "relative_error": coef_error,
                    "ratio": ratio_val,
                }
        except Exception:
            pass

        def normalize_expr(expr):
            terms = sympy.Add.make_args(expand(expr))
            if terms:
                first_coef = terms[0].as_coeff_Mul()[0]
                if first_coef != 0:
                    return sympy.simplify(expr / first_coef)
            return expr

        truth_norm = normalize_expr(truth_expanded)
        pred_norm = normalize_expr(pred_expanded)
        structure_match = sympy.simplify(truth_norm - pred_norm) == 0

        return {
            "match": bool(structure_match),
            "rmse": 0.0 if structure_match else float("inf"),
            "max_error": 0.0 if structure_match else float("inf"),
            "relative_error": 0.0 if structure_match else float("inf"),
            "structure_match": bool(structure_match),
        }
    except Exception as e:
        return {
            "match": False,
            "rmse": float("inf"),
            "max_error": float("inf"),
            "relative_error": float("inf"),
            "error": str(e),
        }


def compare_expressions_numerically(
    truth_str: str,
    pred_str: str,
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 50,
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """Compare two expressions by evaluating them numerically."""
    try:
        truth_expr = sympify(truth_str)
        pred_expr = sympify(pred_str)

        truth_symbols = truth_expr.free_symbols
        pred_symbols = pred_expr.free_symbols
        all_symbols = truth_symbols | pred_symbols

        if len(all_symbols) > 1:
            return compare_multivariate(truth_expr, pred_expr, tolerance)

        X = Symbol("X")
        if all_symbols and X not in all_symbols:
            X = list(all_symbols)[0]

        expr_str_lower = (str(truth_expr).lower() + " " + str(pred_expr).lower())
        needs_positive = any(fn in expr_str_lower for fn in ["log", "ln", "sqrt"])
        if needs_positive and x_range[0] <= 0:
            x_range = (0.1, 10.0)

        truth_fn = sympy.lambdify(X, truth_expr, modules=["numpy"])
        pred_fn = sympy.lambdify(X, pred_expr, modules=["numpy"])

        x_vals = np.linspace(x_range[0], x_range[1], n_points)

        truth_vals = truth_fn(x_vals)
        pred_vals = pred_fn(x_vals)

        if np.isscalar(truth_vals):
            truth_vals = np.full_like(x_vals, truth_vals, dtype=np.float64)
        if np.isscalar(pred_vals):
            pred_vals = np.full_like(x_vals, pred_vals, dtype=np.float64)

        valid_mask = np.isfinite(truth_vals) & np.isfinite(pred_vals)
        if not valid_mask.any():
            return {
                "match": False,
                "rmse": float("inf"),
                "max_error": float("inf"),
                "relative_error": float("inf"),
                "error": "No valid evaluation points",
            }

        truth_vals = np.asarray(truth_vals)[valid_mask]
        pred_vals = np.asarray(pred_vals)[valid_mask]

        errors = np.abs(truth_vals - pred_vals)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        max_error = float(np.max(errors))

        truth_range = float(np.max(truth_vals) - np.min(truth_vals))
        relative_error = rmse / truth_range if truth_range > 1e-8 else rmse

        return {
            "match": relative_error < tolerance,
            "rmse": rmse,
            "max_error": max_error,
            "relative_error": float(relative_error),
        }
    except Exception as e:
        return {
            "match": False,
            "rmse": float("inf"),
            "max_error": float("inf"),
            "relative_error": float("inf"),
            "error": str(e),
        }


def compare_coefficients(truth_str: str, pred_str: str, tolerance: float = 0.1) -> Dict[str, Any]:
    """Compare polynomial coefficients directly."""
    try:
        X = Symbol("X")
        truth_expr = expand(sympify(truth_str))
        pred_expr = expand(sympify(pred_str))

        truth_poly = sympy.Poly(truth_expr, X)
        pred_poly = sympy.Poly(pred_expr, X)

        structure_match = truth_poly.degree() == pred_poly.degree()

        truth_coeffs = truth_poly.as_dict()
        pred_coeffs = pred_poly.as_dict()

        all_powers = set(truth_coeffs.keys()) | set(pred_coeffs.keys())

        total_error = 0.0
        max_coeff = 0.0
        for power in all_powers:
            t_coef = float(truth_coeffs.get(power, 0.0))
            p_coef = float(pred_coeffs.get(power, 0.0))
            total_error += abs(t_coef - p_coef)
            max_coeff = max(max_coeff, abs(t_coef), abs(p_coef))

        if max_coeff > 1e-8 and len(all_powers) > 0:
            relative_coeff_error = total_error / (max_coeff * len(all_powers))
        else:
            relative_coeff_error = total_error

        return {
            "match": bool(structure_match and (relative_coeff_error < tolerance)),
            "coeff_error": float(total_error),
            "relative_coeff_error": float(relative_coeff_error),
            "structure_match": bool(structure_match),
            "truth_degree": int(truth_poly.degree()),
            "pred_degree": int(pred_poly.degree()),
        }
    except Exception as e:
        return {
            "match": False,
            "coeff_error": float("inf"),
            "relative_coeff_error": float("inf"),
            "structure_match": False,
            "error": str(e),
        }


# =============================================================================
# PHASE 5: Test Cases with Seeds
# =============================================================================

TEST_CASES = {
    "polynomial": [
        {"name": "linear_simple_seed0", "subtype": "linear_positive", "seed": 0, "threshold": 0.15},
        {"name": "linear_simple_seed1", "subtype": "linear_positive", "seed": 1, "threshold": 0.15},
        {"name": "quadratic_vertex_seed0", "subtype": "quadratic_up", "seed": 0, "threshold": 0.15},
        {"name": "cubic_seed0", "subtype": "cubic", "seed": 0, "threshold": 0.15},
        {"name": "cubic_seed1", "subtype": "cubic", "seed": 1, "threshold": 0.15},
    ],
    "ar": [
        {"name": "ar1_medium_seed0", "subtype": "ar1_medium", "seed": 0, "threshold": 0.2},
        {"name": "ar1_high_seed0", "subtype": "ar1_high", "seed": 0, "threshold": 0.2},
        {"name": "ar1_high_seed1", "subtype": "ar1_high", "seed": 1, "threshold": 0.2},
    ],
    "transcendental": [
        {"name": "exp_decay_seed0", "subtype": "exp_decay", "seed": 0, "threshold": 0.2},
        {"name": "power_law_seed0", "subtype": "power_positive", "seed": 0, "threshold": 0.2},
        {"name": "logarithmic_seed0", "subtype": "logarithmic", "seed": 0, "threshold": 0.2},
    ],
    "physics_laws": [
        {"name": "newton_gravity_seed0", "subtype": "newton_gravity", "seed": 0, "threshold": 0.2},
        {"name": "coulomb_seed0", "subtype": "coulomb", "seed": 0, "threshold": 0.2},
        {"name": "kinetic_energy_seed0", "subtype": "kinetic_energy", "seed": 0, "threshold": 0.2},
    ],
    "linear_ode": [
        {"name": "first_order_decay_seed0", "subtype": "first_order_decay", "seed": 0, "threshold": 0.25},
        {"name": "first_order_growth_seed0", "subtype": "first_order_growth", "seed": 0, "threshold": 0.25},
        {"name": "underdamped_seed0", "subtype": "underdamped", "seed": 0, "threshold": 0.25},
        {"name": "underdamped_seed1", "subtype": "underdamped", "seed": 1, "threshold": 0.25},
        {"name": "overdamped_seed0", "subtype": "overdamped", "seed": 0, "threshold": 0.25},
        {"name": "undamped_seed0", "subtype": "undamped", "seed": 0, "threshold": 0.25},
    ],
}


# =============================================================================
# Runner
# =============================================================================

def _load_model_and_vocab(model_name: str, tier: int, device: str):
    if model_name not in CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    cfg = CONFIGS[model_name]
    domain = DOMAINS[model_name]

    checkpoint_path = Path("checkpoints") / domain / model_name / f"tier{tier}" / "model_best.pt"
    
    # Legacy fallback
    if not checkpoint_path.exists():
        legacy_path = Path("checkpoints") / domain / model_name / "model_best.pt"
        if legacy_path.exists():
            print(f"⚠️ No tier checkpoint found, using legacy: {legacy_path}")
            checkpoint_path = legacy_path
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or legacy path")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = load_vocab_for_checkpoint(checkpoint_path.parent, ckpt)

    model = SymbolicTransformer(vocab_size=vocab.vocab_size, **cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, vocab, checkpoint_path


def test(args):
    device = get_device()

    try:
        model, vocab, checkpoint_path = _load_model_and_vocab(args.model, args.tier, device)
    except Exception as e:
        print(str(e))
        return

    print(f"Loaded: {checkpoint_path}", flush=True)

    test_cases = TEST_CASES.get(args.model, [])
    if not test_cases:
        print(f"No test cases for {args.model}")
        return

    generator = get_generator(args.model)

    print(f"\nRunning {len(test_cases)} tests for {args.model} (Tier {args.tier})...")
    print("=" * 70)

    passed = 0
    for tc in test_cases:
        subtype = tc.get("subtype")
        threshold = float(tc["threshold"])
        seed = tc.get("seed", 0)
        
        # PHASE 5: Seed the generator
        np.random.seed(seed)

        features, truth_tokens, _actual_subtype = generator.generate(subtype=subtype, tier=args.tier)

        feat_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_ids = model.generate(feat_tensor, vocab)
        pred_tokens = vocab.decode(pred_ids)

        # ODE-aware path
        if args.model == "linear_ode":
            ode_result = compare_linear_ode(truth_tokens, pred_tokens, tolerance=threshold)
            match = bool(ode_result["match"])

            truth_str = " ".join(truth_tokens)
            pred_str = " ".join(pred_tokens)

            status = "PASS" if match else "FAIL"
            color = "\033[92m" if match else "\033[91m"
            print(f"{color}[{status}] {tc['name']}\033[0m")
            print(f"  Truth:     {truth_str}")
            print(f"  Predicted: {pred_str}")
            
            if ode_result.get("mode") == "first_order":
                print(
                    f"  ODE:       a_truth={ode_result['truth_a']:.4f} "
                    f"a_pred={ode_result['pred_a']:.4f} "
                    f"rel_err={ode_result['relative_error']:.4f} "
                    f"sign_ok={ode_result['sign_ok']} (threshold={threshold})"
                )
            elif ode_result.get("mode") == "second_order":
                print(
                    f"  ODE:       C1_truth={ode_result['truth_C1']:.4f} C1_pred={ode_result['pred_C1']:.4f} rel_C1={ode_result['rel_C1']:.4f}"
                )
                print(
                    f"             C2_truth={ode_result['truth_C2']:.4f} C2_pred={ode_result['pred_C2']:.4f} rel_C2={ode_result['rel_C2']:.4f}"
                )
                print(f"             avg_rel_err={ode_result['relative_error']:.4f} (threshold={threshold})")
            
            print("-" * 70)

            if match:
                passed += 1
            continue

        # Non-ODE: convert to strings and compare
        truth_str = tokens_to_string(truth_tokens)
        pred_str = tokens_to_string(pred_tokens)

        if args.cleanup:
            pred_str = cleanup_prediction(pred_str)
            truth_str = cleanup_prediction(truth_str)

        num_result = compare_expressions_numerically(truth_str, pred_str, tolerance=threshold)
        coef_result = compare_coefficients(truth_str, pred_str, tolerance=threshold)

        match = bool(num_result["match"] or coef_result["match"])

        status = "PASS" if match else "FAIL"
        color = "\033[92m" if match else "\033[91m"

        print(f"{color}[{status}] {tc['name']}\033[0m")
        print(f"  Truth:     {truth_str}")
        print(f"  Predicted: {pred_str}")
        print(f"  Numerical: rel_err={num_result['relative_error']:.4f} (threshold={threshold})")

        if coef_result.get("structure_match") is not None:
            struct = "✓" if coef_result["structure_match"] else "✗"
            rel_err = coef_result.get("relative_coeff_error", float("inf"))
            if rel_err != float("inf"):
                print(f"  Coeffs:    rel_err={rel_err:.4f}, structure={struct}")

        print("-" * 70)

        if match:
            passed += 1

    print(f"\nSummary: {passed}/{len(test_cases)} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3], help="Difficulty tier to test")
    parser.add_argument("--cleanup", action="store_true", default=True, help="Apply coefficient cleanup")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false")
    args = parser.parse_args()

    test(args)

#!/usr/bin/env python3
"""
test_overseer.py - Test suite for the Overseer Router

Tests classification accuracy across different data types and edge cases.

Usage:
    python test_overseer.py
    python test_overseer.py --with-inference  # Also test actual model routing
    python test_overseer.py --verbose         # Show detailed output
"""
import os
import sys
import argparse
import numpy as np
from scipy.integrate import odeint

# Color output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _finalize_test(result: bool):
    """Return bool for CLI mode, assert in pytest mode."""
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        assert result
        return None
    return result


def check(condition, message, details=None):
    """Check a condition and print status."""
    if condition:
        print(f"{GREEN}✓{RESET} {message}")
        return True
    else:
        print(f"{RED}✗{RESET} {message}")
        if details:
            print(f"  {YELLOW}Details:{RESET} {details}")
        return False


def test_data_type_detection():
    """Test that data types are correctly detected."""
    print("\n=== Testing Data Type Detection ===")
    from overseer import Overseer, DataType
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Univariate (x, y) data
    x = np.linspace(-5, 5, 100)
    y = x**2 + 2*x + 1
    result = overseer.classify(y, x)
    all_ok &= check(
        result.data_type == DataType.UNIVARIATE,
        "Detects univariate data correctly",
        f"Got {result.data_type}, expected UNIVARIATE"
    )
    
    # Test 2: Timeseries (no x)
    y = np.cumsum(np.random.randn(100))
    result = overseer.classify(y)
    all_ok &= check(
        result.data_type == DataType.TIMESERIES,
        "Detects timeseries (no x) correctly",
        f"Got {result.data_type}, expected TIMESERIES"
    )
    
    # Test 3: ODE trajectory (oscillatory)
    t = np.linspace(0, 10, 200)
    y = np.exp(-0.3 * t) * np.cos(2 * t)
    result = overseer.classify(y, t)
    all_ok &= check(
        result.data_type == DataType.ODE_TRAJECTORY,
        "Detects ODE trajectory (damped oscillator)",
        f"Got {result.data_type}, expected ODE_TRAJECTORY"
    )
    
    # Test 4: Multivariate data
    data = {
        "M1": np.random.uniform(1, 10, 50),
        "M2": np.random.uniform(1, 10, 50),
        "R": np.random.uniform(1, 10, 50),
        "output": np.random.uniform(1, 100, 50),
    }
    result = overseer.classify(data["output"], data=data)
    all_ok &= check(
        result.data_type == DataType.MULTIVARIATE,
        "Detects multivariate data correctly",
        f"Got {result.data_type}, expected MULTIVARIATE"
    )
    
    return _finalize_test(all_ok)


def test_polynomial_classification():
    """Test classification of polynomial data."""
    print("\n=== Testing Polynomial Classification ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Linear
    x = np.linspace(-5, 5, 100)
    y = 2*x + 3
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model == "polynomial",
        f"Linear data → polynomial (got {result.best_model})"
    )
    
    # Test 2: Quadratic
    y = x**2 - 2*x + 1
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model == "polynomial",
        f"Quadratic data → polynomial (got {result.best_model})"
    )
    
    # Test 3: Cubic
    y = 0.5*x**3 - 2*x**2 + x + 1
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model == "polynomial",
        f"Cubic data → polynomial (got {result.best_model})"
    )
    
    # Test 4: High confidence for clean polynomial
    y = 3*x**2 + 2*x + 1
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_confidence > 0.25,
        f"Clean polynomial has reasonable confidence ({result.best_confidence:.2f})"
    )
    
    return _finalize_test(all_ok)


def test_transcendental_classification():
    """Test classification of transcendental functions."""
    print("\n=== Testing Transcendental Classification ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Exponential growth
    x = np.linspace(0, 5, 100)
    y = 2 * np.exp(0.5 * x)
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model in ["transcendental", "linear_ode"],
        f"Exponential growth → transcendental or ODE (got {result.best_model})"
    )
    
    # Test 2: Exponential decay
    y = 5 * np.exp(-0.3 * x)
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model in ["transcendental", "linear_ode"],
        f"Exponential decay → transcendental or ODE (got {result.best_model})"
    )
    
    # Test 3: Power law
    x = np.linspace(0.1, 10, 100)
    y = 2 * x**1.5
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model in ["transcendental", "polynomial"],
        f"Power law → transcendental or polynomial (got {result.best_model})"
    )
    
    # Test 4: Logarithmic (can look like polynomial over limited range)
    y = 3 * np.log(x) + 1
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model in ["transcendental", "polynomial", "linear_ode"],
        f"Logarithmic → transcendental/polynomial/ode (got {result.best_model})"
    )
    
    return _finalize_test(all_ok)


def test_ode_classification():
    """Test classification of ODE trajectories."""
    print("\n=== Testing ODE Classification ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Undamped oscillator (simple harmonic)
    t = np.linspace(0, 10, 200)
    y = np.cos(2 * t)
    result = overseer.classify(y, t)
    all_ok &= check(
        result.best_model == "linear_ode",
        f"Simple harmonic → linear_ode (got {result.best_model})"
    )
    
    # Test 2: Underdamped oscillator
    omega, zeta = 2.0, 0.15
    omega_d = omega * np.sqrt(1 - zeta**2)
    y = np.exp(-zeta * omega * t) * np.cos(omega_d * t)
    result = overseer.classify(y, t)
    all_ok &= check(
        result.best_model == "linear_ode",
        f"Underdamped oscillator → linear_ode (got {result.best_model})"
    )
    
    # Test 3: Overdamped (no oscillation, pure decay)
    C1, C2 = 6.0, 4.0  # zeta > 1
    
    def overdamped_system(state, t):
        x, v = state
        return [v, -C1*v - C2*x]
    
    sol = odeint(overdamped_system, [1.0, 0.0], t)
    y = sol[:, 0]
    result = overseer.classify(y, t)
    # Overdamped looks like exponential decay, so could be transcendental
    all_ok &= check(
        result.best_model in ["linear_ode", "transcendental"],
        f"Overdamped → linear_ode or transcendental (got {result.best_model})"
    )
    
    # Test 4: First-order decay (dx/dt = -0.5x)
    y = np.exp(-0.5 * t)
    result = overseer.classify(y, t)
    all_ok &= check(
        result.best_model in ["linear_ode", "transcendental"],
        f"First-order decay → linear_ode or transcendental (got {result.best_model})"
    )
    
    return _finalize_test(all_ok)


def test_ar_classification():
    """Test classification of AR timeseries."""
    print("\n=== Testing AR Classification ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: AR(1) with high persistence
    np.random.seed(42)
    y = np.zeros(200)
    phi = 0.9
    for i in range(1, 200):
        y[i] = phi * y[i-1] + np.random.normal(0, 0.1)
    
    result = overseer.classify(y)
    all_ok &= check(
        result.best_model == "ar",
        f"AR(1) φ=0.9 → ar (got {result.best_model})"
    )
    
    # Test 2: AR(1) with medium persistence
    np.random.seed(42)
    y = np.zeros(200)
    phi = 0.7
    for i in range(1, 200):
        y[i] = phi * y[i-1] + np.random.normal(0, 0.1)
    
    result = overseer.classify(y)
    all_ok &= check(
        result.best_model == "ar",
        f"AR(1) φ=0.7 → ar (got {result.best_model})"
    )
    
    # Test 3: Check ACF features
    result = overseer.classify(y)
    acf_lag1 = result.features.get("acf_lag1", 0)
    all_ok &= check(
        abs(acf_lag1 - 0.7) < 0.15,
        f"ACF(1) correctly estimated (~0.7, got {acf_lag1:.3f})"
    )
    
    return _finalize_test(all_ok)


def test_physics_classification():
    """Test classification of physics law data."""
    print("\n=== Testing Physics Laws Classification ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Inverse square law
    n = 100
    R = np.random.uniform(1, 10, n)
    M1 = np.random.uniform(1, 5, n)
    M2 = np.random.uniform(1, 5, n)
    F = M1 * M2 / R**2
    
    data = {"M1": M1, "M2": M2, "R": R, "output": F}
    result = overseer.classify(F, data=data)
    all_ok &= check(
        result.best_model == "physics_laws",
        f"Inverse square law → physics_laws (got {result.best_model})"
    )
    
    # Test 2: Linear (Hooke's law)
    k = 100
    x = np.random.uniform(-1, 1, n)
    F = k * x
    
    data = {"k": np.full(n, k), "x": x, "output": F}
    result = overseer.classify(F, data=data)
    all_ok &= check(
        result.best_model == "physics_laws",
        f"Hooke's law → physics_laws (got {result.best_model})"
    )
    
    return _finalize_test(all_ok)


def test_feature_extraction():
    """Test that diagnostic features are correctly computed."""
    print("\n=== Testing Feature Extraction ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Zero crossings
    t = np.linspace(0, 10, 200)
    y = np.sin(2 * t)  # ~6 zero crossings in [0, 10]
    result = overseer.classify(y, t)
    
    zc = result.features.get("zero_crossings", 0)
    all_ok &= check(
        5 <= zc <= 8,
        f"Zero crossings for sin(2t) in [5,8] (got {zc})"
    )
    
    # Test 2: Monotonicity
    x = np.linspace(0, 10, 100)
    y = x**2  # Monotonically increasing for x > 0
    result = overseer.classify(y, x)
    
    mono = result.features.get("is_monotonic", 0)
    all_ok &= check(
        mono > 0.5,
        f"x² on [0,10] detected as monotonic"
    )
    
    # Test 3: Spectral peak
    t = np.linspace(0, 10, 200)
    y = np.cos(2 * t)  # Strong peak at f = 2/(2π) ≈ 0.318 Hz
    result = overseer.classify(y, t)
    
    peak_ratio = result.features.get("spectral_peak_ratio", 0)
    all_ok &= check(
        peak_ratio > 0.3,
        f"Strong spectral peak for cos(2t) (ratio={peak_ratio:.3f})"
    )
    
    # Test 4: Envelope decay
    y = np.exp(-0.3 * t) * np.cos(2 * t)
    result = overseer.classify(y, t)
    
    decay = result.features.get("envelope_decay_rate", 0)
    all_ok &= check(
        decay < 0,  # Just needs to be negative (decaying)
        f"Envelope decay detected ({decay:.3f})"
    )
    
    return _finalize_test(all_ok)


def test_ranking():
    """Test that model rankings are sensible."""
    print("\n=== Testing Model Ranking ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Rankings sum to ~1 (probabilities)
    x = np.linspace(-5, 5, 100)
    y = x**2
    result = overseer.classify(y, x)
    
    total_prob = sum(conf for _, conf in result.rankings)
    all_ok &= check(
        abs(total_prob - 1.0) < 0.01,
        f"Rankings sum to 1.0 (got {total_prob:.4f})"
    )
    
    # Test 2: top_k returns correct number
    all_ok &= check(
        len(result.top_k(3)) == 3,
        "top_k(3) returns 3 models"
    )
    
    # Test 3: Rankings are sorted by confidence
    confs = [conf for _, conf in result.rankings]
    all_ok &= check(
        confs == sorted(confs, reverse=True),
        "Rankings are sorted by confidence (descending)"
    )
    
    return _finalize_test(all_ok)


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    # Test 1: Very short data
    y = np.array([1, 2, 3, 4, 5])
    result = overseer.classify(y)
    all_ok &= check(
        result.best_model is not None,
        "Handles very short data (n=5)"
    )
    
    # Test 2: Constant data
    y = np.ones(100)
    result = overseer.classify(y)
    all_ok &= check(
        result.best_model is not None,
        "Handles constant data"
    )
    
    # Test 3: Data with outliers
    np.random.seed(42)
    x = np.linspace(-5, 5, 100)
    y = x**2 + np.random.randn(100) * 0.1
    y[50] = 1000  # Outlier
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model is not None,
        "Handles data with outliers"
    )
    
    # Test 4: Negative values
    x = np.linspace(-10, 10, 100)
    y = -x**2 - 5
    result = overseer.classify(y, x)
    all_ok &= check(
        result.best_model == "polynomial",
        f"Handles negative values (got {result.best_model})"
    )
    
    return _finalize_test(all_ok)


def test_explanation():
    """Test the explain_classification output."""
    print("\n=== Testing Explanation Generation ===")
    from overseer import Overseer
    
    overseer = Overseer()
    all_ok = True
    
    x = np.linspace(-5, 5, 100)
    y = x**2 + 2*x + 1
    result = overseer.classify(y, x)
    
    explanation = overseer.explain_classification(result)
    
    all_ok &= check(
        "Data Type" in explanation,
        "Explanation includes data type"
    )
    
    all_ok &= check(
        "Best Model" in explanation,
        "Explanation includes best model"
    )
    
    all_ok &= check(
        "Reasoning" in explanation,
        "Explanation includes reasoning"
    )
    
    all_ok &= check(
        "Model Rankings" in explanation,
        "Explanation includes rankings"
    )
    
    return _finalize_test(all_ok)


def test_routing_with_inference(verbose=False):
    """Test actual routing with model inference (requires trained models)."""
    print("\n=== Testing Routing with Inference ===")
    
    try:
        from overseer import Overseer
        from galileo_tools import GalileoTools
        
        # Check if models are available
        tools = GalileoTools()
        available = tools.list_available_models()
        
        if not available:
            print(f"  {YELLOW}Skipping: No trained models available{RESET}")
            return _finalize_test(True)
        
        print(f"  Available models: {available}")
        
        overseer = Overseer()
        all_ok = True
        
        # Test 1: Route polynomial data
        if "polynomial" in available:
            x = np.linspace(-5, 5, 100)
            y = 2*x**2 + 3*x + 1
            
            result = overseer.route(y, x)
            
            if verbose:
                print(f"\n  Polynomial routing:")
                print(f"    Model: {result.model}")
                print(f"    Equation: {result.equation}")
                print(f"    Confidence: {result.confidence:.2f}")
            
            all_ok &= check(
                result.model == "polynomial",
                f"Routes quadratic to polynomial (got {result.model})"
            )
            
            all_ok &= check(
                "x" in result.equation.lower() or "X" in result.equation,
                f"Returns equation with x: {result.equation[:50]}"
            )
        
        # Test 2: Ensemble routing
        if len(available) >= 2:
            x = np.linspace(-5, 5, 100)
            y = x**2
            
            result = overseer.route_ensemble(y, x, top_k=2)
            
            if verbose:
                print(f"\n  Ensemble routing:")
                print(f"    Best: {result.model} ({result.confidence:.2f})")
                print(f"    Alternatives: {len(result.alternatives)}")
            
            all_ok &= check(
                len(result.alternatives) >= 1,
                "Ensemble returns alternatives"
            )
        
        return _finalize_test(all_ok)
        
    except ImportError as e:
        print(f"  {YELLOW}Skipping inference tests: {e}{RESET}")
        return _finalize_test(True)
    except Exception as e:
        print(f"  {RED}Error: {e}{RESET}")
        return _finalize_test(False)


def test_classification_accuracy():
    """Test overall classification accuracy on a variety of data."""
    print("\n=== Testing Classification Accuracy ===")
    from overseer import Overseer
    
    overseer = Overseer()
    
    test_cases = [
        # (name, generate_func, expected_models)
        ("Linear", lambda: (np.linspace(-5, 5, 100), 2*np.linspace(-5, 5, 100) + 1), ["polynomial"]),
        ("Quadratic", lambda: (np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)**2), ["polynomial"]),
        ("Cubic", lambda: (np.linspace(-5, 5, 100), 0.1*np.linspace(-5, 5, 100)**3), ["polynomial"]),
        ("Exp decay", lambda: (np.linspace(0, 10, 100), np.exp(-0.5*np.linspace(0, 10, 100))), ["transcendental", "linear_ode"]),
        ("Damped osc", lambda: (np.linspace(0, 10, 200), np.exp(-0.3*np.linspace(0, 10, 200))*np.cos(2*np.linspace(0, 10, 200))), ["linear_ode"]),
        ("Simple harmonic", lambda: (np.linspace(0, 10, 200), np.cos(2*np.linspace(0, 10, 200))), ["linear_ode"]),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for name, gen_func, expected in test_cases:
        x, y = gen_func()
        result = overseer.classify(y, x)
        
        is_correct = result.best_model in expected
        if is_correct:
            correct += 1
            status = f"{GREEN}✓{RESET}"
        else:
            status = f"{RED}✗{RESET}"
        
        print(f"  {status} {name:20s} → {result.best_model:15s} (expected: {expected})")
    
    accuracy = correct / total
    check(
        accuracy >= 0.8,
        f"Classification accuracy: {correct}/{total} ({accuracy:.0%})"
    )
    
    return _finalize_test(accuracy >= 0.7)


def main():
    parser = argparse.ArgumentParser(description="Test Overseer Router")
    parser.add_argument("--with-inference", action="store_true", 
                        help="Include inference tests (requires trained models)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Overseer Router Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("Data Type Detection", test_data_type_detection()))
    results.append(("Polynomial Classification", test_polynomial_classification()))
    results.append(("Transcendental Classification", test_transcendental_classification()))
    results.append(("ODE Classification", test_ode_classification()))
    results.append(("AR Classification", test_ar_classification()))
    results.append(("Physics Classification", test_physics_classification()))
    results.append(("Feature Extraction", test_feature_extraction()))
    results.append(("Model Ranking", test_ranking()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Explanation", test_explanation()))
    results.append(("Classification Accuracy", test_classification_accuracy()))
    
    if args.with_inference:
        results.append(("Routing with Inference", test_routing_with_inference(args.verbose)))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name:30s} {status}")
    
    print(f"\nOverall: {passed}/{total} test groups passed")
    
    if passed == total:
        print(f"\n{GREEN}✓ All tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some tests failed.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

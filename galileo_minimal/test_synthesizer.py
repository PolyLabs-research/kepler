#!/usr/bin/env python3
"""
test_synthesizer.py - Test suite for the Synthesizer module

Validates all synthesis methods and demonstrates the LLM verification loop.

Usage:
    python test_synthesizer.py
    python test_synthesizer.py --verbose
"""
import os
import sys
import numpy as np
from synthesizer import Synthesizer, SynthesizerError

# Color output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
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


def test_from_equation():
    """Test equation synthesis."""
    print("\n=== Testing from_equation ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Simple polynomial
    x, y = synth.from_equation("x**2", x_range=(-2, 2), n_points=5)
    expected_y = np.array([-2, -1, 0, 1, 2])**2
    all_ok &= check(
        np.allclose(y, expected_y, atol=0.01),
        "x² generates correct values",
        f"Got {y}, expected {expected_y}"
    )
    
    # Test 2: Exponential
    x, y = synth.from_equation("exp(-0.5*x)", x_range=(0, 4), n_points=5)
    all_ok &= check(
        y[0] > y[-1] and y[0] > 0.9 and y[-1] < 0.2,
        "exp(-0.5*x) shows decay from ~1 to ~0.14",
        f"y[0]={y[0]:.3f}, y[-1]={y[-1]:.3f}"
    )
    
    # Test 3: Trigonometric
    x, y = synth.from_equation("sin(x)", x_range=(0, 2*np.pi), n_points=100)
    all_ok &= check(
        abs(y[0]) < 0.1 and abs(y[-1]) < 0.1 and max(y) > 0.95,
        "sin(x) has correct period and amplitude",
        f"y[0]={y[0]:.3f}, max={max(y):.3f}, y[-1]={y[-1]:.3f}"
    )
    
    # Test 4: Logarithmic (auto-adjusts x_range)
    x, y = synth.from_equation("log(x)", x_range=(0, 10), n_points=50)
    all_ok &= check(
        x[0] > 0 and y[-1] > y[0],
        "log(x) auto-adjusts range to avoid log(0)",
        f"x_min={x[0]:.3f}, y grows from {y[0]:.3f} to {y[-1]:.3f}"
    )
    
    # Test 5: With noise
    np.random.seed(42)
    x1, y1 = synth.from_equation("x", n_points=100, noise_std=0.0)
    np.random.seed(42)
    x2, y2 = synth.from_equation("x", n_points=100, noise_std=0.5)
    all_ok &= check(
        not np.allclose(y1, y2) and np.std(y2 - y1) > 0.3,
        "Noise parameter adds Gaussian noise",
        f"std of noise = {np.std(y2 - y1):.3f}"
    )
    
    # Test 6: Custom variable
    x, y = synth.from_equation("t**2 + 1", variable="t", x_range=(0, 3), n_points=4)
    all_ok &= check(
        np.allclose(y, [1, 2, 5, 10], atol=0.1),
        "Custom variable 't' works correctly"
    )
    
    return _finalize_test(all_ok)


def test_from_description():
    """Test natural language parsing."""
    print("\n=== Testing from_description ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Exponential decay
    x, y = synth.from_description("exponential decay with rate 0.5", n_points=50)
    all_ok &= check(
        y[0] > y[-1] and y[0] > 0.9,
        "Exponential decay recognized",
        f"y decays from {y[0]:.3f} to {y[-1]:.3f}"
    )
    
    # Test 2: Exponential growth
    x, y = synth.from_description("exponential growth", n_points=50)
    all_ok &= check(
        y[-1] > y[0],
        "Exponential growth recognized",
        f"y grows from {y[0]:.3f} to {y[-1]:.3f}"
    )
    
    # Test 3: Sine wave
    x, y = synth.from_description("sine wave with frequency 2", n_points=200)
    # Count zero crossings to estimate frequency
    crossings = np.sum(np.diff(np.sign(y)) != 0)
    all_ok &= check(
        crossings > 6,  # Multiple periods - frequency 2 over 4*pi should give ~8 crossings
        "Sine wave with frequency recognized",
        f"Zero crossings: {crossings}"
    )
    
    # Test 4: Quadratic
    x, y = synth.from_description("quadratic", n_points=50)
    all_ok &= check(
        y[0] > 0 and y[-1] > 0 and min(y) < max(y),  # Parabola shape
        "Quadratic recognized",
        f"Shape: min={min(y):.2f}, max={max(y):.2f}"
    )
    
    # Test 5: Logarithmic
    x, y = synth.from_description("logarithmic", n_points=50)
    all_ok &= check(
        x[0] > 0 and y[-1] > y[0],
        "Logarithmic recognized",
        f"x starts at {x[0]:.2f}, y grows from {y[0]:.2f} to {y[-1]:.2f}"
    )
    
    # Test 6: Sigmoid
    x, y = synth.from_description("sigmoid", n_points=100)
    # Sigmoid 1/(1+exp(-x)) goes from ~0 to ~1, centered at 0.5 at x=0
    all_ok &= check(
        y[0] < 0.02 and y[-1] > 0.98,  # sigmoid over (-6, 6) range
        "Sigmoid recognized",
        f"y transitions from {y[0]:.3f} to {y[-1]:.3f}"
    )
    
    # Test 7: Damped oscillation
    x, y = synth.from_description("damped oscillation with damping 0.3 frequency 2", n_points=200)
    peaks = []
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            peaks.append(abs(y[i]))
    all_ok &= check(
        len(peaks) > 2 and peaks[0] > peaks[-1],
        "Damped oscillation shows decaying peaks",
        f"Peak amplitudes: {[f'{p:.3f}' for p in peaks[:4]]}"
    )
    
    # Test 8: Direct equation fallback
    x, y = synth.from_description("2*x + 3", n_points=5)
    all_ok &= check(
        np.allclose(y, 2*x + 3, atol=0.01),
        "Direct equation fallback works"
    )
    
    return _finalize_test(all_ok)


def test_from_ode():
    """Test ODE synthesis."""
    print("\n=== Testing from_ode ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Simple harmonic (x'' + 4x = 0)
    t, x = synth.from_ode("x'' + 4*x = 0", t_range=(0, 10), n_points=200)
    crossings = np.sum(np.diff(np.sign(x)) != 0)
    # omega=2, period=pi≈3.14, so in t=10 we expect ~3 full periods = ~6 crossings
    all_ok &= check(
        4 < crossings < 12,
        "Simple harmonic x''+4x=0 oscillates correctly",
        f"Zero crossings: {crossings} (expected ~6)"
    )
    
    # Test 2: Damped oscillation (x'' + 0.6x' + 4x = 0)
    t, x = synth.from_ode("x'' + 0.6*x' + 4*x = 0", t_range=(0, 10), n_points=200)
    # Find peaks
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > 0.05:
            peaks.append(abs(x[i]))
    all_ok &= check(
        len(peaks) >= 2 and peaks[0] > peaks[-1],
        "Damped ODE shows decaying amplitude",
        f"Peaks decay: {peaks[0]:.3f} → {peaks[-1]:.3f}"
    )
    
    # Test 3: First-order decay (x' = -0.5*x)
    t, x = synth.from_ode("x' = -0.5*x", t_range=(0, 10), y0=[2.0], n_points=100)
    expected_final = 2.0 * np.exp(-0.5 * 10)
    all_ok &= check(
        abs(x[-1] - expected_final) < 0.1,
        "First-order decay x'=-0.5x works",
        f"Final: {x[-1]:.4f}, expected: {expected_final:.4f}"
    )
    
    # Test 4: First-order growth (dx/dt = 0.3*x)
    t, x = synth.from_ode("dx/dt = 0.3*x", t_range=(0, 5), y0=[1.0], n_points=50)
    expected_final = np.exp(0.3 * 5)
    all_ok &= check(
        abs(x[-1] - expected_final) < 0.5,
        "First-order growth dx/dt=0.3x works",
        f"Final: {x[-1]:.3f}, expected: {expected_final:.3f}"
    )
    
    # Test 5: Undamped from y'' notation
    t, x = synth.from_ode("y'' + 9*y = 0", t_range=(0, 4), n_points=100)
    # omega=3, period=2pi/3 ≈ 2.09, so ~2 periods in t=4
    all_ok &= check(
        abs(x[0] - 1.0) < 0.01,  # y0=[1,0] default
        "y'' notation parsed correctly"
    )
    
    return _finalize_test(all_ok)


def test_from_ode_params():
    """Test direct ODE parameter synthesis."""
    print("\n=== Testing from_ode_params ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Underdamped (zeta < 1)
    t, x = synth.from_ode_params(order=2, C1=0.6, C2=4.0, n_points=200)
    info = synth.get_ode_info(0.6, 4.0)
    all_ok &= check(
        info["damping_type"] == "underdamped",
        f"C1=0.6, C2=4.0 is underdamped (zeta={info['zeta']:.3f})"
    )
    
    # Test 2: Overdamped (zeta > 1)
    t, x = synth.from_ode_params(order=2, C1=6.0, C2=4.0, n_points=200)
    info = synth.get_ode_info(6.0, 4.0)
    all_ok &= check(
        info["damping_type"] == "overdamped",
        f"C1=6.0, C2=4.0 is overdamped (zeta={info['zeta']:.3f})"
    )
    # Overdamped should decay monotonically
    crossings = np.sum(np.diff(np.sign(x - x[-1])) != 0)
    all_ok &= check(
        crossings < 3,  # Minimal oscillation
        "Overdamped decays monotonically",
        f"Zero crossings: {crossings}"
    )
    
    # Test 3: Undamped (C1=0)
    t, x = synth.from_ode_params(order=2, C1=0.0, C2=4.0, n_points=200)
    info = synth.get_ode_info(0.0, 4.0)
    all_ok &= check(
        info["damping_type"] == "undamped",
        "C1=0 is undamped"
    )
    # Check amplitude is preserved
    all_ok &= check(
        abs(max(x) - 1.0) < 0.1 and abs(min(x) + 1.0) < 0.1,
        "Undamped preserves amplitude",
        f"Amplitude range: [{min(x):.3f}, {max(x):.3f}]"
    )
    
    # Test 4: First-order
    t, x = synth.from_ode_params(order=1, a=-0.5, y0=[2.0], n_points=100)
    all_ok &= check(
        x[0] > 1.9 and x[-1] < 0.1,
        "First-order decay from params works",
        f"x[0]={x[0]:.3f}, x[-1]={x[-1]:.3f}"
    )
    
    return _finalize_test(all_ok)


def test_from_ar():
    """Test AR timeseries synthesis."""
    print("\n=== Testing from_ar ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: AR(1) with positive phi
    np.random.seed(42)
    y = synth.from_ar([0.7], n_steps=1000, noise_std=0.1)
    # Compute empirical autocorrelation at lag 1
    y_centered = y - np.mean(y)
    empirical_phi = np.mean(y_centered[1:] * y_centered[:-1]) / np.var(y)
    all_ok &= check(
        abs(empirical_phi - 0.7) < 0.15,
        f"AR(1) phi=0.7 has empirical autocorr ≈ 0.7",
        f"Empirical: {empirical_phi:.3f}"
    )
    
    # Test 2: AR(2)
    np.random.seed(42)
    y = synth.from_ar([0.5, 0.3], n_steps=500, noise_std=0.1)
    all_ok &= check(
        len(y) == 500 and np.isfinite(y).all(),
        "AR(2) generates stable timeseries"
    )
    
    # Test 3: Near-unit-root (high persistence)
    np.random.seed(42)
    y = synth.from_ar([0.95], n_steps=200, noise_std=0.05)
    all_ok &= check(
        np.std(y) > 0.1,  # High persistence leads to more variance than the noise alone
        "Near-unit-root AR(1) is highly persistent",
        f"std(y)={np.std(y):.3f}"
    )
    
    return _finalize_test(all_ok)


def test_from_physics():
    """Test physics law synthesis."""
    print("\n=== Testing from_physics ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Newton's gravity
    data = synth.from_physics(
        "newton_gravity",
        M1=5.97e24,  # Earth mass
        M2=1000,     # 1 ton object
        R=np.linspace(6.4e6, 4e8, 100)  # LEO to Moon
    )
    all_ok &= check(
        "output" in data and len(data["output"]) == 100,
        "Newton gravity generates output array"
    )
    all_ok &= check(
        data["output"][0] > data["output"][-1],  # Force decreases with R
        "Gravity decreases with distance",
        f"F at LEO: {data['output'][0]:.2e}, at Moon: {data['output'][-1]:.2e}"
    )
    
    # Test 2: Ideal gas law
    data = synth.from_physics(
        "ideal_gas",
        n=1.0,  # 1 mole
        T=np.linspace(200, 400, 50),  # Temperature range
        V=0.01  # 10 liters
    )
    all_ok &= check(
        data["output"][-1] > data["output"][0],  # P increases with T
        "Ideal gas: pressure increases with temperature",
        f"P at 200K: {data['output'][0]:.2f}, at 400K: {data['output'][-1]:.2f}"
    )
    
    # Test 3: Kinetic energy
    data = synth.from_physics(
        "kinetic_energy",
        m=10,  # 10 kg
        v=np.linspace(0, 10, 20)  # 0 to 10 m/s
    )
    expected = 0.5 * 10 * np.linspace(0, 10, 20)**2
    all_ok &= check(
        np.allclose(data["output"], expected, rtol=0.01),
        "Kinetic energy E=½mv² computed correctly"
    )
    
    # Test 4: Hooke's law
    data = synth.from_physics(
        "hookes_law",
        k=100,  # Spring constant
        x=np.linspace(-0.1, 0.1, 50)  # Displacement
    )
    all_ok &= check(
        data["output"][0] < 0 and data["output"][-1] > 0,
        "Hooke's law: F changes sign with displacement"
    )
    
    # Test 5: With noise
    np.random.seed(42)
    data = synth.from_physics("kinetic_energy", m=10, v=5, n_samples=100, noise_std=0.05)
    all_ok &= check(
        np.std(data["output"]) > 1,  # Noise present
        "Physics laws support noise parameter"
    )
    
    return _finalize_test(all_ok)


def test_from_polynomial():
    """Test polynomial synthesis from coefficients."""
    print("\n=== Testing from_polynomial ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Linear
    x, y = synth.from_polynomial([1, 2], x_range=(0, 5), n_points=6)
    expected = 1 + 2*x
    all_ok &= check(
        np.allclose(y, expected),
        "Linear [1, 2] → 1 + 2x works",
        f"Got {y}, expected {expected}"
    )
    
    # Test 2: Quadratic
    x, y = synth.from_polynomial([0, 0, 1], x_range=(-3, 3), n_points=7)
    expected = x**2
    all_ok &= check(
        np.allclose(y, expected),
        "Quadratic [0, 0, 1] → x² works"
    )
    
    # Test 3: Full cubic
    x, y = synth.from_polynomial([1, -1, 2, -0.5], x_range=(-2, 2), n_points=50)
    expected = 1 - x + 2*x**2 - 0.5*x**3
    all_ok &= check(
        np.allclose(y, expected, atol=0.01),
        "Cubic [1, -1, 2, -0.5] → 1-x+2x²-0.5x³ works"
    )
    
    return _finalize_test(all_ok)


def test_get_ode_info():
    """Test ODE analysis utility."""
    print("\n=== Testing get_ode_info ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Underdamped
    info = synth.get_ode_info(C1=0.6, C2=4.0)
    all_ok &= check(
        info["damping_type"] == "underdamped" and info["zeta"] < 1,
        f"Correctly identifies underdamped (zeta={info['zeta']:.3f})"
    )
    all_ok &= check(
        abs(info["omega"] - 2.0) < 0.01,
        f"Correctly computes omega=2 from C2=4"
    )
    
    # Test 2: Overdamped
    info = synth.get_ode_info(C1=6.0, C2=4.0)
    all_ok &= check(
        info["damping_type"] == "overdamped" and info["zeta"] > 1,
        f"Correctly identifies overdamped (zeta={info['zeta']:.3f})"
    )
    all_ok &= check(
        isinstance(info["roots"][0], float),  # Real roots for overdamped
        "Returns real roots for overdamped"
    )
    
    # Test 3: Critically damped
    info = synth.get_ode_info(C1=4.0, C2=4.0)  # zeta = 1
    all_ok &= check(
        info["damping_type"] == "critically_damped",
        "Correctly identifies critically damped"
    )
    
    # Test 4: Undamped
    info = synth.get_ode_info(C1=0.0, C2=9.0)
    all_ok &= check(
        info["damping_type"] == "undamped" and info["zeta"] == 0,
        "Correctly identifies undamped"
    )
    all_ok &= check(
        abs(info["omega"] - 3.0) < 0.01 and info["damped_frequency"] == info["omega"],
        "Undamped: omega_d = omega"
    )
    
    return _finalize_test(all_ok)


def test_error_handling():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")
    synth = Synthesizer()
    all_ok = True
    
    # Test 1: Invalid equation
    try:
        synth.from_equation("invalid_func(x)")
        all_ok &= check(False, "Should raise error for invalid equation")
    except SynthesizerError:
        all_ok &= check(True, "Raises SynthesizerError for invalid equation")
    except:
        all_ok &= check(False, "Should raise SynthesizerError specifically")
    
    # Test 2: Unparseable description
    try:
        synth.from_description("some random gibberish that doesn't match")
        all_ok &= check(False, "Should raise error for unparseable description")
    except SynthesizerError:
        all_ok &= check(True, "Raises SynthesizerError for unparseable description")
    except:
        all_ok &= check(False, "Should raise SynthesizerError specifically")
    
    # Test 3: Invalid ODE format
    try:
        synth.from_ode("this is not an ODE")
        all_ok &= check(False, "Should raise error for invalid ODE")
    except SynthesizerError:
        all_ok &= check(True, "Raises SynthesizerError for invalid ODE")
    
    # Test 4: Unknown physics law
    try:
        synth.from_physics("made_up_law", x=1)
        all_ok &= check(False, "Should raise error for unknown physics law")
    except SynthesizerError:
        all_ok &= check(True, "Raises SynthesizerError for unknown physics law")
    
    # Test 5: Missing physics variables
    try:
        synth.from_physics("newton_gravity", M1=1e24)  # Missing M2, R
        all_ok &= check(False, "Should raise error for missing variables")
    except SynthesizerError:
        all_ok &= check(True, "Raises SynthesizerError for missing variables")
    
    # Test 6: Unstable AR
    try:
        synth.from_ar([0.8, 0.5])  # Sum > 1, unstable
        all_ok &= check(False, "Should raise error for unstable AR")
    except SynthesizerError:
        all_ok &= check(True, "Raises SynthesizerError for unstable AR")
    
    return _finalize_test(all_ok)


def test_verification_loop():
    """
    Test the LLM verification loop concept:
    Generate data from known equation, then verify it matches expectations.
    """
    print("\n=== Testing Verification Loop Concept ===")
    synth = Synthesizer()
    all_ok = True
    
    # Scenario: LLM hypothesizes "y'' + 4y = 0 has solution cos(2t)"
    print("  Hypothesis: y'' + 4y = 0 has solution y = cos(2t)")
    
    # Step 1: Synthesize from hypothesis
    t, y_hypothesis = synth.from_equation("cos(2*t)", x_range=(0, 10), n_points=200, variable="t")
    
    # Step 2: Synthesize from ODE directly  
    t, y_ode = synth.from_ode("y'' + 4*y = 0", t_range=(0, 10), y0=[1.0, 0.0], n_points=200)
    
    # Step 3: Compare
    max_diff = np.max(np.abs(y_hypothesis - y_ode))
    all_ok &= check(
        max_diff < 0.01,
        f"Verification: cos(2t) matches y''+4y=0 solution",
        f"Max difference: {max_diff:.6f}"
    )
    
    # Scenario 2: Verify first-order solution
    print("  Hypothesis: x' = -0.5x has solution x = e^(-0.5t)")
    
    t, y_hyp = synth.from_equation("exp(-0.5*t)", x_range=(0, 10), n_points=100, variable="t")
    t, y_ode = synth.from_ode("x' = -0.5*x", t_range=(0, 10), y0=[1.0], n_points=100)
    
    max_diff = np.max(np.abs(y_hyp - y_ode))
    all_ok &= check(
        max_diff < 0.01,
        f"Verification: e^(-0.5t) matches x'=-0.5x solution",
        f"Max difference: {max_diff:.6f}"
    )
    
    return _finalize_test(all_ok)


def main():
    """Run all tests."""
    print("=" * 70)
    print("Synthesizer Module Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("from_equation", test_from_equation()))
    results.append(("from_description", test_from_description()))
    results.append(("from_ode", test_from_ode()))
    results.append(("from_ode_params", test_from_ode_params()))
    results.append(("from_ar", test_from_ar()))
    results.append(("from_physics", test_from_physics()))
    results.append(("from_polynomial", test_from_polynomial()))
    results.append(("get_ode_info", test_get_ode_info()))
    results.append(("error_handling", test_error_handling()))
    results.append(("verification_loop", test_verification_loop()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name:25s} {status}")
    
    print(f"\nOverall: {passed}/{total} test groups passed")
    
    if passed == total:
        print(f"\n{GREEN}✓ All tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Some tests failed.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

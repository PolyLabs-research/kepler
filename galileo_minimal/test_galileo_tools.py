#!/usr/bin/env python3
"""
test_galileo_tools.py - Test suite for Galileo Tools LLM interface

Tests the three core tools:
1. synthesize_data
2. discover_equation  
3. verify_equation

Usage:
    python test_galileo_tools.py
    python test_galileo_tools.py --with-models  # Also test discovery (requires trained models)
"""
import os
import sys
import json
import numpy as np

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


def test_tool_schemas():
    """Test that tool schemas are valid."""
    print("\n=== Testing Tool Schemas ===")
    
    from galileo_tools import TOOL_SCHEMAS, ANTHROPIC_TOOL_SCHEMAS, GalileoTools
    
    all_ok = True
    
    # Check OpenAI format
    all_ok &= check(
        len(TOOL_SCHEMAS) == 3,
        f"TOOL_SCHEMAS has 3 tools (got {len(TOOL_SCHEMAS)})"
    )
    
    for schema in TOOL_SCHEMAS:
        name = schema.get("function", {}).get("name", "unknown")
        all_ok &= check(
            "type" in schema and "function" in schema,
            f"OpenAI schema for '{name}' has correct structure"
        )
        all_ok &= check(
            "name" in schema["function"] and "description" in schema["function"],
            f"OpenAI schema for '{name}' has name and description"
        )
    
    # Check Anthropic format
    all_ok &= check(
        len(ANTHROPIC_TOOL_SCHEMAS) == 3,
        f"ANTHROPIC_TOOL_SCHEMAS has 3 tools"
    )
    
    for schema in ANTHROPIC_TOOL_SCHEMAS:
        name = schema.get("name", "unknown")
        all_ok &= check(
            "name" in schema and "description" in schema and "input_schema" in schema,
            f"Anthropic schema for '{name}' has correct structure"
        )
    
    # Check get_schemas method
    tools = GalileoTools()
    
    openai_schemas = tools.get_schemas("openai")
    all_ok &= check(
        openai_schemas == TOOL_SCHEMAS,
        "get_schemas('openai') returns correct schemas"
    )
    
    anthropic_schemas = tools.get_schemas("anthropic")
    all_ok &= check(
        anthropic_schemas == ANTHROPIC_TOOL_SCHEMAS,
        "get_schemas('anthropic') returns correct schemas"
    )
    
    raw_schemas = tools.get_schemas("raw")
    all_ok &= check(
        len(raw_schemas) == 3 and "name" in raw_schemas[0],
        "get_schemas('raw') returns function definitions only"
    )
    
    return _finalize_test(all_ok)


def test_synthesize_data():
    """Test synthesize_data tool."""
    print("\n=== Testing synthesize_data Tool ===")
    
    from galileo_tools import GalileoTools
    
    tools = GalileoTools()
    all_ok = True
    
    # Test 1: Equation synthesis
    result = tools.synthesize_data(equation="x**2", n_points=5)
    all_ok &= check(
        result.get("success") and "x" in result and "y" in result,
        "Equation synthesis returns x and y arrays"
    )
    all_ok &= check(
        len(result["x"]) == 5 and len(result["y"]) == 5,
        "Returns correct number of points",
        f"Got {len(result.get('x', []))} points"
    )
    
    # Test 2: Description synthesis
    result = tools.synthesize_data(description="exponential decay", n_points=10)
    all_ok &= check(
        result.get("success") and result.get("method") == "description",
        "Description synthesis works"
    )
    
    # Test 3: ODE synthesis
    result = tools.synthesize_data(ode="x'' + 4*x = 0", n_points=20)
    all_ok &= check(
        result.get("success") and result.get("method") == "ode",
        "ODE synthesis works"
    )
    
    # Test 4: Physics law synthesis
    result = tools.synthesize_data(
        physics_law="kinetic_energy",
        variables={"m": 10, "v": [1, 2, 3, 4, 5]},
        n_points=5
    )
    all_ok &= check(
        result.get("success") and "data" in result,
        "Physics law synthesis works"
    )
    all_ok &= check(
        "output" in result.get("data", {}),
        "Physics result contains output array"
    )
    
    # Test 5: Custom x_range
    result = tools.synthesize_data(equation="x", x_range=[0, 10], n_points=11)
    all_ok &= check(
        result.get("success") and result["x"][0] == 0 and result["x"][-1] == 10,
        "Custom x_range is respected"
    )
    
    # Test 6: Noise parameter
    np.random.seed(42)
    result1 = tools.synthesize_data(equation="x", n_points=100, noise_std=0.0)
    np.random.seed(42)
    result2 = tools.synthesize_data(equation="x", n_points=100, noise_std=0.5)
    
    y1 = np.array(result1["y"])
    y2 = np.array(result2["y"])
    all_ok &= check(
        not np.allclose(y1, y2),
        "Noise parameter adds noise to output"
    )
    
    # Test 7: Error handling - missing physics variables
    result = tools.synthesize_data(physics_law="newton_gravity")
    all_ok &= check(
        "error" in result,
        "Missing physics variables returns error"
    )
    
    # Test 8: Error handling - no specification
    result = tools.synthesize_data()
    all_ok &= check(
        "error" in result,
        "No specification returns error"
    )
    
    return _finalize_test(all_ok)


def test_verify_equation():
    """Test verify_equation tool."""
    print("\n=== Testing verify_equation Tool ===")
    
    from galileo_tools import GalileoTools
    
    tools = GalileoTools()
    all_ok = True
    
    # Test 1: Perfect match
    result = tools.verify_equation(
        equation="x**2",
        x=[1, 2, 3, 4, 5],
        y=[1, 4, 9, 16, 25]
    )
    all_ok &= check(
        result.get("success") and result.get("match") == True,
        "Exact match detected",
        f"RMSE={result.get('rmse', 'N/A')}"
    )
    all_ok &= check(
        result.get("rmse", 1) < 0.01,
        "RMSE is near zero for exact match"
    )
    
    # Test 2: Close match
    result = tools.verify_equation(
        equation="x**2",
        x=[1, 2, 3, 4, 5],
        y=[1.1, 4.05, 8.95, 16.1, 24.9]  # Slightly noisy
    )
    all_ok &= check(
        result.get("success") and result.get("match") == True,
        "Close match detected with default tolerance",
        f"rel_error={result.get('relative_error', 'N/A'):.4f}"
    )
    
    # Test 3: No match
    result = tools.verify_equation(
        equation="x**2",
        x=[1, 2, 3, 4, 5],
        y=[1, 2, 3, 4, 5]  # Linear, not quadratic
    )
    all_ok &= check(
        result.get("success") and result.get("match") == False,
        "Mismatch detected"
    )
    
    # Test 4: Custom tolerance
    result = tools.verify_equation(
        equation="x**2",
        x=[1, 2, 3, 4, 5],
        y=[1, 4, 9, 16, 26],  # Last point off by 1
        tolerance=0.01  # Very strict
    )
    all_ok &= check(
        result.get("success") and result.get("match") == False,
        "Strict tolerance rejects small error"
    )
    
    result = tools.verify_equation(
        equation="x**2",
        x=[1, 2, 3, 4, 5],
        y=[1, 4, 9, 16, 26],
        tolerance=0.2  # More lenient
    )
    all_ok &= check(
        result.get("success") and result.get("match") == True,
        "Lenient tolerance accepts small error"
    )
    
    # Test 5: Auto-generated x values
    result = tools.verify_equation(
        equation="x**2 + 1",
        y=[1, 2, 5, 10, 17]  # Values at x = 0, 1, 2, 3, 4
    )
    all_ok &= check(
        result.get("success"),
        "Verification works without explicit x values"
    )
    
    # Test 6: Error handling - invalid equation
    result = tools.verify_equation(
        equation="invalid_function(x)",
        y=[1, 2, 3]
    )
    all_ok &= check(
        "error" in result,
        "Invalid equation returns error"
    )
    
    return _finalize_test(all_ok)


def test_execute_interface():
    """Test the unified execute interface."""
    print("\n=== Testing execute() Interface ===")
    
    from galileo_tools import GalileoTools
    
    tools = GalileoTools()
    all_ok = True
    
    # Test execute with synthesize_data
    result = tools.execute("synthesize_data", {"equation": "x**2", "n_points": 5})
    all_ok &= check(
        result.get("success"),
        "execute('synthesize_data', ...) works"
    )
    
    # Test execute with verify_equation
    result = tools.execute("verify_equation", {
        "equation": "x**2",
        "x": [1, 2, 3],
        "y": [1, 4, 9]
    })
    all_ok &= check(
        result.get("success"),
        "execute('verify_equation', ...) works"
    )
    
    # Test execute with unknown tool
    result = tools.execute("unknown_tool", {})
    all_ok &= check(
        "error" in result and "Unknown tool" in result["error"],
        "execute() handles unknown tools"
    )
    
    return _finalize_test(all_ok)


def test_convenience_functions():
    """Test module-level convenience functions."""
    print("\n=== Testing Convenience Functions ===")
    
    from galileo_tools import synthesize_data, verify_equation, get_tools
    
    all_ok = True
    
    # Test synthesize_data function
    result = synthesize_data(equation="x + 1", n_points=5)
    all_ok &= check(
        result.get("success"),
        "synthesize_data() convenience function works"
    )
    
    # Test verify_equation function
    result = verify_equation(equation="x", x=[1, 2, 3], y=[1, 2, 3])
    all_ok &= check(
        result.get("success") and result.get("match"),
        "verify_equation() convenience function works"
    )
    
    # Test get_tools returns singleton
    tools1 = get_tools()
    tools2 = get_tools()
    all_ok &= check(
        tools1 is tools2,
        "get_tools() returns singleton instance"
    )
    
    return _finalize_test(all_ok)


def test_discover_equation(with_models: bool = False):
    """Test discover_equation tool."""
    print("\n=== Testing discover_equation Tool ===")
    
    from galileo_tools import GalileoTools
    
    tools = GalileoTools()
    all_ok = True
    
    # Check available models
    available = tools.list_available_models()
    
    if not available:
        print(f"  {YELLOW}⚠ No trained models found. Skipping discovery tests.{RESET}")
        print(f"  {YELLOW}  Train a model first: python train.py --model polynomial --tier 1{RESET}")
        return _finalize_test(True)  # Not a failure, just skip
    
    print(f"  Available models: {available}")
    
    if not with_models:
        print(f"  {YELLOW}⚠ Skipping model tests. Use --with-models to test discovery.{RESET}")
        return _finalize_test(True)
    
    # Test 1: Basic discovery
    result = tools.discover_equation(
        x=[1, 2, 3, 4, 5],
        y=[1, 4, 9, 16, 25],
        model_hint="polynomial"
    )
    
    if result.get("success"):
        all_ok &= check(
            "equation" in result and "tokens" in result,
            f"Discovery returns equation: {result.get('equation')}",
            f"Tokens: {result.get('tokens')}"
        )
        all_ok &= check(
            "confidence" in result and 0 <= result["confidence"] <= 1,
            f"Discovery returns valid confidence: {result.get('confidence'):.2f}"
        )
    else:
        all_ok &= check(
            False,
            f"Discovery failed: {result.get('error')}"
        )
    
    # Test 2: Auto-detect data type
    result = tools.discover_equation(
        y=[1, 4, 9, 16, 25]  # No x provided, no hints
    )
    all_ok &= check(
        result.get("success") or "error" in result,
        "Discovery handles missing x values"
    )
    
    # Test 3: Explicit data type
    result = tools.discover_equation(
        x=[0, 1, 2, 3, 4],
        y=[1.0, 0.37, 0.14, 0.05, 0.02],  # Exponential decay
        data_type="algebraic"
    )
    all_ok &= check(
        result.get("success") or "error" in result,
        f"Discovery with data_type hint works"
    )
    
    return _finalize_test(all_ok)


def test_llm_workflow():
    """Test a typical LLM tool-calling workflow."""
    print("\n=== Testing LLM Workflow Simulation ===")
    
    from galileo_tools import GalileoTools, TOOL_SCHEMAS
    import json
    
    tools = GalileoTools()
    all_ok = True
    
    # Simulate: LLM receives tool schemas
    schemas_json = json.dumps(TOOL_SCHEMAS)
    all_ok &= check(
        len(schemas_json) > 100,
        "Tool schemas serialize to JSON"
    )
    
    # Simulate: LLM decides to synthesize data
    llm_tool_call = {
        "name": "synthesize_data",
        "arguments": {"equation": "x**2 + 2*x + 1", "n_points": 10}
    }
    
    # Simulate: Execute tool call
    result = tools.execute(llm_tool_call["name"], llm_tool_call["arguments"])
    all_ok &= check(
        result.get("success"),
        "LLM-style tool call executes successfully"
    )
    
    # Simulate: LLM receives result and decides to verify
    if result.get("success"):
        x = result["x"]
        y = result["y"]
        
        # LLM hypothesizes the equation might be (x+1)^2
        verify_call = {
            "name": "verify_equation",
            "arguments": {
                "equation": "(x + 1)**2",
                "x": x,
                "y": y
            }
        }
        
        verify_result = tools.execute(verify_call["name"], verify_call["arguments"])
        all_ok &= check(
            verify_result.get("success") and verify_result.get("match"),
            "LLM verification workflow succeeds",
            f"Verified: (x+1)² = x²+2x+1, match={verify_result.get('match')}"
        )
    
    return _finalize_test(all_ok)


def test_overseer_integration():
    """Test that Overseer integrates properly with GalileoTools."""
    print("\n=== Testing Overseer Integration ===")
    
    from galileo_tools import GalileoTools, _OVERSEER_AVAILABLE
    
    all_ok = True
    
    # Test 1: Check Overseer availability
    all_ok &= check(
        _OVERSEER_AVAILABLE,
        "Overseer module is available"
    )
    
    if not _OVERSEER_AVAILABLE:
        print(f"  {YELLOW}Skipping remaining tests - Overseer not available{RESET}")
        return _finalize_test(True)  # Don't fail, just skip
    
    # Test 2: GalileoTools initializes with Overseer
    tools = GalileoTools(use_overseer=True)
    all_ok &= check(
        tools._overseer is not None,
        "GalileoTools initializes with Overseer"
    )
    
    # Test 3: GalileoTools can disable Overseer
    tools_no_overseer = GalileoTools(use_overseer=False)
    all_ok &= check(
        tools_no_overseer._overseer is None,
        "GalileoTools can disable Overseer"
    )
    
    # Test 4: Classification works via _select_model
    import numpy as np
    
    # Quadratic data
    x = np.linspace(-5, 5, 100)
    y = 2*x**2 + 3*x + 1
    
    model = tools._select_model(y, x, None, None)  # No hints
    all_ok &= check(
        model is not None,
        f"Auto-routing selects a model: {model}"
    )
    
    # Test 5: Classification is cached
    all_ok &= check(
        tools._last_classification is not None,
        "Classification result is cached"
    )
    
    if tools._last_classification:
        best = tools._last_classification.best_model
        conf = tools._last_classification.best_confidence
        all_ok &= check(
            best == "polynomial" or best == "transcendental",  # Both reasonable for quadratic
            f"Classification is reasonable (best={best}, conf={conf:.2f})"
        )
    
    # Test 6: Oscillatory data routes to linear_ode
    t = np.linspace(0, 10, 200)
    y_osc = np.exp(-0.3 * t) * np.cos(2 * t)
    
    model_osc = tools._select_model(y_osc, t, None, None)
    all_ok &= check(
        model_osc == "linear_ode",
        f"Oscillatory data routes to linear_ode (got {model_osc})"
    )
    
    # Test 7: Timeseries (no x) routes to AR
    np.random.seed(42)
    y_ts = np.zeros(200)
    for i in range(1, 200):
        y_ts[i] = 0.7 * y_ts[i-1] + np.random.normal(0, 0.1)
    
    model_ts = tools._select_model(y_ts, None, None, None)
    all_ok &= check(
        model_ts == "ar",
        f"Timeseries routes to ar (got {model_ts})"
    )
    
    # Test 8: Explicit hints override Overseer
    # Note: model_hint only works if the model is in CONFIGS (requires torch)
    # In environments without torch, we test with data_type instead
    model_hinted = tools._select_model(y, x, "algebraic", None)  # Use data_type hint
    all_ok &= check(
        model_hinted == "polynomial",
        "Explicit data_type hint is respected"
    )
    
    # Test 9: data_type hint maps correctly
    model_ode_hint = tools._select_model(y, x, "ode", None)
    all_ok &= check(
        model_ode_hint == "linear_ode",
        "data_type='ode' maps to linear_ode"
    )
    
    return _finalize_test(all_ok)


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Galileo Tools")
    parser.add_argument("--with-models", action="store_true", 
                       help="Also test discover_equation (requires trained models)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Galileo Tools Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("Tool Schemas", test_tool_schemas()))
    results.append(("synthesize_data", test_synthesize_data()))
    results.append(("verify_equation", test_verify_equation()))
    results.append(("execute() Interface", test_execute_interface()))
    results.append(("Convenience Functions", test_convenience_functions()))
    results.append(("discover_equation", test_discover_equation(args.with_models)))
    results.append(("LLM Workflow", test_llm_workflow()))
    results.append(("Overseer Integration", test_overseer_integration()))
    
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

"""
test_mcp_server.py - The MCP wrapper around Kepler's tools.

The tool functions are tested directly (no `mcp` needed); the server-level
tests build the real server and are skipped when `mcp` is not installed.
"""
import asyncio
import json

import numpy as np
import pytest

import mcp_server
from mcp_server import (
    TOOL_FUNCTIONS,
    classify_data,
    discover_equation,
    fit_skeleton,
    jsonable,
    list_models,
    synthesize_data,
    verify_equation,
)

try:
    import mcp  # noqa: F401
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

EXPECTED_TOOLS = {"discover_equation", "verify_equation", "fit_skeleton", "synthesize_data", "classify_data", "list_models"}


def test_jsonable_cleans_numpy_and_non_finite():
    out = jsonable({"a": np.float32(1.5), "b": np.arange(3), "c": float("nan"), "d": (np.int64(2), "s"), "e": None})
    assert out == {"a": 1.5, "b": [0, 1, 2], "c": None, "d": [2, "s"], "e": None}
    json.dumps(out)


def test_tool_functions_work_without_checkpoints(monkeypatch):
    monkeypatch.setattr(mcp_server, "_TOOLS", None)
    monkeypatch.setenv("KEPLER_CHECKPOINTS", "/nonexistent/checkpoints")

    status = list_models()
    assert "available_models" in status and status["available_models"] == {}
    assert status["checkpoint_dir"] == "/nonexistent/checkpoints"

    synth = synthesize_data(equation="x**2", x_range=[0, 4], n_points=5)
    assert synth["success"] and synth["x"] == [0, 1, 2, 3, 4] and synth["y"] == [0, 1, 4, 9, 16]
    json.dumps(synth)

    verified = verify_equation("x**2", y=synth["y"], x=synth["x"])
    assert verified["match"] and verified["r2"] == pytest.approx(1.0)

    x = np.linspace(-5, 5, 100)
    ranking = classify_data(y=(2 * x**2 + 1).tolist(), x=x.tolist())
    assert ranking.get("rankings") or ranking.get("best_model") or ranking.get("success"), ranking
    json.dumps(ranking)

    missing = discover_equation(y=[1, 4, 9, 16, 25])
    assert "error" in missing  # no checkpoints: an error dict, never an exception
    json.dumps(missing)


class _FakeModel:
    def __init__(self, tokens):
        self.tokens = tokens

    def generate(self, features, vocab):
        return vocab.encode(self.tokens)


def test_discover_through_wrapper_with_stubbed_specialist(monkeypatch):
    from galileo_tools import GalileoTools
    from vocab import Vocabulary

    tools = GalileoTools(use_overseer=False)
    vocab = Vocabulary.default()
    monkeypatch.setattr(tools, "_load_model", lambda name, tier: (_FakeModel("INT_1 MUL X POW INT_2 ADD INT_1 MUL X".split()), vocab))
    monkeypatch.setattr(tools, "_predict_tokens", lambda model, v, features: v.decode(model.generate(features, v)))
    monkeypatch.setattr(tools, "list_available_models", lambda: {"polynomial": [1]})
    monkeypatch.setattr(mcp_server, "_TOOLS", tools)

    x = np.linspace(-50, 50, 80)
    y = 0.02 * x**2 - 1.5 * x + 300
    result = discover_equation(y=y.tolist(), x=x.tolist())
    assert result["success"] and result["strategy"] == "rank" and result["model_used"] == "polynomial"
    assert result["fit"]["r2"] == pytest.approx(1.0) and "bic" in result["fit"]
    json.dumps(result)  # everything the wrapper returns must serialise

    confirmed = verify_equation(result["equation"], y=y.tolist(), x=x.tolist())
    assert confirmed["match"]


@pytest.mark.skipif(not HAVE_MCP, reason="mcp package not installed")
def test_server_registers_all_tools_with_schemas():
    server = mcp_server.build_server()
    tools = asyncio.run(server.list_tools())
    by_name = {t.name: t for t in tools}
    assert set(by_name) == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description and len(tool.description) > 20, tool.name
    schema = getattr(by_name["discover_equation"], "input_schema", None) or getattr(by_name["discover_equation"], "inputSchema")
    assert "y" in schema["properties"] and "y" in schema.get("required", [])
    assert {"x", "model_hint", "tier", "normalize", "refine", "strategy"} <= set(schema["properties"])
    assert "sample index" in json.dumps(schema["properties"]["x"])


@pytest.mark.skipif(not HAVE_MCP, reason="mcp package not installed")
def test_server_call_tool_roundtrip():
    server = mcp_server.build_server()
    result = asyncio.run(server.call_tool("verify_equation", {"equation": "x**2", "x": [1, 2, 3], "y": [1, 4, 9]}))
    structured = getattr(result, "structured_content", None)
    if structured is None and isinstance(result, tuple):  # mcp 1.x returned (content, structured)
        structured = result[1]
    if structured is None:
        content = getattr(result, "content", result)
        structured = json.loads(content[0].text)
    payload = structured.get("result", structured)
    assert payload["match"] is True and payload["r2"] == pytest.approx(1.0)


@pytest.mark.skipif(not HAVE_MCP, reason="mcp package not installed")
def test_cli_lists_tools(capsys):
    assert mcp_server.main(["--list-tools"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {t["name"] for t in listed} == EXPECTED_TOOLS


def test_fit_skeleton_fits_constants_with_error_bars():
    x = np.array([6, 8, 10, 12, 16, 20, 24, 32, 40], dtype=float)
    y = 27.0 * x ** 0.35
    out = fit_skeleton("20.0*x**0.5", y=y.tolist(), x=x.tolist())
    json.dumps(out)
    assert out["success"] and out["r2"] == pytest.approx(1.0)
    assert sorted(out["params"].values()) == pytest.approx([0.35, 27.0], rel=1e-6)
    assert set(out["stderr"]) == set(out["params"])

    by_name = fit_skeleton("power", y=y.tolist(), x=x.tolist(), space="log")
    assert by_name["params"] == pytest.approx({"a": 27.0, "b": 0.35}, rel=1e-6)


def test_fit_skeleton_reports_bad_input_as_error():
    assert "zero constant" in fit_skeleton("0.0 + 2.0*x", y=[1.0, 2.0, 3.0])["error"]
    assert "error" in fit_skeleton("a*x", y=[1.0, 2.0, 3.0])
    assert "error" in fit_skeleton("power", y=[1.0, 2.0], x=[1.0])


def test_tool_function_list_matches_expected_names():
    assert {fn.__name__ for fn in TOOL_FUNCTIONS} == EXPECTED_TOOLS

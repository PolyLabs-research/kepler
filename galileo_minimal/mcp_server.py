#!/usr/bin/env python3
"""
mcp_server.py - Kepler's equation tools over the Model Context Protocol (MCP).

Any MCP client (Claude Code, Claude Desktop, an agent framework) can call the
synthesise, discover, verify loop as tools:

    list_models        which specialists and tiers have trained checkpoints
    classify_data      Overseer's model rankings for a dataset, no inference
    discover_equation  data -> equation in original units, ranked and refined
    verify_equation    equation + data -> RMSE, R^2 and a match verdict (never refits)
    fit_skeleton       your equation's constants fitted to data, with error bars
    synthesize_data    equation, description, ODE or physics law -> data

Run (stdio, the transport Claude Code and Claude Desktop use):

    python mcp_server.py

Register with Claude Code from anywhere:

    claude mcp add kepler -- /path/to/galileo_minimal/.venv/bin/python /path/to/galileo_minimal/mcp_server.py

Environment:
    KEPLER_CHECKPOINTS   checkpoint directory (default: <this directory>/checkpoints)
    KEPLER_DEFAULT_TIER  tier used when a call gives none (default: 2)

The tool implementations are plain functions that work without the `mcp`
package, so they can be imported and tested directly; only `build_server`
and `main` need `mcp` (install with `pip install -r requirements-mcp.txt`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from galileo_tools import GalileoTools  # noqa: E402
from kepler.fitting import fit as _fit_law  # noqa: E402  (importable once galileo_tools has loaded)
from kepler.laws import LAWS  # noqa: E402

try:
    from pydantic import Field
except ImportError:  # pragma: no cover - pydantic ships with mcp; the functions still work without it
    def Field(**kwargs: Any) -> Any:  # type: ignore[misc]
        return None


INSTRUCTIONS = """Kepler discovers closed-form equations from numerical data with small
specialist transformer models and least-squares refinement.

Typical loop: call discover_equation with your x and y (in their natural units;
the data is normalised internally and the equation comes back in your units),
read `equation`, `fit.r2` and `ranking_note`, then call verify_equation with the
same data to confirm, or synthesize_data to generate data from a hypothesis and
compare. verify_equation never refits: it scores the equation exactly as written.
To test your own idea of the shape, call fit_skeleton with an equation whose
constants are numbers ('2.0*x**0.5 + 1.0') or a named law ('power'); it fits the
constants and returns them with standard errors and BIC, so several proposals
for the same data can be compared. Call list_models first if you need to know which specialists are
trained. When x is omitted the sample index 0..n-1 is used.

Families covered: polynomials, exponentials, power laws, logarithms, logistic
curves, first and second order linear ODEs, and AR time series. Results carry
`candidates` with every specialist's score when no model_hint is given."""


# =============================================================================
# Shared tool instance and JSON hygiene
# =============================================================================

_TOOLS: Optional[GalileoTools] = None


def get_tools() -> GalileoTools:
    """Lazily build the GalileoTools instance from environment settings."""
    global _TOOLS
    if _TOOLS is None:
        checkpoint_dir = os.environ.get("KEPLER_CHECKPOINTS") or str(HERE / "checkpoints")
        tier = int(os.environ.get("KEPLER_DEFAULT_TIER", "2"))
        _TOOLS = GalileoTools(checkpoint_dir=checkpoint_dir, default_tier=tier)
    return _TOOLS


def jsonable(obj: Any) -> Any:
    """Convert numpy types and non-finite floats so a result serialises as JSON."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


# =============================================================================
# Tool functions (plain Python; registered with the MCP server in build_server)
# =============================================================================

Y = Annotated[List[float], Field(description="Observed output values (dependent variable).")]
X = Annotated[Optional[List[float]], Field(description="Input values (independent variable, or time for ODE data), same length as y. Omit to use the sample index 0..n-1.")]


def discover_equation(
    y: Y,
    x: X = None,
    model_hint: Annotated[Optional[str], Field(description="Force one specialist: polynomial, transcendental, linear_ode, ar or physics_laws. Omit to run every trained specialist and rank them.")] = None,
    data_type: Annotated[Optional[str], Field(description="Coarse hint instead of a model: algebraic, timeseries, ode or physics.")] = None,
    tier: Annotated[Optional[int], Field(description="Restrict to one difficulty tier (1, 2 or 3). Omit to use every tier with a checkpoint.")] = None,
    normalize: Annotated[bool, Field(description="Map the data into each model's training frame before inference and map the equation back. Leave on for real data.")] = True,
    refine: Annotated[bool, Field(description="Refit the constants of the predicted structure to the data by least squares.")] = True,
    strategy: Annotated[str, Field(description="With no model_hint or data_type: 'rank' runs every specialist and returns the best by BIC with all candidates; 'route' lets the Overseer pick one.")] = "rank",
) -> Dict[str, Any]:
    """Discover the symbolic equation underlying numerical data.

    Returns the equation in the units of x and y, the fit quality (RMSE, R^2,
    number of constants, an ill_conditioned flag), the normalisation used, the
    model's raw prediction, and, when ranking, every candidate's BIC and why
    the winner was chosen. Errors come back as {"error": ...}.
    """
    return jsonable(get_tools().discover_equation(
        y=y, x=x, data_type=data_type, model_hint=model_hint, tier=tier,
        normalize=normalize, refine=refine, strategy=strategy,
    ))


def verify_equation(
    equation: Annotated[str, Field(description="Equation in x to test, SymPy syntax, e.g. '2.5*x**2 - 1.3*x + 0.7' or '20 + 80*exp(-0.25*(x - 2000))'.")],
    y: Y,
    x: X = None,
    tolerance: Annotated[float, Field(description="Relative RMSE (RMSE / range of y) below which the equation counts as a match. Default 0.1.")] = 0.1,
) -> Dict[str, Any]:
    """Check how well an equation reproduces the data.

    Returns match (bool), verdict, rmse, max_error, relative_error and r2.
    Use it to confirm a discovered equation or to test your own hypothesis.
    """
    return jsonable(get_tools().verify_equation(equation=equation, y=y, x=x, tolerance=tolerance))


def synthesize_data(
    equation: Annotated[Optional[str], Field(description="Equation in x, e.g. 'x**2 + 2*x - 1' or 'exp(-0.5*x)'.")] = None,
    description: Annotated[Optional[str], Field(description="Natural language, e.g. 'exponential decay with rate 0.5' or 'damped oscillation'.")] = None,
    ode: Annotated[Optional[str], Field(description="ODE, e.g. \"x'' + 4*x = 0\" or \"dx/dt = -0.5*x\".")] = None,
    physics_law: Annotated[Optional[str], Field(description="newton_gravity, coulomb, ideal_gas, kinetic_energy, hookes_law or gravitational_pe.")] = None,
    variables: Annotated[Optional[Dict[str, Any]], Field(description="Variable values for a physics law, scalars or [min, max] ranges, e.g. {\"M1\": 5.97e24, \"M2\": 1000, \"R\": [6.4e6, 4e8]}.")] = None,
    x_range: Annotated[Optional[List[float]], Field(description="[min, max] for x. Default [-5, 5].")] = None,
    t_range: Annotated[Optional[List[float]], Field(description="[min, max] time range for an ODE. Default [0, 10].")] = None,
    n_points: Annotated[int, Field(description="Number of points to generate. Default 100.")] = 100,
    initial_conditions: Annotated[Optional[List[float]], Field(description="Initial conditions for an ODE. Default [1.0] or [1.0, 0.0].")] = None,
    noise_std: Annotated[float, Field(description="Standard deviation of Gaussian noise to add. Default 0.")] = 0.0,
) -> Dict[str, Any]:
    """Generate numerical data from an equation, description, ODE or physics law.

    Give exactly one of equation, description, ode or physics_law. Returns x
    and y arrays (or a data table for physics laws) to feed into
    discover_equation or verify_equation for hypothesis testing.
    """
    kwargs: Dict[str, Any] = {"n_points": n_points, "noise_std": noise_std}
    if equation is not None:
        kwargs["equation"] = equation
    if description is not None:
        kwargs["description"] = description
    if ode is not None:
        kwargs["ode"] = ode
    if physics_law is not None:
        kwargs["physics_law"] = physics_law
    if variables is not None:
        kwargs["variables"] = variables
    if x_range is not None:
        kwargs["x_range"] = x_range
    if t_range is not None:
        kwargs["t_range"] = t_range
    if initial_conditions is not None:
        kwargs["initial_conditions"] = initial_conditions
    return jsonable(get_tools().synthesize_data(**kwargs))


def fit_skeleton(
    equation: Annotated[str, Field(description="Equation in x whose numeric constants are fitted, e.g. '2.0*x**0.5 + 1.0' (integers such as the 2 in x**2 stay fixed; start no constant at 0.0), or a law name: " + ", ".join(LAWS) + ".")],
    y: Y,
    x: X = None,
    sigma: Annotated[Optional[List[float]], Field(description="Uncertainty of each y, same length; points with smaller sigma count for more. Omit for equal weights.")] = None,
    space: Annotated[str, Field(description="'linear', or 'log' to fit log(f(x)) to log(y) for data spanning decades (every y > 0).")] = "linear",
) -> Dict[str, Any]:
    """Fit the constants of a proposed equation (or a named law) to data.

    Keeps the structure you give and fits every numeric constant by (weighted)
    least squares. Returns the fitted equation, each constant with its standard
    error, RMSE, R^2, BIC (lower is better; compare proposals on the same data,
    sigma and space) and an ill_conditioned flag. Errors come back as {"error": ...}.
    """
    try:
        x_values = list(range(len(y))) if x is None else x
        return jsonable(_fit_law(equation, x_values, y, sigma=sigma, space=space).to_dict())
    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}


def classify_data(y: Y, x: X = None) -> Dict[str, Any]:
    """Rank which specialist families suit the data, without running a model.

    Returns the Overseer's rankings with confidences and its reasoning. Cheap;
    useful to decide whether to pass a model_hint to discover_equation.
    """
    return jsonable(get_tools().classify_data(y=y, x=x))


def list_models() -> Dict[str, Any]:
    """Report which specialists and tiers have trained checkpoints, and server status.

    `available_models` maps model name to the tiers that can be run. An empty
    map means no checkpoints were found under the configured directory.
    """
    return jsonable(get_tools().get_status())


TOOL_FUNCTIONS = [discover_equation, verify_equation, fit_skeleton, synthesize_data, classify_data, list_models]


# =============================================================================
# Server
# =============================================================================

def build_server():
    """Create the MCP server with every tool registered (needs the `mcp` package)."""
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x
    server = Server("kepler", instructions=INSTRUCTIONS)
    for fn in TOOL_FUNCTIONS:
        server.tool()(fn)
    return server


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Kepler MCP server.")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports")
    parser.add_argument("--list-tools", action="store_true", help="Print the registered tools as JSON and exit")
    args = parser.parse_args(argv)

    server = build_server()
    if args.list_tools:
        import asyncio
        tools = asyncio.run(server.list_tools())
        print(json.dumps([{"name": t.name, "description": t.description} for t in tools], indent=2))
        return 0
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

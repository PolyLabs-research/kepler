#!/usr/bin/env python3
"""
galileo_tools.py - Universal LLM Tool Interface for Galileo Minimal

Provides three core tools callable by any LLM via function-calling:
    1. synthesize_data - Generate numerical data from equations
    2. discover_equation - Find symbolic equation from numerical data
    3. verify_equation - Check if an equation fits given data

Usage with any LLM:
    from galileo_tools import GalileoTools, TOOL_SCHEMAS
    
    # Initialize
    tools = GalileoTools()
    
    # Get OpenAI-compatible tool schemas
    schemas = TOOL_SCHEMAS  # Pass to LLM
    
    # Execute tool calls from LLM
    result = tools.execute("synthesize_data", {"equation": "x**2 + 2*x - 1"})

Real-data handling (discover_equation):
    1. The data is mapped into the specialist's training frame (see normalize.py)
       so the clipped feature extractors see the shape of the curve.
    2. The predicted structure is kept, but every constant in it is refitted to
       the data by least squares (see refine.py).
    3. The equation is reported in the original units with RMSE and R^2.
    Both steps can be switched off with normalize=False / refine=False.

The TOOL_SCHEMAS are OpenAI-compatible and work with:
- OpenAI (GPT-4, etc.)
- Anthropic Claude (via tool_use)
- Google Gemini
- Local models via Ollama, vLLM, etc.
"""
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from typing import Any, Dict, List, Optional, Tuple, Union

# Local imports
from synthesizer import Synthesizer, SynthesizerError

# Import Overseer for auto-routing (optional but recommended)
_OVERSEER_AVAILABLE = False
try:
    from overseer import Overseer, ClassificationResult
    _OVERSEER_AVAILABLE = True
except ImportError:
    Overseer = None
    ClassificationResult = None

# Configuration, features and post-processing need only numpy/scipy/sympy.
from config import get_device, CONFIGS, DOMAINS
from vocab import Vocabulary
from features import (
    compute_algebraic_features,
    compute_timeseries_features,
    compute_dynamical_features,
    compute_multivariate_features,
)
from expression import (
    ExpressionParseError,
    expression_to_string,
    format_ar_equation,
    format_ode_equation,
    parse_ar_tokens,
    parse_ode_tokens,
    prune_small_terms,
    tokens_to_infix,
    tokens_to_sympy,
)
from normalize import fit_normalization
from refine import (
    cancellation_ratio,
    count_constants,
    r_squared,
    refine_ar,
    refine_expression,
    refine_ode,
    rmse as _rmse,
)
# The refine/expression/normalize shims above have put the kepler package on the path.
from kepler.ranking import MIN_R2, OCCAM_WINDOW, bic as _bic, occam_order

# Running a specialist needs torch and a trained checkpoint.
_DISCOVERY_AVAILABLE = False
try:
    import torch
    from model import SymbolicTransformer
    _DISCOVERY_AVAILABLE = True
except ImportError:
    torch = None
    SymbolicTransformer = None


# =============================================================================
# Tool Schemas (OpenAI-compatible format)
# =============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "synthesize_data",
            "description": "Generate numerical data from a symbolic equation, natural language description, or ODE specification. Use this to create test data for hypothesis verification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "Symbolic equation like 'x**2 + 2*x - 1', 'exp(-0.5*x)', 'sin(2*x)'"
                    },
                    "description": {
                        "type": "string",
                        "description": "Natural language like 'exponential decay with rate 0.5', 'damped oscillation'"
                    },
                    "ode": {
                        "type": "string",
                        "description": "ODE specification like \"x'' + 4*x = 0\" or \"dx/dt = -0.5*x\""
                    },
                    "physics_law": {
                        "type": "string",
                        "enum": ["newton_gravity", "coulomb", "ideal_gas", "kinetic_energy", "hookes_law", "gravitational_pe"],
                        "description": "Name of physics law to use"
                    },
                    "variables": {
                        "type": "object",
                        "description": "Variable values for physics laws, e.g. {\"M1\": 5.97e24, \"M2\": 1000, \"R\": [6.4e6, 4e8]}"
                    },
                    "x_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Range for x values as [min, max]. Default [-5, 5]"
                    },
                    "t_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Time range for ODEs as [min, max]. Default [0, 10]"
                    },
                    "n_points": {
                        "type": "integer",
                        "description": "Number of data points to generate. Default 100"
                    },
                    "initial_conditions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Initial conditions for ODEs. Default [1.0] or [1.0, 0.0]"
                    },
                    "noise_std": {
                        "type": "number",
                        "description": "Standard deviation of Gaussian noise to add. Default 0"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "discover_equation",
            "description": "Analyze numerical data and discover the underlying symbolic equation. Without a model_hint or data_type every trained specialist is run and the refined fits are ranked (BIC over the residuals; ill-conditioned fits are excluded). The data is normalised into each model's frame, the constants of the predicted structure are refitted to the data, and the equation is returned in the original units together with RMSE, R^2 and the ranked candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Input x values (independent variable, or time for ODE data). If omitted, the sample index 0..n-1 is used."
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Output y values (dependent variable)"
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["algebraic", "timeseries", "ode", "physics"],
                        "description": "Type of data to help select the right model. Default: auto-detect"
                    },
                    "model_hint": {
                        "type": "string",
                        "enum": ["polynomial", "transcendental", "ar", "linear_ode", "physics_laws"],
                        "description": "Specific model to use. Default: auto-select based on data_type"
                    },
                    "tier": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "Model tier to use. Default: 2"
                    },
                    "normalize": {
                        "type": "boolean",
                        "description": "Map the data into the model's training frame before inference and map the equation back. Default true; set false only for data already in that frame."
                    },
                    "refine": {
                        "type": "boolean",
                        "description": "Refit the constants of the predicted structure to the data by least squares. Default true."
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["rank", "route"],
                        "description": "Only used when neither model_hint nor data_type is given. 'rank' (default) runs every available specialist and tier and returns the best fit with all candidates; 'route' lets the Overseer pick one model."
                    }
                },
                "required": ["y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_equation",
            "description": "Check if a proposed equation matches the given data. Returns match status, RMSE, max error, relative error and R^2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "The equation to verify, e.g. 'x**2 + 2*x - 1'"
                    },
                    "x": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Input x values. If omitted, the sample index 0..n-1 is used."
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Observed y values to compare against"
                    },
                    "tolerance": {
                        "type": "number",
                        "description": "Relative error tolerance for match. Default 0.1 (10%)"
                    }
                },
                "required": ["equation", "y"]
            }
        }
    }
]

# Anthropic Claude format (slightly different structure)
ANTHROPIC_TOOL_SCHEMAS = [
    {
        "name": schema["function"]["name"],
        "description": schema["function"]["description"],
        "input_schema": schema["function"]["parameters"]
    }
    for schema in TOOL_SCHEMAS
]


# =============================================================================
# GalileoTools - Main Interface Class
# =============================================================================

class GalileoTools:
    """
    Universal tool interface for Galileo equation discovery.
    
    Provides three core capabilities:
    1. synthesize_data - Generate data from equations
    2. discover_equation - Find equations from data  
    3. verify_equation - Check equation against data
    
    Example:
        tools = GalileoTools()
        
        # Synthesize data
        result = tools.synthesize_data(equation="x**2")
        
        # Discover equation
        result = tools.discover_equation(y=[1, 4, 9, 16, 25])
        
        # Verify equation
        result = tools.verify_equation(
            equation="x**2",
            x=[1, 2, 3, 4, 5],
            y=[1, 4, 9, 16, 25]
        )
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", default_tier: int = 2, use_overseer: bool = True):
        """
        Initialize GalileoTools.
        
        Args:
            checkpoint_dir: Path to checkpoints directory
            default_tier: Default model tier to use (1, 2, or 3)
            use_overseer: Whether to use Overseer for auto-routing (default True)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.default_tier = default_tier
        self.device = get_device()
        self.synthesizer = Synthesizer()
        
        # Initialize Overseer for auto-routing
        self.use_overseer = use_overseer and _OVERSEER_AVAILABLE
        self._overseer = None
        if self.use_overseer:
            self._overseer = Overseer(models_dir=checkpoint_dir, default_tier=default_tier)
        
        # Cache for loaded models
        self._model_cache: Dict[str, Tuple[SymbolicTransformer, Vocabulary]] = {}
        
        # Classification cache (populated by Overseer during auto-routing)
        self._last_classification = None
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.
        
        This is the main entry point for LLM tool calls.
        
        Args:
            tool_name: One of "synthesize_data", "discover_equation", "verify_equation"
            arguments: Tool arguments as a dictionary
            
        Returns:
            Tool result as a dictionary
        """
        if tool_name == "synthesize_data":
            return self.synthesize_data(**arguments)
        elif tool_name == "discover_equation":
            return self.discover_equation(**arguments)
        elif tool_name == "verify_equation":
            return self.verify_equation(**arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    # =========================================================================
    # Tool 1: Synthesize Data
    # =========================================================================
    
    def synthesize_data(
        self,
        equation: Optional[str] = None,
        description: Optional[str] = None,
        ode: Optional[str] = None,
        physics_law: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        x_range: Optional[List[float]] = None,
        t_range: Optional[List[float]] = None,
        n_points: int = 100,
        initial_conditions: Optional[List[float]] = None,
        noise_std: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate numerical data from a symbolic specification.
        
        Provide ONE of: equation, description, ode, or physics_law.
        
        Returns:
            Dict with 'x', 'y' arrays and metadata
        """
        try:
            # Determine which synthesis method to use
            if equation:
                x_range = tuple(x_range) if x_range else (-5, 5)
                x, y = self.synthesizer.from_equation(
                    equation, 
                    x_range=x_range,
                    n_points=n_points,
                    noise_std=noise_std
                )
                method = "equation"
                spec = equation
                
            elif description:
                x_range = tuple(x_range) if x_range else None
                x, y = self.synthesizer.from_description(
                    description,
                    x_range=x_range,
                    n_points=n_points,
                    noise_std=noise_std
                )
                method = "description"
                spec = description
                
            elif ode:
                t_range = tuple(t_range) if t_range else (0, 10)
                x, y = self.synthesizer.from_ode(
                    ode,
                    t_range=t_range,
                    y0=initial_conditions,
                    n_points=n_points,
                    noise_std=noise_std
                )
                method = "ode"
                spec = ode
                
            elif physics_law:
                if not variables:
                    return {"error": f"physics_law requires 'variables' parameter"}
                
                # Convert list values to numpy arrays
                processed_vars = {}
                for k, v in variables.items():
                    if isinstance(v, list):
                        processed_vars[k] = np.array(v)
                    else:
                        processed_vars[k] = v
                
                data = self.synthesizer.from_physics(
                    physics_law,
                    n_samples=n_points,
                    noise_std=noise_std,
                    **processed_vars
                )
                
                # Return all variables plus output
                return {
                    "success": True,
                    "method": "physics_law",
                    "specification": physics_law,
                    "data": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in data.items()},
                    "n_points": len(data["output"])
                }
            else:
                return {"error": "Provide one of: equation, description, ode, or physics_law"}
            
            return {
                "success": True,
                "method": method,
                "specification": spec,
                "x": x.tolist(),
                "y": y.tolist(),
                "n_points": len(x),
                "x_range": [float(x.min()), float(x.max())],
                "y_range": [float(y.min()), float(y.max())]
            }
            
        except SynthesizerError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Synthesis failed: {str(e)}"}
    
    # =========================================================================
    # Tool 2: Discover Equation
    # =========================================================================
    
    def discover_equation(
        self,
        y: List[float],
        x: Optional[List[float]] = None,
        data_type: Optional[str] = None,
        model_hint: Optional[str] = None,
        tier: Optional[int] = None,
        include_classification: bool = False,
        normalize: bool = True,
        refine: bool = True,
        strategy: str = "rank",
    ) -> Dict[str, Any]:
        """
        Discover the symbolic equation underlying the data.

        With neither model_hint nor data_type, strategy="rank" (default) runs
        every available specialist and ranks the refined fits (see
        discover_best); strategy="route" asks the Overseer for one model.

        Args:
            y: Output values (required)
            x: Input values, or time for ODE data. If omitted the sample index
               0..n-1 is used and the result says so.
            data_type: One of "algebraic", "timeseries", "ode", "physics"
            model_hint: Specific model to use
            tier: Model tier (1, 2, or 3)
            include_classification: Include Overseer classification details in response
            normalize: Map data into the model's training frame first (default True)
            refine: Refit the predicted constants to the data (default True)

        Returns:
            Dict with the equation in original units, the model's raw prediction,
            the normalisation used, fit quality, and a confidence score.
        """
        try:
            y_arr = np.asarray(y, dtype=np.float64).ravel()
            n = len(y_arr)
            if n == 0:
                return {"error": "y is empty"}
            x_in = None if x is None else np.asarray(x, dtype=np.float64).ravel()
            if x_in is not None and len(x_in) != n:
                return {"error": f"x has {len(x_in)} values but y has {n}"}

            if model_hint is None and data_type is None and strategy == "rank":
                return self.discover_best(y_arr, x_in, tier=tier, normalize=normalize, refine=refine)

            tier = tier or self.default_tier

            # Reset classification cache
            self._last_classification = None

            # Determine model to use (may populate self._last_classification)
            model_name = self._select_model(y_arr, x_in, data_type, model_hint)
            if model_name is None:
                return {"error": "Could not determine appropriate model for this data"}

            model, vocab = self._load_model(model_name, tier)
            if model is None:
                if not _DISCOVERY_AVAILABLE:
                    return {"error": "Discovery requires torch and trained models. Install torch and train a model first."}
                return {"error": f"No checkpoint found for {model_name} tier {tier}"}

            result = self._discover_with_model(model_name, model, vocab, y_arr, x_in, tier, normalize, refine)
            result["auto_routed"] = data_type is None and model_hint is None and self._overseer is not None
            result["strategy"] = "route" if result["auto_routed"] else "hint"

            if include_classification and self._last_classification is not None:
                result["classification"] = {
                    "data_type": self._last_classification.data_type.value,
                    "model_rankings": self._last_classification.rankings,
                    "reasoning": self._last_classification.reasoning,
                }
            return result

        except Exception as e:
            return {"error": f"Discovery failed: {str(e)}"}

    def route_and_discover(
        self,
        y: List[float],
        x: Optional[List[float]] = None,
        tier: Optional[int] = None,
        top_k: int = 3,
        normalize: bool = True,
        refine: bool = True,
    ) -> Dict[str, Any]:
        """
        Use Overseer to route to top-k models and return the best result.

        Every candidate goes through the same normalise, predict, refine path
        as discover_equation; candidates are ranked by a blend of the
        Overseer's confidence and the refined fit's R^2.
        """
        if not _DISCOVERY_AVAILABLE:
            return {"error": "Discovery requires torch and trained models."}
        if self._overseer is None:
            return {"error": "Overseer not available. Use discover_equation instead."}

        try:
            y_arr = np.asarray(y, dtype=np.float64).ravel()
            x_in = None if x is None else np.asarray(x, dtype=np.float64).ravel()
            tier = tier or self.default_tier

            classification = self._overseer.classify(y_arr, x_in)

            results = []
            for model_name, conf in classification.top_k(top_k):
                model, vocab = self._load_model(model_name, tier)
                if model is None:
                    continue
                try:
                    r = self._discover_with_model(model_name, model, vocab, y_arr, x_in, tier, normalize, refine)
                    fit = r.get("fit") or {}
                    r2 = fit.get("r2")
                    fit_score = float(np.clip(r2, 0.0, 1.0)) if r2 is not None and not fit.get("ill_conditioned") else 0.0
                except Exception as e:
                    r = {"model_used": model_name, "equation": f"Error: {e}", "tokens": []}
                    fit_score = 0.0
                r["model"] = model_name
                r["classification_confidence"] = float(conf)
                r["fit_score"] = fit_score
                r["combined_score"] = float(0.5 * conf + 0.5 * fit_score)
                results.append(r)

            if not results:
                return {"error": "No models could process this data"}

            results.sort(key=lambda r: r["combined_score"], reverse=True)
            best = dict(results[0])
            best["success"] = True
            best["confidence"] = best["combined_score"]
            best["confidence_source"] = "overseer+fit"
            best["alternatives"] = [
                {k: r.get(k) for k in ("model", "equation", "fit_score", "classification_confidence", "combined_score", "fit")}
                for r in results[1:]
            ]
            best["classification"] = {
                "data_type": classification.data_type.value,
                "model_rankings": classification.rankings[:5],
                "reasoning": classification.reasoning,
            }
            return best

        except Exception as e:
            return {"error": f"Route and discover failed: {str(e)}"}

    # =========================================================================
    # Run-all-and-rank
    # =========================================================================

    DETERMINISTIC_MODELS = ("polynomial", "transcendental", "linear_ode")
    STOCHASTIC_MODELS = ("ar",)
    MIN_DETERMINISTIC_R2 = MIN_R2
    # Occam window: a richer structure must beat a simpler one by more than this
    # many BIC points to be preferred (Kass & Raftery: 2-6 is "positive", >6 "strong").
    # The rule itself is kepler.ranking's, shared with kepler.rank().
    OCCAM_WINDOW = OCCAM_WINDOW
    # Resampling for the model's eyes: the specialists were trained on evenly
    # spaced samples, so unevenly spaced x is resampled onto an even grid of at
    # least this many points with a shape-preserving interpolant before feature
    # extraction. Refinement always uses the original points.
    RESAMPLE_MIN_POINTS = 64
    RESAMPLE_SPACING_TOL = 1e-3

    def discover_best(
        self,
        y: List[float],
        x: Optional[List[float]] = None,
        tier: Optional[int] = None,
        normalize: bool = True,
        refine: bool = True,
        candidates: Optional[List[Tuple[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Run every available specialist and return the best-ranked fit.

        Deterministic families (polynomial, transcendental, linear_ode) at every
        available tier are ranked by BIC on their residuals in original units,
        with the number of fitted constants as the complexity term. Within an
        Occam window of 6 BIC points of the best score the candidate with the
        fewest constants wins, so a richer structure (an ODE whose solutions
        include an offset exponential, say) has to fit clearly better than the
        closed form to be preferred. Ill-conditioned fits, non-stationary AR
        fits and diverged fits (r2 < 0) are ineligible.

        AR answers a different question (is this a stationary stochastic
        process?) and its one-step residuals are not comparable to curve
        residuals, so it is chosen only when no deterministic family reaches
        R^2 >= 0.5 and the AR fit is stationary.

        Args:
            tier: restrict to one tier (default: every tier with a checkpoint)
            candidates: explicit [(model, tier), ...] to run instead of all
        """
        try:
            y_arr = np.asarray(y, dtype=np.float64).ravel()
            n = len(y_arr)
            if n == 0:
                return {"error": "y is empty"}
            x_in = None if x is None else np.asarray(x, dtype=np.float64).ravel()
            if x_in is not None and len(x_in) != n:
                return {"error": f"x has {len(x_in)} values but y has {n}"}

            if candidates is None:
                available = self.list_available_models()
                candidates = [
                    (m, t)
                    for m in (*self.DETERMINISTIC_MODELS, *self.STOCHASTIC_MODELS)
                    for t in available.get(m, [])
                    if tier is None or t == tier
                ]
            if not candidates:
                if not _DISCOVERY_AVAILABLE:
                    return {"error": "Discovery requires torch and trained models. Install torch and train a model first."}
                where = f" for tier {tier}" if tier else ""
                return {"error": f"No trained specialists found{where} under {self.checkpoint_dir}"}

            rows: List[Dict[str, Any]] = []
            for model_name, t in candidates:
                model, vocab = self._load_model(model_name, t)
                if model is None:
                    rows.append({"success": False, "model_used": model_name, "tier": t, "error": "no checkpoint"})
                    continue
                try:
                    rows.append(self._discover_with_model(model_name, model, vocab, y_arr, x_in, t, normalize, refine))
                except Exception as e:
                    rows.append({"success": False, "model_used": model_name, "tier": t, "error": str(e)})

            ranked, best, best_entry, note = self._rank_candidates(rows, n)
            if best is None:
                return {"error": "No specialist produced a usable fit", "candidates": ranked}

            result = dict(best)
            if best_entry is not None and best_entry.get("bic") is not None:
                # Put the winner's score where a caller reading `fit` expects it,
                # not only in its row of `candidates`.
                result["fit"] = {**(result.get("fit") or {}),
                                 "bic": best_entry["bic"], "delta_bic": best_entry["delta_bic"]}
            result.update(success=True, strategy="rank", ranking_note=note, candidates=ranked)
            return result

        except Exception as e:
            return {"error": f"Ranked discovery failed: {str(e)}"}

    def _rank_candidates(self, rows: List[Dict[str, Any]], n: int):
        """
        Score candidate results by BIC and apply the selection rule.

        Returns (entries, best_row, best_entry, note): every candidate's score
        entry, the winning result row and its entry, and the reason.
        """
        scored = []
        for r in rows:
            fit = r.get("fit") or {}
            model = r.get("model_used")
            track = "stochastic" if model in self.STOCHASTIC_MODELS else "deterministic"
            entry: Dict[str, Any] = {
                "model": model,
                "tier": r.get("tier"),
                "track": track,
                "equation": r.get("equation"),
                "r2": fit.get("r2"),
                "rmse": fit.get("rmse"),
                "n_constants": int(fit.get("n_constants") or fit.get("n_parameters") or 0),
                "ill_conditioned": bool(fit.get("ill_conditioned", False)),
                "bic": None,
                "delta_bic": None,
                "eligible": False,
                "reason": "",
            }
            rmse_v = fit.get("rmse")
            if not r.get("success"):
                entry["reason"] = r.get("error") or r.get("postprocess_error") or "failed"
            elif rmse_v is None or not np.isfinite(rmse_v):
                entry["reason"] = r.get("postprocess_error") or "no fit"
            else:
                entry["bic"] = _bic(n * float(rmse_v) ** 2, n, entry["n_constants"])
                r2_v = fit.get("r2")
                if entry["ill_conditioned"]:
                    entry["reason"] = "ill-conditioned fit (cancelling terms)"
                elif track == "stochastic" and not fit.get("stationary", True):
                    entry["reason"] = "non-stationary AR fit"
                elif r2_v is not None and (not np.isfinite(r2_v) or r2_v < 0.0):
                    # Worse than predicting the mean: a diverged ODE trajectory
                    # or a template that cannot bend to the data. Its BIC is
                    # kept so it still sorts and can serve as a last resort.
                    entry["reason"] = "diverged fit (worse than the mean, r2 < 0)"
                else:
                    entry["eligible"] = True
            scored.append((entry, r))

        det = sorted(((e, r) for e, r in scored if e["track"] == "deterministic" and e["eligible"]), key=lambda er: er[0]["bic"])
        sto = sorted(((e, r) for e, r in scored if e["track"] == "stochastic" and e["eligible"]), key=lambda er: er[0]["bic"])

        best, note = None, ""
        if det:
            # Occam window: simplest candidate among those close to the best BIC.
            floor = det[0][0]["bic"]
            det = occam_order(det, self.OCCAM_WINDOW, lambda er: er[0]["bic"], lambda er: er[0]["n_constants"])
        if det and (det[0][0]["r2"] or 0.0) >= self.MIN_DETERMINISTIC_R2:
            e = det[0][0]
            best = det[0][1]
            note = f"best deterministic fit by BIC: {e['model']} tier {e['tier']} (r2={e['r2']:.4f})"
            if e["bic"] > floor:
                note += f"; chosen over a richer structure within the {self.OCCAM_WINDOW:g}-point Occam window"
        elif sto:
            best = sto[0][1]
            note = "no deterministic family reaches r2 >= 0.5; the series is consistent with a stationary AR process"
            if det:
                note += f" (best deterministic r2={det[0][0]['r2']:.4f})"
        elif det:
            e = det[0][0]
            best = det[0][1]
            note = f"low confidence: best deterministic fit {e['model']} tier {e['tier']} has r2={e['r2']:.4f}"
        else:
            fallback = sorted(((e, r) for e, r in scored if e["bic"] is not None), key=lambda er: er[0]["bic"])
            if fallback:
                best = fallback[0][1]
                note = f"every fit was flagged ({fallback[0][0]['reason']}); reporting the least bad"

        entries = [e for e, _ in scored]
        for track in ("deterministic", "stochastic"):
            bics = [e["bic"] for e in entries if e["track"] == track and e["bic"] is not None]
            if bics:
                floor = min(bics)
                for e in entries:
                    if e["track"] == track and e["bic"] is not None:
                        e["delta_bic"] = float(e["bic"] - floor)
        entries.sort(key=lambda e: (not e["eligible"], e["track"] != "deterministic",
                                    e["bic"] if e["bic"] is not None else float("inf")))
        best_entry = next((e for e, r in scored if r is best), None) if best is not None else None
        return entries, best, best_entry, note

    # =========================================================================
    # Shared discovery path: normalise -> predict -> interpret -> refine
    # =========================================================================

    @staticmethod
    def _resolve_x(x: Optional[np.ndarray], n: int) -> Tuple[np.ndarray, Optional[str]]:
        """Return x as float64, fabricating the sample index when none was given."""
        if x is None:
            return np.arange(n, dtype=np.float64), "index"
        return np.asarray(x, dtype=np.float64), None

    def _resample_for_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_assumed: Optional[str],
        domain: str,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """
        Return the (x, y) the specialist should look at, and a note if it differs.

        The generators sample every family on an evenly spaced grid, so a
        power law sampled at log-spaced x looks to the model like a logarithm
        or an exponential. When x is unevenly spaced the data is sorted and
        resampled onto an even grid of max(n, RESAMPLE_MIN_POINTS) points with a
        shape-preserving (PCHIP) interpolant, which assumes nothing about the
        family and introduces no kinks. Only the prediction uses the resampled
        series; constants are refitted on the original points.

        Not applied when x is the sample index, for time-series and
        multivariate models (their features are index based), with fewer than
        three points, or when x has repeated values.
        """
        n = len(y)
        if x_assumed or domain in ("timeseries", "multivariate") or n < 3:
            return x, y, None
        order = np.argsort(x, kind="stable")
        xs, ys = x[order], y[order]
        gaps = np.diff(xs)
        if np.any(gaps <= 0):
            return xs, ys, None
        if (gaps.max() - gaps.min()) <= self.RESAMPLE_SPACING_TOL * gaps.mean():
            return xs, ys, None
        n_grid = max(n, self.RESAMPLE_MIN_POINTS)
        xg = np.linspace(xs[0], xs[-1], n_grid)
        yg = PchipInterpolator(xs, ys)(xg)
        info = {
            "method": "pchip",
            "n_original": int(n),
            "n_grid": int(n_grid),
            "reason": "x is not evenly spaced; the specialists were trained on evenly spaced samples, "
                      "so the model predicted from a resampled view and the constants were refitted on the original points",
        }
        return xg, yg, info

    def _predict_tokens(self, model: Any, vocab: Any, features: np.ndarray) -> List[str]:
        """Greedy-decode one feature vector into token strings."""
        if torch is None:
            raise RuntimeError("Discovery requires torch")
        feat_tensor = torch.tensor(np.asarray(features, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            pred_ids = model.generate(feat_tensor, vocab)
        return vocab.decode(pred_ids)

    def _discover_with_model(
        self,
        model_name: str,
        model: Any,
        vocab: Any,
        y_arr: np.ndarray,
        x_in: Optional[np.ndarray],
        tier: int,
        normalize: bool,
        refine: bool,
    ) -> Dict[str, Any]:
        """Run one specialist end to end and return the full result dictionary."""
        domain = DOMAINS.get(model_name, "algebraic")
        x_arr, x_assumed = self._resolve_x(x_in, len(y_arr))

        norm = fit_normalization(x_arr, y_arr, model_name, domain, enabled=normalize, x_assumed=x_assumed)
        x_n, y_n = norm.transform(x_arr, y_arr)

        # The model predicts from an evenly spaced view of the data; refinement
        # below fits the predicted structure to the original points (x_n, y_n).
        x_m, y_m, resampling = self._resample_for_model(x_n, y_n, x_assumed, domain)
        features = self._compute_features(y_m, x_m, model_name)
        pred_tokens = self._predict_tokens(model, vocab, features)

        result: Dict[str, Any] = {
            "success": True,
            "model_used": model_name,
            "tier": tier,
            "tokens": pred_tokens,
            "model_equation": self._tokens_to_equation(pred_tokens, model_name),
            "normalization": norm.to_dict(),
            "resampling": resampling,
        }
        if x_assumed:
            result["x_assumed"] = "sample index 0..n-1 (no x provided)"

        result.update(self._postprocess_prediction(domain, model_name, pred_tokens, norm, x_arr, y_arr, x_n, y_n, refine))

        fit = result.get("fit") or {}
        r2 = fit.get("r2")
        if r2 is not None and domain in ("algebraic", "dynamical"):
            result["confidence"] = float(np.clip(r2, 0.0, 1.0))
            result["confidence_source"] = "fit_r2"
            if fit.get("ill_conditioned"):
                result["warning"] = ("the fit relies on large cancelling terms (see fit.cancellation_ratio); "
                                     "its R^2 overstates how well this structure describes the data")
        else:
            # Stochastic (AR) and multivariate paths: R^2 is bounded by noise,
            # so fall back to the structural heuristic.
            result["confidence"] = self._estimate_confidence(features, pred_tokens)
            result["confidence_source"] = "heuristic"
        return result

    @staticmethod
    def _fit_metrics(expr, x_n: np.ndarray, y_n: np.ndarray, y_scale: float) -> Dict[str, Any]:
        """RMSE (original units) and R^2 of an expression evaluated in normalised coordinates."""
        import sympy
        from expression import X as _X
        try:
            func = sympy.lambdify(_X, expr, modules="numpy")
            with np.errstate(all="ignore"):
                pred = func(x_n)
            value = _rmse(pred, y_n)
            ratio = cancellation_ratio(expr, x_n, y_n, _X)
            return {
                "rmse_initial": None,
                "rmse": float(value * y_scale) if np.isfinite(value) else None,
                "r2": float(r_squared(pred, y_n)) if np.isfinite(value) else None,
                "n_parameters": 0,
                "n_constants": count_constants(expr),
                "improved": False,
                "ill_conditioned": bool(np.isfinite(ratio) and ratio > 10.0),
                "cancellation_ratio": float(ratio) if np.isfinite(ratio) else None,
                "message": "constants as predicted (refine=False)",
            }
        except Exception as exc:
            return {"rmse_initial": None, "rmse": None, "r2": None, "n_parameters": 0,
                    "improved": False, "message": f"could not evaluate: {exc}"}

    def _postprocess_prediction(
        self,
        domain: str,
        model_name: str,
        tokens: List[str],
        norm: Any,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        x_n: np.ndarray,
        y_n: np.ndarray,
        refine: bool,
    ) -> Dict[str, Any]:
        """Interpret predicted tokens, refit constants, and express the result in original units."""
        out: Dict[str, Any] = {"refined": False, "fit": None}
        y_scale = norm.y_scale if norm.enabled else 1.0
        try:
            if domain == "algebraic":
                expr_n = tokens_to_sympy(tokens, model_name)
                if refine:
                    res = refine_expression(expr_n, x_n, y_n, affine_wrapper=True,
                                            bound=1e6 if norm.enabled else None)
                    expr_n = res.expression
                    fit = res.fit_dict()
                    for key in ("rmse", "rmse_initial"):
                        if fit[key] is not None:
                            fit[key] = float(fit[key] * y_scale)
                    out["refined"] = bool(res.improved)
                    out["fit"] = fit
                else:
                    out["fit"] = self._fit_metrics(expr_n, x_n, y_n, y_scale)
                tol = 1e-9 * (abs(y_scale) + abs(norm.y_shift) + 1.0)
                readable = prune_small_terms(norm.readable_inverse(expr_n), tol)
                out["equation"] = expression_to_string(readable)
                out["equation_normalized"] = expression_to_string(expr_n)

            elif domain == "timeseries":
                phi = parse_ar_tokens(tokens)
                info: Dict[str, Any] = {"coefficients": phi, "order": len(phi), "mean": float(np.mean(y_arr))}
                if refine:
                    info = refine_ar(y_arr, phi)
                    out["refined"] = bool(info["improved"])
                    out["fit"] = {k: info.get(k) for k in ("rmse_initial", "rmse", "r2", "n_parameters", "n_constants",
                                                           "stationary", "max_root_modulus", "ill_conditioned", "message")}
                    if not info.get("stationary", True):
                        out["warning"] = "the refitted AR coefficients are not stationary; the series is probably not an AR process"
                out["ar"] = {"coefficients": info["coefficients"], "order": info["order"], "mean": info["mean"]}
                out["equation"] = format_ar_equation(info["coefficients"])
                out["note"] = "AR coefficients apply to the series after subtracting its mean."

            elif domain == "dynamical":
                params_n = parse_ode_tokens(tokens)
                if params_n is None:
                    raise ExpressionParseError("tokens do not match a linear ODE template")
                params = norm.inverse_ode(params_n)
                if refine:
                    # Fit in normalised time (well conditioned), then convert back.
                    info = refine_ode(params_n, x_n, y_n, bound=1e6 if norm.enabled else None)
                    fitted_n = {k: info[k] for k in ("order", "a", "C1", "C2") if k in info}
                    params = norm.inverse_ode(fitted_n)
                    out["refined"] = bool(info["improved"])
                    out["fit"] = {k: info.get(k) for k in ("rmse_initial", "rmse", "r2", "n_parameters", "n_constants",
                                                           "condition_number", "ill_conditioned", "message")}
                    for key in ("rmse", "rmse_initial"):
                        if out["fit"][key] is not None:
                            out["fit"][key] = float(out["fit"][key] * y_scale)
                    ic = info.get("initial_conditions") or {}
                    t_scale = norm.x_scale if norm.enabled else 1.0
                    out["initial_conditions"] = {
                        k: float(v * y_scale / (t_scale if k == "v0" else 1.0)) for k, v in ic.items()
                    }
                out["ode"] = params
                out["equation"] = format_ode_equation(params)
                out["equation_normalized"] = format_ode_equation(params_n)

            else:  # multivariate
                out["equation"] = self._tokens_to_equation(tokens, model_name)
                out["warning"] = (
                    "physics_laws discovery through this interface sees only the output column, "
                    "but the model was trained on input-output correlations; treat the result as unreliable."
                )
        except Exception as exc:
            # A malformed prediction must not lose the rest of the result.
            out["equation"] = self._tokens_to_equation(tokens, model_name)
            out["fit"] = None
            out["postprocess_error"] = f"could not interpret the prediction: {type(exc).__name__}: {exc}"
        return out

    # =========================================================================
    # Tool 3: Verify Equation
    # =========================================================================
    
    def verify_equation(
        self,
        equation: str,
        y: List[float],
        x: Optional[List[float]] = None,
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Verify if an equation matches the given data.
        
        The equation is evaluated at exactly the given x values, so irregular
        spacing and unsorted x are fine.
        
        Args:
            equation: Equation to verify
            y: Observed y values
            x: x values (optional; sample index 0..n-1 if omitted)
            tolerance: Relative error tolerance (default 0.1 = 10%)
            
        Returns:
            Dict with match status and error metrics
        """
        try:
            y_observed = np.asarray(y, dtype=np.float64)
            n = len(y_observed)
            
            # Generate x if not provided
            if x is None:
                x = np.arange(n, dtype=np.float64)
            else:
                x = np.asarray(x, dtype=np.float64)
            if x.shape != y_observed.shape:
                return {"error": f"x has {x.size} values but y has {n}"}
            
            # Evaluate the equation at exactly the given x values. No grid,
            # no interpolation, so irregular or unsorted x is handled exactly.
            y_predicted = self.synthesizer.evaluate_at(equation, x)
            undefined = ~np.isfinite(y_predicted)
            if undefined.any():
                return {
                    "error": (
                        f"Equation is not finite at {int(undefined.sum())} of {n} x values "
                        f"(first at x={x[undefined][0]:g})"
                    )
                }
            
            # Compute error metrics
            errors = np.abs(y_observed - y_predicted)
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            max_error = float(np.max(errors))
            r2 = float(r_squared(y_predicted, y_observed))
            
            y_range = float(np.max(y_observed) - np.min(y_observed))
            relative_error = rmse / y_range if y_range > 1e-8 else rmse
            
            # Determine match
            match = relative_error < tolerance
            
            return {
                "success": True,
                "match": bool(match),
                "equation": equation,
                "rmse": rmse,
                "max_error": max_error,
                "r2": r2 if np.isfinite(r2) else None,
                "relative_error": float(relative_error),
                "tolerance": tolerance,
                "n_points": n,
                "verdict": "MATCH" if match else "NO MATCH"
            }
            
        except SynthesizerError as e:
            return {"error": f"Could not evaluate equation: {str(e)}"}
        except Exception as e:
            return {"error": f"Verification failed: {str(e)}"}
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _select_model(
        self, 
        y: np.ndarray, 
        x: np.ndarray, 
        data_type: Optional[str],
        model_hint: Optional[str]
    ) -> Optional[str]:
        """Select the best model for the given data using Overseer when available."""
        
        # If explicit hint, use it
        if model_hint and model_hint in CONFIGS:
            return model_hint
        
        # If data_type specified, map to model
        type_to_model = {
            "algebraic": "polynomial",  # Default algebraic model
            "timeseries": "ar",
            "ode": "linear_ode",
            "physics": "physics_laws"
        }
        
        if data_type and data_type in type_to_model:
            return type_to_model[data_type]
        
        # Use Overseer for auto-routing if available
        if self._overseer is not None:
            try:
                classification = self._overseer.classify(y, x)
                # Store classification for later use (e.g., confidence reporting)
                self._last_classification = classification
                return classification.best_model
            except Exception as e:
                # Fall back to simple heuristics on error
                pass
        
        # Fallback: Simple heuristics (legacy behavior)
        return self._select_model_fallback(y, x)
    
    def _select_model_fallback(
        self,
        y: np.ndarray,
        x: np.ndarray
    ) -> str:
        """Fallback model selection using simple heuristics (when Overseer unavailable)."""
        n = len(y)
        if x is None:
            x = np.arange(n, dtype=np.float64)
        
        # Check for oscillatory behavior (suggests ODE or trig)
        zero_crossings = np.sum(np.diff(np.sign(y - np.mean(y))) != 0)
        is_oscillatory = zero_crossings > n / 10
        
        # Check for exponential behavior
        if np.all(y > 0):
            log_y = np.log(y + 1e-10)
            log_linearity = np.corrcoef(x, log_y)[0, 1] ** 2
            is_exponential = log_linearity > 0.95
        else:
            is_exponential = False
        
        # Check for polynomial behavior
        poly_fit = np.polyfit(x, y, 3)
        poly_pred = np.polyval(poly_fit, x)
        poly_error = np.mean((y - poly_pred) ** 2) / (np.var(y) + 1e-8)
        is_polynomial = poly_error < 0.1
        
        # Decision logic
        if is_oscillatory:
            return "linear_ode"
        elif is_exponential:
            return "transcendental"
        elif is_polynomial:
            return "polynomial"
        else:
            return "polynomial"  # Default fallback
    
    def _load_model(
        self, 
        model_name: str, 
        tier: int
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Load a model from checkpoint, with caching."""
        
        if torch is None or SymbolicTransformer is None:
            return None, None

        cache_key = f"{model_name}_tier{tier}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        if model_name not in CONFIGS:
            return None, None
        
        cfg = CONFIGS[model_name]
        domain = DOMAINS[model_name]
        
        # Try tier-specific checkpoint
        checkpoint_path = self.checkpoint_dir / domain / model_name / f"tier{tier}" / "model_best.pt"
        
        # Fallback to legacy path
        if not checkpoint_path.exists():
            checkpoint_path = self.checkpoint_dir / domain / model_name / "model_best.pt"
        
        if not checkpoint_path.exists():
            return None, None
        
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load vocab
            vocab_path = checkpoint_path.parent / "vocab.json"
            if vocab_path.exists():
                vocab = Vocabulary.load(vocab_path)
            elif "vocab_tokens" in ckpt:
                vocab = Vocabulary(ckpt["vocab_tokens"])
            else:
                vocab = Vocabulary.default()
            
            # Build model
            model = SymbolicTransformer(vocab_size=vocab.vocab_size, **cfg).to(self.device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            
            # Cache it
            self._model_cache[cache_key] = (model, vocab)
            
            return model, vocab
            
        except Exception as e:
            print(f"Warning: Failed to load {model_name}: {e}")
            return None, None
    
    def _compute_features(
        self,
        y: np.ndarray,
        x: np.ndarray,
        model_name: str
    ) -> np.ndarray:
        """Compute the feature vector for the given data and model type."""
        domain = DOMAINS.get(model_name, "algebraic")

        if domain == "timeseries":
            return compute_timeseries_features(y)
        elif domain == "dynamical":
            return compute_dynamical_features(y, t=x)
        elif domain == "multivariate":
            # For physics laws, we need structured data.
            # This is a simplified version using just output.
            return compute_multivariate_features({"output": y})
        else:
            # Algebraic (polynomial, transcendental)
            return compute_algebraic_features(x, y)

    def _tokens_to_equation(self, tokens: List[str], model_name: str) -> str:
        """
        Render the model's raw token prediction as a human-readable string.

        This is the prediction in the model's own (normalised) coordinates;
        discover_equation reports the equation in original units separately.
        """
        tokens = [t for t in tokens if t not in ["PAD", "BOS", "EOS", "UNK"]]
        if not tokens:
            return "0"

        domain = DOMAINS.get(model_name, "algebraic")
        try:
            if domain == "dynamical":
                params = parse_ode_tokens(tokens)
                return format_ode_equation(params) if params else " ".join(tokens)
            if domain == "timeseries":
                return format_ar_equation(parse_ar_tokens(tokens))
            if domain == "algebraic":
                return expression_to_string(tokens_to_sympy(tokens, model_name))
        except ExpressionParseError:
            pass
        return tokens_to_infix(tokens)

    def _estimate_confidence(self, features: np.ndarray, tokens: List[str]) -> float:
        """
        Estimate confidence in the prediction.
        
        This is a heuristic based on:
        - Feature quality (non-zero, finite values)
        - Token sequence validity
        """
        # Check feature quality
        n_finite = np.sum(np.isfinite(features))
        n_nonzero = np.sum(np.abs(features) > 1e-6)
        feature_quality = (n_finite / len(features)) * (n_nonzero / len(features))
        
        # Check token validity
        valid_tokens = [t for t in tokens if t not in ["PAD", "BOS", "EOS", "UNK"]]
        if len(valid_tokens) == 0:
            token_quality = 0.0
        else:
            # Penalize very short or very long sequences
            length_penalty = 1.0 - abs(len(valid_tokens) - 10) / 20
            length_penalty = max(0.5, min(1.0, length_penalty))
            token_quality = length_penalty
        
        # Combine
        confidence = 0.5 * feature_quality + 0.5 * token_quality
        return float(np.clip(confidence, 0.0, 1.0))
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def list_available_models(self) -> Dict[str, List[int]]:
        """List all models with available checkpoints and their tiers."""
        if not _DISCOVERY_AVAILABLE:
            return {}
        
        available = {}
        
        for model_name in CONFIGS.keys():
            domain = DOMAINS.get(model_name)
            if domain is None:
                continue
            
            model_dir = self.checkpoint_dir / domain / model_name
            if not model_dir.exists():
                continue
            
            tiers = []
            
            # Check tier directories
            for tier in [1, 2, 3]:
                tier_dir = model_dir / f"tier{tier}"
                if (tier_dir / "model_best.pt").exists():
                    tiers.append(tier)
            
            # Check legacy (non-tier) checkpoint
            if (model_dir / "model_best.pt").exists() and 1 not in tiers:
                tiers.append(1)  # Treat legacy as tier 1
            
            if tiers:
                available[model_name] = sorted(tiers)
        
        return available
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of GalileoTools including Overseer availability."""
        return {
            "discovery_available": _DISCOVERY_AVAILABLE,
            "overseer_available": _OVERSEER_AVAILABLE,
            "overseer_enabled": self._overseer is not None,
            "device": str(self.device),
            "default_tier": self.default_tier,
            "checkpoint_dir": str(self.checkpoint_dir),
            "available_models": self.list_available_models(),
            "cached_models": list(self._model_cache.keys()),
        }
    
    def classify_data(
        self,
        y: List[float],
        x: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Classify data without running inference - just get model recommendations.
        
        Args:
            y: Output values
            x: Input values (optional)
            
        Returns:
            Classification results from Overseer
        """
        if self._overseer is None:
            return {"error": "Overseer not available"}
        
        try:
            y_arr = np.array(y, dtype=np.float32)
            x_arr = np.array(x, dtype=np.float32) if x is not None else None
            
            classification = self._overseer.classify(y_arr, x_arr)
            
            return {
                "success": True,
                "data_type": classification.data_type.value,
                "best_model": classification.best_model,
                "best_confidence": float(classification.best_confidence),
                "rankings": [(m, float(c)) for m, c in classification.rankings],
                "reasoning": classification.reasoning,
            }
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
    
    def get_schemas(self, format: str = "openai") -> List[Dict]:
        """
        Get tool schemas in specified format.
        
        Args:
            format: "openai" (default), "anthropic", or "raw"
        """
        if format == "anthropic":
            return ANTHROPIC_TOOL_SCHEMAS
        elif format == "raw":
            return [s["function"] for s in TOOL_SCHEMAS]
        else:
            return TOOL_SCHEMAS


# =============================================================================
# Convenience Functions
# =============================================================================

_default_tools = None

def get_tools() -> GalileoTools:
    """Get or create default GalileoTools instance."""
    global _default_tools
    if _default_tools is None:
        _default_tools = GalileoTools()
    return _default_tools


def synthesize_data(**kwargs) -> Dict[str, Any]:
    """Quick access to synthesize_data tool."""
    return get_tools().synthesize_data(**kwargs)


def discover_equation(**kwargs) -> Dict[str, Any]:
    """Quick access to discover_equation tool."""
    return get_tools().discover_equation(**kwargs)


def verify_equation(**kwargs) -> Dict[str, Any]:
    """Quick access to verify_equation tool."""
    return get_tools().verify_equation(**kwargs)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Galileo Tools - LLM Interface")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--schemas", choices=["openai", "anthropic", "raw"], 
                       help="Print tool schemas in specified format")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    
    args = parser.parse_args()
    
    tools = GalileoTools()
    
    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 40)
        available = tools.list_available_models()
        if available:
            for model, tiers in available.items():
                print(f"  {model:20s} tiers: {tiers}")
        else:
            print("  No trained models found.")
            print("  Train a model first: python train.py --model polynomial --tier 1")
        
    elif args.schemas:
        schemas = tools.get_schemas(args.schemas)
        print(json.dumps(schemas, indent=2))
        
    elif args.demo:
        print("=" * 70)
        print("Galileo Tools Demo")
        print("=" * 70)
        
        # Demo 1: Synthesize data
        print("\n1. Synthesizing data from equation: x**2 + 2*x - 1")
        result = tools.synthesize_data(equation="x**2 + 2*x - 1", n_points=5)
        if result.get("success"):
            print(f"   x: {result['x']}")
            print(f"   y: {result['y']}")
        else:
            print(f"   Error: {result.get('error')}")
        
        # Demo 2: Synthesize from description
        print("\n2. Synthesizing from description: 'exponential decay'")
        result = tools.synthesize_data(description="exponential decay", n_points=5)
        if result.get("success"):
            print(f"   x: {[f'{v:.2f}' for v in result['x']]}")
            print(f"   y: {[f'{v:.4f}' for v in result['y']]}")
        
        # Demo 3: Synthesize ODE
        print("\n3. Synthesizing ODE: x'' + 4*x = 0")
        result = tools.synthesize_data(ode="x'' + 4*x = 0", n_points=5)
        if result.get("success"):
            print(f"   t: {[f'{v:.2f}' for v in result['x']]}")
            print(f"   x: {[f'{v:.4f}' for v in result['y']]}")
        
        # Demo 4: Verify equation
        print("\n4. Verifying equation x**2 against [1, 4, 9, 16, 25]")
        result = tools.verify_equation(
            equation="x**2",
            x=[1, 2, 3, 4, 5],
            y=[1, 4, 9, 16, 25]
        )
        print(f"   Match: {result.get('match')}")
        print(f"   RMSE: {result.get('rmse', 'N/A')}")
        
        # Demo 5: Discover equation (requires trained model)
        print("\n5. Discovering equation from [1, 4, 9, 16, 25]")
        result = tools.discover_equation(
            x=[1, 2, 3, 4, 5],
            y=[1, 4, 9, 16, 25]
        )
        if result.get("success"):
            print(f"   Discovered: {result['equation']}")
            print(f"   Model: {result['model_used']}")
            print(f"   Confidence: {result['confidence']:.2f}")
        else:
            print(f"   {result.get('error')}")
            print("   (Train a model first: python train.py --model polynomial --tier 1)")
        
        print("\n" + "=" * 70)
        print("Use --schemas openai to get tool definitions for your LLM")
        print("=" * 70)
    
    else:
        parser.print_help()

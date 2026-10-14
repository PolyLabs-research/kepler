#!/usr/bin/env python3
"""
overseer.py - Overseer Router for Galileo Minimal

Automatically classifies incoming data and routes to the appropriate
specialist model(s). The overseer examines data characteristics and
returns ranked model recommendations with confidence scores.

Architecture:
    Raw Data → Feature Analysis → Classification → Model Ranking → Specialist Inference

The overseer uses a combination of:
1. Heuristic rules (fast, interpretable)
2. Statistical signatures (derivative patterns, periodicity, etc.)
3. Optional learned classifier (can be trained on labeled examples)

Usage:
    from overseer import Overseer
    
    overseer = Overseer()
    
    # Get model recommendations
    ranking = overseer.classify(x, y)
    # Returns: [('polynomial', 0.85), ('transcendental', 0.12), ...]
    
    # Auto-route to best model
    result = overseer.route(x, y)
    # Returns: {'model': 'polynomial', 'equation': '...', 'confidence': 0.85}
    
    # Route with ensemble (try top-k models)
    result = overseer.route_ensemble(x, y, top_k=3)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Suppress numpy warnings for edge cases
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DataType(Enum):
    """Supported data types for routing."""
    UNIVARIATE = "univariate"      # (x, y) pairs
    TIMESERIES = "timeseries"      # y[t] sequence
    MULTIVARIATE = "multivariate"  # {var1: [...], var2: [...], output: [...]}
    ODE_TRAJECTORY = "ode"         # Time series from dynamical system


@dataclass
class ClassificationResult:
    """Result of data classification."""
    rankings: List[Tuple[str, float]]  # [(model_name, confidence), ...]
    data_type: DataType
    features: Dict[str, float]  # Computed diagnostic features
    reasoning: List[str]  # Human-readable explanation

    @property
    def best_model(self) -> str:
        return self.rankings[0][0] if self.rankings else "polynomial"

    @property
    def best_confidence(self) -> float:
        return self.rankings[0][1] if self.rankings else 0.0

    @property
    def confidence_margin(self) -> float:
        """Difference between top-1 and top-2 confidence."""
        if len(self.rankings) < 2:
            return self.best_confidence
        return float(self.rankings[0][1] - self.rankings[1][1])

    @property
    def is_ambiguous(self) -> bool:
        """Low-margin routing should be treated as ambiguous."""
        return self.confidence_margin < 0.08

    def top_k(self, k: int = 3) -> List[Tuple[str, float]]:
        return self.rankings[:k]


@dataclass 
class RoutingResult:
    """Result of routing to a specialist model."""
    model: str
    equation: str
    confidence: float
    tokens: List[str]
    classification: ClassificationResult
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


class Overseer:
    """
    Overseer Router - classifies data and routes to specialist models.
    
    The overseer uses a multi-stage classification pipeline:
    
    Stage 1: Data Type Detection
        - Univariate (x,y pairs) vs Timeseries vs Multivariate
        
    Stage 2: Structural Analysis
        - Monotonicity, periodicity, boundedness
        - Derivative patterns
        - Growth/decay characteristics
        
    Stage 3: Model Scoring
        - Each model gets a compatibility score
        - Scores normalized to probabilities
        
    Stage 4: Ranking & Routing
        - Models ranked by score
        - Top model(s) used for inference
    """
    
    # Model capabilities and characteristics
    MODEL_INFO = {
        "polynomial": {
            "data_type": DataType.UNIVARIATE,
            "feature_dim": 48,
            "characteristics": ["smooth", "non_periodic", "polynomial_residual"],
            "subtypes": ["linear", "quadratic", "cubic", "quartic"],
        },
        "transcendental": {
            "data_type": DataType.UNIVARIATE,
            "feature_dim": 48,
            "characteristics": ["smooth", "non_periodic", "exponential_or_log"],
            "subtypes": ["exp_growth", "exp_decay", "power", "logarithmic", "logistic"],
        },
        "linear_ode": {
            "data_type": DataType.ODE_TRAJECTORY,
            "feature_dim": 64,
            "characteristics": ["smooth", "may_oscillate", "decaying_or_growing"],
            "subtypes": ["first_order", "underdamped", "overdamped", "undamped"],
        },
        "ar": {
            "data_type": DataType.TIMESERIES,
            "feature_dim": 32,
            "characteristics": ["stochastic", "autocorrelated", "stationary"],
            "subtypes": ["ar1", "ar2", "ar3"],
        },
        "physics_laws": {
            "data_type": DataType.MULTIVARIATE,
            "feature_dim": 64,
            "characteristics": ["multivariate", "deterministic", "power_law"],
            "subtypes": ["inverse_square", "ideal_gas", "kinetic", "linear"],
        },
    }
    
    def __init__(self, models_dir: str = "checkpoints", default_tier: int = 2):
        """
        Initialize the Overseer.
        
        Args:
            models_dir: Directory containing trained model checkpoints
            default_tier: Default tier to use for inference (1, 2, or 3)
        """
        self.models_dir = models_dir
        self.default_tier = default_tier
        self._inference_cache = {}
        
    def classify(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
    ) -> ClassificationResult:
        """
        Classify data and return ranked model recommendations.
        
        Args:
            y: Output values (or timeseries if x is None)
            x: Input values (optional for timeseries)
            data: Multivariate data dict (optional, overrides x/y)
            
        Returns:
            ClassificationResult with ranked models and diagnostic info
        """
        # Stage 1: Detect data type
        data_type, reasoning = self._detect_data_type(y, x, data)
        
        # Stage 2: Extract diagnostic features
        features = self._extract_diagnostic_features(y, x, data, data_type)
        
        # Stage 3: Score each model
        scores = self._score_models(features, data_type, reasoning)
        
        # Stage 4: Normalize and rank
        rankings = self._normalize_and_rank(scores)
        
        return ClassificationResult(
            rankings=rankings,
            data_type=data_type,
            features=features,
            reasoning=reasoning,
        )
    
    def route(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
        tier: Optional[int] = None,
    ) -> RoutingResult:
        """
        Classify data and route to the best specialist model.
        
        Args:
            y: Output values
            x: Input values (optional)
            data: Multivariate data (optional)
            tier: Model tier to use (default: self.default_tier)
            
        Returns:
            RoutingResult with equation and metadata
        """
        classification = self.classify(y, x, data)
        tier = tier or self.default_tier
        
        # Route to best model
        best_model = classification.best_model
        equation, tokens = self._run_inference(best_model, y, x, data, tier)
        confidence = classification.best_confidence

        # Ambiguity fallback: if margin is low, evaluate top-2 by fit score.
        if classification.is_ambiguous and len(classification.rankings) >= 2:
            candidates = []
            for model_name, conf in classification.top_k(2):
                eq_i, tok_i = self._run_inference(model_name, y, x, data, tier)
                fit_i = self._verify_fit(eq_i, y, x, data)
                candidates.append((model_name, eq_i, tok_i, conf, fit_i, 0.45 * conf + 0.55 * fit_i))

            candidates.sort(key=lambda t: t[5], reverse=True)
            top = candidates[0]
            best_model, equation, tokens = top[0], top[1], top[2]
            confidence = top[5]

        return RoutingResult(
            model=best_model,
            equation=equation,
            confidence=confidence,
            tokens=tokens,
            classification=classification,
        )
    
    def route_ensemble(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
        tier: Optional[int] = None,
        top_k: int = 3,
    ) -> RoutingResult:
        """
        Route to top-k models and return best result based on verification.
        
        Args:
            y: Output values
            x: Input values (optional)
            data: Multivariate data (optional)
            tier: Model tier to use
            top_k: Number of models to try
            
        Returns:
            RoutingResult with best equation and alternatives
        """
        classification = self.classify(y, x, data)
        tier = tier or self.default_tier
        
        results = []
        for model_name, conf in classification.top_k(top_k):
            try:
                equation, tokens = self._run_inference(model_name, y, x, data, tier)
                
                # Verify fit quality
                fit_score = self._verify_fit(equation, y, x, data)
                
                results.append({
                    "model": model_name,
                    "equation": equation,
                    "tokens": tokens,
                    "classification_confidence": conf,
                    "fit_score": fit_score,
                    "combined_score": 0.5 * conf + 0.5 * fit_score,
                })
            except Exception as e:
                results.append({
                    "model": model_name,
                    "equation": f"Error: {e}",
                    "tokens": [],
                    "classification_confidence": conf,
                    "fit_score": 0.0,
                    "combined_score": 0.0,
                })
        
        # Sort by combined score
        results.sort(key=lambda r: r["combined_score"], reverse=True)
        
        best = results[0]
        return RoutingResult(
            model=best["model"],
            equation=best["equation"],
            confidence=best["combined_score"],
            tokens=best["tokens"],
            classification=classification,
            alternatives=results[1:],
        )
    
    # =========================================================================
    # Stage 1: Data Type Detection
    # =========================================================================
    
    def _detect_data_type(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray],
        data: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[DataType, List[str]]:
        """Detect the type of input data."""
        reasoning = []
        
        # Multivariate check
        if data is not None and len(data) > 2:
            reasoning.append(f"Multivariate data with {len(data)} variables")
            return DataType.MULTIVARIATE, reasoning
        
        # Timeseries vs univariate
        if x is None:
            reasoning.append("No x values provided - treating as timeseries")
            return DataType.TIMESERIES, reasoning
        
        # Check if x looks like time (evenly spaced, starting near 0)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        dx = np.diff(x)
        is_uniform = np.std(dx) / (np.mean(dx) + 1e-10) < 0.01
        starts_near_zero = abs(x[0]) < 1.0
        
        if is_uniform and starts_near_zero:
            # Could be ODE trajectory - check for oscillation/decay patterns
            zero_crossings = np.sum(np.diff(np.sign(y - np.mean(y))) != 0)
            has_oscillation = zero_crossings > 4
            
            # Check for exponential envelope
            abs_y = np.abs(y - np.mean(y))
            peaks_idx = []
            for i in range(1, len(abs_y) - 1):
                if abs_y[i] > abs_y[i-1] and abs_y[i] > abs_y[i+1]:
                    peaks_idx.append(i)
            
            if len(peaks_idx) >= 4 and has_oscillation:
                peak_vals = abs_y[peaks_idx]
                # Check if peaks decay and avoid noise-only false positives
                if peak_vals[0] > peak_vals[-1] * 1.15 and np.std(y_centered := (y - np.mean(y))) > 1e-8:
                    reasoning.append("Detected decaying oscillation - likely ODE")
                    return DataType.ODE_TRAJECTORY, reasoning

            # Strong oscillation signal only
            if has_oscillation and zero_crossings >= max(6, len(y) // 30):
                reasoning.append("Detected oscillatory pattern - likely ODE")
                return DataType.ODE_TRAJECTORY, reasoning

            # Pure exponential-like curves are ambiguous; keep as UNIVARIATE and let scoring decide
            if self._check_exponential_pattern(y):
                reasoning.append("Detected exponential pattern - routing as univariate (transcendental/ODE overlap)")
        
        reasoning.append("Standard (x, y) data - univariate function")
        return DataType.UNIVARIATE, reasoning
    
    def _check_exponential_pattern(self, y: np.ndarray) -> bool:
        """Check if y follows exponential growth/decay."""
        y_pos = y[y > 0]
        if len(y_pos) < 10:
            return False
        
        log_y = np.log(y_pos + 1e-10)
        x_temp = np.arange(len(y_pos))
        
        try:
            coeffs = np.polyfit(x_temp, log_y, 1)
            residual = np.std(log_y - np.polyval(coeffs, x_temp))
            return residual < 0.5  # Good linear fit in log space
        except:
            return False
    
    # =========================================================================
    # Stage 2: Diagnostic Feature Extraction
    # =========================================================================
    
    def _extract_diagnostic_features(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray],
        data: Optional[Dict[str, np.ndarray]],
        data_type: DataType,
    ) -> Dict[str, float]:
        """Extract features useful for model classification."""
        features = {}
        
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        
        # Basic statistics
        features["n_points"] = n
        features["y_mean"] = float(np.mean(y))
        features["y_std"] = float(np.std(y))
        features["y_range"] = float(np.max(y) - np.min(y))
        features["y_min"] = float(np.min(y))
        features["y_max"] = float(np.max(y))
        
        # Monotonicity
        dy = np.diff(y)
        features["frac_increasing"] = float(np.mean(dy > 0))
        features["frac_decreasing"] = float(np.mean(dy < 0))
        features["is_monotonic"] = float(features["frac_increasing"] > 0.95 or 
                                         features["frac_decreasing"] > 0.95)
        
        # Zero crossings (centered)
        y_centered = y - np.mean(y)
        zero_crossings = np.sum(np.diff(np.sign(y_centered)) != 0)
        features["zero_crossings"] = int(zero_crossings)
        features["crossing_rate"] = float(zero_crossings / n)
        
        # Periodicity detection via FFT
        if n >= 16:
            spectrum = np.abs(np.fft.rfft(y_centered))
            freqs = np.fft.rfftfreq(n)
            
            # Skip DC component
            if len(spectrum) > 1:
                peak_idx = np.argmax(spectrum[1:]) + 1
                features["dominant_freq"] = float(freqs[peak_idx])
                features["spectral_peak_ratio"] = float(
                    spectrum[peak_idx] / (np.sum(spectrum[1:]) + 1e-10)
                )
            else:
                features["dominant_freq"] = 0.0
                features["spectral_peak_ratio"] = 0.0
        else:
            features["dominant_freq"] = 0.0
            features["spectral_peak_ratio"] = 0.0
        
        # Autocorrelation (for AR detection)
        if n >= 20:
            y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
            acf_1 = float(np.mean(y_norm[1:] * y_norm[:-1]))
            acf_2 = float(np.mean(y_norm[2:] * y_norm[:-2])) if n > 2 else 0.0
            features["acf_lag1"] = acf_1
            features["acf_lag2"] = acf_2
            features["acf_decay"] = float(acf_1 - acf_2) if acf_1 > 0 else 0.0
        else:
            features["acf_lag1"] = 0.0
            features["acf_lag2"] = 0.0
            features["acf_decay"] = 0.0
        
        # Derivative-based features (if x available)
        if x is not None:
            x = np.asarray(x, dtype=np.float64)
            dx = np.diff(x)
            dx[dx == 0] = 1e-10  # Avoid division by zero
            
            dy_dx = np.diff(y) / dx
            features["deriv_mean"] = float(np.mean(dy_dx))
            features["deriv_std"] = float(np.std(dy_dx))
            
            if len(dy_dx) > 1:
                d2y_dx2 = np.diff(dy_dx) / dx[:-1]
                features["deriv2_mean"] = float(np.mean(d2y_dx2))
                features["deriv2_std"] = float(np.std(d2y_dx2))
            else:
                features["deriv2_mean"] = 0.0
                features["deriv2_std"] = 0.0
            
            # Polynomial fit residuals
            for deg in [1, 2, 3]:
                try:
                    coeffs = np.polyfit(x, y, deg)
                    residual = np.std(y - np.polyval(coeffs, x))
                    features[f"poly{deg}_residual"] = float(residual)
                except:
                    features[f"poly{deg}_residual"] = float('inf')
            
            # Log-linear fit (for exponential detection)
            if np.all(y > 0):
                try:
                    log_coeffs = np.polyfit(x, np.log(y), 1)
                    log_residual = np.std(np.log(y) - np.polyval(log_coeffs, x))
                    features["log_linear_residual"] = float(log_residual)
                except:
                    features["log_linear_residual"] = float('inf')
            else:
                features["log_linear_residual"] = float('inf')

            # Logistic-like signature: bounded S-curve with inflection
            if n >= 30:
                y_rng = max(features["y_range"], 1e-10)
                y_sorted = np.sort(y)
                low_band = max(3, n // 10)
                lo = np.mean(y_sorted[:low_band])
                hi = np.mean(y_sorted[-low_band:])
                span = max(hi - lo, 1e-10)

                # Normalize to [0,1]-ish and look for saturation at both ends
                y_norm = (y - lo) / span
                lower_sat = float(np.mean(y_norm < 0.1))
                upper_sat = float(np.mean(y_norm > 0.9))
                features["lower_saturation_ratio"] = lower_sat
                features["upper_saturation_ratio"] = upper_sat

                # Inflection proxy: derivative peaks in the middle for logistic curves
                grad = np.gradient(y, x)
                seg = n // 3
                g_early = float(np.mean(np.abs(grad[:seg])))
                g_mid = float(np.mean(np.abs(grad[seg:2*seg])))
                g_late = float(np.mean(np.abs(grad[2*seg:])))
                features["mid_slope_ratio"] = float(g_mid / (0.5 * (g_early + g_late) + 1e-10))

                logistic_like = (
                    features["is_monotonic"] > 0.85
                    and lower_sat > 0.08
                    and upper_sat > 0.08
                    and features["mid_slope_ratio"] > 1.25
                )
                features["logistic_signature"] = float(logistic_like)
            else:
                features["lower_saturation_ratio"] = 0.0
                features["upper_saturation_ratio"] = 0.0
                features["mid_slope_ratio"] = 0.0
                features["logistic_signature"] = 0.0
        
        # Envelope decay (for ODE detection)
        abs_y = np.abs(y_centered)
        peaks = []
        for i in range(1, n - 1):
            if abs_y[i] > abs_y[i-1] and abs_y[i] > abs_y[i+1] and abs_y[i] > 0.01 * features["y_range"]:
                peaks.append((i, abs_y[i]))
        
        if len(peaks) >= 3:
            peak_indices = [p[0] for p in peaks]
            peak_values = np.array([p[1] for p in peaks])
            try:
                log_peaks = np.log(peak_values + 1e-10)
                decay_rate, _ = np.polyfit(peak_indices, log_peaks, 1)
                features["envelope_decay_rate"] = float(decay_rate)
            except:
                features["envelope_decay_rate"] = 0.0
        else:
            features["envelope_decay_rate"] = 0.0
        
        # Multivariate features
        if data is not None:
            features["n_variables"] = len(data)
            
            # Check for inverse-square relationship
            for key in data:
                if key.lower() in ['r', 'distance', 'radius']:
                    r = data[key]
                    output = data.get('output', y)
                    try:
                        # Check if output ~ 1/r^2
                        r_inv_sq = 1 / (r ** 2 + 1e-10)
                        corr = np.corrcoef(r_inv_sq, output)[0, 1]
                        features["inverse_square_corr"] = float(corr) if np.isfinite(corr) else 0.0
                    except:
                        features["inverse_square_corr"] = 0.0
                    break
        
        return features
    
    # =========================================================================
    # Stage 3: Model Scoring
    # =========================================================================
    
    def _score_models(
        self,
        features: Dict[str, float],
        data_type: DataType,
        reasoning: List[str],
    ) -> Dict[str, float]:
        """Score each model based on features and data type."""
        scores = {}
        
        # Base scores from data type compatibility
        type_scores = {
            DataType.UNIVARIATE: {
                "polynomial": 0.5, "transcendental": 0.5, "linear_ode": 0.2,
                "ar": 0.1, "physics_laws": 0.1,
            },
            DataType.TIMESERIES: {
                "polynomial": 0.1, "transcendental": 0.1, "linear_ode": 0.3,
                "ar": 0.8, "physics_laws": 0.1,
            },
            DataType.ODE_TRAJECTORY: {
                "polynomial": 0.1, "transcendental": 0.2, "linear_ode": 0.8,
                "ar": 0.2, "physics_laws": 0.1,
            },
            DataType.MULTIVARIATE: {
                "polynomial": 0.1, "transcendental": 0.1, "linear_ode": 0.1,
                "ar": 0.1, "physics_laws": 0.8,
            },
        }
        
        for model, base_score in type_scores[data_type].items():
            scores[model] = base_score
        
        # Adjust based on features
        
        # Polynomial scoring
        if "poly1_residual" in features and "poly2_residual" in features:
            poly_improvement = features["poly1_residual"] - features["poly2_residual"]
            # Avoid over-rewarding polynomial when exponential/log fits are strong
            exp_like = features.get("log_linear_residual", float('inf')) < 0.2 and features.get("is_monotonic", 0)

            if poly_improvement > 0.1 * features.get("y_std", 1) and not exp_like:
                scores["polynomial"] += 0.22
                reasoning.append("Quadratic improves over linear - polynomial likely")

            if (
                features.get("poly3_residual", float('inf')) < 0.05 * features.get("y_std", 1)
                and not exp_like
            ):
                scores["polynomial"] += 0.12
                reasoning.append("Cubic fits well")
        
        # Transcendental scoring
        if features.get("log_linear_residual", float('inf')) < 0.1:
            scores["transcendental"] += 0.4
            reasoning.append("Good log-linear fit - exponential likely")
        
        if features.get("is_monotonic", 0) and features.get("y_min", 0) > 0:
            scores["transcendental"] += 0.2
            reasoning.append("Monotonic positive data - could be exponential/power")

        if (
            features.get("is_monotonic", 0)
            and features.get("log_linear_residual", float('inf')) < 0.25
            and features.get("zero_crossings", 0) < 3
        ):
            scores["transcendental"] += 0.45
            reasoning.append("Monotonic log-linear pattern with low oscillation - transcendental favored")

        if features.get("logistic_signature", 0) > 0.5:
            scores["transcendental"] += 0.55
            scores["polynomial"] -= 0.25
            reasoning.append("Saturation + inflection signature detected - logistic/transcendental favored")
        
        # ODE scoring (gate on oscillation evidence)
        zc = features.get("zero_crossings", 0)
        spr = features.get("spectral_peak_ratio", 0)
        env_decay = abs(features.get("envelope_decay_rate", 0))

        if zc >= 6:
            scores["linear_ode"] += 0.25
            reasoning.append("Multiple zero crossings - oscillatory behavior")

        if spr > 0.45:
            scores["linear_ode"] += 0.2
            reasoning.append("Strong spectral peak - periodic component")

        # Only boost damping when oscillation evidence exists
        if env_decay > 0.01 and zc >= 4:
            scores["linear_ode"] += 0.25
            reasoning.append("Decaying oscillatory envelope detected")

        # Penalize ODE for smooth monotonic curves without oscillation
        if features.get("is_monotonic", 0) and zc < 3 and spr < 0.2:
            scores["linear_ode"] -= 0.25
            reasoning.append("Monotonic non-oscillatory curve - weak ODE evidence")

        # AR scoring (avoid over-trigger on smooth deterministic curves)
        acf1 = abs(features.get("acf_lag1", 0))
        acf_decay = features.get("acf_decay", 0)
        monotonic = features.get("is_monotonic", 0)

        if acf1 > 0.7:
            scores["ar"] += 0.25
            reasoning.append(f"High lag-1 autocorrelation ({features['acf_lag1']:.2f})")

        if acf_decay > 0 and features.get("acf_lag1", 0) > 0:
            scores["ar"] += 0.15
            reasoning.append("Decaying autocorrelation - AR signature")

        # Penalize AR for smooth deterministic-like trends
        if monotonic and features.get("deriv_std", 0) < max(1e-6, 0.15 * features.get("y_std", 1.0)):
            scores["ar"] -= 0.2
            reasoning.append("Smooth monotonic trend - weak AR evidence")
        
        # Physics laws scoring
        if features.get("n_variables", 1) > 2:
            scores["physics_laws"] += 0.4
            reasoning.append(f"Multivariate data ({features['n_variables']} variables)")
        
        if features.get("inverse_square_corr", 0) > 0.9:
            scores["physics_laws"] += 0.3
            reasoning.append("Strong inverse-square relationship detected")
        
        return scores
    
    # =========================================================================
    # Stage 4: Normalization and Ranking
    # =========================================================================
    
    def _normalize_and_rank(
        self,
        scores: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """Normalize scores to probabilities and rank models."""
        # Softmax normalization
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        
        probabilities = {k: v / total for k, v in exp_scores.items()}
        
        # Sort by probability
        rankings = sorted(probabilities.items(), key=lambda x: -x[1])
        
        return rankings
    
    # =========================================================================
    # Inference
    # =========================================================================
    
    def _run_inference(
        self,
        model_name: str,
        y: np.ndarray,
        x: Optional[np.ndarray],
        data: Optional[Dict[str, np.ndarray]],
        tier: int,
    ) -> Tuple[str, List[str]]:
        """Run inference on a specialist model."""
        try:
            # Import here to avoid circular dependencies
            from galileo_tools import GalileoTools
            
            tools = GalileoTools()
            
            # Prepare data for discovery
            if data is not None:
                result = tools.discover_equation(data=data, data_type=model_name, tier=tier)
            elif x is not None:
                result = tools.discover_equation(x=x.tolist(), y=y.tolist(), 
                                                  data_type=model_name, tier=tier)
            else:
                result = tools.discover_equation(y=y.tolist(), data_type=model_name, tier=tier)
            
            if "error" in result:
                return f"Error: {result['error']}", []
            
            return result.get("equation", "Unknown"), result.get("tokens", [])
            
        except Exception as e:
            return f"Inference error: {e}", []
    
    def _verify_fit(
        self,
        equation: str,
        y: np.ndarray,
        x: Optional[np.ndarray],
        data: Optional[Dict[str, np.ndarray]],
    ) -> float:
        """Verify how well an equation fits the data. Returns score 0-1."""
        try:
            from galileo_tools import GalileoTools
            
            tools = GalileoTools()
            
            if x is not None:
                result = tools.verify_equation(
                    x=x.tolist(), y=y.tolist(), equation=equation
                )
            else:
                result = tools.verify_equation(
                    y=y.tolist(), equation=equation
                )
            
            if result.get("match", False):
                return 1.0 - min(result.get("relative_error", 0), 1.0)
            else:
                return max(0.0, 0.5 - result.get("relative_error", 1.0))
                
        except Exception:
            return 0.0
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def available_models(self) -> List[str]:
        """Return list of available models."""
        return list(self.MODEL_INFO.keys())
    
    def model_info(self, model_name: str) -> Dict[str, Any]:
        """Return info about a specific model."""
        return self.MODEL_INFO.get(model_name, {})
    
    def explain_classification(self, result: ClassificationResult) -> str:
        """Generate human-readable explanation of classification."""
        lines = [
            f"Data Type: {result.data_type.value}",
            f"Best Model: {result.best_model} (confidence: {result.best_confidence:.1%})",
            "",
            "Reasoning:",
        ]
        for reason in result.reasoning:
            lines.append(f"  • {reason}")
        
        lines.extend([
            "",
            "Model Rankings:",
        ])
        for model, conf in result.rankings:
            bar = "█" * int(conf * 20)
            lines.append(f"  {model:15s} {bar} {conf:.1%}")
        
        lines.extend([
            "",
            "Key Features:",
        ])
        important_features = [
            "zero_crossings", "acf_lag1", "envelope_decay_rate",
            "poly2_residual", "log_linear_residual", "spectral_peak_ratio"
        ]
        for feat in important_features:
            if feat in result.features:
                lines.append(f"  {feat}: {result.features[feat]:.4f}")
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

_overseer = None

def get_overseer(**kwargs) -> Overseer:
    """Get or create singleton Overseer instance."""
    global _overseer
    if _overseer is None:
        _overseer = Overseer(**kwargs)
    return _overseer


def classify(y, x=None, data=None) -> ClassificationResult:
    """Convenience function for classification."""
    return get_overseer().classify(y, x, data)


def route(y, x=None, data=None, tier=None) -> RoutingResult:
    """Convenience function for routing."""
    return get_overseer().route(y, x, data, tier)


def route_ensemble(y, x=None, data=None, tier=None, top_k=3) -> RoutingResult:
    """Convenience function for ensemble routing."""
    return get_overseer().route_ensemble(y, x, data, tier, top_k)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Overseer Router for Galileo")
    parser.add_argument("--demo", action="store_true", help="Run demo classification")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    overseer = Overseer()
    
    if args.list_models:
        print("Available Models:")
        print("-" * 60)
        for name, info in overseer.MODEL_INFO.items():
            print(f"  {name:15s} [{info['data_type'].value}]")
            print(f"                  Subtypes: {', '.join(info['subtypes'])}")
        exit(0)
    
    if args.demo or True:  # Default to demo
        print("=" * 70)
        print("Overseer Router Demo")
        print("=" * 70)
        
        # Demo 1: Polynomial
        print("\n1. Quadratic data: y = 2x² + 3x + 1")
        x = np.linspace(-5, 5, 100)
        y = 2*x**2 + 3*x + 1
        result = overseer.classify(y, x)
        print(f"   Best: {result.best_model} ({result.best_confidence:.1%})")
        print(f"   Top 3: {result.top_k(3)}")
        
        # Demo 2: Exponential
        print("\n2. Exponential decay: y = 5·e^(-0.3x)")
        x = np.linspace(0, 10, 100)
        y = 5 * np.exp(-0.3 * x)
        result = overseer.classify(y, x)
        print(f"   Best: {result.best_model} ({result.best_confidence:.1%})")
        print(f"   Top 3: {result.top_k(3)}")
        
        # Demo 3: Damped oscillator
        print("\n3. Damped oscillator: x'' + 0.5x' + 4x = 0")
        t = np.linspace(0, 10, 200)
        omega, zeta = 2.0, 0.125
        omega_d = omega * np.sqrt(1 - zeta**2)
        y = np.exp(-zeta * omega * t) * np.cos(omega_d * t)
        result = overseer.classify(y, t)
        print(f"   Best: {result.best_model} ({result.best_confidence:.1%})")
        print(f"   Top 3: {result.top_k(3)}")
        
        # Demo 4: AR process
        print("\n4. AR(1) process: y[t] = 0.7·y[t-1] + noise")
        np.random.seed(42)
        y = np.zeros(200)
        for i in range(1, 200):
            y[i] = 0.7 * y[i-1] + np.random.normal(0, 0.1)
        result = overseer.classify(y)
        print(f"   Best: {result.best_model} ({result.best_confidence:.1%})")
        print(f"   Top 3: {result.top_k(3)}")
        
        # Demo 5: Detailed explanation
        print("\n5. Detailed classification for damped oscillator:")
        t = np.linspace(0, 10, 200)
        y = np.exp(-0.3 * t) * np.cos(2 * t)
        result = overseer.classify(y, t)
        print(overseer.explain_classification(result))

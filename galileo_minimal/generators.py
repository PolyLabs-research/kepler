"""
generators.py - Primary Registry with Tier-Aware Generators (Phase 3 Complete)
"""
import numpy as np
from generators_base import BaseGenerator, float_to_tokens
from generators_dynamical import LinearODEGenerator
from features import compute_algebraic_features, compute_timeseries_features, compute_multivariate_features

# ============================================================================
# Standard Model Generators with Tier Support
# ============================================================================

class PolynomialGenerator(BaseGenerator):
    SUBTYPES = {
        "linear_positive": 1, 
        "linear_negative": 1, 
        "quadratic_up": 2, 
        "quadratic_down": 2, 
        "cubic": 3, 
        "quartic": 4
    }
    WEIGHTS = {
        "linear_positive": 1.0,
        "linear_negative": 1.0,
        "quadratic_up": 1.0,
        "quadratic_down": 1.0,
        "cubic": 3.0,
        "quartic": 1.0,
    }
    
    def __init__(self, n_points: int = 200, x_range: tuple = (-5, 5)):
        self.n_points = n_points
        self.x_range = x_range
    
    def generate(self, subtype=None, tier: int = 1):
        if subtype is None: 
            subtype = self._sample_subtype()
        deg = self.SUBTYPES[subtype]
        
        # TIER LOGIC
        if tier == 1:
            # Discrete coefficients, snap to integers
            coeffs = np.random.choice([-3, -2, -1, 0, 1, 2, 3], size=deg + 1)
            # Ensure non-zero leading coefficient
            while coeffs[0] == 0:
                coeffs[0] = np.random.choice([-3, -2, -1, 1, 2, 3])
        elif tier == 2:
            # Continuous coefficients, 2dp precision
            coeffs = np.random.uniform(-3, 3, size=deg + 1)
        else:  # tier == 3
            # Wider range, potential for more extreme coefficients
            coeffs = np.random.uniform(-5, 5, size=deg + 1)
            
        # Apply subtype constraints
        if subtype.endswith("_positive"): 
            coeffs[-1] = abs(coeffs[-1])
        elif subtype.endswith("_negative"): 
            coeffs[-1] = -abs(coeffs[-1])
        
        x = np.linspace(*self.x_range, self.n_points)
        y = np.polyval(coeffs[::-1], x)
        
        # Add noise for tier 3
        if tier == 3:
            noise = np.random.normal(0, 0.05 * np.std(y), size=y.shape)
            y = y + noise
        
        tokens = []
        for i, c in enumerate(coeffs):
            if abs(c) < 0.01: continue
            if tokens: 
                tokens.append("ADD" if c > 0 else "SUB")
            elif c < 0: 
                tokens.append("NEG")
            
            c_val = abs(c)
            if abs(c_val - 1.0) > 0.01 or i == 0:
                tokens.extend(self._float_to_tokens(c_val))
                if i > 0: tokens.append("MUL")
            
            if i == 1: 
                tokens.append("X")
            elif i > 1: 
                tokens.extend(["X", "POW", f"INT_{i}"])
            
        return compute_algebraic_features(x, y), tokens or ["INT_0"], subtype


class ARGenerator(BaseGenerator):
    SUBTYPES = {"ar1_low": 1, "ar1_medium": 1, "ar1_high": 1, "ar2": 2, "ar3": 3}
    WEIGHTS = {"ar1_low": 1, "ar1_medium": 2, "ar1_high": 2, "ar2": 1, "ar3": 0.5}
    
    def __init__(self, n_steps: int = 200):
        self.n_steps = n_steps
    
    def generate(self, subtype=None, tier: int = 1):
        if subtype is None: 
            subtype = self._sample_subtype()
        order = self.SUBTYPES[subtype]
        
        # TIER LOGIC
        if tier == 1:
            # Discrete parameter sets
            if order == 1:
                phi = np.array([np.random.choice([0.3, 0.5, 0.7, 0.9])])
            else:
                phi = np.random.choice([-0.3, -0.2, 0.0, 0.2, 0.3], size=order)
        elif tier == 2:
            # Continuous parameters (2dp)
            if order == 1:
                phi = np.random.uniform(0.1, 0.95, size=order)
            else:
                phi = np.random.uniform(-0.3, 0.3, size=order)
        else:  # tier == 3
            # Wider range
            if order == 1:
                phi = np.random.uniform(0.05, 0.98, size=order)
            else:
                phi = np.random.uniform(-0.4, 0.4, size=order)
        
        # Stability constraint
        if order > 1 and np.sum(np.abs(phi)) >= 0.95:
            phi = phi * 0.9 / np.sum(np.abs(phi))
        
        y = np.zeros(self.n_steps, dtype=np.float32)
        
        # Noise level depends on tier
        noise_std = 0.05 if tier == 1 else (0.1 if tier == 2 else 0.15)
        
        for t in range(order, self.n_steps):
            y[t] = sum(phi[i] * y[t - i - 1] for i in range(order)) + np.random.normal(0, noise_std)
            y[t] = np.clip(y[t], -1e6, 1e6)
        
        if not np.isfinite(y).all(): 
            return self.generate(subtype, tier)
        
        tokens = ["X"]
        for p in phi: 
            tokens.extend(["MUL"] + self._float_to_tokens(p))
        
        return compute_timeseries_features(y), tokens, subtype


class TranscendentalGenerator(BaseGenerator):
    SUBTYPES = {
        "exp_growth": "exp", "exp_decay": "exp", 
        "power_positive": "power", "power_negative": "power", 
        "logarithmic": "log", "logistic": "logistic"
    }
    WEIGHTS = {k: 1.0 for k in SUBTYPES}
    
    def __init__(self, n_points: int = 200, x_range: tuple = (0.1, 10)):
        self.n_points = n_points
        self.x_range = x_range
    
    def generate(self, subtype=None, tier: int = 1):
        if subtype is None: 
            subtype = self._sample_subtype()
        form = self.SUBTYPES[subtype]
        x = np.linspace(*self.x_range, self.n_points)
        
        # TIER LOGIC - parameter ranges
        if tier == 1:
            # Snap to nice values
            a_choices = [0.5, 1.0, 1.5, 2.0]
            b_choices = [0.1, 0.2, 0.3, 0.5, 1.0]
        elif tier == 2:
            # Continuous 2dp
            a_choices = None
            b_choices = None
        else:  # tier == 3
            # Wider ranges
            a_choices = None
            b_choices = None
        
        if form == "exp":
            if tier == 1:
                a = np.random.choice(a_choices)
                b = np.random.choice(b_choices) * (1 if "growth" in subtype else -1)
            else:
                a = np.random.uniform(0.5, 2.5 if tier == 3 else 2)
                b = np.random.uniform(0.1, 0.6 if tier == 3 else 0.5) * (1 if "growth" in subtype else -1)
            y = a * np.exp(b * x)
            tokens = self._float_to_tokens(a) + ["MUL", "EXP"] + self._float_to_tokens(b) + ["MUL", "X"]
            
        elif form == "power":
            if tier == 1:
                a = np.random.choice(a_choices)
                b = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5]) * (1 if "positive" in subtype else -1)
            else:
                a = np.random.uniform(0.5, 2.5 if tier == 3 else 2)
                b = np.random.uniform(0.5, 3.5 if tier == 3 else 3) * (1 if "positive" in subtype else -1)
            y = a * np.power(x, b)
            tokens = self._float_to_tokens(a) + ["MUL", "X", "POW"] + self._float_to_tokens(b)
            
        elif form == "log":
            if tier == 1:
                a = np.random.choice([1.0, 1.5, 2.0, 2.5])
                b = np.random.choice([-2, -1, 0, 1, 2])
            else:
                a = np.random.uniform(0.5, 3.5 if tier == 3 else 3)
                b = np.random.uniform(-2.5 if tier == 3 else -2, 2.5 if tier == 3 else 2)
            y = a * np.log(x) + b
            tokens = self._float_to_tokens(a) + ["MUL", "LN", "X", "ADD"] + self._float_to_tokens(b)
            
        else:  # logistic
            if tier == 1:
                L = np.random.choice([5, 8, 10])
                k = np.random.choice([0.5, 1.0, 1.5, 2.0])
                x0 = np.random.choice([3, 4, 5, 6, 7])
            else:
                L = np.random.uniform(1, 12 if tier == 3 else 10)
                k = np.random.uniform(0.5, 2.5 if tier == 3 else 2)
                x0 = np.random.uniform(3, 8 if tier == 3 else 7)
            y = L / (1 + np.exp(-k * (x - x0)))
            tokens = self._float_to_tokens(L) + ["DIV", "INT_1", "ADD", "EXP", "NEG"] + self._float_to_tokens(k) + ["MUL", "X", "SUB"] + self._float_to_tokens(x0)
        
        # Add noise for tier 3
        if tier == 3:
            noise = np.random.normal(0, 0.02 * np.std(y), size=y.shape)
            y = y + noise
        
        return compute_algebraic_features(x, y), tokens, subtype


class PhysicsLawsGenerator(BaseGenerator):
    SUBTYPES = {
        "newton_gravity": "inverse_square", 
        "coulomb": "inverse_square", 
        "ideal_gas": "ideal_gas", 
        "kinetic_energy": "kinetic", 
        "hookes_law": "linear"
    }
    WEIGHTS = {k: 1.0 for k in SUBTYPES}
    
    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples

    def generate(self, subtype=None, tier: int = 1):
        if subtype is None: 
            subtype = self._sample_subtype()
        law = self.SUBTYPES[subtype]
        
        # TIER LOGIC - parameter ranges and constant precision
        if tier == 1:
            # Small integers, snap constants
            range_low, range_high = 1, 5
        elif tier == 2:
            # Continuous with 2dp
            range_low, range_high = 1, 10
        else:  # tier == 3
            # Wider ranges
            range_low, range_high = 0.1, 15
        
        if law == "inverse_square":
            vars = ["M1", "M2", "R"] if subtype == "newton_gravity" else ["Q1", "Q2", "R"]
            data = {v: np.random.uniform(range_low, range_high, self.n_samples) for v in vars}
            
            if tier == 1:
                C = np.random.choice([1, 2, 3, 4, 5])
            else:
                C = np.random.uniform(0.5, 5)
                
            y = C * data[vars[0]] * data[vars[1]] / (data["R"]**2)
            tokens = self._float_to_tokens(C) + ["MUL", vars[0], "MUL", vars[1], "DIV", "R", "POW", "INT_2"]
            
        elif law == "ideal_gas":
            data = {
                "N_MOLES": np.random.uniform(0.1 if tier == 3 else 0.5, 5, self.n_samples),  # ← ADD self.n_samples
                "T": np.random.uniform(200, 400 if tier <= 2 else 500, self.n_samples),
                "V": np.random.uniform(0.1 if tier == 3 else 0.5, 10, self.n_samples)
            }
            y = data["N_MOLES"] * 8.314 * data["T"] / data["V"]
            tokens = ["N_MOLES", "MUL", "R_GAS", "MUL", "T", "DIV", "V"]
            
        elif law == "kinetic":
            data = {
                "M": np.random.uniform(range_low, range_high * 10, self.n_samples),
                "V": np.random.uniform(0.1, range_high, self.n_samples)
            }
            y = 0.5 * data["M"] * data["V"]**2
            tokens = ["HALF", "MUL", "M", "MUL", "V", "POW", "INT_2"]
            
        else:  # linear (Hooke's law)
            data = {
                "K": np.random.uniform(range_low, range_high * 10, self.n_samples),
                "X": np.random.uniform(-1, 1, self.n_samples)
            }
            y = data["K"] * data["X"]
            tokens = ["K", "MUL", "X"]
        
        # Add noise for tier 3
        if tier == 3:
            noise = np.random.normal(0, 0.02 * np.std(y), size=y.shape)
            y = y + noise
        
        data["output"] = y
        return compute_multivariate_features(data), tokens, subtype


# ============================================================================
# Generator Registry
# ============================================================================

GENERATORS = {
    "polynomial": PolynomialGenerator,
    "ar": ARGenerator,
    "transcendental": TranscendentalGenerator,
    "physics_laws": PhysicsLawsGenerator,
    "linear_ode": LinearODEGenerator,
}

def get_generator(model_type, **kwargs):
    if model_type not in GENERATORS:
        raise ValueError(f"Unknown model type: {model_type}")
    return GENERATORS[model_type](**kwargs)

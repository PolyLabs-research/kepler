"""
generators_dynamical.py - ODE Generators with Phase 4 Second-Order Fix
"""
import numpy as np
from scipy.integrate import odeint
from generators_base import BaseGenerator, float_to_tokens
from features import compute_dynamical_features

class LinearODEGenerator(BaseGenerator):
    SUBTYPES = {
        "first_order_decay": {"order": 1, "a_sign": -1},
        "first_order_growth": {"order": 1, "a_sign": 1},
        "underdamped": {"order": 2, "zeta": 0.3},
        "overdamped": {"order": 2, "zeta": 1.5},
        "undamped": {"order": 2, "zeta": 0.0},
    }
    WEIGHTS = {k: 1.0 for k in SUBTYPES}

    def __init__(self, t_span=10.0, n_points=200):
        self.t = np.linspace(0, t_span, n_points)

    def generate(self, subtype=None, tier: int = 1):
        if subtype is None: 
            subtype = self._sample_subtype()
        config = self.SUBTYPES[subtype]
        
        if config["order"] == 1:
            # First-order ODE: dx/dt = a*x
            
            # TIER LOGIC
            if tier == 1:
                # Discrete values
                a_mag = np.random.choice([0.2, 0.5, 1.0, 1.5, 2.0])
            elif tier == 2:
                # Continuous 2dp
                a_mag = np.random.uniform(0.1, 2.0)
            else:  # tier == 3
                # Wider range
                a_mag = np.random.uniform(0.05, 2.5)
            
            a = a_mag * config["a_sign"]
            
            # Initial condition varies by tier
            if tier == 1:
                y0 = [np.random.choice([1.0, 2.0, 3.0, 5.0])]
            else:
                y0 = [np.random.uniform(0.5, 5.0)]
            
            t_local = self.t
            y = odeint(lambda y, t: a * y, y0, self.t).flatten()
            
            # Tokenize: DX_DT EQ <a> MUL X
            tokens = ["DX_DT", "EQ"] + self._float_to_tokens(a) + ["MUL", "X"]
            
        else:
            # Second-order ODE: x'' + 2*zeta*omega*x' + omega^2*x = 0
            # PHASE 4 FIX: Encode coefficients C1 and C2 instead of template
            
            # TIER LOGIC
            if tier == 1:
                # Discrete parameter sets
                omega = float(np.random.choice([1, 2, 3, 4]))
                zeta_base = config["zeta"]
                if zeta_base == 0.0:
                    zeta = 0.0
                elif zeta_base < 1.0:  # underdamped
                    zeta = np.random.choice([0.0, 0.3, 0.5, 0.7])
                else:  # overdamped
                    zeta = np.random.choice([1.0, 1.5, 2.0])
            elif tier == 2:
                # Continuous with 2dp
                omega = np.random.uniform(1.0, 5.0)
                zeta_base = config["zeta"]
                if zeta_base == 0.0:
                    zeta = 0.0
                elif zeta_base < 1.0:
                    zeta = np.random.uniform(0.0, 0.9)
                else:
                    zeta = np.random.uniform(1.0, 2.0)
            else:  # tier == 3
                # Wider range + noise
                omega = np.random.uniform(0.5, 6.0)
                zeta_base = config["zeta"]
                if zeta_base == 0.0:
                    zeta = 0.0
                elif zeta_base < 1.0:
                    zeta = np.random.uniform(0.0, 0.95)
                else:
                    zeta = np.random.uniform(1.0, 2.5)
            
            # Compute coefficients
            C1 = 2 * zeta * omega  # coefficient of x'
            C2 = omega ** 2         # coefficient of x
            # Adaptive t_span for overdamped: ensure >= 5 time constants
            # of the slow root are captured. For underdamped/undamped,
            # discriminant <= 0 so this is skipped and self.t is used as-is.
            disc = C1**2 - 4 * C2
            if disc > 0:
                r1 = (-C1 + np.sqrt(disc)) / 2  # slow root (closer to 0)
                t_span = max(self.t[-1], 5.0 / abs(r1))
                t_local = np.linspace(0, t_span, len(self.t))
            else:
                t_local = self.t

            # Initial conditions
            if tier == 1:
                y0 = [1.0, 0.0]
            else:
                y0 = [np.random.uniform(0.5, 2.0), np.random.uniform(-0.5, 0.5)]
            
            # Simulate
            def system(state, t):
                x, v = state
                return [v, -C1 * v - C2 * x]
            
            sol = odeint(system, y0, t_local)
            y = sol[:, 0]
            
            # Add noise for tier 3
            if tier == 3:
                noise = np.random.normal(0, 0.01 * np.std(y), size=y.shape)
                y = y + noise
            
            # PHASE 4 FIX: Tokenize as D2X_DT2 + C1*DX_DT + C2*X = 0
            # Format: D2X_DT2 ADD <C1_tokens> MUL DX_DT ADD <C2_tokens> MUL X EQ INT_0
            tokens = ["D2X_DT2", "ADD"] + self._float_to_tokens(C1) + ["MUL", "DX_DT", "ADD"] + self._float_to_tokens(C2) + ["MUL", "X", "EQ", "INT_0"]
        
        features = compute_dynamical_features(y, t=t_local)
        return features, tokens, subtype

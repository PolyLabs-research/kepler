#!/usr/bin/env python3
"""
synthesizer.py - Data Synthesizer for Galileo Minimal

Generates numerical data from symbolic equations, enabling the LLM verification loop:
    LLM hypothesizes → synthesize_data → discover_equation → verify match

This is the reverse of the discovery pipeline: instead of data→equation,
we do equation→data, allowing hypothesis testing.

Usage:
    from synthesizer import Synthesizer
    
    synth = Synthesizer()
    
    # From equation string
    x, y = synth.from_equation("2*x**2 + 3*x - 1", x_range=(-5, 5))
    
    # From natural language description
    x, y = synth.from_description("exponential decay with rate 0.5")
    
    # From ODE specification
    t, y = synth.from_ode("x'' + 4*x = 0", t_range=(0, 10), y0=[1, 0])
    
    # From physics law
    data = synth.from_physics("newton_gravity", M1=5.97e24, M2=7.35e22, R=3.84e8)
"""
import re
import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Dict, List, Optional, Union
import sympy
from sympy import Symbol, sympify, lambdify, sin, cos, exp, log, sqrt, pi, E


class SynthesizerError(Exception):
    """Raised when synthesis fails."""
    pass


class Synthesizer:
    """
    Data synthesizer that generates numerical arrays from symbolic specifications.
    
    Supports five equation families matching Galileo's specialist models:
    - Algebraic: polynomials, transcendental functions
    - Dynamical: linear ODEs (first and second order)
    - Timeseries: autoregressive processes
    - Multivariate: physics laws
    """
    
    # Natural language patterns for equation recognition
    NL_PATTERNS = {
        # Exponential patterns
        r"exponential\s+(?:growth|increase)(?:\s+(?:with\s+)?rate\s+([0-9.]+))?": 
            lambda m: f"exp({m.group(1) or '1'}*x)",
        r"exponential\s+decay(?:\s+(?:with\s+)?rate\s+([0-9.]+))?": 
            lambda m: f"exp(-{m.group(1) or '1'}*x)",
        r"exp\s*\(\s*([0-9.-]+)\s*\*?\s*x\s*\)":
            lambda m: f"exp({m.group(1)}*x)",
            
        # Power law patterns
        r"(?:power\s+law|power)\s+(?:with\s+)?(?:exponent|power)\s+([0-9.-]+)":
            lambda m: f"x**{m.group(1)}",
        r"(?:square\s+root|sqrt)(?:\s+of\s+x)?":
            lambda m: "sqrt(x)",
        r"inverse(?:\s+square)?(?:\s+law)?":
            lambda m: "1/x**2",
        r"quadratic":
            lambda m: "x**2",
        r"cubic":
            lambda m: "x**3",
        r"linear":
            lambda m: "x",
            
        # Trigonometric patterns
        r"(?:sine|sin)(?:\s+wave)?(?:\s+(?:with\s+)?(?:frequency|freq)\s+([0-9.]+))?":
            lambda m: f"sin({m.group(1) or '1'}*x)",
        r"(?:cosine|cos)(?:\s+wave)?(?:\s+(?:with\s+)?(?:frequency|freq)\s+([0-9.]+))?":
            lambda m: f"cos({m.group(1) or '1'}*x)",
            
        # Logarithmic patterns
        r"(?:logarithm|logarithmic|log|ln)(?:\s+of\s+x)?":
            lambda m: "log(x)",
            
        # Logistic patterns  
        r"logistic(?:\s+(?:with\s+)?(?:L|capacity)\s*=?\s*([0-9.]+))?(?:\s+(?:k|rate)\s*=?\s*([0-9.]+))?(?:\s+(?:x0|midpoint)\s*=?\s*([0-9.]+))?":
            lambda m: f"{m.group(1) or '10'}/(1 + exp(-{m.group(2) or '1'}*(x - {m.group(3) or '5'})))",
        r"sigmoid":
            lambda m: "1/(1 + exp(-x))",
            
        # Damped oscillation patterns
        r"damped\s+(?:oscillation|harmonic)(?:\s+(?:with\s+)?(?:damping|zeta)\s+([0-9.]+))?(?:\s+(?:frequency|omega)\s+([0-9.]+))?":
            lambda m: f"exp(-{m.group(1) or '0.3'}*{m.group(2) or '2'}*x)*cos({m.group(2) or '2'}*x)",
        r"(?:simple\s+)?harmonic(?:\s+motion)?(?:\s+(?:with\s+)?(?:frequency|omega)\s+([0-9.]+))?":
            lambda m: f"cos({m.group(1) or '2'}*x)",
    }
    
    # ODE patterns for parsing
    ODE_PATTERNS = {
        # Second order: x'' + C1*x' + C2*x = 0
        r"x''\s*\+\s*([0-9.]+)\s*\*?\s*x'\s*\+\s*([0-9.]+)\s*\*?\s*x\s*=\s*0":
            lambda m: {"order": 2, "C1": float(m.group(1)), "C2": float(m.group(2))},
        r"x''\s*\+\s*([0-9.]+)\s*\*?\s*x\s*=\s*0":
            lambda m: {"order": 2, "C1": 0.0, "C2": float(m.group(1))},
        r"y''\s*\+\s*([0-9.]+)\s*\*?\s*y'\s*\+\s*([0-9.]+)\s*\*?\s*y\s*=\s*0":
            lambda m: {"order": 2, "C1": float(m.group(1)), "C2": float(m.group(2))},
        r"y''\s*\+\s*([0-9.]+)\s*\*?\s*y\s*=\s*0":
            lambda m: {"order": 2, "C1": 0.0, "C2": float(m.group(1))},
            
        # First order: x' = a*x or dx/dt = a*x
        r"x'\s*=\s*([0-9.-]+)\s*\*?\s*x":
            lambda m: {"order": 1, "a": float(m.group(1))},
        r"y'\s*=\s*([0-9.-]+)\s*\*?\s*y":
            lambda m: {"order": 1, "a": float(m.group(1))},
        r"dx/dt\s*=\s*([0-9.-]+)\s*\*?\s*x":
            lambda m: {"order": 1, "a": float(m.group(1))},
        r"dy/dt\s*=\s*([0-9.-]+)\s*\*?\s*y":
            lambda m: {"order": 1, "a": float(m.group(1))},
    }
    
    # Physics law templates
    PHYSICS_TEMPLATES = {
        "newton_gravity": {
            "formula": "G * M1 * M2 / R**2",
            "constants": {"G": 6.67430e-11},
            "required_vars": ["M1", "M2", "R"],
            "description": "Newton's law of gravitation: F = G*M1*M2/R²"
        },
        "coulomb": {
            "formula": "k * Q1 * Q2 / R**2",
            "constants": {"k": 8.9875517923e9},
            "required_vars": ["Q1", "Q2", "R"],
            "description": "Coulomb's law: F = k*Q1*Q2/R²"
        },
        "ideal_gas": {
            "formula": "n * R_gas * T / V",
            "constants": {"R_gas": 8.31446},
            "required_vars": ["n", "T", "V"],
            "description": "Ideal gas law: P = nRT/V"
        },
        "kinetic_energy": {
            "formula": "0.5 * m * v**2",
            "constants": {},
            "required_vars": ["m", "v"],
            "description": "Kinetic energy: E = ½mv²"
        },
        "hookes_law": {
            "formula": "k * x",
            "constants": {},
            "required_vars": ["k", "x"],
            "description": "Hooke's law: F = kx"
        },
        "gravitational_pe": {
            "formula": "m * g * h",
            "constants": {"g": 9.80665},
            "required_vars": ["m", "h"],
            "description": "Gravitational PE: E = mgh"
        },
    }
    
    def __init__(self, default_n_points: int = 200):
        """
        Initialize the synthesizer.
        
        Args:
            default_n_points: Default number of data points to generate
        """
        self.default_n_points = default_n_points
        
    def from_equation(
        self,
        equation: str,
        x_range: Tuple[float, float] = (-5, 5),
        n_points: Optional[int] = None,
        noise_std: float = 0.0,
        variable: str = "x"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from a symbolic equation string.
        
        Args:
            equation: Equation string like "2*x**2 + 3*x - 1" or "exp(-0.5*x)"
            x_range: (min, max) range for x values
            n_points: Number of points to generate
            noise_std: Standard deviation of Gaussian noise (0 = no noise)
            variable: Variable name in the equation (default "x")
            
        Returns:
            Tuple of (x, y) numpy arrays
            
        Examples:
            >>> synth = Synthesizer()
            >>> x, y = synth.from_equation("x**2 + 2*x + 1")
            >>> x, y = synth.from_equation("exp(-0.5*x)", x_range=(0, 10))
            >>> x, y = synth.from_equation("sin(2*t)", variable="t")
        """
        n_points = n_points or self.default_n_points
        
        # Handle special x_range for log/sqrt
        if any(fn in equation.lower() for fn in ["log(", "ln(", "sqrt("]):
            if x_range[0] <= 0:
                x_range = (0.1, x_range[1])
        
        func = self._compile(equation, variable)

        try:
            # Generate x values
            x = np.linspace(x_range[0], x_range[1], n_points)
            
            # Evaluate
            y = func(x)
            
            # Handle scalar results (constant functions)
            if np.isscalar(y):
                y = np.full_like(x, y)
                
            # Add noise if requested
            if noise_std > 0:
                y = y + np.random.normal(0, noise_std, size=y.shape)
                
            return x.astype(np.float32), y.astype(np.float32)
            
        except Exception as e:
            raise SynthesizerError(f"Failed to evaluate equation '{equation}': {e}")

    def _compile(self, equation: str, variable: str = "x"):
        """Parse an equation string into a numpy-callable of one variable."""
        try:
            expr = sympify(equation)
            return lambdify(Symbol(variable), expr, modules=["numpy"])
        except Exception as e:
            raise SynthesizerError(f"Failed to parse equation '{equation}': {e}")

    def evaluate_at(
        self,
        equation: str,
        x: Union[np.ndarray, List[float]],
        variable: str = "x",
    ) -> np.ndarray:
        """
        Evaluate an equation string at exactly the given x values.

        Unlike from_equation, no grid is built and nothing is interpolated, so
        the result is exact for irregularly spaced or unsorted x. Points where
        the equation is undefined come back as nan or inf; the caller decides
        what to do with them.

        Args:
            equation: Equation string in the given variable
            x: Values to evaluate at
            variable: Variable name in the equation (default "x")

        Returns:
            float64 array with the same shape as x
        """
        func = self._compile(equation, variable)
        x = np.asarray(x, dtype=np.float64)
        try:
            with np.errstate(all="ignore"):
                y = np.asarray(func(x))
        except Exception as e:
            raise SynthesizerError(f"Failed to evaluate equation '{equation}': {e}")
        if np.iscomplexobj(y):
            y = np.where(np.abs(y.imag) <= 1e-12 * np.maximum(1.0, np.abs(y.real)), y.real, np.nan)
        if y.ndim == 0:
            y = np.full(x.shape, float(y), dtype=np.float64)
        return np.asarray(y, dtype=np.float64).reshape(x.shape)
    
    def from_description(
        self,
        description: str,
        x_range: Optional[Tuple[float, float]] = None,
        n_points: Optional[int] = None,
        noise_std: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from a natural language description.
        
        Args:
            description: Natural language like "exponential decay with rate 0.5"
            x_range: Optional (min, max) range (auto-selected if not provided)
            n_points: Number of points to generate
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Tuple of (x, y) numpy arrays
            
        Examples:
            >>> synth = Synthesizer()
            >>> x, y = synth.from_description("exponential decay with rate 0.5")
            >>> x, y = synth.from_description("sine wave with frequency 2")
            >>> x, y = synth.from_description("quadratic")
            >>> x, y = synth.from_description("damped harmonic with damping 0.3 frequency 2")
        """
        description = description.lower().strip()
        
        # Try each pattern
        for pattern, equation_func in self.NL_PATTERNS.items():
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                equation = equation_func(match)
                
                # Auto-select x_range based on equation type
                if x_range is None:
                    if "log" in equation or "sqrt" in equation:
                        x_range = (0.1, 10)
                    elif "1/(1" in equation.replace(" ", ""):  # Sigmoid/logistic
                        x_range = (-6, 6)
                    elif "exp(" in equation and "-" not in equation:
                        x_range = (0, 5)  # Exponential growth - limited range
                    elif "exp(-" in equation and "1/(1" not in equation.replace(" ", ""):
                        x_range = (0, 10)  # Exponential decay
                    elif "sin" in equation or "cos" in equation:
                        x_range = (0, 4 * np.pi)  # A few periods
                    elif "logistic" in description.lower():
                        x_range = (-6, 14)  # Cover the transition
                    else:
                        x_range = (-5, 5)
                
                return self.from_equation(equation, x_range, n_points, noise_std)
        
        # If no pattern matched, try direct parse
        try:
            return self.from_equation(description, x_range or (-5, 5), n_points, noise_std)
        except:
            raise SynthesizerError(
                f"Could not parse description: '{description}'\n"
                f"Supported patterns include: exponential growth/decay, power law, "
                f"sine/cosine, logarithmic, logistic, damped oscillation, quadratic, cubic, linear"
            )
    
    def from_ode(
        self,
        ode_spec: str,
        t_range: Tuple[float, float] = (0, 10),
        y0: Optional[List[float]] = None,
        n_points: Optional[int] = None,
        noise_std: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from an ODE specification.
        
        Supports first-order (dx/dt = a*x) and second-order (x'' + C1*x' + C2*x = 0).
        
        Args:
            ode_spec: ODE string like "x'' + 4*x = 0" or "dx/dt = -0.5*x"
            t_range: (min, max) time range
            y0: Initial conditions [x(0)] for first-order or [x(0), x'(0)] for second-order
            n_points: Number of time points
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Tuple of (t, x) numpy arrays
            
        Examples:
            >>> synth = Synthesizer()
            >>> t, x = synth.from_ode("x'' + 4*x = 0")  # Simple harmonic
            >>> t, x = synth.from_ode("x'' + 0.6*x' + 4*x = 0")  # Damped
            >>> t, x = synth.from_ode("dx/dt = -0.5*x", y0=[2.0])  # First-order decay
        """
        n_points = n_points or self.default_n_points
        
        # Parse ODE specification
        ode_params = None
        ode_spec_clean = ode_spec.replace(" ", "")
        
        for pattern, parser in self.ODE_PATTERNS.items():
            match = re.search(pattern, ode_spec_clean, re.IGNORECASE)
            if match:
                ode_params = parser(match)
                break
        
        if ode_params is None:
            raise SynthesizerError(
                f"Could not parse ODE: '{ode_spec}'\n"
                f"Supported formats:\n"
                f"  First-order:  x' = a*x, dx/dt = a*x\n"
                f"  Second-order: x'' + C1*x' + C2*x = 0, x'' + C2*x = 0"
            )
        
        t = np.linspace(t_range[0], t_range[1], n_points)
        
        if ode_params["order"] == 1:
            # First-order: dx/dt = a*x
            a = ode_params["a"]
            y0 = y0 or [1.0]
            
            x = odeint(lambda y, t: a * y, y0, t).flatten()
            
        else:
            # Second-order: x'' + C1*x' + C2*x = 0
            C1 = ode_params["C1"]
            C2 = ode_params["C2"]
            y0 = y0 or [1.0, 0.0]
            
            def system(state, t):
                x, v = state
                return [v, -C1 * v - C2 * x]
            
            sol = odeint(system, y0, t)
            x = sol[:, 0]
        
        # Add noise if requested
        if noise_std > 0:
            x = x + np.random.normal(0, noise_std, size=x.shape)
            
        return t.astype(np.float32), x.astype(np.float32)
    
    def from_ode_params(
        self,
        order: int,
        C1: float = 0.0,
        C2: float = 1.0,
        a: Optional[float] = None,
        t_range: Tuple[float, float] = (0, 10),
        y0: Optional[List[float]] = None,
        n_points: Optional[int] = None,
        noise_std: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ODE data from explicit parameters.
        
        Args:
            order: 1 for first-order, 2 for second-order
            C1: Damping coefficient (second-order only)
            C2: Stiffness coefficient (second-order only)  
            a: Growth/decay rate (first-order only)
            t_range: (min, max) time range
            y0: Initial conditions
            n_points: Number of time points
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Tuple of (t, x) numpy arrays
            
        Examples:
            >>> synth = Synthesizer()
            >>> t, x = synth.from_ode_params(order=2, C1=0.6, C2=4.0)  # Underdamped
            >>> t, x = synth.from_ode_params(order=2, C1=6.0, C2=4.0)  # Overdamped
            >>> t, x = synth.from_ode_params(order=1, a=-0.5)  # Decay
        """
        n_points = n_points or self.default_n_points
        t = np.linspace(t_range[0], t_range[1], n_points)
        
        if order == 1:
            if a is None:
                raise SynthesizerError("First-order ODE requires 'a' parameter")
            y0 = y0 or [1.0]
            x = odeint(lambda y, t: a * y, y0, t).flatten()
            
        elif order == 2:
            y0 = y0 or [1.0, 0.0]
            
            # Adaptive t_span for overdamped (same as generator)
            disc = C1**2 - 4 * C2
            if disc > 0:
                r1 = (-C1 + np.sqrt(disc)) / 2
                t_span = max(t_range[1], 5.0 / abs(r1))
                t = np.linspace(t_range[0], t_span, n_points)
            
            def system(state, t):
                x, v = state
                return [v, -C1 * v - C2 * x]
            
            sol = odeint(system, y0, t)
            x = sol[:, 0]
        else:
            raise SynthesizerError(f"Unsupported ODE order: {order}")
        
        if noise_std > 0:
            x = x + np.random.normal(0, noise_std, size=x.shape)
            
        return t.astype(np.float32), x.astype(np.float32)
    
    def from_ar(
        self,
        phi: List[float],
        n_steps: int = 200,
        initial: Optional[List[float]] = None,
        noise_std: float = 0.1
    ) -> np.ndarray:
        """
        Generate autoregressive timeseries data.
        
        Args:
            phi: AR coefficients [phi_1, phi_2, ...] for AR(p) process
            n_steps: Number of time steps to generate
            initial: Initial values (random if not provided)
            noise_std: Innovation noise standard deviation
            
        Returns:
            1D numpy array of timeseries values
            
        Examples:
            >>> synth = Synthesizer()
            >>> y = synth.from_ar([0.7])  # AR(1) with phi=0.7
            >>> y = synth.from_ar([0.5, 0.3])  # AR(2)
        """
        order = len(phi)
        phi = np.array(phi)
        
        # Check stability
        if order > 1 and np.sum(np.abs(phi)) >= 1.0:
            raise SynthesizerError(
                f"AR process is unstable: sum(|phi|) = {np.sum(np.abs(phi)):.3f} >= 1.0"
            )
        
        y = np.zeros(n_steps, dtype=np.float32)
        
        # Initialize
        if initial is not None:
            y[:order] = initial[:order]
        
        # Generate process
        for t in range(order, n_steps):
            y[t] = sum(phi[i] * y[t - i - 1] for i in range(order))
            y[t] += np.random.normal(0, noise_std)
            y[t] = np.clip(y[t], -1e6, 1e6)
        
        return y
    
    def from_physics(
        self,
        law: str,
        n_samples: int = 100,
        noise_std: float = 0.0,
        **variables
    ) -> Dict[str, np.ndarray]:
        """
        Generate data from a physics law.
        
        Args:
            law: Name of physics law (newton_gravity, coulomb, ideal_gas, 
                 kinetic_energy, hookes_law, gravitational_pe)
            n_samples: Number of samples to generate
            noise_std: Relative noise as fraction of output
            **variables: Variable values - either scalars or arrays
            
        Returns:
            Dict mapping variable names to arrays, including "output"
            
        Examples:
            >>> synth = Synthesizer()
            >>> # Newton's gravity with varying R
            >>> data = synth.from_physics("newton_gravity", 
            ...     M1=5.97e24, M2=1000, R=np.linspace(6.4e6, 4e8, 100))
            >>> 
            >>> # Ideal gas law with varying T and V
            >>> data = synth.from_physics("ideal_gas",
            ...     n=1.0, T=np.linspace(200, 400, 50), V=0.01)
        """
        if law not in self.PHYSICS_TEMPLATES:
            available = ", ".join(self.PHYSICS_TEMPLATES.keys())
            raise SynthesizerError(f"Unknown physics law: '{law}'. Available: {available}")
        
        template = self.PHYSICS_TEMPLATES[law]
        
        # Check required variables
        missing = [v for v in template["required_vars"] if v.lower() not in 
                   {k.lower() for k in variables}]
        if missing:
            raise SynthesizerError(
                f"Missing required variables for {law}: {missing}\n"
                f"Required: {template['required_vars']}\n"
                f"Description: {template['description']}"
            )
        
        # Normalize variable names to match formula
        var_map = {}
        for req_var in template["required_vars"]:
            for provided in variables:
                if provided.lower() == req_var.lower():
                    var_map[req_var] = variables[provided]
                    break
        
        # Determine array length from any array inputs
        array_len = n_samples
        for value in var_map.values():
            if not np.isscalar(value):
                array_len = len(value)
                break
        
        # Convert scalars to arrays (broadcast to array_len)
        data = {}
        for var_name, value in var_map.items():
            if np.isscalar(value):
                data[var_name] = np.full(array_len, value, dtype=np.float64)
            else:
                data[var_name] = np.array(value, dtype=np.float64)
        
        # Add constants
        for const_name, const_value in template["constants"].items():
            data[const_name] = const_value
        
        # Evaluate formula directly using numpy operations
        # Build evaluation namespace
        eval_namespace = {"np": np}
        for k, v in data.items():
            eval_namespace[k] = v
        
        # Convert sympy formula to evaluable expression
        formula = template["formula"]
        # Replace ** with np.power for safety (though ** works in numpy)
        output = eval(formula, {"__builtins__": {}}, eval_namespace)
        
        if np.isscalar(output):
            output = np.full(array_len, output)
        
        # Add noise
        if noise_std > 0:
            output = output + np.random.normal(0, noise_std * np.abs(output), size=output.shape)
        
        # Build result dict
        result = {k: v.astype(np.float32) for k, v in data.items() if isinstance(v, np.ndarray)}
        result["output"] = output.astype(np.float32)
        
        return result
    
    def from_polynomial(
        self,
        coeffs: List[float],
        x_range: Tuple[float, float] = (-5, 5),
        n_points: Optional[int] = None,
        noise_std: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate polynomial data from coefficients.
        
        Args:
            coeffs: Coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
            x_range: (min, max) range for x values
            n_points: Number of points
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Tuple of (x, y) numpy arrays
            
        Examples:
            >>> synth = Synthesizer()
            >>> x, y = synth.from_polynomial([1, 2, 1])  # 1 + 2x + x²
            >>> x, y = synth.from_polynomial([0, 0, 1])  # x²
        """
        n_points = n_points or self.default_n_points
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        y = np.polyval(coeffs[::-1], x)
        
        if noise_std > 0:
            y = y + np.random.normal(0, noise_std, size=y.shape)
            
        return x.astype(np.float32), y.astype(np.float32)
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def list_physics_laws(self) -> Dict[str, str]:
        """Return dict of available physics laws with descriptions."""
        return {k: v["description"] for k, v in self.PHYSICS_TEMPLATES.items()}
    
    def list_nl_patterns(self) -> List[str]:
        """Return list of supported natural language patterns."""
        patterns = [
            "exponential growth [with rate X]",
            "exponential decay [with rate X]",
            "power law with exponent X",
            "square root",
            "inverse square",
            "quadratic / cubic / linear",
            "sine/cosine wave [with frequency X]",
            "logarithmic / log / ln",
            "logistic [with L=X k=X x0=X]",
            "sigmoid",
            "damped oscillation [with damping X frequency X]",
            "simple harmonic [with frequency X]",
        ]
        return patterns
    
    def get_ode_info(self, C1: float, C2: float) -> Dict[str, any]:
        """
        Analyze second-order ODE parameters.
        
        Args:
            C1: Damping coefficient
            C2: Stiffness coefficient
            
        Returns:
            Dict with omega, zeta, damping type, roots, etc.
        """
        omega = np.sqrt(C2)
        zeta = C1 / (2 * omega) if omega > 0 else 0
        
        disc = C1**2 - 4 * C2
        
        if disc < 0:
            # Complex roots (underdamped)
            omega_d = omega * np.sqrt(1 - zeta**2)
            roots = (
                complex(-C1/2, np.sqrt(-disc)/2),
                complex(-C1/2, -np.sqrt(-disc)/2)
            )
            damping_type = "underdamped"
        elif disc > 0:
            # Real roots (overdamped)
            r1 = (-C1 + np.sqrt(disc)) / 2
            r2 = (-C1 - np.sqrt(disc)) / 2
            roots = (r1, r2)
            omega_d = None
            damping_type = "overdamped"
        else:
            # Repeated roots (critically damped)
            r = -C1 / 2
            roots = (r, r)
            omega_d = None
            damping_type = "critically_damped"
        
        if abs(C1) < 1e-10:
            damping_type = "undamped"
            omega_d = omega
        
        return {
            "C1": C1,
            "C2": C2,
            "omega": omega,
            "zeta": zeta,
            "damping_type": damping_type,
            "damped_frequency": omega_d,
            "roots": roots,
            "discriminant": disc,
        }


# =============================================================================
# Convenience functions for quick access
# =============================================================================

_default_synth = None

def _get_synth() -> Synthesizer:
    """Get or create default synthesizer instance."""
    global _default_synth
    if _default_synth is None:
        _default_synth = Synthesizer()
    return _default_synth


def synthesize_equation(equation: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Quick access to from_equation."""
    return _get_synth().from_equation(equation, **kwargs)


def synthesize_description(description: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Quick access to from_description."""
    return _get_synth().from_description(description, **kwargs)


def synthesize_ode(ode_spec: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Quick access to from_ode."""
    return _get_synth().from_ode(ode_spec, **kwargs)


def synthesize_physics(law: str, **kwargs) -> Dict[str, np.ndarray]:
    """Quick access to from_physics."""
    return _get_synth().from_physics(law, **kwargs)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Synthesizer for Galileo Minimal")
    parser.add_argument("--equation", "-e", type=str, help="Equation to synthesize")
    parser.add_argument("--description", "-d", type=str, help="Natural language description")
    parser.add_argument("--ode", type=str, help="ODE specification")
    parser.add_argument("--physics", type=str, help="Physics law name")
    parser.add_argument("--x-range", type=float, nargs=2, default=[-5, 5], help="X range")
    parser.add_argument("--n-points", type=int, default=20, help="Number of points to show")
    parser.add_argument("--list-physics", action="store_true", help="List available physics laws")
    parser.add_argument("--list-patterns", action="store_true", help="List NL patterns")
    
    args = parser.parse_args()
    
    synth = Synthesizer()
    
    if args.list_physics:
        print("\nAvailable Physics Laws:")
        print("-" * 60)
        for name, desc in synth.list_physics_laws().items():
            print(f"  {name:20s} {desc}")
        exit(0)
    
    if args.list_patterns:
        print("\nSupported Natural Language Patterns:")
        print("-" * 60)
        for pattern in synth.list_nl_patterns():
            print(f"  • {pattern}")
        exit(0)
    
    if args.equation:
        print(f"\nSynthesizing: {args.equation}")
        print(f"Range: {args.x_range}")
        x, y = synth.from_equation(args.equation, tuple(args.x_range), args.n_points)
        print("\n  x       |  y")
        print("-" * 25)
        for xi, yi in zip(x, y):
            print(f"  {xi:7.3f} | {yi:10.4f}")
    
    elif args.description:
        print(f"\nParsing description: '{args.description}'")
        x, y = synth.from_description(args.description, n_points=args.n_points)
        print("\n  x       |  y")
        print("-" * 25)
        for xi, yi in zip(x, y):
            print(f"  {xi:7.3f} | {yi:10.4f}")
    
    elif args.ode:
        print(f"\nSolving ODE: {args.ode}")
        t, y = synth.from_ode(args.ode, n_points=args.n_points)
        print("\n  t       |  x")
        print("-" * 25)
        for ti, yi in zip(t, y):
            print(f"  {ti:7.3f} | {yi:10.4f}")
    
    elif args.physics:
        print(f"\nPhysics law: {args.physics}")
        print("(Run with actual variable values for data)")
        template = synth.PHYSICS_TEMPLATES.get(args.physics)
        if template:
            print(f"  Formula: {template['formula']}")
            print(f"  Required: {template['required_vars']}")
            print(f"  Constants: {template['constants']}")
        else:
            print(f"  Unknown law. Use --list-physics to see available laws.")
    
    else:
        # Demo
        print("=" * 70)
        print("Galileo Minimal - Data Synthesizer Demo")
        print("=" * 70)
        
        print("\n1. From equation: x**2 + 2*x - 1")
        x, y = synth.from_equation("x**2 + 2*x - 1", n_points=5)
        for xi, yi in zip(x, y):
            print(f"   x={xi:6.2f} → y={yi:8.3f}")
        
        print("\n2. From description: 'exponential decay with rate 0.5'")
        x, y = synth.from_description("exponential decay with rate 0.5", n_points=5)
        for xi, yi in zip(x, y):
            print(f"   x={xi:6.2f} → y={yi:8.3f}")
        
        print("\n3. From ODE: x'' + 4*x = 0 (simple harmonic)")
        t, y = synth.from_ode("x'' + 4*x = 0", n_points=5)
        for ti, yi in zip(t, y):
            print(f"   t={ti:6.2f} → x={yi:8.3f}")
        
        print("\n4. From AR: AR(1) with phi=0.7")
        y = synth.from_ar([0.7], n_steps=10)
        print(f"   y = {y.round(3).tolist()}")
        
        print("\n5. ODE info for C1=0.6, C2=4.0:")
        info = synth.get_ode_info(0.6, 4.0)
        print(f"   Type: {info['damping_type']}")
        print(f"   omega={info['omega']:.3f}, zeta={info['zeta']:.3f}")
        print(f"   omega_d={info['damped_frequency']:.3f}")
        
        print("\n" + "=" * 70)
        print("Use --help for CLI options, or import Synthesizer in your code.")
        print("=" * 70)

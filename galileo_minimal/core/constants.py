import math

# Mathematical constants
MATHEMATICAL_CONSTANTS = {
    "PI": math.pi,
    "E": math.e,
    "PHI": (1 + math.sqrt(5)) / 2,
    "TAU": 2 * math.pi,
    "SQRT2": math.sqrt(2),
    "SQRT3": math.sqrt(3),
    "HALF": 0.5,
    "THIRD": 1/3,
    "QUARTER": 0.25,
}

# Small integers as specialized tokens
INTEGER_CONSTANTS = {f"INT_{i}": float(i) for i in range(11)}
INTEGER_CONSTANTS.update({"NEG_1": -1.0, "NEG_2": -2.0})

# Physical constants from the larger spec
PHYSICS_CONSTANTS = {
    "G_EARTH": 9.80665,
    "G_NEWTON": 6.67430e-11,
    "C_LIGHT": 299792458,
    "R_GAS": 8.31446,
}

ALL_CONSTANTS = {**MATHEMATICAL_CONSTANTS, **INTEGER_CONSTANTS, **PHYSICS_CONSTANTS}

def snap_to_token(value: float, tolerance: float = 0.001) -> str | None:
    """Check if a float is close enough to a known constant to replace it."""
    for token, const_val in ALL_CONSTANTS.items():
        if const_val == 0:
            if abs(value) < tolerance: return token
        else:
            # Use relative error for non-zero constants
            if abs(value - const_val) / abs(const_val) < tolerance:
                return token
    return None
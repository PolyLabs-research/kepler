"""Run in galileo_minimal. Check whether features.py has the enriched signature."""
import sys
sys.path.insert(0, ".")
import inspect
from features import compute_dynamical_features
sig = inspect.signature(compute_dynamical_features)
print(f"Signature: compute_dynamical_features{sig}")
params = list(sig.parameters.keys())
if "t" in params:
    print("✓ 't' parameter present — adaptive t_span will propagate correctly")
else:
    print("✗ 't' parameter MISSING — need to copy features_new.py → features.py")
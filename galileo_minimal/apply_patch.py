"""
Run in galileo_minimal.
Applies all three changes to generators_dynamical.py.
Uses line-by-line search so whitespace differences don't cause failures.
"""
from pathlib import Path

path = Path("generators_dynamical.py")
lines = path.read_text().splitlines(keepends=True)

# --- Find key lines by content ---
idx = {}
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "C2 = omega ** 2         # coefficient of x":
        idx["C2_line"] = i
    if stripped == "# Initial conditions" and "C2_line" in idx and i > idx["C2_line"]:
        if "init_conditions" not in idx:  # first occurrence after C2
            idx["init_conditions"] = i
    if stripped == "sol = odeint(system, y0, self.t)":
        idx["odeint_line"] = i
    if stripped == "y = odeint(lambda y, t: a * y, y0, self.t).flatten()":
        idx["first_order_odeint"] = i
    if stripped == "features = compute_dynamical_features(y)":
        idx["features_line"] = i

print("Found target lines:")
for k, v in idx.items():
    print(f"  {k}: line {v+1} → {lines[v].rstrip()}")

missing = [k for k in ["C2_line","init_conditions","odeint_line","first_order_odeint","features_line"] if k not in idx]
if missing:
    print(f"\n✗ FAILED: could not find lines: {missing}")
    print("  Check generators_dynamical.py manually.")
    exit(1)

# --- PATCH 1: Insert adaptive t_span block between C2_line and init_conditions ---
# Replace everything from C2_line+1 up to (not including) init_conditions
# with the adaptive block. This handles any number of blank lines between them.
adaptive_block = """\
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

"""
# Remove lines between C2 and init_conditions, insert adaptive block
lines[idx["C2_line"]+1 : idx["init_conditions"]] = [adaptive_block]
print("✓ Patch 1 applied: adaptive t_span block inserted")

# Re-scan after insertion (line numbers shifted)
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "sol = odeint(system, y0, self.t)":
        idx["odeint_line"] = i
    if stripped == "y = odeint(lambda y, t: a * y, y0, self.t).flatten()":
        idx["first_order_odeint"] = i
    if stripped == "features = compute_dynamical_features(y)":
        idx["features_line"] = i

# --- PATCH 2: odeint uses t_local ---
lines[idx["odeint_line"]] = lines[idx["odeint_line"]].replace("self.t)", "t_local)")
print("✓ Patch 2 applied: odeint now uses t_local")

# --- PATCH 3a: set t_local = self.t in first-order branch (before its odeint) ---
indent = "            "  # match indentation
lines.insert(idx["first_order_odeint"], indent + "t_local = self.t\n")
print("✓ Patch 3a applied: t_local = self.t in first-order branch")

# Re-scan for features line (shifted by insert)
for i, line in enumerate(lines):
    if line.strip() == "features = compute_dynamical_features(y)":
        idx["features_line"] = i

# --- PATCH 3b: feature call passes t=t_local ---
lines[idx["features_line"]] = lines[idx["features_line"]].replace(
    "compute_dynamical_features(y)",
    "compute_dynamical_features(y, t=t_local)"
)
print("✓ Patch 3b applied: feature extraction passes t_local")

# Write
path.write_text("".join(lines))
print(f"\n  Written to {path}")

# --- Verification: print the patched second-order block ---
print("\n" + "=" * 70)
print("VERIFICATION: second-order block after patch")
print("=" * 70)
result = path.read_text().splitlines()
printing = False
for i, line in enumerate(result):
    if "# Compute coefficients" in line:
        printing = True
    if printing:
        print(f"  {i+1:4d} | {line}")
    if printing and "features = compute_dynamical_features" in line:
        break
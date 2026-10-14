"""
Run in galileo_minimal.
Overdamped seed 9 failed badly (C1_err=0.44, C2_err=0.60).
Check: what are the actual ODE parameters, and is this a regime
where the trajectory is degenerate (e.g. very fast decay, two
real roots that are close together, making C1/C2 hard to separate)?

For overdamped: zeta > 1, roots are real:
  r1 = -zeta*omega + omega*sqrt(zeta^2 - 1)
  r2 = -zeta*omega - omega*sqrt(zeta^2 - 1)
If zeta is only slightly > 1, the roots are close and the trajectory
looks almost like a single exponential — C1 and C2 become entangled.
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from generators import get_generator
from generators_dynamical import LinearODEGenerator
from scipy.integrate import odeint

gen = get_generator("linear_ode")

# Reproduce seed 9 overdamped
np.random.seed(9)
feat, tokens, _ = gen.generate(subtype="overdamped", tier=2)
print(f"Tokens: {' '.join(tokens)}")
print(f"Feature vector nonzero: {int(np.sum(np.abs(feat) > 1e-6))}/64")
print()

# Now reconstruct the actual parameters
# Need to replicate the generator's internal logic for seed 9
# The generator uses np.random internally, so seed 9 at generate() entry
# gives specific omega/zeta draws. Let's just parse from tokens.
def parse_coeff_tokens(tokens, start_token, end_token):
    """Extract coefficient tokens between start and end markers."""
    try:
        start = tokens.index(start_token)
        end = tokens.index(end_token, start)
        return tokens[start+1:end]
    except:
        return []

token_list = list(tokens)
# Parse: D2X_DT2 ADD <C1> MUL DX_DT ADD <C2> MUL X EQ INT_0
add1 = token_list.index("ADD")
mul1 = token_list.index("MUL", add1)
add2 = token_list.index("ADD", mul1)
mul2 = token_list.index("MUL", add2)
c1_tokens = token_list[add1+1:mul1]
c2_tokens = token_list[add2+1:mul2]

def tokens_to_float(toks):
    s = ""
    neg = False
    for t in toks:
        if t == "CONST": continue
        elif t == "NEG": neg = True
        elif t == "POINT": s += "."
        elif t.startswith("INT_"): return float(t.split("_")[1]) * (-1 if neg else 1)
        else: s += t
    return float(s) * (-1 if neg else 1) if s else None

C1 = tokens_to_float(c1_tokens)
C2 = tokens_to_float(c2_tokens)
print(f"C1 = {C1}, C2 = {C2}")

# Derived quantities
omega = np.sqrt(C2)
zeta  = C1 / (2 * omega)
print(f"omega = {omega:.4f}, zeta = {zeta:.4f}")

# Roots of characteristic equation: r^2 + C1*r + C2 = 0
discriminant = C1**2 - 4*C2
print(f"Discriminant = C1^2 - 4*C2 = {discriminant:.4f}")
if discriminant >= 0:
    r1 = (-C1 + np.sqrt(discriminant)) / 2
    r2 = (-C1 - np.sqrt(discriminant)) / 2
    print(f"Roots: r1 = {r1:.4f}, r2 = {r2:.4f}")
    print(f"Root separation: |r1-r2| = {abs(r1-r2):.4f}")
    print(f"Root ratio: r1/r2 = {r1/r2:.4f}")
    # If roots are close, the trajectory looks like a single exponential
    # and C1/C2 become hard to separate
    if abs(r1 - r2) < 0.5:
        print(f"\n  ⚠️  NEAR-CRITICAL DAMPING: roots are very close.")
        print(f"      Trajectory looks like single exponential.")
        print(f"      C1 and C2 are entangled — hard to recover independently.")

# Generate and plot the trajectory (print values)
t = np.linspace(0, 10.0, 200)
y0 = [1.0, 0.0]
def system(state, t):
    x, v = state
    return [v, -C1*v - C2*x]
sol = odeint(system, y0, t)
y = sol[:, 0]

print(f"\nTrajectory stats:")
print(f"  y[0]={y[0]:.4f}, y[50]={y[50]:.6f}, y[100]={y[100]:.8f}, y[-1]={y[-1]:.2e}")
print(f"  Time to 1% of initial: ", end="")
for i, yi in enumerate(y):
    if abs(yi) < 0.01:
        print(f"t={t[i]:.2f}")
        break
else:
    print("never (still > 1% at t=10)")

# Compare with a "normal" overdamped case for contrast
print("\n--- COMPARISON: overdamped seed 0 ---")
np.random.seed(0)
feat0, tokens0, _ = gen.generate(subtype="overdamped", tier=2)
print(f"Tokens: {' '.join(tokens0)}")
token_list0 = list(tokens0)
add1 = token_list0.index("ADD")
mul1 = token_list0.index("MUL", add1)
add2 = token_list0.index("ADD", mul1)
mul2 = token_list0.index("MUL", add2)
C1_0 = tokens_to_float(token_list0[add1+1:mul1])
C2_0 = tokens_to_float(token_list0[add2+1:mul2])
omega_0 = np.sqrt(C2_0)
zeta_0 = C1_0 / (2*omega_0)
disc_0 = C1_0**2 - 4*C2_0
r1_0 = (-C1_0 + np.sqrt(disc_0))/2
r2_0 = (-C1_0 - np.sqrt(disc_0))/2
print(f"C1={C1_0}, C2={C2_0}, omega={omega_0:.4f}, zeta={zeta_0:.4f}")
print(f"Roots: r1={r1_0:.4f}, r2={r2_0:.4f}, separation={abs(r1_0-r2_0):.4f}")
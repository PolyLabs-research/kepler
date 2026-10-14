"""
Run in galileo_minimal.
For overdamped seed 9 (r1=-0.3949), the 1/e decay time is 1/|r1| = 2.53.
So in t=10 we only see ~4 decay constants on the slow mode.
Check: what t_span would give 5 decay constants? And does the generator
use a fixed t_span or adaptive?
Also check: across all overdamped tier 2 cases, what's the distribution
of the slow root r1, and how many have |r1| < 0.5 (slow decay)?
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from generators import get_generator

gen = get_generator("linear_ode")

print("=== OVERDAMPED ROOT DISTRIBUTION (200 samples, tier 2) ===\n")

slow_roots = []
all_roots = []

for seed in range(200):
    np.random.seed(seed)
    feat, tokens, _ = gen.generate(subtype="overdamped", tier=2)

    # Parse C1, C2
    toks = list(tokens)
    try:
        add1 = toks.index("ADD")
        mul1 = toks.index("MUL", add1)
        add2 = toks.index("ADD", mul1)
        mul2 = toks.index("MUL", add2)
        c1_toks = toks[add1+1:mul1]
        c2_toks = toks[add2+1:mul2]

        def t2f(ts):
            s, neg = "", False
            for t in ts:
                if t == "CONST": continue
                elif t == "NEG": neg = True
                elif t == "POINT": s += "."
                elif t.startswith("INT_"): return float(t.split("_")[1]) * (-1 if neg else 1)
                else: s += t
            return float(s) * (-1 if neg else 1) if s else None

        C1 = t2f(c1_toks)
        C2 = t2f(c2_toks)
        if C1 is None or C2 is None: continue

        disc = C1**2 - 4*C2
        if disc < 0: continue  # shouldn't happen for overdamped

        r1 = (-C1 + np.sqrt(disc)) / 2  # slow root (closer to 0)
        r2 = (-C1 - np.sqrt(disc)) / 2  # fast root

        all_roots.append({"seed": seed, "C1": C1, "C2": C2, "r1": r1, "r2": r2,
                          "tau_slow": 1.0/abs(r1), "decay_in_10": np.exp(r1*10)})

        if abs(r1) < 0.5:
            slow_roots.append(all_roots[-1])
    except:
        pass

print(f"  Total overdamped samples parsed: {len(all_roots)}")
print(f"  Slow root |r1| < 0.5 (tau > 2.0): {len(slow_roots)} ({100*len(slow_roots)/len(all_roots):.1f}%)")
print(f"  Slow root |r1| < 1.0 (tau > 1.0): {sum(1 for r in all_roots if abs(r['r1']) < 1.0)} ({100*sum(1 for r in all_roots if abs(r['r1']) < 1.0)/len(all_roots):.1f}%)")

print(f"\n  Slow root distribution:")
r1_vals = np.array([r["r1"] for r in all_roots])
print(f"    min={r1_vals.min():.4f} max={r1_vals.max():.4f} mean={r1_vals.mean():.4f} median={np.median(r1_vals):.4f}")

# Show the slow cases
if slow_roots:
    print(f"\n  Cases with |r1| < 0.5 (hardest to recover):")
    print(f"    {'seed':>4s} {'C1':>6s} {'C2':>6s} {'r1':>7s} {'tau':>6s} {'y(10)':>8s}")
    for r in sorted(slow_roots, key=lambda x: abs(x["r1"])):
        print(f"    {r['seed']:4d} {r['C1']:6.2f} {r['C2']:6.2f} {r['r1']:7.4f} {r['tau_slow']:6.2f} {r['decay_in_10']:8.4f}")

# Check: does the generator use fixed t_span?
print(f"\n=== GENERATOR t_span CHECK ===")
# Look at generators_dynamical.py for t_span logic
import inspect
from generators_dynamical import LinearODEGenerator
source = inspect.getsource(LinearODEGenerator.generate)
# Find t_span references
lines = source.split("\n")
for i, line in enumerate(lines):
    if "t_span" in line or "linspace" in line or "t =" in line.strip()[:4]:
        print(f"  line {i}: {line.strip()}")
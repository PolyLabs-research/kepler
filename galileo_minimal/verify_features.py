"""
Verify the new compute_dynamical_features.
Reproduces exactly the same checks as diagnose_features.py
so the output is directly comparable.
Run in galileo_minimal AFTER copying features_new.py there as features.py.
"""
import numpy as np
from generators import get_generator
from features import compute_dynamical_features   # will be the new version

gen = get_generator("linear_ode")

subtypes_2nd = ["underdamped", "overdamped", "undamped"]

print("=== RAW FEATURE VECTORS (second-order subtypes, seed 0) ===\n")
for st in subtypes_2nd:
    np.random.seed(0)
    feat, tokens, _ = gen.generate(subtype=st, tier=1)
    nonzero = int(np.sum(np.abs(feat) > 1e-6))

    print(f"  [{st}]")
    print(f"    tokens: {' '.join(tokens)}")
    print(f"    nonzero dims: {nonzero}/64")

    # Show the key parameter-encoding dims
    print(f"    [order]   d8={feat[8]:6.2f} (zc) d10={feat[10]:4.1f} (mono) d15={feat[15]:6.3f} (exp_rate)")
    print(f"    [C2/omega] d16={feat[16]:6.3f} (|dy/y|) d17={feat[17]:6.3f} (|d2y/y|) d19={feat[19]:6.3f} (omega)")
    print(f"    [C1/damp]  d37={feat[37]:6.3f} (|sigma|) d38={feat[38]:6.3f} (C1_est) d46={feat[46]:6.3f} (E_decay)")
    print(f"    [waveform] d54-63: [{', '.join(f'{feat[54+i]:.3f}' for i in range(10))}]")
    print()

print("=== WITHIN-SUBTYPE FEATURE VARIANCE ===\n")
print("  (Compare to original: underdamped had 5 varying dims, overdamped had 4)\n")

for st in subtypes_2nd:
    all_feats = []
    all_tokens = []
    for seed in range(50):
        np.random.seed(seed * 7 + 13)
        feat, tokens, _ = gen.generate(subtype=st, tier=1)
        all_feats.append(feat)
        all_tokens.append(tokens)

    all_feats = np.array(all_feats)
    varying = int(np.sum(np.std(all_feats, axis=0) > 0.001))

    print(f"  [{st}] — {len(all_feats)} samples, {varying} varying dims (was 4-5)")

    # Group by C1 token and check which dims separate C1 values
    c1_groups = {}
    for i, tokens in enumerate(all_tokens):
        try:
            add1_idx = tokens.index("ADD")
            mul1_idx = tokens.index("MUL", add1_idx)
            c1_tok = " ".join(tokens[add1_idx + 1 : mul1_idx])
            c1_groups.setdefault(c1_tok, []).append(i)
        except:
            pass

    if len(c1_groups) > 1:
        print(f"    C1 groups: {list(c1_groups.keys())}")

        # Find dims that separate C1 with Cohen's d > 0.8 (large effect)
        separating = []
        for dim in range(64):
            group_means = {
                c1: np.mean([all_feats[i, dim] for i in idxs])
                for c1, idxs in c1_groups.items()
            }
            vals = list(group_means.values())
            spread = max(vals) - min(vals)
            if spread > 0.05:
                separating.append((dim, spread, group_means))

        print(f"    Dims separating C1 (spread > 0.05): {len(separating)}")
        # Show top 5 by spread
        separating.sort(key=lambda x: -x[1])
        for dim, spread, gm in separating[:5]:
            summary = {k: f"{v:.3f}" for k, v in sorted(gm.items())}
            print(f"      dim {dim:2d} (spread={spread:.3f}): {summary}")

    # Same for C2
    c2_groups = {}
    for i, tokens in enumerate(all_tokens):
        try:
            add1_idx = tokens.index("ADD")
            mul1_idx = tokens.index("MUL", add1_idx)
            add2_idx = tokens.index("ADD", mul1_idx)
            mul2_idx = tokens.index("MUL", add2_idx)
            c2_tok = " ".join(tokens[add2_idx + 1 : mul2_idx])
            c2_groups.setdefault(c2_tok, []).append(i)
        except:
            pass

    if len(c2_groups) > 1:
        print(f"    C2 groups: {list(c2_groups.keys())}")
        separating_c2 = []
        for dim in range(64):
            group_means = {
                c2: np.mean([all_feats[i, dim] for i in idxs])
                for c2, idxs in c2_groups.items()
            }
            vals = list(group_means.values())
            spread = max(vals) - min(vals)
            if spread > 0.05:
                separating_c2.append((dim, spread, group_means))

        print(f"    Dims separating C2 (spread > 0.05): {len(separating_c2)}")
        separating_c2.sort(key=lambda x: -x[1])
        for dim, spread, gm in separating_c2[:5]:
            summary = {k: f"{v:.3f}" for k, v in sorted(gm.items())}
            print(f"      dim {dim:2d} (spread={spread:.3f}): {summary}")
    print()

# Final summary: feature density across all subtypes combined
print("=== OVERALL FEATURE DENSITY ===\n")
all_combined = []
subtypes_all = ["first_order_decay", "first_order_growth",
                "underdamped", "overdamped", "undamped"]
for st in subtypes_all:
    for seed in range(100):
        np.random.seed(seed * 7 + hash(st) % 1000)
        feat, _, _ = gen.generate(subtype=st, tier=1)
        all_combined.append(feat)

all_combined = np.array(all_combined)
active = int(np.sum(np.std(all_combined, axis=0) > 0.001))
max_val = np.max(np.abs(all_combined))
print(f"  Active dims: {active}/64  (was 5-6)")
print(f"  Max |feature value|: {max_val:.2f}")
print(f"  Dims with max > 50:  {int(np.sum(np.max(np.abs(all_combined), axis=0) > 50))}")
import numpy as np
from generators import get_generator
from features import compute_dynamical_features

gen = get_generator("linear_ode")

# Collect features for known second-order cases
subtypes_2nd = ["underdamped", "overdamped", "undamped"]

print("=== RAW FEATURE VECTORS (second-order subtypes, seed 0) ===\n")
for st in subtypes_2nd:
    np.random.seed(0)
    feat, tokens, _ = gen.generate(subtype=st, tier=1)
    
    # Count non-zero dims
    nonzero = int(np.sum(np.abs(feat) > 1e-6))
    
    print(f"  [{st}]")
    print(f"    tokens: {' '.join(tokens)}")
    print(f"    nonzero dims: {nonzero}/64")
    print(f"    feat[0:10]: {feat[0:10].tolist()}")
    print(f"    feat[10:20]: {feat[10:20].tolist()}")
    
    # Show ALL nonzero dims
    nonzero_dims = [(i, feat[i]) for i in range(64) if abs(feat[i]) > 1e-6]
    print(f"    all nonzero: {nonzero_dims}")
    print()

# Now the critical test: generate 50 samples of each second-order subtype
# and check feature variance WITHIN each subtype. If features don't vary
# when C1/C2 change, the model literally cannot recover them.
print("=== WITHIN-SUBTYPE FEATURE VARIANCE ===\n")
print("  (If std≈0 for a dim, that dim doesn't encode C1/C2 variation)\n")

for st in subtypes_2nd:
    all_feats = []
    all_tokens = []
    for seed in range(50):
        np.random.seed(seed * 7 + 13)
        feat, tokens, _ = gen.generate(subtype=st, tier=1)
        all_feats.append(feat)
        all_tokens.append(tokens)
    
    all_feats = np.array(all_feats)
    
    print(f"  [{st}] — {len(all_feats)} samples")
    # Show dims that actually vary
    for d in range(64):
        std = all_feats[:, d].std()
        if std > 0.001:
            print(f"    dim {d:2d}: mean={all_feats[:, d].mean():8.4f}  std={std:8.4f}  range=[{all_feats[:, d].min():.4f}, {all_feats[:, d].max():.4f}]")
    
    # Check: do different C1 values produce different features?
    # Group by C1 token
    c1_groups = {}
    for i, tokens in enumerate(all_tokens):
        try:
            add1_idx = tokens.index("ADD")
            mul1_idx = tokens.index("MUL", add1_idx)
            c1_tok = " ".join(tokens[add1_idx+1:mul1_idx])
            c1_groups.setdefault(c1_tok, []).append(i)
        except:
            pass
    
    if len(c1_groups) > 1:
        print(f"\n    C1 groups found: {list(c1_groups.keys())}")
        for dim in range(64):
            group_means = {}
            for c1_tok, indices in c1_groups.items():
                group_means[c1_tok] = np.mean([all_feats[i, dim] for i in indices])
            vals = list(group_means.values())
            spread = max(vals) - min(vals)
            if spread > 0.01:
                print(f"    dim {dim:2d} separates C1: {dict((k, f'{v:.4f}') for k,v in group_means.items())}")
    print()
"""
Run in galileo_minimal.
Generalization test for linear_ode tier 2.
Seeds 0-9, all subtypes. Reports exact match rate and error statistics
broken down by subtype and coefficient type.
"""
import numpy as np
import sys, torch
from pathlib import Path

sys.path.insert(0, ".")
from generators import get_generator
from vocab import Vocabulary
from model import SymbolicTransformer
from config import get_device, CONFIGS

device = get_device()
vocab  = Vocabulary.default()
cfg    = CONFIGS["linear_ode"]
gen    = get_generator("linear_ode")

# Load tier 2 model
ckpt_path = Path("checkpoints/dynamical/linear_ode/tier2/model_best.pt")
model = SymbolicTransformer(vocab_size=vocab.vocab_size, **cfg).to(device)
ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

def predict(features):
    feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = [vocab.bos_token]
        for _ in range(cfg["max_seq_len"]):
            tgt    = torch.tensor([generated], dtype=torch.long).to(device)
            logits = model(feat_t, tgt)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == vocab.eos_token:
                break
            generated.append(next_tok)
    return vocab.decode(generated[1:])

def parse_float_tokens(tokens):
    """Parse a CONST or INT_* token subsequence back to a float value."""
    s = ""
    neg = False
    is_const = False
    for t in tokens:
        if t == "CONST":
            is_const = True
        elif t == "NEG":
            neg = True
        elif t == "POINT":
            s += "."
        elif t.startswith("INT_"):
            return float(t.split("_")[1]) * (-1 if neg else 1)
        elif t.startswith("NEG_"):
            return -float(t.split("_")[1])
        elif t in "0123456789":
            s += t
    if not s:
        return None
    try:
        val = float(s) * (-1 if neg else 1)
        return val
    except:
        return None

def extract_ode_params(tokens):
    """Extract C1, C2 (or 'a') from token sequence."""
    token_list = list(tokens)
    if "D2X_DT2" in token_list:
        # Second order: D2X_DT2 ADD <C1> MUL DX_DT ADD <C2> MUL X EQ INT_0
        try:
            add1 = token_list.index("ADD")
            mul1 = token_list.index("MUL", add1)
            add2 = token_list.index("ADD", mul1)
            mul2 = token_list.index("MUL", add2)
            c1 = parse_float_tokens(token_list[add1+1:mul1])
            c2 = parse_float_tokens(token_list[add2+1:mul2])
            return {"order": 2, "C1": c1, "C2": c2}
        except:
            return {"order": 2, "C1": None, "C2": None}
    elif "DX_DT" in token_list:
        # First order: DX_DT EQ <a> MUL X
        try:
            eq_idx = token_list.index("EQ")
            mul_idx = token_list.index("MUL", eq_idx)
            a = parse_float_tokens(token_list[eq_idx+1:mul_idx])
            return {"order": 1, "a": a}
        except:
            return {"order": 1, "a": None}
    return {}

subtypes = ["first_order_decay", "first_order_growth",
            "underdamped", "overdamped", "undamped"]

# Per-subtype error tracking
errors_by_subtype = {st: {"c1_errs": [], "c2_errs": [], "a_errs": [], "exact": 0, "total": 0, "parse_fail": 0} for st in subtypes}
all_misses = []

print("=" * 80)
print("TIER 2 GENERALIZATION — seeds 0-9, all subtypes")
print("=" * 80)

for st in subtypes:
    print(f"\n  [{st}]")
    for seed in range(10):
        np.random.seed(seed)
        feat, tokens, _ = gen.generate(subtype=st, tier=2)
        pred_tokens = predict(feat)

        truth_str = " ".join(tokens)
        pred_str  = " ".join(pred_tokens)
        exact     = (truth_str == pred_str)

        truth_params = extract_ode_params(tokens)
        pred_params  = extract_ode_params(pred_tokens)

        errors_by_subtype[st]["total"] += 1
        if exact:
            errors_by_subtype[st]["exact"] += 1

        # Compute parameter errors
        if truth_params.get("order") == 1:
            a_t = truth_params.get("a")
            a_p = pred_params.get("a")
            if a_t is not None and a_p is not None:
                rel = abs(a_t - a_p) / (abs(a_t) + 1e-8)
                errors_by_subtype[st]["a_errs"].append(rel)
            else:
                errors_by_subtype[st]["parse_fail"] += 1
        elif truth_params.get("order") == 2:
            c1_t = truth_params.get("C1")
            c1_p = pred_params.get("C1")
            c2_t = truth_params.get("C2")
            c2_p = pred_params.get("C2")
            if c1_t is not None and c1_p is not None:
                # C1=0 special case: use absolute error
                if abs(c1_t) < 1e-6:
                    errors_by_subtype[st]["c1_errs"].append(abs(c1_p))
                else:
                    errors_by_subtype[st]["c1_errs"].append(abs(c1_t - c1_p) / abs(c1_t))
            if c2_t is not None and c2_p is not None:
                errors_by_subtype[st]["c2_errs"].append(abs(c2_t - c2_p) / (abs(c2_t) + 1e-8))

        # Print line
        status = "EXACT" if exact else "~"
        if not exact:
            # Show error magnitudes
            errs = []
            if truth_params.get("order") == 1:
                a_t = truth_params.get("a")
                a_p = pred_params.get("a")
                if a_t and a_p:
                    errs.append(f"a_err={abs(a_t-a_p)/abs(a_t):.4f}")
            else:
                c1_t, c1_p = truth_params.get("C1"), pred_params.get("C1")
                c2_t, c2_p = truth_params.get("C2"), pred_params.get("C2")
                if c1_t is not None and c1_p is not None:
                    if abs(c1_t) < 1e-6:
                        errs.append(f"C1_abs={abs(c1_p):.4f}")
                    else:
                        errs.append(f"C1_err={abs(c1_t-c1_p)/abs(c1_t):.4f}")
                if c2_t is not None and c2_p is not None:
                    errs.append(f"C2_err={abs(c2_t-c2_p)/abs(c2_t):.4f}")
            status += f"  [{', '.join(errs)}]"
            all_misses.append((st, seed, truth_str, pred_str))

        print(f"    seed{seed} {status}")
        if not exact:
            print(f"      truth: {truth_str}")
            print(f"      pred:  {pred_str}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY BY SUBTYPE")
print("=" * 80)

for st in subtypes:
    d = errors_by_subtype[st]
    print(f"\n  [{st}]")
    print(f"    Exact match: {d['exact']}/{d['total']}")
    if d["a_errs"]:
        print(f"    'a' rel error:  mean={np.mean(d['a_errs']):.4f}  max={np.max(d['a_errs']):.4f}")
    if d["c1_errs"]:
        print(f"    C1 rel error:   mean={np.mean(d['c1_errs']):.4f}  max={np.max(d['c1_errs']):.4f}")
    if d["c2_errs"]:
        print(f"    C2 rel error:   mean={np.mean(d['c2_errs']):.4f}  max={np.max(d['c2_errs']):.4f}")
    if d["parse_fail"]:
        print(f"    Parse failures: {d['parse_fail']}")

# Overall
all_c1 = [e for st in subtypes for e in errors_by_subtype[st]["c1_errs"]]
all_c2 = [e for st in subtypes for e in errors_by_subtype[st]["c2_errs"]]
all_a  = [e for st in subtypes for e in errors_by_subtype[st]["a_errs"]]
total_exact = sum(errors_by_subtype[st]["exact"] for st in subtypes)
total_count = sum(errors_by_subtype[st]["total"] for st in subtypes)

print(f"\n  OVERALL")
print(f"    Exact match: {total_exact}/{total_count}")
print(f"    All within threshold (0.25): ", end="")
all_ok = True
if all_a  and max(all_a)  > 0.25: all_ok = False
if all_c1 and max(all_c1) > 0.25: all_ok = False
if all_c2 and max(all_c2) > 0.25: all_ok = False
print("YES" if all_ok else "NO")
if all_a:
    print(f"    'a':  mean={np.mean(all_a):.4f}  max={np.max(all_a):.4f}")
if all_c1:
    print(f"    C1:   mean={np.mean(all_c1):.4f}  max={np.max(all_c1):.4f}")
if all_c2:
    print(f"    C2:   mean={np.mean(all_c2):.4f}  max={np.max(all_c2):.4f}")
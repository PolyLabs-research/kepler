"""
Run in galileo_minimal.
Tests linear_ode on parameter values the model hasn't memorized:
  - Seeds 2-9 for each subtype (test.py only used seeds 0-1)
  - Checks whether predictions are exact or approximate
  - Separates "seen-style" (integer C1/C2) from "unseen-style" (CONST float C1/C2)
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
vocab = Vocabulary.default()
cfg = CONFIGS["linear_ode"]
gen = get_generator("linear_ode")

# Load model
checkpoint_dir = Path("checkpoints/dynamical/linear_ode/tier1")
ckpt_path = checkpoint_dir / "model_best.pt"
model = SymbolicTransformer(vocab_size=vocab.vocab_size, **cfg).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

def predict(features):
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = [vocab.bos_token]
        for _ in range(cfg["max_seq_len"]):
            tgt = torch.tensor([generated], dtype=torch.long).to(device)
            logits = model(feat_tensor, tgt)
            next_token = logits[0, -1].argmax().item()
            if next_token == vocab.eos_token:
                break
            generated.append(next_token)
    return vocab.decode(generated[1:])  # strip BOS

subtypes = ["first_order_decay", "first_order_growth",
            "underdamped", "overdamped", "undamped"]

exact_match = 0
approx_match = 0  # structurally correct but wrong coefficients
total = 0
exact_by_type = {"int_coeff": 0, "float_coeff": 0}
total_by_type = {"int_coeff": 0, "float_coeff": 0}

print("=" * 80)
print("GENERALIZATION TEST — seeds 0-9, all subtypes")
print("=" * 80)

for st in subtypes:
    print(f"\n  [{st}]")
    for seed in range(10):
        np.random.seed(seed)
        feat, tokens, _ = gen.generate(subtype=st, tier=1)
        pred_tokens = predict(feat)
        
        truth_str = " ".join(tokens)
        pred_str  = " ".join(pred_tokens)
        
        match = "EXACT" if truth_str == pred_str else "MISS "
        
        # Classify as int_coeff or float_coeff
        has_const = "CONST" in tokens
        coeff_type = "float_coeff" if has_const else "int_coeff"
        
        total += 1
        total_by_type[coeff_type] += 1
        if match == "EXACT":
            exact_match += 1
            exact_by_type[coeff_type] += 1
        
        # Only print misses and a few hits for brevity
        if match == "MISS " or seed < 2:
            print(f"    seed{seed} [{match}] ({coeff_type})")
            print(f"      truth: {truth_str}")
            if match == "MISS ":
                print(f"      pred:  {pred_str}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n  Overall:      {exact_match}/{total} exact match ({100*exact_match/total:.0f}%)")
print(f"  Int coeffs:   {exact_by_type['int_coeff']}/{total_by_type['int_coeff']} exact ({100*exact_by_type['int_coeff']/max(1,total_by_type['int_coeff']):.0f}%)")
print(f"  Float coeffs: {exact_by_type['float_coeff']}/{total_by_type['float_coeff']} exact ({100*exact_by_type['float_coeff']/max(1,total_by_type['float_coeff']):.0f}%)")
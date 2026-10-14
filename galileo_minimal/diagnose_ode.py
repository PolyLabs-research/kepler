import numpy as np
from generators import get_generator
from vocab import Vocabulary

gen = get_generator("linear_ode")
vocab = Vocabulary.default()

# --- 1. SUBTYPE DISTRIBUTION over 1000 samples ---
subtype_counts = {}
for _ in range(1000):
    _, _, st = gen.generate(tier=1)
    subtype_counts[st] = subtype_counts.get(st, 0) + 1

print("=== SUBTYPE DISTRIBUTION (1000 samples) ===")
for st, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
    print(f"  {st:30s} {count:4d}  ({count/10:.1f}%)")

# --- 2. TOKEN-LEVEL breakdown: first-order vs second-order ---
first_order_count = 0
second_order_count = 0
second_order_token_seqs = []  # collect raw token sequences

for i in range(200):
    np.random.seed(i)
    _, tokens, st = gen.generate(tier=1)
    if "D2X_DT2" in tokens:
        second_order_count += 1
        second_order_token_seqs.append((st, tokens))
    else:
        first_order_count += 1

print(f"\n=== ORDER SPLIT (200 seeded samples) ===")
print(f"  First-order:  {first_order_count}")
print(f"  Second-order: {second_order_count}")

# --- 3. Show actual second-order token sequences ---
print(f"\n=== SECOND-ORDER TOKEN SEQUENCES (first 10) ===")
for st, tokens in second_order_token_seqs[:10]:
    token_str = " ".join(tokens)
    encoded = [vocab.bos_token] + vocab.encode(tokens) + [vocab.eos_token]
    # Flag any UNK tokens
    unk_positions = [i for i, t in enumerate(tokens) if vocab.token_to_id.get(t) is None]
    unk_flag = f"  *** UNK at positions {unk_positions}: {[tokens[i] for i in unk_positions]}" if unk_positions else ""
    print(f"  [{st}]")
    print(f"    tokens:  {token_str}")
    print(f"    ids:     {encoded}")
    print(f"    length:  {len(encoded)}{unk_flag}")

# --- 4. Check sequence length distribution for second-order ---
print(f"\n=== SECOND-ORDER SEQUENCE LENGTHS ===")
lengths = []
for st, tokens in second_order_token_seqs:
    encoded = [vocab.bos_token] + vocab.encode(tokens) + [vocab.eos_token]
    lengths.append(len(encoded))
if lengths:
    print(f"  min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")
    print(f"  max_seq_len in config = 64")
    print(f"  Any truncated? {any(l > 64 for l in lengths)}")

# --- 5. Check C1=0 cases (undamped) ---
print(f"\n=== C1=0 CASES (undamped) ===")
undamped_seqs = [(st, t) for st, t in second_order_token_seqs if st == "undamped"]
for st, tokens in undamped_seqs[:3]:
    print(f"  {' '.join(tokens)}")

# --- 6. Binary data check: what's actually stored in the compiled data ---
print(f"\n=== BINARY DATA TOKEN DISTRIBUTION ===")
try:
    import json
    from pathlib import Path
    meta_path = Path("data/linear_ode/tier1/metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        targets = np.memmap("data/linear_ode/tier1/targets.bin", dtype='int32', mode='r',
                           shape=(meta["n_samples"], meta["max_seq_len"]))
        
        # Sample 1000 sequences, check how many are second-order
        d2x_id = vocab.token_to_id.get("D2X_DT2")
        dx_id = vocab.token_to_id.get("DX_DT")
        const_id = vocab.token_to_id.get("CONST")
        
        second_in_data = 0
        first_in_data = 0
        const_in_second = 0
        
        sample_indices = np.random.choice(meta["n_samples"], min(2000, meta["n_samples"]), replace=False)
        for idx in sample_indices:
            seq = targets[idx]
            if d2x_id in seq:
                second_in_data += 1
                if const_id in seq:
                    const_in_second += 1
            elif dx_id in seq:
                first_in_data += 1
        
        print(f"  Sampled {len(sample_indices)} from {meta['n_samples']} total")
        print(f"  First-order in data:  {first_in_data} ({first_in_data/len(sample_indices)*100:.1f}%)")
        print(f"  Second-order in data: {second_in_data} ({second_in_data/len(sample_indices)*100:.1f}%)")
        print(f"  Second-order WITH CONST token: {const_in_second} ({const_in_second/max(1,second_in_data)*100:.1f}% of 2nd order)")
        
        # Show a few raw second-order sequences from binary data
        print(f"\n  Raw second-order sequences from binary data (first 5):")
        count = 0
        for idx in sample_indices:
            seq = targets[idx]
            if d2x_id in seq and count < 5:
                # Decode non-pad tokens
                non_pad = [int(t) for t in seq if t != vocab.pad_token]
                decoded = vocab.decode(non_pad[1:-1])  # strip BOS/EOS
                print(f"    {' '.join(decoded)}")
                count += 1
    else:
        print("  No binary data found at data/linear_ode/tier1/")
except Exception as e:
    print(f"  Error: {e}")
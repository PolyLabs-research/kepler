import os
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from generators import get_generator
from vocab import Vocabulary
from config import CONFIGS

def _generate_batch(args):
    """Worker function for a single process batch."""
    model_type, tier, n_samples_local = args
    generator = get_generator(model_type)
    vocab = Vocabulary.default()
    cfg = CONFIGS[model_type]
    
    batch_features = []
    batch_targets = []
    
    for _ in range(n_samples_local):
        feat, tokens, _ = generator.generate(tier=tier)
        ids = [vocab.bos_token] + vocab.encode(tokens) + [vocab.eos_token]
        # Pad/Truncate
        ids = ids[:cfg["max_seq_len"]]
        ids += [vocab.pad_token] * (cfg["max_seq_len"] - len(ids))
        
        batch_features.append(feat)
        batch_targets.append(ids)
        
    return np.array(batch_features, dtype='float32'), np.array(batch_targets, dtype='int32')

def compile_dataset_parallel(model_type: str, tier: int = 1, n_samples: int = 100000):
    cfg = CONFIGS[model_type]
    out_dir = Path(f"data/{model_type}/tier{tier}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate chunks
    n_workers = max(1, cpu_count() - 2) # Leave 2 cores for system stability
    samples_per_worker = n_samples // n_workers
    tasks = [(model_type, tier, samples_per_worker) for _ in range(n_workers)]
    
    print(f"🚀 Parallel Compiling {n_samples} samples (Tier {tier}) using {n_workers} CPU cores...")
    
    # Memory-mapped pre-allocation
    features_mm = np.memmap(out_dir / "features.bin", dtype='float32', mode='w+', shape=(n_samples, cfg["feature_dim"]))
    targets_mm = np.memmap(out_dir / "targets.bin", dtype='int32', mode='w+', shape=(n_samples, cfg["max_seq_len"]))

    current_idx = 0
    with Pool(n_workers) as pool:
        for i, (f_batch, t_batch) in enumerate(pool.imap_unordered(_generate_batch, tasks)):
            batch_size = f_batch.shape[0]
            features_mm[current_idx : current_idx + batch_size] = f_batch
            targets_mm[current_idx : current_idx + batch_size] = t_batch
            current_idx += batch_size
            print(f"  Done {(current_idx/n_samples)*100:.1f}%...")

    features_mm.flush()
    targets_mm.flush()
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "n_samples": n_samples,
            "feature_dim": cfg["feature_dim"],
            "max_seq_len": cfg["max_seq_len"],
            "tier": tier
        }, f)

    print(f"✅ Fast Compilation complete: {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3], help="Difficulty tier (1=easy, 2=medium, 3=hard)")
    parser.add_argument("--samples", type=int, default=100000)
    args = parser.parse_args()
    compile_dataset_parallel(args.model, args.tier, args.samples)

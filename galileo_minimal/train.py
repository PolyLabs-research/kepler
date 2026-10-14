#!/usr/bin/env python3
"""
train.py - Unified training script with tier support
"""
import os
import argparse
import sys
import json
import signal
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# NVIDIA ADA Optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config import get_device, auto_batch_size, CONFIGS, TRAINING, DOMAINS
from vocab import Vocabulary
from model import SymbolicTransformer
from generators import get_generator

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, model_type: str, tier: int = 1):
        data_dir = Path(f"data/{model_type}/tier{tier}")
        
        # Fallback to legacy non-tier path if tier path doesn't exist
        if not data_dir.exists():
            legacy_dir = Path(f"data/{model_type}")
            if legacy_dir.exists():
                print(f"⚠️ Using legacy data path (no tier). Consider migrating to tier structure.")
                data_dir = legacy_dir
            else:
                raise FileNotFoundError(f"No data found at {data_dir} or legacy path")
        
        with open(data_dir / "metadata.json", "r") as f:
            self.meta = json.load(f)
            
        self.features = np.memmap(
            data_dir / "features.bin", dtype='float32', mode='r', 
            shape=(self.meta["n_samples"], self.meta["feature_dim"])
        )
        self.targets = np.memmap(
            data_dir / "targets.bin", dtype='int32', mode='r', 
            shape=(self.meta["n_samples"], self.meta["max_seq_len"])
        )

    def __len__(self): return self.meta["n_samples"]
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx].copy()).float(),
            torch.from_numpy(self.targets[idx].copy()).long()
        )

class OnlineDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator, vocab, max_seq_len: int, tier: int = 1):
        self.generator = generator
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.tier = tier
    
    def __iter__(self):
        while True:
            features, tokens, _ = self.generator.generate(tier=self.tier)
            ids = [self.vocab.bos_token] + self.vocab.encode(tokens) + [self.vocab.eos_token]
            ids = ids[:self.max_seq_len] + [self.vocab.pad_token] * max(0, self.max_seq_len - len(ids))
            yield torch.tensor(features, dtype=torch.float32), torch.tensor(ids, dtype=torch.long)

def train(args):
    device = get_device()
    cfg, domain, vocab = CONFIGS[args.model], DOMAINS[args.model], Vocabulary.default()
    
    # Tier-scoped checkpoint directory
    checkpoint_dir = Path("checkpoints") / domain / args.model / f"tier{args.tier}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save vocab with checkpoint for future-proofing
    vocab.save(checkpoint_dir / "vocab.json")
    
    # Dataset Selection
    data_dir = Path(f"data/{args.model}/tier{args.tier}")
    legacy_dir = Path(f"data/{args.model}")
    
    if (data_dir / "features.bin").exists():
        print(f"📦 Found tier {args.tier} binary data. Using BinaryDataset.", flush=True)
        dataset, num_workers = BinaryDataset(args.model, args.tier), 2
    elif (legacy_dir / "features.bin").exists():
        print(f"⚠️ Found legacy binary data (no tier). Using BinaryDataset.", flush=True)
        dataset, num_workers = BinaryDataset(args.model, args.tier), 2
    else:
        print(f"⚠️ No binary data found. Using OnlineDataset (tier {args.tier}).", flush=True)
        dataset = OnlineDataset(get_generator(args.model), vocab, cfg["max_seq_len"], args.tier)
        num_workers = 0

    batch_size = args.batch_size or auto_batch_size(512)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    model = SymbolicTransformer(vocab_size=vocab.vocab_size, **cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token)

    # RESUME LOGIC - try tier-specific first, then legacy
    step, best_loss = 0, float("inf")
    resume_path = checkpoint_dir / "model_best.pt"
    
    # Legacy fallback
    if args.resume and not resume_path.exists():
        legacy_checkpoint = Path("checkpoints") / domain / args.model / "model_best.pt"
        if legacy_checkpoint.exists():
            print(f"⚠️ No tier checkpoint found. Attempting legacy checkpoint: {legacy_checkpoint}")
            resume_path = legacy_checkpoint
    
    if args.resume and resume_path.exists():
        print(f"🔄 Resuming from {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        best_loss = ckpt.get("loss", float("inf"))

    running_loss, interrupted = 0.0, False
    def handle_interrupt(sig, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, handle_interrupt)

    print(f"🚀 Training {args.model} (Tier {args.tier}) on {device} from step {step}...")
    model.train()
    data_iter = iter(loader)
    max_steps = args.steps or TRAINING["max_steps"]

    while step < max_steps and not interrupted:
        try: features, targets = next(data_iter)
        except StopIteration: data_iter = iter(loader); features, targets = next(data_iter)

        features, targets = features.to(device), targets.to(device)
        lr = TRAINING["learning_rate"] * min(1.0, step / TRAINING["warmup_steps"])
        for pg in optimizer.param_groups: pg["lr"] = lr

        logits = model(features, targets[:, :-1])
        loss = criterion(logits.reshape(-1, vocab.vocab_size), targets[:, 1:].reshape(-1))

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["gradient_clip"])
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % TRAINING["log_interval"] == 0:
            avg = running_loss / TRAINING["log_interval"]
            print(f"[{step}/{max_steps}] loss={loss.item():.4f} avg={avg:.4f} lr={lr:.2e}")
            if avg < best_loss:
                best_loss = avg
                torch.save({
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "step": step, 
                    "loss": avg,
                    "tier": args.tier
                }, checkpoint_dir / "model_best.pt")
                print(f"  💾 Saved BEST checkpoint")
            running_loss = 0.0

    final_name = "model_interrupted.pt" if interrupted else "model_final.pt"
    torch.save({"model": model.state_dict(), "step": step, "tier": args.tier}, checkpoint_dir / final_name)
    print(f"✅ Training finished. Checkpoint: {checkpoint_dir / final_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3], help="Difficulty tier (1=easy, 2=medium, 3=hard)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    train(parser.parse_args())

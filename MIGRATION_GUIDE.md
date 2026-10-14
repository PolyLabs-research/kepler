# Galileo Minimal Tier Migration Guide

## Overview

This guide walks you through migrating your existing Galileo Minimal system to the new tier-aware architecture. The migration is designed to be **safe and reversible**, with backward compatibility during the transition.

## What Changed

### High-Level Summary

1. **Tier plumbing added everywhere** - All scripts now accept `--tier` parameter (default=1)
2. **Directory structure is tier-scoped** - Data and checkpoints organized by tier
3. **Generator API includes tier parameter** - All generators accept `tier` kwarg
4. **Tier meanings standardized** - Consistent difficulty progression across all models
5. **Linear ODE second-order fixed** - Now encodes actual coefficients instead of template
6. **Tests are deterministic** - All test cases have explicit seeds
7. **Backward compatibility** - Legacy checkpoints and data still work

### Tier Definitions (Now Consistent)

- **Tier 1**: Discrete parameters (snap to integers/fractions), short sequences, low noise
- **Tier 2**: Continuous parameters (2dp precision), medium complexity, stable
- **Tier 3**: Wider ranges, more noise, edge cases, longer expressions

---

## Pre-Migration Checklist

Before you start, ensure you have:

- [ ] A backup of your `checkpoints/` directory
- [ ] A backup of your `data/` directory (if you have precompiled data)
- [ ] Git commit or zip of your current working state
- [ ] Tested that your current system works (run a test to verify)

### Creating a Backup

```bash
# Create timestamped backup
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp -r checkpoints/ $BACKUP_DIR/
cp -r data/ $BACKUP_DIR/  # Only if you have precompiled data
echo "Backup created at $BACKUP_DIR"
```

---

## Migration Steps

### Step 1: Replace Core Files

Replace these files with the new versions:

- `compile_data.py`
- `train.py`
- `test.py`
- `generators_base.py`
- `generators.py`
- `generators_dynamical.py`

**Action:**
```bash
# If you have the new files in a migration folder:
cp migration/*.py .
```

### Step 2: Migrate Existing Checkpoints (Optional but Recommended)

Your existing checkpoints will work as-is due to legacy fallback, but for cleanliness, move them to tier1 structure:

```bash
# For each model you've trained:
cd checkpoints/algebraic/polynomial/
mkdir -p tier1
mv model_best.pt tier1/
mv model_final.pt tier1/ 2>/dev/null || true
cd ../../..

# Repeat for other models:
# checkpoints/timeseries/ar/
# checkpoints/dynamical/linear_ode/
# etc.
```

**Example script to automate:**

```bash
#!/bin/bash
# migrate_checkpoints.sh

for domain in checkpoints/*/; do
    for model in $domain*/; do
        if [ -f "$model/model_best.pt" ]; then
            echo "Migrating $model"
            mkdir -p "$model/tier1"
            mv "$model/model_best.pt" "$model/tier1/" 2>/dev/null
            mv "$model/model_final.pt" "$model/tier1/" 2>/dev/null
        fi
    done
done

echo "✅ Checkpoint migration complete"
```

### Step 3: Migrate Existing Data (Optional)

If you have precompiled binary data, move it to tier1:

```bash
# For each model with precompiled data:
cd data/polynomial/
mkdir -p tier1
mv features.bin tier1/
mv targets.bin tier1/
mv metadata.json tier1/
cd ../..

# Update metadata.json to include tier field
# Add: "tier": 1
```

**Or just delete and recompile:**

```bash
# Simpler approach - recompile with tier awareness
rm -rf data/polynomial/
python compile_data.py --model polynomial --tier 1 --samples 100000
```

### Step 4: Verify Legacy Compatibility

Test that your old checkpoints still work:

```bash
# This should work even without moving checkpoints
python test.py --model polynomial --tier 1

# You should see:
# "⚠️ No tier checkpoint found, using legacy: checkpoints/algebraic/polynomial/model_best.pt"
```

### Step 5: Train New Tier-Aware Models

Now train new models with explicit tier awareness:

```bash
# Tier 1 (easy) - good for initial testing
python train.py --model polynomial --tier 1 --steps 5000

# Tier 2 (medium) - main production tier
python train.py --model polynomial --tier 2 --steps 10000

# Tier 3 (hard) - edge cases and robustness
python train.py --model polynomial --tier 3 --steps 15000
```

### Step 6: Recompile Data (If Using Binary Data)

If you prefer precompiled data over online generation:

```bash
# Compile data for all tiers
python compile_data.py --model polynomial --tier 1 --samples 100000
python compile_data.py --model polynomial --tier 2 --samples 100000
python compile_data.py --model polynomial --tier 3 --samples 100000
```

**Directory structure after compilation:**

```
data/
  polynomial/
    tier1/
      features.bin
      targets.bin
      metadata.json
    tier2/
      features.bin
      targets.bin
      metadata.json
    tier3/
      features.bin
      targets.bin
      metadata.json
```

### Step 7: Test All Tiers

Run deterministic tests across all tiers:

```bash
python test.py --model polynomial --tier 1
python test.py --model polynomial --tier 2
python test.py --model polynomial --tier 3
```

---

## Model-Specific Migration Notes

### Polynomial Generator

**What Changed:**
- Tier 1: Discrete coefficients from {-3, -2, -1, 0, 1, 2, 3}
- Tier 2: Continuous coefficients in [-3, 3] with 2dp
- Tier 3: Wider range [-5, 5] + noise

**Action Required:** None (automatic)

**Test Output:** You should see snapped coefficients in Tier 1, smooth coefficients in Tier 2/3

---

### AR Generator

**What Changed:**
- Tier 1: Discrete phi values like {0.3, 0.5, 0.7, 0.9}
- Tier 2: Continuous phi with 2dp
- Tier 3: Wider range + higher noise

**Action Required:** None (automatic)

---

### Transcendental Generator

**What Changed:**
- Tier 1: Parameters snap to nice values (0.5, 1.0, 1.5, 2.0)
- Tier 2: Continuous parameters
- Tier 3: Wider ranges + noise

**Action Required:** None (automatic)

---

### Physics Laws Generator

**What Changed:**
- Tier 1: Small integer ranges [1, 5], discrete constants
- Tier 2: Continuous [1, 10]
- Tier 3: Wide range [0.1, 15] + noise

**Action Required:** None (automatic)

---

### Linear ODE Generator ⚠️ BREAKING CHANGE

**What Changed (PHASE 4 FIX):**

Second-order ODEs now encode **actual coefficients** instead of fixed template:

**Old (broken):**
```
D2X_DT2 ADD DX_DT ADD X EQ INT_0
```
All underdamped/overdamped/undamped looked identical!

**New (correct):**
```
D2X_DT2 ADD <C1> MUL DX_DT ADD <C2> MUL X EQ INT_0
```
Where C1 = 2*zeta*omega, C2 = omega^2

**Action Required:**

1. **Retrain all linear_ode models** - old checkpoints won't work for second-order
2. **Delete old linear_ode checkpoints:**
   ```bash
   rm -rf checkpoints/dynamical/linear_ode/
   ```
3. **Retrain from scratch:**
   ```bash
   python train.py --model linear_ode --tier 1 --steps 10000
   python train.py --model linear_ode --tier 2 --steps 10000
   python train.py --model linear_ode --tier 3 --steps 10000
   ```

**Why This Matters:**

The old approach let the model "cheat" by memorizing templates. The new approach forces actual coefficient recovery, making it consistent with other generators.

**New Test Comparator:**

The test script now parses C1 and C2 from tokens and checks relative error:

```python
# test.py now does this for second-order:
truth_C1, truth_C2 = extract_coefficients(truth_tokens)
pred_C1, pred_C2 = extract_coefficients(pred_tokens)
rel_C1 = abs(truth_C1 - pred_C1) / abs(truth_C1)
rel_C2 = abs(truth_C2 - pred_C2) / abs(truth_C2)
match = (rel_C1 < 0.25) and (rel_C2 < 0.25)
```

---

## Workflow After Migration

### Daily Usage

**Compile data for a tier:**
```bash
python compile_data.py --model polynomial --tier 2 --samples 100000
```

**Train on a tier:**
```bash
python train.py --model polynomial --tier 2 --steps 10000
```

**Test a tier:**
```bash
python test.py --model polynomial --tier 2
```

**Resume training:**
```bash
python train.py --model polynomial --tier 2 --resume --steps 20000
```

### Multi-Tier Training Strategy

```bash
#!/bin/bash
# train_all_tiers.sh

MODEL="polynomial"

for TIER in 1 2 3; do
    echo "🚀 Training $MODEL Tier $TIER"
    python train.py --model $MODEL --tier $TIER --steps 10000
    python test.py --model $MODEL --tier $TIER
done
```

---

## Troubleshooting

### Problem: "No checkpoint found"

**Cause:** You're trying to test tier 2/3 but only have legacy (non-tier) checkpoint

**Solution:**
```bash
# Train that tier first
python train.py --model polynomial --tier 2 --steps 5000

# Or test tier 1 with legacy fallback
python test.py --model polynomial --tier 1
```

---

### Problem: "Could not parse ODE coefficients"

**Cause:** You're testing an old linear_ode checkpoint that uses the old template format

**Solution:**
```bash
# Delete old checkpoint and retrain
rm -rf checkpoints/dynamical/linear_ode/
python train.py --model linear_ode --tier 1 --steps 10000
```

---

### Problem: Tests pass with legacy but fail with new tier checkpoints

**Cause:** Tier mismatch - you compiled tier 1 data but are testing tier 2

**Solution:**
```bash
# Ensure tier consistency:
python compile_data.py --model polynomial --tier 2 --samples 100000
python train.py --model polynomial --tier 2 --steps 10000
python test.py --model polynomial --tier 2
```

---

### Problem: "Legacy data path" warning during training

**This is expected** during transition. To clean it up:

```bash
# Move legacy data to tier1
cd data/polynomial/
mkdir tier1
mv features.bin targets.bin metadata.json tier1/
cd ../..
```

---

### Problem: All test cases fail for linear_ode

**Cause:** Using old checkpoint with new test comparator

**Solution:**
```bash
# Retrain from scratch with new tokenization
rm -rf checkpoints/dynamical/linear_ode/
python train.py --model linear_ode --tier 1 --steps 15000
```

Note: linear_ode may take longer to converge now because it's learning actual coefficients instead of templates.

---

## Verification Checklist

After migration, verify:

- [ ] `python test.py --model polynomial --tier 1` passes (using legacy or new checkpoint)
- [ ] New tier-aware training works: `python train.py --model polynomial --tier 2 --steps 1000`
- [ ] Directory structure is correct: `ls checkpoints/algebraic/polynomial/tier2/`
- [ ] Vocab is saved with checkpoints: `ls checkpoints/algebraic/polynomial/tier1/vocab.json`
- [ ] linear_ode second-order shows distinct C1/C2 values in test output
- [ ] Tests are deterministic (same seed gives same result)

---

## Rollback Procedure

If you need to revert:

```bash
# Restore from backup
BACKUP_DIR="backup_YYYYMMDD_HHMMSS"  # Your backup timestamp
rm -rf checkpoints/ data/
cp -r $BACKUP_DIR/checkpoints/ .
cp -r $BACKUP_DIR/data/ .  # If you had precompiled data

# Restore old Python files from git
git checkout HEAD~1 *.py

# Or restore from your pre-migration zip/backup
```

---

## Benefits of Tier System

After migration, you gain:

1. **Curriculum learning** - Train on easy examples first, then harder
2. **Controlled difficulty** - Know exactly what complexity level you're testing
3. **No silent mixing** - Tier 1 data never contaminates Tier 3 training
4. **Reproducible tests** - Seeded test cases give consistent results
5. **True parameter recovery** - linear_ode can't cheat with templates anymore
6. **Future-proof** - Easy to add Tier 4/5 later if needed

---

## Next Steps

1. **Complete the migration** following steps 1-7 above
2. **Retrain linear_ode** if you use it (required due to PHASE 4 fix)
3. **Experiment with curriculum learning** (Tier 1 → 2 → 3)
4. **Add your own test cases** with seeds for reproducibility
5. **Consider per-tier evaluation** to understand model capabilities

---

## Questions?

Common gotchas:

- **"Why do my polynomial tests fail now?"** → You may need higher thresholds for Tier 3 due to noise
- **"Can I train Tier 2 with Tier 1 checkpoint?"** → No, tiers are independent (but you can use tier=1 as warm start)
- **"Do I need to compile data for all tiers?"** → No, online generation works fine. Compilation is optional for speed.

---

## Appendix: Manual File Changes (If Not Using Provided Files)

If you need to manually update your existing files:

### compile_data.py

**Line 7:** Add tier parameter to worker function:
```python
def _generate_batch(args):
    model_type, tier, n_samples_local = args  # Added tier
```

**Line 16:** Pass tier to generator:
```python
feat, tokens, _ = generator.generate(tier=tier)  # Added tier=tier
```

**Line 25:** Add tier parameter to main function:
```python
def compile_dataset_parallel(model_type: str, tier: int = 1, n_samples: int = 100000):
```

**Line 27:** Update output path:
```python
out_dir = Path(f"data/{model_type}/tier{tier}")  # Added /tier{tier}
```

**Line 56:** Update argparse:
```python
parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
```

### Similar updates needed for train.py, test.py, generators.py, etc.

**See provided files for complete implementations.**

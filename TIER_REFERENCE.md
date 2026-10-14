# Tier System Quick Reference

## Command Templates

### Compile Data
```bash
python compile_data.py --model <MODEL> --tier <1|2|3> --samples <N>
```

### Train Model
```bash
python train.py --model <MODEL> --tier <1|2|3> [--steps N] [--resume]
```

### Test Model
```bash
python test.py --model <MODEL> --tier <1|2|3> [--no-cleanup]
```

---

## Tier Characteristics

| Aspect | Tier 1 | Tier 2 | Tier 3 |
|--------|--------|--------|--------|
| **Parameters** | Discrete (snap to integers) | Continuous (2dp) | Wide range |
| **Noise** | None | Low | Medium |
| **Coefficients** | {-3,-2,-1,0,1,2,3} | [-3, 3] continuous | [-5, 5] continuous |
| **Purpose** | Easy validation | Main training | Robustness |
| **Use Case** | Quick testing | Production | Edge cases |

---

## Directory Structure

```
project/
├── data/
│   └── polynomial/
│       ├── tier1/
│       │   ├── features.bin
│       │   ├── targets.bin
│       │   └── metadata.json
│       ├── tier2/
│       └── tier3/
│
├── checkpoints/
│   └── algebraic/
│       └── polynomial/
│           ├── tier1/
│           │   ├── model_best.pt
│           │   ├── model_final.pt
│           │   └── vocab.json
│           ├── tier2/
│           └── tier3/
```

---

## Model-Specific Tier Rules

### Polynomial
- **Tier 1**: Integer coefficients only
- **Tier 2**: Float coefficients with 2 decimal places
- **Tier 3**: Wider coefficient range + 5% noise

### AR (Autoregressive)
- **Tier 1**: phi ∈ {0.3, 0.5, 0.7, 0.9}
- **Tier 2**: phi ∈ [0.1, 0.95] continuous
- **Tier 3**: phi ∈ [0.05, 0.98] + higher noise

### Transcendental
- **Tier 1**: Parameters snap to {0.5, 1.0, 1.5, 2.0}
- **Tier 2**: Continuous parameters
- **Tier 3**: Wider ranges + 2% noise

### Physics Laws
- **Tier 1**: Variables in [1, 5], discrete constants
- **Tier 2**: Variables in [1, 10], 2dp constants
- **Tier 3**: Variables in [0.1, 15] + 2% noise

### Linear ODE
- **Tier 1**: Discrete omega ∈ {1,2,3,4}, discrete zeta
- **Tier 2**: Continuous omega ∈ [1,5], continuous zeta
- **Tier 3**: Wide omega ∈ [0.5,6] + noise in trajectory

**⚠️ Important:** Second-order ODEs now encode C1 and C2 coefficients explicitly

---

## Example Workflows

### Basic Training Pipeline
```bash
# 1. Compile data (optional)
python compile_data.py --model polynomial --tier 1 --samples 100000

# 2. Train
python train.py --model polynomial --tier 1 --steps 10000

# 3. Test
python test.py --model polynomial --tier 1
```

### Multi-Tier Progression
```bash
# Train easy first
python train.py --model polynomial --tier 1 --steps 5000

# Then medium
python train.py --model polynomial --tier 2 --steps 10000

# Finally hard
python train.py --model polynomial --tier 3 --steps 15000

# Test all tiers
for tier in 1 2 3; do
    python test.py --model polynomial --tier $tier
done
```

### Resume Training
```bash
# Train for 5k steps
python train.py --model polynomial --tier 2 --steps 5000

# Continue for 5k more (total 10k)
python train.py --model polynomial --tier 2 --steps 10000 --resume
```

---

## Common Errors and Fixes

### "No checkpoint found"
**Fix:** Train that tier first
```bash
python train.py --model polynomial --tier 2 --steps 5000
```

### "Could not parse ODE coefficients"
**Fix:** Retrain linear_ode (old format is incompatible)
```bash
rm -rf checkpoints/dynamical/linear_ode/
python train.py --model linear_ode --tier 1 --steps 10000
```

### Tests fail but loss is low
**Fix:** Check tier consistency
```bash
# Ensure data, training, and testing use same tier
python compile_data.py --model polynomial --tier 2 --samples 100000
python train.py --model polynomial --tier 2 --steps 10000
python test.py --model polynomial --tier 2
```

---

## Test Case Seeds (Phase 5)

All test cases now have explicit seeds for reproducibility:

```python
# Example from TEST_CASES
{"name": "cubic_seed0", "subtype": "cubic", "seed": 0, "threshold": 0.15}
{"name": "cubic_seed1", "subtype": "cubic", "seed": 1, "threshold": 0.15}
```

Running the same test twice gives identical results.

---

## Backward Compatibility

Legacy (non-tier) checkpoints and data still work:

```bash
# If you have: checkpoints/algebraic/polynomial/model_best.pt
# This still works:
python test.py --model polynomial --tier 1

# You'll see:
# ⚠️ No tier checkpoint found, using legacy: checkpoints/algebraic/polynomial/model_best.pt
```

---

## Performance Tips

1. **Use precompiled data for large-scale training**
   - Online generation: ~500 samples/sec
   - Binary mmap: ~50,000 samples/sec

2. **Tier 1 trains fastest** (discrete parameters, no noise)
3. **Tier 3 needs more steps** (noise and edge cases are harder)
4. **VRAM auto-scales** - batch size adjusts to your GPU

---

## Tier Selection Guide

**Use Tier 1 when:**
- Quick validation of architecture changes
- Debugging tokenization/feature extraction
- Initial model development

**Use Tier 2 when:**
- Main production training
- Most realistic use cases
- Best balance of difficulty and speed

**Use Tier 3 when:**
- Testing robustness to noise
- Edge case handling
- Final model hardening

---

## Metrics by Tier (Typical)

| Model | Tier 1 Loss | Tier 2 Loss | Tier 3 Loss |
|-------|-------------|-------------|-------------|
| Polynomial | 0.5 | 0.8 | 1.2 |
| AR | 0.6 | 0.9 | 1.3 |
| Transcendental | 0.7 | 1.0 | 1.4 |
| Physics | 0.8 | 1.1 | 1.5 |
| Linear ODE | 0.9 | 1.2 | 1.6 |

*Lower is better. These are rough targets after 10k steps.*

---

## Checklist for New Model Types

When adding a new generator:

- [ ] Define Tier 1/2/3 parameter ranges
- [ ] Implement tier parameter in `generate()`
- [ ] Add to `GENERATORS` registry
- [ ] Add config to `CONFIGS` in `config.py`
- [ ] Add domain mapping to `DOMAINS`
- [ ] Create seeded test cases in `test.py`
- [ ] Document tier meanings for that model

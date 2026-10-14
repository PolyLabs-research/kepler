# Migration Cheat Sheet - Galileo Minimal Tier System

## ⚡ Quick Migration (5 Minutes)

```bash
# 1. Backup (IMPORTANT!)
python migrate_to_tier.py --backup --all

# 2. Validate
python validate_migration.py

# 3. Test legacy compatibility
python test.py --model polynomial --tier 1

# 4. Train new tier
python train.py --model polynomial --tier 2 --steps 5000

# 5. Test new tier
python test.py --model polynomial --tier 2
```

## 📁 New Directory Structure

```
checkpoints/
  algebraic/
    polynomial/
      tier1/model_best.pt  ← Moved here
      tier2/model_best.pt  ← New tier
      tier3/model_best.pt  ← New tier

data/
  polynomial/
    tier1/features.bin    ← Moved here
    tier2/features.bin    ← New tier
```

## 🎯 Essential Commands

| Task | Command |
|------|---------|
| Compile | `python compile_data.py --model X --tier 2 --samples 100k` |
| Train | `python train.py --model X --tier 2 --steps 10000` |
| Resume | `python train.py --model X --tier 2 --steps 20000 --resume` |
| Test | `python test.py --model X --tier 2` |

## 🔢 Tier Meanings (Memorize This!)

| Tier | Description | Use Case |
|------|-------------|----------|
| **1** | Discrete params, no noise | Quick tests, debugging |
| **2** | Continuous params, 2dp | Main production tier |
| **3** | Wide ranges, noise | Robustness, edge cases |

## ⚠️ ONE Breaking Change

**linear_ode MUST be retrained:**
```bash
rm -rf checkpoints/dynamical/linear_ode/
python train.py --model linear_ode --tier 1 --steps 10000
```

**Why?** Second-order ODEs now encode actual coefficients (C1, C2) instead of templates.

## 🐛 Troubleshooting (3 Most Common)

### Problem: "No checkpoint found"
**Fix:** Train that tier: `python train.py --model X --tier 2 --steps 5000`

### Problem: "Could not parse ODE coefficients"  
**Fix:** Retrain linear_ode (see above)

### Problem: Tests pass with --tier 1, fail with --tier 2
**Fix:** Tier mismatch. Ensure data, training, and testing all use same tier.

## 📊 Verification

Run this to check everything:
```bash
python validate_migration.py
```

Should see:
```
✓ Imports
✓ Generator Tier Support
✓ ODE Tokenization
✓ Determinism
✓ Tier Differences
✓ File Structure
✓ CLI Arguments

Overall: 7/7 checks passed
✓ Migration validation successful!
```

## 💡 Pro Tips

1. **Start with Tier 1** - Fastest to train, easiest to debug
2. **Tier 2 is production** - Best balance of difficulty and speed
3. **Use Tier 3 for robustness** - Test model on edge cases
4. **Seeds are your friend** - Tests are now deterministic
5. **Vocab is saved** - Future-proofs your checkpoints

## 🔄 Rollback (If Needed)

```bash
# Restore from backup
cp -r backup_YYYYMMDD/* .

# Verify old system
python test.py --model polynomial
```

## 📚 Full Documentation

- **MIGRATION_GUIDE.md** - Complete step-by-step (READ THIS FIRST)
- **TIER_REFERENCE.md** - Daily use reference
- **IMPLEMENTATION_SUMMARY.md** - Technical details

## ✅ Success Checklist

- [ ] Backed up existing files
- [ ] Ran `validate_migration.py` (all pass)
- [ ] Tested legacy checkpoint works
- [ ] Trained one tier 2 model
- [ ] linear_ode retrained (if used)
- [ ] Read MIGRATION_GUIDE.md

## 🎓 New Workflow Example

```bash
# Old way (single tier, implicit)
python train.py --model polynomial --steps 10000

# New way (explicit tier, scalable)
python train.py --model polynomial --tier 2 --steps 10000

# Multi-tier progression
for tier in 1 2 3; do
    python train.py --model polynomial --tier $tier --steps 10000
    python test.py --model polynomial --tier $tier
done
```

## 🚀 Next Steps

1. Validate: `python validate_migration.py`
2. Test: `python test.py --model polynomial --tier 1`
3. Train: `python train.py --model ar --tier 2 --steps 5000`
4. Experiment: Try curriculum learning (T1→T2→T3)

---

**Questions?** Check MIGRATION_GUIDE.md or run `python validate_migration.py` for diagnostics.

**All clear?** Start with: `python migrate_to_tier.py --backup --all`

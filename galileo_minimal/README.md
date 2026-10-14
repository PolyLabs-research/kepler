# Galileo Trainer (Minimal)

Lightweight symbolic equation discovery using transformer models. Trains specialist models to predict mathematical formulas from numerical data.

## Quick Start

### Windows

```cmd
REM 1) Setup (CUDA by default; use --cpu for CPU-only)
setup.bat --dev

REM 2) Activate environment (if not already)
.venv\Scripts\activate.bat

REM 3) Run tests
pytest

REM 4) Train a model
python train.py --model polynomial
```

### Linux/Mac

```bash
chmod +x setup.sh
./setup.sh --dev        # auto-detects CUDA (or use --cpu / --cuda)
source .venv/bin/activate
pytest
python train.py --model polynomial
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (or CPU, but slower)
- ~4GB+ VRAM recommended (batch size auto-scales)

**Dependencies**:
- Core: `requirements.txt` (numpy, scipy, sympy, pyyaml)
- Dev/test: `requirements-dev.txt` (+ pytest)
- PyTorch: installed by setup scripts (CPU or CUDA)

---

## Training

### Basic Usage

```bash
python train.py --model <model_type>
```

### Available Models

| Model | Description | Feature Dim |
|-------|-------------|-------------|
| `polynomial` | y = Σaᵢxⁱ (linear, quadratic, cubic, etc.) | 48 |
| `ar` | Autoregressive timeseries y[t] = Σφᵢy[t-i] | 32 |
| `transcendental` | Exponential, power law, logarithmic, logistic | 48 |
| `physics_laws` | Multivariate physics (F = GMm/r², etc.) | 64 |

### Options

```bash
python train.py --model polynomial              # Basic training
python train.py --model polynomial --steps 10000  # Custom step count
python train.py --model polynomial --resume       # Resume from checkpoint
python train.py --model polynomial --batch-size 256  # Override batch size
```

### What Happens During Training

1. **Online data generation** - samples generated on-the-fly (no pre-compilation)
2. **Checkpoints saved** to `checkpoints/<domain>/<model>/`
   - `model_best.pt` - best loss so far
   - `model_final.pt` - end of training
   - `model_interrupted.pt` - if you Ctrl+C
3. **Logging** every 100 steps showing loss and learning rate

### Typical Output

```
Device: cuda
Batch size: 512
Model parameters: 4,234,567
[100/50000] loss=2.3421 avg=2.5123 lr=3.00e-05
[200/50000] loss=1.8934 avg=2.1245 lr=6.00e-05
Saved BEST checkpoint (loss=2.1245)
...
```

### Training Tips

- **Start with polynomial** - simplest, validates your setup
- **5000-10000 steps** is usually enough to see if it's learning
- **Watch the loss** - should drop below 1.0 for polynomial within a few thousand steps
- **Ctrl+C is safe** - saves an interrupt checkpoint you can resume from

---

## Testing

### Basic Usage

```bash
python test.py --model polynomial
```

### Options

```bash
python test.py --model polynomial              # With cleanup (default)
python test.py --model polynomial --no-cleanup # Raw model output
```

### What Cleanup Does

The model outputs approximate coefficients. Post-processing cleans them up:

| Raw Output | After Cleanup |
|------------|---------------|
| `0.98 * X + 0.03` | `X` |
| `2.03 * X + 0.97` | `2 * X + 1` |
| `0.51 * X ** 2` | `1/2 * X ** 2` |

Cleanup includes:
- **Pruning** tiny terms (|coef| < 0.1)
- **Snapping** to integers (0.98 → 1)
- **Snapping** to common fractions (0.51 → 1/2, 0.67 → 2/3)

### Typical Output

```
Loaded: checkpoints/algebraic/polynomial/model_best.pt

Running 3 tests for polynomial...
======================================================================
[PASS] linear_simple
  Truth:     X
  Predicted: X
----------------------------------------------------------------------
[PASS] quadratic_vertex
  Truth:     X ** 2
  Predicted: X ** 2
----------------------------------------------------------------------

Summary: 2/3 passed
```

---

## File Structure

```
galileo_minimal/
├── setup.bat         # Windows setup
├── setup.sh          # Linux/Mac setup
├── requirements.txt  # Runtime dependencies
├── requirements-dev.txt # Dev/test dependencies
├── pytest.ini        # Pytest defaults
├── config.py         # Hardware detection, model configs
├── vocab.py          # Token vocabulary
├── model.py          # SymbolicTransformer architecture
├── generators.py     # Data generators + feature extraction
├── train.py          # Training script
├── test.py           # Evaluation script
└── checkpoints/      # Created during training
    ├── algebraic/
    │   └── polynomial/
    │       └── model_best.pt
    └── timeseries/
        └── ar/
            └── model_best.pt
```

---

## Hardware Notes

### VRAM Auto-Scaling

The trainer automatically adjusts batch size based on available VRAM:

| VRAM | Batch Size |
|------|------------|
| 16GB+ | 512 |
| 8GB | 256 |
| 4GB | 128 |
| CPU | 128 |

Override with `--batch-size` if needed.

### CPU Training

Works but slow. Useful for debugging:

```cmd
REM Windows
set CUDA_VISIBLE_DEVICES=
python train.py --model polynomial --steps 100
```

```bash
# Linux/Mac
CUDA_VISIBLE_DEVICES="" python train.py --model polynomial --steps 100
```

---

## Migrating Checkpoints

Checkpoints from the full Galileo Trainer are compatible. Just copy the folder:

```cmd
REM Windows
xcopy /E /I C:\old\path\checkpoints checkpoints
```

Then test:

```cmd
python test.py --model polynomial
```

---

## Troubleshooting

### "CUDA out of memory"

Reduce batch size:
```cmd
python train.py --model polynomial --batch-size 128
```

### "No checkpoint found"

Train first:
```cmd
python train.py --model polynomial --steps 5000
```

### Loss not decreasing

- Check GPU is being used: run `nvidia-smi` in another terminal
- Try a simpler model first (polynomial)
- Ensure batch size isn't too small (<64 can be unstable)

### Ctrl+C doesn't stop cleanly on Windows

This is a known Windows/Python issue. The checkpoint should still save. If it hangs:
- Close the terminal window
- Your `model_best.pt` from the last save is safe

### "DLL load failed" or CUDA errors on Windows

- Ensure NVIDIA drivers are up to date
- Verify CUDA toolkit matches PyTorch version (we use CUDA 12.1)
- Try: `nvidia-smi` to confirm GPU is detected

---

## Phase 3 Prep (Before Training)

A pre-training toolkit is included in `phase3/`:

- `prepare_routing_dataset.py` generates labeled routing datasets (JSONL/CSV)
- `calibration_dry_run.py` produces calibration baseline metrics
- `PHASE3_PRETRAIN_CHECKLIST.md` documents handoff steps

Quick run:

```bash
python phase3/prepare_routing_dataset.py
python phase3/calibration_dry_run.py
```

## Extending

### Adding a New Generator

1. Add class to `generators.py` following the `BaseGenerator` pattern
2. Add config to `CONFIGS` in `config.py`
3. Add domain mapping to `DOMAINS` in `config.py`
4. Add test cases to `TEST_CASES` in `test.py`

### Adding New Tokens

Edit `vocab.py` → `Vocabulary.default()` to add tokens.

**Note**: Changing vocabulary invalidates existing checkpoints.

---

## License

MIT

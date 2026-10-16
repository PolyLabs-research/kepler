# Galileo Trainer (Minimal)

Lightweight symbolic equation discovery using transformer models. Trains specialist models to predict mathematical formulas from numerical data.

## The `kepler` library

The classical half of Kepler (expression parsing, normalisation, least-squares refinement and
the BIC ranking rule) is an installable package in `src/kepler`, with a public fit, rank and
predict API. It needs numpy, scipy and sympy only; the transformers and the MCP server are
optional extras.

```bash
pip install -e .              # the library
pip install -e ".[models]"    # + torch, for galileo_minimal/train.py and friends
pip install -e ".[mcp]"       # + mcp, for galileo_minimal/mcp_server.py
pytest                        # the library's tests (tests/); galileo_minimal has its own
```

```python
from kepler import fit, rank, predict, LAWS

# Constants with standard errors; sigma weights each point, space="log" fits log y.
fits = [fit(name, L, T, sigma=T_err) for name in ("logarithmic", "power", "exponential")]
ranking = rank(fits)                      # BIC, 6-point Occam window, eligibility reasons
best = ranking.best.fit                   # None when no law fits: that is an answer too
best.params, best.stderr                  # e.g. (27.2, 0.350), (0.98, 0.013)
T_hat, low, high = predict(best, [48, 64, 128])   # 95% band from the covariance
```

- `fit(skeleton, x, y, sigma=None, space="linear")` takes a law name from `kepler.LAWS`
  (constant, logarithmic, linear, power, exponential, exponential_area, logistic,
  shifted_power), a `Law`, or an equation whose Float literals are the constants
  (`"20.0*x**0.5"`). A zero literal is refused: SymPy drops `0.0 + x` to `x`.
- `rank(fits, n, occam_window=6.0, min_r2=0.5)` is the rule `GalileoTools.discover_best` uses,
  now shared code. It refuses to compare fits of different data, weights or spaces.
- `predict(fit, x, level=0.95)` returns the curve and its confidence band. A held-out
  measurement's own error is the caller's to add. `predict_covariance(fit, x)` gives the full
  covariance of predictions at several x (correlated through the shared constants), for a joint
  test of held-out points.
- The MCP server gains `fit_skeleton`: a language model proposes an equation and gets its
  constants fitted. `verify_equation` never refits.
- Model comparison is likelihood-based: with absolute sigmas a fit's BIC is chi-square +
  k ln n (known errors), otherwise n ln(RSS/n) + k ln n; `Fit.aicc` is the small-sample AIC.
  With `inflate`, `rank` divides every chi-square by one shared overdispersion factor (the
  best law's reduced chi-square, at least 1), so errors that are too small cannot make a
  verdict look decisive.
- `separation(truth, rival, x, relative=0.1, space="log", truth_params={...})` says before a
  run whether a design can tell two laws apart: the expected BIC difference, the factor by
  which the seeds must grow to reach a margin, and (with `simulations=N`) the chance of a
  decisive right or wrong verdict.

`galileo_minimal/expression.py`, `normalize.py` and `refine.py` are now shims that hand back
the package modules, so the scripts and tests there run unchanged from inside that folder.

## Quick Start

```bash
# 1. Setup (creates venv, installs PyTorch+CUDA)
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate

# 2. Train a model
python train.py --model polynomial

# 3. Test it
python test.py --model polynomial
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (or CPU, but slower)
- ~4GB+ VRAM recommended (batch size auto-scales)

**Dependencies** (installed by setup.sh):
- torch (with CUDA 12.1)
- numpy, scipy, sympy, pyyaml

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
├── setup.sh          # Environment setup
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

```bash
CUDA_VISIBLE_DEVICES="" python train.py --model polynomial --steps 100
```

---

## Migrating Checkpoints

Checkpoints from the full Galileo Trainer are compatible. Just copy:

```bash
cp -r /old/system/checkpoints ./checkpoints
```

Then test:

```bash
python test.py --model polynomial
```

---

## Troubleshooting

### "CUDA out of memory"

Reduce batch size:
```bash
python train.py --model polynomial --batch-size 128
```

### "No checkpoint found"

Train first:
```bash
python train.py --model polynomial --steps 5000
```

### Loss not decreasing

- Check your GPU is being used (`nvidia-smi`)
- Try a simpler model first (polynomial)
- Ensure batch size isn't too small (<64 can be unstable)

### Test results all FAIL but predictions look close

This is usually a comparison issue. Check:
- Are coefficients approximately correct? (cleanup helps)
- Is the structure right even if coefficients differ?

---

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

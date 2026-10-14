# Kepler Cloud GPU Training Plan (Pre-Execution)

## 1) What to retain vs exclude

### Retain (required)
- `train.py`, `compile_data.py`
- `config.py`, `model.py`, `vocab.py`, `features.py`
- `generators.py`, `generators_dynamical.py`, `generators_base.py`
- `galileo_tools.py`, `overseer.py`, `synthesizer.py`
- `core/`
- `requirements.txt`, `requirements-dev.txt`
- `phase3/` (dataset prep + calibration baselines)

### Retain (optional but useful)
- `test.py`, `test_*.py`
- `README.md`

### Exclude from upload (save time/cost)
- `.venv*/`
- `__pycache__/`
- `backup_20260129_150049/`
- local `checkpoints/` (unless warm-starting intentionally)
- large `data/` binaries (unless resuming from precompiled data)

Current local sizes (approx):
- `checkpoints`: 1.7G
- `data`: 440M
- `backup_...`: 606M

## 2) Cloud training strategy

Use cloud GPU for **training only** and keep inference/testing local.

### Recommended flow
1. Upload lean project (code + configs + scripts; no bulky artifacts).
2. Build env on cloud.
3. Train targeted models/tier matrix.
4. Download only resulting checkpoint folders.
5. Place checkpoints locally under `checkpoints/<domain>/<model>/tier<tier>/`.
6. Run local validation (`pytest`, plus real-world report scripts).

## 3) Suggested priority model set

Based on recent results:
1. `transcendental` (highest priority; logistic weakness)
2. `linear_ode`
3. `ar`
4. `polynomial`
5. `physics_laws`

Train tiers 1→2→3 per model, reusing resume.

## 4) Cloud commands (generic Ubuntu GPU VM)

```bash
# On cloud box
sudo apt-get update
sudo apt-get install -y git rsync python3-venv

# Receive project (example path)
mkdir -p ~/kepler && cd ~/kepler
# (upload tar/rsync from local)

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# GPU torch (choose one)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Train (example)
```bash
source .venv/bin/activate
python train.py --model transcendental --tier 1 --steps 60000
python train.py --model transcendental --tier 2 --steps 80000
python train.py --model transcendental --tier 3 --steps 100000

python train.py --model linear_ode --tier 1 --steps 60000
python train.py --model linear_ode --tier 2 --steps 80000
python train.py --model linear_ode --tier 3 --steps 100000
```

### Resume safety
```bash
python train.py --model transcendental --tier 2 --resume
```

## 5) Pull models back locally

Download only checkpoint folders:

```bash
# From local machine
rsync -avz cloud:~/kepler/checkpoints/ /home/michael/Documents/kepler/galileo_minimal/checkpoints/
```

## 6) Local post-download validation

```bash
cd /home/michael/Documents/kepler/galileo_minimal
source .venv-linux/bin/activate
pytest -q
python phase3/calibration_dry_run.py
```

Then re-run your real-world benchmark/report script.

## 7) Cost/risk controls

- Use spot/preemptible GPUs where available.
- Save checkpoints every run; use `--resume` after interruption.
- Train one model family at a time; evaluate before broadening scope.
- Keep a simple run log (model/tier/steps/loss/device/time).

## 8) Practical recommended first cloud run

Run only:
- `transcendental` tier1/2/3
- `linear_ode` tier1/2

Then pull checkpoints and re-run Kepler reports. If logistic improves and no ODE regressions, continue with the rest.

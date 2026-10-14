# Phase 3 (Pre-Training) Checklist

This folder prepares Kepler for the next training/calibration cycle without actually training.

## Added in this pass

- `prepare_routing_dataset.py`
  - Generates labeled routing-feature datasets (JSONL + CSV)
  - Includes polynomial, transcendental (exp/logistic), ODE, and AR scenarios

- `calibration_dry_run.py`
  - Runs a held-out synthetic benchmark
  - Reports top-1 accuracy, top-2 coverage, and ambiguity rate

## Run steps

```bash
cd galileo_minimal
source .venv-linux/bin/activate
python phase3/prepare_routing_dataset.py
python phase3/calibration_dry_run.py
```

## Expected outputs

- `phase3/output/routing_dataset.jsonl`
- `phase3/output/routing_dataset.csv`
- `phase3/output/phase3_calibration_dry_run.json`

## Training handoff (next phase)

1. Use `routing_dataset.csv` to train a lightweight classifier/calibrator.
2. Compare post-train metrics against `phase3_calibration_dry_run.json` baseline.
3. Integrate calibrator in `Overseer._score_models` or post-softmax re-ranker.
4. Re-run real-world report suite and validate no regression on AR/ODE.

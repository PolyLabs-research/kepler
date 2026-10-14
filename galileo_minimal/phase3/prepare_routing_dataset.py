#!/usr/bin/env python3
"""
Prepare a labeled routing dataset for Overseer calibration/training.

This script does NOT train anything. It creates structured data that can be
used to train/calibrate a classifier in the next phase.

Outputs:
  - phase3/output/routing_dataset.jsonl
  - phase3/output/routing_dataset.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from overseer import Overseer, DataType


def generate_scenarios(seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    # Polynomial family
    for _ in range(80):
        x = np.linspace(-5, 5, 140)
        a, b, c = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-1, 1)
        y = a * x**2 + b * x + c + rng.normal(0, 0.2, size=len(x))
        rows.append({"name": "quadratic", "x": x, "y": y, "label": "polynomial"})

    # Exponential / transcendental
    for _ in range(80):
        x = np.linspace(0, 12, 140)
        a = rng.uniform(10, 120)
        k = rng.uniform(0.08, 0.6)
        c = rng.uniform(0, 8)
        y = c + a * np.exp(-k * x) + rng.normal(0, 0.5, size=len(x))
        rows.append({"name": "exp_decay", "x": x, "y": y, "label": "transcendental"})

    # Logistic / transcendental
    for _ in range(120):
        x = np.linspace(0, 20, 180)
        K = rng.uniform(300, 2000)
        P0 = rng.uniform(5, 80)
        r = rng.uniform(0.15, 0.6)
        A = (K - P0) / P0
        y = K / (1 + A * np.exp(-r * x)) + rng.normal(0, rng.uniform(1.0, 10.0), size=len(x))
        rows.append({"name": "logistic", "x": x, "y": y, "label": "transcendental"})

    # ODE oscillatory
    for _ in range(90):
        t = np.linspace(0, 20, 240)
        damp = rng.uniform(0.03, 0.2)
        omega = rng.uniform(1.2, 3.2)
        y = np.exp(-damp * t) * np.cos(omega * t) + rng.normal(0, 0.02, size=len(t))
        rows.append({"name": "damped_osc", "x": t, "y": y, "label": "linear_ode"})

    # AR(1)
    for _ in range(80):
        n = 260
        y = np.zeros(n)
        phi = rng.uniform(0.6, 0.92)
        sigma = rng.uniform(0.1, 0.5)
        for i in range(1, n):
            y[i] = phi * y[i - 1] + rng.normal(0, sigma)
        rows.append({"name": "ar1", "x": None, "y": y, "label": "ar"})

    return rows


def main() -> int:
    base = Path(__file__).resolve().parent
    out_dir = base / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    overseer = Overseer()
    scenarios = generate_scenarios()

    jsonl_path = out_dir / "routing_dataset.jsonl"
    csv_path = out_dir / "routing_dataset.csv"

    records: List[Dict] = []
    for s in scenarios:
        y = np.asarray(s["y"], dtype=float)
        x = None if s["x"] is None else np.asarray(s["x"], dtype=float)

        data_type, _reason = overseer._detect_data_type(y, x, None)
        feats = overseer._extract_diagnostic_features(y, x, None, data_type)

        record = {
            "scenario": s["name"],
            "label": s["label"],
            "data_type": data_type.value,
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in feats.items()},
        }
        records.append(record)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    fieldnames = sorted({k for r in records for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"Wrote {len(records)} rows")
    print(jsonl_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

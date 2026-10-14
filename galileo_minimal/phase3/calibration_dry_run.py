#!/usr/bin/env python3
"""
Phase 3 calibration dry-run.

Runs Overseer on a held-out synthetic suite and reports:
- top-1 accuracy
- top-2 coverage
- ambiguous-case rate

No training is performed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from overseer import Overseer


def build_suite(seed: int = 123) -> List[Dict]:
    rng = np.random.default_rng(seed)
    suite: List[Dict] = []

    # Logistic heavy suite for current weak spot
    for _ in range(60):
        x = np.linspace(0, 20, 180)
        K = rng.uniform(200, 2000)
        P0 = rng.uniform(8, 80)
        r = rng.uniform(0.18, 0.55)
        A = (K - P0) / P0
        y = K / (1 + A * np.exp(-r * x)) + rng.normal(0, rng.uniform(2, 12), size=len(x))
        suite.append({"name": "logistic", "x": x, "y": y, "expected": ["transcendental"]})

    # Control groups
    for _ in range(40):
        x = np.linspace(0, 12, 140)
        y = 20 + 80 * np.exp(-0.25 * x) + rng.normal(0, 0.6, size=len(x))
        suite.append({"name": "exp", "x": x, "y": y, "expected": ["transcendental", "linear_ode"]})

    for _ in range(40):
        t = np.linspace(0, 20, 220)
        y = np.exp(-0.12 * t) * np.cos(2.2 * t) + rng.normal(0, 0.02, size=len(t))
        suite.append({"name": "ode", "x": t, "y": y, "expected": ["linear_ode"]})

    for _ in range(40):
        n = 260
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.8 * y[i - 1] + rng.normal(0, 0.3)
        suite.append({"name": "ar", "x": None, "y": y, "expected": ["ar"]})

    return suite


def main() -> int:
    over = Overseer()
    suite = build_suite()

    top1 = 0
    top2 = 0
    ambiguous = 0

    details = []
    for s in suite:
        y = np.asarray(s["y"], dtype=float)
        x = None if s["x"] is None else np.asarray(s["x"], dtype=float)
        c = over.classify(y, x)
        t2 = [m for m, _ in c.top_k(2)]

        ok1 = c.best_model in s["expected"]
        ok2 = any(m in s["expected"] for m in t2)

        top1 += int(ok1)
        top2 += int(ok2)
        ambiguous += int(c.is_ambiguous)

        details.append({
            "scenario": s["name"],
            "best": c.best_model,
            "top2": t2,
            "expected": s["expected"],
            "margin": c.confidence_margin,
            "ambiguous": c.is_ambiguous,
            "ok1": ok1,
            "ok2": ok2,
        })

    out = {
        "n": len(suite),
        "top1_accuracy": top1 / len(suite),
        "top2_coverage": top2 / len(suite),
        "ambiguous_rate": ambiguous / len(suite),
        "details": details,
    }

    out_path = Path(__file__).resolve().parent / "output" / "phase3_calibration_dry_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(out_path)
    print(f"top1={out['top1_accuracy']:.3f} top2={out['top2_coverage']:.3f} ambiguous={out['ambiguous_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

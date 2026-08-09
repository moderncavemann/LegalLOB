#!/usr/bin/env python3
"""Run and aggregate the nine official FI-2010 anchored folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/fi2010/noauction_zscore_full")
    parser.add_argument("--output-root", default="checkpoints_fi2010/anchored_cv")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folds = parse_ints(args.folds)
    seeds = parse_ints(args.seeds)
    root = Path(args.output_root)
    for seed in seeds:
        for fold in folds:
            save_dir = root / f"seed{seed}" / f"cf{fold}"
            cmd = [
                sys.executable,
                "scripts/train_fi2010_deeplob.py",
                "--processed-dir",
                args.processed_dir,
                "--save-dir",
                str(save_dir),
                "--fold",
                str(fold),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(seed),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            print("$ " + " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    if args.dry_run:
        return 0

    rows = []
    for seed in seeds:
        for fold in folds:
            path = root / f"seed{seed}" / f"cf{fold}" / "metrics.json"
            payload = json.loads(path.read_text())
            metrics = payload["test"][f"test_cf{fold}"]
            rows.append({"seed": seed, "fold": fold, **metrics})
    summary = {
        "protocol": "FI-2010 no-auction z-score official anchored folds",
        "folds": folds,
        "seeds": seeds,
        "rows": rows,
        "aggregate": {},
    }
    for metric in ["loss", "accuracy", "macro_f1", "balanced_accuracy"]:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary["aggregate"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "anchored_cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

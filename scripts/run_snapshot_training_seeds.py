#!/usr/bin/env python3
"""Run snapshot_l2 projection ablations across multiple training seeds.

This is an orchestration layer for paper-grade robustness. It calls
``run_snapshot_ablation.py`` once per training seed with isolated checkpoint and
result directories, then optionally aggregates the per-training-seed
``seed_summary.json`` files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_seed_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def metric_keys(rows: list[dict]) -> list[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(key)
    return sorted(keys)


def aggregate_training_seed_summaries(result_root: Path, train_seeds: list[int]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for seed in train_seeds:
        path = result_root / f"train_seed{seed}" / "seed_summary.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        for row in rows:
            grouped.setdefault(row["method"], []).append(row)

    summary = []
    for method, rows in sorted(grouped.items()):
        out = {
            "method": method,
            "training_seeds": ",".join(str(s) for s in train_seeds),
            "n_training_seeds_found": len(rows),
        }
        for key in metric_keys(rows):
            values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                out[f"{key}_training_seed_mean"] = float(values.mean())
                out[f"{key}_training_seed_std"] = float(values.std(ddof=0))
        summary.append(out)
    return summary


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("No training-seed summaries found.\n")
        return
    preferred = [
        "method",
        "n_training_seeds_found",
        "mse_raw_mean_training_seed_mean",
        "mse_raw_mean_training_seed_std",
        "raw_crossed_book_rate_mean_training_seed_mean",
        "raw_crossed_book_rate_mean_training_seed_std",
        "raw_ask_ladder_violation_mean_training_seed_mean",
        "raw_bid_ladder_violation_mean_training_seed_mean",
        "projection_distance_mean_training_seed_mean",
        "sampler_cumulative_projection_l2_mean_mean_training_seed_mean",
    ]
    cols = [c for c in preferred if any(c in row for row in rows)]
    lines = ["# Training Seed Ablation Summary", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run snapshot ablation over multiple training seeds")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="0,1,2")
    parser.add_argument("--snowball-seeds", default="0,1,2")
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--cfg-drop", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--price-reference", choices=["auto", "dataset", "history_last_mid"], default="auto")
    parser.add_argument("--sampling-mode", choices=["uniform", "head", "full"], default="uniform")
    parser.add_argument("--skip-physics-baseline", action="store_true")
    parser.add_argument("--include-matched-baseline", action="store_true")
    parser.add_argument("--physics-loss-weight", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-snowball", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_seeds = parse_seed_list(args.train_seeds)
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root or f"checkpoints_training_seeds/{args.dataset.lower()}_{args.epochs}ep")

    for seed in train_seeds:
        raw_dir = checkpoint_root / f"train_seed{seed}" / "raw"
        legal_dir = checkpoint_root / f"train_seed{seed}" / "legal_lob"
        matched_dir = checkpoint_root / f"train_seed{seed}" / "matched_lob"
        physics_dir = checkpoint_root / f"train_seed{seed}" / "raw_physics"
        result_dir = output_root / f"train_seed{seed}"
        cmd = [
            sys.executable,
            "scripts/run_snapshot_ablation.py",
            "--dataset",
            args.dataset,
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(result_dir),
            "--raw-save-dir",
            str(raw_dir),
            "--legal-save-dir",
            str(legal_dir),
            "--matched-save-dir",
            str(matched_dir),
            "--physics-save-dir",
            str(physics_dir),
            "--epochs",
            str(args.epochs),
            "--hist-len",
            str(args.hist_len),
            "--pred-len",
            str(args.pred_len),
            "--batch-size",
            str(args.batch_size),
            "--n-batches",
            str(args.n_batches),
            "--ddim-steps",
            str(args.ddim_steps),
            "--cfg-drop",
            str(args.cfg_drop),
            "--lr",
            str(args.lr),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--price-reference",
            args.price_reference,
            "--sampling-mode",
            args.sampling_mode,
            "--num-workers",
            str(args.num_workers),
            "--train-seed",
            str(seed),
            "--eval-seeds",
            args.eval_seeds,
            "--snowball-seeds",
            args.snowball_seeds,
            "--device",
            args.device,
        ]
        if not args.skip_physics_baseline:
            cmd.extend([
                "--include-physics-baseline",
                "--physics-loss-weight",
                str(args.physics_loss_weight),
            ])
        if args.include_matched_baseline:
            cmd.append("--include-matched-baseline")
        if args.skip_training:
            cmd.append("--skip-training")
        if args.skip_snowball:
            cmd.append("--skip-snowball")
        if args.dry_run:
            cmd.append("--dry-run")
        run(cmd, args.dry_run)

    if not args.dry_run:
        summary = aggregate_training_seed_summaries(output_root, train_seeds)
        (output_root / "training_seed_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        write_markdown(output_root / "training_seed_summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

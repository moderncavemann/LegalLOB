#!/usr/bin/env python3
"""Run the audited public-ITCH 1 Hz snapshot ablation panel sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def validate_panel_audit(path: Path, required_dataset_ids: list[str]) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Panel audit does not exist: {path}")
    payload = json.loads(path.read_text())
    if not payload.get("all_schema_checks_pass", False):
        raise ValueError(f"Panel audit has not passed: {path}")
    rows = {row.get("dataset"): row for row in payload.get("datasets", [])}
    missing = [dataset_id for dataset_id in required_dataset_ids if dataset_id not in rows]
    failed = [
        dataset_id
        for dataset_id in required_dataset_ids
        if dataset_id in rows and rows[dataset_id].get("status") not in {"pass", "pass_with_warnings"}
    ]
    if missing or failed:
        raise ValueError(f"Panel audit is incomplete; missing={missing}, failed={failed}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="PUBLIC_ITCH_2019_SAMPLES")
    parser.add_argument("--symbols", default="AAPL,AMZN,GOOG")
    parser.add_argument("--mode", choices=["1s", "event"], default="1s")
    parser.add_argument("--audit", default="PUBLIC_ITCH_PANEL_AUDIT.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", default="results/public_itch_1s_panel_full")
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints_training_seeds/public_itch_1s_panel_50ep",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="0,1,2")
    parser.add_argument("--snowball-seeds", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--physics-loss-weight", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip-physics-baseline", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-snowball", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = _csv(args.symbols)
    if not symbols:
        raise ValueError("At least one symbol is required")
    dataset_ids = [f"{args.prefix}_{symbol}_{args.mode}" for symbol in symbols]
    audit = validate_panel_audit(Path(args.audit), dataset_ids)

    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_ids": dataset_ids,
        "audit": str(args.audit),
        "audit_scope": audit.get("scope"),
        "epochs": args.epochs,
        "train_seeds": _csv(args.train_seeds),
        "eval_seeds": _csv(args.eval_seeds),
        "snowball_seeds": _csv(args.snowball_seeds),
        "device": args.device,
        "claim_scope": "multi-day multi-stock 1 Hz continuous structural legality",
    }
    (output_root / "panel_run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    for dataset_id in dataset_ids:
        symbol = dataset_id.removeprefix(f"{args.prefix}_").removesuffix(f"_{args.mode}")
        cmd = [
            sys.executable,
            "scripts/run_snapshot_training_seeds.py",
            "--dataset",
            dataset_id,
            "--data-dir",
            args.data_dir,
            "--output-root",
            str(output_root / symbol),
            "--checkpoint-root",
            str(checkpoint_root / symbol),
            "--epochs",
            str(args.epochs),
            "--train-seeds",
            args.train_seeds,
            "--eval-seeds",
            args.eval_seeds,
            "--snowball-seeds",
            args.snowball_seeds,
            "--batch-size",
            str(args.batch_size),
            "--n-batches",
            str(args.n_batches),
            "--ddim-steps",
            str(args.ddim_steps),
            "--sampling-mode",
            "uniform",
            "--physics-loss-weight",
            str(args.physics_loss_weight),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--include-matched-baseline",
            "--device",
            args.device,
        ]
        if args.skip_physics_baseline:
            cmd.append("--skip-physics-baseline")
        if args.skip_training:
            cmd.append("--skip-training")
        if args.skip_snowball:
            cmd.append("--skip-snowball")
        if args.dry_run:
            cmd.append("--dry-run")
        print("\n$ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        (output_root / "panel_completion.json").write_text(
            json.dumps({"status": "complete", **run_config}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

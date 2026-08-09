#!/usr/bin/env python3
"""Run a bounded all-method 100-step rollout panel on frozen checkpoints."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


METHODS = (
    ("raw", "raw", "none", "none"),
    ("raw_posthoc_pava", "raw", "exact", "none"),
    ("raw_inloop_pava", "raw", "none", "exact"),
    ("matched", "matched_lob", "none", "none"),
    ("matched_posthoc_pava", "matched_lob", "exact", "none"),
    ("legal_lob", "legal_lob", "none", "none"),
)


def checkpoint_filename(value: str) -> str:
    """Accept one checkpoint filename, never a path traversal."""
    path = Path(value)
    if not value or value in {".", ".."} or path.name != value:
        raise argparse.ArgumentTypeError(
            "--checkpoint-name must be a single filename (for example best.pt)"
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="GOOG_databento_20190130_mbp10_rth_v2"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints_training_seeds/goog_databento_rth_v2_50ep",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=checkpoint_filename,
        default="best.pt",
        help="Checkpoint filename within each seed/representation directory.",
    )
    parser.add_argument(
        "--output-root", default="results/all_method_rollout_goog_event_640"
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--methods", nargs="+", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.methods or [item[0] for item in METHODS])
    unknown = selected - {item[0] for item in METHODS}
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    manifest = {
        "dataset": args.dataset,
        "checkpoint_root": args.checkpoint_root,
        "checkpoint_name": args.checkpoint_name,
        "train_seeds": args.train_seeds,
        "eval_seed": args.eval_seed,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "ddim_steps": args.ddim_steps,
        "methods": sorted(selected),
        "runs": [],
    }

    for train_seed in args.train_seeds:
        for method_id, representation, projection_mode, sampler_mode in METHODS:
            if method_id not in selected:
                continue
            checkpoint_representation = {
                "raw": "raw",
                "matched_lob": "matched_lob",
                "legal_lob": "legal_lob",
            }[representation]
            checkpoint = (
                Path(args.checkpoint_root)
                / f"train_seed{train_seed}"
                / checkpoint_representation
                / args.checkpoint_name
            )
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            output_dir = output_root / f"train_seed{train_seed}" / method_id
            stats_path = output_dir / "snowball_stats.json"
            if args.skip_existing and stats_path.exists():
                print(f"SKIP {stats_path}")
                manifest["runs"].append(
                    {
                        "train_seed": train_seed,
                        "method": method_id,
                        "checkpoint": str(checkpoint),
                        "output_dir": str(output_dir),
                        "status": "existing",
                    }
                )
                (output_root / "run_manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                args.python,
                "scripts/snowball_test.py",
                "--genlob-ckpt",
                str(checkpoint),
                "--weights",
                "ema",
                "--dataset",
                args.dataset,
                "--data-dir",
                args.data_dir,
                "--hist-len",
                "50",
                "--pred-len",
                "10",
                "--chunk-size",
                "10",
                "--horizon",
                str(args.horizon),
                "--ddim-steps",
                str(args.ddim_steps),
                "--representation",
                "auto",
                "--projection-mode",
                projection_mode,
                "--sampler-projection-mode",
                sampler_mode,
                "--compute-projection-distance",
                "--skip-plots",
                "--action-mode",
                "hold",
                "--batch-size",
                str(args.batch_size),
                "--n-batches",
                str(args.n_batches),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(args.eval_seed),
            ]
            if args.device:
                command.extend(["--device", args.device])
            command_text = shlex.join(command)
            (output_dir / "command.txt").write_text(command_text + "\n")
            print(f"RUN train_seed={train_seed} method={method_id}")
            subprocess.run(command, check=True)
            manifest["runs"].append(
                {
                    "train_seed": train_seed,
                    "method": method_id,
                    "checkpoint": str(checkpoint),
                    "output_dir": str(output_dir),
                    "command": command_text,
                    "status": "completed",
                }
            )
            (output_root / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )

    print(f"All-method rollout panel complete: {output_root}")


if __name__ == "__main__":
    main()

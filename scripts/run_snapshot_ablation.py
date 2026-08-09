#!/usr/bin/env python3
"""Run a generic snapshot_l2 raw-vs-LegalLOB projection ablation.

This is a convenience orchestrator around the existing training/evaluation
scripts. It does not introduce a new metric protocol; it standardizes command
execution for any processed ``snapshot_l2`` dataset id.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _seed_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-save-dir", default=None)
    parser.add_argument("--legal-save-dir", default=None)
    parser.add_argument("--matched-save-dir", default=None)
    parser.add_argument("--physics-save-dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
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
    parser.add_argument("--include-physics-baseline", action="store_true")
    parser.add_argument("--include-matched-baseline", action="store_true")
    parser.add_argument("--physics-loss-weight", type=float, default=1e-5,
                        help="Calibrated to contribute roughly 5% of the initial DDPM loss on Databento RTH v2.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", default="0,1,2")
    parser.add_argument("--snowball-seeds", default="0,1,2")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-resume-existing", dest="resume_existing", action="store_false")
    parser.set_defaults(resume_existing=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-snowball", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_slug = args.dataset.lower()
    raw_save_dir = args.raw_save_dir or f"checkpoints_raw/{dataset_slug}_{args.epochs}ep"
    legal_save_dir = args.legal_save_dir or f"checkpoints_legal/{dataset_slug}_{args.epochs}ep"
    matched_save_dir = args.matched_save_dir or f"checkpoints_matched/{dataset_slug}_{args.epochs}ep"
    physics_save_dir = args.physics_save_dir or f"checkpoints_raw_physics/{dataset_slug}_{args.epochs}ep"
    output_dir = Path(args.output_dir)
    device_args = [] if args.device == "auto" else ["--device", args.device]

    if not args.skip_training:
        training_specs = [
            ("legal_lob", legal_save_dir, 0.0),
            ("raw", raw_save_dir, 0.0),
        ]
        if args.include_matched_baseline:
            training_specs.insert(1, ("matched_lob", matched_save_dir, 0.0))
        if args.include_physics_baseline:
            training_specs.append(("raw", physics_save_dir, args.physics_loss_weight))
        for representation, save_dir, physics_weight in training_specs:
            save_path = Path(save_dir)
            if args.resume_existing and (save_path / "completion.json").exists():
                print(f"\n# completed training exists, skipping: {save_path}", flush=True)
                continue
            resume_args = []
            if args.resume_existing and (save_path / "last.pt").exists():
                resume_args = ["--resume", str(save_path / "last.pt")]
            _run(
                [
                    sys.executable,
                    "scripts/train_v2.py",
                    "--dataset",
                    args.dataset,
                    "--data-dir",
                    args.data_dir,
                    "--hist-len",
                    str(args.hist_len),
                    "--pred-len",
                    str(args.pred_len),
                    "--batch-size",
                    str(args.batch_size),
                    "--epochs",
                    str(args.epochs),
                    "--cfg-drop",
                    str(args.cfg_drop),
                    "--lr",
                    str(args.lr),
                    "--early-stopping-patience",
                    str(args.early_stopping_patience),
                    "--representation",
                    representation,
                    "--action-conditioning",
                    "none",
                    "--price-reference",
                    args.price_reference,
                    "--physics-loss-weight",
                    str(physics_weight),
                    "--action-contrastive-weight",
                    "0",
                    "--save-dir",
                    save_dir,
                    "--num-workers",
                    str(args.num_workers),
                    "--seed",
                    str(args.train_seed),
                ] + device_args + resume_args,
                args.dry_run,
            )

    if not args.skip_eval:
        methods = [
            ("raw_no_projection", raw_save_dir, "none", "none"),
            ("raw_posthoc_pava", raw_save_dir, "exact", "none"),
            ("raw_inloop_pava", raw_save_dir, "none", "exact"),
            ("legal_no_projection", legal_save_dir, "none", "none"),
        ]
        if args.include_matched_baseline:
            methods.extend([
                ("matched_no_projection", matched_save_dir, "none", "none"),
                ("matched_posthoc_pava", matched_save_dir, "exact", "none"),
            ])
        if args.include_physics_baseline:
            methods.append(("raw_physics_no_projection", physics_save_dir, "none", "none"))
        for seed in _seed_list(args.eval_seeds):
            for method, save_dir, projection_mode, sampler_projection_mode in methods:
                result_dir = output_dir / method if seed == 0 else output_dir / "seeds" / f"seed{seed}" / method
                _run(
                    [
                        sys.executable,
                        "scripts/evaluate_v2.py",
                        "--checkpoint",
                        str(Path(save_dir) / "best.pt"),
                        "--weights",
                        "ema",
                        "--dataset",
                        args.dataset,
                        "--data-dir",
                        args.data_dir,
                        "--split",
                        "test",
                        "--output-dir",
                        str(result_dir),
                        "--n-batches",
                        str(args.n_batches),
                        "--sampling-mode",
                        args.sampling_mode,
                        "--batch-size",
                        str(args.batch_size),
                        "--ddim-steps",
                        str(args.ddim_steps),
                        "--projection-mode",
                        projection_mode,
                        "--sampler-projection-mode",
                        sampler_projection_mode,
                        "--representation",
                        "auto",
                        "--action-conditioning",
                        "auto",
                        "--price-reference",
                        "auto",
                        "--seed",
                        str(seed),
                    ] + device_args,
                    args.dry_run,
                )

        _run(
            [
                sys.executable,
                "scripts/evaluate_snapshot_baseline.py",
                "--dataset",
                args.dataset,
                "--data-dir",
                args.data_dir,
                "--output-dir",
                str(output_dir / "persistence"),
                "--hist-len",
                str(args.hist_len),
                "--pred-len",
                str(args.pred_len),
                "--batch-size",
                str(args.batch_size),
                "--n-batches",
                str(args.n_batches),
                "--sampling-mode",
                args.sampling_mode,
                "--price-reference",
                args.price_reference,
            ],
            args.dry_run,
        )

        for seed in _seed_list(args.eval_seeds):
            result_dir = (
                output_dir / "moving_block_bootstrap"
                if seed == 0
                else output_dir / "seeds" / f"seed{seed}" / "moving_block_bootstrap"
            )
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_snapshot_baseline.py",
                    "--dataset",
                    args.dataset,
                    "--data-dir",
                    args.data_dir,
                    "--output-dir",
                    str(result_dir),
                    "--baseline",
                    "moving_block_bootstrap",
                    "--hist-len",
                    str(args.hist_len),
                    "--pred-len",
                    str(args.pred_len),
                    "--batch-size",
                    str(args.batch_size),
                    "--n-batches",
                    str(args.n_batches),
                    "--sampling-mode",
                    args.sampling_mode,
                    "--price-reference",
                    args.price_reference,
                    "--seed",
                    str(seed),
                ],
                args.dry_run,
            )

        for baseline_name in ["persistence", "moving_block_bootstrap"]:
            _run(
                [
                    sys.executable,
                    "scripts/block_bootstrap_ablation.py",
                    "--method-a",
                    str(output_dir / baseline_name),
                    "--method-b",
                    str(output_dir / "legal_no_projection"),
                    "--output",
                    str(output_dir / f"block_bootstrap_{baseline_name}_vs_legal.json"),
                ],
                args.dry_run,
            )

        if args.include_matched_baseline:
            _run(
                [
                    sys.executable,
                    "scripts/block_bootstrap_ablation.py",
                    "--method-a",
                    str(output_dir / "matched_no_projection"),
                    "--method-b",
                    str(output_dir / "legal_no_projection"),
                    "--output",
                    str(output_dir / "block_bootstrap_matched_vs_legal.json"),
                ],
                args.dry_run,
            )

        _run(
            [
                sys.executable,
                "scripts/block_bootstrap_ablation.py",
                "--method-a",
                str(output_dir / "raw_no_projection"),
                "--method-b",
                str(output_dir / "legal_no_projection"),
                "--output",
                str(output_dir / "block_bootstrap_raw_vs_legal.json"),
            ],
            args.dry_run,
        )

    if not args.skip_snowball:
        for seed in _seed_list(args.snowball_seeds):
            result_dir = (
                output_dir / "legal_snowball_100_replay"
                if seed == 0
                else output_dir / "seeds" / f"seed{seed}" / "legal_snowball_100_replay"
            )
            _run(
                [
                    sys.executable,
                    "scripts/snowball_test.py",
                    "--genlob-ckpt",
                    str(Path(legal_save_dir) / "best.pt"),
                    "--weights",
                    "ema",
                    "--dataset",
                    args.dataset,
                    "--data-dir",
                    args.data_dir,
                    "--hist-len",
                    str(args.hist_len),
                    "--pred-len",
                    str(args.pred_len),
                    "--horizon",
                    "100",
                    "--ddim-steps",
                    str(args.ddim_steps),
                    "--representation",
                    "auto",
                    "--projection-mode",
                    "none",
                    "--compute-projection-distance",
                    "--action-mode",
                    "hold",
                    "--batch-size",
                    str(args.batch_size),
                    "--n-batches",
                    str(args.n_batches),
                    "--output-dir",
                    str(result_dir),
                    "--seed",
                    str(seed),
                ] + device_args,
                args.dry_run,
            )

    if not args.skip_eval or not args.skip_snowball:
        _run(
            [sys.executable, "scripts/summarize_projection_ablation.py", "--results-dir", str(output_dir)],
            args.dry_run,
        )
        _run(
            [sys.executable, "scripts/summarize_seed_ablation.py", "--results-dir", str(output_dir)],
            args.dry_run,
        )
        _run(
            [
                sys.executable,
                "scripts/summarize_ensemble_generation.py",
                "--results-dir",
                str(output_dir),
                "--seeds",
                args.eval_seeds,
            ],
            args.dry_run,
        )

    if not args.skip_eval or not args.skip_snowball:
        _run(
            [
                sys.executable,
                "scripts/write_ablation_provenance.py",
                "--results-dir",
                str(output_dir),
                "--experiment",
                f"{dataset_slug}_{args.epochs}ep_projection_ablation",
                "--dataset",
                args.dataset,
                "--raw-checkpoint",
                str(Path(raw_save_dir) / "best.pt"),
                "--legal-checkpoint",
                str(Path(legal_save_dir) / "best.pt"),
            ],
            args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

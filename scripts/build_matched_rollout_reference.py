#!/usr/bin/env python3
"""Build a strict Matched+Post reference CSV for the consumer audit.

The input roots must be produced by ``run_all_method_rollout_panel.py`` with
``--methods matched_posthoc_pava``.  This collector fails closed on incomplete
train/evaluation cells, protocol drift, orphan statistics, or path mismatches.
It records content hashes for every manifest, command, statistic, checkpoint,
and the resulting reference CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shlex
from pathlib import Path
from typing import Any


METHOD_ID = "matched_posthoc_pava"

CSV_FIELDS = (
    "train_seed",
    "eval_seed",
    "method_id",
    "method",
    "horizon",
    "n_histories",
    "final_invalid_snapshot_rate",
    "final_crossed_book_rate",
    "final_ask_ladder_violation",
    "final_bid_ladder_violation",
    "final_negative_volume_rate",
    "diagnostic_projection_l2",
    "spread_relative_abs_error",
    "volume_relative_abs_error",
    "return_std_relative_abs_error",
    "temporal_correlation",
    "return_w1",
    "spread_w1",
    "volume_w1",
    "path_pre_projection_invalid_rate",
    "path_cumulative_projection_l2",
    "path_projection_calls",
    "rollout_seconds",
    "trajectories_per_second",
    "lob_steps_per_second",
    "stats_path",
    "stats_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "command_path",
    "command_sha256",
    "manifest_path",
    "manifest_sha256",
)

CONSUMER_REFERENCE_FIELDS = (
    "path_cumulative_projection_l2",
    "path_projection_calls",
    "spread_relative_abs_error",
    "volume_relative_abs_error",
    "return_std_relative_abs_error",
    "temporal_correlation",
    "path_pre_projection_invalid_rate",
)


def checkpoint_filename(value: str) -> str:
    """Accept one checkpoint filename, never a path traversal."""
    path = Path(value)
    if not value or value in {".", ".."} or path.name != value:
        raise argparse.ArgumentTypeError(
            "--checkpoint-name must be a single filename (for example best.pt)"
        )
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve_declared(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{label}: observed={observed!r} expected={expected!r}")


def _require_finite(label: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _relative_error(label: str, generated: Any, real: Any) -> float:
    generated_value = _require_finite(f"{label}.generated", generated)
    real_value = _require_finite(f"{label}.real", real)
    if abs(real_value) < 1e-12:
        raise ValueError(f"{label}.real is too close to zero")
    return abs(generated_value - real_value) / abs(real_value)


def _command_value(tokens: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(tokens) if token == flag]
    if len(positions) != 1:
        raise ValueError(f"command must contain {flag} exactly once")
    index = positions[0]
    if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
        raise ValueError(f"command is missing a value for {flag}")
    return tokens[index + 1]


def _require_flag(tokens: list[str], flag: str) -> None:
    if tokens.count(flag) != 1:
        raise ValueError(f"command must contain {flag} exactly once")


def _validate_command(
    command: str,
    *,
    dataset: str,
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    eval_seed: int,
    hist_len: int,
    pred_len: int,
    chunk_size: int,
    horizon: int,
    ddim_steps: int,
    batch_size: int,
    n_batches: int,
    device: str,
) -> None:
    tokens = shlex.split(command)
    expected_values: dict[str, str] = {
        "--weights": "ema",
        "--dataset": dataset,
        "--hist-len": str(hist_len),
        "--pred-len": str(pred_len),
        "--chunk-size": str(chunk_size),
        "--horizon": str(horizon),
        "--ddim-steps": str(ddim_steps),
        "--representation": "auto",
        "--projection-mode": "exact",
        "--sampler-projection-mode": "none",
        "--action-mode": "hold",
        "--batch-size": str(batch_size),
        "--n-batches": str(n_batches),
        "--seed": str(eval_seed),
        "--device": device,
    }
    for flag, expected in expected_values.items():
        _require_equal(f"command {flag}", _command_value(tokens, flag), expected)
    _require_equal(
        "command --data-dir",
        _resolve_declared(_command_value(tokens, "--data-dir")),
        data_dir,
    )
    _require_equal(
        "command --genlob-ckpt",
        _resolve_declared(_command_value(tokens, "--genlob-ckpt")),
        checkpoint,
    )
    _require_equal(
        "command --output-dir",
        _resolve_declared(_command_value(tokens, "--output-dir")),
        output_dir,
    )
    _require_flag(tokens, "--compute-projection-distance")
    _require_flag(tokens, "--skip-plots")


def _extract_row(
    stats: dict[str, Any],
    *,
    train_seed: int,
    eval_seed: int,
    horizon: int,
    stats_path: Path,
    stats_hash: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    command_path: Path,
    command_hash: str,
    manifest_path: Path,
    manifest_hash: str,
) -> dict[str, Any]:
    step = (stats.get("generated") or {}).get(f"step_{horizon}")
    if not isinstance(step, dict):
        raise ValueError(f"Missing generated.step_{horizon}: {stats_path}")
    projection = stats.get("rollout_projection_stats") or {}
    horizon_projection = (
        (projection.get("by_horizon") or {}).get(f"step_{horizon}") or {}
    )
    posthoc = (
        horizon_projection.get("posthoc_chunk_projection")
        or projection.get("posthoc_chunk_projection")
    )
    if not isinstance(posthoc, dict):
        raise ValueError(f"Missing post-hoc projection statistics: {stats_path}")
    violations = step.get("violations_raw") or {}
    distance = step.get("projection_distance") or {}
    runtime = stats.get("runtime") or {}
    config = stats.get("config") or {}
    trajectories_per_second = _require_finite(
        "runtime.trajectories_per_second", runtime.get("trajectories_per_second")
    )
    rollout_seconds = _require_finite(
        "runtime.rollout_seconds", runtime.get("rollout_seconds")
    )
    if trajectories_per_second <= 0 or rollout_seconds <= 0:
        raise ValueError(f"Non-positive runtime in {stats_path}")
    row = {
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "method_id": METHOD_ID,
        "method": "Matched + post-hoc PAVA",
        "horizon": horizon,
        "n_histories": int(config["n_histories"]),
        "final_invalid_snapshot_rate": violations.get(
            "any_invalid_snapshot_rate"
        ),
        "final_crossed_book_rate": violations.get("crossed_book_rate"),
        "final_ask_ladder_violation": violations.get("ask_ladder_violation"),
        "final_bid_ladder_violation": violations.get("bid_ladder_violation"),
        "final_negative_volume_rate": violations.get("negative_volume_rate"),
        "diagnostic_projection_l2": distance.get("total_l2_distance"),
        "spread_relative_abs_error": _relative_error(
            "spread", step.get("spread_mean"), step.get("spread_real_mean")
        ),
        "volume_relative_abs_error": _relative_error(
            "volume", step.get("volume_mean"), step.get("volume_real_mean")
        ),
        "return_std_relative_abs_error": _relative_error(
            "return_std",
            step.get("mid_return_std"),
            step.get("mid_return_real_std"),
        ),
        "temporal_correlation": step.get("temporal_corr"),
        "return_w1": step.get("w1_mid_return"),
        "spread_w1": step.get("w1_spread"),
        "volume_w1": step.get("w1_volume"),
        "path_pre_projection_invalid_rate": posthoc.get(
            "pre_projection_any_invalid_snapshot_rate_mean"
        ),
        "path_cumulative_projection_l2": posthoc.get(
            "cumulative_projection_l2_mean"
        ),
        "path_projection_calls": posthoc.get(
            "n_projection_calls_per_trajectory"
        ),
        "rollout_seconds": rollout_seconds,
        "trajectories_per_second": trajectories_per_second,
        "lob_steps_per_second": trajectories_per_second * horizon,
        "stats_path": str(stats_path),
        "stats_sha256": stats_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "command_path": str(command_path),
        "command_sha256": command_hash,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
    }
    for field in CONSUMER_REFERENCE_FIELDS:
        _require_finite(field, row[field])
    invalid_rate = float(row["path_pre_projection_invalid_rate"])
    if not 0.0 <= invalid_rate <= 1.0:
        raise ValueError(f"path invalidity is outside [0, 1]: {invalid_rate}")
    if float(row["path_cumulative_projection_l2"]) < 0.0:
        raise ValueError("cumulative projection L2 is negative")
    return row


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def collect_reference(
    *,
    results_roots: list[Path],
    dataset: str,
    data_dir: Path,
    checkpoint_root: Path,
    train_seeds: list[int],
    eval_seeds: list[int],
    hist_len: int,
    pred_len: int,
    chunk_size: int,
    horizon: int,
    ddim_steps: int,
    batch_size: int,
    n_batches: int,
    device: str,
    checkpoint_name: str = "best.pt",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_filename(checkpoint_name)
    if len(set(train_seeds)) != len(train_seeds):
        raise ValueError("Duplicate training seeds")
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("Duplicate evaluation seeds")
    if len(results_roots) != len(eval_seeds):
        raise ValueError(
            f"Expected one result root per eval seed: roots={len(results_roots)} "
            f"eval_seeds={len(eval_seeds)}"
        )
    resolved_roots = [path.resolve() for path in results_roots]
    if len(set(resolved_roots)) != len(resolved_roots):
        raise ValueError("Duplicate result roots")
    data_dir = data_dir.resolve()
    checkpoint_root = checkpoint_root.resolve()
    expected_histories = batch_size * n_batches
    expected_calls = math.ceil(horizon / chunk_size)
    checkpoint_hashes: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    root_receipts: list[dict[str, Any]] = []
    observed_eval_seeds: set[int] = set()

    for root in resolved_roots:
        manifest_path = root / "run_manifest.json"
        manifest = _load_json(manifest_path)
        manifest_hash = sha256_file(manifest_path)
        _require_equal("manifest.dataset", manifest.get("dataset"), dataset)
        _require_equal(
            "manifest.checkpoint_root",
            _resolve_declared(manifest.get("checkpoint_root", "")),
            checkpoint_root,
        )
        _require_equal(
            "manifest.checkpoint_name",
            manifest.get("checkpoint_name", "best.pt"),
            checkpoint_name,
        )
        _require_equal("manifest.train_seeds", manifest.get("train_seeds"), train_seeds)
        eval_seed = int(manifest.get("eval_seed"))
        if eval_seed not in eval_seeds:
            raise ValueError(f"Unexpected eval seed {eval_seed}: {manifest_path}")
        if eval_seed in observed_eval_seeds:
            raise ValueError(f"Duplicate eval seed {eval_seed}")
        observed_eval_seeds.add(eval_seed)
        _require_equal("manifest.n_batches", manifest.get("n_batches"), n_batches)
        _require_equal("manifest.batch_size", manifest.get("batch_size"), batch_size)
        _require_equal("manifest.horizon", manifest.get("horizon"), horizon)
        _require_equal("manifest.ddim_steps", manifest.get("ddim_steps"), ddim_steps)
        _require_equal("manifest.methods", manifest.get("methods"), [METHOD_ID])
        runs = manifest.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"manifest.runs is not a list: {manifest_path}")
        if len(runs) != len(train_seeds):
            raise ValueError(
                f"manifest run count {len(runs)} != {len(train_seeds)}: "
                f"{manifest_path}"
            )
        run_by_seed: dict[int, dict[str, Any]] = {}
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"Malformed run in {manifest_path}")
            train_seed = int(run.get("train_seed"))
            if train_seed in run_by_seed:
                raise ValueError(f"Duplicate train seed {train_seed}: {manifest_path}")
            run_by_seed[train_seed] = run
        _require_equal("manifest run seeds", sorted(run_by_seed), sorted(train_seeds))

        expected_stats_paths: set[Path] = set()
        for train_seed in train_seeds:
            run = run_by_seed[train_seed]
            _require_equal("run.method", run.get("method"), METHOD_ID)
            _require_equal("run.status", run.get("status"), "completed")
            expected_checkpoint = (
                checkpoint_root
                / f"train_seed{train_seed}"
                / "matched_lob"
                / checkpoint_name
            ).resolve()
            if not expected_checkpoint.is_file():
                raise FileNotFoundError(expected_checkpoint)
            _require_equal(
                "run.checkpoint",
                _resolve_declared(run.get("checkpoint", "")),
                expected_checkpoint,
            )
            output_dir = (root / f"train_seed{train_seed}" / METHOD_ID).resolve()
            _require_equal(
                "run.output_dir",
                _resolve_declared(run.get("output_dir", "")),
                output_dir,
            )
            stats_path = output_dir / "snowball_stats.json"
            command_path = output_dir / "command.txt"
            expected_stats_paths.add(stats_path)
            stats = _load_json(stats_path)
            command = command_path.read_text().strip() if command_path.is_file() else None
            if not command:
                raise FileNotFoundError(command_path)
            _require_equal("run.command", run.get("command"), command)
            _validate_command(
                command,
                dataset=dataset,
                data_dir=data_dir,
                checkpoint=expected_checkpoint,
                output_dir=output_dir,
                eval_seed=eval_seed,
                hist_len=hist_len,
                pred_len=pred_len,
                chunk_size=chunk_size,
                horizon=horizon,
                ddim_steps=ddim_steps,
                batch_size=batch_size,
                n_batches=n_batches,
                device=device,
            )
            config = stats.get("config") or {}
            expected_config = {
                "dataset": dataset,
                "hist_len": hist_len,
                "pred_len": pred_len,
                "chunk_size": chunk_size,
                "horizon": horizon,
                "ddim_steps": ddim_steps,
                "batch_size": batch_size,
                "n_histories": expected_histories,
                "action_mode": "hold",
                "projection_mode": "exact",
                "sampler_projection_mode": "none",
                "compute_projection_distance": True,
                "representation": "matched_lob",
                "price_reference": "history_last_mid",
                "weights": "ema",
            }
            for field, expected in expected_config.items():
                _require_equal(f"stats.config.{field}", config.get(field), expected)
            _require_equal(
                "stats.config.data_dir",
                _resolve_declared(config.get("data_dir", "")),
                data_dir,
            )
            stats_checkpoint = stats.get("checkpoint") or {}
            _require_equal(
                "stats.checkpoint.path",
                _resolve_declared(stats_checkpoint.get("path", "")),
                expected_checkpoint,
            )
            _require_finite("stats.checkpoint.epoch", stats_checkpoint.get("epoch"))
            _require_finite(
                "stats.checkpoint.val_loss", stats_checkpoint.get("val_loss")
            )
            projection = stats.get("rollout_projection_stats") or {}
            horizon_projection = (
                (projection.get("by_horizon") or {}).get(f"step_{horizon}") or {}
            )
            posthoc = (
                horizon_projection.get("posthoc_chunk_projection")
                or projection.get("posthoc_chunk_projection")
            )
            if not isinstance(posthoc, dict):
                raise ValueError(f"Missing post-hoc projection stats: {stats_path}")
            _require_equal("projection.n_batches", posthoc.get("n_batches"), n_batches)
            _require_equal(
                "projection.n_trajectories",
                posthoc.get("n_trajectories"),
                expected_histories,
            )
            _require_equal(
                "projection.calls_per_trajectory",
                posthoc.get("n_projection_calls_per_trajectory"),
                expected_calls,
            )
            _require_equal(
                "projection.kernel_calls",
                posthoc.get("n_projection_kernel_calls"),
                n_batches * expected_calls,
            )
            if expected_checkpoint not in checkpoint_hashes:
                checkpoint_hashes[expected_checkpoint] = sha256_file(
                    expected_checkpoint
                )
            stats_hash = sha256_file(stats_path)
            command_hash = sha256_file(command_path)
            checkpoint_hash = checkpoint_hashes[expected_checkpoint]
            row = _extract_row(
                stats,
                train_seed=train_seed,
                eval_seed=eval_seed,
                horizon=horizon,
                stats_path=stats_path,
                stats_hash=stats_hash,
                checkpoint_path=expected_checkpoint,
                checkpoint_hash=checkpoint_hash,
                command_path=command_path,
                command_hash=command_hash,
                manifest_path=manifest_path,
                manifest_hash=manifest_hash,
            )
            rows.append(row)
            cells.append(
                {
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "status": "completed",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_hash,
                    "stats_path": str(stats_path),
                    "stats_sha256": stats_hash,
                    "checkpoint_path": str(expected_checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "command_path": str(command_path),
                    "command_sha256": command_hash,
                }
            )

        observed_stats_paths = {
            path.resolve() for path in root.glob("train_seed*/*/snowball_stats.json")
        }
        if observed_stats_paths != expected_stats_paths:
            raise ValueError(
                "stats file set mismatch "
                f"missing={sorted(expected_stats_paths-observed_stats_paths)} "
                f"orphan={sorted(observed_stats_paths-expected_stats_paths)}"
            )
        root_receipts.append(
            {
                "eval_seed": eval_seed,
                "root": str(root),
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
            }
        )

    _require_equal("observed eval seeds", sorted(observed_eval_seeds), sorted(eval_seeds))
    expected_cells = {
        (train_seed, eval_seed)
        for train_seed in train_seeds
        for eval_seed in eval_seeds
    }
    observed_cells = {
        (int(row["train_seed"]), int(row["eval_seed"])) for row in rows
    }
    _require_equal("reference cell set", observed_cells, expected_cells)
    if len(rows) != len(expected_cells):
        raise ValueError("Duplicate reference rows")
    rows.sort(key=lambda row: (int(row["train_seed"]), int(row["eval_seed"])))
    cells.sort(key=lambda cell: (cell["train_seed"], cell["eval_seed"]))
    root_receipts.sort(key=lambda receipt: receipt["eval_seed"])

    repo_root = Path(__file__).resolve().parent.parent
    source_paths = (
        Path(__file__).resolve(),
        repo_root / "scripts/run_all_method_rollout_panel.py",
        repo_root / "scripts/snowball_test.py",
        repo_root / "scripts/summarize_intermediate_consumer.py",
    )
    dataset_dir = data_dir / "processed" / "snapshot_l2" / dataset
    dataset_files = (dataset_dir / "manifest.yaml", dataset_dir / "normalization.json")
    for path in dataset_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = {
        "schema_version": 1,
        "status": "validated",
        "method_id": METHOD_ID,
        "expected_protocol": {
            "dataset": dataset,
            "data_dir": str(data_dir),
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_name": checkpoint_name,
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "hist_len": hist_len,
            "pred_len": pred_len,
            "chunk_size": chunk_size,
            "horizon": horizon,
            "ddim_steps": ddim_steps,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "n_histories_per_cell": expected_histories,
            "device": device,
            "weights": "ema",
            "action_mode": "hold",
            "projection_mode": "exact",
            "sampler_projection_mode": "none",
        },
        "completeness": {
            "expected_cells": len(expected_cells),
            "observed_cells": len(rows),
            "all_cells_complete": observed_cells == expected_cells,
        },
        "result_roots": root_receipts,
        "cells": cells,
        "dataset_receipt": {
            "dataset_dir": str(dataset_dir.resolve()),
            "manifest_sha256": sha256_file(dataset_files[0]),
            "normalization_sha256": sha256_file(dataset_files[1]),
        },
        "source_sha256": {
            str(path.resolve().relative_to(repo_root)): sha256_file(path)
            for path in source_paths
        },
    }
    return rows, validation


def write_reference(
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
    *,
    output_csv: Path,
    validation_json: Path,
) -> None:
    output_csv = output_csv.resolve()
    validation_json = validation_json.resolve()
    if output_csv == validation_json:
        raise ValueError("CSV and validation JSON paths must differ")
    for path in (output_csv, validation_json):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    csv_payload = _csv_bytes(rows)
    validation = {
        **validation,
        "reference_csv": {
            "path": str(output_csv),
            "sha256": sha256_bytes(csv_payload),
            "rows": len(rows),
        },
    }
    json_payload = (json.dumps(validation, indent=2, allow_nan=False) + "\n").encode()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    validation_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("xb") as handle:
        handle.write(csv_payload)
    with validation_json.open("xb") as handle:
        handle.write(json_payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-name",
        type=checkpoint_filename,
        default="best.pt",
        help="Checkpoint filename within each seed/representation directory.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--device", default="mps")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    rows, validation = collect_reference(
        results_roots=args.results_roots,
        dataset=args.dataset,
        data_dir=args.data_dir,
        checkpoint_root=args.checkpoint_root,
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        chunk_size=args.chunk_size,
        horizon=args.horizon,
        ddim_steps=args.ddim_steps,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
    )
    write_reference(
        rows,
        validation,
        output_csv=args.output_csv,
        validation_json=args.validation_json,
    )
    print(
        json.dumps(
            {
                "status": "validated",
                "cells": len(rows),
                "reference_csv": str(args.output_csv.resolve()),
                "validation_json": str(args.validation_json.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

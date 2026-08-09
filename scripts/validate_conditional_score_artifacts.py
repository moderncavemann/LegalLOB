#!/usr/bin/env python3
"""Fail-closed validation for the P2 conditional-score sampling artifacts.

The validator is intentionally separate from the sampler and summarizer.  It
does not generate or repair results: it verifies the frozen protocol, paired
histories, array contents, checkpoint contracts, and provenance before writing
one immutable JSON receipt.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


EXPECTED_ASSETS = ("AAPL", "AMZN", "GOOG")
EXPECTED_TRAIN_SEEDS = (0, 1, 2)
EXPECTED_SAMPLING_SEEDS = tuple(range(32))
EXPECTED_SHAPE = (512, 2, 20, 10)
METHOD_CONTRACT = {
    "raw_posthoc_pava": {
        "representation": "raw",
        "projection_mode": "exact",
        "evaluated_file": "generated_projected.npy",
        "feasibility_file": "generated_projected.npy",
        "feasibility_metrics_key": "structural_validity_projected",
    },
    "matched_posthoc_pava": {
        "representation": "matched_lob",
        "projection_mode": "exact",
        "evaluated_file": "generated_projected.npy",
        "feasibility_file": "generated_projected.npy",
        "feasibility_metrics_key": "structural_validity_projected",
    },
    "legal_lob": {
        "representation": "legal_lob",
        "projection_mode": "none",
        "evaluated_file": "generated_raw.npy",
        "feasibility_file": "generated_raw.npy",
        "feasibility_metrics_key": "structural_validity_raw",
    },
}
REQUIRED_RESULT_FILES = (
    "command.txt",
    "run.log",
    "metrics.json",
    "per_sample_metrics.npz",
    "generated_raw.npy",
    "generated_projected.npy",
    "real_future.npy",
    "persistence.npy",
)
DATA_INPUT_FILES = (
    "manifest.yaml",
    "metadata.json",
    "normalization.json",
    "schema_validation.json",
    "states.npy",
    "states_raw.npy",
    "split.npy",
    "segment_id.npy",
    "day_id.npy",
    "actions.npy",
    "source_row_idx.npy",
    "timestamps.npy",
)
SOURCE_FILES = (
    "scripts/run_conditional_score_panel.py",
    "scripts/evaluate_v2.py",
    "scripts/summarize_conditional_scores.py",
    "scripts/validate_conditional_score_artifacts.py",
    "genlob/models/genlob_v2.py",
    "genlob/models/sde.py",
    "genlob/models/sampling.py",
    "genlob/data/nasdaq.py",
    "genlob/eval/metrics.py",
    "genlob/utils/legal_lob.py",
    "genlob/utils/physics.py",
    "genlob/utils/device.py",
)
CAMPAIGN_REQUIRED_SOURCES = (
    "scripts/run_conditional_score_panel.py",
    "scripts/evaluate_v2.py",
    "genlob/models/sampling.py",
    "genlob/utils/legal_lob.py",
    "genlob/utils/physics.py",
)
CAMPAIGN_FROZEN_DATA_FILES = (
    "metadata.json",
    "manifest.yaml",
    "normalization.json",
    "schema_validation.json",
)


class ValidationError(RuntimeError):
    """Raised when a formal artifact violates the frozen P2 contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _file_receipt(path: Path, repo_root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required file missing: {path}")
    try:
        display = str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        display = str(path.resolve())
    return {
        "path": display,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_campaign_manifest(
    path: Path, *, repo_root: Path, data_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify validation-time inputs against the launch/preflight receipt."""
    _require(path.is_file(), f"campaign manifest missing: {path}")
    campaign = json.loads(path.read_text())
    _require(campaign.get("schema_version") == 1, "campaign schema_version must be 1")
    _require(bool(campaign.get("campaign_id")), "campaign_id missing")
    frozen_sources = campaign.get("sources_sha256")
    _require(isinstance(frozen_sources, dict), "campaign sources_sha256 missing")
    source_checks: dict[str, Any] = {}
    for relative in CAMPAIGN_REQUIRED_SOURCES:
        _require(relative in frozen_sources, f"campaign source hash missing: {relative}")
        actual = sha256_file(repo_root / relative)
        expected = frozen_sources[relative]
        _require(actual == expected, f"campaign source hash mismatch: {relative}")
        source_checks[relative] = {
            "launch_preflight_sha256": expected,
            "validation_time_sha256": actual,
            "matches": True,
        }

    frozen_datasets = campaign.get("datasets")
    _require(isinstance(frozen_datasets, dict), "campaign datasets missing")
    dataset_checks: dict[str, Any] = {}
    for asset in EXPECTED_ASSETS:
        dataset = f"PUBLIC_ITCH_2019_SAMPLES_{asset}_1s"
        frozen = frozen_datasets.get(dataset)
        _require(isinstance(frozen, dict), f"campaign dataset hashes missing: {dataset}")
        dataset_dir = data_root / "processed" / "snapshot_l2" / dataset
        file_checks: dict[str, Any] = {}
        for name in CAMPAIGN_FROZEN_DATA_FILES:
            _require(name in frozen, f"campaign data hash missing: {dataset}/{name}")
            actual = sha256_file(dataset_dir / name)
            expected = frozen[name]
            _require(actual == expected, f"campaign data hash mismatch: {dataset}/{name}")
            file_checks[name] = {
                "launch_preflight_sha256": expected,
                "validation_time_sha256": actual,
                "matches": True,
            }
        dataset_checks[dataset] = file_checks

    frozen_checkpoints = campaign.get("frozen_checkpoints_sha256")
    _require(
        isinstance(frozen_checkpoints, dict),
        "campaign frozen_checkpoints_sha256 missing",
    )
    evidence = {
        "manifest": _file_receipt(path, repo_root),
        "campaign_id": campaign["campaign_id"],
        "status_at_validation": campaign.get("status"),
        "source_hash_checks": source_checks,
        "dataset_hash_checks": dataset_checks,
    }
    return campaign, evidence


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        value = vars(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _finite_numbers(value: Any, prefix: str = "metrics") -> tuple[int, int]:
    """Require every numeric JSON leaf to be finite; null is non-numeric."""
    numeric = 0
    nulls = 0
    if value is None:
        return numeric, 1
    if isinstance(value, bool):
        return numeric, nulls
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"non-finite number at {prefix}")
        return 1, nulls
    if isinstance(value, dict):
        for key, item in value.items():
            n_item, null_item = _finite_numbers(item, f"{prefix}.{key}")
            numeric += n_item
            nulls += null_item
    elif isinstance(value, list):
        for index, item in enumerate(value):
            n_item, null_item = _finite_numbers(item, f"{prefix}[{index}]")
            numeric += n_item
            nulls += null_item
    return numeric, nulls


def _flag_value(tokens: list[str], flag: str) -> str:
    _require(flag in tokens, f"command is missing {flag}")
    index = tokens.index(flag)
    _require(index + 1 < len(tokens), f"command has no value after {flag}")
    return tokens[index + 1]


def _expected_checkpoint(
    checkpoint_root: Path, train_seed: int, method: str
) -> Path:
    representation = METHOD_CONTRACT[method]["representation"]
    return checkpoint_root / f"train_seed{train_seed}" / representation / "best.pt"


def validate_manifest_protocol(
    manifest: dict[str, Any],
    *,
    asset: str,
    result_root: Path,
    repo_root: Path,
) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Validate the exact 3 x 32 x 3 manifest and return its indexed runs."""
    expected_dataset = f"PUBLIC_ITCH_2019_SAMPLES_{asset}_1s"
    _require(manifest.get("dataset") == expected_dataset, f"{asset}: dataset mismatch")
    _require(
        manifest.get("train_seeds") == list(EXPECTED_TRAIN_SEEDS),
        f"{asset}: train seeds must be {list(EXPECTED_TRAIN_SEEDS)}",
    )
    _require(
        manifest.get("sampling_seeds") == list(EXPECTED_SAMPLING_SEEDS),
        f"{asset}: sampling seeds must be 0..31",
    )
    _require(manifest.get("n_batches") == 4, f"{asset}: n_batches must be 4")
    _require(manifest.get("batch_size") == 128, f"{asset}: batch_size must be 128")
    _require(manifest.get("n_histories") == 512, f"{asset}: n_histories must be 512")
    _require(manifest.get("ddim_steps") == 20, f"{asset}: ddim_steps must be 20")
    _require(
        sorted(manifest.get("methods", [])) == sorted(METHOD_CONTRACT),
        f"{asset}: method set mismatch",
    )

    expected_keys = {
        (train_seed, sampling_seed, method)
        for train_seed in EXPECTED_TRAIN_SEEDS
        for sampling_seed in EXPECTED_SAMPLING_SEEDS
        for method in METHOD_CONTRACT
    }
    runs = manifest.get("runs")
    _require(isinstance(runs, list), f"{asset}: manifest runs must be a list")
    _require(len(runs) == 288, f"{asset}: expected 288 runs, found {len(runs)}")
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    checkpoint_root = _resolved(manifest.get("checkpoint_root", ""), repo_root)
    for run in runs:
        key = (run.get("train_seed"), run.get("sampling_seed"), run.get("method"))
        _require(key in expected_keys, f"{asset}: unexpected run key {key}")
        _require(key not in indexed, f"{asset}: duplicate run key {key}")
        _require(run.get("status") == "completed", f"{asset}: {key} is not completed")
        _require(isinstance(run.get("command"), str), f"{asset}: {key} lacks command")
        expected_output = (
            result_root
            / f"train_seed{key[0]}"
            / f"sampling_seed{key[1]}"
            / str(key[2])
        ).resolve()
        _require(
            _resolved(run.get("output_dir", ""), repo_root) == expected_output,
            f"{asset}: {key} output_dir does not match canonical layout",
        )
        expected_checkpoint = _expected_checkpoint(
            checkpoint_root, int(key[0]), str(key[2])
        ).resolve()
        _require(
            _resolved(run.get("checkpoint", ""), repo_root) == expected_checkpoint,
            f"{asset}: {key} checkpoint path mismatch",
        )
        indexed[key] = run
    _require(set(indexed) == expected_keys, f"{asset}: incomplete run key grid")
    return indexed


def _checkpoint_receipt(
    path: Path,
    *,
    repo_root: Path,
    dataset: str,
    train_seed: int,
    representation: str,
) -> dict[str, Any]:
    receipt = _file_receipt(path, repo_root)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), f"checkpoint is not a mapping: {path}")
    args = checkpoint.get("args")
    config = checkpoint.get("config")
    _require(isinstance(args, dict), f"checkpoint args missing: {path}")
    _require(args.get("dataset") == dataset, f"checkpoint dataset mismatch: {path}")
    _require(args.get("seed") == train_seed, f"checkpoint train seed mismatch: {path}")
    _require(
        args.get("representation") == representation,
        f"checkpoint representation mismatch: {path}",
    )
    expected_args = {
        "hist_len": 50,
        "pred_len": 10,
        "action_conditioning": "none",
        "price_reference": "history_last_mid",
        "min_spread": 0.1,
        "min_gap": 0.1,
    }
    for key, expected in expected_args.items():
        _require(args.get(key) == expected, f"checkpoint {key} mismatch: {path}")
    config_json = _jsonable(config)
    args_json = _jsonable(args)
    _require(config_json is not None, f"checkpoint config missing: {path}")
    receipt.update(
        {
            "config_sha256": sha256_json(config_json),
            "args_sha256": sha256_json(args_json),
            "representation": representation,
            "dataset": dataset,
            "train_seed": train_seed,
        }
    )
    del checkpoint
    gc.collect()
    return receipt


def bind_campaign_checkpoint_hash(
    receipt: dict[str, Any], campaign_manifest: dict[str, Any]
) -> dict[str, Any]:
    checkpoint_key = receipt["path"]
    frozen = campaign_manifest.get("frozen_checkpoints_sha256")
    _require(isinstance(frozen, dict), "campaign frozen_checkpoints_sha256 missing")
    _require(
        checkpoint_key in frozen,
        f"campaign checkpoint hash missing: {checkpoint_key}",
    )
    expected = frozen[checkpoint_key]
    _require(
        receipt["sha256"] == expected,
        f"campaign checkpoint hash mismatch: {checkpoint_key}",
    )
    return {
        **receipt,
        "launch_preflight_sha256": expected,
        "matches_launch_preflight": True,
    }


def _validate_command(
    command: str,
    command_file: Path,
    *,
    run: dict[str, Any],
    dataset: str,
    train_seed: int,
    sampling_seed: int,
    method: str,
    repo_root: Path,
    data_root: Path,
    output_dir: Path,
) -> None:
    recorded = command_file.read_text().strip()
    _require(recorded == command, f"command manifest/file mismatch: {command_file}")
    tokens = shlex.split(command)
    contract = METHOD_CONTRACT[method]
    _require(
        _resolved(_flag_value(tokens, "--checkpoint"), repo_root)
        == _resolved(run["checkpoint"], repo_root),
        f"checkpoint flag mismatch: {command_file}",
    )
    _require(
        _resolved(_flag_value(tokens, "--data-dir"), repo_root) == data_root.resolve(),
        f"data directory flag mismatch: {command_file}",
    )
    _require(
        _resolved(_flag_value(tokens, "--output-dir"), repo_root)
        == output_dir.resolve(),
        f"output directory flag mismatch: {command_file}",
    )
    expected_flags = {
        "--dataset": dataset,
        "--split": "test",
        "--weights": "ema",
        "--n-batches": "4",
        "--batch-size": "128",
        "--ddim-steps": "20",
        "--projection-mode": contract["projection_mode"],
        "--sampler-projection-mode": "none",
        "--representation": "auto",
        "--action-conditioning": "none",
        "--seed": str(sampling_seed),
    }
    for flag, expected in expected_flags.items():
        _require(
            _flag_value(tokens, flag) == expected,
            f"{flag} mismatch in {command_file}",
        )
    _require("--skip-plots" in tokens, f"--skip-plots missing in {command_file}")


def _load_required_array(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(array.shape == EXPECTED_SHAPE, f"shape mismatch for {path}: {array.shape}")
    _require(np.issubdtype(array.dtype, np.number), f"non-numeric array: {path}")
    _require(bool(np.isfinite(array).all()), f"non-finite array values: {path}")
    return array


def _structural_audit(
    array: np.ndarray, *, min_spread: float, min_gap: float
) -> dict[str, Any]:
    prices = np.asarray(array[:, 0], dtype=np.float64)
    volumes = np.asarray(array[:, 1], dtype=np.float64)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    tolerance = max(1e-7, abs(min_spread) * 2e-5, abs(min_gap) * 2e-5)
    spread = ask[:, 0] - bid[:, 0]
    ask_gap = ask[:, 1:] - ask[:, :-1]
    bid_gap = bid[:, :-1] - bid[:, 1:]
    counts = {
        "spread_shortfall_count": int(np.count_nonzero(spread < min_spread - tolerance)),
        "ask_gap_shortfall_count": int(np.count_nonzero(ask_gap < min_gap - tolerance)),
        "bid_gap_shortfall_count": int(np.count_nonzero(bid_gap < min_gap - tolerance)),
        "negative_volume_count": int(np.count_nonzero(volumes < -tolerance)),
    }
    counts["feasible"] = all(value == 0 for value in counts.values())
    counts["tolerance"] = tolerance
    return counts


def _validate_metrics(
    metrics: dict[str, Any], *, method: str, metrics_path: Path
) -> dict[str, int]:
    contract = METHOD_CONTRACT[method]
    _require(
        metrics.get("representation") == contract["representation"],
        f"metrics representation mismatch: {metrics_path}",
    )
    _require(
        metrics.get("projection_mode") == contract["projection_mode"],
        f"metrics projection mode mismatch: {metrics_path}",
    )
    _require(
        metrics.get("sampler_projection_mode") == "none",
        f"sampler projection mode mismatch: {metrics_path}",
    )
    protocol = metrics.get("evaluation_protocol") or {}
    expected_protocol = {
        "sampling_mode": "uniform",
        "selected_windows": 512,
        "action_conditioning": "none",
    }
    for key, expected in expected_protocol.items():
        _require(protocol.get(key) == expected, f"metrics protocol {key} mismatch: {metrics_path}")
    for key in ("mse_raw", "mse_projected", "mae_raw", "mae_projected"):
        _require(key in metrics, f"required metric {key} missing: {metrics_path}")
    numeric, nulls = _finite_numbers(metrics, str(metrics_path))
    _require(numeric > 0, f"no numeric metrics found: {metrics_path}")
    structural = metrics.get(contract["feasibility_metrics_key"]) or {}
    for key in (
        "any_invalid_snapshot_rate",
        "any_invalid_trajectory_rate",
        "max_spread_shortfall",
        "max_ask_gap_shortfall",
        "max_bid_gap_shortfall",
        "max_negative_volume_magnitude",
    ):
        _require(structural.get(key) == 0.0, f"nonzero {key}: {metrics_path}")
    return {"numeric_leaves": numeric, "null_leaves": nulls}


def validate_method_artifact(
    run: dict[str, Any],
    *,
    repo_root: Path,
    result_root: Path,
    dataset: str,
    train_seed: int,
    sampling_seed: int,
    method: str,
    data_root: Path | None = None,
) -> dict[str, Any]:
    output_dir = (
        result_root
        / f"train_seed{train_seed}"
        / f"sampling_seed{sampling_seed}"
        / method
    )
    for name in REQUIRED_RESULT_FILES:
        _require((output_dir / name).is_file(), f"missing result file: {output_dir / name}")
    _validate_command(
        run["command"],
        output_dir / "command.txt",
        run=run,
        dataset=dataset,
        train_seed=train_seed,
        sampling_seed=sampling_seed,
        method=method,
        repo_root=repo_root,
        data_root=(data_root or repo_root / "data"),
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics_scan = _validate_metrics(metrics, method=method, metrics_path=metrics_path)

    arrays = {
        name: _load_required_array(output_dir / name)
        for name in (
            "generated_raw.npy",
            "generated_projected.npy",
            "real_future.npy",
            "persistence.npy",
        )
    }
    with np.load(output_dir / "per_sample_metrics.npz", allow_pickle=False) as sample_metrics:
        _require("dataset_index" in sample_metrics.files, f"dataset_index missing: {output_dir}")
        _require(sample_metrics["dataset_index"].shape == (512,), f"dataset_index shape mismatch: {output_dir}")
        for name in sample_metrics.files:
            value = sample_metrics[name]
            if np.issubdtype(value.dtype, np.number):
                _require(bool(np.isfinite(value).all()), f"non-finite per-sample metric {name}: {output_dir}")
        alignment = {
            name: np.asarray(sample_metrics[name]).copy()
            for name in ("dataset_index", "start_indices", "day_id", "segment_id")
            if name in sample_metrics.files
        }

    contract = METHOD_CONTRACT[method]
    feasibility = _structural_audit(
        arrays[contract["feasibility_file"]],
        min_spread=float(metrics.get("min_spread", 0.1)),
        min_gap=float(metrics.get("min_gap", 0.1)),
    )
    _require(feasibility["feasible"], f"structurally infeasible evaluated output: {output_dir}")
    hashes = {
        name: _file_receipt(output_dir / name, repo_root)
        for name in REQUIRED_RESULT_FILES
    }
    return {
        "alignment": alignment,
        "real_future": arrays["real_future.npy"],
        "persistence": arrays["persistence.npy"],
        "evaluated": arrays[contract["evaluated_file"]],
        "receipt": {
            "method": method,
            "output_dir": str(output_dir.resolve()),
            "metrics_scan": metrics_scan,
            "structural_feasibility": feasibility,
            "result_files": hashes,
        },
    }


def _pairwise_rmse(method_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    output: dict[str, float] = {}
    names = sorted(method_arrays)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            difference = np.asarray(method_arrays[left], dtype=np.float64) - np.asarray(
                method_arrays[right], dtype=np.float64
            )
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            _require(math.isfinite(rmse), f"non-finite pairwise RMSE: {left}/{right}")
            _require(rmse > 1e-6, f"pairwise RMSE <= 1e-6: {left}/{right} ({rmse})")
            output[f"{left}__vs__{right}"] = rmse
    return output


def _ensemble_pairwise_rmse(samples: list[np.ndarray], label: str) -> float:
    """Match the summarizer's ordered-pair ensemble-dispersion convention."""
    _require(len(samples) == 32, f"{label}: expected 32 sampling members")
    pair_sum = 0.0
    for left_index, left in enumerate(samples):
        for right in samples[left_index + 1 :]:
            difference = np.asarray(left, dtype=np.float64) - np.asarray(
                right, dtype=np.float64
            )
            per_history = np.sqrt(np.mean(np.square(difference), axis=(1, 2, 3)))
            pair_sum += float(np.mean(per_history))
    # This is E over ordered pairs, including the zero diagonal, as used by
    # summarize_conditional_scores.state_energy_and_crps.
    value = (2.0 / (len(samples) * len(samples))) * pair_sum
    _require(math.isfinite(value), f"{label}: non-finite ensemble pairwise RMSE")
    _require(value > 1e-6, f"{label}: ensemble pairwise RMSE <= 1e-6 ({value})")
    return value


def _require_aligned(method_outputs: dict[str, dict[str, Any]], label: str) -> None:
    reference_name = "legal_lob"
    reference = method_outputs[reference_name]
    for method, output in method_outputs.items():
        if method == reference_name:
            continue
        _require(
            set(output["alignment"]) == set(reference["alignment"]),
            f"alignment metadata fields differ for {label}: {method}",
        )
        for key, values in output["alignment"].items():
            _require(
                np.array_equal(values, reference["alignment"][key]),
                f"{key} mismatch for {label}: {method}",
            )
        _require(
            np.array_equal(output["real_future"], reference["real_future"]),
            f"real_future mismatch for {label}: {method}",
        )
        _require(
            np.array_equal(output["persistence"], reference["persistence"]),
            f"persistence mismatch for {label}: {method}",
        )


def validate_asset(
    asset: str,
    result_root: Path,
    *,
    repo_root: Path,
    data_root: Path,
    campaign_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = result_root / "run_manifest.json"
    _require(manifest_path.is_file(), f"manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    runs = validate_manifest_protocol(
        manifest, asset=asset, result_root=result_root, repo_root=repo_root
    )
    dataset = str(manifest["dataset"])
    checkpoint_root = _resolved(manifest["checkpoint_root"], repo_root)

    dataset_dir = data_root / "processed" / "snapshot_l2" / dataset
    data_receipts = {
        name: _file_receipt(dataset_dir / name, repo_root) for name in DATA_INPUT_FILES
    }
    frozen_dataset = campaign_manifest["datasets"][dataset]
    for name in CAMPAIGN_FROZEN_DATA_FILES:
        data_receipts[name]["launch_preflight_sha256"] = frozen_dataset[name]
        data_receipts[name]["matches_launch_preflight"] = (
            data_receipts[name]["sha256"] == frozen_dataset[name]
        )
        _require(
            data_receipts[name]["matches_launch_preflight"],
            f"campaign data hash mismatch: {dataset}/{name}",
        )
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    _require(metadata.get("dataset_id") == dataset, f"metadata dataset mismatch: {dataset_dir}")
    _require(metadata.get("symbol") == asset, f"metadata symbol mismatch: {dataset_dir}")
    _require(metadata.get("sampling") == "1s", f"metadata sampling mismatch: {dataset_dir}")

    checkpoint_receipts: dict[str, Any] = {}
    for train_seed in EXPECTED_TRAIN_SEEDS:
        for method, contract in METHOD_CONTRACT.items():
            checkpoint = _expected_checkpoint(checkpoint_root, train_seed, method).resolve()
            key = f"train_seed{train_seed}/{contract['representation']}"
            checkpoint_receipts[key] = _checkpoint_receipt(
                checkpoint,
                repo_root=repo_root,
                dataset=dataset,
                train_seed=train_seed,
                representation=str(contract["representation"]),
            )
            checkpoint_receipts[key] = bind_campaign_checkpoint_hash(
                checkpoint_receipts[key], campaign_manifest
            )

    cell_receipts: list[dict[str, Any]] = []
    cross_method_pairwise_min = float("inf")
    ensemble_pairwise_receipts: list[dict[str, Any]] = []
    ensemble_pairwise_min = float("inf")
    for train_seed in EXPECTED_TRAIN_SEEDS:
        ensemble_outputs: dict[str, list[np.ndarray]] = {
            method: [] for method in METHOD_CONTRACT
        }
        for sampling_seed in EXPECTED_SAMPLING_SEEDS:
            outputs = {
                method: validate_method_artifact(
                    runs[(train_seed, sampling_seed, method)],
                    repo_root=repo_root,
                    result_root=result_root,
                    dataset=dataset,
                    train_seed=train_seed,
                    sampling_seed=sampling_seed,
                    method=method,
                    data_root=data_root,
                )
                for method in METHOD_CONTRACT
            }
            label = f"{asset}/train_seed{train_seed}/sampling_seed{sampling_seed}"
            _require_aligned(outputs, label)
            pairwise = _pairwise_rmse(
                {method: output["evaluated"] for method, output in outputs.items()}
            )
            cross_method_pairwise_min = min(
                cross_method_pairwise_min, *pairwise.values()
            )
            for method, output in outputs.items():
                ensemble_outputs[method].append(
                    np.asarray(output["evaluated"], dtype=np.float32).copy()
                )
            cell_receipts.append(
                {
                    "train_seed": train_seed,
                    "sampling_seed": sampling_seed,
                    "pairwise_rmse": pairwise,
                    "methods": {
                        method: output["receipt"] for method, output in outputs.items()
                    },
                }
            )
            del outputs
        for method, samples in ensemble_outputs.items():
            label = f"{asset}/train_seed{train_seed}/{method}"
            value = _ensemble_pairwise_rmse(samples, label)
            ensemble_pairwise_min = min(ensemble_pairwise_min, value)
            ensemble_pairwise_receipts.append(
                {
                    "train_seed": train_seed,
                    "method": method,
                    "n_sampling_members": len(samples),
                    "pairwise_rmse": value,
                }
            )
        del ensemble_outputs

    return {
        "asset": asset,
        "dataset": dataset,
        "result_root": str(result_root.resolve()),
        "manifest": _file_receipt(manifest_path, repo_root),
        "expected_cells": 288,
        "completed_cells": len(runs),
        "failed_cells": 0,
        "expected_paired_cells": 96,
        "paired_cells": len(cell_receipts),
        "minimum_cross_method_pairwise_rmse": cross_method_pairwise_min,
        "minimum_ensemble_pairwise_rmse": ensemble_pairwise_min,
        "ensemble_pairwise_rmse": ensemble_pairwise_receipts,
        "data": data_receipts,
        "checkpoints": checkpoint_receipts,
        "cells": cell_receipts,
    }


def build_validation_receipt(
    asset_roots: dict[str, Path],
    *,
    repo_root: Path,
    data_root: Path,
    campaign_manifest_path: Path,
) -> dict[str, Any]:
    _require(
        set(asset_roots) == set(EXPECTED_ASSETS),
        f"asset roots must be exactly {list(EXPECTED_ASSETS)}",
    )
    campaign_manifest, campaign_evidence = validate_campaign_manifest(
        campaign_manifest_path, repo_root=repo_root, data_root=data_root
    )
    source_receipts = {
        relative: _file_receipt(repo_root / relative, repo_root)
        for relative in SOURCE_FILES
    }
    for relative in CAMPAIGN_REQUIRED_SOURCES:
        source_receipts[relative]["launch_preflight_sha256"] = campaign_manifest[
            "sources_sha256"
        ][relative]
        source_receipts[relative]["matches_launch_preflight"] = True
    assets = {
        asset: validate_asset(
            asset,
            asset_roots[asset].resolve(),
            repo_root=repo_root,
            data_root=data_root,
            campaign_manifest=campaign_manifest,
        )
        for asset in EXPECTED_ASSETS
    }
    return {
        "schema_version": 1,
        "status": "VERIFIED",
        "scope": "P2 cross-setting conditional proper-score artifact integrity",
        "protocol": {
            "assets": list(EXPECTED_ASSETS),
            "train_seeds": list(EXPECTED_TRAIN_SEEDS),
            "sampling_seeds": list(EXPECTED_SAMPLING_SEEDS),
            "methods": list(METHOD_CONTRACT),
            "n_histories": 512,
            "array_shape": list(EXPECTED_SHAPE),
            "expected_completed_cells_per_asset": 288,
            "expected_paired_cells_per_asset": 96,
            "cross_method_pairwise_rmse_floor": 1e-6,
            "ensemble_pairwise_rmse_floor": 1e-6,
        },
        "checks": {
            "campaign_launch_preflight_hashes_match": True,
            "manifest_protocol": True,
            "all_cells_completed": True,
            "checkpoint_and_projection_contract": True,
            "common_history_alignment": True,
            "array_shape_and_finiteness": True,
            "evaluated_outputs_structurally_feasible": True,
            "cross_method_outputs_nonidentical": True,
            "sampling_ensembles_noncollapsed": True,
            "source_data_checkpoint_result_hashes_recorded": True,
        },
        "campaign_manifest": campaign_evidence,
        "denominators": {
            "per_asset": {
                asset: {
                    "expected_cells": 288,
                    "completed_cells": assets[asset]["completed_cells"],
                    "failed_cells": assets[asset]["failed_cells"],
                    "expected_paired_cells": 96,
                    "paired_cells": assets[asset]["paired_cells"],
                }
                for asset in EXPECTED_ASSETS
            },
            "cross_asset": {
                "expected_cells": 864,
                "completed_cells": sum(
                    assets[asset]["completed_cells"] for asset in EXPECTED_ASSETS
                ),
                "failed_cells": sum(
                    assets[asset]["failed_cells"] for asset in EXPECTED_ASSETS
                ),
                "expected_paired_cells": 288,
                "paired_cells": sum(
                    assets[asset]["paired_cells"] for asset in EXPECTED_ASSETS
                ),
            },
        },
        "sources": source_receipts,
        "assets": assets,
    }


def parse_asset_roots(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        asset, separator, path = value.partition("=")
        asset = asset.upper()
        if not separator or not asset or not path:
            raise argparse.ArgumentTypeError(
                f"invalid --asset-root {value!r}; expected ASSET=PATH"
            )
        if asset in result:
            raise argparse.ArgumentTypeError(f"duplicate asset root: {asset}")
        result[asset] = Path(path)
    return result


def write_validation_receipt(receipt: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite validation JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        "--asset-root",
        dest="asset_roots",
        action="append",
        required=True,
        help="Repeat exactly three times as AAPL=PATH, AMZN=PATH, GOOG=PATH",
    )
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.validation_json.exists():
        raise FileExistsError(
            f"Refusing to overwrite validation JSON: {args.validation_json}"
        )
    repo_root = args.repo_root.resolve()
    data_root = _resolved(args.data_root, repo_root)
    asset_roots = {
        asset: _resolved(path, repo_root)
        for asset, path in parse_asset_roots(args.asset_roots).items()
    }
    receipt = build_validation_receipt(
        asset_roots,
        repo_root=repo_root,
        data_root=data_root,
        campaign_manifest_path=_resolved(args.campaign_manifest, repo_root),
    )
    write_validation_receipt(receipt, args.validation_json)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "validation_json": str(args.validation_json.resolve()),
                "completed_cells": {
                    asset: details["completed_cells"]
                    for asset, details in receipt["assets"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

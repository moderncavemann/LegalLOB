#!/usr/bin/env python3
"""Validate GenLOB dataset manifests and snapshot Level-2 tensor schemas."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
PURGED_SPLIT = -1


def _resolve_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _rate(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def _split_sequence(split: np.ndarray) -> list[int]:
    if split.size == 0:
        return []
    sequence = [int(split[0])]
    for value in split[1:]:
        value = int(value)
        if value != sequence[-1]:
            sequence.append(value)
    return sequence


def _scan_raw_exchange_validity(
    raw: np.ndarray,
    tick_size: float,
    chunk_size: int = 100_000,
) -> dict[str, float | int]:
    """Stream raw snapshots to audit tick-grid and integer-share semantics."""
    if raw.ndim != 3 or raw.shape[1:] != (2, 20):
        raise ValueError(f"raw states must have shape [T,2,20], got {raw.shape}")
    counts = {
        "prices": 0,
        "off_tick_prices": 0,
        "volumes": 0,
        "noninteger_volumes": 0,
        "negative_volumes": 0,
    }
    max_price_tick_error = 0.0
    max_volume_integer_error = 0.0
    for start in range(0, len(raw), chunk_size):
        block = np.asarray(raw[start:start + chunk_size], dtype=np.float64)
        prices = block[:, 0]
        volumes = block[:, 1]
        tick_values = prices / tick_size
        price_error = np.abs(tick_values - np.rint(tick_values))
        volume_error = np.abs(volumes - np.rint(volumes))
        counts["prices"] += int(prices.size)
        counts["off_tick_prices"] += int((price_error > 1e-6).sum())
        counts["volumes"] += int(volumes.size)
        counts["noninteger_volumes"] += int((volume_error > 1e-6).sum())
        counts["negative_volumes"] += int((volumes < 0).sum())
        max_price_tick_error = max(max_price_tick_error, float(price_error.max(initial=0.0)))
        max_volume_integer_error = max(
            max_volume_integer_error, float(volume_error.max(initial=0.0))
        )
    return {
        "off_tick_price_rate": counts["off_tick_prices"] / max(counts["prices"], 1),
        "noninteger_volume_rate": counts["noninteger_volumes"] / max(counts["volumes"], 1),
        "negative_volume_rate": counts["negative_volumes"] / max(counts["volumes"], 1),
        "max_price_tick_error": max_price_tick_error,
        "max_volume_integer_error": max_volume_integer_error,
        "rows_scanned": int(len(raw)),
    }


def validate_snapshot_dataset(
    dataset_id: str,
    dataset_cfg: dict[str, Any],
    manifest_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    required_paths = {
        "states": "state_path",
        "actions": "action_path",
        "timestamps": "timestamps_path",
        "day_id": "day_id_path",
        "split": "split_path",
        "normalization": "normalization_path",
        "metadata": "metadata_path",
    }
    paths = {}
    for label, key in required_paths.items():
        value = dataset_cfg.get(key)
        if not value:
            errors.append(f"missing manifest key: {key}")
            continue
        path = _resolve_path(value, manifest_path)
        paths[label] = path
        if not path.exists():
            errors.append(f"missing {label} file: {path}")

    if errors:
        return {
            "dataset": dataset_id,
            "status": "fail",
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }

    states = np.load(paths["states"], mmap_mode="r")
    actions = np.load(paths["actions"], mmap_mode="r")
    timestamps = np.load(paths["timestamps"], mmap_mode="r")
    day_id = np.load(paths["day_id"], mmap_mode="r")
    split = np.load(paths["split"], mmap_mode="r")
    normalization = _load_json(paths["normalization"])
    metadata = _load_json(paths["metadata"])

    raw_path_value = dataset_cfg.get("raw_state_path")
    if raw_path_value:
        raw_path = _resolve_path(raw_path_value, manifest_path)
        if not raw_path.exists():
            errors.append(f"missing raw states file: {raw_path}")
        else:
            raw_states = np.load(raw_path, mmap_mode="r")
            tick_size = float(normalization.get("tick_size_raw", 0.0))
            if tick_size <= 0:
                warnings.append("raw states exist but tick_size_raw is unavailable")
            else:
                try:
                    raw_validity = _scan_raw_exchange_validity(raw_states, tick_size)
                    metrics["raw_exchange_validity"] = raw_validity
                    for key in [
                        "off_tick_price_rate", "noninteger_volume_rate", "negative_volume_rate"
                    ]:
                        if raw_validity[key] > 0:
                            errors.append(f"raw {key} is nonzero: {raw_validity[key]:.8g}")
                except ValueError as exc:
                    errors.append(str(exc))

    metrics["paths"] = {k: str(v) for k, v in paths.items()}
    metrics["shapes"] = {
        "states": list(states.shape),
        "actions": list(actions.shape),
        "timestamps": list(timestamps.shape),
        "day_id": list(day_id.shape),
        "split": list(split.shape),
    }

    if states.ndim != 3 or states.shape[1:] != (2, 20):
        errors.append(f"states must have shape [T, 2, 20], got {states.shape}")
    if actions.ndim != 2:
        errors.append(f"actions must have shape [T-1, A], got {actions.shape}")

    if states.ndim == 3:
        T = states.shape[0]
        if actions.ndim == 2 and actions.shape[0] != T - 1:
            errors.append(f"actions length must be T-1={T - 1}, got {actions.shape[0]}")
        for label, arr in [("timestamps", timestamps), ("day_id", day_id), ("split", split)]:
            if arr.shape[0] != T:
                errors.append(f"{label} length must be T={T}, got {arr.shape[0]}")

    if not np.isfinite(states).all():
        errors.append("states contain NaN or Inf")
    if not np.isfinite(actions).all():
        errors.append("actions contain NaN or Inf")
    if not np.isfinite(timestamps).all():
        errors.append("timestamps contain NaN or Inf")
    if len(timestamps) > 1 and len(day_id) == len(timestamps):
        same_day = day_id[1:] == day_id[:-1]
        timestamp_reversal = same_day & (timestamps[1:] < timestamps[:-1])
        metrics["timestamp_reversal_rate_within_day"] = _rate(timestamp_reversal)
        if timestamp_reversal.any():
            errors.append(
                f"timestamps reverse within a day at {int(timestamp_reversal.sum())} transitions"
            )

    if states.ndim == 3 and states.shape[1:] == (2, 20):
        ask = states[:, 0, :10]
        bid = states[:, 0, 10:]
        volumes = states[:, 1, :]
        min_spread = float(normalization.get("min_spread_normalized", 1e-6))
        min_gap = float(normalization.get("min_level_gap_normalized", 0.0))
        metrics["min_spread"] = min_spread
        metrics["min_gap"] = min_gap
        effective_tolerance = max(
            float(tolerance),
            1e-7,
            abs(min_spread) * 2e-5,
            abs(min_gap) * 2e-5,
        )
        metrics["constraint_tolerance"] = effective_tolerance

        ask_violation = ask[:, 1:] - ask[:, :-1] < min_gap - effective_tolerance
        bid_violation = bid[:, :-1] - bid[:, 1:] < min_gap - effective_tolerance
        crossed = ask[:, 0] - bid[:, 0] < min_spread - effective_tolerance
        negative_volume = volumes < -effective_tolerance

        metrics["legality"] = {
            "crossed_book_rate": _rate(crossed),
            "ask_ladder_violation": _rate(ask_violation),
            "bid_ladder_violation": _rate(bid_violation),
            "negative_volume_rate": _rate(negative_volume),
            "spread_mean": float((ask[:, 0] - bid[:, 0]).mean()),
            "volume_mean": float(volumes.mean()),
        }
        for key, value in metrics["legality"].items():
            if key.endswith("rate") or key.endswith("violation"):
                if value > 0:
                    errors.append(f"{key} is nonzero: {value:.8g}")

    valid_split_values = set(np.unique(split).astype(int).tolist())
    metrics["split_counts"] = {
        ("purged" if value == PURGED_SPLIT else SPLIT_NAMES.get(value, str(value))):
        int((split[:-1] == value).sum())
        for value in sorted(valid_split_values)
    }
    if not valid_split_values.issubset({PURGED_SPLIT, 0, 1, 2}):
        errors.append(
            f"split values must be subset of {{-1,0,1,2}}, got {sorted(valid_split_values)}"
        )
    sequence = _split_sequence(split[split != PURGED_SPLIT])
    metrics["split_sequence"] = [SPLIT_NAMES.get(value, str(value)) for value in sequence]
    if sequence != sorted(sequence):
        errors.append(f"split must be contiguous train->val->test, got sequence {sequence}")

    unique_day = np.unique(day_id)
    placeholder_day = bool(unique_day.size == 1 and int(unique_day[0]) < 0)
    metrics["day_id"] = {
        "unique_count": int(unique_day.size),
        "placeholder": placeholder_day,
        "status": metadata.get("day_id_status", "unknown"),
    }
    if placeholder_day:
        warnings.append("day_id is a placeholder; split/day leakage cannot be audited for this dataset")
    else:
        split_policy = metadata.get("split_policy") or {}
        single_day_chronological = str(
            split_policy.get("type") or split_policy.get("policy") or ""
        ).startswith("single_day_")
        for day in unique_day:
            day_splits = {
                int(value) for value in split[day_id == day].tolist()
                if int(value) != PURGED_SPLIT
            }
            if len(day_splits) > 1:
                message = f"day_id={int(day)} spans multiple splits: {sorted(day_splits)}"
                if single_day_chronological:
                    warnings.append(
                        message + "; accepted only as a purged single-day chronological validation protocol"
                    )
                else:
                    errors.append(message)

    if not dataset_cfg.get("has_true_timestamps", False):
        warnings.append("timestamps are not true exchange timestamps")
    if not dataset_cfg.get("has_message_data", False):
        warnings.append("raw LOBSTER message/orderbook data is missing; message-level baselines are blocked")
    if dataset_cfg.get("action_label_status") != "message_level_aggressor_side":
        warnings.append("actions are not message-level aggressor-side labels; action causality tests are diagnostic only")

    if normalization.get("fit_split") not in {"train", "train_only"}:
        warnings.append(
            "normalization fit_split is not independently verified as train-only: "
            f"{normalization.get('fit_split')}"
        )

    status = "fail" if errors else ("pass_with_warnings" if warnings else "pass")
    return {
        "dataset": dataset_id,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict-warnings", action="store_true",
                        help="Return nonzero when validation has warnings.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    datasets = manifest.get("datasets", {})
    if args.dataset not in datasets:
        print(json.dumps({
            "dataset": args.dataset,
            "status": "fail",
            "errors": [f"dataset not found in manifest: {args.dataset}"],
        }, indent=2))
        return 1

    report = validate_snapshot_dataset(
        args.dataset,
        datasets[args.dataset],
        manifest_path,
        tolerance=args.tolerance,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")

    if report["errors"]:
        return 1
    if report["warnings"] and args.strict_warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Day-stratified audit of frozen public-ITCH 1 Hz generations.

This script never trains or samples a model.  It verifies the frozen evaluation
panel, slices the paired histories by their true test-day labels, and aggregates
evaluation seeds within each checkpoint before reporting variation across
training seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.eval.metrics import wasserstein_1d
from scripts.summarize_financial_sensitivity import (
    admissible_snapshot_mask,
    functional_vectors,
)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    output_file: str
    representation: str
    projection_mode: str
    deterministic: bool = False


METHODS = {
    item.method_id: item
    for item in (
        MethodSpec(
            "persistence",
            "Persistence",
            "generated_raw.npy",
            "deterministic_baseline",
            "none",
            True,
        ),
        MethodSpec(
            "raw_posthoc_pava",
            "Raw+Post",
            "generated_projected.npy",
            "raw",
            "exact",
        ),
        MethodSpec(
            "matched_no_projection",
            "Matched",
            "generated_raw.npy",
            "matched_lob",
            "none",
        ),
        MethodSpec(
            "matched_posthoc_pava",
            "Matched+Post",
            "generated_projected.npy",
            "matched_lob",
            "exact",
        ),
        MethodSpec(
            "legal_no_projection",
            "LegalLOB",
            "generated_raw.npy",
            "legal_lob",
            "none",
        ),
    )
}

METRIC_DIRECTIONS = {
    "output_mse": "down",
    "final_invalid_snapshot_rate": "down",
    "final_invalid_trajectory_rate": "down",
    "diagnostic_projection_l2": "down",
    "microprice_offset_w1_ticks": "down",
    "liquidity_5tick_w1_target_units": "down",
    "market_order_slippage_w1_ticks": "down",
}

REQUIRED_SAMPLE_KEYS = (
    "dataset_index",
    "day_id",
    "segment_id",
    "projection_l2",
)
START_INDEX_KEYS = ("start_indices", "start_index")


class IntegrityError(RuntimeError):
    """Frozen inputs violate the declared paired-panel contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_signature(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def result_dir(
    root: Path, symbol: str, train_seed: int, eval_seed: int, method_id: str
) -> Path:
    base = root / symbol / f"train_seed{train_seed}"
    if eval_seed:
        base = base / "seeds" / f"seed{eval_seed}"
    return base / method_id


def _positive_float(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping.get(key, 0.0))
    if not math.isfinite(value) or value <= 0:
        raise IntegrityError(f"normalization field {key!r} must be finite and positive")
    return value


def _method_eval_seeds(spec: MethodSpec, eval_seeds: tuple[int, ...]) -> tuple[int, ...]:
    if not spec.deterministic:
        return eval_seeds
    if 0 not in eval_seeds:
        raise IntegrityError("Persistence requires eval seed 0 in --eval-seeds")
    return (0,)


def _metric_row(
    generated: np.ndarray,
    real: np.ndarray,
    projection_l2: np.ndarray,
    normalization: dict[str, Any],
    target_size: float,
) -> dict[str, float | int]:
    generated = np.asarray(generated)
    real = np.asarray(real)
    n_histories = int(generated.shape[0])
    horizon = int(generated.shape[-1])
    finite_snapshot = np.isfinite(generated).all(axis=(1, 2))
    admissible = admissible_snapshot_mask(generated, normalization).reshape(
        n_histories, horizon
    )
    valid = finite_snapshot & admissible

    generated_vectors, _ = functional_vectors(
        generated, normalization, target_size=target_size
    )
    real_vectors, _ = functional_vectors(real, normalization, target_size=target_size)
    squared_error = (
        generated.astype(np.float64, copy=False)
        - real.astype(np.float64, copy=False)
    ) ** 2
    projection_values = np.asarray(projection_l2, dtype=np.float64)
    projection_values = projection_values[np.isfinite(projection_values)]
    return {
        "n_histories": n_histories,
        "n_snapshots": n_histories * horizon,
        "output_mse": float(np.mean(squared_error)),
        "final_invalid_snapshot_rate": float(1.0 - valid.mean()),
        "final_invalid_trajectory_rate": float(np.mean(np.any(~valid, axis=1))),
        "diagnostic_projection_l2": (
            float(projection_values.mean()) if projection_values.size else float("nan")
        ),
        "microprice_offset_w1_ticks": wasserstein_1d(
            generated_vectors["microprice_offset"],
            real_vectors["microprice_offset"],
        ),
        "liquidity_5tick_w1_target_units": wasserstein_1d(
            generated_vectors["liquidity_5tick"],
            real_vectors["liquidity_5tick"],
        ),
        "market_order_slippage_w1_ticks": wasserstein_1d(
            generated_vectors["market_order_slippage"],
            real_vectors["market_order_slippage"],
        ),
    }


def hierarchical_aggregate(
    rows: list[dict[str, Any]],
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Average eval seeds first, then summarize independent checkpoints."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["symbol"], row["day_id"], row["method_id"], row["method"])
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        deterministic = bool(group[0]["deterministic"])
        expected_eval = (0,) if deterministic else eval_seeds
        checkpoint_values: dict[str, list[float]] = {
            metric: [] for metric in METRIC_DIRECTIONS
        }
        eval_counts: list[int] = []
        for train_seed in train_seeds:
            seed_rows = [row for row in group if row["train_seed"] == train_seed]
            observed_eval = tuple(sorted(int(row["eval_seed"]) for row in seed_rows))
            if observed_eval != expected_eval:
                raise IntegrityError(
                    f"{key}: train_seed={train_seed} eval cells {observed_eval} "
                    f"!= {expected_eval}"
                )
            eval_counts.append(len(seed_rows))
            for metric in METRIC_DIRECTIONS:
                values = [float(row[metric]) for row in seed_rows]
                if not all(math.isfinite(value) for value in values):
                    raise IntegrityError(f"{key}: non-finite {metric}")
                checkpoint_values[metric].append(float(np.mean(values)))

        item: dict[str, Any] = {
            "symbol": key[0],
            "day_id": key[1],
            "method_id": key[2],
            "method": key[3],
            "deterministic": deterministic,
            "n_source_training_seed_copies": len(train_seeds),
            "n_independent_training_seeds": 0 if deterministic else len(train_seeds),
            "n_eval_seeds_per_training_seed": eval_counts,
            "n_histories_per_cell": int(group[0]["n_histories"]),
        }
        for metric, values in checkpoint_values.items():
            if deterministic:
                if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
                    raise IntegrityError(
                        f"{key}: deterministic copies disagree for {metric}: {values}"
                    )
                item[f"{metric}_mean"] = values[0]
                item[f"{metric}_std"] = None
            else:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
            item[f"{metric}_min"] = float(np.min(values))
            item[f"{metric}_max"] = float(np.max(values))
        output.append(item)
    return output


def worst_day_aggregate(
    rows: list[dict[str, Any]],
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["symbol"], row["method_id"], row["method"])
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        deterministic = bool(group[0]["deterministic"])
        item: dict[str, Any] = {
            "symbol": key[0],
            "method_id": key[1],
            "method": key[2],
            "deterministic": deterministic,
            "n_independent_training_seeds": 0 if deterministic else len(train_seeds),
        }
        for metric, direction in METRIC_DIRECTIONS.items():
            worst_values: list[float] = []
            worst_days: Counter[int] = Counter()
            source_train_seeds = (train_seeds[0],) if deterministic else train_seeds
            for train_seed in source_train_seeds:
                seed_rows = (
                    group
                    if deterministic
                    else [row for row in group if row["train_seed"] == train_seed]
                )
                day_values: dict[int, float] = {}
                for day in sorted({int(row["day_id"]) for row in seed_rows}):
                    values = [
                        float(row[metric])
                        for row in seed_rows
                        if int(row["day_id"]) == day
                        and (deterministic or int(row["train_seed"]) == train_seed)
                    ]
                    day_values[day] = float(np.mean(values))
                chooser = max if direction == "down" else min
                worst_value = chooser(day_values.values())
                worst_values.append(worst_value)
                for day, value in day_values.items():
                    if math.isclose(value, worst_value, rel_tol=1e-12, abs_tol=1e-12):
                        worst_days[day] += 1
            item[f"{metric}_worst_mean"] = float(np.mean(worst_values))
            item[f"{metric}_worst_std"] = (
                None
                if deterministic or len(worst_values) < 2
                else float(np.std(worst_values, ddof=1))
            )
            item[f"{metric}_worst_day_occurrences"] = dict(sorted(worst_days.items()))
        output.append(item)
    return output


def _validate_window_membership(
    starts: np.ndarray,
    reported_days: np.ndarray,
    reported_segments: np.ndarray,
    test_days: np.ndarray,
    test_segments: np.ndarray,
    hist_len: int,
    pred_len: int,
) -> None:
    width = hist_len + pred_len
    for position, start_value in enumerate(starts):
        start = int(start_value)
        stop = start + width
        if start < 0 or stop > len(test_days):
            raise IntegrityError(f"selected window [{start}, {stop}) is out of test bounds")
        if not np.all(test_days[start:stop] == test_days[start]):
            raise IntegrityError(f"selected window {start} crosses a trading day")
        if not np.all(test_segments[start:stop] == test_segments[start]):
            raise IntegrityError(f"selected window {start} crosses a continuity segment")
        if int(reported_days[position]) != int(test_days[start]):
            raise IntegrityError(f"reported day_id disagrees at selected window {start}")
        if int(reported_segments[position]) != int(test_segments[start]):
            raise IntegrityError(f"reported segment_id disagrees at selected window {start}")


def collect_rows(
    results_root: Path,
    data_root: Path,
    symbols: tuple[str, ...],
    days: tuple[int, ...],
    method_ids: tuple[str, ...],
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    hist_len: int,
    pred_len: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    if len(set(symbols)) != len(symbols) or len(set(days)) != len(days):
        raise IntegrityError("symbols and days must not contain duplicates")
    if len(set(train_seeds)) != len(train_seeds) or len(set(eval_seeds)) != len(eval_seeds):
        raise IntegrityError("seed lists must not contain duplicates")
    unknown = set(method_ids) - set(METHODS)
    if unknown:
        raise IntegrityError(f"unknown methods: {sorted(unknown)}")
    if hist_len <= 0 or pred_len <= 0:
        raise IntegrityError("history and prediction lengths must be positive")

    source_records: dict[Path, dict[str, Any]] = {}

    def record(path: Path) -> str:
        resolved = path.resolve()
        if not resolved.exists():
            raise IntegrityError(f"missing frozen input: {resolved}")
        if resolved not in source_records:
            source_records[resolved] = {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        return str(source_records[resolved]["sha256"])

    rows: list[dict[str, Any]] = []
    targets: dict[str, float] = {}
    symbol_receipts: dict[str, Any] = {}
    for symbol in symbols:
        dataset_dir = data_root / f"PUBLIC_ITCH_2019_SAMPLES_{symbol}_1s"
        dataset_paths = {
            name: dataset_dir / name
            for name in (
                "states.npy",
                "split.npy",
                "day_id.npy",
                "segment_id.npy",
                "normalization.json",
            )
        }
        for path in dataset_paths.values():
            record(path)
        normalization = json.loads(dataset_paths["normalization.json"].read_text())
        _positive_float(normalization, "ticks_per_price_unit")
        _positive_float(normalization, "volume_scale")
        _positive_float(normalization, "min_spread_normalized")
        _positive_float(normalization, "min_level_gap_normalized")
        states = np.load(dataset_paths["states.npy"], mmap_mode="r")
        split = np.load(dataset_paths["split.npy"], mmap_mode="r")
        full_days = np.load(dataset_paths["day_id.npy"], mmap_mode="r")
        full_segments = np.load(dataset_paths["segment_id.npy"], mmap_mode="r")
        if states.ndim != 3 or states.shape[1:] != (2, 20):
            raise IntegrityError(f"{symbol}: dataset states shape {states.shape}")
        if not (len(states) == len(split) == len(full_days) == len(full_segments)):
            raise IntegrityError(f"{symbol}: dataset arrays are not row-aligned")
        test_mask = np.asarray(split) == 2
        test_days = np.asarray(full_days[test_mask])
        test_segments = np.asarray(full_segments[test_mask])
        observed_test_days = tuple(sorted(int(value) for value in np.unique(test_days)))
        if observed_test_days != tuple(sorted(days)):
            raise IntegrityError(
                f"{symbol}: test days {observed_test_days} != requested {tuple(sorted(days))}"
            )
        test_states = np.asarray(states[test_mask])[..., None]
        _, target_size = functional_vectors(test_states, normalization)
        if not math.isfinite(target_size) or target_size <= 0:
            raise IntegrityError(f"{symbol}: invalid full-test target size {target_size}")
        targets[symbol] = float(target_size)

        pairing_signature: dict[str, str] | None = None
        paired_real_sha256: str | None = None
        expected_day_counts: dict[int, int] | None = None
        for train_seed in train_seeds:
            for method_id in method_ids:
                spec = METHODS[method_id]
                for eval_seed in _method_eval_seeds(spec, eval_seeds):
                    path = result_dir(
                        results_root, symbol, train_seed, eval_seed, method_id
                    )
                    required = {
                        "samples": path / "per_sample_metrics.npz",
                        "raw": path / "generated_raw.npy",
                        "projected": path / "generated_projected.npy",
                        "real": path / "real_future.npy",
                        "metrics": path / "metrics.json",
                    }
                    hashes = {name: record(file_path) for name, file_path in required.items()}
                    metrics = json.loads(required["metrics"].read_text())
                    if metrics.get("representation") != spec.representation:
                        raise IntegrityError(
                            f"{path}: representation={metrics.get('representation')} "
                            f"!= {spec.representation}"
                        )
                    if metrics.get("projection_mode") != spec.projection_mode:
                        raise IntegrityError(
                            f"{path}: projection_mode={metrics.get('projection_mode')} "
                            f"!= {spec.projection_mode}"
                        )

                    with np.load(required["samples"], allow_pickle=False) as sample_file:
                        missing = set(REQUIRED_SAMPLE_KEYS) - set(sample_file.files)
                        if missing:
                            raise IntegrityError(f"{path}: missing sample keys {sorted(missing)}")
                        sample = {key: np.asarray(sample_file[key]) for key in REQUIRED_SAMPLE_KEYS}
                        present_start_keys = [
                            key for key in START_INDEX_KEYS if key in sample_file.files
                        ]
                        if not present_start_keys:
                            raise IntegrityError(
                                f"{path}: missing sample key start_indices/start_index"
                            )
                        start_indices = np.asarray(sample_file[present_start_keys[0]])
                        if len(present_start_keys) == 2 and not np.array_equal(
                            start_indices, np.asarray(sample_file[present_start_keys[1]])
                        ):
                            raise IntegrityError(
                                f"{path}: start_indices and start_index disagree"
                            )
                        sample["start_indices"] = start_indices
                    for key in ("dataset_index", "start_indices", "day_id", "segment_id"):
                        if not np.issubdtype(sample[key].dtype, np.integer):
                            raise IntegrityError(f"{path}: sample key {key} must be integral")
                        sample[key] = sample[key].astype(np.int64, copy=False)
                    n = len(sample["day_id"])
                    if any(len(sample[key]) != n for key in REQUIRED_SAMPLE_KEYS):
                        raise IntegrityError(f"{path}: per-sample arrays are not aligned")
                    signature = {
                        key: array_signature(sample[key])
                        for key in ("dataset_index", "start_indices", "day_id", "segment_id")
                    }
                    if pairing_signature is None:
                        pairing_signature = signature
                        _validate_window_membership(
                            sample["start_indices"],
                            sample["day_id"],
                            sample["segment_id"],
                            test_days,
                            test_segments,
                            hist_len,
                            pred_len,
                        )
                    elif signature != pairing_signature:
                        raise IntegrityError(f"{path}: paired history metadata mismatch")

                    observed_days, counts = np.unique(sample["day_id"], return_counts=True)
                    day_counts = {
                        int(day): int(count) for day, count in zip(observed_days, counts, strict=True)
                    }
                    if tuple(sorted(day_counts)) != tuple(sorted(days)):
                        raise IntegrityError(f"{path}: selected test days {sorted(day_counts)}")
                    if expected_day_counts is None:
                        expected_day_counts = day_counts
                    elif day_counts != expected_day_counts:
                        raise IntegrityError(f"{path}: day counts {day_counts} changed")

                    raw = np.load(required["raw"], mmap_mode="r")
                    projected = np.load(required["projected"], mmap_mode="r")
                    real = np.load(required["real"], mmap_mode="r")
                    if raw.shape != projected.shape or raw.shape != real.shape:
                        raise IntegrityError(f"{path}: generated/real shapes differ")
                    if raw.ndim != 4 or raw.shape[1:3] != (2, 20) or len(raw) != n:
                        raise IntegrityError(f"{path}: unexpected sequence shape {raw.shape}")
                    if raw.shape[-1] != pred_len:
                        raise IntegrityError(
                            f"{path}: saved horizon {raw.shape[-1]} != --pred-len {pred_len}"
                        )
                    if paired_real_sha256 is None:
                        paired_real_sha256 = hashes["real"]
                    elif hashes["real"] != paired_real_sha256:
                        raise IntegrityError(f"{path}: real_future is not paired byte-for-byte")
                    generated = raw if spec.output_file == "generated_raw.npy" else projected
                    for day in days:
                        day_mask = sample["day_id"] == day
                        metric_values = _metric_row(
                            generated[day_mask],
                            real[day_mask],
                            sample["projection_l2"][day_mask],
                            normalization,
                            target_size,
                        )
                        rows.append(
                            {
                                "symbol": symbol,
                                "day_id": int(day),
                                "train_seed": int(train_seed),
                                "eval_seed": int(eval_seed),
                                "method_id": method_id,
                                "method": spec.label,
                                "deterministic": spec.deterministic,
                                "target_size_clipped_shares": float(target_size),
                                "result_dir": str(path.resolve()),
                                **metric_values,
                            }
                        )
        symbol_receipts[symbol] = {
            "dataset_dir": str(dataset_dir.resolve()),
            "test_days": list(observed_test_days),
            "selected_histories_per_day": expected_day_counts,
            "paired_history_signature": pairing_signature,
            "paired_real_sha256": paired_real_sha256,
            "full_test_snapshots": int(test_mask.sum()),
            "target_size_clipped_shares": float(target_size),
            "target_size_source": "all snapshots in split==2",
        }

    expected_cells = sum(
        len(train_seeds) * len(_method_eval_seeds(METHODS[method], eval_seeds))
        for method in method_ids
    ) * len(symbols)
    integrity = {
        "schema_version": 1,
        "status": "passed",
        "expected_input_cells": expected_cells,
        "observed_input_cells": expected_cells,
        "expected_per_cell_day_rows": expected_cells * len(days),
        "observed_per_cell_day_rows": len(rows),
        "gates": {
            "all_required_files_present_and_hashed": True,
            "dataset_arrays_aligned": True,
            "requested_days_equal_test_split_days": True,
            "windows_stay_within_day_and_segment": True,
            "paired_history_metadata_identical_within_symbol": True,
            "real_future_byte_identical_within_symbol": True,
            "method_representation_and_projection_contracts_match": True,
            "sequence_shapes_and_denominators_match": True,
            "target_size_fixed_from_full_test_split": True,
        },
        "symbols": symbol_receipts,
    }
    return rows, targets, integrity, [source_records[path] for path in sorted(source_records)]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _fmt(row: dict[str, Any], metric: str, digits: int = 4) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "NA"
    if std is None:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def render_markdown(
    day_summary: list[dict[str, Any]],
    worst_summary: list[dict[str, Any]],
    targets: dict[str, float],
) -> str:
    lines = [
        "# Public ITCH Test-Day Robustness",
        "",
        "Frozen endpoint generations only. Evaluation seeds are averaged within",
        "each checkpoint; uncertainty is sample SD across three training seeds.",
        "Persistence is deterministic and therefore has no training-seed SD.",
        "All metric arrows point down. The market-order size is frozen once per",
        "asset from every snapshot in the complete test split, not re-estimated by day.",
        "",
        "| Symbol | Test day | Method | Invalid snapshots ↓ | MSE ↓ | Projection L2 ↓ | Microprice W1 ↓ | Liq@5 W1 ↓ | Slippage W1 ↓ |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in day_summary:
        lines.append(
            f"| {row['symbol']} | {row['day_id']} | {row['method']} | "
            f"{_fmt(row, 'final_invalid_snapshot_rate', 6)} | "
            f"{_fmt(row, 'output_mse')} | "
            f"{_fmt(row, 'diagnostic_projection_l2')} | "
            f"{_fmt(row, 'microprice_offset_w1_ticks')} | "
            f"{_fmt(row, 'liquidity_5tick_w1_target_units')} | "
            f"{_fmt(row, 'market_order_slippage_w1_ticks')} |"
        )
    lines.extend(
        [
            "",
            "## Frozen target sizes",
            "",
            "| Symbol | Clipped shares |",
            "|---|---:|",
        ]
    )
    for symbol, value in sorted(targets.items()):
        lines.append(f"| {symbol} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "The two dates are fixed test conditions, not independent stochastic",
            "replicates; no day-level p-value is reported. Worst-day rows are retained",
            "in `worst_day_summary.csv`. This audit supports endpoint robustness across",
            "two held-out 2019 dates within XNAS, not unseen-asset, cross-venue, or",
            "anytime-consumer generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def _git_receipt(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def write_outputs(
    output_root: Path,
    rows: list[dict[str, Any]],
    day_summary: list[dict[str, Any]],
    worst_summary: list[dict[str, Any]],
    targets: dict[str, float],
    integrity: dict[str, Any],
    source_files: list[dict[str, Any]],
    config: dict[str, Any],
    command: str,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.tmp-", dir=output_root.parent
    ) as temp_name:
        temp = Path(temp_name)
        write_csv(temp / "per_cell_day_metrics.csv", rows)
        write_csv(temp / "hierarchical_day_summary.csv", day_summary)
        write_csv(temp / "worst_day_summary.csv", worst_summary)
        (temp / "integrity_report.json").write_text(
            json.dumps(json_safe(integrity), indent=2, allow_nan=False) + "\n"
        )
        summary_payload = {
            "schema_version": 1,
            "scope": "frozen 1 Hz endpoint generations stratified by held-out date",
            "metric_directions": METRIC_DIRECTIONS,
            "target_sizes_clipped_shares": targets,
            "hierarchical_day_summary": day_summary,
            "worst_day_summary": worst_summary,
        }
        (temp / "summary.json").write_text(
            json.dumps(json_safe(summary_payload), indent=2, allow_nan=False) + "\n"
        )
        (temp / "summary.md").write_text(
            render_markdown(day_summary, worst_summary, targets)
        )
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(temp.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "config": config,
            "metric_directions": METRIC_DIRECTIONS,
            "git": _git_receipt(root),
            "analysis_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "source_files": source_files,
            "output_sha256": output_hashes,
        }
        (temp / "run_manifest.json").write_text(
            json.dumps(json_safe(manifest), indent=2, allow_nan=False) + "\n"
        )
        temp.rename(output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/public_itch_1s_panel_full")
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/processed/snapshot_l2")
    )
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "AMZN", "GOOG"])
    parser.add_argument("--days", nargs="+", type=int, default=[20191030, 20191230])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output root: {args.output_root}"
        )
    symbols = tuple(str(value).upper() for value in args.symbols)
    days = tuple(int(value) for value in args.days)
    methods = tuple(str(value) for value in args.methods)
    train_seeds = tuple(int(value) for value in args.train_seeds)
    eval_seeds = tuple(int(value) for value in args.eval_seeds)
    rows, targets, integrity, source_files = collect_rows(
        args.results_root,
        args.data_root,
        symbols,
        days,
        methods,
        train_seeds,
        eval_seeds,
        args.hist_len,
        args.pred_len,
    )
    day_summary = hierarchical_aggregate(rows, train_seeds, eval_seeds)
    worst_summary = worst_day_aggregate(rows, train_seeds, eval_seeds)
    config = {
        "results_root": str(args.results_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "symbols": symbols,
        "days": days,
        "methods": methods,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "aggregation": "eval-seed mean within checkpoint, then mean/sample-SD across training seeds",
        "worst_day": "max for down metrics; min for up metrics",
        "target_size": "median positive best-level depth over all split==2 snapshots, once per symbol",
    }
    write_outputs(
        args.output_root,
        rows,
        day_summary,
        worst_summary,
        targets,
        integrity,
        source_files,
        config,
        shlex.join(sys.argv),
    )
    print(f"Wrote day-stratified audit to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

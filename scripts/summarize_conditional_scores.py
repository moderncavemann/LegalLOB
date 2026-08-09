#!/usr/bin/env python3
"""Compute proper conditional scores from a saved sampling ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    ("persistence", "Persistence"),
    ("raw_posthoc_pava", "Raw + post-hoc PAVA"),
    ("matched_posthoc_pava", "Matched + post-hoc PAVA"),
    ("legal_lob", "LegalLOB"),
)


def evaluated_output(path: Path) -> np.ndarray:
    metrics = json.loads((path / "metrics.json").read_text())
    projection_mode = metrics.get("projection_mode", "none")
    filename = (
        "generated_projected.npy"
        if projection_mode in {"exact", "fast", "auto"}
        else "generated_raw.npy"
    )
    return np.load(path / filename, mmap_mode="r")


def state_energy_and_crps(samples: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-history state energy score, marginal CRPS, and diversity."""
    sample = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    first_l2 = np.sqrt(np.mean(np.square(sample - truth[None]), axis=(2, 3, 4))).mean(0)
    first_l1 = np.mean(np.abs(sample - truth[None]), axis=(0, 2, 3, 4))

    pair_l2 = np.zeros(len(truth), dtype=np.float64)
    pair_l1 = np.zeros(len(truth), dtype=np.float64)
    n_members = len(sample)
    for left in range(n_members):
        for right in range(left + 1, n_members):
            difference = sample[left] - sample[right]
            pair_l2 += np.sqrt(np.mean(np.square(difference), axis=(1, 2, 3)))
            pair_l1 += np.mean(np.abs(difference), axis=(1, 2, 3))
    # E over ordered pairs includes both (i,j) and (j,i), with zero diagonal.
    pair_factor = 2.0 / max(n_members * n_members, 1)
    pair_l2 *= pair_factor
    pair_l1 *= pair_factor
    return {
        "energy_score": first_l2 - 0.5 * pair_l2,
        "marginal_crps": first_l1 - 0.5 * pair_l1,
        "pairwise_rmse": pair_l2,
    }


def trajectory_features(lob: np.ndarray) -> np.ndarray:
    """Create a compact 29-D temporal feature vector per trajectory."""
    array = np.asarray(lob, dtype=np.float64)
    ask = array[..., 0, :10, :]
    bid = array[..., 0, 10:, :]
    ask_depth = np.maximum(array[..., 1, :10, :], 0.0)
    bid_depth = np.maximum(array[..., 1, 10:, :], 0.0)
    mid = 0.5 * (ask[..., 0, :] + bid[..., 0, :])
    spread = ask[..., 0, :] - bid[..., 0, :]
    ask_total = ask_depth.sum(axis=-2)
    bid_total = bid_depth.sum(axis=-2)
    imbalance = np.divide(
        bid_total - ask_total,
        bid_total + ask_total,
        out=np.zeros_like(bid_total),
        where=(bid_total + ask_total) > 0,
    )
    return np.concatenate([np.diff(mid, axis=-1), spread, imbalance], axis=-1)


def variogram_score(
    samples: np.ndarray,
    target: np.ndarray,
    *,
    power: float = 0.5,
) -> np.ndarray:
    """Per-history variogram score on standardized temporal LOB features."""
    sample_features = trajectory_features(samples)
    target_features = trajectory_features(target)
    scale = np.std(target_features, axis=0, ddof=1)
    scale = np.where(scale > 1e-6, scale, 1.0)
    sample_features = sample_features / scale
    target_features = target_features / scale
    n_features = target_features.shape[-1]
    result = np.zeros(len(target_features), dtype=np.float64)
    pair_count = 0
    for left in range(n_features):
        for right in range(left + 1, n_features):
            observed = np.abs(
                target_features[:, left] - target_features[:, right]
            ) ** power
            forecast = np.mean(
                np.abs(
                    sample_features[..., left] - sample_features[..., right]
                ) ** power,
                axis=0,
            )
            result += np.square(observed - forecast)
            pair_count += 1
    return result / max(pair_count, 1)


def interval_statistics(samples: np.ndarray, target: np.ndarray) -> dict[str, float]:
    sample = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    result: dict[str, float] = {}
    for nominal, lower_q, upper_q in ((0.5, 0.25, 0.75), (0.9, 0.05, 0.95)):
        lower = np.quantile(sample, lower_q, axis=0)
        upper = np.quantile(sample, upper_q, axis=0)
        coverage = float(np.mean((truth >= lower) & (truth <= upper)))
        key = int(100 * nominal)
        result[f"coverage_{key}"] = coverage
        result[f"coverage_error_{key}"] = abs(coverage - nominal)
        result[f"interval_width_{key}"] = float(np.mean(upper - lower))
    return result


def summarize_method(samples: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores = state_energy_and_crps(samples, target)
    variogram = variogram_score(samples, target)
    result = {
        name: float(np.mean(values)) for name, values in scores.items()
    }
    result["variogram_score"] = float(np.mean(variogram))
    if len(samples) > 1:
        result.update(interval_statistics(samples, target))
    else:
        for key in (50, 90):
            result[f"coverage_{key}"] = float("nan")
            result[f"coverage_error_{key}"] = float("nan")
            result[f"interval_width_{key}"] = 0.0
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method_id, label in METHODS:
        group = [row for row in rows if row["method_id"] == method_id]
        if not group:
            continue
        item: dict[str, Any] = {
            "method_id": method_id,
            "method": label,
            "n_training_seeds": len(group),
            "n_sampling_members": group[0]["n_sampling_members"],
            "n_histories": group[0]["n_histories"],
        }
        fields = [
            key
            for key, value in group[0].items()
            if key not in {"method_id", "method", "train_seed", "n_sampling_members", "n_histories"}
            and isinstance(value, (int, float))
        ]
        for field in fields:
            values = np.asarray([row[field] for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                item[f"{field}_mean"] = float(finite.mean())
                item[f"{field}_std"] = (
                    float(finite.std(ddof=1)) if finite.size > 1 else 0.0
                )
        output.append(item)
    return output


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", default="results/conditional_scores_goog_event_32x512"
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sampling-seeds", type=int, default=32)
    parser.add_argument("--output-dir", default="paper_data")
    parser.add_argument(
        "--table-path", default="paper_tables/table_conditional_scores.md"
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = []
    for train_seed in args.train_seeds:
        first = root / f"train_seed{train_seed}" / "sampling_seed0" / "legal_lob"
        target = np.asarray(np.load(first / "real_future.npy", mmap_mode="r"))
        persistence = np.asarray(np.load(first / "persistence.npy", mmap_mode="r"))[None]
        for method_id, label in METHODS:
            if method_id == "persistence":
                samples = persistence
            else:
                members = []
                reference_indices = None
                for sampling_seed in range(args.sampling_seeds):
                    path = (
                        root
                        / f"train_seed{train_seed}"
                        / f"sampling_seed{sampling_seed}"
                        / method_id
                    )
                    indices = np.load(path / "per_sample_metrics.npz")["dataset_index"]
                    if reference_indices is None:
                        reference_indices = indices
                    elif not np.array_equal(reference_indices, indices):
                        raise ValueError(f"History mismatch: {path}")
                    candidate_target = np.load(path / "real_future.npy", mmap_mode="r")
                    if not np.array_equal(candidate_target, target):
                        raise ValueError(f"Real future mismatch: {path}")
                    members.append(np.asarray(evaluated_output(path)))
                samples = np.stack(members)
            row = {
                "method_id": method_id,
                "method": label,
                "train_seed": train_seed,
                "n_sampling_members": len(samples),
                "n_histories": len(target),
            }
            row.update(summarize_method(samples, target))
            rows.append(row)

    summary = aggregate(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "conditional_scores_per_train_seed.csv", rows)
    _write_csv(output_dir / "conditional_scores_summary.csv", summary)
    (output_dir / "conditional_scores_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )

    def pm(row: dict[str, Any], field: str, digits: int = 4) -> str:
        mean = row.get(f"{field}_mean")
        std = row.get(f"{field}_std")
        if mean is None:
            return "--"
        return f"{mean:.{digits}f} +/- {(std or 0.0):.{digits}f}"

    lines = [
        "# Conditional Generative Scores: GOOG-Event",
        "",
        "Representative fixed-checkpoint panel with 512 common histories, 32",
        "sampling trajectories per learned method, and three independently",
        "trained checkpoints. Mean +/- sample SD is across training seeds.",
        "Energy score and marginal CRPS use the normalized 10-frame state;",
        "variogram score uses standardized mid-return, spread, and depth-imbalance",
        "features. Persistence is deterministic, so interval coverage is not defined.",
        "",
        "| Method | Energy score | Marginal CRPS | Variogram score | 50% coverage | 90% coverage | Pairwise RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {energy} | {crps} | {variogram} | {cov50} | {cov90} | {pair} |".format(
                method=row["method"],
                energy=pm(row, "energy_score"),
                crps=pm(row, "marginal_crps"),
                variogram=pm(row, "variogram_score"),
                cov50=pm(row, "coverage_50", 3),
                cov90=pm(row, "coverage_90", 3),
                pair=pm(row, "pairwise_rmse"),
            )
        )
    table_path = Path(args.table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} per-training-seed rows and {len(summary)} aggregates")
    print(f"Table: {table_path}")


if __name__ == "__main__":
    main()

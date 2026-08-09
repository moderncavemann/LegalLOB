#!/usr/bin/env python3
"""Paired moving-block bootstrap over chronological held-out histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def moving_block_means(
    diff: np.ndarray,
    block_size: int,
    n_boot: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    if block_size <= 0 or block_size > len(diff):
        raise ValueError("block_size must be in [1, number of paired histories]")
    rng = np.random.default_rng(seed)
    groups = np.zeros(len(diff), dtype=np.int64) if groups is None else np.asarray(groups)
    if len(groups) != len(diff):
        raise ValueError("groups must align with diff")
    group_positions = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pieces = []
        for positions in group_positions:
            values = diff[positions]
            local_block = min(block_size, len(values))
            starts = np.arange(0, len(values) - local_block + 1)
            n_blocks = int(np.ceil(len(values) / local_block))
            chosen = rng.choice(starts, size=n_blocks, replace=True)
            pieces.append(
                np.concatenate([values[s:s + local_block] for s in chosen])[:len(values)]
            )
        sample = np.concatenate(pieces)
        boot[i] = sample.mean()
    return boot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-a", required=True, help="Directory containing per_sample_metrics.npz")
    parser.add_argument("--method-b", required=True, help="Directory containing per_sample_metrics.npz")
    parser.add_argument(
        "--metrics",
        default=(
            "mse_raw,any_invalid_snapshot_rate,crossed_book_rate,"
            "ask_ladder_violation,bid_ladder_violation,projection_l2"
        ),
    )
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    path_a = Path(args.method_a) / "per_sample_metrics.npz"
    path_b = Path(args.method_b) / "per_sample_metrics.npz"
    a = np.load(path_a)
    b = np.load(path_b)
    if "dataset_index" not in a or "dataset_index" not in b:
        raise ValueError("Both evaluations must save dataset_index for paired bootstrap")
    if not np.array_equal(a["dataset_index"], b["dataset_index"]):
        raise ValueError("Evaluation histories are not paired in the same chronological order")
    group_key = "segment_id" if "segment_id" in a and "segment_id" in b else (
        "day_id" if "day_id" in a and "day_id" in b else None
    )
    if group_key and not np.array_equal(a[group_key], b[group_key]):
        raise ValueError(f"Evaluation {group_key} values are not paired")
    groups_all = np.asarray(a[group_key]) if group_key else np.zeros(len(a["dataset_index"]))

    rows = []
    for metric in [item.strip() for item in args.metrics.split(",") if item.strip()]:
        if metric not in a or metric not in b:
            rows.append({"metric": metric, "error": "missing from one or both inputs"})
            continue
        values_a = np.asarray(a[metric], dtype=np.float64)
        values_b = np.asarray(b[metric], dtype=np.float64)
        finite = np.isfinite(values_a) & np.isfinite(values_b)
        diff = values_b[finite] - values_a[finite]
        groups = groups_all[finite]
        boot = moving_block_means(diff, args.block_size, args.n_boot, args.seed, groups)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({
            "metric": metric,
            "n_paired_histories": int(len(diff)),
            "mean_a": float(values_a[finite].mean()),
            "mean_b": float(values_b[finite].mean()),
            "mean_diff_b_minus_a": float(diff.mean()),
            "block_bootstrap_ci95": [float(low), float(high)],
            "probability_diff_le_zero": float(np.mean(boot <= 0)),
            "probability_diff_ge_zero": float(np.mean(boot >= 0)),
        })
    payload = {
        "method_a": str(args.method_a),
        "method_b": str(args.method_b),
        "block_size_in_selected_histories": args.block_size,
        "block_grouping": group_key or "single_contiguous_fallback",
        "n_groups": int(len(np.unique(groups_all))),
        "n_boot": args.n_boot,
        "interpretation": "Data-uncertainty CI over chronological held-out blocks; not a substitute for training-seed variation.",
        "comparisons": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

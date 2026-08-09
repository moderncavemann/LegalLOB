#!/usr/bin/env python3
"""Audit whether exact PAVA changes model-based stress-scenario selection.

The audit is offline and uses paired frozen candidates. It does not simulate a
trading agent and must not be interpreted as causal action-response evidence.
For every seed history, candidates are ranked before and after PAVA by two
snapshot-level stress scores: maximum fixed-size market-order slippage and
minimum five-tick liquidity over the generated horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarize_financial_sensitivity import (
    _liquidity_within_ticks,
    _market_order_slippage,
    _to_economic_units,
    functional_vectors,
)


METHODS = ("raw_posthoc_pava", "matched_posthoc_pava", "legal_lob")
RISK_NAMES = ("max_market_order_slippage", "negative_min_liquidity_5tick")


def _numeric_suffix(path: Path, prefix: str) -> int:
    return int(path.name.removeprefix(prefix))


def discover_sampling_dirs(train_dir: Path) -> list[Path]:
    paths = [path for path in train_dir.glob("sampling_seed*") if path.is_dir()]
    return sorted(paths, key=lambda path: _numeric_suffix(path, "sampling_seed"))


def trajectory_risks(
    lob: np.ndarray,
    normalization: dict[str, Any],
    target_size: float,
) -> dict[str, np.ndarray]:
    """Return one stress score per trajectory for [N, 2, 20, T] input."""
    n_trajectories, _, _, horizon = lob.shape
    prices, volumes, _ = _to_economic_units(lob, normalization)
    ask_price = prices[:, :10]
    bid_price = prices[:, 10:]
    ask_volume = volumes[:, :10]
    bid_volume = volumes[:, 10:]

    buy_slippage, buy_full = _market_order_slippage(
        prices, volumes, target_size, buy=True
    )
    sell_slippage, sell_full = _market_order_slippage(
        prices, volumes, target_size, buy=False
    )
    buy_slippage = buy_slippage.reshape(n_trajectories, horizon)
    sell_slippage = sell_slippage.reshape(n_trajectories, horizon)
    buy_full = buy_full.reshape(n_trajectories, horizon)
    sell_full = sell_full.reshape(n_trajectories, horizon)

    finite_slippage = np.concatenate([buy_slippage, sell_slippage], axis=1)
    with np.errstate(all="ignore"):
        max_slippage = np.nanmax(finite_slippage, axis=1)
    any_incomplete = ~(buy_full & sell_full).all(axis=1)
    # An order that cannot be filled within top-10 depth is more stressed than
    # every finite-slippage candidate. The separate incomplete rate makes this
    # ordering convention explicit in the output.
    max_slippage[any_incomplete] = np.inf

    liquidity = _liquidity_within_ticks(
        ask_price,
        bid_price,
        ask_volume,
        bid_volume,
        ticks=5.0,
        target_size=target_size,
    ).reshape(n_trajectories, horizon)
    min_liquidity = np.min(liquidity, axis=1)

    return {
        "max_market_order_slippage": max_slippage,
        # Higher scores always mean more stressed in the ranking code.
        "negative_min_liquidity_5tick": -min_liquidity,
    }


def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Small-vector Kendall tau-b with ties and infinities."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    concordant = discordant = tie_x = tie_y = 0
    for left in range(len(x) - 1):
        x_right = x[left + 1 :]
        y_right = y[left + 1 :]
        dx = np.where(x[left] > x_right, 1, np.where(x[left] < x_right, -1, 0))
        dy = np.where(y[left] > y_right, 1, np.where(y[left] < y_right, -1, 0))
        concordant += int(np.sum((dx == dy) & (dx != 0)))
        discordant += int(np.sum((dx == -dy) & (dx != 0) & (dy != 0)))
        tie_x += int(np.sum((dx == 0) & (dy != 0)))
        tie_y += int(np.sum((dy == 0) & (dx != 0)))
    denominator = math.sqrt(
        (concordant + discordant + tie_x)
        * (concordant + discordant + tie_y)
    )
    if denominator == 0:
        return 1.0 if np.array_equal(x, y) else float("nan")
    return (concordant - discordant) / denominator


def _descending_order(scores: np.ndarray) -> np.ndarray:
    candidate_ids = np.arange(len(scores))
    return np.lexsort((candidate_ids, -scores))


def _finite_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        f"{prefix}_mean": float(np.mean(finite)) if finite.size else float("nan"),
        f"{prefix}_median": float(np.median(finite)) if finite.size else float("nan"),
        f"{prefix}_p05": float(np.quantile(finite, 0.05)) if finite.size else float("nan"),
        f"{prefix}_p95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        f"{prefix}_finite_rate": float(finite.size / max(values.size, 1)),
    }


def _safe_difference(left: float, right: float) -> float:
    if not (math.isfinite(left) and math.isfinite(right)):
        return 0.0 if left == right else float("nan")
    return float(left - right)


def compare_rankings(
    pre: np.ndarray,
    post: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    """Compare [N histories, C candidates] risk matrices."""
    if pre.shape != post.shape or pre.ndim != 2:
        raise ValueError(f"Expected matching [N, C] matrices, got {pre.shape}, {post.shape}")
    if not 1 <= top_k <= pre.shape[1]:
        raise ValueError(f"top_k must be in [1, {pre.shape[1]}]")

    taus: list[float] = []
    jaccards: list[float] = []
    flips: list[bool] = []
    changed_top_k: list[bool] = []
    final_risk_gain: list[float] = []
    selected_repair_shift: list[float] = []
    pre_max_ties: list[int] = []
    post_max_ties: list[int] = []

    for pre_scores, post_scores in zip(pre, post, strict=True):
        pre_order = _descending_order(pre_scores)
        post_order = _descending_order(post_scores)
        pre_top = set(pre_order[:top_k].tolist())
        post_top = set(post_order[:top_k].tolist())
        pre_selected = int(pre_order[0])
        post_selected = int(post_order[0])

        taus.append(_kendall_tau_b(pre_scores, post_scores))
        jaccards.append(len(pre_top & post_top) / len(pre_top | post_top))
        flips.append(pre_selected != post_selected)
        changed_top_k.append(pre_top != post_top)
        final_risk_gain.append(
            _safe_difference(post_scores[post_selected], post_scores[pre_selected])
        )
        selected_repair_shift.append(
            _safe_difference(post_scores[pre_selected], pre_scores[pre_selected])
        )
        pre_max_ties.append(int(np.sum(pre_scores == pre_scores[pre_selected])))
        post_max_ties.append(int(np.sum(post_scores == post_scores[post_selected])))

    tau_array = np.asarray(taus)
    jaccard_array = np.asarray(jaccards)
    result = {
        "histories": int(pre.shape[0]),
        "candidates_per_history": int(pre.shape[1]),
        "top_k": int(top_k),
        "worst_candidate_flip_rate": float(np.mean(flips)),
        "top_k_membership_change_rate": float(np.mean(changed_top_k)),
        "top_k_jaccard_mean": float(np.mean(jaccard_array)),
        "top_k_jaccard_median": float(np.median(jaccard_array)),
        "pre_infinite_candidate_rate": float(np.mean(np.isinf(pre))),
        "post_infinite_candidate_rate": float(np.mean(np.isinf(post))),
        "pre_worst_tie_count_mean": float(np.mean(pre_max_ties)),
        "post_worst_tie_count_mean": float(np.mean(post_max_ties)),
    }
    result.update(_finite_summary(tau_array, "kendall_tau_b"))
    result.update(_finite_summary(np.asarray(final_risk_gain), "selected_final_risk_gain"))
    result.update(
        _finite_summary(np.asarray(selected_repair_shift), "selected_repair_shift")
    )
    return result


def load_method_risks(
    sampling_dirs: list[Path],
    method: str,
    normalization: dict[str, Any],
    target_size: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pre_by_risk: dict[str, list[np.ndarray]] = {name: [] for name in RISK_NAMES}
    post_by_risk: dict[str, list[np.ndarray]] = {name: [] for name in RISK_NAMES}
    expected_shape: tuple[int, ...] | None = None
    for sampling_dir in sampling_dirs:
        method_dir = sampling_dir / method
        raw = np.load(method_dir / "generated_raw.npy", mmap_mode="r")
        projected = np.load(method_dir / "generated_projected.npy", mmap_mode="r")
        if raw.shape != projected.shape:
            raise ValueError(f"Paired shape mismatch in {method_dir}: {raw.shape} vs {projected.shape}")
        if expected_shape is None:
            expected_shape = raw.shape
        elif raw.shape != expected_shape:
            raise ValueError(f"Candidate shape mismatch in {method_dir}: {raw.shape} vs {expected_shape}")
        pre_risks = trajectory_risks(raw, normalization, target_size)
        post_risks = trajectory_risks(projected, normalization, target_size)
        for name in RISK_NAMES:
            pre_by_risk[name].append(pre_risks[name])
            post_by_risk[name].append(post_risks[name])

    # Candidate files are outermost; transpose to [history, candidate].
    return (
        {name: np.stack(values, axis=1) for name, values in pre_by_risk.items()},
        {name: np.stack(values, axis=1) for name, values in post_by_risk.items()},
    )


def audit_train_seed(
    results_root: Path,
    train_seed: int,
    normalization: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    train_dir = results_root / f"train_seed{train_seed}"
    sampling_dirs = discover_sampling_dirs(train_dir)
    if not sampling_dirs:
        raise FileNotFoundError(f"No sampling candidates found under {train_dir}")

    reference_real = np.load(
        sampling_dirs[0] / "raw_posthoc_pava" / "real_future.npy", mmap_mode="r"
    )
    for sampling_dir in sampling_dirs[1:]:
        candidate_real = np.load(
            sampling_dir / "raw_posthoc_pava" / "real_future.npy", mmap_mode="r"
        )
        if not np.array_equal(reference_real, candidate_real):
            raise ValueError(
                f"Candidate histories are not paired: {sampling_dir / 'raw_posthoc_pava'}"
            )
    _, target_size = functional_vectors(reference_real, normalization)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        pre_by_risk, post_by_risk = load_method_risks(
            sampling_dirs, method, normalization, target_size
        )
        for risk_name in RISK_NAMES:
            comparison = compare_rankings(
                pre_by_risk[risk_name], post_by_risk[risk_name], top_k=top_k
            )
            rows.append(
                {
                    "train_seed": train_seed,
                    "method": method,
                    "risk": risk_name,
                    "market_order_target_size_clipped_shares": target_size,
                    **comparison,
                }
            )
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "kendall_tau_b_mean",
        "top_k_jaccard_mean",
        "worst_candidate_flip_rate",
        "top_k_membership_change_rate",
        "selected_final_risk_gain_mean",
        "selected_repair_shift_mean",
    )
    aggregates: list[dict[str, Any]] = []
    for method in METHODS:
        for risk_name in RISK_NAMES:
            group = [
                row for row in rows if row["method"] == method and row["risk"] == risk_name
            ]
            aggregate: dict[str, Any] = {
                "method": method,
                "risk": risk_name,
                "train_seed_count": len(group),
            }
            for metric in metrics:
                values = np.asarray([row[metric] for row in group], dtype=np.float64)
                finite = values[np.isfinite(values)]
                aggregate[f"{metric}_across_seed_mean"] = (
                    float(np.mean(finite)) if finite.size else float("nan")
                )
                aggregate[f"{metric}_across_seed_sd"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                )
            aggregates.append(aggregate)
    return aggregates


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (output_dir / "scenario_selection_per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    go_rows = [
        row
        for row in rows
        if row["method"] != "legal_lob"
        and (
            row["worst_candidate_flip_rate"] > args.go_flip_rate
            or row["top_k_jaccard_mean"] < args.go_jaccard
        )
    ]
    aggregates = aggregate_rows(rows)
    report = {
        "audit_type": "offline_model_based_stress_selection",
        "causal_interpretation": False,
        "results_root": str(args.results_root),
        "normalization": str(args.normalization),
        "train_seeds": args.train_seeds,
        "top_k": args.top_k,
        "go_no_go_rule": {
            "worst_candidate_flip_rate_gt": args.go_flip_rate,
            "top_k_jaccard_mean_lt": args.go_jaccard,
        },
        "recursive_followup_warranted": bool(go_rows),
        "triggered_rows": [
            {"train_seed": row["train_seed"], "method": row["method"], "risk": row["risk"]}
            for row in go_rows
        ],
        "rows": rows,
        "aggregates": aggregates,
    }
    (output_dir / "scenario_selection_summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Offline Model-Based Stress Selection Audit",
        "",
        "This is a paired scenario-ranking diagnostic, not causal action-response evidence.",
        "",
        "| Train seed | Method | Risk | Kendall tau-b | Top-k Jaccard | Worst flip | Top-k change |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['train_seed']} | {row['method']} | {row['risk']} | "
            f"{row['kendall_tau_b_mean']:.4f} | {row['top_k_jaccard_mean']:.4f} | "
            f"{row['worst_candidate_flip_rate']:.4f} | "
            f"{row['top_k_membership_change_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Across-Training-Seed Summary",
            "",
            "| Method | Risk | Kendall tau-b | Top-k Jaccard | Worst flip | Top-k change |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['method']} | {row['risk']} | "
            f"{row['kendall_tau_b_mean_across_seed_mean']:.4f} "
            f"+/- {row['kendall_tau_b_mean_across_seed_sd']:.4f} | "
            f"{row['top_k_jaccard_mean_across_seed_mean']:.4f} "
            f"+/- {row['top_k_jaccard_mean_across_seed_sd']:.4f} | "
            f"{row['worst_candidate_flip_rate_across_seed_mean']:.4f} "
            f"+/- {row['worst_candidate_flip_rate_across_seed_sd']:.4f} | "
            f"{row['top_k_membership_change_rate_across_seed_mean']:.4f} "
            f"+/- {row['top_k_membership_change_rate_across_seed_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Recursive follow-up warranted by the predeclared rule: **{bool(go_rows)}**.",
        ]
    )
    (output_dir / "scenario_selection_summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/conditional_scores_goog_event_32x512"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path(
            "data/processed/snapshot_l2/"
            "GOOG_databento_20190130_mbp10_rth_v2/normalization.json"
        ),
    )
    parser.add_argument("--train-seeds", type=parse_seed_list, default=[0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--go-flip-rate", type=float, default=0.05)
    parser.add_argument("--go-jaccard", type=float, default=0.90)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/scenario_selection_goog_event_32x512"),
    )
    args = parser.parse_args()

    normalization = json.loads(args.normalization.read_text())
    rows: list[dict[str, Any]] = []
    for train_seed in args.train_seeds:
        rows.extend(
            audit_train_seed(
                args.results_root,
                train_seed,
                normalization,
                top_k=args.top_k,
            )
        )
    write_outputs(rows, args.output_dir, args)
    print((args.output_dir / "scenario_selection_summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

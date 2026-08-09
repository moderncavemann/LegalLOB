#!/usr/bin/env python3
"""Aggregate projection ablation metrics across sampling seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_projection_ablation import _row_from_metrics, _row_from_snowball


AGGREGATE_COLUMNS = [
    "mse_raw",
    "mse_projected",
    "mae_raw",
    "mae_projected",
    "price_corr_raw",
    "price_corr_projected",
    "volume_corr_raw",
    "volume_corr_projected",
    "temporal_corr_raw",
    "temporal_corr_projected",
    "raw_crossed_book_rate",
    "raw_ask_ladder_violation",
    "raw_bid_ladder_violation",
    "raw_negative_volume_rate",
    "projected_crossed_book_rate",
    "projected_ask_ladder_violation",
    "projected_bid_ladder_violation",
    "projected_negative_volume_rate",
    "projection_distance",
    "price_projection_distance",
    "volume_projection_distance",
    "real_spread_mean",
    "raw_spread_mean",
    "projected_spread_mean",
    "spread_shift_raw",
    "spread_shift_projected",
    "sampler_per_step_projection_l2_mean",
    "sampler_cumulative_projection_l2_mean",
    "sampler_pre_projection_crossed_mean",
    "sampler_pre_projection_ask_ladder_mean",
    "sampler_pre_projection_bid_ladder_mean",
    "w1_mid_return",
    "w1_spread",
    "w1_volume",
    "w1_depth_imbalance",
    "w1_queue_imbalance",
    "ks_mid_return",
    "ks_spread",
    "ks_volume",
    "js_mid_return",
    "js_spread",
    "js_volume",
    "mmd2_mid_return",
    "volume_difference_correlation_mae",
    "horizon_mse_final",
    "persistence_mse_final",
    "persistence_skill_mean",
    "persistence_skill_final",
    "off_tick_spread_rate",
    "off_tick_gap_rate",
    "noninteger_volume_rate",
    "any_invalid_snapshot_rate",
    "any_invalid_trajectory_rate",
    "max_spread_shortfall",
    "max_ask_gap_shortfall",
    "max_bid_gap_shortfall",
    "max_negative_volume_magnitude",
    "repair_any_snapshot_modified_rate",
    "repair_price_level_modified_rate",
    "repair_mean_ticks_moved_modified_levels",
    "acf_mid_return_lag1_real",
    "acf_mid_return_lag1_generated",
    "acf_abs_return_lag1_real",
    "acf_abs_return_lag1_generated",
    "acf_spread_lag1_real",
    "acf_spread_lag1_generated",
]


DISPLAY_COLUMNS = [
    "method",
    "n_seeds",
    "representation",
    "projection_mode",
    "sampler_projection_mode",
    "mse_raw",
    "mse_projected",
    "mae_raw",
    "mae_projected",
    "raw_crossed_book_rate",
    "raw_ask_ladder_violation",
    "raw_bid_ladder_violation",
    "raw_negative_volume_rate",
    "projected_crossed_book_rate",
    "projection_distance",
    "sampler_cumulative_projection_l2_mean",
    "sampler_pre_projection_crossed_mean",
    "real_spread_mean",
    "raw_spread_mean",
    "projected_spread_mean",
    "price_corr_raw",
    "price_corr_projected",
    "volume_corr_raw",
    "volume_corr_projected",
    "persistence_skill_mean",
    "ks_mid_return",
    "ks_spread",
    "off_tick_gap_rate",
    "noninteger_volume_rate",
    "any_invalid_snapshot_rate",
    "repair_price_level_modified_rate",
    "repair_mean_ticks_moved_modified_levels",
]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mu = _mean(values)
    assert mu is not None
    return (sum((x - mu) ** 2 for x in values) / len(values)) ** 0.5


def _seed_from_path(root: Path, path: Path) -> int | None:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "seeds" and parts[1].startswith("seed"):
        return int(parts[1].replace("seed", ""))
    if len(parts) == 2:
        return 0
    return None


def _collect_rows(root: Path, include_smoke: bool) -> list[dict[str, Any]]:
    rows = []
    metric_paths = list(root.glob("*/metrics.json")) + list(root.glob("seeds/seed*/*/metrics.json"))
    snowball_paths = list(root.glob("*/snowball_stats.json")) + list(root.glob("seeds/seed*/*/snowball_stats.json"))
    for path in sorted(metric_paths):
        if not include_smoke and "smoke" in path.parent.name:
            continue
        seed = _seed_from_path(root, path)
        if seed is None:
            continue
        row = _row_from_metrics(path)
        row["seed"] = seed
        rows.append(row)
    for path in sorted(snowball_paths):
        if not include_smoke and "smoke" in path.parent.name:
            continue
        seed = _seed_from_path(root, path)
        if seed is None:
            continue
        row = _row_from_snowball(path)
        row["seed"] = seed
        rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    aggregated = []
    for method, method_rows in sorted(by_method.items()):
        method_rows = sorted(method_rows, key=lambda row: row["seed"])
        first = method_rows[0]
        out: dict[str, Any] = {
            "method": method,
            "seeds": ",".join(str(row["seed"]) for row in method_rows),
            "n_seeds": len(method_rows),
            "representation": first.get("representation"),
            "projection_mode": first.get("projection_mode"),
            "sampler_projection_mode": first.get("sampler_projection_mode"),
        }
        for col in AGGREGATE_COLUMNS:
            values = [float(row[col]) for row in method_rows if isinstance(row.get(col), (int, float))]
            out[f"{col}_mean"] = _mean(values)
            out[f"{col}_std"] = _std(values)
        aggregated.append(out)
    return aggregated


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _format_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(DISPLAY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(DISPLAY_COLUMNS)) + " |",
    ]
    for row in rows:
        values = []
        for col in DISPLAY_COLUMNS:
            if col in {"method", "n_seeds", "representation", "projection_mode", "sampler_projection_mode"}:
                values.append(_format_value(row.get(col)))
                continue
            mean = row.get(f"{col}_mean")
            std = row.get(f"{col}_std")
            if mean is None:
                values.append("")
            elif std is None:
                values.append(_format_value(mean))
            else:
                values.append(f"{_format_value(mean)} +/- {_format_value(std)}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/ablation_projection")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_dir)
    rows = _collect_rows(root, include_smoke=args.include_smoke)
    if not rows:
        raise SystemExit(f"No seeded ablation metrics found under {root}")

    aggregated = _aggregate(rows)
    output_prefix = Path(args.output_prefix) if args.output_prefix else root / "seed_summary"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(aggregated, indent=2) + "\n")
    fieldnames = list(aggregated[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)
    md_path.write_text(_format_md(aggregated))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

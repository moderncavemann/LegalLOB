#!/usr/bin/env python3
"""Summarize projection ablation outputs into JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _method_label(path: Path) -> str:
    return path.parent.name


def _row_from_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    raw_viol = data.get("violations_raw", {})
    proj_viol = data.get("violations_proj", {})
    proj_dist = data.get("projection_distance", {})
    sampler_stats = data.get("sampler_projection_stats") or {}
    projection_mode = data.get("projection_mode", "unknown")
    use_projected = projection_mode in {"exact", "fast", "auto"}
    metric_output = "projected" if use_projected else "raw"
    facts = data.get(f"stylized_facts_{metric_output}") or {}
    distributions = facts.get("distribution") or {}
    acf = facts.get("acf") or {}
    horizon = data.get(f"horizon_fidelity_{metric_output}") or {}
    discrete = data.get(f"discrete_validity_{metric_output}") or {}
    structural = data.get(f"structural_validity_{metric_output}") or {}
    repair = data.get("diagnostic_projection_repair") or {}
    generated_spread = (
        data.get("spread_mean_proj")
        if projection_mode in {"exact", "fast", "auto"}
        else data.get("spread_mean_raw")
    )
    spread_distortion = None
    if data.get("spread_mean_proj") is not None and data.get("spread_mean_raw") is not None:
        spread_distortion = float(data["spread_mean_proj"]) - float(data["spread_mean_raw"])

    return {
        "method": _method_label(path),
        "source": str(path),
        "kind": "single_step",
        "metric_output": metric_output,
        "representation": data.get("representation"),
        "projection_mode": projection_mode,
        "sampler_projection_mode": data.get("sampler_projection_mode", "none"),
        "mse_raw": _float(data.get("mse_raw")),
        "mse_projected": _float(data.get("mse_projected", data.get("mse_proj"))),
        "mae_raw": _float(data.get("mae_raw")),
        "mae_projected": _float(data.get("mae_projected", data.get("mae_proj"))),
        "price_corr_raw": _float(data.get("corr_price_raw")),
        "price_corr_projected": _float(data.get("corr_price_projected", data.get("corr_price"))),
        "volume_corr_raw": _float(data.get("corr_volume_raw")),
        "volume_corr_projected": _float(data.get("corr_volume_projected", data.get("corr_volume"))),
        "temporal_corr_raw": _float(data.get("temporal_correlation_raw")),
        "temporal_corr_projected": _float(data.get("temporal_correlation_projected", data.get("temporal_correlation"))),
        "raw_crossed_book_rate": _float(raw_viol.get("crossed_book_rate")),
        "raw_ask_ladder_violation": _float(raw_viol.get("ask_ladder_violation")),
        "raw_bid_ladder_violation": _float(raw_viol.get("bid_ladder_violation")),
        "raw_negative_volume_rate": _float(raw_viol.get("negative_volume_rate")),
        "projected_crossed_book_rate": _float(proj_viol.get("crossed_book_rate")),
        "projected_ask_ladder_violation": _float(proj_viol.get("ask_ladder_violation")),
        "projected_bid_ladder_violation": _float(proj_viol.get("bid_ladder_violation")),
        "projected_negative_volume_rate": _float(proj_viol.get("negative_volume_rate")),
        "projection_distance": _float(proj_dist.get("total_l2_distance")),
        "price_projection_distance": _float(proj_dist.get("price_l2_distance")),
        "volume_projection_distance": _float(proj_dist.get("volume_l2_distance")),
        "real_spread_mean": _float(data.get("spread_mean_real")),
        "generated_spread_mean": _float(generated_spread),
        "raw_spread_mean": _float(data.get("spread_mean_raw")),
        "projected_spread_mean": _float(data.get("spread_mean_projected", data.get("spread_mean_proj"))),
        "spread_shift_raw": _float(data.get("spread_shift_raw")),
        "spread_shift_projected": _float(data.get("spread_shift_projected")),
        "spread_distortion_proj_minus_raw": spread_distortion,
        "sampler_projection_calls": _float(sampler_stats.get("n_projection_calls")),
        "sampler_per_step_projection_l2_mean": _float(sampler_stats.get("per_step_projection_l2_mean")),
        "sampler_per_step_projection_l2_max": _float(sampler_stats.get("per_step_projection_l2_max")),
        "sampler_cumulative_projection_l2_mean": _float(sampler_stats.get("cumulative_projection_l2_mean")),
        "sampler_cumulative_projection_l2_max": _float(sampler_stats.get("cumulative_projection_l2_max")),
        "sampler_pre_projection_crossed_mean": _float(sampler_stats.get("pre_projection_crossed_book_rate_mean")),
        "sampler_pre_projection_ask_ladder_mean": _float(sampler_stats.get("pre_projection_ask_ladder_violation_mean")),
        "sampler_pre_projection_bid_ladder_mean": _float(sampler_stats.get("pre_projection_bid_ladder_violation_mean")),
        "w1_mid_return": _float((distributions.get("mid_return") or {}).get("wasserstein_1")),
        "w1_spread": _float((distributions.get("spread") or {}).get("wasserstein_1")),
        "w1_volume": _float((distributions.get("total_volume") or {}).get("wasserstein_1")),
        "w1_depth_imbalance": _float((distributions.get("depth_imbalance") or {}).get("wasserstein_1")),
        "w1_queue_imbalance": _float((distributions.get("queue_imbalance") or {}).get("wasserstein_1")),
        "ks_mid_return": _float((distributions.get("mid_return") or {}).get("ks")),
        "ks_spread": _float((distributions.get("spread") or {}).get("ks")),
        "ks_volume": _float((distributions.get("total_volume") or {}).get("ks")),
        "js_mid_return": _float((distributions.get("mid_return") or {}).get("js")),
        "js_spread": _float((distributions.get("spread") or {}).get("js")),
        "js_volume": _float((distributions.get("total_volume") or {}).get("js")),
        "mmd2_mid_return": _float((distributions.get("mid_return") or {}).get("mmd2_rbf_biased")),
        "volume_difference_correlation_mae": _float(
            (facts.get("volume_difference_cross_level_correlation") or {}).get("mean_absolute_error")
        ),
        "horizon_mse_final": _float((horizon.get("mse") or [None])[-1]),
        "persistence_mse_final": _float((horizon.get("persistence_mse") or [None])[-1]),
        "persistence_skill_mean": (
            sum(horizon["persistence_skill_score"]) / len(horizon["persistence_skill_score"])
            if horizon.get("persistence_skill_score")
            else None
        ),
        "persistence_skill_final": _float(
            (horizon.get("persistence_skill_score") or [None])[-1]
        ),
        "discrete_validity_available": discrete.get("available"),
        "off_tick_spread_rate": _float(discrete.get("off_tick_spread_rate")),
        "off_tick_gap_rate": _float(discrete.get("off_tick_gap_rate")),
        "noninteger_volume_rate": _float(discrete.get("noninteger_volume_rate")),
        "any_invalid_snapshot_rate": _float(structural.get("any_invalid_snapshot_rate")),
        "any_invalid_trajectory_rate": _float(structural.get("any_invalid_trajectory_rate")),
        "max_spread_shortfall": _float(structural.get("max_spread_shortfall")),
        "max_ask_gap_shortfall": _float(structural.get("max_ask_gap_shortfall")),
        "max_bid_gap_shortfall": _float(structural.get("max_bid_gap_shortfall")),
        "max_negative_volume_magnitude": _float(
            structural.get("max_negative_volume_magnitude")
        ),
        "repair_any_snapshot_modified_rate": _float(repair.get("any_snapshot_modified_rate")),
        "repair_price_level_modified_rate": _float(repair.get("price_level_modified_rate")),
        "repair_mean_ticks_moved_modified_levels": _float(
            repair.get("mean_ticks_moved_modified_levels")
        ),
        "acf_mid_return_lag1_real": _float(((acf.get("mid_return") or {}).get("1") or {}).get("real")),
        "acf_mid_return_lag1_generated": _float(((acf.get("mid_return") or {}).get("1") or {}).get("generated")),
        "acf_abs_return_lag1_real": _float(((acf.get("abs_mid_return") or {}).get("1") or {}).get("real")),
        "acf_abs_return_lag1_generated": _float(((acf.get("abs_mid_return") or {}).get("1") or {}).get("generated")),
        "acf_spread_lag1_real": _float(((acf.get("spread") or {}).get("1") or {}).get("real")),
        "acf_spread_lag1_generated": _float(((acf.get("spread") or {}).get("1") or {}).get("generated")),
        "notes": "",
    }


def _row_from_snowball(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    generated = data.get("generated", {})
    key = "step_100" if "step_100" in generated else sorted(generated)[-1]
    step = generated[key]
    viol = step.get("violations_raw", {})
    proj_dist = step.get("projection_distance") or {}
    return {
        "method": _method_label(path),
        "source": str(path),
        "kind": key,
        "metric_output": "raw",
        "representation": data.get("config", {}).get("representation"),
        "projection_mode": data.get("config", {}).get("projection_mode"),
        "sampler_projection_mode": "none",
        "mse_raw": None,
        "mse_projected": None,
        "mae_raw": None,
        "mae_projected": None,
        "price_corr_raw": _float(step.get("price_corr")),
        "price_corr_projected": None,
        "volume_corr_raw": _float(step.get("volume_corr")),
        "volume_corr_projected": None,
        "temporal_corr_raw": _float(step.get("temporal_corr")),
        "temporal_corr_projected": None,
        "raw_crossed_book_rate": _float(viol.get("crossed_book_rate")),
        "raw_ask_ladder_violation": _float(viol.get("ask_ladder_violation")),
        "raw_bid_ladder_violation": _float(viol.get("bid_ladder_violation")),
        "raw_negative_volume_rate": _float(viol.get("negative_volume_rate")),
        "projected_crossed_book_rate": None,
        "projected_ask_ladder_violation": None,
        "projected_bid_ladder_violation": None,
        "projected_negative_volume_rate": None,
        "projection_distance": _float(proj_dist.get("total_l2_distance")),
        "price_projection_distance": _float(proj_dist.get("price_l2_distance")),
        "volume_projection_distance": _float(proj_dist.get("volume_l2_distance")),
        "real_spread_mean": _float(step.get("spread_real_mean")),
        "generated_spread_mean": _float(step.get("spread_mean")),
        "raw_spread_mean": _float(step.get("spread_mean")),
        "projected_spread_mean": None,
        "spread_shift_raw": (
            _float(step.get("spread_mean")) - _float(step.get("spread_real_mean"))
            if _float(step.get("spread_mean")) is not None and _float(step.get("spread_real_mean")) is not None
            else None
        ),
        "spread_shift_projected": None,
        "spread_distortion_proj_minus_raw": None,
        "sampler_projection_calls": None,
        "sampler_per_step_projection_l2_mean": None,
        "sampler_per_step_projection_l2_max": None,
        "sampler_cumulative_projection_l2_mean": None,
        "sampler_cumulative_projection_l2_max": None,
        "sampler_pre_projection_crossed_mean": None,
        "sampler_pre_projection_ask_ladder_mean": None,
        "sampler_pre_projection_bid_ladder_mean": None,
        "w1_spread": _float(step.get("w1_spread")),
        "w1_volume": _float(step.get("w1_volume")),
        "w1_mid_return": _float(step.get("w1_mid_return")),
        "w1_depth_imbalance": _float(step.get("w1_order_imbalance")),
        "w1_queue_imbalance": _float(step.get("w1_queue_imbalance")),
        "mmd2_mid_return": _float(step.get("mmd_mid_return")),
        "ks_mid_return": None,
        "ks_spread": None,
        "ks_volume": None,
        "js_mid_return": None,
        "js_spread": None,
        "js_volume": None,
        "volume_difference_correlation_mae": None,
        "horizon_mse_final": None,
        "persistence_mse_final": None,
        "persistence_skill_mean": None,
        "persistence_skill_final": None,
        "discrete_validity_available": None,
        "off_tick_spread_rate": None,
        "off_tick_gap_rate": None,
        "noninteger_volume_rate": None,
        "any_invalid_snapshot_rate": None,
        "any_invalid_trajectory_rate": None,
        "max_spread_shortfall": None,
        "max_ask_gap_shortfall": None,
        "max_bid_gap_shortfall": None,
        "max_negative_volume_magnitude": None,
        "repair_any_snapshot_modified_rate": None,
        "repair_price_level_modified_rate": None,
        "repair_mean_ticks_moved_modified_levels": None,
        "acf_mid_return_lag1_real": _float((step.get("acf_mid_return_real") or {}).get("lag_1")),
        "acf_mid_return_lag1_generated": _float((step.get("acf_mid_return_generated") or {}).get("lag_1")),
        "acf_abs_return_lag1_real": _float((step.get("acf_abs_mid_return_real") or {}).get("lag_1")),
        "acf_abs_return_lag1_generated": _float((step.get("acf_abs_mid_return_generated") or {}).get("lag_1")),
        "acf_spread_lag1_real": _float((step.get("acf_spread_real") or {}).get("lag_1")),
        "acf_spread_lag1_generated": _float((step.get("acf_spread_generated") or {}).get("lag_1")),
        "notes": "rollout summary",
    }


def _format_md(rows: list[dict[str, Any]]) -> str:
    columns = [
        "method",
        "kind",
        "metric_output",
        "representation",
        "projection_mode",
        "sampler_projection_mode",
        "mse_raw",
        "mse_projected",
        "mae_raw",
        "mae_projected",
        "price_corr_raw",
        "price_corr_projected",
        "volume_corr_raw",
        "volume_corr_projected",
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
        "spread_shift_raw",
        "spread_shift_projected",
        "w1_spread",
        "w1_volume",
        "w1_mid_return",
        "w1_depth_imbalance",
        "w1_queue_imbalance",
        "ks_mid_return",
        "ks_spread",
        "js_mid_return",
        "js_spread",
        "horizon_mse_final",
        "persistence_mse_final",
        "persistence_skill_mean",
        "persistence_skill_final",
        "off_tick_spread_rate",
        "off_tick_gap_rate",
        "noninteger_volume_rate",
        "any_invalid_snapshot_rate",
        "any_invalid_trajectory_rate",
        "repair_price_level_modified_rate",
        "repair_mean_ticks_moved_modified_levels",
        "acf_mid_return_lag1_real",
        "acf_mid_return_lag1_generated",
        "acf_abs_return_lag1_real",
        "acf_abs_return_lag1_generated",
        "acf_spread_lag1_real",
        "acf_spread_lag1_generated",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/ablation_projection")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--include-smoke", action="store_true",
                        help="Include smoke-test outputs in the summary table.")
    args = parser.parse_args()

    root = Path(args.results_dir)
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        if not args.include_smoke and "smoke" in path.parent.name:
            continue
        rows.append(_row_from_metrics(path))
    for path in sorted(root.glob("*/snowball_stats.json")):
        if not args.include_smoke and "smoke" in path.parent.name:
            continue
        rows.append(_row_from_snowball(path))

    if not rows:
        raise SystemExit(f"No ablation outputs found under {root}")

    output_prefix = Path(args.output_prefix) if args.output_prefix else root / "summary"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(_format_md(rows))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate the bounded all-method recursive rollout panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHOD_LABELS = {
    "raw": "Raw",
    "raw_posthoc_pava": "Raw + post-hoc PAVA",
    "raw_inloop_pava": "Raw + in-loop PAVA",
    "matched": "Matched",
    "matched_posthoc_pava": "Matched + post-hoc PAVA",
    "legal_lob": "LegalLOB",
}
HORIZONS = (10, 20, 50, 100)


def _relative_error(generated: Any, real: Any) -> float | None:
    if generated is None or real is None:
        return None
    generated = float(generated)
    real = float(real)
    if not math.isfinite(generated) or not math.isfinite(real) or abs(real) < 1e-12:
        return None
    return abs(generated - real) / abs(real)


def collect_rows(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    eval_seed = int(manifest.get("eval_seed", 0))
    rows = []
    for seed_dir in sorted(root.glob("train_seed*")):
        train_seed = int(seed_dir.name.removeprefix("train_seed"))
        for method_id, label in METHOD_LABELS.items():
            stats_path = seed_dir / method_id / "snowball_stats.json"
            if not stats_path.exists():
                raise FileNotFoundError(stats_path)
            stats = json.loads(stats_path.read_text())
            runtime = stats.get("runtime") or {}
            config = stats.get("config") or {}
            trajectories_per_second = runtime.get("trajectories_per_second")
            rollout_horizon = config.get("horizon")
            lob_steps_per_second = (
                float(trajectories_per_second) * float(rollout_horizon)
                if trajectories_per_second is not None and rollout_horizon is not None
                else None
            )
            projection = stats.get("rollout_projection_stats") or {}
            for horizon in HORIZONS:
                step = (stats.get("generated") or {}).get(f"step_{horizon}")
                if step is None:
                    continue
                horizon_projection = (
                    (projection.get("by_horizon") or {}).get(f"step_{horizon}")
                    or {}
                )
                sampler = (
                    horizon_projection.get("sampler_projection")
                    or projection.get("sampler_projection")
                    or {}
                )
                posthoc = (
                    horizon_projection.get("posthoc_chunk_projection")
                    or projection.get("posthoc_chunk_projection")
                    or {}
                )
                repair = sampler or posthoc
                violations = step.get("violations_raw") or {}
                distance = step.get("projection_distance") or {}
                rows.append(
                    {
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "method_id": method_id,
                        "method": label,
                        "horizon": horizon,
                        "n_histories": config.get("n_histories"),
                        "final_invalid_snapshot_rate": violations.get(
                            "any_invalid_snapshot_rate"
                        ),
                        "final_crossed_book_rate": violations.get("crossed_book_rate"),
                        "final_ask_ladder_violation": violations.get(
                            "ask_ladder_violation"
                        ),
                        "final_bid_ladder_violation": violations.get(
                            "bid_ladder_violation"
                        ),
                        "final_negative_volume_rate": violations.get(
                            "negative_volume_rate"
                        ),
                        "diagnostic_projection_l2": distance.get("total_l2_distance"),
                        "spread_relative_abs_error": _relative_error(
                            step.get("spread_mean"), step.get("spread_real_mean")
                        ),
                        "volume_relative_abs_error": _relative_error(
                            step.get("volume_mean"), step.get("volume_real_mean")
                        ),
                        "return_std_relative_abs_error": _relative_error(
                            step.get("mid_return_std"),
                            step.get("mid_return_real_std"),
                        ),
                        "temporal_correlation": step.get("temporal_corr"),
                        "return_w1": step.get("w1_mid_return"),
                        "spread_w1": step.get("w1_spread"),
                        "volume_w1": step.get("w1_volume"),
                        "path_pre_projection_invalid_rate": repair.get(
                            "pre_projection_any_invalid_snapshot_rate_mean"
                        ),
                        "path_cumulative_projection_l2": repair.get(
                            "cumulative_projection_l2_mean"
                        ),
                        "path_projection_calls": repair.get(
                            "n_projection_calls_per_trajectory"
                        ),
                        "rollout_seconds": runtime.get("rollout_seconds"),
                        "trajectories_per_second": runtime.get(
                            "trajectories_per_second"
                        ),
                        "lob_steps_per_second": lob_steps_per_second,
                        "stats_path": str(stats_path),
                    }
                )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["method_id"], row["method"], row["horizon"]) for row in rows})
    excluded = {
        "train_seed",
        "eval_seed",
        "method_id",
        "method",
        "horizon",
        "stats_path",
    }
    fields = sorted({key for row in rows for key in row} - excluded)
    output = []
    for method_id, method, horizon in keys:
        group = [
            row for row in rows
            if row["method_id"] == method_id and row["horizon"] == horizon
        ]
        train_seed_groups: dict[int, list[dict[str, Any]]] = {}
        for row in group:
            train_seed_groups.setdefault(int(row["train_seed"]), []).append(row)
        item = {
            "method_id": method_id,
            "method": method,
            "horizon": horizon,
            "n_training_seeds": len(train_seed_groups),
            "n_eval_seeds_per_training_seed": sorted(
                {len(seed_rows) for seed_rows in train_seed_groups.values()}
            ),
        }
        for field in fields:
            values = []
            for seed_rows in train_seed_groups.values():
                eval_values = [
                    float(row[field])
                    for row in seed_rows
                    if row.get(field) is not None
                    and math.isfinite(float(row[field]))
                ]
                if eval_values:
                    values.append(float(np.mean(eval_values)))
            if values:
                item[f"{field}_mean"] = float(np.mean(values))
                item[f"{field}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
        output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(row: dict[str, Any], field: str, digits: int = 3) -> str:
    mean = row.get(f"{field}_mean")
    if mean is None:
        return "--"
    std = row.get(f"{field}_std", 0.0)
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _fmt_scaled(
    row: dict[str, Any], field: str, scale: float, digits: int = 2
) -> str:
    mean = row.get(f"{field}_mean")
    if mean is None:
        return "--"
    std = row.get(f"{field}_std", 0.0)
    return f"{scale * mean:.{digits}f} +/- {scale * std:.{digits}f}"


def markdown(summary: list[dict[str, Any]]) -> str:
    rows_by_method = {
        row["method_id"]: row for row in summary if row["horizon"] == 100
    }
    eval_seed_counts = sorted(
        {
            int(count)
            for row in summary
            for count in row.get("n_eval_seeds_per_training_seed", [])
        }
    )
    eval_seed_text = (
        str(eval_seed_counts[0])
        if len(eval_seed_counts) == 1
        else "/".join(str(value) for value in eval_seed_counts)
    )
    lines = [
        "# GOOG-Event All-Method 100-Step Rollout",
        "",
        f"Bounded representative panel: 640 histories, {eval_seed_text} paired sampling seed(s),",
        "and three independently trained checkpoints per method. Sampling seeds",
        "are averaged within checkpoints; mean +/- sample SD is across training seeds.",
        "This panel tests recursive mechanism behavior;",
        "it is not a calibrated probabilistic forecast comparison.",
        "",
        "One LOB step is one generated future Level-2 snapshot; throughput therefore",
        "counts 100 steps per completed trajectory rather than DDIM reverse updates.",
        "",
        "| Method | Final invalid % | Pre-repair/path invalid % | Cumulative repair L2 | Spread drift | Volume drift | Return-std drift | Temporal corr | LOB steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in METHOD_LABELS:
        row = rows_by_method[method_id]
        lines.append(
            "| {method} | {final} | {pre} | {repair} | {spread} | {volume} | {ret} | {corr} | {speed} |".format(
                method=row["method"],
                final=_fmt_scaled(row, "final_invalid_snapshot_rate", 100.0),
                pre=_fmt_scaled(
                    row, "path_pre_projection_invalid_rate", 100.0, digits=1
                ),
                repair=_fmt(row, "path_cumulative_projection_l2", 2),
                spread=_fmt(row, "spread_relative_abs_error", 3),
                volume=_fmt(row, "volume_relative_abs_error", 3),
                ret=_fmt(row, "return_std_relative_abs_error", 3),
                corr=_fmt(row, "temporal_correlation", 3),
                speed=_fmt(row, "lob_steps_per_second", 1),
            )
        )
    horizon_rows = {
        (row["method_id"], row["horizon"]): row for row in summary
    }
    lines.extend(
        [
            "",
            "## Horizon-wise projection burden",
            "",
            "Pre/path invalidity is measured immediately before each projection call; cumulative L2 sums repair magnitudes through the reported horizon.",
            "",
            "| Horizon | Raw + post pre % | Raw + post cum. L2 | Matched + post pre % | Matched + post cum. L2 | Raw + in-loop pre % | Raw + in-loop cum. L2 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in HORIZONS:
        raw_post = horizon_rows[("raw_posthoc_pava", horizon)]
        matched_post = horizon_rows[("matched_posthoc_pava", horizon)]
        raw_inloop = horizon_rows[("raw_inloop_pava", horizon)]
        lines.append(
            "| {horizon} | {raw_pre} | {raw_l2} | {matched_pre} | {matched_l2} | {inloop_pre} | {inloop_l2} |".format(
                horizon=horizon,
                raw_pre=_fmt_scaled(
                    raw_post, "path_pre_projection_invalid_rate", 100.0, digits=1
                ),
                raw_l2=_fmt(raw_post, "path_cumulative_projection_l2", 2),
                matched_pre=_fmt_scaled(
                    matched_post,
                    "path_pre_projection_invalid_rate",
                    100.0,
                    digits=1,
                ),
                matched_l2=_fmt(
                    matched_post, "path_cumulative_projection_l2", 2
                ),
                inloop_pre=_fmt_scaled(
                    raw_inloop,
                    "path_pre_projection_invalid_rate",
                    100.0,
                    digits=1,
                ),
                inloop_l2=_fmt(raw_inloop, "path_cumulative_projection_l2", 2),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-roots",
        nargs="+",
        default=["results/all_method_rollout_goog_event_640"],
    )
    parser.add_argument(
        "--output-prefix", default="paper_data/all_method_rollout_goog_event_640"
    )
    parser.add_argument(
        "--table-path", default="paper_tables/table_all_method_rollout.md"
    )
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for root in args.results_roots:
        rows.extend(collect_rows(Path(root)))
    summary = aggregate(rows)
    _write_csv(prefix.with_name(prefix.name + "_per_run.csv"), rows)
    _write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary)
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    table_path.write_text(markdown(summary))
    print(f"Wrote {len(rows)} rows and {len(summary)} aggregates")
    print(f"Table: {table_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build paper-facing feasibility, repair, and temporal summaries.

The script is deliberately read-only with respect to experiment outputs. It
uses saved evaluation arrays and JSON files, first averages fixed-checkpoint
sampling seeds within each training seed, and then reports mean and sample
standard deviation across the three independent training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.eval.metrics import wasserstein_1d
from scripts.audit_saved_generation import repair_distribution


TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    result_root: Path
    normalization: Path


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    output: str


DATASETS = (
    DatasetSpec(
        "GOOG_DATABENTO_RTH_V2",
        "GOOG Databento RTH v2",
        Path("results/goog_databento_rth_v2_full"),
        Path("data/processed/snapshot_l2/GOOG_databento_20190130_mbp10_rth_v2/normalization.json"),
    ),
    DatasetSpec(
        "AAPL_PUBLIC_ITCH_1S",
        "AAPL public ITCH 1 Hz",
        Path("results/public_itch_1s_panel_full/AAPL"),
        Path("data/processed/snapshot_l2/PUBLIC_ITCH_2019_SAMPLES_AAPL_1s/normalization.json"),
    ),
    DatasetSpec(
        "AMZN_PUBLIC_ITCH_1S",
        "AMZN public ITCH 1 Hz",
        Path("results/public_itch_1s_panel_full/AMZN"),
        Path("data/processed/snapshot_l2/PUBLIC_ITCH_2019_SAMPLES_AMZN_1s/normalization.json"),
    ),
    DatasetSpec(
        "GOOG_PUBLIC_ITCH_1S",
        "GOOG public ITCH 1 Hz",
        Path("results/public_itch_1s_panel_full/GOOG"),
        Path("data/processed/snapshot_l2/PUBLIC_ITCH_2019_SAMPLES_GOOG_1s/normalization.json"),
    ),
)


METHODS = (
    MethodSpec("persistence", "Persistence", "raw"),
    MethodSpec("raw_no_projection", "Raw", "raw"),
    MethodSpec("raw_physics_no_projection", "Raw + soft penalty", "raw"),
    MethodSpec("raw_posthoc_pava", "Raw + post-hoc PAVA", "projected"),
    MethodSpec("raw_inloop_pava", "Raw + in-loop PAVA", "raw"),
    MethodSpec("matched_no_projection", "Matched", "raw"),
    MethodSpec("matched_posthoc_pava", "Matched + post-hoc PAVA", "projected"),
    MethodSpec("legal_no_projection", "LegalLOB", "raw"),
)


MAIN_METHOD_IDS = (
    "persistence",
    "raw_no_projection",
    "matched_no_projection",
    "raw_posthoc_pava",
    "matched_posthoc_pava",
    "legal_no_projection",
)


SCALAR_FIELDS = (
    "mse_raw",
    "mse_projected",
    "mse_output",
    "invalid_raw",
    "invalid_output",
    "return_w1",
    "abs_return_acf_iae",
    "spread_acf_iae",
    "volume_difference_correlation_mae",
    "horizon10_mse",
    "horizon10_persistence_skill",
    "repair_snapshot_modified_rate",
    "repair_price_level_modified_rate",
    "repair_volume_level_modified_rate",
    "repair_depth_invalid_snapshot_rate",
    "repair_depth_level_clamped_rate",
    "repair_depth_norm_p50",
    "repair_depth_norm_p95",
    "repair_depth_norm_max",
    "repair_depth_share_equiv_p50",
    "repair_depth_share_equiv_p95",
    "repair_depth_share_equiv_max",
    "repair_ticks_p50",
    "repair_ticks_p95",
    "repair_ticks_max",
    "repair_snapshot_gt_0_1_ticks",
    "repair_snapshot_gt_0_5_ticks",
    "repair_snapshot_gt_1_tick",
    "repair_price_level_gt_0_1_ticks",
    "repair_price_level_gt_0_5_ticks",
    "repair_price_level_gt_1_tick",
    "repair_return_w1_raw_real",
    "repair_return_w1_projected_real",
    "repair_return_w1_raw_projected",
    "repair_spread_w1_raw_real",
    "repair_spread_w1_projected_real",
    "repair_spread_w1_raw_projected",
    "repair_gap_w1_raw_real",
    "repair_gap_w1_projected_real",
    "repair_gap_w1_raw_projected",
    "inloop_cumulative_projection_l2",
    "inloop_pre_projection_cbr",
)


SNOWBALL_FIELDS = (
    "invalid_output",
    "projection_distance",
    "return_w1",
    "spread_w1",
    "volume_w1",
    "temporal_correlation",
    "spread_relative_abs_error",
    "volume_relative_abs_error",
    "return_std_relative_abs_error",
)


TEMPORAL_FIELDS = (
    "return_w1",
    "abs_return_acf_iae",
    "spread_acf_iae",
    "volume_difference_correlation_mae",
    "horizon10_mse",
    "horizon10_persistence_skill",
)


def result_dir(root: Path, train_seed: int, eval_seed: int, method_id: str) -> Path:
    base = root / f"train_seed{train_seed}"
    if eval_seed:
        base = base / "seeds" / f"seed{eval_seed}"
    return base / method_id


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def acf_integrated_absolute_error(facts: dict[str, Any], series: str) -> float | None:
    """Sum absolute ACF error over all finite lags saved by the evaluator."""
    entries = ((facts.get("acf") or {}).get(series) or {})
    errors = []
    for lag in sorted(entries, key=lambda item: int(item)):
        values = entries[lag] or {}
        real = _as_float(values.get("real"))
        generated = _as_float(values.get("generated"))
        if real is not None and generated is not None:
            errors.append(abs(real - generated))
    return float(sum(errors)) if errors else None


def _selected_facts(metrics: dict[str, Any], output: str) -> dict[str, Any]:
    return metrics.get(f"stylized_facts_{output}") or {}


def _selected_horizon(metrics: dict[str, Any], output: str) -> dict[str, Any]:
    return metrics.get(f"horizon_fidelity_{output}") or {}


def _selected_validity(metrics: dict[str, Any], output: str) -> dict[str, Any]:
    return metrics.get(f"structural_validity_{output}") or {}


def _last(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    return _as_float(values[-1])


def _distribution_features(lob: np.ndarray) -> dict[str, np.ndarray]:
    prices = np.asarray(lob[:, 0], dtype=np.float64)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    mid = 0.5 * (ask[:, 0] + bid[:, 0])
    spread = ask[:, 0] - bid[:, 0]
    ask_gaps = ask[:, 1:] - ask[:, :-1]
    bid_gaps = bid[:, :-1] - bid[:, 1:]
    return {
        "return": np.diff(mid, axis=-1).reshape(-1),
        "spread": spread.reshape(-1),
        "gap": np.concatenate([ask_gaps.reshape(-1), bid_gaps.reshape(-1)]),
    }


def depth_repair_severity(
    raw: np.ndarray,
    projected: np.ndarray,
    *,
    tolerance: float,
    volume_scale: float,
) -> dict[str, float]:
    """Measure clamp incidence and magnitude independently of price PAVA."""
    raw_volume = np.asarray(raw[:, 1], dtype=np.float64)
    projected_volume = np.asarray(projected[:, 1], dtype=np.float64)
    correction = np.abs(projected_volume - raw_volume)
    modified = correction > tolerance
    values = correction[modified]
    # Negative normalized sqrt-depth has no physical inverse. Squaring the
    # correction magnitude after restoring the training scale gives a
    # transparent share-equivalent severity for the clamp-to-zero operation.
    share_equiv = np.square(values * volume_scale)

    def quantile(array: np.ndarray, q: float) -> float:
        return float(np.quantile(array, q)) if array.size else 0.0

    return {
        "repair_depth_invalid_snapshot_rate": float(
            np.mean(np.any(raw_volume < -tolerance, axis=1))
        ),
        "repair_depth_level_clamped_rate": float(np.mean(modified)),
        "repair_depth_norm_p50": quantile(values, 0.50),
        "repair_depth_norm_p95": quantile(values, 0.95),
        "repair_depth_norm_max": float(values.max()) if values.size else 0.0,
        "repair_depth_share_equiv_p50": quantile(share_equiv, 0.50),
        "repair_depth_share_equiv_p95": quantile(share_equiv, 0.95),
        "repair_depth_share_equiv_max": (
            float(share_equiv.max()) if share_equiv.size else 0.0
        ),
    }


def _repair_metrics(
    path: Path,
    metrics: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, float]:
    raw = np.load(path / "generated_raw.npy", mmap_mode="r")
    projected = np.load(path / "generated_projected.npy", mmap_mode="r")
    real = np.load(path / "real_future.npy", mmap_mode="r")
    tolerance = float(
        (metrics.get("output_projection_repair") or {}).get(
            "change_tolerance",
            max(1e-8, float(normalization["min_level_gap_normalized"]) * 1e-4),
        )
    )
    repair = repair_distribution(raw, projected, normalization, tolerance)
    moved = repair["ticks_moved_modified_price_levels"]
    depth_repair = depth_repair_severity(
        raw,
        projected,
        tolerance=tolerance,
        volume_scale=float(normalization.get("volume_scale", 1.0)),
    )
    raw_features = _distribution_features(raw)
    projected_features = _distribution_features(projected)
    real_features = _distribution_features(real)

    result = {
        "repair_snapshot_modified_rate": float(repair["any_snapshot_modified_rate"]),
        "repair_price_level_modified_rate": float(repair["price_level_modified_rate"]),
        "repair_volume_level_modified_rate": float(repair["volume_level_modified_rate"]),
        **depth_repair,
        "repair_ticks_p50": float(moved["p50"]) if moved else 0.0,
        "repair_ticks_p95": float(moved["p95"]) if moved else 0.0,
        "repair_ticks_max": float(moved["max"]) if moved else 0.0,
        "repair_snapshot_gt_0_1_ticks": float(
            repair["tick_threshold_exceedance"]["0.1"]["snapshot_rate"]
        ),
        "repair_snapshot_gt_0_5_ticks": float(
            repair["tick_threshold_exceedance"]["0.5"]["snapshot_rate"]
        ),
        "repair_snapshot_gt_1_tick": float(
            repair["tick_threshold_exceedance"]["1"]["snapshot_rate"]
        ),
        "repair_price_level_gt_0_1_ticks": float(
            repair["tick_threshold_exceedance"]["0.1"]["price_level_rate"]
        ),
        "repair_price_level_gt_0_5_ticks": float(
            repair["tick_threshold_exceedance"]["0.5"]["price_level_rate"]
        ),
        "repair_price_level_gt_1_tick": float(
            repair["tick_threshold_exceedance"]["1"]["price_level_rate"]
        ),
    }
    for name in ("return", "spread", "gap"):
        result[f"repair_{name}_w1_raw_real"] = wasserstein_1d(
            raw_features[name], real_features[name]
        )
        result[f"repair_{name}_w1_projected_real"] = wasserstein_1d(
            projected_features[name], real_features[name]
        )
        result[f"repair_{name}_w1_raw_projected"] = wasserstein_1d(
            raw_features[name], projected_features[name]
        )
    return result


def collect_evaluation_rows(
    datasets: Iterable[DatasetSpec] = DATASETS,
    methods: Iterable[MethodSpec] = METHODS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        normalization = json.loads(dataset.normalization.read_text())
        for train_seed in TRAIN_SEEDS:
            for method in methods:
                method_eval_seeds = (0,) if method.method_id == "persistence" else EVAL_SEEDS
                for eval_seed in method_eval_seeds:
                    path = result_dir(dataset.result_root, train_seed, eval_seed, method.method_id)
                    metrics_path = path / "metrics.json"
                    if not metrics_path.exists():
                        raise FileNotFoundError(f"Missing frozen result: {metrics_path}")
                    metrics = json.loads(metrics_path.read_text())
                    facts = _selected_facts(metrics, method.output)
                    distributions = facts.get("distribution") or {}
                    horizon = _selected_horizon(metrics, method.output)
                    raw_validity = _selected_validity(metrics, "raw")
                    output_validity = _selected_validity(metrics, method.output)
                    row: dict[str, Any] = {
                        "dataset_id": dataset.dataset_id,
                        "dataset": dataset.label,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "method_id": method.method_id,
                        "method": method.label,
                        "output": method.output,
                        "result_dir": str(path),
                        "mse_raw": _as_float(metrics.get("mse_raw")),
                        "mse_projected": _as_float(metrics.get("mse_projected")),
                        "mse_output": _as_float(metrics.get(f"mse_{method.output}")),
                        "invalid_raw": _as_float(raw_validity.get("any_invalid_snapshot_rate")),
                        "invalid_output": _as_float(
                            output_validity.get("any_invalid_snapshot_rate")
                        ),
                        "return_w1": _as_float(
                            ((distributions.get("mid_return") or {}).get("wasserstein_1"))
                        ),
                        "abs_return_acf_iae": acf_integrated_absolute_error(
                            facts, "abs_mid_return"
                        ),
                        "spread_acf_iae": acf_integrated_absolute_error(facts, "spread"),
                        "volume_difference_correlation_mae": _as_float(
                            (
                                facts.get("volume_difference_cross_level_correlation") or {}
                            ).get("mean_absolute_error")
                        ),
                        "horizon10_mse": _last(horizon.get("mse")),
                        "horizon10_persistence_skill": _last(
                            horizon.get("persistence_skill_score")
                        ),
                        "repair_snapshot_modified_rate": None,
                        "repair_price_level_modified_rate": None,
                        "repair_volume_level_modified_rate": None,
                        "repair_depth_invalid_snapshot_rate": None,
                        "repair_depth_level_clamped_rate": None,
                        "repair_depth_norm_p50": None,
                        "repair_depth_norm_p95": None,
                        "repair_depth_norm_max": None,
                        "repair_depth_share_equiv_p50": None,
                        "repair_depth_share_equiv_p95": None,
                        "repair_depth_share_equiv_max": None,
                        "repair_ticks_p50": None,
                        "repair_ticks_p95": None,
                        "repair_ticks_max": None,
                        "repair_snapshot_gt_0_1_ticks": None,
                        "repair_snapshot_gt_0_5_ticks": None,
                        "repair_snapshot_gt_1_tick": None,
                        "repair_price_level_gt_0_1_ticks": None,
                        "repair_price_level_gt_0_5_ticks": None,
                        "repair_price_level_gt_1_tick": None,
                        "repair_return_w1_raw_real": None,
                        "repair_return_w1_projected_real": None,
                        "repair_return_w1_raw_projected": None,
                        "repair_spread_w1_raw_real": None,
                        "repair_spread_w1_projected_real": None,
                        "repair_spread_w1_raw_projected": None,
                        "repair_gap_w1_raw_real": None,
                        "repair_gap_w1_projected_real": None,
                        "repair_gap_w1_raw_projected": None,
                        "inloop_cumulative_projection_l2": _as_float(
                            (metrics.get("sampler_projection_stats") or {}).get(
                                "cumulative_projection_l2_mean"
                            )
                        ),
                        "inloop_pre_projection_cbr": _as_float(
                            (metrics.get("sampler_projection_stats") or {}).get(
                                "pre_projection_crossed_book_rate_mean"
                            )
                        ),
                    }
                    if method.method_id.endswith("_posthoc_pava"):
                        row.update(_repair_metrics(path, metrics, normalization))
                    rows.append(row)
    return rows


def hierarchical_aggregate(
    rows: list[dict[str, Any]],
    fields: Iterable[str],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Average eval seeds first, then summarize independent training seeds."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    output = []
    for key, group in sorted(groups.items()):
        train_seed_means: dict[int, dict[str, float]] = {}
        for train_seed in TRAIN_SEEDS:
            seed_rows = [row for row in group if row["train_seed"] == train_seed]
            if not seed_rows:
                continue
            seed_values: dict[str, float] = {}
            for field in fields:
                values = [
                    float(row[field])
                    for row in seed_rows
                    if row.get(field) is not None and math.isfinite(float(row[field]))
                ]
                if values:
                    seed_values[field] = float(np.mean(values))
            train_seed_means[train_seed] = seed_values

        item: dict[str, Any] = dict(zip(group_fields, key))
        item["n_training_seeds"] = len(train_seed_means)
        item["n_eval_seeds_per_training_seed"] = sorted(
            {len([row for row in group if row["train_seed"] == seed]) for seed in train_seed_means}
        )
        item["training_seed_means"] = train_seed_means
        for field in fields:
            values = [values[field] for values in train_seed_means.values() if field in values]
            if values:
                item[f"{field}_mean"] = float(np.mean(values))
                item[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                item[f"{field}_min"] = float(np.min(values))
                item[f"{field}_max"] = float(np.max(values))
                run_values = [
                    float(row[field])
                    for row in group
                    if row.get(field) is not None and math.isfinite(float(row[field]))
                ]
                item[f"{field}_global_min"] = float(np.min(run_values))
                item[f"{field}_global_max"] = float(np.max(run_values))
        output.append(item)
    return output


def collect_snowball_rows(datasets: Iterable[DatasetSpec] = DATASETS) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for train_seed in TRAIN_SEEDS:
            for eval_seed in EVAL_SEEDS:
                path = result_dir(
                    dataset.result_root,
                    train_seed,
                    eval_seed,
                    "legal_snowball_100_replay",
                )
                stats_path = path / "snowball_stats.json"
                if not stats_path.exists():
                    raise FileNotFoundError(f"Missing frozen rollout result: {stats_path}")
                stats = json.loads(stats_path.read_text())
                step = (stats.get("generated") or {}).get("step_100") or {}
                violations = step.get("violations_raw") or {}
                invalid = max(
                    float(violations.get(name, 0.0))
                    for name in (
                        "crossed_book_rate",
                        "ask_ladder_violation",
                        "bid_ladder_violation",
                        "negative_volume_rate",
                    )
                )
                real_spread = float(step["spread_real_mean"])
                real_volume = float(step["volume_real_mean"])
                real_return_std = float(step["mid_return_real_std"])
                rows.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "dataset": dataset.label,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "method_id": "legal_no_projection",
                        "method": "LegalLOB",
                        "horizon": 100,
                        "result_dir": str(path),
                        "invalid_output": invalid,
                        "projection_distance": _as_float(
                            (step.get("projection_distance") or {}).get("total_l2_distance")
                        ),
                        "return_w1": _as_float(step.get("w1_mid_return")),
                        "spread_w1": _as_float(step.get("w1_spread")),
                        "volume_w1": _as_float(step.get("w1_volume")),
                        "temporal_correlation": _as_float(step.get("temporal_corr")),
                        "spread_relative_abs_error": abs(
                            float(step["spread_mean"]) - real_spread
                        )
                        / max(abs(real_spread), 1e-12),
                        "volume_relative_abs_error": abs(
                            float(step["volume_mean"]) - real_volume
                        )
                        / max(abs(real_volume), 1e-12),
                        "return_std_relative_abs_error": abs(
                            float(step["mid_return_std"]) - real_return_std
                        )
                        / max(abs(real_return_std), 1e-12),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _method_order(method_id: str) -> int:
    return MAIN_METHOD_IDS.index(method_id) if method_id in MAIN_METHOD_IDS else 99


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if value == 0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _fmt_pm(row: dict[str, Any], field: str, digits: int = 4) -> str:
    mean = row.get(f"{field}_mean")
    std = row.get(f"{field}_std")
    if mean is None:
        return "NA"
    return f"{_fmt(float(mean), digits)} +/- {_fmt(float(std), digits)}"


def write_feasibility_table(path: Path, summary: list[dict[str, Any]]) -> None:
    selected = [row for row in summary if row["method_id"] in MAIN_METHOD_IDS]
    selected.sort(key=lambda row: (row["dataset_id"], _method_order(row["method_id"])))
    lines = [
        "# Cross-Asset Feasibility and Repair",
        "",
        "All values first average fixed-checkpoint sampling seeds `0/1/2` within each training seed, then report mean +/- sample SD across training seeds `0/1/2`. Output MSE and final invalidity refer to the method output actually evaluated; post-hoc rows therefore use projected output.",
        "",
        "| Dataset | Method | Output MSE | Raw invalid snapshot rate | Final invalid snapshot rate | Snapshots modified | Price levels modified |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["method"],
                    _fmt_pm(row, "mse_output"),
                    _fmt_pm(row, "invalid_raw"),
                    _fmt_pm(row, "invalid_output"),
                    _fmt_pm(row, "repair_snapshot_modified_rate"),
                    _fmt_pm(row, "repair_price_level_modified_rate"),
                ]
            )
            + " |"
        )

    repair_rows = [
        row for row in summary if row["method_id"] in {"raw_posthoc_pava", "matched_posthoc_pava"}
    ]
    repair_rows.sort(key=lambda row: (row["dataset_id"], row["method_id"]))
    lines.extend(
        [
            "",
            "## Exact Post-hoc PAVA Repair",
            "",
            "Tick p50/p95 values are run-level quantiles averaged hierarchically; `max ticks` is the maximum observed over all nine runs. Threshold columns are snapshot rates with at least one price correction above the stated tick-equivalent magnitude. W1 columns show error versus real before -> after projection.",
            "",
            "| Dataset | Repair baseline | Projected - raw MSE | p50 ticks | p95 ticks | Max ticks | >0.1 tick | >0.5 tick | >1 tick | Return W1 raw -> projected | Spread W1 raw -> projected | Gap W1 raw -> projected |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in repair_rows:
        mse_delta = float(row["mse_projected_mean"]) - float(row["mse_raw_mean"])
        max_ticks = row.get("repair_ticks_max_global_max")
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["method"],
                    _fmt(mse_delta),
                    _fmt_pm(row, "repair_ticks_p50"),
                    _fmt_pm(row, "repair_ticks_p95"),
                    _fmt(_as_float(max_ticks)),
                    _fmt_pm(row, "repair_snapshot_gt_0_1_ticks"),
                    _fmt_pm(row, "repair_snapshot_gt_0_5_ticks"),
                    _fmt_pm(row, "repair_snapshot_gt_1_tick"),
                    f"{_fmt(row.get('repair_return_w1_raw_real_mean'))} -> {_fmt(row.get('repair_return_w1_projected_real_mean'))}",
                    f"{_fmt(row.get('repair_spread_w1_raw_real_mean'))} -> {_fmt(row.get('repair_spread_w1_projected_real_mean'))}",
                    f"{_fmt(row.get('repair_gap_w1_raw_real_mean'))} -> {_fmt(row.get('repair_gap_w1_projected_real_mean'))}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Depth Clamp Severity",
            "",
            "Depth correction is reported separately from price PAVA. Share-equivalent correction maps the magnitude of a negative normalized sqrt-depth through the recorded training volume scale before squaring; it is a clamp-severity diagnostic, not recovered uncensored shares.",
            "",
            "| Dataset | Repair baseline | Depth-invalid snapshots | Depth levels clamped | p50 normalized | p95 normalized | p95 share-equivalent | Max share-equivalent |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in repair_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["method"],
                    _fmt_pm(row, "repair_depth_invalid_snapshot_rate"),
                    _fmt_pm(row, "repair_depth_level_clamped_rate"),
                    _fmt_pm(row, "repair_depth_norm_p50"),
                    _fmt_pm(row, "repair_depth_norm_p95"),
                    _fmt_pm(row, "repair_depth_share_equiv_p95"),
                    _fmt(_as_float(row.get("repair_depth_share_equiv_max_global_max"))),
                ]
            )
            + " |"
        )

    inloop = [row for row in summary if row["method_id"] == "raw_inloop_pava"]
    inloop.sort(key=lambda row: row["dataset_id"])
    lines.extend(
        [
            "",
            "## In-loop Exact PAVA Path Repair",
            "",
            "Near-zero final diagnostic distance is not used as repair cost. The table reports cumulative denoising-path correction saved by the sampler.",
            "",
            "| Dataset | Output MSE | Final invalid rate | Cumulative projection L2 | Mean pre-projection CBR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in inloop:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    _fmt_pm(row, "mse_output"),
                    _fmt_pm(row, "invalid_output"),
                    _fmt_pm(row, "inloop_cumulative_projection_l2"),
                    _fmt_pm(row, "inloop_pre_projection_cbr"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Interpretation: exact post-hoc PAVA is a cheap and effective repair baseline, but it modifies generated samples. Matched + PAVA confirms that positive links, rather than midpoint/gap coordinate semantics alone, are what make LegalLOB feasible before repair. Fidelity and distribution shifts remain asset dependent.",
            "",
            "Invalidity is limited to the continuous minimum-spread, monotone-ladder, and non-negative-volume contract; it is not exchange-complete legality.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_temporal_table(
    path: Path,
    evaluation_summary: list[dict[str, Any]],
    snowball_summary: list[dict[str, Any]],
) -> None:
    selected = [row for row in evaluation_summary if row["method_id"] in MAIN_METHOD_IDS]
    selected.sort(key=lambda row: (row["dataset_id"], _method_order(row["method_id"])))
    lines = [
        "# Temporal and Distributional Diagnostics",
        "",
        "The 10-frame table uses saved one-shot test trajectories. ACF integrated absolute error (IAE) is the sum of absolute real/generated ACF differences over the finite saved lags `{1,2,5}`. Lower is better. Statistics use the same hierarchical seed aggregation as the feasibility table.",
        "Persistence has identically zero generated returns, so its absolute-return ACF is undefined and reported as `NA`.",
        "",
        "| Dataset | Method | Return W1 | Abs-return ACF IAE | Spread ACF IAE | Volume-difference corr. MAE | Horizon-10 MSE | Persistence skill at h=10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["method"],
                    _fmt_pm(row, "return_w1"),
                    _fmt_pm(row, "abs_return_acf_iae"),
                    _fmt_pm(row, "spread_acf_iae"),
                    _fmt_pm(row, "volume_difference_correlation_mae"),
                    _fmt_pm(row, "horizon10_mse"),
                    _fmt_pm(row, "horizon10_persistence_skill"),
                ]
            )
            + " |"
        )

    snowball_summary = sorted(snowball_summary, key=lambda row: row["dataset_id"])
    lines.extend(
        [
            "",
            "## LegalLOB 100-step Chunk Rollout",
            "",
            "Only LegalLOB has a frozen 100-step rollout panel for every asset and seed. Relative drift is the absolute generated/real difference divided by the absolute real statistic at step 100.",
            "",
            "| Dataset | Max per-element violation rate | Diagnostic projection L2 | Return W1 | Spread W1 | Volume W1 | Spread mean drift | Volume mean drift | Return-std drift | Temporal corr. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in snowball_summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    _fmt_pm(row, "invalid_output"),
                    _fmt_pm(row, "projection_distance"),
                    _fmt_pm(row, "return_w1"),
                    _fmt_pm(row, "spread_w1"),
                    _fmt_pm(row, "volume_w1"),
                    _fmt_pm(row, "spread_relative_abs_error"),
                    _fmt_pm(row, "volume_relative_abs_error"),
                    _fmt_pm(row, "return_std_relative_abs_error"),
                    _fmt_pm(row, "temporal_correlation"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "The tiny nonzero AMZN violation rate is a float32 boundary effect under repeated anchor re-centering (maximum per-run rate below `9e-5`; diagnostic PAVA L2 below `5e-7`), not a crossed-book or negative-volume failure. The mathematical coordinate map remains feasibility preserving, while the emitted long-rollout tensors are feasible only up to finite-precision tolerance.",
            "",
            "Interpretation: LegalLOB preserves the stated continuous feasibility contract through 100-step rollouts up to finite-precision boundary effects, but it does not consistently improve temporal or distributional realism. These diagnostics are evidence of an open dynamics limitation, not a forecasting-superiority claim.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _acf_seed_values(
    dataset: DatasetSpec,
    method: MethodSpec,
    series: str,
    lags: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    real_by_seed = []
    generated_by_seed = []
    for train_seed in TRAIN_SEEDS:
        real_eval = []
        generated_eval = []
        for eval_seed in EVAL_SEEDS:
            path = result_dir(dataset.result_root, train_seed, eval_seed, method.method_id)
            metrics = json.loads((path / "metrics.json").read_text())
            facts = _selected_facts(metrics, method.output)
            acf = ((facts.get("acf") or {}).get(series) or {})
            real_eval.append([float(acf[str(lag)]["real"]) for lag in lags])
            generated_eval.append([float(acf[str(lag)]["generated"]) for lag in lags])
        real_by_seed.append(np.mean(real_eval, axis=0))
        generated_by_seed.append(np.mean(generated_eval, axis=0))
    return np.asarray(real_by_seed), np.asarray(generated_by_seed)


def write_acf_figure(path: Path, dataset: DatasetSpec) -> None:
    import matplotlib.pyplot as plt

    method_ids = (
        "raw_no_projection",
        "raw_posthoc_pava",
        "matched_posthoc_pava",
        "legal_no_projection",
    )
    methods = {method.method_id: method for method in METHODS}
    lags = (1, 2, 5)
    series = (
        ("abs_mid_return", "Absolute mid-return ACF"),
        ("spread", "Spread ACF"),
        ("total_volume", "Total-volume ACF"),
    )
    colors = {
        "raw_no_projection": "#C44E52",
        "raw_posthoc_pava": "#DD8452",
        "matched_posthoc_pava": "#8172B2",
        "legal_no_projection": "#4C72B0",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    for axis, (series_id, title) in zip(axes, series):
        for method_index, method_id in enumerate(method_ids):
            method = methods[method_id]
            real, generated = _acf_seed_values(dataset, method, series_id, lags)
            if method_index == 0:
                axis.plot(lags, real.mean(axis=0), "o-", color="black", label="Real", linewidth=1.8)
            axis.errorbar(
                lags,
                generated.mean(axis=0),
                yerr=generated.std(axis=0, ddof=1),
                marker="o",
                linewidth=1.4,
                capsize=2,
                color=colors[method_id],
                label=method.label,
            )
        axis.set_title(title)
        axis.set_xlabel("Lag (frames)")
        axis.set_xticks(lags)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Autocorrelation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(f"{dataset.label}: one-shot 10-frame temporal diagnostics", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-data-dir", default="paper_data")
    parser.add_argument("--paper-table-dir", default="paper_tables")
    parser.add_argument("--paper-figure-dir", default="paper_figures")
    parser.add_argument(
        "--skip-acf-figure",
        action="store_true",
        help="Write evidence tables without importing the optional plotting stack.",
    )
    args = parser.parse_args()

    paper_data = Path(args.paper_data_dir)
    paper_tables = Path(args.paper_table_dir)
    paper_figures = Path(args.paper_figure_dir)
    paper_data.mkdir(parents=True, exist_ok=True)

    rows = collect_evaluation_rows()
    summary = hierarchical_aggregate(
        rows,
        SCALAR_FIELDS,
        group_fields=("dataset_id", "dataset", "method_id", "method", "output"),
    )
    temporal_summary = hierarchical_aggregate(
        rows,
        TEMPORAL_FIELDS,
        group_fields=("dataset_id", "dataset", "method_id", "method", "output"),
    )
    snowball_rows = collect_snowball_rows()
    snowball_summary = hierarchical_aggregate(
        snowball_rows,
        SNOWBALL_FIELDS,
        group_fields=("dataset_id", "dataset", "method_id", "method", "horizon"),
    )

    protocol = {
        "scope": "saved arrays and JSON only; no retraining and no new diffusion sampling",
        "training_seeds": list(TRAIN_SEEDS),
        "evaluation_seeds": list(EVAL_SEEDS),
        "aggregation": (
            "mean over evaluation seeds within each training seed, followed by mean and "
            "sample standard deviation over independent training seeds"
        ),
        "acf_integrated_absolute_error_lags": [1, 2, 5],
        "claim_boundary": (
            "continuous minimum-spread, monotone-ladder, and non-negative-volume feasibility"
        ),
    }
    feasibility_payload = {
        "protocol": protocol,
        "datasets": [dataset.dataset_id for dataset in DATASETS],
        "summary": summary,
    }
    temporal_payload = {
        "protocol": protocol,
        "one_shot_summary": temporal_summary,
        "legal_lob_100_step_summary": snowball_summary,
    }
    (paper_data / "cross_asset_feasibility_summary.json").write_text(
        json.dumps(feasibility_payload, indent=2) + "\n"
    )
    (paper_data / "temporal_distribution_summary.json").write_text(
        json.dumps(temporal_payload, indent=2) + "\n"
    )
    _write_csv(paper_data / "cross_asset_feasibility_per_run.csv", rows)
    _write_csv(paper_data / "legal_lob_100_step_per_run.csv", snowball_rows)
    write_feasibility_table(paper_tables / "table_cross_asset_feasibility.md", summary)
    write_temporal_table(
        paper_tables / "table_temporal_distribution.md", temporal_summary, snowball_summary
    )
    if not args.skip_acf_figure:
        write_acf_figure(
            paper_figures / "figure_rth_temporal_acf.png",
            DATASETS[0],
        )

    print(f"Validated and summarized {len(rows)} fixed-checkpoint evaluations")
    print(f"Validated and summarized {len(snowball_rows)} LegalLOB rollouts")
    print(f"Wrote {paper_tables / 'table_cross_asset_feasibility.md'}")
    print(f"Wrote {paper_tables / 'table_temporal_distribution.md'}")
    if not args.skip_acf_figure:
        print(f"Wrote {paper_figures / 'figure_rth_temporal_acf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize snapshot-level financial functionals from frozen generations.

This audit is deliberately offline: it reads the paired arrays already saved
by ``evaluate_v2.py`` and never resamples or retrains a model. Prices are
reported in exchange-tick units. The square-root depth transform is inverted
to the clipped share scale recorded by each dataset's normalization manifest.

The fixed market-order size is the median real best-level depth within each
saved evaluation run. This is a normalized-book sensitivity diagnostic, not a
queue-aware execution backtest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.eval.metrics import wasserstein_1d
from scripts.summarize_feasibility_evidence import (
    DATASETS,
    EVAL_SEEDS,
    METHODS,
    TRAIN_SEEDS,
    DatasetSpec,
    MethodSpec,
    hierarchical_aggregate,
    result_dir,
)


OUTPUT_METHOD_IDS = (
    "persistence",
    "raw_no_projection",
    "raw_posthoc_pava",
    "raw_inloop_pava",
    "matched_no_projection",
    "matched_posthoc_pava",
    "legal_no_projection",
)

SUMMARY_FIELDS = (
    "admissible_snapshot_rate",
    "microprice_offset_w1_ticks",
    "imbalance_l1_w1",
    "imbalance_l5_w1",
    "imbalance_l10_w1",
    "market_order_slippage_w1_ticks",
    "market_order_fill_rate_abs_error",
    "liquidity_1tick_w1_target_units",
    "liquidity_5tick_w1_target_units",
    "liquidity_10tick_w1_target_units",
    "spread_p95_abs_error_ticks",
    "total_depth_p95_relative_abs_error",
    "repair_microprice_abs_shift_ticks",
    "repair_imbalance_l10_abs_shift",
    "repair_market_order_slippage_abs_shift_ticks",
    "repair_liquidity_5tick_abs_shift_target_units",
    "repair_raw_negative_slippage_rate",
    "repair_final_negative_slippage_rate",
)


def _flatten_lob(lob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [N, 2, 20, T] into price/depth matrices [N*T, 20]."""
    array = np.asarray(lob)
    if array.ndim != 4 or array.shape[1:3] != (2, 20):
        raise ValueError(f"Expected [N, 2, 20, T], got {array.shape}")
    prices = np.transpose(array[:, 0], (0, 2, 1)).reshape(-1, 20)
    volumes = np.transpose(array[:, 1], (0, 2, 1)).reshape(-1, 20)
    return prices.astype(np.float64, copy=False), volumes.astype(np.float64, copy=False)


def _to_economic_units(
    lob: np.ndarray,
    normalization: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    prices, volumes = _flatten_lob(lob)
    ticks_per_unit = float(normalization["ticks_per_price_unit"])
    volume_scale = float(normalization["volume_scale"])
    prices_ticks = prices * ticks_per_unit
    # Negative raw depths have no execution semantics. Clipping here is used
    # only to keep algebraic sensitivity diagnostics finite; admissibility is
    # reported separately and remains the gating metric.
    volumes_shares = np.square(np.maximum(volumes, 0.0) * volume_scale)
    return prices_ticks, volumes_shares, ticks_per_unit


def admissible_snapshot_mask(
    lob: np.ndarray,
    normalization: dict[str, Any],
) -> np.ndarray:
    prices, volumes = _flatten_lob(lob)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    min_spread = float(normalization["min_spread_normalized"])
    min_gap = float(normalization["min_level_gap_normalized"])
    tolerance = max(1e-8, min(min_spread, max(min_gap, 1e-12)) * 2e-5)
    return (
        (ask[:, 0] - bid[:, 0] >= min_spread - tolerance)
        & np.all(np.diff(ask, axis=1) >= min_gap - tolerance, axis=1)
        & np.all(-np.diff(bid, axis=1) >= min_gap - tolerance, axis=1)
        & np.all(volumes >= -tolerance, axis=1)
    )


def _imbalance(ask_volume: np.ndarray, bid_volume: np.ndarray, levels: int) -> np.ndarray:
    ask = ask_volume[:, :levels].sum(axis=1)
    bid = bid_volume[:, :levels].sum(axis=1)
    denominator = ask + bid
    return np.divide(
        bid - ask,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def _market_order_slippage(
    prices: np.ndarray,
    volumes: np.ndarray,
    target_size: float,
    buy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tick slippage and full-fill mask for a fixed-size market order."""
    if buy:
        side_prices = prices[:, :10]
        side_volumes = volumes[:, :10]
        best = side_prices[:, 0]
    else:
        side_prices = prices[:, 10:]
        side_volumes = volumes[:, 10:]
        best = side_prices[:, 0]

    cumulative = np.cumsum(side_volumes, axis=1)
    full = cumulative[:, -1] + 1e-12 >= target_size
    previous = np.concatenate(
        [np.zeros((len(side_volumes), 1)), cumulative[:, :-1]], axis=1
    )
    take = np.clip(target_size - previous, 0.0, side_volumes)
    vwap = np.divide(
        (take * side_prices).sum(axis=1),
        target_size,
        out=np.full(len(side_prices), np.nan),
        where=target_size > 0,
    )
    slippage = vwap - best if buy else best - vwap
    slippage[~full] = np.nan
    return slippage, full


def _liquidity_within_ticks(
    ask_price: np.ndarray,
    bid_price: np.ndarray,
    ask_volume: np.ndarray,
    bid_volume: np.ndarray,
    ticks: float,
    target_size: float,
) -> np.ndarray:
    ask_near = ask_price <= ask_price[:, [0]] + ticks + 1e-9
    bid_near = bid_price >= bid_price[:, [0]] - ticks - 1e-9
    liquidity = 0.5 * (
        (ask_volume * ask_near).sum(axis=1)
        + (bid_volume * bid_near).sum(axis=1)
    )
    return liquidity / max(target_size, 1e-12)


def functional_vectors(
    lob: np.ndarray,
    normalization: dict[str, Any],
    target_size: float | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    prices, volumes, _ = _to_economic_units(lob, normalization)
    ask_price = prices[:, :10]
    bid_price = prices[:, 10:]
    ask_volume = volumes[:, :10]
    bid_volume = volumes[:, 10:]

    if target_size is None:
        best_depth = np.concatenate([ask_volume[:, 0], bid_volume[:, 0]])
        positive = best_depth[np.isfinite(best_depth) & (best_depth > 0)]
        if positive.size == 0:
            raise ValueError("Cannot define fixed market-order size from zero real depth")
        target_size = float(np.median(positive))

    mid = 0.5 * (ask_price[:, 0] + bid_price[:, 0])
    best_total = ask_volume[:, 0] + bid_volume[:, 0]
    microprice = np.divide(
        ask_price[:, 0] * bid_volume[:, 0]
        + bid_price[:, 0] * ask_volume[:, 0],
        best_total,
        out=mid.copy(),
        where=best_total > 0,
    )
    buy_slippage, buy_full = _market_order_slippage(
        prices, volumes, target_size, buy=True
    )
    sell_slippage, sell_full = _market_order_slippage(
        prices, volumes, target_size, buy=False
    )
    market_slippage = np.concatenate([buy_slippage, sell_slippage])
    full_fill = np.concatenate([buy_full, sell_full]).astype(np.float64)

    vectors = {
        "microprice_offset": microprice - mid,
        "imbalance_l1": _imbalance(ask_volume, bid_volume, 1),
        "imbalance_l5": _imbalance(ask_volume, bid_volume, 5),
        "imbalance_l10": _imbalance(ask_volume, bid_volume, 10),
        "market_order_slippage": market_slippage,
        "market_order_full_fill": full_fill,
        "spread": ask_price[:, 0] - bid_price[:, 0],
        "total_depth": ask_volume.sum(axis=1) + bid_volume.sum(axis=1),
    }
    for ticks in (1, 5, 10):
        vectors[f"liquidity_{ticks}tick"] = _liquidity_within_ticks(
            ask_price,
            bid_price,
            ask_volume,
            bid_volume,
            ticks=float(ticks),
            target_size=target_size,
        )
    return vectors, target_size


def _finite_mean_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if np.any(mask) else float("nan")


def _quantile_abs_error(a: np.ndarray, b: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not a.size or not b.size:
        return float("nan")
    return float(abs(np.quantile(a, q) - np.quantile(b, q)))


def summarize_output(
    generated: np.ndarray,
    real: np.ndarray,
    normalization: dict[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], float]:
    real_vectors, target_size = functional_vectors(real, normalization)
    generated_vectors, _ = functional_vectors(generated, normalization, target_size)
    admissible = admissible_snapshot_mask(generated, normalization)

    result = {
        "admissible_snapshot_rate": float(admissible.mean()),
        "market_order_target_size_clipped_shares": target_size,
        "microprice_offset_w1_ticks": wasserstein_1d(
            generated_vectors["microprice_offset"], real_vectors["microprice_offset"]
        ),
        "imbalance_l1_w1": wasserstein_1d(
            generated_vectors["imbalance_l1"], real_vectors["imbalance_l1"]
        ),
        "imbalance_l5_w1": wasserstein_1d(
            generated_vectors["imbalance_l5"], real_vectors["imbalance_l5"]
        ),
        "imbalance_l10_w1": wasserstein_1d(
            generated_vectors["imbalance_l10"], real_vectors["imbalance_l10"]
        ),
        "market_order_slippage_w1_ticks": wasserstein_1d(
            generated_vectors["market_order_slippage"],
            real_vectors["market_order_slippage"],
        ),
        "market_order_fill_rate_abs_error": abs(
            float(generated_vectors["market_order_full_fill"].mean())
            - float(real_vectors["market_order_full_fill"].mean())
        ),
        "spread_p95_abs_error_ticks": _quantile_abs_error(
            generated_vectors["spread"], real_vectors["spread"], 0.95
        ),
    }
    real_depth_p95 = float(np.quantile(real_vectors["total_depth"], 0.95))
    result["total_depth_p95_relative_abs_error"] = (
        _quantile_abs_error(generated_vectors["total_depth"], real_vectors["total_depth"], 0.95)
        / max(real_depth_p95, 1e-12)
    )
    for ticks in (1, 5, 10):
        result[f"liquidity_{ticks}tick_w1_target_units"] = wasserstein_1d(
            generated_vectors[f"liquidity_{ticks}tick"],
            real_vectors[f"liquidity_{ticks}tick"],
        )
    return result, generated_vectors, real_vectors, target_size


def summarize_paired_repair(
    raw: np.ndarray,
    projected: np.ndarray,
    normalization: dict[str, Any],
    target_size: float,
) -> dict[str, float]:
    raw_vectors, _ = functional_vectors(raw, normalization, target_size)
    final_vectors, _ = functional_vectors(projected, normalization, target_size)
    raw_slippage = raw_vectors["market_order_slippage"]
    final_slippage = final_vectors["market_order_slippage"]
    return {
        "repair_microprice_abs_shift_ticks": _finite_mean_abs_difference(
            raw_vectors["microprice_offset"], final_vectors["microprice_offset"]
        ),
        "repair_imbalance_l10_abs_shift": _finite_mean_abs_difference(
            raw_vectors["imbalance_l10"], final_vectors["imbalance_l10"]
        ),
        "repair_market_order_slippage_abs_shift_ticks": _finite_mean_abs_difference(
            raw_slippage, final_slippage
        ),
        "repair_liquidity_5tick_abs_shift_target_units": _finite_mean_abs_difference(
            raw_vectors["liquidity_5tick"], final_vectors["liquidity_5tick"]
        ),
        "repair_raw_negative_slippage_rate": float(
            np.mean(raw_slippage[np.isfinite(raw_slippage)] < -1e-8)
        ),
        "repair_final_negative_slippage_rate": float(
            np.mean(final_slippage[np.isfinite(final_slippage)] < -1e-8)
        ),
    }


def collect_rows(
    datasets: Iterable[DatasetSpec] = DATASETS,
    methods: Iterable[MethodSpec] = METHODS,
) -> list[dict[str, Any]]:
    selected_methods = [m for m in methods if m.method_id in OUTPUT_METHOD_IDS]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        normalization = json.loads(dataset.normalization.read_text())
        for train_seed in TRAIN_SEEDS:
            for method in selected_methods:
                eval_seeds = (0,) if method.method_id == "persistence" else EVAL_SEEDS
                for eval_seed in eval_seeds:
                    path = result_dir(dataset.result_root, train_seed, eval_seed, method.method_id)
                    raw_path = path / "generated_raw.npy"
                    projected_path = path / "generated_projected.npy"
                    real_path = path / "real_future.npy"
                    for required in (raw_path, projected_path, real_path):
                        if not required.exists():
                            raise FileNotFoundError(f"Missing frozen array: {required}")
                    output_path = projected_path if method.output == "projected" else raw_path
                    generated = np.load(output_path, mmap_mode="r")
                    real = np.load(real_path, mmap_mode="r")
                    metrics, _, _, target_size = summarize_output(
                        generated, real, normalization
                    )
                    row: dict[str, Any] = {
                        "dataset_id": dataset.dataset_id,
                        "dataset": dataset.label,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "method_id": method.method_id,
                        "method": method.label,
                        "output": method.output,
                        "result_dir": str(path),
                    }
                    row.update(metrics)
                    for field in SUMMARY_FIELDS:
                        row.setdefault(field, None)
                    if method.method_id.endswith("_posthoc_pava"):
                        row.update(
                            summarize_paired_repair(
                                np.load(raw_path, mmap_mode="r"),
                                np.load(projected_path, mmap_mode="r"),
                                normalization,
                                target_size,
                            )
                        )
                    rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(mean: float | None, std: float | None, digits: int = 3) -> str:
    if mean is None:
        return "--"
    return f"{mean:.{digits}f} +/- {(std or 0.0):.{digits}f}"


def _markdown_table(summary: list[dict[str, Any]]) -> str:
    keep = {
        "persistence",
        "raw_posthoc_pava",
        "raw_inloop_pava",
        "matched_posthoc_pava",
        "legal_no_projection",
    }
    lines = [
        "# Snapshot Financial-Functional Sensitivity",
        "",
        "All values are mean +/- sample SD across training seeds after averaging",
        "fixed-checkpoint sampling seeds. Prices are in ticks; market-order size",
        "is the median real best-level depth in the corresponding frozen run.",
        "Depth is on the inverse-transformed, train-clipped share scale. This is",
        "a snapshot sensitivity audit, not queue-aware execution backtesting.",
        "",
        "| Dataset | Method | Invalid % | Microprice W1 | OI10 W1 | MO slip W1 | Liq@5 W1/Q | Fill error pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["method_id"] not in keep:
            continue
        invalid = 100.0 * (1.0 - row.get("admissible_snapshot_rate_mean", float("nan")))
        lines.append(
            "| {dataset} | {method} | {invalid:.3f} | {micro} | {imb} | {slip} | {liq} | {fill} |".format(
                dataset=row["dataset"],
                method=row["method"],
                invalid=invalid,
                micro=_fmt(row.get("microprice_offset_w1_ticks_mean"), row.get("microprice_offset_w1_ticks_std")),
                imb=_fmt(row.get("imbalance_l10_w1_mean"), row.get("imbalance_l10_w1_std")),
                slip=_fmt(row.get("market_order_slippage_w1_ticks_mean"), row.get("market_order_slippage_w1_ticks_std")),
                liq=_fmt(row.get("liquidity_5tick_w1_target_units_mean"), row.get("liquidity_5tick_w1_target_units_std")),
                fill=_fmt(
                    100.0 * row.get("market_order_fill_rate_abs_error_mean", float("nan")),
                    100.0 * row.get("market_order_fill_rate_abs_error_std", 0.0),
                    digits=2,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Paired post-hoc repair sensitivity",
            "",
            "Raw-level market-order functionals are algebraic diagnostics only when",
            "the parent snapshot is structurally invalid.",
            "",
            "| Dataset | Repair | Microprice shift | OI10 shift | MO slip shift | Liq@5 shift/Q | Negative slip raw/final % |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        if not row["method_id"].endswith("_posthoc_pava"):
            continue
        raw_negative = 100.0 * row.get("repair_raw_negative_slippage_rate_mean", float("nan"))
        final_negative = 100.0 * row.get("repair_final_negative_slippage_rate_mean", float("nan"))
        lines.append(
            "| {dataset} | {method} | {micro} | {imb} | {slip} | {liq} | {raw:.2f}/{final:.2f} |".format(
                dataset=row["dataset"],
                method=row["method"],
                micro=_fmt(row.get("repair_microprice_abs_shift_ticks_mean"), row.get("repair_microprice_abs_shift_ticks_std")),
                imb=_fmt(row.get("repair_imbalance_l10_abs_shift_mean"), row.get("repair_imbalance_l10_abs_shift_std")),
                slip=_fmt(row.get("repair_market_order_slippage_abs_shift_ticks_mean"), row.get("repair_market_order_slippage_abs_shift_ticks_std")),
                liq=_fmt(row.get("repair_liquidity_5tick_abs_shift_target_units_mean"), row.get("repair_liquidity_5tick_abs_shift_target_units_std")),
                raw=raw_negative,
                final=final_negative,
            )
        )
    lines.append("")
    return "\n".join(lines)


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
    parser.add_argument("--output-dir", default="paper_data")
    parser.add_argument(
        "--table-path",
        default="paper_tables/table_financial_functional_sensitivity.md",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    summary = hierarchical_aggregate(
        rows,
        SUMMARY_FIELDS,
        group_fields=("dataset_id", "dataset", "method_id", "method", "output"),
    )
    _write_csv(output_dir / "financial_functional_sensitivity_per_run.csv", rows)
    _write_csv(output_dir / "financial_functional_sensitivity_summary.csv", summary)
    (output_dir / "financial_functional_sensitivity_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    table_path.write_text(_markdown_table(summary))
    print(f"Wrote {len(rows)} per-run rows and {len(summary)} aggregates")
    print(f"Table: {table_path}")


if __name__ == "__main__":
    main()

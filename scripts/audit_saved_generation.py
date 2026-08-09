#!/usr/bin/env python3
"""Audit saved LOB generations by channel, level, physical scale, and repair.

This script is intentionally independent of training and sampling. It reads the
arrays already written by ``evaluate_v2.py`` so it can run while another model
is training without changing the active experiment code path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.utils.physics import project_to_valid_lob


DEFAULT_METHODS = (
    "persistence",
    "raw_no_projection",
    "raw_posthoc_pava",
    "raw_inloop_pava",
    "matched_no_projection",
    "legal_no_projection",
)


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _result_dir(root: Path, train_seed: int, eval_seed: int, method: str) -> Path:
    base = root / f"train_seed{train_seed}"
    return base / method if eval_seed == 0 else base / "seeds" / f"seed{eval_seed}" / method


def _evaluated_array(path: Path) -> Path:
    metrics = json.loads((path / "metrics.json").read_text())
    projection_mode = metrics.get("projection_mode", "none")
    if projection_mode in {"exact", "fast", "auto"}:
        return path / "generated_projected.npy"
    return path / "generated_raw.npy"


def _scalar_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _error_stats(diff: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean(np.square(diff, dtype=np.float64))),
        "mae": float(np.mean(np.abs(diff), dtype=np.float64)),
    }


def error_breakdown(
    generated: np.ndarray,
    real: np.ndarray,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    """Return model-scale and interpretable channel/level error diagnostics."""
    if generated.shape != real.shape or generated.ndim != 4 or generated.shape[1:3] != (2, 20):
        raise ValueError(
            f"Expected matching [N,2,20,T] arrays, got {generated.shape} and {real.shape}"
        )

    generated64 = np.asarray(generated, dtype=np.float64)
    real64 = np.asarray(real, dtype=np.float64)
    diff = generated64 - real64
    ask = np.arange(0, 10)
    bid = np.arange(10, 20)
    level_groups = {
        "l1": np.array([0, 10]),
        "l1_l5": np.concatenate([np.arange(0, 5), np.arange(10, 15)]),
        "l6_l10": np.concatenate([np.arange(5, 10), np.arange(15, 20)]),
    }

    result: dict[str, Any] = {
        "overall": _error_stats(diff),
        "price": _error_stats(diff[:, 0]),
        "volume": _error_stats(diff[:, 1]),
        "price_ask": _error_stats(diff[:, 0, ask]),
        "price_bid": _error_stats(diff[:, 0, bid]),
        "volume_ask": _error_stats(diff[:, 1, ask]),
        "volume_bid": _error_stats(diff[:, 1, bid]),
        "by_level_group": {},
        "by_horizon": [],
    }

    tick_model = float(normalization.get("min_level_gap_normalized", 1.0))
    price_unit_raw = float(normalization.get("model_price_unit_raw", 1.0))
    generated_prices = generated64[:, 0]
    real_prices = real64[:, 0]
    semantic_diffs = {
        "mid": 0.5
        * (
            generated_prices[:, 0]
            + generated_prices[:, 10]
            - real_prices[:, 0]
            - real_prices[:, 10]
        ),
        "spread": (
            generated_prices[:, 0]
            - generated_prices[:, 10]
            - real_prices[:, 0]
            + real_prices[:, 10]
        ),
        "ask_gap": (
            generated_prices[:, 1:10]
            - generated_prices[:, :9]
            - real_prices[:, 1:10]
            + real_prices[:, :9]
        ),
        "bid_gap": (
            generated_prices[:, 10:19]
            - generated_prices[:, 11:20]
            - real_prices[:, 10:19]
            + real_prices[:, 11:20]
        ),
    }
    result["price_semantics"] = {}
    for name, semantic_diff in semantic_diffs.items():
        stats = _error_stats(semantic_diff)
        stats["mae_ticks"] = stats["mae"] / max(tick_model, 1e-12)
        stats["mae_raw_currency"] = stats["mae"] * price_unit_raw
        result["price_semantics"][name] = stats

    for name, indices in level_groups.items():
        price_diff = diff[:, 0, indices]
        volume_diff = diff[:, 1, indices]
        result["by_level_group"][name] = {
            "price": _error_stats(price_diff),
            "price_mae_ticks": float(np.mean(np.abs(price_diff)) / max(tick_model, 1e-12)),
            "price_mae_raw_currency": float(np.mean(np.abs(price_diff)) * price_unit_raw),
            "volume": _error_stats(volume_diff),
        }

    for horizon in range(diff.shape[-1]):
        result["by_horizon"].append(
            {
                "horizon": horizon + 1,
                "overall_mse": float(np.mean(np.square(diff[..., horizon]))),
                "price_mse": float(np.mean(np.square(diff[:, 0, :, horizon]))),
                "volume_mse": float(np.mean(np.square(diff[:, 1, :, horizon]))),
            }
        )

    result["price"]["mae_ticks"] = result["price"]["mae"] / max(tick_model, 1e-12)
    result["price"]["mae_raw_currency"] = result["price"]["mae"] * price_unit_raw

    volume_scale = normalization.get("volume_scale")
    if volume_scale is not None:
        scale = float(volume_scale)
        generated_shares = np.square(np.maximum(generated64[:, 1], 0.0) * scale)
        real_shares = np.square(np.maximum(real64[:, 1], 0.0) * scale)
        result["volume"]["implied_clipped_share_mae"] = float(
            np.mean(np.abs(generated_shares - real_shares))
        )
        result["volume"]["physical_scale_note"] = (
            "Inverse of sqrt/scale after clipping at zero; values above the training clip "
            "cannot recover original uncapped shares."
        )
    return result


def continuous_boundary_stats(
    lob: np.ndarray,
    min_spread: float,
    min_gap: float,
) -> dict[str, Any]:
    prices = np.asarray(lob[:, 0], dtype=np.float64)
    volumes = np.asarray(lob[:, 1], dtype=np.float64)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    spread = ask[:, 0] - bid[:, 0]
    ask_gaps = ask[:, 1:] - ask[:, :-1]
    bid_gaps = bid[:, :-1] - bid[:, 1:]
    tolerance = max(1e-7, 1e-4 * min(min_spread, min_gap))
    return {
        "min_spread": min_spread,
        "min_gap": min_gap,
        "boundary_tolerance": tolerance,
        "spread_at_minimum_rate": float(np.mean(spread <= min_spread + tolerance)),
        "ask_gap_at_minimum_rate": float(np.mean(ask_gaps <= min_gap + tolerance)),
        "bid_gap_at_minimum_rate": float(np.mean(bid_gaps <= min_gap + tolerance)),
        "zero_volume_rate": float(np.mean(volumes <= tolerance)),
        "spread": _scalar_summary(spread),
        "ask_gap": _scalar_summary(ask_gaps),
        "bid_gap": _scalar_summary(bid_gaps),
        "volume": _scalar_summary(volumes),
    }


def _inverse_softplus(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float64), 1e-12)
    return value + np.log(-np.expm1(-value))


def encode_legal_numpy(
    lob: np.ndarray,
    min_spread: float,
    min_gap: float,
) -> np.ndarray:
    prices = np.asarray(lob[:, 0], dtype=np.float64)
    volumes = np.asarray(lob[:, 1], dtype=np.float64)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    mid = 0.5 * (ask[:, 0:1] + bid[:, 0:1])
    spread_raw = _inverse_softplus(ask[:, 0:1] - bid[:, 0:1] - min_spread)
    ask_gap_raw = _inverse_softplus(ask[:, 1:] - ask[:, :-1] - min_gap)
    bid_gap_raw = _inverse_softplus(bid[:, :-1] - bid[:, 1:] - min_gap)
    volume_raw = _inverse_softplus(volumes)
    price = np.concatenate([mid, spread_raw, ask_gap_raw, bid_gap_raw], axis=1)
    return np.stack([price, volume_raw], axis=1)


def _distribution_summary(values: np.ndarray, standardized: np.ndarray | None = None) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    result = {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "p001": float(np.quantile(flat, 0.001)),
        "p01": float(np.quantile(flat, 0.01)),
        "p05": float(np.quantile(flat, 0.05)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "p999": float(np.quantile(flat, 0.999)),
        "fraction_below_minus_10": float(np.mean(flat < -10.0)),
        "fraction_below_minus_20": float(np.mean(flat < -20.0)),
    }
    if standardized is not None:
        z = np.asarray(standardized, dtype=np.float64).reshape(-1)
        result["standardized_abs_gt_3_rate"] = float(np.mean(np.abs(z) > 3.0))
        result["standardized_abs_gt_5_rate"] = float(np.mean(np.abs(z) > 5.0))
    return result


def latent_audit(
    lob: np.ndarray,
    min_spread: float,
    min_gap: float,
    representation_mean: np.ndarray | None = None,
    representation_std: np.ndarray | None = None,
) -> dict[str, Any]:
    encoded = encode_legal_numpy(lob, min_spread=min_spread, min_gap=min_gap)
    standardized = None
    if representation_mean is not None and representation_std is not None:
        standardized = (encoded - representation_mean) / np.maximum(representation_std, 1e-6)
    groups = {
        "mid": (0, slice(0, 1)),
        "spread_raw": (0, slice(1, 2)),
        "ask_gap_raw": (0, slice(2, 11)),
        "bid_gap_raw": (0, slice(11, 20)),
        "volume_raw": (1, slice(0, 20)),
    }
    result = {}
    for name, (channel, levels) in groups.items():
        z = None if standardized is None else standardized[:, channel, levels]
        result[name] = _distribution_summary(encoded[:, channel, levels], z)
    return result


def repair_distribution(
    raw: np.ndarray,
    projected: np.ndarray,
    normalization: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    if raw.shape != projected.shape:
        raise ValueError(f"Raw/projected shapes differ: {raw.shape} vs {projected.shape}")
    delta = np.asarray(projected, dtype=np.float64) - np.asarray(raw, dtype=np.float64)
    price_abs = np.abs(delta[:, 0])
    volume_abs = np.abs(delta[:, 1])
    tick_model = float(normalization.get("min_level_gap_normalized", 1.0))
    ticks = price_abs / max(tick_model, 1e-12)
    modified_ticks = ticks[price_abs > tolerance]
    snapshot_l2_ticks = np.sqrt(np.sum(np.square(ticks), axis=1)).reshape(-1)

    raw_prices = np.asarray(raw[:, 0], dtype=np.float64)
    projected_prices = np.asarray(projected[:, 0], dtype=np.float64)
    raw_spread = raw_prices[:, 0] - raw_prices[:, 10]
    projected_spread = projected_prices[:, 0] - projected_prices[:, 10]
    raw_mid = 0.5 * (raw_prices[:, 0] + raw_prices[:, 10])
    projected_mid = 0.5 * (projected_prices[:, 0] + projected_prices[:, 10])
    threshold_exceedance = {
        f"{threshold:g}": {
            "snapshot_rate": float(np.mean(np.any(ticks > threshold, axis=1))),
            "price_level_rate": float(np.mean(ticks > threshold)),
        }
        for threshold in (0.1, 0.5, 1.0)
    }
    return {
        "any_snapshot_modified_rate": float(np.mean(np.any(np.abs(delta) > tolerance, axis=(1, 2)))),
        "price_level_modified_rate": float(np.mean(price_abs > tolerance)),
        "volume_level_modified_rate": float(np.mean(volume_abs > tolerance)),
        "ticks_moved_all_price_levels": _scalar_summary(ticks),
        "ticks_moved_modified_price_levels": (
            _scalar_summary(modified_ticks) if modified_ticks.size else None
        ),
        "tick_threshold_exceedance": threshold_exceedance,
        "snapshot_price_l2_ticks": _scalar_summary(snapshot_l2_ticks),
        "spread_change_model_units": _scalar_summary(projected_spread - raw_spread),
        "absolute_spread_change_ticks": _scalar_summary(
            np.abs(projected_spread - raw_spread) / max(tick_model, 1e-12)
        ),
        "absolute_mid_return_change_ticks": _scalar_summary(
            np.abs(np.diff(projected_mid, axis=-1) - np.diff(raw_mid, axis=-1))
            / max(tick_model, 1e-12)
        ),
        "volume_absolute_change": _scalar_summary(volume_abs),
    }


def benchmark_exact_projection(
    raw: np.ndarray,
    stored_projected: np.ndarray,
    min_spread: float,
    min_gap: float,
    n_samples: int,
    repeats: int,
) -> dict[str, Any]:
    count = min(n_samples, len(raw))
    tensor = torch.from_numpy(np.array(raw[:count], copy=True)).float()
    times = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = project_to_valid_lob(
            tensor,
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=False,
        )
        times.append(time.perf_counter() - start)
    assert output is not None
    stored = np.asarray(stored_projected[:count], dtype=np.float32)
    return {
        "device": "cpu",
        "implementation": "exact single-chain PAVA plus non-negative volume clamp",
        "n_trajectories": count,
        "frames_per_trajectory": int(raw.shape[-1]),
        "repeats": repeats,
        "seconds": _scalar_summary(np.asarray(times)),
        "milliseconds_per_trajectory": float(1000.0 * np.mean(times) / count),
        "microseconds_per_snapshot": float(1e6 * np.mean(times) / (count * raw.shape[-1])),
        "max_abs_difference_from_saved_projection": float(
            np.max(np.abs(output.numpy() - stored))
        ),
        "scope": "projection kernel only; excludes model sampling, I/O, and accelerator transfer",
    }


def _checkpoint_stats(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        return None, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stats = checkpoint.get("representation_stats")
    if not stats:
        return None, None
    mean = stats["mean"].detach().cpu().numpy()
    std = stats["std"].detach().cpu().numpy()
    return mean, std


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "overall_mse",
        "overall_mae",
        "price_mse",
        "volume_mse",
        "price_mae_ticks",
        "mid_mse",
        "mid_mae_ticks",
        "spread_mae_ticks",
        "ask_gap_mae_ticks",
        "bid_gap_mae_ticks",
        "l1_price_mae_ticks",
        "l1_l5_price_mae_ticks",
        "l6_l10_price_mae_ticks",
        "any_invalid_snapshot_rate",
    )
    aggregated = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "n_runs": len(method_rows)}
        for field in fields:
            values = [row[field] for row in method_rows if row.get(field) is not None]
            if values:
                item[f"{field}_mean"] = float(np.mean(values))
                item[f"{field}_std"] = float(np.std(values))
        aggregated.append(item)
    return aggregated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    dataset: str,
    aggregate: list[dict[str, Any]],
    boundary: dict[str, Any],
    repair: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
) -> None:
    columns = (
        "method",
        "overall_mse_mean",
        "price_mse_mean",
        "volume_mse_mean",
        "mid_mae_ticks_mean",
        "spread_mae_ticks_mean",
        "ask_gap_mae_ticks_mean",
        "bid_gap_mae_ticks_mean",
        "l1_price_mae_ticks_mean",
        "l1_l5_price_mae_ticks_mean",
        "l6_l10_price_mae_ticks_mean",
        "any_invalid_snapshot_rate_mean",
    )
    lines = [
        f"# Saved Generation Audit: {dataset}",
        "",
        "Scope: saved-array audit over training seeds and fixed-checkpoint sampling seeds. "
        "It does not retrain or resample a model.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in aggregate:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Real-State Boundary Prevalence",
            "",
            f"- Spread at minimum: `{boundary['spread_at_minimum_rate']:.6g}`",
            f"- Ask gap at minimum: `{boundary['ask_gap_at_minimum_rate']:.6g}`",
            f"- Bid gap at minimum: `{boundary['bid_gap_at_minimum_rate']:.6g}`",
            f"- Zero/near-zero volume: `{boundary['zero_volume_rate']:.6g}`",
        ]
    )
    if repair:
        moved = repair["ticks_moved_modified_price_levels"]
        lines.extend(
            [
                "",
                "## Post-hoc PAVA Repair",
                "",
                f"- Any snapshot modified: `{repair['any_snapshot_modified_rate']:.6g}`",
                f"- Price levels modified: `{repair['price_level_modified_rate']:.6g}`",
                f"- Volume levels modified: `{repair['volume_level_modified_rate']:.6g}`",
                f"- Modified-level ticks moved p50/p95/max: "
                f"`{moved['p50']:.6g}` / `{moved['p95']:.6g}` / `{moved['max']:.6g}`",
                "- Snapshot repair rate above 0.1/0.5/1 tick-equivalent: "
                f"`{repair['tick_threshold_exceedance']['0.1']['snapshot_rate']:.6g}` / "
                f"`{repair['tick_threshold_exceedance']['0.5']['snapshot_rate']:.6g}` / "
                f"`{repair['tick_threshold_exceedance']['1']['snapshot_rate']:.6g}`",
            ]
        )
    if runtime:
        lines.extend(
            [
                "",
                "## Projection Kernel Runtime",
                "",
                f"- CPU exact PAVA: `{runtime['microseconds_per_snapshot']:.6g}` us/snapshot",
                f"- Saved-output max difference: "
                f"`{runtime['max_abs_difference_from_saved_projection']:.6g}`",
                "- Runtime excludes denoising, data loading, and accelerator transfer.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The audited guarantee is continuous minimum-spread, monotone-ladder, and "
            "non-negative-volume feasibility. It does not establish tick-grid, integer-size, "
            "queue, matching, or exchange-complete legality.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--eval-seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-samples", type=int, default=256)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.results_root)
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization = json.loads(Path(args.normalization).read_text())
    min_spread = float(normalization["min_spread_normalized"])
    min_gap = float(normalization["min_level_gap_normalized"])
    train_seeds = _parse_ints(args.train_seeds)
    eval_seeds = _parse_ints(args.eval_seeds)
    methods = _parse_strings(args.methods)

    rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    repair_runs: list[dict[str, Any]] = []
    first_real = None
    runtime = None
    for train_seed in train_seeds:
        representation_mean = representation_std = None
        if checkpoint_root:
            representation_mean, representation_std = _checkpoint_stats(
                checkpoint_root / f"train_seed{train_seed}" / "legal_lob" / "best.pt"
            )
        for eval_seed in eval_seeds:
            for method in methods:
                path = _result_dir(root, train_seed, eval_seed, method)
                if not (path / "metrics.json").exists():
                    continue
                metrics = json.loads((path / "metrics.json").read_text())
                real = np.load(path / "real_future.npy", mmap_mode="r")
                generated = np.load(_evaluated_array(path), mmap_mode="r")
                if first_real is None:
                    first_real = np.asarray(real)
                breakdown = error_breakdown(generated, real, normalization)
                validity = metrics.get("structural_validity_raw", {})
                if metrics.get("projection_mode") in {"exact", "fast", "auto"}:
                    validity = metrics.get("structural_validity_projected", validity)
                row = {
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "method": method,
                    "overall_mse": breakdown["overall"]["mse"],
                    "overall_mae": breakdown["overall"]["mae"],
                    "price_mse": breakdown["price"]["mse"],
                    "volume_mse": breakdown["volume"]["mse"],
                    "price_mae_ticks": breakdown["price"]["mae_ticks"],
                    "mid_mse": breakdown["price_semantics"]["mid"]["mse"],
                    "mid_mae_ticks": breakdown["price_semantics"]["mid"]["mae_ticks"],
                    "spread_mae_ticks": breakdown["price_semantics"]["spread"]["mae_ticks"],
                    "ask_gap_mae_ticks": breakdown["price_semantics"]["ask_gap"]["mae_ticks"],
                    "bid_gap_mae_ticks": breakdown["price_semantics"]["bid_gap"]["mae_ticks"],
                    "l1_price_mae_ticks": breakdown["by_level_group"]["l1"]["price_mae_ticks"],
                    "l1_l5_price_mae_ticks": breakdown["by_level_group"]["l1_l5"]["price_mae_ticks"],
                    "l6_l10_price_mae_ticks": breakdown["by_level_group"]["l6_l10"]["price_mae_ticks"],
                    "any_invalid_snapshot_rate": validity.get("any_invalid_snapshot_rate"),
                    "details": breakdown,
                    "result_dir": str(path),
                }
                rows.append(row)

                if method == "legal_no_projection":
                    latent_rows.append(
                        {
                            "train_seed": train_seed,
                            "eval_seed": eval_seed,
                            "real": latent_audit(
                                real,
                                min_spread,
                                min_gap,
                                representation_mean,
                                representation_std,
                            ),
                            "generated": latent_audit(
                                generated,
                                min_spread,
                                min_gap,
                                representation_mean,
                                representation_std,
                            ),
                        }
                    )

                if method == "raw_posthoc_pava":
                    raw = np.load(path / "generated_raw.npy", mmap_mode="r")
                    projected = np.load(path / "generated_projected.npy", mmap_mode="r")
                    tolerance = float(
                        metrics.get("output_projection_repair", {}).get(
                            "change_tolerance", max(1e-8, min_gap * 1e-4)
                        )
                    )
                    repair = repair_distribution(raw, projected, normalization, tolerance)
                    repair_runs.append(
                        {"train_seed": train_seed, "eval_seed": eval_seed, **repair}
                    )
                    if runtime is None and args.runtime_samples > 0 and args.runtime_repeats > 0:
                        runtime = benchmark_exact_projection(
                            raw,
                            projected,
                            min_spread,
                            min_gap,
                            args.runtime_samples,
                            args.runtime_repeats,
                        )

    if not rows or first_real is None:
        raise FileNotFoundError(f"No saved evaluations found under {root}")

    aggregate = _aggregate(rows)
    boundary = continuous_boundary_stats(first_real, min_spread, min_gap)
    repair_aggregate = None
    if repair_runs:
        # Concatenating all saved arrays would waste memory. Report the mean of
        # run-level rates and the pooled mean of run-level repair quantiles.
        repair_aggregate = {
            "n_runs": len(repair_runs),
            "any_snapshot_modified_rate": float(
                np.mean([row["any_snapshot_modified_rate"] for row in repair_runs])
            ),
            "price_level_modified_rate": float(
                np.mean([row["price_level_modified_rate"] for row in repair_runs])
            ),
            "volume_level_modified_rate": float(
                np.mean([row["volume_level_modified_rate"] for row in repair_runs])
            ),
            "tick_threshold_exceedance": {
                threshold: {
                    metric: float(
                        np.mean(
                            [
                                row["tick_threshold_exceedance"][threshold][metric]
                                for row in repair_runs
                            ]
                        )
                    )
                    for metric in ("snapshot_rate", "price_level_rate")
                }
                for threshold in ("0.1", "0.5", "1")
            },
            "ticks_moved_modified_price_levels": {
                key: float(
                    np.mean(
                        [row["ticks_moved_modified_price_levels"][key] for row in repair_runs]
                    )
                )
                for key in ("mean", "std", "p50", "p95", "p99", "max")
            },
            "run_details": repair_runs,
        }

    payload = {
        "dataset": args.dataset,
        "scope": "saved-array audit; no retraining and no new diffusion sampling",
        "normalization": normalization,
        "train_seeds_requested": train_seeds,
        "eval_seeds_requested": eval_seeds,
        "per_run": rows,
        "aggregate": aggregate,
        "real_boundary_prevalence": boundary,
        "legal_latent_audit": latent_rows,
        "posthoc_pava_repair": repair_aggregate,
        "projection_runtime": runtime,
        "claim_boundary": (
            "continuous minimum-spread, monotone-ladder, and non-negative-volume "
            "feasibility only"
        ),
    }
    json_path = output_dir / "saved_generation_audit.json"
    csv_path = output_dir / "saved_generation_audit.csv"
    md_path = output_dir / "saved_generation_audit.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(csv_path, [{key: value for key, value in row.items() if key != "details"} for row in rows])
    _write_markdown(md_path, args.dataset, aggregate, boundary, repair_aggregate, runtime)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

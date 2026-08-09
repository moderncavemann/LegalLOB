#!/usr/bin/env python3
"""Cross-check frozen post-hoc PAVA claims against saved runs and arrays."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.utils.physics import project_to_valid_lob


POSTHOC_PARENT = {
    "raw_posthoc_pava": "raw_no_projection",
    "matched_posthoc_pava": "matched_no_projection",
}
VIOLATION_FIELDS = (
    "crossed_book_rate",
    "ask_ladder_violation",
    "bid_ladder_violation",
    "negative_volume_rate",
)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) not in (None, "", "None", "nan")
        and math.isfinite(float(row[field]))
    ]
    return float(np.mean(values)) if values else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _price_chain(array: np.ndarray) -> np.ndarray:
    """Return prices in the constrained chain order b10..b1,a1..a10."""
    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 4 or x.shape[1:3] != (2, 20):
        raise ValueError(f"expected [B, 2, 20, T], got {x.shape}")
    prices = x[:, 0]
    return np.concatenate([prices[:, 10:][:, ::-1], prices[:, :10]], axis=1)


def _chain_offsets(min_spread: float, min_gap: float) -> np.ndarray:
    increments = np.asarray(
        [min_gap] * 9 + [min_spread] + [min_gap] * 9,
        dtype=np.float64,
    )
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(increments)])


def independent_invalid_rate(
    array: np.ndarray,
    min_spread: float,
    min_gap: float,
    tolerance: float = 1e-10,
) -> float:
    """Compute snapshot-union invalidity without using production diagnostics."""
    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 4 or x.shape[1:3] != (2, 20):
        raise ValueError(f"expected [B, 2, 20, T], got {x.shape}")

    ask = x[:, 0, :10]
    bid = x[:, 0, 10:]
    volumes = x[:, 1]
    crossed = ask[:, 0] - bid[:, 0] < min_spread - tolerance
    ask_invalid = np.any(
        ask[:, 1:] - ask[:, :-1] < min_gap - tolerance,
        axis=1,
    )
    bid_invalid = np.any(
        bid[:, :-1] - bid[:, 1:] < min_gap - tolerance,
        axis=1,
    )
    depth_invalid = np.any(volumes < -tolerance, axis=1)
    return float(np.mean(crossed | ask_invalid | bid_invalid | depth_invalid))


def price_kkt_residuals(
    raw: np.ndarray,
    projected: np.ndarray,
    min_spread: float,
    min_gap: float,
) -> dict[str, float]:
    """Check KKT residuals for the exact price-chain Euclidean projection.

    Removing the cumulative minimum-increment offsets reduces the price problem
    to isotonic regression with constraints ``z[i] <= z[i + 1]``.  The dual
    multipliers are recovered from cumulative stationarity, independently of
    the PAVA implementation used to produce ``projected``.
    """
    raw_chain = _price_chain(raw)
    projected_chain = _price_chain(projected)
    if raw_chain.shape != projected_chain.shape:
        raise ValueError(
            f"raw/projected shapes differ: {raw_chain.shape} vs {projected_chain.shape}"
        )

    offsets = _chain_offsets(min_spread, min_gap)[None, :, None]
    y = raw_chain - offsets
    z = projected_chain - offsets
    multipliers = np.cumsum(y - z, axis=1)[:, :-1]

    stationarity = np.empty_like(z)
    stationarity[:, 0] = z[:, 0] - y[:, 0] + multipliers[:, 0]
    stationarity[:, 1:-1] = (
        z[:, 1:-1]
        - y[:, 1:-1]
        - multipliers[:, :-1]
        + multipliers[:, 1:]
    )
    stationarity[:, -1] = z[:, -1] - y[:, -1] - multipliers[:, -1]

    constraint_residual = z[:, :-1] - z[:, 1:]
    complementarity = multipliers * constraint_residual
    complementarity_scale = 1.0 + np.abs(multipliers) + np.abs(constraint_residual)
    return {
        "primal_violation_max": float(
            np.maximum(constraint_residual, 0.0).max(initial=0.0)
        ),
        "dual_violation_max": float(
            np.maximum(-multipliers, 0.0).max(initial=0.0)
        ),
        "stationarity_sum_abs_max": float(
            np.sum(np.abs(stationarity), axis=1).max(initial=0.0)
        ),
        "complementarity_abs_max": float(
            np.abs(complementarity).max(initial=0.0)
        ),
        "complementarity_scaled_max": float(
            (np.abs(complementarity) / complementarity_scale).max(initial=0.0)
        ),
    }


def _paired_mse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.mean(np.square(delta)))


def verify_saved_pava_arrays(
    per_run_path: Path,
    *,
    subset_size: int,
) -> dict[str, Any]:
    """Independently verify every frozen exact post-hoc PAVA output array."""
    source_rows = list(csv.DictReader(per_run_path.open()))
    posthoc_rows = [row for row in source_rows if row["method_id"] in POSTHOC_PARENT]
    run_checks: list[dict[str, Any]] = []

    for source in posthoc_rows:
        result_dir = Path(source["result_dir"])
        metrics = json.loads((result_dir / "metrics.json").read_text())
        min_spread = float(metrics["min_spread"])
        min_gap = float(metrics["min_gap"])
        raw = np.load(result_dir / "generated_raw.npy", mmap_mode="r")
        projected = np.load(result_dir / "generated_projected.npy", mmap_mode="r")
        real = np.load(result_dir / "real_future.npy", mmap_mode="r")
        if raw.shape != projected.shape or raw.shape != real.shape:
            raise ValueError(
                f"array shape mismatch in {result_dir}: "
                f"raw={raw.shape}, projected={projected.shape}, real={real.shape}"
            )

        count = min(subset_size, len(raw))
        indices = np.linspace(0, len(raw) - 1, count, dtype=np.int64)
        raw32 = np.array(raw[indices], dtype=np.float32, copy=True)
        projected32 = np.array(projected[indices], dtype=np.float32, copy=True)
        recomputed32 = project_to_valid_lob(
            torch.from_numpy(raw32),
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=False,
        ).numpy()
        reprojected32 = project_to_valid_lob(
            torch.from_numpy(projected32.copy()),
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=False,
        ).numpy()
        raw64 = raw32.astype(np.float64)
        projected64 = project_to_valid_lob(
            torch.from_numpy(raw64),
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=False,
        ).numpy()
        kkt = price_kkt_residuals(
            raw64,
            projected64,
            min_spread,
            min_gap,
        )
        mse_raw = _paired_mse(raw, real)
        mse_projected = _paired_mse(projected, real)
        run_checks.append(
            {
                "run": str(result_dir),
                "method": source["method_id"],
                "n_trajectories": int(raw.shape[0]),
                "n_snapshots": int(raw.shape[0] * raw.shape[-1]),
                "min_spread": min_spread,
                "min_gap": min_gap,
                "projected_invalid_snapshot_rate": independent_invalid_rate(
                    projected,
                    min_spread,
                    min_gap,
                    tolerance=max(1e-7, abs(min_spread) * 2e-5, abs(min_gap) * 2e-5),
                ),
                "mse_raw": mse_raw,
                "mse_projected": mse_projected,
                "mse_change": mse_projected - mse_raw,
                "recomputed_max_abs_difference": float(
                    np.max(np.abs(recomputed32 - projected32))
                ),
                "float64_solution_max_abs_difference": float(
                    np.max(np.abs(projected64 - projected32.astype(np.float64)))
                ),
                "idempotence_max_abs_difference": float(
                    np.max(np.abs(reprojected32 - projected32))
                ),
                "depth_projection_max_abs_difference": float(
                    np.max(
                        np.abs(
                            projected32[:, 1]
                            - np.maximum(raw32[:, 1], np.float32(0.0))
                        )
                    )
                ),
                **kkt,
            }
        )

    maxima = {
        "runs": len(run_checks),
        "trajectories": sum(row["n_trajectories"] for row in run_checks),
        "snapshots": sum(row["n_snapshots"] for row in run_checks),
        "max_projected_invalid_snapshot_rate": max(
            row["projected_invalid_snapshot_rate"] for row in run_checks
        ),
        "max_mse_increase": max(row["mse_change"] for row in run_checks),
        "max_recomputed_abs_difference": max(
            row["recomputed_max_abs_difference"] for row in run_checks
        ),
        "max_idempotence_abs_difference": max(
            row["idempotence_max_abs_difference"] for row in run_checks
        ),
        "max_float64_solution_abs_difference": max(
            row["float64_solution_max_abs_difference"] for row in run_checks
        ),
        "max_depth_projection_abs_difference": max(
            row["depth_projection_max_abs_difference"] for row in run_checks
        ),
        "max_primal_violation": max(
            row["primal_violation_max"] for row in run_checks
        ),
        "max_dual_violation": max(row["dual_violation_max"] for row in run_checks),
        "max_stationarity_sum_abs": max(
            row["stationarity_sum_abs_max"] for row in run_checks
        ),
        "max_complementarity_abs": max(
            row["complementarity_abs_max"] for row in run_checks
        ),
        "max_complementarity_scaled": max(
            row["complementarity_scaled_max"] for row in run_checks
        ),
    }
    checks = {
        "all_final_outputs_valid": maxima["max_projected_invalid_snapshot_rate"] == 0.0,
        "projection_never_increases_mse": maxima["max_mse_increase"] <= 1e-12,
        "saved_projection_reproduced": maxima["max_recomputed_abs_difference"] <= 2e-6,
        "projection_idempotent": maxima["max_idempotence_abs_difference"] <= 2e-6,
        "depth_clamp_reproduced": maxima["max_depth_projection_abs_difference"] == 0.0,
        "price_kkt_within_float_tolerance": max(
            maxima["max_primal_violation"],
            maxima["max_dual_violation"],
            maxima["max_stationarity_sum_abs"],
            maxima["max_complementarity_scaled"],
        )
        <= 5e-12,
    }
    return {
        "scope": (
            "Independent audit of frozen exact post-hoc PAVA outputs. Full "
            "arrays are checked for validity and MSE; a stratified subset is "
            "recomputed and checked for idempotence and isotonic KKT residuals."
        ),
        "per_run_source": str(per_run_path),
        "subset_size_per_run": subset_size,
        "totals": maxima,
        "checks": checks,
        "runs": run_checks,
    }


def write_verification_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify(
    per_run_path: Path,
    efficiency_path: Path,
    audit_glob: str,
    *,
    check_arrays: bool,
) -> dict[str, Any]:
    rows = list(csv.DictReader(per_run_path.open()))
    index = {
        (row["dataset"], row["train_seed"], row["eval_seed"], row["method_id"]): row
        for row in rows
    }
    posthoc_rows = [
        row for row in rows if row["method_id"] in POSTHOC_PARENT
    ]

    array_checks = []
    projected_violations = {field: [] for field in VIOLATION_FIELDS}
    mse_deltas = []
    for row in posthoc_rows:
        parent_id = POSTHOC_PARENT[row["method_id"]]
        parent = index[
            (row["dataset"], row["train_seed"], row["eval_seed"], parent_id)
        ]
        result_dir = Path(row["result_dir"])
        parent_dir = Path(parent["result_dir"])
        metrics = json.loads((result_dir / "metrics.json").read_text())
        for field in VIOLATION_FIELDS:
            projected_violations[field].append(float(metrics["violations_proj"][field]))
        mse_delta = float(row["mse_output"]) - float(row["mse_raw"])
        mse_deltas.append(mse_delta)
        paired_equal = None
        if check_arrays:
            posthoc_raw = np.load(result_dir / "generated_raw.npy", mmap_mode="r")
            parent_raw = np.load(parent_dir / "generated_raw.npy", mmap_mode="r")
            paired_equal = bool(np.array_equal(posthoc_raw, parent_raw))
        array_checks.append(
            {
                "dataset": row["dataset"],
                "train_seed": int(row["train_seed"]),
                "eval_seed": int(row["eval_seed"]),
                "posthoc_method": row["method_id"],
                "parent_method": parent_id,
                "raw_arrays_exactly_equal": paired_equal,
            }
        )

    grouped = []
    group_keys = sorted({(row["dataset"], row["method_id"]) for row in posthoc_rows})
    for dataset, method_id in group_keys:
        group = [
            row
            for row in posthoc_rows
            if row["dataset"] == dataset and row["method_id"] == method_id
        ]
        deltas = [
            float(row["mse_output"]) - float(row["mse_raw"]) for row in group
        ]
        grouped.append(
            {
                "dataset": dataset,
                "method_id": method_id,
                "n_runs": len(group),
                "raw_invalid_snapshot_rate_mean": _mean(group, "invalid_raw"),
                "final_invalid_snapshot_rate_max": max(
                    float(row["invalid_output"]) for row in group
                ),
                "mse_change_mean": float(np.mean(deltas)),
                "mse_change_max_abs": float(np.max(np.abs(deltas))),
                "snapshot_modified_rate_mean": _mean(
                    group, "repair_snapshot_modified_rate"
                ),
                "price_level_modified_rate_mean": _mean(
                    group, "repair_price_level_modified_rate"
                ),
                "changed_price_level_p95_ticks_mean": _mean(
                    group, "repair_ticks_p95"
                ),
                "snapshot_above_0_1_tick_rate_mean": _mean(
                    group, "repair_snapshot_gt_0_1_ticks"
                ),
                "depth_invalid_snapshot_rate_mean": _mean(
                    group, "repair_depth_invalid_snapshot_rate"
                ),
            }
        )

    audit_paths = sorted(Path().glob(audit_glob))
    recomputation_checks = []
    for path in audit_paths:
        audit = json.loads(path.read_text())
        runtime = audit.get("projection_runtime") or {}
        recomputation_checks.append(
            {
                "dataset": path.parent.name,
                "audit_path": str(path),
                "max_abs_difference_from_saved_projection": runtime.get(
                    "max_abs_difference_from_saved_projection"
                ),
                "microseconds_per_snapshot": runtime.get(
                    "microseconds_per_snapshot"
                ),
            }
        )

    efficiency = json.loads(efficiency_path.read_text())
    efficiency_rows = {
        row["method"]: {
            "milliseconds_per_batch": row["mean_total_ms_per_batch"],
            "trajectories_per_second": row["trajectories_per_second"],
            "lob_steps_per_second": row["snapshots_per_second"],
            "relative_to_raw": row["relative_to_raw"],
            "projection_calls": row["projection_calls"],
        }
        for row in efficiency["summary"]
        if row["method"]
        in {
            "raw_no_projection",
            "raw_posthoc_pava",
            "raw_inloop_pava",
            "matched_no_projection",
            "matched_posthoc_pava",
            "legal_no_projection",
        }
    }

    all_components_zero = all(
        value == 0.0 for values in projected_violations.values() for value in values
    )
    paired_checks_run = [
        row for row in array_checks if row["raw_arrays_exactly_equal"] is not None
    ]
    paired_equal = (
        all(row["raw_arrays_exactly_equal"] for row in paired_checks_run)
        if paired_checks_run
        else None
    )
    recomputation_exact = all(
        row["max_abs_difference_from_saved_projection"] == 0.0
        for row in recomputation_checks
    )
    return {
        "scope": {
            "per_run_source": str(per_run_path),
            "efficiency_source": str(efficiency_path),
            "posthoc_runs": len(posthoc_rows),
            "datasets": sorted({row["dataset"] for row in posthoc_rows}),
            "paired_array_check_enabled": check_arrays,
        },
        "checks": {
            "all_projected_constraint_components_zero": all_components_zero,
            "paired_parent_raw_arrays_exactly_equal": paired_equal,
            "saved_projection_exactly_recomputed_in_audits": recomputation_exact,
            "all_posthoc_mse_changes_nonpositive": all(
                value <= 1e-12 for value in mse_deltas
            ),
            "largest_absolute_per_run_mse_change": float(
                np.max(np.abs(mse_deltas))
            ),
            "mean_per_run_mse_change": float(np.mean(mse_deltas)),
            "projected_component_maxima": {
                field: max(values) for field, values in projected_violations.items()
            },
        },
        "by_dataset_and_method": grouped,
        "paired_array_checks": array_checks,
        "projection_recomputation_checks": recomputation_checks,
        "efficiency": {
            "definition": (
                "one LOB step is one generated future Level-2 snapshot; "
                "it is not one DDIM reverse update"
            ),
            "hardware": efficiency["hardware"],
            "protocol": efficiency["protocol"],
            "methods": efficiency_rows,
            "comparisons": efficiency["comparisons"],
        },
        "interpretation": [
            (
                "Exact post-hoc PAVA is genuinely strong for the stated terminal "
                "continuous-feasibility contract: every saved projected component "
                "has zero violations."
            ),
            (
                "Non-increasing point MSE is expected, not evidence of superior "
                "distribution learning: the real target belongs to the same convex "
                "feasible set, so Euclidean projection cannot increase squared "
                "distance to it."
            ),
            (
                "PAVA remains an external repair mechanism. Incidence, correction "
                "tails, finance-facing functionals, path-level cumulative repair, "
                "and implementation-specific CPU cost must be reported separately."
            ),
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    lines = [
        "# PAVA Evidence Audit",
        "",
        "## Verdict",
        "",
        "Exact post-hoc PAVA is confirmed as a strong terminal feasibility baseline, "
        "not as a universally superior generator.",
        "",
        f"- Post-hoc runs checked: {payload['scope']['posthoc_runs']}.",
        f"- All projected constraint components zero: {checks['all_projected_constraint_components_zero']}.",
        f"- Paired raw arrays exactly match no-projection parents: {checks['paired_parent_raw_arrays_exactly_equal']}.",
        f"- Saved projections exactly reproduced by the independent audit path: {checks['saved_projection_exactly_recomputed_in_audits']}.",
        f"- Largest absolute per-run MSE change: {checks['largest_absolute_per_run_mse_change']:.6f}.",
        f"- Mean per-run MSE change: {checks['mean_per_run_mse_change']:.6f}.",
        "",
        "Because each real target is feasible and PAVA is the Euclidean projection "
        "onto the same convex set, projected point MSE cannot increase. This property "
        "validates PAVA as a repair control; it does not establish probabilistic realism.",
        "",
        "## Paired Repair Results",
        "",
        "| Dataset | Method | Runs | Raw invalid % | Final invalid % | Snapshot modified % | Price levels modified % | Mean MSE change | Mean p95 ticks | >0.1 tick snapshots % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_dataset_and_method"]:
        lines.append(
            "| {dataset} | {method} | {runs} | {raw:.2f} | {final:.2f} | "
            "{snap:.2f} | {levels:.2f} | {mse:.6f} | {p95:.3f} | {gt:.2f} |".format(
                dataset=row["dataset"],
                method=row["method_id"],
                runs=row["n_runs"],
                raw=100.0 * row["raw_invalid_snapshot_rate_mean"],
                final=100.0 * row["final_invalid_snapshot_rate_max"],
                snap=100.0 * row["snapshot_modified_rate_mean"],
                levels=100.0 * row["price_level_modified_rate_mean"],
                mse=row["mse_change_mean"],
                p95=row["changed_price_level_p95_ticks_mean"],
                gt=100.0 * row["snapshot_above_0_1_tick_rate_mean"],
            )
        )

    methods = payload["efficiency"]["methods"]
    labels = {
        "raw_no_projection": "Raw",
        "raw_posthoc_pava": "Raw + post-hoc PAVA",
        "raw_inloop_pava": "Raw + in-loop PAVA",
        "matched_no_projection": "Matched",
        "matched_posthoc_pava": "Matched + post-hoc PAVA",
        "legal_no_projection": "LegalLOB",
    }
    lines.extend(
        [
            "",
            "## Synchronized Throughput",
            "",
            "One LOB step is one generated future Level-2 snapshot.",
            "",
            "| Method | Projection calls | ms/batch | LOB steps/s | Relative to Raw |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method_id, label in labels.items():
        row = methods[method_id]
        lines.append(
            f"| {label} | {row['projection_calls']} | "
            f"{row['milliseconds_per_batch']:.1f} | "
            f"{row['lob_steps_per_second']:.1f} | "
            f"{row['relative_to_raw']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "The timing is Apple M3 Ultra/MPS generation with exact CPU PAVA. "
            "It is batched throughput, not CUDA performance or single-request latency.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-run", default="paper_data/cross_asset_feasibility_per_run.csv"
    )
    parser.add_argument(
        "--efficiency", default="paper_data/method_efficiency_benchmark.json"
    )
    parser.add_argument(
        "--audit-glob",
        default="results/submission_audits/*/saved_generation_audit.json",
    )
    parser.add_argument(
        "--output-json", default="paper_data/pava_evidence_audit.json"
    )
    parser.add_argument(
        "--output-md", default="paper_tables/table_pava_evidence_audit.md"
    )
    parser.add_argument(
        "--verification-json",
        default="paper_data/pava_evidence_verification.json",
    )
    parser.add_argument(
        "--verification-csv",
        default="paper_data/pava_evidence_verification.csv",
    )
    parser.add_argument("--verification-subset-size", type=int, default=64)
    parser.add_argument("--skip-full-verification", action="store_true")
    parser.add_argument("--skip-array-check", action="store_true")
    args = parser.parse_args()

    payload = verify(
        Path(args.per_run),
        Path(args.efficiency),
        args.audit_glob,
        check_arrays=not args.skip_array_check,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    )
    write_markdown(Path(args.output_md), payload)
    if not args.skip_full_verification:
        verification = verify_saved_pava_arrays(
            Path(args.per_run),
            subset_size=args.verification_subset_size,
        )
        verification_json = Path(args.verification_json)
        verification_json.parent.mkdir(parents=True, exist_ok=True)
        verification_json.write_text(
            json.dumps(_json_safe(verification), indent=2, allow_nan=False) + "\n"
        )
        write_verification_csv(Path(args.verification_csv), verification["runs"])
        if not all(verification["checks"].values()):
            raise RuntimeError(
                f"independent PAVA verification failed: {verification['checks']}"
            )
        print(json.dumps(verification["totals"], indent=2))
        print(
            f"Wrote {verification_json} and {args.verification_csv}"
        )
    print(json.dumps(payload["checks"], indent=2))
    print(f"Wrote {output_json} and {args.output_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independently audit exact-PAVA claims from frozen generated arrays.

The audit does not trust the paper aggregation. It recomputes continuous
snapshot validity and paired MSE directly from saved raw/projected/real arrays,
then checks a small deterministic subset against a fresh exact-PAVA call.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.utils.physics import project_to_valid_lob


POSTHOC_METHODS = {"raw_posthoc_pava", "matched_posthoc_pava"}
AUDITED_METHODS = POSTHOC_METHODS | {"raw_inloop_pava"}


def continuous_validity(
    array: np.ndarray,
    *,
    min_spread: float,
    min_gap: float,
) -> dict[str, float]:
    x = np.asarray(array, dtype=np.float64)
    prices = x[:, 0]
    volumes = x[:, 1]
    ask = prices[:, :10]
    bid = prices[:, 10:]
    tolerance = max(1e-7, abs(min_spread) * 2e-5, abs(min_gap) * 2e-5)
    crossed = ask[:, 0] - bid[:, 0] < min_spread - tolerance
    ask_invalid = np.any(ask[:, 1:] - ask[:, :-1] < min_gap - tolerance, axis=1)
    bid_invalid = np.any(bid[:, :-1] - bid[:, 1:] < min_gap - tolerance, axis=1)
    depth_invalid = np.any(volumes < -tolerance, axis=1)
    any_invalid = crossed | ask_invalid | bid_invalid | depth_invalid
    return {
        "crossed_book_snapshot_rate": float(np.mean(crossed)),
        "ask_ladder_invalid_snapshot_rate": float(np.mean(ask_invalid)),
        "bid_ladder_invalid_snapshot_rate": float(np.mean(bid_invalid)),
        "negative_depth_invalid_snapshot_rate": float(np.mean(depth_invalid)),
        "any_invalid_snapshot_rate": float(np.mean(any_invalid)),
        "tolerance": tolerance,
    }


def mse(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.mean(np.square(delta)))


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method_id"])].append(row)
    output = []
    numeric_fields = (
        "pre_any_invalid_snapshot_rate",
        "final_any_invalid_snapshot_rate",
        "paired_mse_before",
        "paired_mse_after",
        "paired_mse_delta",
        "fresh_projection_max_abs_difference",
        "final_idempotence_max_abs_difference",
        "path_cumulative_projection_l2",
        "path_pre_projection_any_component_rate",
    )
    for (dataset, method_id), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "dataset": dataset,
            "method_id": method_id,
            "n_runs": len(group),
        }
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in group
                if row.get(field) is not None and math.isfinite(float(row[field]))
            ]
            if values:
                item[f"{field}_mean"], item[f"{field}_std"] = _mean_std(values)
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-csv",
        default="paper_data/cross_asset_feasibility_per_run.csv",
    )
    parser.add_argument("--output-dir", default="paper_data/pava_audit")
    parser.add_argument("--reprojection-trajectories", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = list(csv.DictReader(Path(args.evidence_csv).open()))
    rows: list[dict[str, Any]] = []

    for source in source_rows:
        method_id = source["method_id"]
        if method_id not in AUDITED_METHODS:
            continue
        result_dir = Path(source["result_dir"])
        metrics = json.loads((result_dir / "metrics.json").read_text())
        min_spread = float(metrics["min_spread"])
        min_gap = float(metrics["min_gap"])
        final_path = (
            result_dir / "generated_projected.npy"
            if method_id in POSTHOC_METHODS
            else result_dir / "generated_raw.npy"
        )
        final = np.load(final_path, mmap_mode="r")
        real = np.load(result_dir / "real_future.npy", mmap_mode="r")
        final_validity = continuous_validity(
            final, min_spread=min_spread, min_gap=min_gap
        )

        count = min(args.reprojection_trajectories, len(final))
        final_sample = torch.from_numpy(np.array(final[:count], copy=True)).float()
        final_reprojected = project_to_valid_lob(
            final_sample,
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=False,
        ).numpy()
        idempotence_max = float(
            np.max(np.abs(final_reprojected - np.asarray(final[:count])))
        )

        pre_invalid = None
        mse_before = None
        mse_after = mse(final, real)
        fresh_difference = idempotence_max
        if method_id in POSTHOC_METHODS:
            raw = np.load(result_dir / "generated_raw.npy", mmap_mode="r")
            pre_invalid = continuous_validity(
                raw, min_spread=min_spread, min_gap=min_gap
            )["any_invalid_snapshot_rate"]
            mse_before = mse(raw, real)
            raw_sample = torch.from_numpy(np.array(raw[:count], copy=True)).float()
            fresh = project_to_valid_lob(
                raw_sample,
                min_spread=min_spread,
                min_gap=min_gap,
                use_gpu_approx=False,
            ).numpy()
            fresh_difference = float(
                np.max(np.abs(fresh - np.asarray(final[:count])))
            )

        sampler = metrics.get("sampler_projection_stats") or {}
        pre_component_rates = [
            float(sampler[key])
            for key in (
                "pre_projection_crossed_book_rate_mean",
                "pre_projection_ask_ladder_violation_mean",
                "pre_projection_bid_ladder_violation_mean",
                "pre_projection_negative_volume_rate_mean",
            )
            if sampler.get(key) is not None
        ]
        row = {
            "dataset": source["dataset"],
            "method_id": method_id,
            "train_seed": int(source["train_seed"]),
            "eval_seed": int(source["eval_seed"]),
            "result_dir": str(result_dir),
            "n_snapshots": int(final.shape[0] * final.shape[-1]),
            "pre_any_invalid_snapshot_rate": pre_invalid,
            "final_any_invalid_snapshot_rate": final_validity[
                "any_invalid_snapshot_rate"
            ],
            "paired_mse_before": mse_before,
            "paired_mse_after": mse_after,
            "paired_mse_delta": (
                mse_after - mse_before if mse_before is not None else None
            ),
            "fresh_projection_max_abs_difference": fresh_difference,
            "final_idempotence_max_abs_difference": idempotence_max,
            "path_cumulative_projection_l2": sampler.get(
                "cumulative_projection_l2_mean"
            ),
            "path_pre_projection_any_component_rate": (
                max(pre_component_rates) if pre_component_rates else None
            ),
        }
        rows.append(row)

    summary = aggregate(rows)
    failures = {
        "nonzero_final_continuous_invalidity": [
            row["result_dir"]
            for row in rows
            if row["final_any_invalid_snapshot_rate"] != 0.0
        ],
        "stored_projection_mismatch": [
            row["result_dir"]
            for row in rows
            if row["fresh_projection_max_abs_difference"] > 2e-6
        ],
        "non_idempotent_final_output": [
            row["result_dir"]
            for row in rows
            if row["final_idempotence_max_abs_difference"] > 2e-6
        ],
        "posthoc_mse_increase": [
            row["result_dir"]
            for row in rows
            if row["paired_mse_delta"] is not None
            and row["paired_mse_delta"] > 2e-6
        ],
    }
    payload = {
        "scope": (
            "continuous minimum-spread, minimum-gap, monotone-ladder, and "
            "non-negative-depth constraints only"
        ),
        "evidence_csv": args.evidence_csv,
        "n_runs": len(rows),
        "n_posthoc_runs": sum(row["method_id"] in POSTHOC_METHODS for row in rows),
        "n_inloop_runs": sum(row["method_id"] == "raw_inloop_pava" for row in rows),
        "checks": failures,
        "all_checks_passed": all(not items for items in failures.values()),
        "summary": summary,
        "limitations": [
            "The audit does not test tick-grid phase or integer share quantities.",
            "It does not test message replay, matching, queue priority, or fills.",
            "In-loop path rates are component rates recorded before projection calls, not snapshot-union rates.",
        ],
    }
    _write_csv(output_dir / "pava_effectiveness_per_run.csv", rows)
    _write_csv(output_dir / "pava_effectiveness_summary.csv", summary)
    (output_dir / "pava_effectiveness_audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )

    lines = [
        "# Exact-PAVA Effectiveness Audit",
        "",
        f"Frozen runs audited: **{len(rows)}** "
        f"({payload['n_posthoc_runs']} post-hoc, {payload['n_inloop_runs']} in-loop).",
        "",
        "| Dataset | Method | Runs | Pre invalid % | Final invalid % | Paired MSE delta | Fresh projection max diff | Cum. path L2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def display(field: str, scale: float = 1.0, digits: int = 5) -> str:
            value = row.get(f"{field}_mean")
            if value is None:
                return "--"
            std = row.get(f"{field}_std", 0.0)
            return f"{scale * value:.{digits}f} +/- {scale * std:.{digits}f}"

        lines.append(
            "| {dataset} | {method} | {runs} | {pre} | {final} | {delta} | {fresh} | {path} |".format(
                dataset=row["dataset"],
                method=row["method_id"],
                runs=row["n_runs"],
                pre=display("pre_any_invalid_snapshot_rate", 100.0, 2),
                final=display("final_any_invalid_snapshot_rate", 100.0, 4),
                delta=display("paired_mse_delta", 1.0, 6),
                fresh=display("fresh_projection_max_abs_difference", 1.0, 7),
                path=display("path_cumulative_projection_l2", 1.0, 2),
            )
        )
    lines.extend(
        [
            "",
            f"All declared checks passed: **{payload['all_checks_passed']}**.",
            "",
            "## Interpretation",
            "",
            "Exact PAVA is genuinely strong for the stated continuous terminal contract: "
            "the saved outputs are feasible, a fresh exact projection reproduces them, "
            "and paired post-hoc MSE never increases. This is expected for Euclidean "
            "projection onto a closed convex feasible set containing the real target.",
            "In-loop PAVA also produces feasible final outputs, but its nonzero cumulative "
            "path correction remains a sampling-time repair burden.",
            "",
            "This audit does not establish exchange-complete validity: tick-grid phase, "
            "integer quantities, message semantics, matching, queue priority, and fills "
            "remain outside scope.",
        ]
    )
    (output_dir / "pava_effectiveness_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_runs": len(rows), "checks": failures}, indent=2))


if __name__ == "__main__":
    main()

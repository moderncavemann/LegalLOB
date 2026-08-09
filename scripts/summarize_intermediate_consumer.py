#!/usr/bin/env python3
"""Aggregate and validate the intermediate-state consumer experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHOD_LABELS = {
    "legal_lob": "LegalLOB",
    "matched_posthoc_pava": "Matched+Post",
    "matched_repeated_pava": "Matched+Repeated PAVA",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_cells(
    root: Path,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]], list[str]]:
    """Load only the exact manifest cell set and audit hashes/orphans."""
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get("config") or {}
    expected = {
        (int(train_seed), int(eval_seed))
        for train_seed in config.get("train_seeds", [])
        for eval_seed in config.get("eval_seeds", [])
    }
    errors: list[str] = []
    entries = manifest.get("cells") or []
    entry_keys: list[tuple[int, int]] = []
    loaded: list[tuple[Path, dict[str, Any]]] = []
    manifest_paths: set[Path] = set()
    for entry in entries:
        try:
            key = (int(entry["train_seed"]), int(entry["eval_seed"]))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"malformed manifest cell: {error}")
            continue
        entry_keys.append(key)
        if entry.get("status") != "completed":
            errors.append(f"cell {key} status={entry.get('status')}")
            continue
        path = Path(entry.get("path", ""))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        manifest_paths.add(path)
        if not path.exists():
            errors.append(f"cell {key} missing file {path}")
            continue
        expected_hash = entry.get("sha256")
        actual_hash = sha256_file(path)
        if not expected_hash or actual_hash != expected_hash:
            errors.append(f"cell {key} sha256 mismatch")
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") != "completed":
            errors.append(f"cell {key} payload status={payload.get('status')}")
            continue
        payload_key = (int(payload["train_seed"]), int(payload["eval_seed"]))
        if payload_key != key:
            errors.append(f"cell {key} payload key={payload_key}")
            continue
        if set(payload.get("methods", {})) != set(METHOD_LABELS):
            errors.append(f"cell {key} method set mismatch")
            continue
        loaded.append((path, payload))

    observed = set(entry_keys)
    if len(entry_keys) != len(observed):
        errors.append("duplicate train/eval cell entries")
    if observed != expected:
        errors.append(
            f"manifest cell set mismatch missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    if len(entries) != len(expected):
        errors.append(
            f"manifest cell count {len(entries)} != expected {len(expected)}"
        )
    orphan_paths = {
        path.resolve()
        for path in root.glob("train_seed*/eval_seed*/consumer_stats.json")
    } - manifest_paths
    if orphan_paths:
        errors.append(
            "orphan consumer files: " + ", ".join(map(str, sorted(orphan_paths)))
        )
    return manifest, loaded, errors


def _get(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _relative_error(generated: Any, real: Any) -> float | None:
    if generated is None or real is None:
        return None
    generated = float(generated)
    real = float(real)
    if not math.isfinite(generated) or not math.isfinite(real) or abs(real) < 1e-12:
        return None
    return abs(generated - real) / abs(real)


def collect_rows(
    cells: list[tuple[Path, dict[str, Any]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    method_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    matched_delta_rows: list[dict[str, Any]] = []
    for path, cell in cells:
        train_seed = int(cell["train_seed"])
        eval_seed = int(cell["eval_seed"])
        cell_method_rows: dict[str, dict[str, Any]] = {}
        for method_id, method in cell["methods"].items():
            horizon = max(
                int(key.removeprefix("step_"))
                for key in method["rollout"]
                if key.startswith("step_")
            )
            rollout = method["rollout"][f"step_{horizon}"]
            row = {
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "method_id": method_id,
                "method": METHOD_LABELS[method_id],
                "horizon": horizon,
                "n_histories": method["n_histories"],
                "anytime_pre_frame_requests": _get(
                    method, "anytime_consumer.pre.frame_requests"
                ),
                "anytime_pre_block_requests": _get(
                    method, "anytime_consumer.pre.block_requests"
                ),
                "anytime_delivered_frame_requests": _get(
                    method, "anytime_consumer.delivered.frame_requests"
                ),
                "anytime_delivered_block_requests": _get(
                    method, "anytime_consumer.delivered.block_requests"
                ),
                "endpoint_pre_frame_requests": _get(
                    method, "endpoint_consumer.pre.frame_requests"
                ),
                "endpoint_pre_block_requests": _get(
                    method, "endpoint_consumer.pre.block_requests"
                ),
                "endpoint_delivered_frame_requests": _get(
                    method, "endpoint_consumer.delivered.frame_requests"
                ),
                "endpoint_delivered_block_requests": _get(
                    method, "endpoint_consumer.delivered.block_requests"
                ),
                "anytime_pre_frame_acceptance": _get(
                    method, "anytime_consumer.pre.frame_acceptance_rate"
                ),
                "anytime_pre_block_acceptance": _get(
                    method, "anytime_consumer.pre.block_acceptance_rate"
                ),
                "anytime_delivered_frame_acceptance": _get(
                    method, "anytime_consumer.delivered.frame_acceptance_rate"
                ),
                "anytime_delivered_block_acceptance": _get(
                    method, "anytime_consumer.delivered.block_acceptance_rate"
                ),
                "anytime_delivered_zero_tolerance_frame_acceptance": _get(
                    method,
                    "anytime_consumer.delivered.zero_tolerance_frame_acceptance_rate",
                ),
                "anytime_pre_completed_trajectory_rate": _get(
                    method,
                    "anytime_consumer.pre_completed_without_rejection_rate",
                ),
                "anytime_delivered_completed_trajectory_rate": _get(
                    method,
                    "anytime_consumer.delivered_completed_without_rejection_rate",
                ),
                "endpoint_pre_frame_acceptance": _get(
                    method, "endpoint_consumer.pre.frame_acceptance_rate"
                ),
                "endpoint_delivered_frame_acceptance": _get(
                    method, "endpoint_consumer.delivered.frame_acceptance_rate"
                ),
                "endpoint_delivered_block_acceptance": _get(
                    method, "endpoint_consumer.delivered.block_acceptance_rate"
                ),
                "endpoint_delivered_zero_tolerance_frame_acceptance": _get(
                    method,
                    "endpoint_consumer.delivered.zero_tolerance_frame_acceptance_rate",
                ),
                "projection_calls_per_trajectory": _get(
                    method, "projection.calls_per_trajectory"
                ),
                "state_changing_call_rate": _get(
                    method, "projection.state_changing_call_rate"
                ),
                "projection_l2_per_call": _get(
                    method, "projection.mean_correction_per_call"
                ),
                "projection_l2_cumulative": _get(
                    method,
                    "projection.mean_cumulative_correction_per_trajectory",
                ),
                "risk_alarm_disagreement": _get(
                    method,
                    "anytime_consumer.algebraic_preprojection_shift_diagnostic.risk_alarm_disagreement_rate",
                ),
                "repair_slippage_shift_ticks": _get(
                    method,
                    "anytime_consumer.algebraic_preprojection_shift_diagnostic.market_order_slippage_abs_shift_mean_ticks",
                ),
                "repair_liquidity_shift_target_units": _get(
                    method,
                    "anytime_consumer.algebraic_preprojection_shift_diagnostic.liquidity_5tick_abs_shift_mean_target_units",
                ),
                "feedback_roundtrip_l2_per_state": _get(
                    method, "feedback_roundtrip_diagnostic.mean_l2_per_state"
                ),
                "feedback_roundtrip_l2_max": _get(
                    method, "feedback_roundtrip_diagnostic.max_l2"
                ),
                "market_order_slippage_w1_ticks": _get(
                    method, "risk_fidelity.market_order_slippage_w1_ticks"
                ),
                "liquidity_5tick_w1_target_units": _get(
                    method, "risk_fidelity.liquidity_5tick_w1_target_units"
                ),
                "risk_alarm_rate_abs_error": _get(
                    method, "risk_fidelity.risk_alarm_rate_abs_error"
                ),
                "underfill_rate_abs_error": _get(
                    method, "risk_fidelity.underfill_rate_abs_error"
                ),
                "spread_relative_abs_error": _relative_error(
                    rollout.get("spread_mean"), rollout.get("spread_real_mean")
                ),
                "volume_relative_abs_error": _relative_error(
                    rollout.get("volume_mean"), rollout.get("volume_real_mean")
                ),
                "return_std_relative_abs_error": _relative_error(
                    rollout.get("mid_return_std"),
                    rollout.get("mid_return_real_std"),
                ),
                "temporal_correlation": rollout.get("temporal_corr"),
                "runtime_seconds": _get(method, "runtime.seconds"),
                "generated_frames_per_second": _get(
                    method, "runtime.generated_frames_per_second"
                ),
                "cell_path": str(path),
            }
            method_rows.append(row)
            cell_method_rows[method_id] = row

        for horizon_key, values in cell["paired_matched_path_divergence"].items():
            paired_rows.append(
                {
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "horizon": int(horizon_key.removeprefix("step_")),
                    **values,
                    "cell_path": str(path),
                }
            )
        post = cell_method_rows["matched_posthoc_pava"]
        repeated = cell_method_rows["matched_repeated_pava"]
        delta_fields = (
            "anytime_delivered_frame_acceptance",
            "endpoint_delivered_frame_acceptance",
            "projection_calls_per_trajectory",
            "state_changing_call_rate",
            "projection_l2_per_call",
            "projection_l2_cumulative",
            "market_order_slippage_w1_ticks",
            "liquidity_5tick_w1_target_units",
            "risk_alarm_rate_abs_error",
            "spread_relative_abs_error",
            "volume_relative_abs_error",
            "return_std_relative_abs_error",
            "temporal_correlation",
            "runtime_seconds",
            "generated_frames_per_second",
        )
        delta_row: dict[str, Any] = {
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "comparison": "matched_repeated_minus_post",
            "cell_path": str(path),
        }
        for field in delta_fields:
            left = repeated.get(field)
            right = post.get(field)
            delta_row[f"{field}_delta"] = (
                float(left) - float(right)
                if left is not None
                and right is not None
                and math.isfinite(float(left))
                and math.isfinite(float(right))
                else None
            )
        matched_delta_rows.append(delta_row)
    return method_rows, paired_rows, matched_delta_rows


def hierarchical_aggregate(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    excluded = {
        "train_seed",
        "eval_seed",
        "cell_path",
        *group_fields,
    }
    numeric_fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    groups = sorted({tuple(row[field] for field in group_fields) for row in rows})
    output: list[dict[str, Any]] = []
    for group_key in groups:
        group = [
            row
            for row in rows
            if tuple(row[field] for field in group_fields) == group_key
        ]
        train_groups: dict[int, list[dict[str, Any]]] = {}
        for row in group:
            train_groups.setdefault(int(row["train_seed"]), []).append(row)
        item: dict[str, Any] = {
            field: value for field, value in zip(group_fields, group_key, strict=True)
        }
        item["n_training_seeds"] = len(train_groups)
        item["n_eval_seeds_per_training_seed"] = sorted(
            len(seed_rows) for seed_rows in train_groups.values()
        )
        for metric in numeric_fields:
            checkpoint_values: list[float] = []
            eval_counts: list[int] = []
            missing_cells = 0
            for seed_rows in train_groups.values():
                eval_values = [
                    float(row[metric])
                    for row in seed_rows
                    if row.get(metric) is not None
                    and math.isfinite(float(row[metric]))
                ]
                eval_counts.append(len(eval_values))
                missing_cells += len(seed_rows) - len(eval_values)
                if eval_values:
                    checkpoint_values.append(float(np.mean(eval_values)))
            if checkpoint_values:
                item[f"{metric}_mean"] = float(np.mean(checkpoint_values))
                item[f"{metric}_std"] = (
                    float(np.std(checkpoint_values, ddof=1))
                    if len(checkpoint_values) > 1
                    else None
                )
            item[f"{metric}_n_training_seeds"] = len(checkpoint_values)
            item[f"{metric}_n_eval_seeds_per_training_seed"] = sorted(eval_counts)
            item[f"{metric}_missing_cells"] = missing_cells
        output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(row: dict[str, Any], field: str, digits: int = 3) -> str:
    mean = row.get(f"{field}_mean")
    if mean is None:
        return "--"
    std = row.get(f"{field}_std")
    if std is None:
        return f"{mean:.{digits}f} (SD n/a)"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _fmt_percent(row: dict[str, Any], field: str, digits: int = 1) -> str:
    mean = row.get(f"{field}_mean")
    if mean is None:
        return "--"
    std = row.get(f"{field}_std")
    if std is None:
        return f"{100 * mean:.{digits}f} (SD n/a)"
    return f"{100 * mean:.{digits}f} +/- {100 * std:.{digits}f}"


def markdown(
    summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    delta_summary: list[dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    rows = {row["method_id"]: row for row in summary}
    lines = [
        "# Intermediate-State Snapshot-Risk Consumer",
        "",
        f"Implementation verified: **{validation['implementation_verified']}**; "
        f"narrow interface GO: **{validation['narrow_scientific_go']}**.",
        "",
        "Evaluation seeds are averaged within checkpoints; mean +/- sample SD is",
        "reported across independently trained checkpoints. Anytime rates use",
        "predicted-clean frame calls, which are correlated reverse-path requests,",
        "not independent market observations. Rejected calls remain in the audit",
        "denominator while generation continues. The consumer is fail-closed for",
        "risk outputs; this is not queue-aware execution or causal control.",
        "",
        "| Method | Anytime pre frame % | Anytime pre block % | Anytime delivered frame % | Endpoint delivered frame % | PAVA calls/path | State-changing % | Cumulative L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in METHOD_LABELS:
        row = rows[method_id]
        lines.append(
            "| {method} | {pre} | {pre_block} | {delivered} | {endpoint} | {calls} | {changing} | {cumulative} |".format(
                method=row["method"],
                pre=_fmt_percent(row, "anytime_pre_frame_acceptance", 2),
                pre_block=_fmt_percent(
                    row, "anytime_pre_block_acceptance", 2
                ),
                delivered=_fmt_percent(
                    row, "anytime_delivered_frame_acceptance", 2
                ),
                endpoint=_fmt_percent(
                    row, "endpoint_delivered_frame_acceptance", 2
                ),
                calls=_fmt(row, "projection_calls_per_trajectory", 1),
                changing=_fmt_percent(row, "state_changing_call_rate", 1),
                cumulative=_fmt(row, "projection_l2_cumulative", 3),
            )
        )
    lines.extend(
        [
            "",
            "## Endpoint Risk/Fidelity Guardrails",
            "",
            "| Method | Slippage W1 | Liq@5 W1 | Alarm-rate error % | Spread rel. error % | Volume rel. error % | Temporal corr. |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method_id in METHOD_LABELS:
        row = rows[method_id]
        lines.append(
            f"| {row['method']} | {_fmt(row, 'market_order_slippage_w1_ticks', 3)} | "
            f"{_fmt(row, 'liquidity_5tick_w1_target_units', 3)} | "
            f"{_fmt_percent(row, 'risk_alarm_rate_abs_error', 2)} | "
            f"{_fmt_percent(row, 'spread_relative_abs_error', 2)} | "
            f"{_fmt_percent(row, 'volume_relative_abs_error', 2)} | "
            f"{_fmt(row, 'temporal_correlation', 3)} |"
        )
    lines.extend(
        [
            "",
            "## Paired Matched Path Divergence",
            "",
            "| Horizon | Trajectory L2 | P95 L2 | Risk-alarm disagreement % |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(paired_summary, key=lambda item: item["horizon"]):
        lines.append(
            f"| {row['horizon']} | {_fmt(row, 'trajectory_l2_mean', 3)} | "
            f"{_fmt(row, 'trajectory_l2_p95', 3)} | "
            f"{_fmt_percent(row, 'risk_alarm_disagreement_rate', 2)} |"
        )
    if delta_summary:
        delta = delta_summary[0]
        lines.extend(
            [
                "",
                "## Matched+Repeated minus Matched+Post",
                "",
                "Signed deltas are paired within each train/evaluation cell before hierarchical aggregation.",
                "",
                "| Calls/path | Cumulative L2 | Slippage W1 | Liq@5 W1 | Alarm error % | Spread error % | Volume error % |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                "| {calls} | {correction} | {slip} | {liq} | {alarm} | {spread} | {volume} |".format(
                    calls=_fmt(
                        delta, "projection_calls_per_trajectory_delta", 1
                    ),
                    correction=_fmt(
                        delta, "projection_l2_cumulative_delta", 3
                    ),
                    slip=_fmt(
                        delta, "market_order_slippage_w1_ticks_delta", 3
                    ),
                    liq=_fmt(
                        delta, "liquidity_5tick_w1_target_units_delta", 3
                    ),
                    alarm=_fmt_percent(
                        delta, "risk_alarm_rate_abs_error_delta", 2
                    ),
                    spread=_fmt_percent(
                        delta, "spread_relative_abs_error_delta", 2
                    ),
                    volume=_fmt_percent(
                        delta, "volume_relative_abs_error_delta", 2
                    ),
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _isclose(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1e-5, abs_tol=5e-7
        )
    except (TypeError, ValueError):
        return False


def reference_reproduction(
    manifest: dict[str, Any], method_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    receipt = manifest.get("provenance", {}).get("reference_rollout", {})
    path = Path(receipt.get("path", ""))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return {"passed": False, "errors": [f"missing reference CSV {path}"]}
    if sha256_file(path) != receipt.get("sha256"):
        return {"passed": False, "errors": ["reference CSV hash mismatch"]}
    reference_rows = {
        (int(row["train_seed"]), int(row["eval_seed"])): row
        for row in csv.DictReader(path.open())
        if row.get("method_id") == "matched_posthoc_pava"
        and int(row["horizon"]) == int(manifest["config"]["horizon"])
    }
    field_map = {
        "projection_l2_cumulative": "path_cumulative_projection_l2",
        "projection_calls_per_trajectory": "path_projection_calls",
        "spread_relative_abs_error": "spread_relative_abs_error",
        "volume_relative_abs_error": "volume_relative_abs_error",
        "return_std_relative_abs_error": "return_std_relative_abs_error",
        "temporal_correlation": "temporal_correlation",
    }
    comparisons = 0
    max_abs_difference: dict[str, float] = {field: 0.0 for field in field_map}
    max_abs_difference["endpoint_pre_invalid_rate"] = 0.0
    errors: list[str] = []
    for row in method_rows:
        if row["method_id"] != "matched_posthoc_pava":
            continue
        key = (int(row["train_seed"]), int(row["eval_seed"]))
        reference = reference_rows.get(key)
        if reference is None:
            errors.append(f"missing reference Matched+Post row {key}")
            continue
        comparisons += 1
        for current_field, reference_field in field_map.items():
            current = row.get(current_field)
            expected = reference.get(reference_field)
            if not _isclose(current, expected):
                errors.append(
                    f"{key} {current_field}: current={current} reference={expected}"
                )
            if current is not None and expected not in (None, ""):
                max_abs_difference[current_field] = max(
                    max_abs_difference[current_field],
                    abs(float(current) - float(expected)),
                )
        current_invalid = 1.0 - float(row["endpoint_pre_frame_acceptance"])
        expected_invalid = float(reference["path_pre_projection_invalid_rate"])
        if not _isclose(current_invalid, expected_invalid):
            errors.append(
                f"{key} endpoint_pre_invalid_rate: current={current_invalid} "
                f"reference={expected_invalid}"
            )
        max_abs_difference["endpoint_pre_invalid_rate"] = max(
            max_abs_difference["endpoint_pre_invalid_rate"],
            abs(current_invalid - expected_invalid),
        )
    expected_comparisons = len(manifest["config"]["train_seeds"]) * len(
        manifest["config"]["eval_seeds"]
    )
    if comparisons != expected_comparisons:
        errors.append(
            f"reference comparisons {comparisons} != {expected_comparisons}"
        )
    return {
        "passed": not errors,
        "comparisons": comparisons,
        "tolerance": {"relative": 1e-5, "absolute": 5e-7},
        "max_abs_difference": max_abs_difference,
        "errors": errors,
    }


def validate(
    manifest: dict[str, Any],
    cells: list[tuple[Path, dict[str, Any]]],
    manifest_errors: list[str],
    method_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    delta_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    config = manifest["config"]
    train_seeds = [int(value) for value in config["train_seeds"]]
    eval_seeds = [int(value) for value in config["eval_seeds"]]
    expected_cells = len(train_seeds) * len(eval_seeds)
    chunks = math.ceil(int(config["horizon"]) / int(config["chunk_size"]))
    n_histories = int(config["n_histories"])
    expected_denominators = {
        "anytime_pre_frame_requests": n_histories
        * chunks
        * int(config["ddim_steps"])
        * int(config["pred_len"]),
        "anytime_pre_block_requests": n_histories
        * chunks
        * int(config["ddim_steps"]),
        "anytime_delivered_frame_requests": n_histories
        * chunks
        * int(config["ddim_steps"])
        * int(config["pred_len"]),
        "anytime_delivered_block_requests": n_histories
        * chunks
        * int(config["ddim_steps"]),
        "endpoint_pre_frame_requests": n_histories * int(config["horizon"]),
        "endpoint_pre_block_requests": n_histories * chunks,
        "endpoint_delivered_frame_requests": n_histories
        * int(config["horizon"]),
        "endpoint_delivered_block_requests": n_histories * chunks,
    }
    cell_gate_failures: list[str] = []
    rng_failures: list[str] = []
    for _, payload in cells:
        key = f"train={payload['train_seed']} eval={payload['eval_seed']}"
        validation = payload.get("validation") or {}
        checks = validation.get("implementation_checks")
        if not isinstance(checks, dict):
            cell_gate_failures.append(f"{key} missing implementation_checks")
        else:
            for name, passed in checks.items():
                if not passed:
                    cell_gate_failures.append(f"{key} {name}")
        rng_hashes = (payload.get("post_model_load_rng_sha256") or {}).values()
        if len(set(rng_hashes)) != 1:
            rng_failures.append(f"{key} post-load RNG mismatch")

    denominator_failures: list[str] = []
    expected_calls = {
        "legal_lob": 0.0,
        "matched_posthoc_pava": float(chunks),
        "matched_repeated_pava": float(chunks * int(config["ddim_steps"])),
    }
    for row in method_rows:
        key = (
            f"train={row['train_seed']} eval={row['eval_seed']} "
            f"method={row['method_id']}"
        )
        if int(row["n_histories"]) != n_histories:
            denominator_failures.append(f"{key} n_histories")
        for field, expected in expected_denominators.items():
            if int(row[field]) != expected:
                denominator_failures.append(
                    f"{key} {field}={row[field]} expected={expected}"
                )
        if not math.isclose(
            float(row["projection_calls_per_trajectory"]),
            expected_calls[row["method_id"]],
            abs_tol=1e-12,
        ):
            denominator_failures.append(f"{key} projection calls")

    source_hash_failures: list[str] = []
    for relative, expected_hash in manifest.get("provenance", {}).get(
        "sources_sha256", {}
    ).items():
        path = Path(relative)
        if not path.exists() or sha256_file(path) != expected_hash:
            source_hash_failures.append(relative)

    required_metrics = (
        "anytime_pre_frame_acceptance",
        "anytime_pre_block_acceptance",
        "anytime_delivered_frame_acceptance",
        "endpoint_delivered_frame_acceptance",
        "projection_calls_per_trajectory",
        "state_changing_call_rate",
        "projection_l2_cumulative",
        "market_order_slippage_w1_ticks",
        "liquidity_5tick_w1_target_units",
        "risk_alarm_rate_abs_error",
        "spread_relative_abs_error",
        "volume_relative_abs_error",
        "return_std_relative_abs_error",
        "temporal_correlation",
    )
    summary_failures: list[str] = []
    if len(summary) != len(METHOD_LABELS):
        summary_failures.append(f"method summary rows={len(summary)}")
    for row in summary:
        if row.get("n_training_seeds") != len(train_seeds):
            summary_failures.append(f"{row.get('method_id')} training seed count")
        if row.get("n_eval_seeds_per_training_seed") != [
            len(eval_seeds)
        ] * len(train_seeds):
            summary_failures.append(f"{row.get('method_id')} eval seed counts")
        for metric in required_metrics:
            if row.get(f"{metric}_n_training_seeds") != len(train_seeds):
                summary_failures.append(
                    f"{row.get('method_id')} {metric} training denominator"
                )
            if row.get(f"{metric}_missing_cells") != 0:
                summary_failures.append(
                    f"{row.get('method_id')} {metric} missing cells"
                )
    if not paired_summary:
        summary_failures.append("missing paired path summary")
    if len(delta_summary) != 1:
        summary_failures.append("missing signed Matched delta summary")

    by_train: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in method_rows:
        by_train.setdefault(int(row["train_seed"]), {}).setdefault(
            row["method_id"], []
        ).append(row)
    per_training_seed_science: dict[str, dict[str, bool]] = {}
    for train_seed in train_seeds:
        grouped = by_train.get(train_seed, {})

        def mean(method: str, field: str) -> float:
            return float(np.mean([row[field] for row in grouped[method]]))

        per_training_seed_science[str(train_seed)] = {
            "legal_anytime_delivered_acceptance": math.isclose(
                mean("legal_lob", "anytime_delivered_frame_acceptance"),
                1.0,
                abs_tol=1e-12,
            ),
            "matched_post_anytime_gap_over_1pct": mean(
                "matched_posthoc_pava", "anytime_pre_block_acceptance"
            )
            < 0.99,
            "matched_post_endpoint_delivered_acceptance": math.isclose(
                mean(
                    "matched_posthoc_pava",
                    "endpoint_delivered_frame_acceptance",
                ),
                1.0,
                abs_tol=1e-12,
            ),
            "repeated_anytime_delivered_acceptance": math.isclose(
                mean(
                    "matched_repeated_pava",
                    "anytime_delivered_frame_acceptance",
                ),
                1.0,
                abs_tol=1e-12,
            ),
            "repeated_nonzero_repair_over_1pct": mean(
                "matched_repeated_pava", "state_changing_call_rate"
            )
            > 0.01,
        }
    science_checks = {
        name: all(checks[name] for checks in per_training_seed_science.values())
        for name in next(iter(per_training_seed_science.values()))
    }
    reproduction = reference_reproduction(manifest, method_rows)
    implementation_verified = (
        manifest.get("status") == "completed"
        and len(cells) == expected_cells
        and len(method_rows) == expected_cells * len(METHOD_LABELS)
        and not manifest_errors
        and not cell_gate_failures
        and not rng_failures
        and not denominator_failures
        and not source_hash_failures
        and not summary_failures
        and reproduction["passed"]
    )
    science_checks_pass = all(science_checks.values())
    return {
        "manifest_status": manifest.get("status"),
        "expected_cells": expected_cells,
        "completed_cells": len(cells),
        "method_rows": len(method_rows),
        "manifest_integrity_errors": manifest_errors,
        "cell_implementation_gate_failures": cell_gate_failures,
        "rng_pairing_failures": rng_failures,
        "denominator_failures": denominator_failures,
        "source_hash_failures": source_hash_failures,
        "summary_completeness_failures": summary_failures,
        "reference_reproduction": reproduction,
        "per_training_seed_science_checks": per_training_seed_science,
        "science_checks": science_checks,
        "science_checks_pass": science_checks_pass,
        "implementation_verified": implementation_verified,
        "narrow_scientific_go": implementation_verified and science_checks_pass,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/intermediate_consumer_goog_event"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_root
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, cells, manifest_errors = load_manifest_cells(args.results_root)
    method_rows, paired_rows, delta_rows = collect_rows(cells)
    if not method_rows:
        raise FileNotFoundError(f"No completed consumer cells under {args.results_root}")
    method_summary = hierarchical_aggregate(
        method_rows, group_fields=("method_id", "method")
    )
    paired_summary = hierarchical_aggregate(paired_rows, group_fields=("horizon",))
    delta_summary = hierarchical_aggregate(
        delta_rows, group_fields=("comparison",)
    )
    validation = validate(
        manifest,
        cells,
        manifest_errors,
        method_rows,
        method_summary,
        paired_summary,
        delta_summary,
    )
    (output_dir / "intermediate_consumer_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    if not validation["implementation_verified"]:
        print(json.dumps(validation, indent=2))
        return 1
    _write_csv(output_dir / "intermediate_consumer_per_run.csv", method_rows)
    _write_csv(output_dir / "intermediate_consumer_summary.csv", method_summary)
    _write_csv(output_dir / "intermediate_consumer_paired_per_run.csv", paired_rows)
    _write_csv(
        output_dir / "intermediate_consumer_paired_summary.csv", paired_summary
    )
    _write_csv(
        output_dir / "intermediate_consumer_matched_delta_per_run.csv",
        delta_rows,
    )
    _write_csv(
        output_dir / "intermediate_consumer_matched_delta_summary.csv",
        delta_summary,
    )
    (output_dir / "intermediate_consumer_summary.json").write_text(
        json.dumps(method_summary, indent=2) + "\n"
    )
    (output_dir / "intermediate_consumer_paired_summary.json").write_text(
        json.dumps(paired_summary, indent=2) + "\n"
    )
    (output_dir / "intermediate_consumer_matched_delta_summary.json").write_text(
        json.dumps(delta_summary, indent=2) + "\n"
    )
    (output_dir / "intermediate_consumer_summary.md").write_text(
        markdown(method_summary, paired_summary, delta_summary, validation)
    )
    print((output_dir / "intermediate_consumer_summary.md").read_text())
    print(json.dumps(validation, indent=2))
    return 0 if validation["implementation_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

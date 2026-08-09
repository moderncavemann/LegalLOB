#!/usr/bin/env python3
"""Validate and summarize the multi-asset event-time robustness study.

The input is the four per-asset ``run_intermediate_consumer_panel.py`` output
directories.  The script recomputes exact rejection counts from the frozen
cell payloads, validates the complete 4 x 3 x 3 evaluation grid, and keeps
implementation integrity separate from the registered scientific outcome.

No cross-asset inferential statistic is computed: the four assets are fixed
replication panels, not random replicates.  Means and sample standard
deviations for continuous metrics use the existing hierarchical aggregation
(evaluation seeds within checkpoint, then the three training checkpoints).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_intermediate_consumer import (  # noqa: E402
    collect_rows,
    hierarchical_aggregate,
    load_manifest_cells,
    sha256_file,
    validate,
)


P4_ASSETS = ("AAPL", "AMZN", "GOOG", "INTC")
P4_DATASETS = {asset: f"PUBLIC_ITCH_2019_SAMPLES_{asset}_event" for asset in P4_ASSETS}
METRIC_CLARIFICATION_PATH = (
    REPO_ROOT / "experiments/multi_asset_event_time/metric_definition.md"
)
METRIC_CLARIFICATION_SHA256 = (
    "52f378090c996abf49fce85154724d28bfbb3dc2fe72c65f8868d32948b64b4d"
)
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = (0, 1, 2)
METHODS = (
    "legal_lob",
    "matched_posthoc_pava",
    "matched_repeated_pava",
)
METHOD_LABELS = {
    "legal_lob": "LegalLOB",
    "matched_posthoc_pava": "Matched+Post",
    "matched_repeated_pava": "Repeated PAVA",
}

EXPECTED_CELLS_PER_ASSET = 9
EXPECTED_METHOD_ROWS_PER_ASSET = 27
EXPECTED_EVALUATION_STAGES = 21
EXPECTED_PER_CELL = {
    "paths": 640,
    "pre_projection_frames": 1_280_000,
    "intermediate_blocks": 128_000,
    "endpoint_frames": 64_000,
}
EXPECTED_PER_ASSET = {
    key: value * EXPECTED_CELLS_PER_ASSET for key, value in EXPECTED_PER_CELL.items()
}
EXPECTED_CALLS_PER_PATH = {
    "legal_lob": 0.0,
    "matched_posthoc_pava": 10.0,
    "matched_repeated_pava": 200.0,
}
REFERENCE_ENDPOINT_INVALIDITY_ABS_TOLERANCE = 5e-5
REFERENCE_PARITY_POLICY = {
    "non_endpoint_relative_tolerance": 1e-5,
    "non_endpoint_absolute_tolerance": 5e-7,
    "endpoint_invalidity_absolute_tolerance": (
        REFERENCE_ENDPOINT_INVALIDITY_ABS_TOLERANCE
    ),
    "endpoint_frames_per_cell": 64_000,
    "endpoint_tolerance_equivalent_frames_per_cell": 3.2,
    "rule": (
        "Only endpoint pre-invalidity receives the declared 5e-5 diagnostic "
        "parity tolerance; every other reference error remains disallowed."
    ),
}

# These are scientific acceptance checks despite their historical placement
# under ``implementation_checks`` in each consumer cell.  Any other failed
# item in that dictionary remains an implementation-integrity failure.
SCIENCE_ONLY_CELL_CHECKS = frozenset(
    {
        "legal_lob_endpoint_delivered_acceptance",
        "matched_posthoc_pava_endpoint_delivered_acceptance",
        "matched_repeated_pava_endpoint_delivered_acceptance",
        "repeated_anytime_pava_output_acceptance",
    }
)


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    direction: str = "down"
    digits: int = 3
    exact_count: bool = False


METRICS = (
    Metric(
        "pre_projection_invalidity",
        "Pre-projection invalidity",
        exact_count=True,
    ),
    Metric(
        "intermediate_consumer_rejection",
        "Intermediate consumer rejection",
        exact_count=True,
    ),
    Metric("endpoint_invalidity", "Endpoint invalidity", exact_count=True),
    Metric("projection_calls_per_path", "Projection calls/path", digits=1),
    Metric(
        "correction_per_call",
        "Decoded-state L2 per scheduled projection call",
        digits=5,
    ),
    Metric("cumulative_repair", "Cumulative repair", digits=4),
    Metric(
        "market_order_slippage_w1_ticks",
        "Market-order slippage W1 (ticks)",
        digits=3,
    ),
)
METRIC_BY_KEY = {metric.key: metric for metric in METRICS}

COUNT_SOURCE_FIELDS = {
    "pre_projection_invalidity": (
        "anytime_pre_frame_acceptance",
        "anytime_pre_frame_requests",
    ),
    "intermediate_consumer_rejection": (
        "anytime_pre_block_acceptance",
        "anytime_pre_block_requests",
    ),
    "endpoint_invalidity": (
        "endpoint_delivered_frame_acceptance",
        "endpoint_delivered_frame_requests",
    ),
}
CONTINUOUS_SOURCE_FIELDS = {
    "projection_calls_per_path": "projection_calls_per_trajectory",
    "correction_per_call": "projection_l2_per_call",
    "cumulative_repair": "projection_l2_cumulative",
    "market_order_slippage_w1_ticks": "market_order_slippage_w1_ticks",
}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _close(left: Any, right: Any, *, abs_tol: float = 1e-10) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=abs_tol)
    )


def _exact_rejections(
    row: dict[str, Any], acceptance_field: str, request_field: str
) -> tuple[int, int]:
    """Recover an exact integer numerator from rate plus exact denominator."""

    rate = row.get(acceptance_field)
    requests = row.get(request_field)
    if not _finite(rate) or not _finite(requests):
        raise ValueError(f"nonfinite {acceptance_field} or {request_field}")
    rate = float(rate)
    request_float = float(requests)
    request_count = int(round(request_float))
    if not _close(request_float, request_count, abs_tol=1e-8):
        raise ValueError(f"non-integer request denominator {requests}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"acceptance rate outside [0,1]: {rate}")
    accepted_float = rate * request_count
    accepted_count = int(round(accepted_float))
    # Stored rates originate from exact counts.  This tolerance permits only
    # ordinary JSON floating-point roundoff, never a materially rounded rate.
    if abs(accepted_float - accepted_count) > 1e-6:
        raise ValueError(
            f"{acceptance_field} does not encode an exact count: "
            f"rate={rate} requests={request_count}"
        )
    return request_count - accepted_count, request_count


def _sample_sd(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _hierarchical_metric(
    rows: list[dict[str, Any]], field: str
) -> tuple[float, float, list[float]]:
    checkpoint_means: list[float] = []
    for train_seed in TRAIN_SEEDS:
        values = [
            float(row[field])
            for row in rows
            if int(row["train_seed"]) == train_seed and _finite(row.get(field))
        ]
        if len(values) != len(EVAL_SEEDS):
            raise ValueError(
                f"{field}: train seed {train_seed} has {len(values)} evaluation values"
            )
        checkpoint_means.append(sum(values) / len(values))
    return (
        sum(checkpoint_means) / len(checkpoint_means),
        _sample_sd(checkpoint_means),
        checkpoint_means,
    )


def _exact_count_metric(rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    acceptance_field, request_field = COUNT_SOURCE_FIELDS[metric_key]
    rejected = 0
    requests = 0
    by_training_seed: dict[str, dict[str, Any]] = {}
    for train_seed in TRAIN_SEEDS:
        seed_rejected = 0
        seed_requests = 0
        seed_rows = [row for row in rows if int(row["train_seed"]) == train_seed]
        if len(seed_rows) != len(EVAL_SEEDS):
            raise ValueError(
                f"{metric_key}: train seed {train_seed} row denominator mismatch"
            )
        for row in seed_rows:
            item_rejected, item_requests = _exact_rejections(
                row, acceptance_field, request_field
            )
            seed_rejected += item_rejected
            seed_requests += item_requests
        by_training_seed[str(train_seed)] = {
            "invalid_or_rejected": seed_rejected,
            "requests": seed_requests,
            "rate": seed_rejected / seed_requests,
        }
        rejected += seed_rejected
        requests += seed_requests
    return {
        "invalid_or_rejected": rejected,
        "requests": requests,
        "rate": rejected / requests,
        "by_training_seed": by_training_seed,
    }


def build_asset_summary(
    asset: str,
    method_rows: list[dict[str, Any]],
    *,
    input_root: Path | str,
    input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build exact counts, hierarchical metrics, and registered study checks."""

    expected_grid = {
        (train_seed, eval_seed, method)
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
        for method in METHODS
    }
    observed_grid: list[tuple[int, int, str]] = []
    for row in method_rows:
        observed_grid.append(
            (
                int(row["train_seed"]),
                int(row["eval_seed"]),
                str(row["method_id"]),
            )
        )
    if (
        len(observed_grid) != EXPECTED_METHOD_ROWS_PER_ASSET
        or len(observed_grid) != len(set(observed_grid))
        or set(observed_grid) != expected_grid
    ):
        raise ValueError(f"{asset}: method row grid is not the exact 3x3x3 denominator")

    output_rows: list[dict[str, Any]] = []
    by_method: dict[str, dict[str, Any]] = {}
    for method_id in METHODS:
        rows = [row for row in method_rows if row["method_id"] == method_id]
        item: dict[str, Any] = {
            "asset": asset,
            "dataset": P4_DATASETS[asset],
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
            "n_training_seeds": len(TRAIN_SEEDS),
            "n_evaluation_seeds_per_training_seed": len(EVAL_SEEDS),
            "n_paths": EXPECTED_PER_ASSET["paths"],
        }
        for metric_key in COUNT_SOURCE_FIELDS:
            item[metric_key] = _exact_count_metric(rows, metric_key)
        for metric_key, source_field in CONTINUOUS_SOURCE_FIELDS.items():
            mean, std, checkpoint_means = _hierarchical_metric(rows, source_field)
            if metric_key == "correction_per_call" and _close(
                (
                    item["projection_calls_per_path"]["mean"]
                    if isinstance(item.get("projection_calls_per_path"), dict)
                    else 0.0
                ),
                0.0,
            ):
                # This branch is normally reached after calls/path is inserted
                # below; the second explicit pass after the loop is the guard.
                value: dict[str, Any] | None = None
            else:
                value = {
                    "mean": mean,
                    "sample_sd_across_training_seeds": std,
                    "training_seed_means": checkpoint_means,
                }
            item[metric_key] = value
        calls = item["projection_calls_per_path"]
        assert isinstance(calls, dict)
        if _close(calls["mean"], 0.0):
            item["correction_per_call"] = None
        item["exact_projection_calls"] = int(
            round(float(calls["mean"]) * EXPECTED_PER_ASSET["paths"])
        )
        output_rows.append(item)
        by_method[method_id] = item

    # Registered strict mechanism rule from the frozen study protocol.
    legal = by_method["legal_lob"]
    post = by_method["matched_posthoc_pava"]
    repeated = by_method["matched_repeated_pava"]
    repeated_rows = [
        row for row in method_rows if row["method_id"] == "matched_repeated_pava"
    ]
    repeated_delivered_rejections = 0
    repeated_delivered_requests = 0
    for row in repeated_rows:
        rejected, requests = _exact_rejections(
            row,
            "anytime_delivered_frame_acceptance",
            "anytime_delivered_frame_requests",
        )
        repeated_delivered_rejections += rejected
        repeated_delivered_requests += requests
    changing_mean, changing_sd, changing_seed_means = _hierarchical_metric(
        repeated_rows, "state_changing_call_rate"
    )
    checks = {
        "legal_zero_pre_projection_invalidity": legal["pre_projection_invalidity"][
            "invalid_or_rejected"
        ]
        == 0,
        "legal_zero_intermediate_consumer_rejection": legal[
            "intermediate_consumer_rejection"
        ]["invalid_or_rejected"]
        == 0,
        "legal_zero_endpoint_invalidity": legal["endpoint_invalidity"][
            "invalid_or_rejected"
        ]
        == 0,
        "legal_zero_projection_calls": legal["exact_projection_calls"] == 0,
        "matched_post_nonzero_intermediate_rejection": post[
            "intermediate_consumer_rejection"
        ]["invalid_or_rejected"]
        > 0,
        "matched_post_zero_delivered_endpoint_invalidity": post["endpoint_invalidity"][
            "invalid_or_rejected"
        ]
        == 0,
        "repeated_pava_zero_delivered_intermediate_invalidity": (
            repeated_delivered_rejections == 0
        ),
        "repeated_pava_has_state_changing_calls": changing_mean > 0.0,
    }
    fidelity_values = {
        method: by_method[method]["market_order_slippage_w1_ticks"]["mean"]
        for method in METHODS
    }
    fidelity_all_finite = all(_finite(value) for value in fidelity_values.values())
    fidelity_warning = fidelity_all_finite and fidelity_values["legal_lob"] > max(
        fidelity_values["matched_posthoc_pava"],
        fidelity_values["matched_repeated_pava"],
    )
    scientific = {
        "mechanism_checks": checks,
        "mechanism_go": all(checks.values()),
        "repeated_delivered_intermediate_exact_counts": {
            "invalid": repeated_delivered_rejections,
            "requests": repeated_delivered_requests,
        },
        "repeated_state_changing_call_rate": {
            "mean": changing_mean,
            "sample_sd_across_training_seeds": changing_sd,
            "training_seed_means": changing_seed_means,
        },
        "fidelity_guardrail_all_finite": fidelity_all_finite,
        "fidelity_warning": fidelity_warning,
        "fidelity_warning_rule": (
            "LegalLOB slippage W1 is strictly worse than both Matched comparators"
        ),
    }
    return {
        "asset": asset,
        "dataset": P4_DATASETS[asset],
        "input_root": str(Path(input_root).resolve()),
        "input_hashes": dict(sorted((input_hashes or {}).items())),
        "method_rows": output_rows,
        "scientific_outcome": scientific,
    }


def _validate_config(asset: str, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    config = manifest.get("config") or {}
    expected = {
        "dataset": P4_DATASETS[asset],
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "hist_len": 50,
        "pred_len": 10,
        "chunk_size": 10,
        "horizon": 100,
        "ddim_steps": 20,
        "batch_size": 128,
        "n_batches": 5,
        "n_histories": 640,
        "action_mode": "hold",
        "weights": "ema",
        "checkpoint_name": "final.pt",
        "device": "mps",
    }
    for field, expected_value in expected.items():
        if config.get(field) != expected_value:
            errors.append(
                f"{asset}: config {field}={config.get(field)!r} "
                f"expected={expected_value!r}"
            )
    return errors


def _validate_final_checkpoints(
    asset: str, cells: list[tuple[Path, dict[str, Any]]]
) -> list[str]:
    errors: list[str] = []
    observed_hashes: dict[Path, str] = {}
    for _, payload in cells:
        cell_key = f"train={payload.get('train_seed')} eval={payload.get('eval_seed')}"
        train_seed = payload.get("train_seed")
        checkpoints = payload.get("checkpoints") or {}
        for variant in ("legal_lob", "matched_lob"):
            checkpoint = checkpoints.get(variant) or {}
            path = Path(str(checkpoint.get("path", "")))
            resolved = path.resolve()
            if path.name != "final.pt":
                errors.append(
                    f"{asset}: {cell_key} {variant} checkpoint is not final.pt"
                )
            expected_path_tokens = (
                "p4_public_itch_event_time_20260807_v1",
                asset,
                f"train_seed{train_seed}",
                variant,
            )
            if any(token not in resolved.parts for token in expected_path_tokens):
                errors.append(
                    f"{asset}: {cell_key} {variant} checkpoint path is outside "
                    "the frozen event-time asset/seed/representation root"
                )
            if checkpoint.get("epoch") != 50:
                errors.append(
                    f"{asset}: {cell_key} {variant} checkpoint epoch "
                    f"{checkpoint.get('epoch')!r} != 50"
                )
            digest = checkpoint.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(
                    f"{asset}: {cell_key} {variant} malformed checkpoint hash"
                )
            elif not resolved.is_file():
                errors.append(
                    f"{asset}: {cell_key} {variant} checkpoint file is missing"
                )
            else:
                if resolved not in observed_hashes:
                    observed_hashes[resolved] = sha256_file(resolved)
                actual = observed_hashes[resolved]
                if actual != digest:
                    errors.append(
                        f"{asset}: {cell_key} {variant} checkpoint SHA-256 mismatch"
                    )
        signatures = {
            (checkpoints.get(variant) or {}).get("config_sha256")
            for variant in ("legal_lob", "matched_lob")
        }
        if len(signatures) != 1 or None in signatures:
            errors.append(f"{asset}: {cell_key} backbone config signature mismatch")
    return errors


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_pairing_receipts(
    asset: str,
    manifest: dict[str, Any],
    cells: list[tuple[Path, dict[str, Any]]],
) -> list[str]:
    errors: list[str] = []
    noise_by_eval: dict[int, list[list[str]]] = {seed: [] for seed in EVAL_SEEDS}
    pairing_receipts: dict[str, list[str]] = {
        "history_starts_sha256": [],
        "real_futures_sha256": [],
    }
    expected_starts_hash = ((manifest.get("provenance") or {}).get("data") or {}).get(
        "start_indices_sha256"
    )
    if not _valid_sha256(expected_starts_hash):
        errors.append(f"{asset}: manifest history-start hash is missing or malformed")
    for _, payload in cells:
        cell_key = f"train={payload.get('train_seed')} eval={payload.get('eval_seed')}"
        eval_seed = int(payload.get("eval_seed"))
        post_load = payload.get("post_model_load_rng_sha256") or {}
        if set(post_load) != set(METHODS) or len(set(post_load.values())) != 1:
            errors.append(f"{asset}: {cell_key} post-model-load RNG is not paired")
        elif not all(_valid_sha256(value) for value in post_load.values()):
            errors.append(f"{asset}: {cell_key} malformed post-model-load RNG hash")
        noise = payload.get("paired_initial_noise_sha256")
        if (
            not isinstance(noise, list)
            or len(noise) != 50
            or not all(_valid_sha256(value) for value in noise)
        ):
            errors.append(
                f"{asset}: {cell_key} initial-noise receipt is not 50 SHA-256 values"
            )
        else:
            noise_by_eval[eval_seed].append(noise)
        for field in pairing_receipts:
            value = payload.get(field)
            if not _valid_sha256(value):
                errors.append(f"{asset}: {cell_key} missing or malformed {field}")
            else:
                pairing_receipts[field].append(value)
        if (
            _valid_sha256(expected_starts_hash)
            and payload.get("history_starts_sha256") != expected_starts_hash
        ):
            errors.append(
                f"{asset}: {cell_key} history starts do not match manifest receipt"
            )
        if payload.get("real_futures_shape") != [640, 2, 20, 100]:
            errors.append(f"{asset}: {cell_key} real-future shape mismatch")
        if payload.get("real_futures_dtype") != "float32":
            errors.append(f"{asset}: {cell_key} real-future dtype mismatch")
    for eval_seed, receipts in noise_by_eval.items():
        if len(receipts) != len(TRAIN_SEEDS) or any(
            receipt != receipts[0] for receipt in receipts[1:]
        ):
            errors.append(
                f"{asset}: eval seed {eval_seed} initial noise differs across checkpoints"
            )
    for field, values in pairing_receipts.items():
        if len(values) != EXPECTED_CELLS_PER_ASSET or len(set(values)) != 1:
            errors.append(f"{asset}: {field} differs or is missing across nine cells")
    return errors


def _science_only_failure(failure: str) -> bool:
    return any(failure.endswith(name) for name in SCIENCE_ONLY_CELL_CHECKS)


def classify_reference_reproduction(reproduction: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen endpoint-only diagnostic tolerance to base parity."""

    raw_errors = [str(error) for error in reproduction.get("errors") or []]
    endpoint_field = "endpoint_pre_invalid_rate"
    endpoint_errors = [error for error in raw_errors if endpoint_field in error]
    non_endpoint_errors = [error for error in raw_errors if endpoint_field not in error]
    maximum = (reproduction.get("max_abs_difference") or {}).get(endpoint_field)
    base_tolerance = reproduction.get("tolerance") or {}
    base_tolerance_matches_policy = (
        base_tolerance.get("relative")
        == REFERENCE_PARITY_POLICY["non_endpoint_relative_tolerance"]
        and base_tolerance.get("absolute")
        == REFERENCE_PARITY_POLICY["non_endpoint_absolute_tolerance"]
    )
    base_flag_consistent = (reproduction.get("passed") is True) == (
        len(raw_errors) == 0
    )
    endpoint_within_tolerance = _finite(maximum) and float(maximum) <= (
        REFERENCE_ENDPOINT_INVALIDITY_ABS_TOLERANCE
    )
    comparisons = reproduction.get("comparisons")
    verified = (
        comparisons == EXPECTED_CELLS_PER_ASSET
        and not non_endpoint_errors
        and endpoint_within_tolerance
        and base_tolerance_matches_policy
        and base_flag_consistent
    )
    return {
        "verified": verified,
        "base_passed": reproduction.get("passed") is True,
        "comparisons": comparisons,
        "expected_comparisons": EXPECTED_CELLS_PER_ASSET,
        "base_tolerance": base_tolerance,
        "base_tolerance_matches_policy": base_tolerance_matches_policy,
        "base_pass_flag_consistent_with_errors": base_flag_consistent,
        "endpoint_max_abs_difference": maximum,
        "endpoint_within_declared_tolerance": endpoint_within_tolerance,
        "endpoint_only_base_errors_reclassified": endpoint_errors,
        "non_endpoint_errors": non_endpoint_errors,
        "policy": REFERENCE_PARITY_POLICY,
    }


def validate_asset(asset: str, consumer_root: Path) -> dict[str, Any]:
    """Validate one raw consumer panel and return an auditable asset record."""

    errors: list[str] = []
    input_hashes: dict[str, str] = {}
    try:
        manifest, cells, manifest_errors = load_manifest_cells(consumer_root)
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError) as exc:
        return {
            "asset": asset,
            "dataset": P4_DATASETS[asset],
            "input_root": str(consumer_root.resolve()),
            "integrity_verified": False,
            "integrity_errors": [f"{asset}: cannot load consumer panel: {exc}"],
            "method_rows": [],
            "scientific_outcome": None,
            "input_hashes": {},
        }
    manifest_path = consumer_root / "run_manifest.json"
    input_hashes[str(manifest_path.resolve())] = sha256_file(manifest_path)
    for path, _ in cells:
        input_hashes[str(path.resolve())] = sha256_file(path)

    errors.extend(f"{asset}: {error}" for error in manifest_errors)
    errors.extend(_validate_config(asset, manifest))
    errors.extend(_validate_final_checkpoints(asset, cells))
    errors.extend(_validate_pairing_receipts(asset, manifest, cells))

    try:
        method_rows, paired_rows, delta_rows = collect_rows(cells)
        method_summary = hierarchical_aggregate(
            method_rows, group_fields=("method_id", "method")
        )
        paired_summary = hierarchical_aggregate(paired_rows, group_fields=("horizon",))
        delta_summary = hierarchical_aggregate(delta_rows, group_fields=("comparison",))
        base = validate(
            manifest,
            cells,
            manifest_errors,
            method_rows,
            method_summary,
            paired_summary,
            delta_summary,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError) as exc:
        errors.append(f"{asset}: cannot aggregate or validate cells: {exc}")
        method_rows = []
        base = {}

    reference_parity: dict[str, Any] | None = None
    if base:
        if base.get("manifest_status") != "completed":
            errors.append(f"{asset}: manifest is not completed")
        if base.get("completed_cells") != EXPECTED_CELLS_PER_ASSET:
            errors.append(f"{asset}: completed cell denominator is not 9")
        if base.get("method_rows") != EXPECTED_METHOD_ROWS_PER_ASSET:
            errors.append(f"{asset}: method row denominator is not 27")
        for field in (
            "manifest_integrity_errors",
            "rng_pairing_failures",
            "denominator_failures",
            "source_hash_failures",
            "summary_completeness_failures",
        ):
            for failure in base.get(field) or []:
                errors.append(f"{asset}: {field}: {failure}")
        reproduction = base.get("reference_reproduction") or {}
        reference_parity = classify_reference_reproduction(reproduction)
        if reference_parity["verified"] is not True:
            errors.append(
                f"{asset}: Matched+Post reference reproduction failed: "
                f"non_endpoint_errors={reference_parity['non_endpoint_errors']} "
                "endpoint_max_abs_difference="
                f"{reference_parity['endpoint_max_abs_difference']!r}"
            )
        for failure in base.get("cell_implementation_gate_failures") or []:
            if not _science_only_failure(str(failure)):
                errors.append(f"{asset}: implementation cell failure: {failure}")

    asset_summary: dict[str, Any] | None = None
    if method_rows:
        try:
            asset_summary = build_asset_summary(
                asset,
                method_rows,
                input_root=consumer_root,
                input_hashes=input_hashes,
            )
            for row in asset_summary["method_rows"]:
                method_id = row["method_id"]
                expected_calls = EXPECTED_CALLS_PER_PATH[method_id]
                observed_calls = row["projection_calls_per_path"]["mean"]
                if not _close(observed_calls, expected_calls, abs_tol=1e-12):
                    errors.append(
                        f"{asset}: {method_id} calls/path={observed_calls} "
                        f"expected={expected_calls}"
                    )
                for metric_key, expected_denominator in (
                    (
                        "pre_projection_invalidity",
                        EXPECTED_PER_ASSET["pre_projection_frames"],
                    ),
                    (
                        "intermediate_consumer_rejection",
                        EXPECTED_PER_ASSET["intermediate_blocks"],
                    ),
                    (
                        "endpoint_invalidity",
                        EXPECTED_PER_ASSET["endpoint_frames"],
                    ),
                ):
                    if row[metric_key]["requests"] != expected_denominator:
                        errors.append(
                            f"{asset}: {method_id} {metric_key} denominator "
                            f"{row[metric_key]['requests']} != {expected_denominator}"
                        )
                for metric_key in CONTINUOUS_SOURCE_FIELDS:
                    value = row[metric_key]
                    if value is None:
                        if not (
                            metric_key == "correction_per_call"
                            and _close(observed_calls, 0.0)
                        ):
                            errors.append(
                                f"{asset}: {method_id} unexpected N/A {metric_key}"
                            )
                    elif not _finite(value.get("mean")) or not _finite(
                        value.get("sample_sd_across_training_seeds")
                    ):
                        errors.append(f"{asset}: {method_id} nonfinite {metric_key}")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"{asset}: exact-count summary failed: {exc}")

    if asset_summary is None:
        return {
            "asset": asset,
            "dataset": P4_DATASETS[asset],
            "input_root": str(consumer_root.resolve()),
            "integrity_verified": False,
            "integrity_errors": sorted(set(errors)),
            "method_rows": [],
            "scientific_outcome": None,
            "reference_parity": reference_parity,
            "input_hashes": dict(sorted(input_hashes.items())),
        }
    asset_summary["integrity_verified"] = not errors
    asset_summary["integrity_errors"] = sorted(set(errors))
    asset_summary["legacy_base_validation"] = base
    asset_summary["reference_parity"] = reference_parity
    asset_summary["scientific_outcome"]["classification"] = (
        "not_classified_due_to_integrity_failure"
        if errors
        else (
            "verified_go"
            if asset_summary["scientific_outcome"]["mechanism_go"]
            else "verified_scientific_no_go"
        )
    )
    return asset_summary


def _metric_value(row: dict[str, Any], metric: Metric) -> float | None:
    value = row.get(metric.key)
    if value is None:
        return None
    if metric.exact_count:
        return float(value["rate"])
    return float(value["mean"])


def dense_ranks(
    rows: Iterable[dict[str, Any]], metric: Metric
) -> dict[str, int | None]:
    """Dense ranks within one asset; ties at numerical tolerance share rank."""

    rows = list(rows)
    values = [
        value for row in rows if (value := _metric_value(row, metric)) is not None
    ]
    unique: list[float] = []
    for value in sorted(values, reverse=metric.direction == "up"):
        if not unique or not _close(value, unique[-1]):
            unique.append(value)
    output: dict[str, int | None] = {}
    for row in rows:
        value = _metric_value(row, metric)
        output[row["method_id"]] = (
            None
            if value is None
            else next(
                rank
                for rank, candidate in enumerate(unique, start=1)
                if _close(value, candidate)
            )
        )
    return output


def _format_mean_sd(value: dict[str, Any], digits: int) -> str:
    return (
        f"{float(value['mean']):.{digits}f} $\\pm$ "
        f"{float(value['sample_sd_across_training_seeds']):.{digits}f}"
    )


def _format_metric(row: dict[str, Any], metric: Metric, *, latex: bool) -> str:
    value = row.get(metric.key)
    if value is None:
        return "N/A"
    if metric.exact_count:
        numerator = int(value["invalid_or_rejected"])
        denominator = int(value["requests"])
        if latex:
            return f"{numerator:,}/{denominator:,}".replace(",", "{,}")
        return f"{numerator:,} / {denominator:,} ({100 * value['rate']:.5f}%)"
    return _format_mean_sd(value, metric.digits)


def _decorate(value: str, rank: int | None, *, latex: bool) -> str:
    if rank == 1:
        return f"\\textbf{{{value}}}" if latex else f"**{value}**"
    if rank == 2:
        return f"\\underline{{{value}}}" if latex else f"<u>{value}</u>"
    return value


def _ranked_rows(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset in assets:
        asset_rows = asset.get("method_rows") or []
        ranks = {metric.key: dense_ranks(asset_rows, metric) for metric in METRICS}
        for row in asset_rows:
            item = dict(row)
            item["ranks"] = {
                metric.key: ranks[metric.key][row["method_id"]] for metric in METRICS
            }
            rows.append(item)
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Panel C: Multi-asset event-time replication",
        "",
        "All seven registered metrics are lower-is-better ($\\downarrow$). "
        "Invalidity and rejection entries are exact pooled counts, not independent "
        "sample estimates. Continuous entries are mean $\\pm$ sample SD across "
        "three training checkpoints after averaging the three paired evaluation "
        "seeds within checkpoint.",
        "Correction/call is decoded-state L2 per scheduled projection call; it is "
        "N/A when a method schedules no external projection.",
        "",
        "| Asset | Method | Pre-invalid $\\downarrow$ | Consumer reject $\\downarrow$ | Endpoint invalid $\\downarrow$ | Calls/path $\\downarrow$ | Correction/call $\\downarrow$ | Cumulative repair $\\downarrow$ | Slippage W1 $\\downarrow$ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["panel_rows"]:
        values = []
        for metric in METRICS:
            rendered = _format_metric(row, metric, latex=False)
            values.append(_decorate(rendered, row["ranks"][metric.key], latex=False))
        lines.append(
            "| {asset} | {method} | {values} |".format(
                asset=row["asset"], method=row["method"], values=" | ".join(values)
            )
        )
    validation = summary["validation"]
    lines.extend(
        [
            "",
            f"Campaign classification: **{validation['campaign_result_class']}**.  ",
            "Implementation integrity and the scientific outcome are classified "
            "separately; a verified negative result is not an integrity failure.",
            "",
            "| Asset | Integrity | Registered mechanism outcome | Fidelity warning |",
            "|---|---:|---:|---:|",
        ]
    )
    for asset in P4_ASSETS:
        item = validation["asset_classifications"][asset]
        lines.append(
            f"| {asset} | {item['integrity_verified']} | "
            f"{item['scientific_classification']} | {item['fidelity_warning']} |"
        )
    lines.extend(
        [
            "",
            "The fidelity guardrail is descriptive and is not a model-selection "
            "rule; absence of a warning does not establish absence of fidelity "
            "degradation. No cross-asset p-values are computed because the four "
            "pre-fixed assets are not random replicates.",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(summary: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated event-time panel. All metrics are lower-is-better.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Panel C: Multi-asset event-time replication. Invalidity and "
        "rejection entries are exact counts; other entries are mean $\\pm$ sample "
        "SD across three training checkpoints. Correction/call is decoded-state "
        "L2 per scheduled projection call and is N/A when no projection is "
        "scheduled.}",
        "\\label{tab:p4_event_time_replication}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Asset & Method & Pre-invalid $\\downarrow$ & Consumer reject $\\downarrow$ & "
        "Endpoint invalid $\\downarrow$ & Calls/path $\\downarrow$ & "
        "Correction/call $\\downarrow$ & Cumulative repair $\\downarrow$ & "
        "Slippage W1 $\\downarrow$ \\\\",
        "\\midrule",
    ]
    previous_asset: str | None = None
    for row in summary["panel_rows"]:
        if previous_asset is not None and row["asset"] != previous_asset:
            lines.append("\\addlinespace")
        values = []
        for metric in METRICS:
            rendered = _format_metric(row, metric, latex=True)
            values.append(_decorate(rendered, row["ranks"][metric.key], latex=True))
        lines.append(
            f"{row['asset']} & {row['method']} & " + " & ".join(values) + r" \\"
        )
        previous_asset = row["asset"]
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_rows(panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in panel_rows:
        item: dict[str, Any] = {
            "asset": row["asset"],
            "dataset": row["dataset"],
            "method_id": row["method_id"],
            "method": row["method"],
            "n_training_seeds": row["n_training_seeds"],
            "n_evaluation_seeds_per_training_seed": row[
                "n_evaluation_seeds_per_training_seed"
            ],
            "n_paths": row["n_paths"],
            "exact_projection_calls": row["exact_projection_calls"],
        }
        for metric in METRICS:
            value = row[metric.key]
            item[f"{metric.key}_direction"] = metric.direction
            item[f"{metric.key}_rank"] = row["ranks"][metric.key]
            if metric.exact_count:
                item[f"{metric.key}_count"] = value["invalid_or_rejected"]
                item[f"{metric.key}_requests"] = value["requests"]
                item[f"{metric.key}_rate"] = value["rate"]
            else:
                item[f"{metric.key}_mean"] = None if value is None else value["mean"]
                item[f"{metric.key}_sample_sd"] = (
                    None if value is None else value["sample_sd_across_training_seeds"]
                )
        output.append(item)
    return output


def build_campaign_summary(
    asset_results: list[dict[str, Any]],
    *,
    evaluation_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if [result.get("asset") for result in asset_results] != list(P4_ASSETS):
        raise ValueError(
            "Results must preserve the fixed AAPL/AMZN/GOOG/INTC order"
        )
    protocol_binding = validate_metric_clarification()
    evaluation_binding = evaluation_binding or {
        "verified": protocol_binding["verified"],
        "errors": [] if protocol_binding["verified"] else [protocol_binding["error"]],
        "plan_path": None,
        "plan_sha256": None,
        "ledger_observation": None,
    }
    integrity_verified = evaluation_binding["verified"] and all(
        result.get("integrity_verified") is True for result in asset_results
    )
    mechanism_go = integrity_verified and all(
        (result.get("scientific_outcome") or {}).get("mechanism_go") is True
        for result in asset_results
    )
    fidelity_warnings = [
        result["asset"]
        for result in asset_results
        if (result.get("scientific_outcome") or {}).get("fidelity_warning") is True
    ]
    campaign_class = (
        "terminal_with_declared_failures"
        if not integrity_verified
        else ("verified_go" if mechanism_go else "verified_scientific_no_go")
    )
    validation = {
        "implementation_integrity_verified": integrity_verified,
        "global_integrity_errors": (list(evaluation_binding.get("errors") or [])),
        "campaign_result_class": campaign_class,
        "four_asset_mechanism_go": mechanism_go,
        "headline_four_asset_replication_allowed": mechanism_go,
        "fidelity_warning_assets": fidelity_warnings,
        # This single descriptive W1 guardrail can flag a clear adverse result,
        # but absence of that warning is not evidence of no degradation (and
        # finite-fill W1 does not account for underfill).  Never turn the
        # one-way warning rule into permission for an absence claim.
        "absence_of_fidelity_degradation_language_allowed": False,
        "fidelity_language_policy": (
            "descriptive guardrail only; no absence-of-degradation claim is "
            "authorized by the registered event-time study"
        ),
        "assets_are_fixed_replication_panels_not_random_replicates": True,
        "p_values_computed": False,
        "reference_parity_policy": REFERENCE_PARITY_POLICY,
        "asset_classifications": {
            result["asset"]: {
                "integrity_verified": result.get("integrity_verified"),
                "integrity_errors": result.get("integrity_errors") or [],
                "scientific_classification": (
                    result.get("scientific_outcome") or {}
                ).get("classification"),
                "mechanism_go": (result.get("scientific_outcome") or {}).get(
                    "mechanism_go"
                ),
                "fidelity_warning": (result.get("scientific_outcome") or {}).get(
                    "fidelity_warning"
                ),
                "reference_parity_verified": (result.get("reference_parity") or {}).get(
                    "verified"
                ),
                "reference_endpoint_max_abs_difference": (
                    result.get("reference_parity") or {}
                ).get("endpoint_max_abs_difference"),
            }
            for result in asset_results
        },
    }
    panel_rows = _ranked_rows(asset_results)
    return {
        "schema_version": 1,
        "campaign_id": "p4_public_itch_event_time_20260807_v1",
        "scope": (
            "separately trained event-time sampled Level-2 snapshot replication; "
            "not message-level, replay, queue/fill, or unseen-asset generalization"
        ),
        "metric_registry": [
            {
                "key": metric.key,
                "label": metric.label,
                "direction": metric.direction,
                "exact_count": metric.exact_count,
            }
            for metric in METRICS
        ],
        "aggregation": {
            "exact_rates": "pooled integer count divided by fixed pooled requests",
            "continuous_metrics": (
                "evaluation-seed mean within checkpoint, then mean and sample SD "
                "across three training checkpoints"
            ),
            "statistical_unit_for_sd": "training checkpoint seed",
        },
        "reference_parity_policy": REFERENCE_PARITY_POLICY,
        "asset_results": asset_results,
        "panel_rows": panel_rows,
        "validation": validation,
        "metric_clarification_binding": protocol_binding,
        "evaluation_plan_binding": evaluation_binding,
    }


def validate_metric_clarification() -> dict[str, Any]:
    path = METRIC_CLARIFICATION_PATH.resolve()
    if not path.is_file():
        return {
            "path": str(path),
            "expected_sha256": METRIC_CLARIFICATION_SHA256,
            "actual_sha256": None,
            "verified": False,
            "error": "missing outcome-blind event-time metric definition",
        }
    actual = sha256_file(path)
    return {
        "path": str(path),
        "expected_sha256": METRIC_CLARIFICATION_SHA256,
        "actual_sha256": actual,
        "verified": actual == METRIC_CLARIFICATION_SHA256,
        "error": (
            None
            if actual == METRIC_CLARIFICATION_SHA256
            else "Event-time metric definition SHA-256 mismatch"
        ),
    }


def validate_evaluation_binding(evaluation_root: Path) -> dict[str, Any]:
    """Bind the panel to the immutable evaluation plan and completed dependencies."""

    errors: list[str] = []
    plan_path = evaluation_root / "_orchestrator" / "p4_evaluation_plan.json"
    ledger_path = evaluation_root / "_orchestrator" / "p4_evaluation_ledger.json"
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "verified": False,
            "errors": [f"cannot read frozen event-time evaluation plan: {exc}"],
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": None,
            "ledger_observation": None,
        }
    try:
        ledger = json.loads(ledger_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        ledger = {}
        errors.append(f"cannot read event-time evaluation ledger: {exc}")

    if plan.get("campaign_id") != "p4_public_itch_event_time_20260807_v1_evaluation":
        errors.append("evaluation plan campaign ID mismatch")
    protocol = plan.get("protocol") or {}
    expected_protocol = {
        "assets": list(P4_ASSETS),
        "datasets": [P4_DATASETS[asset] for asset in P4_ASSETS],
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "methods": list(METHODS),
        "checkpoint_name": "final.pt",
        "n_histories_per_cell": 640,
        "horizon": 100,
        "ddim_steps": 20,
        "reference_reproduction": REFERENCE_PARITY_POLICY,
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            errors.append(
                f"evaluation plan protocol {field}={protocol.get(field)!r} "
                f"expected={expected!r}"
            )
    sources = (plan.get("provenance") or {}).get("sources") or {}
    source_expectations = {
        "experiments/multi_asset_event_time/metric_definition.md": (
            METRIC_CLARIFICATION_PATH,
            METRIC_CLARIFICATION_SHA256,
        ),
        "experiments/multi_asset_event_time/summarize_results.py": (
            Path(__file__).resolve(),
            sha256_file(Path(__file__).resolve()),
        ),
    }
    for relative, (path, expected_hash) in source_expectations.items():
        receipt = sources.get(relative) or {}
        if receipt.get("sha256") != expected_hash:
            errors.append(f"evaluation plan source hash mismatch: {relative}")
        if not path.is_file() or sha256_file(path) != expected_hash:
            errors.append(f"live source hash mismatch: {relative}")
    plan_hash = sha256_file(plan_path)
    if ledger:
        if ledger.get("plan_sha256") != plan_hash:
            errors.append("evaluation ledger does not bind the current frozen plan")
        if ledger.get("campaign_id") != plan.get("campaign_id"):
            errors.append("evaluation ledger campaign ID mismatch")
        stages = ledger.get("stages") or []
        if len(stages) != EXPECTED_EVALUATION_STAGES:
            errors.append(
                f"evaluation ledger has {len(stages)} stages, "
                f"expected {EXPECTED_EVALUATION_STAGES}"
            )
        panel_stages = [stage for stage in stages if stage.get("kind") == "panel_c"]
        if len(panel_stages) != 1:
            errors.append("evaluation ledger does not contain exactly one panel stage")
        else:
            dependencies = set(panel_stages[0].get("dependencies") or [])
            status_by_id = {
                stage.get("stage_id"): stage.get("status") for stage in stages
            }
            incomplete = sorted(
                stage_id
                for stage_id in dependencies
                if status_by_id.get(stage_id) != "completed"
            )
            if incomplete:
                errors.append(f"panel dependencies are not completed: {incomplete}")
    clarification = validate_metric_clarification()
    if not clarification["verified"]:
        errors.append(str(clarification["error"]))
    return {
        "verified": not errors,
        "errors": errors,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan_hash,
        "ledger_observation": {
            "path": str(ledger_path.resolve()),
            "revision": ledger.get("revision"),
            "status": ledger.get("status"),
            "panel_dependency_count": (
                len(panel_stages[0].get("dependencies") or [])
                if ledger and len(panel_stages) == 1
                else None
            ),
        },
        "metric_clarification_sha256": clarification["actual_sha256"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty event-time CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_outputs(evaluation_root: Path, output_dir: Path) -> int:
    """Validate all fixed assets and atomically write a never-overwritten bundle."""

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing event-time output {output_dir}"
        )
    asset_results = [
        validate_asset(asset, evaluation_root / asset / "consumer")
        for asset in P4_ASSETS
    ]
    evaluation_binding = validate_evaluation_binding(evaluation_root)
    summary = build_campaign_summary(
        asset_results, evaluation_binding=evaluation_binding
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        validation = summary["validation"]
        (staging / "p4_event_robustness_validation.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        # Preserve integrity failures, but do not create a publication panel
        # whose inputs failed the frozen evaluation contract.
        if validation["implementation_integrity_verified"]:
            (staging / "p4_event_robustness_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            _write_csv(
                staging / "p4_event_robustness_summary.csv",
                _csv_rows(summary["panel_rows"]),
            )
            (staging / "p4_event_robustness_panel_c.md").write_text(
                render_markdown(summary)
            )
            (staging / "p4_event_robustness_panel_c.tex").write_text(
                render_latex(summary)
            )
        input_hashes = {
            path: digest
            for result in asset_results
            for path, digest in (result.get("input_hashes") or {}).items()
        }
        outputs = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        receipt = {
            "schema_version": 1,
            "campaign_id": summary["campaign_id"],
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "metric_registry_sha256": _sha256_json(summary["metric_registry"]),
            "reference_parity_policy": REFERENCE_PARITY_POLICY,
            "reference_parity_policy_sha256": _sha256_json(REFERENCE_PARITY_POLICY),
            "reference_parity_by_asset": {
                result["asset"]: {
                    "verified": (result.get("reference_parity") or {}).get("verified"),
                    "endpoint_max_abs_difference": (
                        result.get("reference_parity") or {}
                    ).get("endpoint_max_abs_difference"),
                    "non_endpoint_errors": (result.get("reference_parity") or {}).get(
                        "non_endpoint_errors"
                    ),
                }
                for result in asset_results
            },
            "metric_clarification_binding": summary["metric_clarification_binding"],
            "evaluation_plan_binding": summary["evaluation_plan_binding"],
            "input_hashes": dict(sorted(input_hashes.items())),
            "output_hashes_before_receipt": outputs,
            "result_class": validation["campaign_result_class"],
        }
        (staging / "p4_event_robustness_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging)
        raise
    return 0 if summary["validation"]["implementation_integrity_verified"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.evaluation_root / "panel_c")
    return write_outputs(args.evaluation_root, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed, descriptive cross-asset summary for the P0 consumer panel.

This script never treats reverse-diffusion requests as independent samples and
never treats the three assets as random replicates.  It validates each frozen
asset panel (including source, checkpoint, reference, cell, and outcome hashes)
before producing per-asset comparisons and pooled *exact counts* only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = ("AAPL", "AMZN", "GOOG")
DATASETS = {
    asset: f"PUBLIC_ITCH_2019_SAMPLES_{asset}_1s" for asset in ASSETS
}
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
    "matched_repeated_pava": "Matched+Repeated PAVA",
}
CHECKPOINT_VARIANTS = ("legal_lob", "matched_lob")
REJECTION_REASONS = (
    "nonfinite",
    "spread",
    "ask_chain",
    "bid_chain",
    "negative_depth",
)

EXPECTED_CELLS_PER_ASSET = 9
EXPECTED_TRAJECTORIES_PER_CELL = 640
EXPECTED_ANYTIME_FRAMES_PER_CELL = 1_280_000
EXPECTED_ANYTIME_BLOCKS_PER_CELL = 128_000
EXPECTED_ENDPOINT_FRAMES_PER_CELL = 64_000
EXPECTED_ENDPOINT_BLOCKS_PER_CELL = 6_400
EXPECTED_ASSET_TOTALS = {
    "trajectories": 5_760,
    "anytime_frame_requests": 11_520_000,
    "anytime_block_requests": 1_152_000,
    "endpoint_frame_requests": 576_000,
}
EXPECTED_PROJECTION_CALLS = {
    "legal_lob": 0.0,
    "matched_posthoc_pava": 10.0,
    "matched_repeated_pava": 200.0,
}
REQUIRED_SOURCE_PATHS = {
    "scripts/run_intermediate_consumer_panel.py",
    "scripts/summarize_intermediate_consumer.py",
    "scripts/snowball_test.py",
    "scripts/summarize_financial_sensitivity.py",
    "tests/test_intermediate_consumer.py",
    "tests/test_summarize_intermediate_consumer.py",
    "genlob/models/sampling.py",
    "genlob/utils/legal_lob.py",
    "genlob/utils/physics.py",
}
REQUIRED_REFERENCE_SOURCE_PATHS = {
    "scripts/build_matched_rollout_reference.py",
    "scripts/run_all_method_rollout_panel.py",
    "scripts/snowball_test.py",
    "scripts/summarize_intermediate_consumer.py",
}
INPUT_FILENAMES = (
    "run_manifest.json",
    "intermediate_consumer_outcome_per_run.csv",
    "intermediate_consumer_outcome_summary.json",
    "intermediate_consumer_outcome_counts.json",
    "intermediate_consumer_outcome_validation.json",
    "intermediate_consumer_outcome_receipt.json",
)
INTEGRITY_RECLASSIFICATION_RULE = {
    "schema_version": 1,
    "purpose": "separate implementation integrity from registered science outcomes",
    "required_manifest_status": "completed",
    "required_expected_cells": 9,
    "required_completed_cells": 9,
    "required_method_rows": 27,
    "required_empty_base_fields": [
        "manifest_integrity_errors",
        "rng_pairing_failures",
        "denominator_failures",
        "source_hash_failures",
        "summary_completeness_failures",
    ],
    "required_declared_outcome_field": {
        "reference_rollout_reproduced": True
    },
    "base_fields_not_used_directly_for_implementation_integrity": [
        "base_pre_registered_validation.cell_implementation_gate_failures",
        "non_performance_cell_failures",
    ],
    "science_only_cell_failure_name_tokens": [
        "acceptance",
        "rejection",
    ],
    "independently_revalidated_implementation_contracts": [
        "projection calls per trajectory",
        "cell and manifest hashes",
        "paired RNG receipts",
        "request denominators",
    ],
    "scientific_fields_not_used_for_implementation_integrity": [
        "science_checks",
        "science_checks_pass",
        "pre_registered_strict_go",
    ],
    "classification": (
        "method acceptance and rejection failures are scientific outcomes; "
        "projection-call, hash, RNG, denominator, completeness, and reference "
        "failures remain implementation-integrity failures"
    ),
    "threshold_policy": "no scientific or diagnostic tolerance is changed",
}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    direction: str
    digits: int = 3
    percent: bool = False


INTERFACE_METRICS = (
    Metric("anytime_pre_frame_acceptance", "Anytime pre frame", "up", 3, True),
    Metric("anytime_pre_block_acceptance", "Anytime pre block", "up", 3, True),
    Metric(
        "anytime_delivered_frame_acceptance",
        "Delivered anytime frame",
        "up",
        5,
        True,
    ),
    Metric(
        "anytime_delivered_completed_trajectory_rate",
        "No-rejection trajectories",
        "up",
        3,
        True,
    ),
    Metric(
        "endpoint_delivered_frame_acceptance",
        "Endpoint delivered frame",
        "up",
        3,
        True,
    ),
    Metric("projection_calls_per_trajectory", "Projection calls/path", "down", 1),
    Metric("projection_l2_cumulative", "Cumulative projection L2", "down", 3),
)
FIDELITY_METRICS = (
    Metric("market_order_slippage_w1_ticks", "Slippage W1", "down", 3),
    Metric("liquidity_5tick_w1_target_units", "Liq@5 W1", "down", 3),
    Metric("risk_alarm_rate_abs_error", "Alarm error", "down", 2, True),
    Metric("spread_relative_abs_error", "Spread rel. error", "down", 2, True),
    Metric("volume_relative_abs_error", "Volume rel. error", "down", 2, True),
    Metric(
        "return_std_relative_abs_error", "Return-vol. rel. error", "down", 2, True
    ),
    Metric("temporal_correlation", "Temporal correlation", "up", 3),
)
ALL_METRICS = INTERFACE_METRICS + FIDELITY_METRICS


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve_recorded_path(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_json(path: Path, errors: list[str], label: str) -> Any:
    if not path.is_file():
        errors.append(f"{label}: missing {path}")
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: unreadable JSON {path}: {exc}")
        return None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _isclose(left: Any, right: Any, *, abs_tol: float = 1e-12) -> bool:
    return _finite(left) and _finite(right) and math.isclose(
        float(left), float(right), rel_tol=1e-10, abs_tol=abs_tol
    )


def _get(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _is_science_cell_failure(name: str) -> bool:
    return any(
        token in name
        for token in INTEGRITY_RECLASSIFICATION_RULE[
            "science_only_cell_failure_name_tokens"
        ]
    )


def _hash_matches(path: Path, expected: Any, errors: list[str], label: str) -> bool:
    if not path.is_file():
        errors.append(f"{label}: missing file {path}")
        return False
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"{label}: missing or malformed SHA-256")
        return False
    actual = sha256_file(path)
    if actual != expected:
        errors.append(f"{label}: SHA-256 mismatch for {path}")
        return False
    return True


def _load_cell_payloads(
    asset: str,
    root: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> list[tuple[Path, dict[str, Any]]]:
    expected_keys = {(train, eval_) for train in TRAIN_SEEDS for eval_ in EVAL_SEEDS}
    entries = manifest.get("cells")
    if not isinstance(entries, list):
        errors.append(f"{asset}: manifest cells is not a list")
        return []
    seen: list[tuple[int, int]] = []
    manifest_paths: set[Path] = set()
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"{asset}: manifest cell {index} is not an object")
            continue
        try:
            key = (int(entry["train_seed"]), int(entry["eval_seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{asset}: malformed manifest cell {index}: {exc}")
            continue
        seen.append(key)
        if entry.get("status") != "completed":
            errors.append(f"{asset}: cell {key} status={entry.get('status')}")
            continue
        path = _resolve_recorded_path(entry.get("path", ""))
        manifest_paths.add(path)
        if not _hash_matches(path, entry.get("sha256"), errors, f"{asset} cell {key}"):
            continue
        payload = _read_json(path, errors, f"{asset} cell {key}")
        if not isinstance(payload, dict):
            continue
        try:
            payload_key = (int(payload["train_seed"]), int(payload["eval_seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{asset}: malformed cell payload {key}: {exc}")
            continue
        if payload_key != key:
            errors.append(f"{asset}: cell {key} payload key={payload_key}")
            continue
        if payload.get("status") != "completed":
            errors.append(f"{asset}: cell {key} payload status={payload.get('status')}")
            continue
        if set((payload.get("methods") or {}).keys()) != set(METHODS):
            errors.append(f"{asset}: cell {key} method set mismatch")
            continue
        loaded.append((path, payload))

    if len(seen) != len(set(seen)):
        errors.append(f"{asset}: duplicate manifest cell keys")
    if set(seen) != expected_keys:
        errors.append(
            f"{asset}: cell set mismatch missing={sorted(expected_keys-set(seen))} "
            f"extra={sorted(set(seen)-expected_keys)}"
        )
    if len(entries) != EXPECTED_CELLS_PER_ASSET:
        errors.append(
            f"{asset}: manifest has {len(entries)} cells, expected "
            f"{EXPECTED_CELLS_PER_ASSET}"
        )
    orphan_paths = {
        path.resolve()
        for path in root.glob("train_seed*/eval_seed*/consumer_stats.json")
    } - manifest_paths
    if orphan_paths:
        errors.append(
            f"{asset}: orphan consumer files: "
            + ", ".join(str(path) for path in sorted(orphan_paths))
        )
    return loaded


def _validate_sources_and_data(
    asset: str,
    manifest: dict[str, Any],
    errors: list[str],
    input_hashes: dict[str, str],
) -> None:
    provenance = manifest.get("provenance") or {}
    sources = provenance.get("sources_sha256") or {}
    missing_sources = REQUIRED_SOURCE_PATHS - set(sources)
    if missing_sources:
        errors.append(f"{asset}: missing source receipts {sorted(missing_sources)}")
    for relative, expected in sorted(sources.items()):
        path = _resolve_recorded_path(relative)
        if _hash_matches(path, expected, errors, f"{asset} source {relative}"):
            input_hashes[str(path)] = str(expected)

    data = provenance.get("data") or {}
    dataset_dir = _resolve_recorded_path(data.get("dataset_dir", ""))
    data_files = {
        "normalization.json": data.get("normalization_sha256"),
        "manifest.yaml": data.get("manifest_sha256"),
        **(data.get("arrays_sha256") or {}),
    }
    required_data = {
        "normalization.json",
        "manifest.yaml",
        "states.npy",
        "states_raw.npy",
        "split.npy",
        "segment_id.npy",
        "actions.npy",
        "source_row_idx.npy",
    }
    if not required_data.issubset(data_files):
        errors.append(
            f"{asset}: incomplete data hash receipt; missing "
            f"{sorted(required_data-set(data_files))}"
        )
    start_indices_sha256 = data.get("start_indices_sha256")
    if not isinstance(start_indices_sha256, str) or len(start_indices_sha256) != 64:
        errors.append(f"{asset}: missing derived start-index SHA-256 receipt")
    for name, expected in sorted(data_files.items()):
        path = dataset_dir / name
        if _hash_matches(path, expected, errors, f"{asset} data {name}"):
            input_hashes[str(path)] = str(expected)


def _validate_checkpoints(
    asset: str,
    manifest: dict[str, Any],
    cells: list[tuple[Path, dict[str, Any]]],
    errors: list[str],
    input_hashes: dict[str, str],
) -> None:
    checkpoint_root = _resolve_recorded_path(
        (manifest.get("config") or {}).get("checkpoint_root", "")
    )
    seen_by_train: dict[int, dict[str, dict[str, Any]]] = {}
    verified_paths: set[Path] = set()
    for _, cell in cells:
        train_seed = int(cell["train_seed"])
        receipts = cell.get("checkpoints")
        if not isinstance(receipts, dict) or set(receipts) != set(CHECKPOINT_VARIANTS):
            errors.append(f"{asset}: train={train_seed} checkpoint receipt set mismatch")
            continue
        previous = seen_by_train.get(train_seed)
        if previous is not None and receipts != previous:
            errors.append(f"{asset}: train={train_seed} checkpoint receipts vary by eval seed")
        seen_by_train.setdefault(train_seed, receipts)
        signatures = {receipt.get("config_sha256") for receipt in receipts.values()}
        if len(signatures) != 1 or None in signatures:
            errors.append(f"{asset}: train={train_seed} checkpoint config signatures differ")
        for variant, receipt in receipts.items():
            expected_path = (
                checkpoint_root / f"train_seed{train_seed}" / variant / "best.pt"
            ).resolve()
            recorded_path = _resolve_recorded_path(receipt.get("path", ""))
            if recorded_path != expected_path:
                errors.append(
                    f"{asset}: train={train_seed} {variant} checkpoint path "
                    f"{recorded_path} != {expected_path}"
                )
            if recorded_path not in verified_paths and _hash_matches(
                recorded_path,
                receipt.get("sha256"),
                errors,
                f"{asset} train={train_seed} {variant} checkpoint",
            ):
                verified_paths.add(recorded_path)
                input_hashes[str(recorded_path)] = str(receipt["sha256"])
    if set(seen_by_train) != set(TRAIN_SEEDS):
        errors.append(f"{asset}: checkpoint train-seed coverage mismatch")


def _validate_reference(
    asset: str,
    manifest: dict[str, Any],
    errors: list[str],
    input_hashes: dict[str, str],
) -> None:
    reference = (manifest.get("provenance") or {}).get("reference_rollout") or {}
    reference_path = _resolve_recorded_path(reference.get("path", ""))
    if _hash_matches(
        reference_path, reference.get("sha256"), errors, f"{asset} reference CSV"
    ):
        input_hashes[str(reference_path)] = str(reference["sha256"])
        try:
            with reference_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            errors.append(f"{asset}: cannot read reference CSV: {exc}")
            rows = []
        keys: list[tuple[int, int]] = []
        for row in rows:
            try:
                if row.get("method_id") != "matched_posthoc_pava":
                    raise ValueError("wrong method")
                if int(row["horizon"]) != 100 or int(row["n_histories"]) != 640:
                    raise ValueError("wrong horizon or history count")
                keys.append((int(row["train_seed"]), int(row["eval_seed"])))
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{asset}: malformed reference row: {exc}")
        expected = {(train, eval_) for train in TRAIN_SEEDS for eval_ in EVAL_SEEDS}
        if len(rows) != 9 or len(keys) != len(set(keys)) or set(keys) != expected:
            errors.append(f"{asset}: reference CSV does not contain the exact 3x3 cells")

    validation_path = reference_path.with_name(
        f"{reference_path.stem}_validation.json"
    )
    validation = _read_json(validation_path, errors, f"{asset} reference validation")
    if not isinstance(validation, dict):
        return
    input_hashes[str(validation_path)] = sha256_file(validation_path)
    completeness = validation.get("completeness") or {}
    protocol = validation.get("expected_protocol") or {}
    reference_receipt = validation.get("reference_csv") or {}
    if validation.get("status") != "validated" or validation.get(
        "method_id"
    ) != "matched_posthoc_pava":
        errors.append(f"{asset}: reference validation status is not validated")
    if protocol.get("dataset") != DATASETS[asset]:
        errors.append(f"{asset}: reference protocol dataset mismatch")
    if protocol.get("train_seeds") != list(TRAIN_SEEDS) or protocol.get(
        "eval_seeds"
    ) != list(EVAL_SEEDS):
        errors.append(f"{asset}: reference protocol seed grid mismatch")
    expected_protocol = {
        "hist_len": 50,
        "pred_len": 10,
        "chunk_size": 10,
        "horizon": 100,
        "ddim_steps": 20,
        "batch_size": 128,
        "n_batches": 5,
        "n_histories_per_cell": 640,
        "device": "mps",
        "weights": "ema",
        "action_mode": "hold",
        "projection_mode": "exact",
        "sampler_projection_mode": "none",
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            errors.append(
                f"{asset}: reference protocol {field}={protocol.get(field)} "
                f"expected={expected}"
            )
    expected_data_dir = (REPO_ROOT / "data").resolve()
    expected_checkpoint_root = (
        REPO_ROOT
        / "checkpoints_training_seeds/public_itch_1s_panel_50ep"
        / asset
    ).resolve()
    if _resolve_recorded_path(protocol.get("data_dir", "")) != expected_data_dir:
        errors.append(f"{asset}: reference protocol data_dir mismatch")
    if (
        _resolve_recorded_path(protocol.get("checkpoint_root", ""))
        != expected_checkpoint_root
    ):
        errors.append(f"{asset}: reference protocol checkpoint_root mismatch")
    if completeness != {
        "expected_cells": 9,
        "observed_cells": 9,
        "all_cells_complete": True,
    }:
        errors.append(f"{asset}: reference completeness receipt mismatch")
    if (
        _resolve_recorded_path(reference_receipt.get("path", "")) != reference_path
        or reference_receipt.get("sha256") != reference.get("sha256")
        or reference_receipt.get("rows") != 9
    ):
        errors.append(f"{asset}: reference CSV receipt mismatch")
    validation_cells = validation.get("cells") or []
    keys = []
    for item in validation_cells:
        try:
            key = (int(item["train_seed"]), int(item["eval_seed"]))
            keys.append(key)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{asset}: malformed reference receipt cell: {exc}")
            continue
        if item.get("status") != "completed":
            errors.append(f"{asset}: reference receipt cell {key} not completed")
        for path_field, hash_field in (
            ("manifest_path", "manifest_sha256"),
            ("stats_path", "stats_sha256"),
            ("checkpoint_path", "checkpoint_sha256"),
            ("command_path", "command_sha256"),
        ):
            path = _resolve_recorded_path(item.get(path_field, ""))
            expected_hash = item.get(hash_field)
            if _hash_matches(path, expected_hash, errors, f"{asset} reference {key} {path_field}"):
                input_hashes[str(path)] = str(expected_hash)
    expected_keys = {(train, eval_) for train in TRAIN_SEEDS for eval_ in EVAL_SEEDS}
    if len(validation_cells) != 9 or len(keys) != len(set(keys)) or set(keys) != expected_keys:
        errors.append(f"{asset}: reference validation cell set mismatch")
    dataset_receipt = validation.get("dataset_receipt") or {}
    reference_dataset_dir = _resolve_recorded_path(
        dataset_receipt.get("dataset_dir", "")
    )
    expected_dataset_dir = (
        REPO_ROOT / "data/processed/snapshot_l2" / DATASETS[asset]
    ).resolve()
    if reference_dataset_dir != expected_dataset_dir:
        errors.append(f"{asset}: reference dataset directory mismatch")
    for name, hash_field in (
        ("manifest.yaml", "manifest_sha256"),
        ("normalization.json", "normalization_sha256"),
    ):
        path = reference_dataset_dir / name
        expected_hash = dataset_receipt.get(hash_field)
        if _hash_matches(path, expected_hash, errors, f"{asset} reference data {name}"):
            input_hashes[str(path)] = str(expected_hash)
    reference_sources = validation.get("source_sha256") or {}
    missing_reference_sources = REQUIRED_REFERENCE_SOURCE_PATHS - set(
        reference_sources
    )
    if missing_reference_sources:
        errors.append(
            f"{asset}: missing reference source receipts "
            f"{sorted(missing_reference_sources)}"
        )
    for relative, expected_hash in reference_sources.items():
        path = _resolve_recorded_path(relative)
        if _hash_matches(path, expected_hash, errors, f"{asset} reference source {relative}"):
            input_hashes[str(path)] = str(expected_hash)


def _validate_cell_denominators(
    asset: str,
    cells: list[tuple[Path, dict[str, Any]]],
    errors: list[str],
) -> list[str]:
    science_failures: list[str] = []
    request_expectations = {
        "anytime_consumer.pre.frame_requests": EXPECTED_ANYTIME_FRAMES_PER_CELL,
        "anytime_consumer.pre.block_requests": EXPECTED_ANYTIME_BLOCKS_PER_CELL,
        "anytime_consumer.delivered.frame_requests": EXPECTED_ANYTIME_FRAMES_PER_CELL,
        "anytime_consumer.delivered.block_requests": EXPECTED_ANYTIME_BLOCKS_PER_CELL,
        "endpoint_consumer.pre.frame_requests": EXPECTED_ENDPOINT_FRAMES_PER_CELL,
        "endpoint_consumer.pre.block_requests": EXPECTED_ENDPOINT_BLOCKS_PER_CELL,
        "endpoint_consumer.delivered.frame_requests": EXPECTED_ENDPOINT_FRAMES_PER_CELL,
        "endpoint_consumer.delivered.block_requests": EXPECTED_ENDPOINT_BLOCKS_PER_CELL,
    }
    for _, cell in cells:
        cell_key = (cell["train_seed"], cell["eval_seed"])
        rng_hashes = set((cell.get("post_model_load_rng_sha256") or {}).values())
        if len(rng_hashes) != 1:
            errors.append(f"{asset}: cell {cell_key} paired RNG receipt mismatch")
        checks = (cell.get("validation") or {}).get("implementation_checks")
        if not isinstance(checks, dict):
            errors.append(f"{asset}: cell {cell_key} lacks implementation checks")
        else:
            for name, passed in checks.items():
                if passed:
                    continue
                classified = f"train={cell_key[0]} eval={cell_key[1]} {name}"
                if _is_science_cell_failure(name):
                    science_failures.append(classified)
                else:
                    errors.append(
                        f"{asset}: cell {cell_key} implementation failure {name}"
                    )
        for method_id, method in cell["methods"].items():
            label = f"{asset} cell={cell_key} method={method_id}"
            if int(method.get("n_histories", -1)) != EXPECTED_TRAJECTORIES_PER_CELL:
                errors.append(f"{label}: n_histories mismatch")
            if int(_get(method, "anytime_consumer.trajectories") or -1) != 640:
                errors.append(f"{label}: trajectory denominator mismatch")
            for dotted, expected in request_expectations.items():
                value = _get(method, dotted)
                if not _finite(value) or int(float(value)) != expected:
                    errors.append(f"{label}: {dotted}={value} expected={expected}")
            calls = _get(method, "projection.calls_per_trajectory")
            if not _isclose(calls, EXPECTED_PROJECTION_CALLS[method_id]):
                errors.append(f"{label}: projection calls={calls}")
    return sorted(science_failures)


def _counts_from_cells(
    cells: list[tuple[Path, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for method_id in METHODS:
        totals: dict[str, Any] = {
            "anytime_frame_requests": 0,
            "anytime_frame_rejections": 0,
            "anytime_block_requests": 0,
            "anytime_block_rejections": 0,
            "endpoint_frame_requests": 0,
            "endpoint_frame_rejections": 0,
            "trajectories": 0,
            "trajectories_with_any_anytime_rejection": 0,
            "rejection_reason_counts": {reason: 0 for reason in REJECTION_REASONS},
        }
        for _, cell in cells:
            method = cell["methods"][method_id]
            anytime = method["anytime_consumer"]
            delivered = anytime["delivered"]
            endpoint = method["endpoint_consumer"]["delivered"]
            frame_requests = int(delivered["frame_requests"])
            block_requests = int(delivered["block_requests"])
            endpoint_requests = int(endpoint["frame_requests"])
            trajectories = int(anytime["trajectories"])
            totals["anytime_frame_requests"] += frame_requests
            totals["anytime_frame_rejections"] += frame_requests - round(
                float(delivered["frame_acceptance_rate"]) * frame_requests
            )
            totals["anytime_block_requests"] += block_requests
            totals["anytime_block_rejections"] += block_requests - round(
                float(delivered["block_acceptance_rate"]) * block_requests
            )
            totals["endpoint_frame_requests"] += endpoint_requests
            totals["endpoint_frame_rejections"] += endpoint_requests - round(
                float(endpoint["frame_acceptance_rate"]) * endpoint_requests
            )
            totals["trajectories"] += trajectories
            totals["trajectories_with_any_anytime_rejection"] += trajectories - round(
                float(anytime["delivered_completed_without_rejection_rate"])
                * trajectories
            )
            for reason in REJECTION_REASONS:
                totals["rejection_reason_counts"][reason] += int(
                    delivered["rejection_reason_counts"][reason]
                )
        totals["anytime_frame_acceptance_rate"] = 1.0 - (
            totals["anytime_frame_rejections"] / totals["anytime_frame_requests"]
        )
        totals["anytime_block_acceptance_rate"] = 1.0 - (
            totals["anytime_block_rejections"] / totals["anytime_block_requests"]
        )
        totals["completed_without_anytime_rejection_rate"] = 1.0 - (
            totals["trajectories_with_any_anytime_rejection"]
            / totals["trajectories"]
        )
        output[method_id] = totals
    return output


def _validate_counts(
    asset: str,
    stored: Any,
    recomputed: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    if not isinstance(stored, dict) or set(stored) != set(METHODS):
        errors.append(f"{asset}: outcome count method set mismatch")
        return
    if stored != recomputed:
        errors.append(f"{asset}: outcome exact counts differ from cell recomputation")
    for method_id in METHODS:
        counts = stored[method_id]
        for field, expected in EXPECTED_ASSET_TOTALS.items():
            if counts.get(field) != expected:
                errors.append(
                    f"{asset}: {method_id} {field}={counts.get(field)} "
                    f"expected={expected}"
                )
        for reject_field, request_field in (
            ("anytime_frame_rejections", "anytime_frame_requests"),
            ("anytime_block_rejections", "anytime_block_requests"),
            ("endpoint_frame_rejections", "endpoint_frame_requests"),
            ("trajectories_with_any_anytime_rejection", "trajectories"),
        ):
            rejected = counts.get(reject_field)
            requests = counts.get(request_field)
            if not isinstance(rejected, int) or not isinstance(requests, int):
                errors.append(f"{asset}: {method_id} non-integer exact count")
            elif not (0 <= rejected <= requests):
                errors.append(f"{asset}: {method_id} impossible {reject_field}")


def _load_and_validate_per_run_csv(
    asset: str,
    path: Path,
    errors: list[str],
) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        errors.append(f"{asset}: cannot read outcome per-run CSV: {exc}")
        return []
    expected = {
        (train, eval_, method)
        for train in TRAIN_SEEDS
        for eval_ in EVAL_SEEDS
        for method in METHODS
    }
    keys: list[tuple[int, int, str]] = []
    for row in rows:
        try:
            key = (
                int(row["train_seed"]),
                int(row["eval_seed"]),
                row["method_id"],
            )
            keys.append(key)
            if int(float(row["n_histories"])) != 640 or int(
                float(row["horizon"])
            ) != 100:
                raise ValueError("wrong history count or horizon")
            for metric in ALL_METRICS:
                if not _finite(row.get(metric.key)):
                    raise ValueError(f"nonfinite {metric.key}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{asset}: malformed per-run outcome row: {exc}")
    if len(rows) != 27 or len(keys) != len(set(keys)) or set(keys) != expected:
        errors.append(f"{asset}: outcome per-run CSV does not contain exact 3x3x3 rows")
    return rows


def _validate_summary(
    asset: str,
    summary: Any,
    per_run_rows: list[dict[str, str]],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(summary, list):
        errors.append(f"{asset}: outcome summary is not a list")
        return []
    by_method = {
        row.get("method_id"): row for row in summary if isinstance(row, dict)
    }
    if len(summary) != 3 or set(by_method) != set(METHODS):
        errors.append(f"{asset}: outcome summary method set mismatch")
        return []
    output: list[dict[str, Any]] = []
    for method_id in METHODS:
        row = by_method[method_id]
        if row.get("n_training_seeds") != 3 or row.get(
            "n_eval_seeds_per_training_seed"
        ) != [3, 3, 3]:
            errors.append(f"{asset}: {method_id} summary seed denominator mismatch")
        item: dict[str, Any] = {
            "asset": asset,
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
        }
        for metric in ALL_METRICS:
            mean_key = f"{metric.key}_mean"
            std_key = f"{metric.key}_std"
            if not _finite(row.get(mean_key)) or not _finite(row.get(std_key)):
                errors.append(f"{asset}: {method_id} nonfinite {metric.key} summary")
                continue
            if (
                row.get(f"{metric.key}_n_training_seeds") != 3
                or row.get(f"{metric.key}_n_eval_seeds_per_training_seed")
                != [3, 3, 3]
                or row.get(f"{metric.key}_missing_cells") != 0
            ):
                errors.append(f"{asset}: {method_id} {metric.key} denominator mismatch")
            item[mean_key] = float(row[mean_key])
            item[std_key] = float(row[std_key])
            matching = [
                per_run
                for per_run in per_run_rows
                if per_run.get("method_id") == method_id
            ]
            checkpoint_means = []
            for train_seed in TRAIN_SEEDS:
                values = [
                    float(per_run[metric.key])
                    for per_run in matching
                    if int(per_run["train_seed"]) == train_seed
                    and _finite(per_run.get(metric.key))
                ]
                if len(values) != 3:
                    errors.append(
                        f"{asset}: {method_id} {metric.key} per-run eval "
                        f"denominator mismatch for train={train_seed}"
                    )
                elif values:
                    checkpoint_means.append(sum(values) / len(values))
            if len(checkpoint_means) == 3:
                recomputed_mean = sum(checkpoint_means) / 3
                recomputed_std = math.sqrt(
                    sum((value - recomputed_mean) ** 2 for value in checkpoint_means)
                    / 2
                )
                if not _isclose(row.get(mean_key), recomputed_mean):
                    errors.append(
                        f"{asset}: {method_id} {metric.key} summary mean "
                        "does not reproduce per-run CSV"
                    )
                if not _isclose(row.get(std_key), recomputed_std):
                    errors.append(
                        f"{asset}: {method_id} {metric.key} summary SD "
                        "does not reproduce per-run CSV"
                    )
        output.append(item)
    return output


def _validate_outcome_gate(
    asset: str,
    validation: Any,
    counts: dict[str, dict[str, Any]],
    raw_science_failures: list[str],
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(validation, dict):
        errors.append(f"{asset}: missing outcome validation object")
        return {}
    legacy_integrity = validation.get("experiment_integrity_verified")
    if legacy_integrity not in (True, False):
        errors.append(f"{asset}: malformed legacy experiment-integrity flag")
    strict_go = validation.get("pre_registered_strict_go")
    expected_class = "strict_go" if strict_go is True else "validated_strict_no_go"
    if strict_go not in (True, False) or validation.get("result_class") != expected_class:
        errors.append(f"{asset}: inconsistent scientific result class")
    base = validation.get("base_pre_registered_validation") or {}
    required_empty_fields = (
        "manifest_integrity_errors",
        "rng_pairing_failures",
        "denominator_failures",
        "source_hash_failures",
        "summary_completeness_failures",
    )
    reclassification_failures: list[str] = []
    expected_scalars = {
        "manifest_status": "completed",
        "expected_cells": 9,
        "completed_cells": 9,
        "method_rows": 27,
    }
    for field, expected in expected_scalars.items():
        if base.get(field) != expected:
            reclassification_failures.append(
                f"{field}={base.get(field)!r} expected={expected!r}"
            )
    for field in required_empty_fields:
        if base.get(field) != []:
            reclassification_failures.append(f"{field} is not empty")
    if validation.get("reference_rollout_reproduced") is not True:
        reclassification_failures.append("reference_rollout_reproduced is not true")

    base_cell_failures = sorted(base.get("cell_implementation_gate_failures") or [])
    if base_cell_failures != sorted(raw_science_failures):
        errors.append(
            f"{asset}: raw method acceptance/performance failures do not "
            "reproduce the legacy base failure list"
        )
    unclassified_cell_failures = [
        failure
        for failure in base_cell_failures
        if not _is_science_cell_failure(failure)
    ]
    if unclassified_cell_failures:
        errors.append(
            f"{asset}: non-scientific cell gate failures require implementation "
            f"review: {unclassified_cell_failures}"
        )
    reclassified_base_integrity = not reclassification_failures
    if not reclassified_base_integrity:
        errors.append(
            f"{asset}: reclassified implementation integrity failed: "
            + "; ".join(reclassification_failures)
        )

    science_checks = base.get("science_checks") or {}
    failed_science_checks = sorted(
        name for name, passed in science_checks.items() if passed is not True
    )
    if base.get("science_checks_pass") is not strict_go:
        errors.append(f"{asset}: strict-GO flag does not match base science checks")
    repeated = counts.get("matched_repeated_pava") or {}
    expected_repeated = {
        "repeated_anytime_frame_rejections": repeated.get(
            "anytime_frame_rejections"
        ),
        "repeated_anytime_frame_requests": repeated.get("anytime_frame_requests"),
        "repeated_trajectories_with_rejection": repeated.get(
            "trajectories_with_any_anytime_rejection"
        ),
        "repeated_trajectories": repeated.get("trajectories"),
    }
    for field, expected in expected_repeated.items():
        if validation.get(field) != expected:
            errors.append(f"{asset}: outcome validation {field} count mismatch")
    return {
        "legacy_experiment_integrity_verified": legacy_integrity,
        "reclassified_base_integrity_verified": reclassified_base_integrity,
        "reclassified_experiment_integrity_verified": reclassified_base_integrity,
        "integrity_reclassification_rule_sha256": sha256_json(
            INTEGRITY_RECLASSIFICATION_RULE
        ),
        "pre_registered_strict_go": strict_go,
        "registered_scientific_result_class": validation.get("result_class"),
        "legacy_reason_for_strict_no_go": validation.get(
            "reason_for_strict_no_go"
        ),
        "failed_science_checks": failed_science_checks,
        "method_acceptance_performance_cell_failures": base_cell_failures,
        "legacy_non_performance_cell_failures": validation.get(
            "non_performance_cell_failures"
        ),
        "legacy_reference_reproduction_diagnostic": base.get(
            "reference_reproduction"
        ),
        "declared_reference_rollout_reproduced": validation.get(
            "reference_rollout_reproduced"
        ),
        "reclassification_failures": reclassification_failures,
    }


def _validate_outcome_receipt(
    asset: str,
    root: Path,
    receipt: Any,
    errors: list[str],
    input_hashes: dict[str, str],
) -> None:
    if not isinstance(receipt, dict):
        errors.append(f"{asset}: missing outcome receipt")
        return
    receipt_targets = {
        "run_manifest_sha256": root / "run_manifest.json",
        "validation_sha256": root / "intermediate_consumer_outcome_validation.json",
        "summary_sha256": root / "intermediate_consumer_outcome_summary.json",
    }
    for field, path in receipt_targets.items():
        if _hash_matches(path, receipt.get(field), errors, f"{asset} outcome receipt {field}"):
            input_hashes[str(path.resolve())] = str(receipt[field])
    script_targets = {
        "analysis_script_sha256": receipt.get("analysis_script"),
        "base_summarizer_sha256": "scripts/summarize_intermediate_consumer.py",
    }
    for field, recorded_path in script_targets.items():
        path = _resolve_recorded_path(recorded_path or "")
        if _hash_matches(path, receipt.get(field), errors, f"{asset} outcome receipt {field}"):
            input_hashes[str(path)] = str(receipt[field])


def validate_asset(asset: str, root: Path) -> dict[str, Any]:
    errors: list[str] = []
    input_hashes: dict[str, str] = {}
    for filename in INPUT_FILENAMES:
        path = root / filename
        if path.is_file():
            input_hashes[str(path.resolve())] = sha256_file(path)
        else:
            errors.append(f"{asset}: missing required outcome artifact {path}")

    manifest = _read_json(root / "run_manifest.json", errors, f"{asset} manifest")
    summary = _read_json(
        root / "intermediate_consumer_outcome_summary.json",
        errors,
        f"{asset} outcome summary",
    )
    counts = _read_json(
        root / "intermediate_consumer_outcome_counts.json",
        errors,
        f"{asset} outcome counts",
    )
    validation = _read_json(
        root / "intermediate_consumer_outcome_validation.json",
        errors,
        f"{asset} outcome validation",
    )
    receipt = _read_json(
        root / "intermediate_consumer_outcome_receipt.json",
        errors,
        f"{asset} outcome receipt",
    )
    per_run_path = root / "intermediate_consumer_outcome_per_run.csv"
    if not all(
        isinstance(value, dict)
        for value in (manifest, counts, validation, receipt)
    ) or not isinstance(summary, list):
        return {
            "asset": asset,
            "root": str(root.resolve()),
            "passed": False,
            "errors": errors,
            "input_hashes": input_hashes,
            "method_rows": [],
            "counts": counts if isinstance(counts, dict) else {},
            "counts_verified": False,
            "outcome": {},
        }

    config = manifest.get("config") or {}
    expected_config = {
        "dataset": DATASETS[asset],
        "data_dir": "data",
        "checkpoint_root": (
            f"checkpoints_training_seeds/public_itch_1s_panel_50ep/{asset}"
        ),
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
        "device": "mps",
    }
    if manifest.get("status") != "completed":
        errors.append(f"{asset}: manifest status={manifest.get('status')}")
    if manifest.get("completed_cells") != 9 or manifest.get("failed_cells") != 0:
        errors.append(f"{asset}: manifest terminal cell counts are not 9 completed / 0 failed")
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            errors.append(
                f"{asset}: config {field}={config.get(field)} expected={expected}"
            )
    risk_contract = manifest.get("risk_contract") or {}
    if (
        risk_contract.get("threshold_train_snapshots") != 70_200
        or risk_contract.get("threshold_source") != "training_split_only"
        or not _finite(risk_contract.get("target_size_shares"))
        or not _finite(risk_contract.get("slippage_alarm_ticks"))
        or not _finite(risk_contract.get("liquidity_alarm_target_units"))
        or not isinstance(risk_contract.get("threshold_index_sha256"), str)
        or len(risk_contract.get("threshold_index_sha256", "")) != 64
    ):
        errors.append(f"{asset}: frozen training-only risk contract mismatch")

    cells = _load_cell_payloads(asset, root, manifest, errors)
    for path, _ in cells:
        input_hashes[str(path.resolve())] = sha256_file(path)
    _validate_sources_and_data(asset, manifest, errors, input_hashes)
    _validate_checkpoints(asset, manifest, cells, errors, input_hashes)
    _validate_reference(asset, manifest, errors, input_hashes)
    raw_science_failures = _validate_cell_denominators(asset, cells, errors)
    recomputed_counts = _counts_from_cells(cells) if len(cells) == 9 else {}
    count_error_start = len(errors)
    if recomputed_counts:
        _validate_counts(asset, counts, recomputed_counts, errors)
    counts_verified = bool(recomputed_counts) and len(errors) == count_error_start
    per_run_rows = _load_and_validate_per_run_csv(asset, per_run_path, errors)
    method_rows = _validate_summary(asset, summary, per_run_rows, errors)
    outcome = _validate_outcome_gate(
        asset, validation, counts, raw_science_failures, errors
    )
    _validate_outcome_receipt(asset, root, receipt, errors, input_hashes)
    outcome["reclassified_experiment_integrity_verified"] = not errors
    outcome["scientific_result_class"] = (
        outcome.get("registered_scientific_result_class")
        if not errors
        else "withheld_integrity_failure"
    )
    return {
        "asset": asset,
        "dataset": config.get("dataset"),
        "root": str(root.resolve()),
        "passed": not errors,
        "errors": errors,
        "cells": len(cells),
        "input_hashes": dict(sorted(input_hashes.items())),
        "method_rows": method_rows,
        "counts": counts,
        "counts_verified": counts_verified,
        "outcome": outcome,
    }


def dense_ranks(rows: list[dict[str, Any]], metric: Metric) -> dict[str, int]:
    """Rank distinct values; ties share rank and do not consume the next rank."""
    ordered = sorted(
        ((row["method_id"], float(row[f"{metric.key}_mean"])) for row in rows),
        key=lambda item: item[1],
        reverse=metric.direction == "up",
    )
    representatives: list[float] = []
    ranks: dict[str, int] = {}
    for method_id, value in ordered:
        rank = None
        for index, representative in enumerate(representatives, start=1):
            if math.isclose(value, representative, rel_tol=1e-9, abs_tol=1e-12):
                rank = index
                break
        if rank is None:
            representatives.append(value)
            rank = len(representatives)
        ranks[method_id] = rank
    return ranks


def add_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    for asset in ASSETS:
        asset_rows = [row for row in output if row["asset"] == asset]
        for metric in ALL_METRICS:
            ranks = dense_ranks(asset_rows, metric)
            for row in asset_rows:
                row[f"{metric.key}_direction"] = metric.direction
                row[f"{metric.key}_rank"] = ranks[row["method_id"]]
    return output


def pool_counts(asset_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for method_id in METHODS:
        pooled: dict[str, Any] = {
            "assets": len(asset_results),
            "anytime_frame_requests": 0,
            "anytime_frame_rejections": 0,
            "anytime_block_requests": 0,
            "anytime_block_rejections": 0,
            "endpoint_frame_requests": 0,
            "endpoint_frame_rejections": 0,
            "trajectories": 0,
            "trajectories_with_any_anytime_rejection": 0,
            "rejection_reason_counts": {reason: 0 for reason in REJECTION_REASONS},
        }
        for result in asset_results:
            counts = result["counts"][method_id]
            for field in (
                "anytime_frame_requests",
                "anytime_frame_rejections",
                "anytime_block_requests",
                "anytime_block_rejections",
                "endpoint_frame_requests",
                "endpoint_frame_rejections",
                "trajectories",
                "trajectories_with_any_anytime_rejection",
            ):
                pooled[field] += int(counts[field])
            for reason in REJECTION_REASONS:
                pooled["rejection_reason_counts"][reason] += int(
                    counts["rejection_reason_counts"][reason]
                )
        pooled["anytime_frame_acceptance_rate"] = 1.0 - (
            pooled["anytime_frame_rejections"] / pooled["anytime_frame_requests"]
        )
        pooled["anytime_block_acceptance_rate"] = 1.0 - (
            pooled["anytime_block_rejections"] / pooled["anytime_block_requests"]
        )
        pooled["completed_without_anytime_rejection_rate"] = 1.0 - (
            pooled["trajectories_with_any_anytime_rejection"]
            / pooled["trajectories"]
        )
        output[method_id] = pooled
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(row: dict[str, Any], metric: Metric) -> str:
    mean = float(row[f"{metric.key}_mean"])
    std = float(row[f"{metric.key}_std"])
    scale = 100.0 if metric.percent else 1.0
    text = f"{scale*mean:.{metric.digits}f} +/- {scale*std:.{metric.digits}f}"
    rank = int(row[f"{metric.key}_rank"])
    if rank == 1:
        return f"**{text}**"
    if rank == 2:
        return f"<u>{text}</u>"
    return text


def _render_metric_table(rows: list[dict[str, Any]], metrics: Iterable[Metric]) -> list[str]:
    metrics = tuple(metrics)
    arrows = {"up": "↑", "down": "↓"}
    lines = [
        "| Asset | Method | "
        + " | ".join(
            f"{metric.label} {arrows[metric.direction]}"
            + (" (%)" if metric.percent else "")
            for metric in metrics
        )
        + " |",
        "|---|---|" + "---:|" * len(metrics),
    ]
    for asset in ASSETS:
        for row in [item for item in rows if item["asset"] == asset]:
            lines.append(
                f"| {asset} | {row['method']} | "
                + " | ".join(_format_metric(row, metric) for metric in metrics)
                + " |"
            )
    return lines


def _count_ranks(
    count_rows: list[dict[str, Any]], asset: str | None, field: str
) -> dict[str, int]:
    selected = [row for row in count_rows if asset is None or row.get("asset") == asset]
    ordered = sorted((row["method_id"], int(row[field])) for row in selected)
    distinct: list[int] = []
    ranks: dict[str, int] = {}
    for method, value in ordered:
        if value not in distinct:
            distinct.append(value)
        ranks[method] = distinct.index(value) + 1
    return ranks


def _style_count(value: str, rank: int) -> str:
    if rank == 1:
        return f"**{value}**"
    if rank == 2:
        return f"<u>{value}</u>"
    return value


def render_markdown(
    rows: list[dict[str, Any]],
    count_rows: list[dict[str, Any]],
    pooled_rows: list[dict[str, Any]],
    asset_results: list[dict[str, Any]],
    result_class: str,
) -> str:
    lines = [
        "# P0 Cross-Asset Intermediate-Consumer Summary",
        "",
        "Experiment integrity: **verified for AAPL, AMZN, and GOOG**.  "
        f"Cross-asset result class: **{result_class}**.",
        "",
        "This is a descriptive per-asset comparison. The three assets are not",
        "treated as random replicates, and correlated reverse-diffusion frames",
        "are not treated as independent samples. No cross-asset p-values are",
        "computed. Evaluation seeds are averaged within each trained checkpoint;",
        "each cell reports mean +/- sample SD across three trained checkpoints.",
        "",
        "Within each asset and metric, the best distinct value is **bold** and the",
        "second-best distinct value is <u>underlined</u>. Ties use a relative",
        "tolerance of 1e-9 (absolute 1e-12), share a dense rank, and therefore all",
        "tie-best values are bold; if no second distinct value exists, no value is",
        "underlined. Arrows state the preferred direction.",
        "",
        "## Immediate-consumption and intervention metrics",
        "",
        *_render_metric_table(rows, INTERFACE_METRICS),
        "",
        "## Endpoint risk/fidelity guardrails",
        "",
        *_render_metric_table(rows, FIDELITY_METRICS),
        "",
        "## Exact per-asset rejection counts",
        "",
        "| Asset | Method | Anytime frame rejects / requests ↓ | Anytime block rejects / requests ↓ | Trajectories with rejection / total ↓ | Endpoint frame rejects / requests ↓ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    count_fields = (
        ("anytime_frame_rejections", "anytime_frame_requests"),
        ("anytime_block_rejections", "anytime_block_requests"),
        ("trajectories_with_any_anytime_rejection", "trajectories"),
        ("endpoint_frame_rejections", "endpoint_frame_requests"),
    )
    for asset in ASSETS:
        ranks = {field: _count_ranks(count_rows, asset, field) for field, _ in count_fields}
        for row in [item for item in count_rows if item["asset"] == asset]:
            cells = []
            for numerator, denominator in count_fields:
                text = f"{row[numerator]} / {row[denominator]}"
                cells.append(_style_count(text, ranks[numerator][row["method_id"]]))
            lines.append(
                f"| {asset} | {row['method']} | " + " | ".join(cells) + " |"
            )
    lines.extend(
        [
            "",
            "## Pooled exact counts across the three fixed assets",
            "",
            "Pooling below is integer bookkeeping only; it is not an inferential",
            "sample-size increase.",
            "",
            "| Method | Anytime frame rejects / requests ↓ | Anytime block rejects / requests ↓ | Trajectories with rejection / total ↓ | Endpoint frame rejects / requests ↓ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    pooled_ranks = {
        field: _count_ranks(pooled_rows, None, field) for field, _ in count_fields
    }
    for row in pooled_rows:
        cells = []
        for numerator, denominator in count_fields:
            text = f"{row[numerator]} / {row[denominator]}"
            cells.append(
                _style_count(text, pooled_ranks[numerator][row["method_id"]])
            )
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Per-asset registered outcomes", ""])
    for result in asset_results:
        outcome = result["outcome"]
        failures = outcome.get("failed_science_checks") or []
        suffix = f" — failed checks: {', '.join(failures)}" if failures else ""
        lines.append(
            f"- {result['asset']}: "
            f"`{outcome.get('scientific_result_class')}`{suffix}"
        )
    lines.extend(
        [
            "",
            "The admissible claim boundary is immediate consumption under the",
            "paper's continuous snapshot contract and lower projection burden.",
            "These results do not establish queue/fill realism, message-level",
            "exchange legality, causal control, or superior trading performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def _flatten_count_rows(asset_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in asset_results:
        for method_id in METHODS:
            counts = result["counts"][method_id]
            row = {
                "asset": result["asset"],
                "method_id": method_id,
                "method": METHOD_LABELS[method_id],
                **{key: value for key, value in counts.items() if key != "rejection_reason_counts"},
            }
            for reason, value in counts["rejection_reason_counts"].items():
                row[f"rejection_reason_{reason}"] = value
            rows.append(row)
    return rows


def _flatten_pooled_rows(pooled: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in METHODS:
        counts = pooled[method_id]
        row = {
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
            **{key: value for key, value in counts.items() if key != "rejection_reason_counts"},
        }
        for reason, value in counts["rejection_reason_counts"].items():
            row[f"rejection_reason_{reason}"] = value
        rows.append(row)
    return rows


def write_outputs(input_root: Path, output_root: Path) -> int:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    observed_assets = {
        path.name for path in input_root.iterdir() if path.is_dir()
    } if input_root.is_dir() else set()
    root_errors: list[str] = []
    if observed_assets != set(ASSETS):
        root_errors.append(
            f"input asset directories {sorted(observed_assets)} != {list(ASSETS)}"
        )
    asset_results = [
        validate_asset(asset, input_root / asset) for asset in ASSETS
    ]
    all_errors = root_errors + [
        error for result in asset_results for error in result["errors"]
    ]
    passed_assets = [result["asset"] for result in asset_results if result["passed"]]
    failed_assets = [result["asset"] for result in asset_results if not result["passed"]]
    verified_failure_counts = {
        result["asset"]: result["counts"]
        for result in asset_results
        if result.get("counts_verified")
    }
    validation = {
        "schema_version": 1,
        "status": "passed" if not all_errors else "failed",
        "experiment_integrity_verified": not all_errors,
        "scientific_summary_status": (
            "eligible" if not all_errors else "withheld_integrity_failure"
        ),
        "integrity_reclassification": {
            "rule": INTEGRITY_RECLASSIFICATION_RULE,
            "rule_sha256": sha256_json(INTEGRITY_RECLASSIFICATION_RULE),
            "asset_flags": {
                result["asset"]: {
                    "legacy_experiment_integrity_verified": result[
                        "outcome"
                    ].get("legacy_experiment_integrity_verified"),
                    "reclassified_experiment_integrity_verified": result[
                        "outcome"
                    ].get("reclassified_experiment_integrity_verified"),
                }
                for result in asset_results
            },
        },
        "contract": {
            "assets": list(ASSETS),
            "cells_per_asset": 9,
            "methods_per_cell": 3,
            "total_cells": 27,
            "trajectories_per_asset_method": 5_760,
            "anytime_blocks_per_asset_method": 1_152_000,
            "anytime_frames_per_asset_method": 11_520_000,
            "inference": "descriptive_per_asset_and_pooled_exact_counts_only",
            "correlated_frames_are_not_independent_samples": True,
            "assets_are_not_random_replicates": True,
            "p_values_computed": False,
        },
        "asset_accounting": {
            "expected_assets": list(ASSETS),
            "observed_asset_directories": sorted(observed_assets),
            "integrity_passed_assets": passed_assets,
            "integrity_failed_assets": failed_assets,
            "expected_asset_count": 3,
            "observed_asset_directory_count": len(observed_assets),
            "failed_asset_count": len(failed_assets),
            "expected_cells": 27,
            "observed_completed_cells": sum(
                int(result.get("cells", 0) or 0) for result in asset_results
            ),
        },
        "asset_results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"method_rows", "counts", "input_hashes"}
            }
            for result in asset_results
        ],
        "validation_only_exact_counts": (
            verified_failure_counts if all_errors else {}
        ),
        "validation_only_counts_note": (
            "These exact counts are retained only for failure auditing. They are "
            "not pooled and no scientific summary is emitted when integrity fails."
        ),
        "errors": all_errors,
    }
    input_hashes = {
        path: digest
        for result in asset_results
        for path, digest in result["input_hashes"].items()
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-", dir=output_root.parent
        )
    )
    try:
        validation_path = staging_root / "cross_asset_validation.json"
        validation_path.write_text(json.dumps(validation, indent=2) + "\n")
        if all_errors:
            receipt = {
                "schema_version": 1,
                "status": "failed",
                "publication": "validation_only",
                "analysis_script": str(Path(__file__).resolve()),
                "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
                "input_root": str(input_root),
                "output_root": str(output_root),
                "validation_sha256": sha256_file(validation_path),
                "integrity_reclassification": validation[
                    "integrity_reclassification"
                ],
                "input_file_sha256": dict(sorted(input_hashes.items())),
                "outputs_withheld": True,
                "withheld_outputs": [
                    "cross_asset_summary.json",
                    "cross_asset_per_asset_method.csv",
                    "cross_asset_exact_counts.csv",
                    "cross_asset_pooled_exact_counts.csv",
                    "cross_asset_summary.md",
                ],
            }
            (staging_root / "cross_asset_receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            if output_root.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output root: {output_root}"
                )
            staging_root.rename(output_root)
            print(json.dumps(validation, indent=2))
            return 1

        method_rows = add_ranks(
            [row for result in asset_results for row in result["method_rows"]]
        )
        count_rows = _flatten_count_rows(asset_results)
        pooled = pool_counts(asset_results)
        pooled_rows = _flatten_pooled_rows(pooled)
        no_go_assets = [
            result["asset"]
            for result in asset_results
            if result["outcome"]["pre_registered_strict_go"] is not True
        ]
        result_class = (
            "all_assets_strict_go"
            if not no_go_assets
            else "integrity_verified_with_strict_no_go_assets"
        )
        summary = {
            "schema_version": 1,
            "status": "validated",
            "result_class": result_class,
            "strict_no_go_assets": no_go_assets,
            "inference_contract": validation["contract"],
            "ranking_contract": {
                "scope": "within_asset_per_metric",
                "best": "bold",
                "second_best_distinct": "underline",
                "direction": {
                    metric.key: metric.direction for metric in ALL_METRICS
                },
                "tie_rule": "dense rank with rel_tol=1e-9 and abs_tol=1e-12",
            },
            "asset_outcomes": {
                result["asset"]: result["outcome"] for result in asset_results
            },
            "per_asset_method_rows": method_rows,
            "per_asset_exact_counts": {
                result["asset"]: result["counts"] for result in asset_results
            },
            "pooled_exact_counts": pooled,
            "claim_boundary": (
                "Immediate consumption under the continuous snapshot contract and "
                "projection burden only; not queue/fill realism, message legality, "
                "causal control, or trading performance."
            ),
        }
        summary_json = staging_root / "cross_asset_summary.json"
        method_csv = staging_root / "cross_asset_per_asset_method.csv"
        count_csv = staging_root / "cross_asset_exact_counts.csv"
        pooled_csv = staging_root / "cross_asset_pooled_exact_counts.csv"
        markdown_path = staging_root / "cross_asset_summary.md"
        summary_json.write_text(json.dumps(summary, indent=2) + "\n")
        _write_csv(method_csv, method_rows)
        _write_csv(count_csv, count_rows)
        _write_csv(pooled_csv, pooled_rows)
        markdown_path.write_text(
            render_markdown(
                method_rows, count_rows, pooled_rows, asset_results, result_class
            )
        )
        output_hashes = {
            str(path.name): sha256_file(path)
            for path in (
                validation_path,
                summary_json,
                method_csv,
                count_csv,
                pooled_csv,
                markdown_path,
            )
        }
        receipt = {
            "schema_version": 1,
            "status": "validated",
            "publication": "scientific_summary",
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "integrity_reclassification": validation[
                "integrity_reclassification"
            ],
            "input_file_sha256": dict(sorted(input_hashes.items())),
            "output_file_sha256": output_hashes,
        }
        (staging_root / "cross_asset_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        if output_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output root: {output_root}"
            )
        staging_root.rename(output_root)
        print((output_root / "cross_asset_summary.md").read_text())
        print(json.dumps(validation, indent=2))
        return 0
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return write_outputs(args.input_root, args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())

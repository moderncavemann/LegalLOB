#!/usr/bin/env python3
"""Execute the frozen 24-cell multi-asset event-time training campaign once.

The orchestrator owns a new campaign root, records the full denominator before
launching a child process, and never retries a started cell.  All four assets
are trained independently with the same optimizer-update budget.  A successful
artifact gate proves the update count from ``final.pt`` rather than inferring it
from epochs or filenames.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CAMPAIGN_ID = "p4_public_itch_event_time_20260807_v1"
CHECKPOINT_ROOT_ARG = (
    "checkpoints_training_seeds/p4_public_itch_event_time_20260807_v1"
)
CONTRACT_PATH = "experiments/multi_asset_event_time/protocol.md"
ASSETS = (
    ("AAPL", "PUBLIC_ITCH_2019_SAMPLES_AAPL_event"),
    ("AMZN", "PUBLIC_ITCH_2019_SAMPLES_AMZN_event"),
    ("GOOG", "PUBLIC_ITCH_2019_SAMPLES_GOOG_event"),
    ("INTC", "PUBLIC_ITCH_2019_SAMPLES_INTC_event"),
)
TRAIN_SEEDS = (0, 1, 2)
REPRESENTATIONS = ("legal_lob", "matched_lob")
EPOCHS = 50
TRAIN_BATCHES_PER_EPOCH = 548
VAL_BATCHES_PER_EPOCH = 183
TARGET_OPTIMIZER_STEPS = EPOCHS * TRAIN_BATCHES_PER_EPOCH
BATCH_SIZE = 128
VALIDATION_SEED = 10000
LEDGER_NAME = "p4_training_ledger.json"
SUMMARY_NAME = "p4_training_summary.json"
LOCK_NAME = ".p4_event_training.lock"
TERMINAL_STATUSES = frozenset({"completed", "failed"})

EXPECTED_DAYS = [20190130, 20190327, 20190730, 20190830, 20191030, 20191230]
EXPECTED_DAY_MAPPING = {
    "20190130": 0,
    "20190327": 0,
    "20190730": 0,
    "20190830": 1,
    "20191030": 2,
    "20191230": 2,
}

SOURCE_PATHS = (
    CONTRACT_PATH,
    "experiments/multi_asset_event_time/run_training.py",
    "experiments/multi_asset_event_time/train_fixed_steps.py",
    "scripts/train_v2.py",
    "tests/test_event_time_training.py",
    "tests/test_train_v2_uniform_sampling.py",
    "genlob/models/genlob_v2.py",
    "genlob/models/sde.py",
    "genlob/data/nasdaq.py",
    "genlob/data/normalize.py",
    "genlob/data/action_templates.py",
    "genlob/utils/device.py",
    "genlob/utils/legal_lob.py",
    "genlob/utils/physics.py",
)

DATA_PATHS = (
    "DATASET_CARD.md",
    "actions.npy",
    "day_id.npy",
    "manifest.yaml",
    "metadata.json",
    "normalization.json",
    "schema_validation.json",
    "segment_id.npy",
    "source_row_idx.npy",
    "split.npy",
    "states.npy",
    "states_raw.npy",
    "timestamps.npy",
)

EXPECTED_MODEL_CONFIG = {
    "input_dim": 2,
    "output_dim": 2,
    "base_channels": 128,
    "emb_dim": 128,
    "num_layers": 16,
    "n_features": 20,
    "num_heads": 8,
    "num_steps": 100,
    "action_dim": 5,
    "dropout": 0.0,
    "cfg_drop_prob": 0.1,
}


class CampaignError(RuntimeError):
    """A fail-closed study protocol or artifact error."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"Required provenance file is missing: {path}")
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = file_sha256(path)
    return record


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any], *, create: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if create:
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise CampaignError(f"Refusing to overwrite immutable file: {path}") from exc
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _git_head(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _dataset_dir(project_root: Path, dataset: str) -> Path:
    return project_root / "data" / "processed" / "snapshot_l2" / dataset


def validate_dataset_contract(
    project_root: Path, symbol: str, dataset: str
) -> dict[str, Any]:
    dataset_dir = _dataset_dir(project_root, dataset)
    metadata = read_json(dataset_dir / "metadata.json")
    normalization = read_json(dataset_dir / "normalization.json")
    schema = read_json(dataset_dir / "schema_validation.json")

    required_metadata = {
        "dataset_id": dataset,
        "symbol": symbol,
        "sampling": "event",
        "source": "nasdaq_public_itch_samples",
        "has_true_timestamps": True,
        "has_true_day_id": True,
        "has_message_data": False,
        "has_raw_itch_source": True,
        "processed_event_l3_available": False,
        "action_label_status": "itch_event_proxy_diagnostic_not_causal",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise CampaignError(
                f"{dataset} metadata {key!r}: expected {expected!r}, "
                f"got {metadata.get(key)!r}"
            )
    if metadata.get("days") != EXPECTED_DAYS:
        raise CampaignError(f"{dataset} does not contain the frozen six dates")
    split_policy = metadata.get("split_policy") or {}
    observed_mapping = {
        str(key): int(value)
        for key, value in (split_policy.get("day_mapping") or {}).items()
    }
    if (
        split_policy.get("policy") != "explicit_day_split"
        or observed_mapping != EXPECTED_DAY_MAPPING
        or split_policy.get("purge_gap") != 100
    ):
        raise CampaignError(f"{dataset} split policy differs from the frozen contract")
    if (
        normalization.get("fit_split") != "train_only"
        or normalization.get("price_reference_protocol") != "history_last_mid"
    ):
        raise CampaignError(f"{dataset} normalization is not train-only/history-relative")
    for key in ("min_spread_normalized", "min_level_gap_normalized"):
        value = normalization.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise CampaignError(f"{dataset} has no positive {key}")
    if schema.get("status") not in {"pass", "pass_with_warnings"} or schema.get("errors"):
        raise CampaignError(f"{dataset} schema validation did not pass without errors")

    arrays = {
        name: np.load(dataset_dir / name, mmap_mode="r")
        for name in (
            "states.npy",
            "states_raw.npy",
            "actions.npy",
            "split.npy",
            "day_id.npy",
            "segment_id.npy",
            "source_row_idx.npy",
            "timestamps.npy",
        )
    }
    state_count = int(arrays["states.npy"].shape[0])
    if tuple(arrays["states.npy"].shape[1:]) != (2, 20):
        raise CampaignError(f"{dataset} states.npy has the wrong state shape")
    if arrays["states_raw.npy"].shape != arrays["states.npy"].shape:
        raise CampaignError(f"{dataset} raw/normalized state shapes differ")
    action_count = int(arrays["actions.npy"].shape[0])
    if state_count not in {action_count, action_count + 1}:
        raise CampaignError(f"{dataset} state/action lengths do not align")
    aligned_count = action_count
    for name in ("split.npy", "day_id.npy", "segment_id.npy", "source_row_idx.npy", "timestamps.npy"):
        if int(arrays[name].shape[0]) not in {aligned_count, state_count}:
            raise CampaignError(f"{dataset} {name} length does not align")
    split = np.asarray(arrays["split.npy"][:aligned_count])
    counts = {str(value): int(np.sum(split == value)) for value in (0, 1, 2)}
    required_train_rows = TRAIN_BATCHES_PER_EPOCH * BATCH_SIZE + 60
    required_val_rows = VAL_BATCHES_PER_EPOCH * BATCH_SIZE + 60
    if counts["0"] < required_train_rows or counts["1"] < required_val_rows:
        raise CampaignError(f"{dataset} is too small for the frozen capped loaders")
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "state_count": state_count,
        "aligned_state_action_count": aligned_count,
        "split_counts": counts,
        "normalization": {
            "fit_split": normalization["fit_split"],
            "price_reference_protocol": normalization["price_reference_protocol"],
            "min_spread_normalized": float(normalization["min_spread_normalized"]),
            "min_level_gap_normalized": float(normalization["min_level_gap_normalized"]),
        },
        "schema_status": schema["status"],
        "schema_warnings": list(schema.get("warnings") or []),
    }


def build_training_command(
    dataset: str, save_dir_arg: str, seed: int, representation: str
) -> list[str]:
    return [
        sys.executable,
        "experiments/multi_asset_event_time/train_fixed_steps.py",
        "--dataset", dataset,
        "--data-dir", "data",
        "--hist-len", "50",
        "--pred-len", "10",
        "--batch-size", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--max-train-batches", str(TRAIN_BATCHES_PER_EPOCH),
        "--max-val-batches", str(VAL_BATCHES_PER_EPOCH),
        "--representation-stat-batches", str(TRAIN_BATCHES_PER_EPOCH),
        "--cfg-drop", "0.1",
        "--lr", "0.0001",
        "--early-stopping-patience", "0",
        "--representation", representation,
        "--action-conditioning", "none",
        "--price-reference", "auto",
        "--physics-loss-weight", "0",
        "--action-contrastive-weight", "0",
        "--save-every", "1000",
        "--save-dir", save_dir_arg,
        "--num-workers", "0",
        "--seed", str(seed),
        "--validation-seed", str(VALIDATION_SEED),
        "--device", "mps",
    ]


def expected_checkpoint_args(
    dataset: str,
    save_dir_arg: str,
    seed: int,
    representation: str,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "data_dir": "data",
        "hist_len": 50,
        "pred_len": 10,
        "base_channels": 128,
        "num_layers": 16,
        "action_dim": 5,
        "action_conditioning": "none",
        "representation": representation,
        "price_reference": "history_last_mid",
        "representation_stat_batches": TRAIN_BATCHES_PER_EPOCH,
        "representation_eps": 1e-6,
        "beta_min": 0.1,
        "beta_max": 20.0,
        "num_scales": 100,
        "continuous": False,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "clip_grad": 1.0,
        "ema_rate": 0.999,
        "cfg_drop": 0.1,
        "physics_loss_weight": 0.0,
        "action_contrastive_weight": 0.0,
        "action_margin": 0.0,
        "allow_diagnostic_action_loss": False,
        "min_spread": float(normalization["min_spread_normalized"]),
        "min_gap": float(normalization["min_level_gap_normalized"]),
        "save_dir": save_dir_arg,
        "save_every": 1000,
        "resume": None,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "log_every": 50,
        "max_train_batches": TRAIN_BATCHES_PER_EPOCH,
        "max_val_batches": VAL_BATCHES_PER_EPOCH,
        "fixed_train_order": False,
        "device": "mps",
        "num_workers": 0,
        "seed": seed,
        "validation_seed": VALIDATION_SEED,
        "p4_uniform_representation_stats": True,
        "p4_uniform_validation": True,
        "p4_training_entrypoint": "experiments/multi_asset_event_time/train_fixed_steps.py",
    }


def build_cell_plans(
    project_root: Path,
    checkpoint_root_arg: str,
    dataset_contracts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    checkpoint_root = (project_root / checkpoint_root_arg).resolve()
    slot = 0
    for symbol, dataset in ASSETS:
        normalization = dataset_contracts[symbol]["normalization"]
        for seed in TRAIN_SEEDS:
            for representation in REPRESENTATIONS:
                slot += 1
                cell_id = f"{symbol.lower()}__train_seed{seed}__{representation}"
                save_dir_arg = (
                    f"{checkpoint_root_arg}/{symbol}/train_seed{seed}/{representation}"
                )
                save_dir = (project_root / save_dir_arg).resolve()
                log_path = checkpoint_root / "_orchestrator" / "logs" / f"{cell_id}.log"
                command = build_training_command(
                    dataset, save_dir_arg, seed, representation
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        "denominator_slot": slot,
                        "symbol": symbol,
                        "dataset": dataset,
                        "train_seed": seed,
                        "representation": representation,
                        "save_dir_arg": save_dir_arg,
                        "save_dir": str(save_dir),
                        "log_path": str(log_path),
                        "command": command,
                        "command_shell": shlex.join(command),
                        "command_sha256": object_sha256(command),
                        "expected_checkpoint_args": expected_checkpoint_args(
                            dataset,
                            save_dir_arg,
                            seed,
                            representation,
                            normalization,
                        ),
                    }
                )
    return cells


def validate_exact_cell_denominator(cells: list[dict[str, Any]]) -> None:
    expected = []
    slot = 0
    for symbol, dataset in ASSETS:
        for seed in TRAIN_SEEDS:
            for representation in REPRESENTATIONS:
                slot += 1
                expected.append(
                    (
                        slot,
                        symbol,
                        dataset,
                        seed,
                        representation,
                        f"{symbol.lower()}__train_seed{seed}__{representation}",
                    )
                )
    observed = [
        (
            cell.get("denominator_slot"),
            cell.get("symbol"),
            cell.get("dataset"),
            cell.get("train_seed"),
            cell.get("representation"),
            cell.get("cell_id"),
        )
        for cell in cells
    ]
    if observed != expected:
        raise CampaignError("Training ledger is not the frozen canonical 24-cell matrix")


def build_campaign_snapshot(
    project_root: Path,
    checkpoint_root_arg: str = CHECKPOINT_ROOT_ARG,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    dataset_contracts = {
        symbol: validate_dataset_contract(project_root, symbol, dataset)
        for symbol, dataset in ASSETS
    }
    source_files = {
        relative: file_record(project_root / relative) for relative in SOURCE_PATHS
    }
    data_files: dict[str, dict[str, Any]] = {}
    for symbol, dataset in ASSETS:
        dataset_dir = _dataset_dir(project_root, dataset)
        for name in DATA_PATHS:
            data_files[f"{symbol}/{name}"] = file_record(dataset_dir / name)
    cells = build_cell_plans(
        project_root, checkpoint_root_arg, dataset_contracts
    )
    validate_exact_cell_denominator(cells)
    checkpoint_root = (project_root / checkpoint_root_arg).resolve()
    uname = os.uname()
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "contract_path": str((project_root / CONTRACT_PATH).resolve()),
        "research_boundary": (
            "Separately trained event-time sampled 2019 XNAS Level-2 snapshots; "
            "not message-level generation, replay, queue/fill, causal action, "
            "unseen-asset transfer, or cross-venue generalization."
        ),
        "checkpoint_root_arg": checkpoint_root_arg,
        "checkpoint_root": str(checkpoint_root),
        "dataset_contracts": dataset_contracts,
        "protocol": {
            "assets": [symbol for symbol, _ in ASSETS],
            "datasets": [dataset for _, dataset in ASSETS],
            "train_seeds": list(TRAIN_SEEDS),
            "representations": list(REPRESENTATIONS),
            "epochs": EPOCHS,
            "train_batches_per_epoch": TRAIN_BATCHES_PER_EPOCH,
            "target_optimizer_steps": TARGET_OPTIMIZER_STEPS,
            "val_batches_per_epoch": VAL_BATCHES_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "representation_stat_sampling": "uniform_train_only",
            "validation_sampling": "uniform_validation_only",
            "checkpoint_for_evaluation": "final.pt",
            "early_stopping_patience": 0,
            "automatic_retry": False,
        },
        "denominators": {
            "expected_cells": 24,
            "assets": 4,
            "training_seeds_per_asset": 3,
            "representations_per_seed": 2,
            "expected_by_asset": {symbol: 6 for symbol, _ in ASSETS},
            "expected_by_representation": {
                representation: 12 for representation in REPRESENTATIONS
            },
            "cell_ids": [cell["cell_id"] for cell in cells],
        },
        "cells": cells,
        "source_files": source_files,
        "data_files": data_files,
        "source_bundle_sha256": object_sha256(source_files),
        "data_bundle_sha256": object_sha256(data_files),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "uname": {
                "sysname": uname.sysname,
                "release": uname.release,
                "version": uname.version,
                "machine": uname.machine,
            },
            "git_head": _git_head(project_root),
        },
    }


def _check_receipt_group(
    expected_group: dict[str, dict[str, Any]], *, full_hash: bool
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    observed_group: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for logical_name, expected in sorted(expected_group.items()):
        try:
            observed = file_record(
                Path(expected["path"]), include_hash=full_hash
            )
        except Exception as exc:
            errors.append(
                {
                    "logical_name": logical_name,
                    "kind": "unreadable_or_missing",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        observed_group[logical_name] = observed
        fields = ("path", "size_bytes", "mtime_ns", "sha256") if full_hash else (
            "path",
            "size_bytes",
            "mtime_ns",
        )
        for field in fields:
            if observed.get(field) != expected.get(field):
                errors.append(
                    {
                        "logical_name": logical_name,
                        "kind": f"{field}_mismatch",
                        "expected": expected.get(field),
                        "observed": observed.get(field),
                    }
                )
    return observed_group, errors


def check_frozen_receipts(
    snapshot: dict[str, Any], phase: str, *, full_data_hash: bool = False
) -> dict[str, Any]:
    _, source_errors = _check_receipt_group(
        snapshot["source_files"], full_hash=True
    )
    _, data_errors = _check_receipt_group(
        snapshot["data_files"], full_hash=full_data_hash
    )
    errors = [
        *({"group": "source_files", **error} for error in source_errors),
        *({"group": "data_files", **error} for error in data_errors),
    ]
    return {
        "phase": phase,
        "checked_at_utc": utc_now(),
        "full_data_hash": full_data_hash,
        "status": "passed" if not errors else "drift",
        "errors": errors,
    }


def ledger_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    validate_exact_cell_denominator(ledger["cells"])
    counts = {status: 0 for status in ("completed", "failed", "not_started", "running")}
    by_asset = {
        symbol: {status: 0 for status in counts} | {"expected": 0}
        for symbol, _ in ASSETS
    }
    by_representation = {
        representation: {status: 0 for status in counts} | {"expected": 0}
        for representation in REPRESENTATIONS
    }
    for cell in ledger["cells"]:
        status = cell["status"]
        if status not in counts:
            raise CampaignError(f"Unknown ledger status: {status}")
        counts[status] += 1
        by_asset[cell["symbol"]]["expected"] += 1
        by_asset[cell["symbol"]][status] += 1
        by_representation[cell["representation"]]["expected"] += 1
        by_representation[cell["representation"]][status] += 1
    expected = 24
    accounted = sum(counts.values())
    if accounted != expected:
        raise CampaignError(f"Ledger accounts for {accounted}, expected {expected}")
    launched = sum(cell.get("subprocess_launch_count") == 1 for cell in ledger["cells"])
    attempted = sum(cell.get("attempt_count") == 1 for cell in ledger["cells"])
    if any(cell.get("subprocess_launch_count") not in (0, 1) for cell in ledger["cells"]):
        raise CampaignError("A cell has an invalid subprocess launch count")
    if any(cell.get("attempt_count") not in (0, 1) for cell in ledger["cells"]):
        raise CampaignError("A cell has an invalid attempt count")
    terminal = counts["completed"] + counts["failed"]
    if counts["completed"] == expected and (ledger.get("final_integrity") or {}).get("status") == "passed":
        status = "completed"
    elif terminal == expected:
        status = "terminal_with_declared_failures"
    else:
        status = "in_progress"
    return {
        "campaign_id": CAMPAIGN_ID,
        "status": status,
        "expected": expected,
        "completed": counts["completed"],
        "failed": counts["failed"],
        "not_started": counts["not_started"],
        "running": counts["running"],
        "terminal": terminal,
        "launched_once": int(launched),
        "attempted_once": int(attempted),
        "by_asset": by_asset,
        "by_representation": by_representation,
        "protocol_drift_latched": bool(ledger["protocol_drift"]["latched"]),
        "final_integrity": ledger.get("final_integrity"),
        "ledger_revision": ledger["revision"],
        "updated_at_utc": ledger["updated_at_utc"],
    }


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    ledger["revision"] = int(ledger["revision"]) + 1
    ledger["updated_at_utc"] = utc_now()
    ledger["summary"] = ledger_summary(ledger)
    ledger["summary"]["ledger_revision"] = ledger["revision"]
    ledger["summary"]["updated_at_utc"] = ledger["updated_at_utc"]
    _atomic_json(path, ledger, create=False)
    _atomic_json(path.parent / SUMMARY_NAME, ledger["summary"], create=False)


def new_ledger(snapshot: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    cells = []
    for plan in snapshot["cells"]:
        cell = dict(plan)
        cell.update(
            {
                "status": "not_started",
                "attempt_count": 0,
                "subprocess_launch_count": 0,
                "started_at_utc": None,
                "terminal_at_utc": None,
                "pid": None,
                "exit_code": None,
                "log_sha256": None,
                "artifacts": None,
                "failure": None,
                "protocol_receipt_checks": [],
            }
        )
        cells.append(cell)
    ledger = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "checkpoint_root": snapshot["checkpoint_root"],
        "created_at_utc": now,
        "updated_at_utc": now,
        "revision": 0,
        "snapshot_sha256": object_sha256(snapshot),
        "snapshot": snapshot,
        "cells": cells,
        "invocations": [],
        "recovery_events": [],
        "protocol_drift": {
            "latched": False,
            "first_detected_at_utc": None,
            "first_cell_id": None,
            "first_phase": None,
        },
        "final_integrity": None,
    }
    ledger["summary"] = ledger_summary(ledger)
    return ledger


def verify_matching_ledger(ledger: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if ledger.get("schema_version") != 1 or ledger.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("Existing root belongs to another campaign/schema")
    validate_exact_cell_denominator(ledger.get("cells") or [])
    if object_sha256(ledger.get("snapshot")) != ledger.get("snapshot_sha256"):
        raise CampaignError("Stored ledger snapshot hash is internally inconsistent")
    if object_sha256(snapshot) != ledger["snapshot_sha256"]:
        raise CampaignError(
            "Current study protocol/data/source snapshot differs from the ledger"
        )
    for stored, planned in zip(ledger["cells"], snapshot["cells"], strict=True):
        for key, value in planned.items():
            if stored.get(key) != value:
                raise CampaignError(f"Ledger plan drift for {planned['cell_id']} field {key}")
        if stored.get("status") not in {"not_started", "running", *TERMINAL_STATUSES}:
            raise CampaignError(f"Illegal status for {planned['cell_id']}")
        if stored.get("attempt_count") not in (0, 1):
            raise CampaignError(f"Illegal attempt count for {planned['cell_id']}")
        if stored.get("subprocess_launch_count") not in (0, 1):
            raise CampaignError(f"Illegal launch count for {planned['cell_id']}")


@contextlib.contextmanager
def campaign_lock(checkpoint_root: Path):
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_root / LOCK_NAME
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(
                f"Another event-time orchestrator holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize_or_load_ledger(
    checkpoint_root: Path,
    snapshot: dict[str, Any],
    *,
    resume_pending: bool,
) -> tuple[Path, dict[str, Any], bool]:
    ledger_path = checkpoint_root / LEDGER_NAME
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        verify_matching_ledger(ledger, snapshot)
        if not resume_pending:
            raise CampaignError(
                "The study ledger already exists; use --resume-pending only for "
                "never-started cells"
            )
        return ledger_path, ledger, False
    unexpected = [
        entry.name for entry in checkpoint_root.iterdir() if entry.name != LOCK_NAME
    ]
    if unexpected:
        raise CampaignError(
            "Checkpoint root exists without its authoritative ledger and is not empty: "
            f"{sorted(unexpected)}"
        )
    ledger = new_ledger(snapshot)
    _atomic_json(ledger_path, ledger, create=True)
    _atomic_json(checkpoint_root / SUMMARY_NAME, ledger["summary"], create=True)
    return ledger_path, ledger, True


def _process_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _ensure_log(cell: dict[str, Any], message: str) -> str:
    path = Path(cell["log_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((message.rstrip() + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def seal_stale_running_cells(
    ledger_path: Path, ledger: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    stale = [cell for cell in ledger["cells"] if cell["status"] == "running"]
    live = {cell["cell_id"]: cell.get("pid") for cell in stale if _process_alive(cell.get("pid"))}
    if live:
        raise CampaignError(
            f"Recorded event-time subprocesses may still be alive: {live}"
        )
    for cell in stale:
        receipt = check_frozen_receipts(snapshot, "stale_recovery")
        cell["protocol_receipt_checks"].append(receipt)
        cell.update(
            {
                "status": "failed",
                "terminal_at_utc": utc_now(),
                "failure": {
                    "kind": "unrecorded_terminal_state",
                    "message": "Started cell had no ledgered terminal state; it is retained as failed without retry.",
                },
            }
        )
        cell["log_sha256"] = _ensure_log(
            cell, "RECOVERY: started cell sealed failed without retry"
        )
        ledger["recovery_events"].append(
            {"cell_id": cell["cell_id"], "at_utc": utc_now(), "decision": "failed_no_retry"}
        )
    if stale:
        save_ledger(ledger_path, ledger)


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, float):
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise CampaignError(f"{context}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise CampaignError(f"{context}: expected {expected!r}, got {actual!r}")


def _config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise CampaignError(f"Unsupported checkpoint config type: {type(value).__name__}")


def _tensor_sha256(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("ascii")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def _validate_checkpoint_common(
    checkpoint: dict[str, Any], cell: dict[str, Any], artifact_name: str
) -> dict[str, Any]:
    required = {
        "epoch",
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "val_loss",
        "best_val_loss",
        "config",
        "args",
        "representation_stats",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise CampaignError(f"{artifact_name} missing fields: {missing}")
    args = checkpoint["args"]
    expected_args = cell["expected_checkpoint_args"]
    if not isinstance(args, dict) or set(args) != set(expected_args):
        raise CampaignError(f"{artifact_name} args differ from the frozen train_v2 namespace")
    for key, expected in expected_args.items():
        _assert_equal(args[key], expected, f"{artifact_name} args[{key!r}]")
    config = _config_dict(checkpoint["config"])
    if set(config) != set(EXPECTED_MODEL_CONFIG):
        raise CampaignError(f"{artifact_name} model config fields differ")
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        _assert_equal(config[key], expected, f"{artifact_name} config[{key!r}]")
    stats = checkpoint["representation_stats"]
    if not isinstance(stats, dict) or set(stats) != {"mean", "std"}:
        raise CampaignError(f"{artifact_name} representation stats are malformed")
    for key in ("mean", "std"):
        tensor = torch.as_tensor(stats[key])
        if tuple(tensor.shape) != (1, 2, 20, 1) or not bool(torch.isfinite(tensor).all()):
            raise CampaignError(f"{artifact_name} representation stats {key} are invalid")
    if not bool((torch.as_tensor(stats["std"]) > 0).all()):
        raise CampaignError(f"{artifact_name} representation std is not positive")
    return {
        "mean_sha256": _tensor_sha256(stats["mean"]),
        "std_sha256": _tensor_sha256(stats["std"]),
        "combined_sha256": object_sha256(
            {"mean": _tensor_sha256(stats["mean"]), "std": _tensor_sha256(stats["std"])}
        ),
    }


def _optimizer_steps(checkpoint: dict[str, Any]) -> list[int]:
    optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise CampaignError("final.pt optimizer state is missing")
    values: list[int] = []
    for state in optimizer["state"].values():
        if not isinstance(state, dict) or "step" not in state:
            raise CampaignError("An Adam parameter state has no step counter")
        raw = state["step"]
        scalar = float(raw.item() if hasattr(raw, "item") else raw)
        if not math.isfinite(scalar) or not scalar.is_integer():
            raise CampaignError(f"Non-integral Adam step counter: {scalar}")
        values.append(int(scalar))
    if not values:
        raise CampaignError("final.pt contains no Adam parameter states")
    return values


def validate_cell_artifacts(cell: dict[str, Any]) -> dict[str, Any]:
    save_dir = Path(cell["save_dir"])
    paths = {
        name: save_dir / name
        for name in ("completion.json", "history.json", "best.pt", "last.pt", "final.pt")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise CampaignError(f"Training artifacts are missing: {missing}")
    completion = read_json(paths["completion.json"])
    expected_completion_keys = {
        "status",
        "last_epoch",
        "requested_epochs",
        "best_val_loss",
        "early_stopped",
        "fixed_validation_seed",
    }
    if set(completion) != expected_completion_keys:
        raise CampaignError("completion.json schema differs from the frozen contract")
    if (
        completion["status"] != "complete"
        or completion["last_epoch"] != EPOCHS
        or completion["requested_epochs"] != EPOCHS
        or completion["early_stopped"] is not False
        or completion["fixed_validation_seed"] != VALIDATION_SEED
        or not math.isfinite(float(completion["best_val_loss"]))
    ):
        raise CampaignError("completion.json does not prove a complete fixed-budget run")
    history = read_json(paths["history.json"])
    rows = history.get("epochs")
    if not isinstance(rows, list) or len(rows) != EPOCHS:
        raise CampaignError("history.json does not contain exactly 50 epochs")
    expected_fields = {
        "epoch",
        "train_loss",
        "val_loss",
        "best_val_loss",
        "learning_rate",
        "elapsed_seconds",
        "is_best",
    }
    for expected_epoch, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != expected_fields or row.get("epoch") != expected_epoch:
            raise CampaignError("history.json fields/epoch order differ from the frozen schema")
        for key in ("train_loss", "val_loss", "best_val_loss", "learning_rate", "elapsed_seconds"):
            if not isinstance(row[key], (int, float)) or not math.isfinite(float(row[key])):
                raise CampaignError(f"history epoch {expected_epoch} has non-finite {key}")
        if not isinstance(row["is_best"], bool):
            raise CampaignError(f"history epoch {expected_epoch} has invalid is_best")

    loaded: dict[str, dict[str, Any]] = {}
    for name in ("best.pt", "last.pt", "final.pt"):
        try:
            payload = torch.load(paths[name], map_location="cpu", weights_only=False)
        except Exception as exc:
            raise CampaignError(f"Cannot load {name}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CampaignError(f"{name} payload is not a dictionary")
        loaded[name] = payload
    final_stats = _validate_checkpoint_common(loaded["final.pt"], cell, "final.pt")
    _validate_checkpoint_common(loaded["last.pt"], cell, "last.pt")
    _validate_checkpoint_common(loaded["best.pt"], cell, "best.pt")
    if loaded["final.pt"]["epoch"] != EPOCHS or loaded["last.pt"]["epoch"] != EPOCHS:
        raise CampaignError("final/last checkpoint is not the 50th epoch")
    steps = _optimizer_steps(loaded["final.pt"])
    unique_steps = sorted(set(steps))
    if unique_steps != [TARGET_OPTIMIZER_STEPS]:
        raise CampaignError(
            f"Adam steps {unique_steps} do not equal the fixed budget {TARGET_OPTIMIZER_STEPS}"
        )
    if not math.isclose(
        float(loaded["final.pt"]["val_loss"]),
        float(rows[-1]["val_loss"]),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise CampaignError("final.pt validation loss differs from history epoch 50")
    best_epoch = loaded["best.pt"]["epoch"]
    if not isinstance(best_epoch, int) or not (1 <= best_epoch <= EPOCHS):
        raise CampaignError("best.pt epoch is outside the completed budget")
    if not rows[best_epoch - 1]["is_best"]:
        raise CampaignError("best.pt epoch is not marked as a best validation epoch")

    receipts = {name: file_record(path) for name, path in paths.items()}
    return {
        "files": receipts,
        "final_epoch": EPOCHS,
        "optimizer_step_min": min(steps),
        "optimizer_step_max": max(steps),
        "optimizer_parameter_states": len(steps),
        "representation_stats": final_stats,
        "evaluation_checkpoint": receipts["final.pt"],
    }


def assert_not_started_paths_clean(cells: list[dict[str, Any]]) -> None:
    contaminated = []
    for cell in cells:
        if cell["status"] != "not_started":
            continue
        for path in (Path(cell["save_dir"]), Path(cell["log_path"])):
            if path.exists():
                contaminated.append(str(path))
    if contaminated:
        raise CampaignError(
            "Never-started cells already have filesystem artifacts; refusing overwrite: "
            f"{sorted(contaminated)}"
        )


def _latch_drift(
    ledger: dict[str, Any], cell: dict[str, Any], receipt: dict[str, Any]
) -> None:
    latch = ledger["protocol_drift"]
    if not latch["latched"]:
        latch.update(
            {
                "latched": True,
                "first_detected_at_utc": receipt["checked_at_utc"],
                "first_cell_id": cell["cell_id"],
                "first_phase": receipt["phase"],
            }
        )


def _mark_failed(
    cell: dict[str, Any], kind: str, message: str, *, exit_code: int | None = None
) -> None:
    cell.update(
        {
            "status": "failed",
            "terminal_at_utc": utc_now(),
            "exit_code": exit_code,
            "failure": {"kind": kind, "message": message},
        }
    )


def launch_subprocess(
    command: list[str],
    cwd: Path,
    log_handle: Any,
    on_started: Callable[[int], None],
) -> int:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(cwd) if not existing else f"{cwd}{os.pathsep}{existing}"
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    on_started(process.pid)
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def _final_integrity(ledger: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    receipt = check_frozen_receipts(snapshot, "campaign_terminal", full_data_hash=True)
    errors = list(receipt["errors"])
    grouped: dict[tuple[str, str], set[str]] = {}
    for cell in ledger["cells"]:
        if cell["status"] != "completed":
            continue
        stats_hash = cell["artifacts"]["representation_stats"]["combined_sha256"]
        grouped.setdefault((cell["symbol"], cell["representation"]), set()).add(stats_hash)
    for key, hashes in sorted(grouped.items()):
        if len(hashes) != 1:
            errors.append(
                {
                    "group": "representation_stats",
                    "kind": "training_seed_mismatch",
                    "asset_representation": list(key),
                    "observed_hashes": sorted(hashes),
                }
            )
    receipt["representation_stats_groups"] = {
        f"{symbol}/{representation}": sorted(hashes)
        for (symbol, representation), hashes in sorted(grouped.items())
    }
    receipt["errors"] = errors
    receipt["status"] = "passed" if not errors else "failed"
    return receipt


def execute_cells(
    project_root: Path,
    ledger_path: Path,
    ledger: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    launcher: Callable[[list[str], Path, Any, Callable[[int], None]], int] = launch_subprocess,
    validator: Callable[[dict[str, Any]], dict[str, Any]] = validate_cell_artifacts,
) -> dict[str, Any]:
    candidates = [cell for cell in ledger["cells"] if cell["status"] == "not_started"]
    assert_not_started_paths_clean(candidates)
    invocation = {
        "invocation_id": f"{utc_now()}__pid{os.getpid()}",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "selected_cell_ids": [cell["cell_id"] for cell in candidates],
        "status": "running",
    }
    ledger["invocations"].append(invocation)
    ledger["final_integrity"] = None
    save_ledger(ledger_path, ledger)
    drift_latched = bool(ledger["protocol_drift"]["latched"])
    try:
        for index, cell in enumerate(candidates, start=1):
            if cell["status"] != "not_started" or cell["attempt_count"] != 0:
                raise CampaignError(f"Refusing repeated attempt for {cell['cell_id']}")
            preflight = check_frozen_receipts(snapshot, f"pre_launch:{cell['cell_id']}")
            cell["protocol_receipt_checks"].append(preflight)
            if drift_latched or preflight["status"] != "passed":
                if preflight["status"] != "passed":
                    _latch_drift(ledger, cell, preflight)
                drift_latched = True
                _mark_failed(
                    cell,
                    "protocol_drift",
                    "Frozen source/data receipts drifted; cell was not launched.",
                )
                cell["log_sha256"] = _ensure_log(cell, "PROTOCOL_DRIFT: not launched")
                save_ledger(ledger_path, ledger)
                continue
            print(
                f"EVENT-TIME TRAIN {index}/{len(candidates)} {cell['cell_id']} "
                f"target_steps={TARGET_OPTIMIZER_STEPS}",
                flush=True,
            )
            cell.update(
                {
                    "status": "running",
                    "attempt_count": 1,
                    "started_at_utc": utc_now(),
                    "pid": None,
                    "invocation_id": invocation["invocation_id"],
                }
            )
            save_ledger(ledger_path, ledger)
            log_path = Path(cell["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                log_handle = log_path.open("xb")
            except FileExistsError:
                _mark_failed(cell, "log_collision", f"Refusing to overwrite {log_path}")
                cell["log_sha256"] = file_sha256(log_path)
                save_ledger(ledger_path, ledger)
                continue
            exit_code: int | None = None
            interrupted: BaseException | None = None
            try:
                log_handle.write(
                    (
                        f"campaign_id={CAMPAIGN_ID}\ncell_id={cell['cell_id']}\n"
                        f"started_at_utc={cell['started_at_utc']}\n"
                        f"command={cell['command_shell']}\n\n"
                    ).encode("utf-8")
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())

                def on_started(pid: int) -> None:
                    if cell["subprocess_launch_count"] != 0:
                        raise CampaignError(f"Subprocess already launched for {cell['cell_id']}")
                    cell["pid"] = int(pid)
                    cell["subprocess_launch_count"] = 1
                    cell["process_started_at_utc"] = utc_now()
                    save_ledger(ledger_path, ledger)

                exit_code = int(launcher(cell["command"], project_root, log_handle, on_started))
            except BaseException as exc:
                interrupted = exc
                with contextlib.suppress(Exception):
                    log_handle.write(
                        ("\nORCHESTRATOR EXCEPTION\n" + "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        )).encode("utf-8", errors="replace")
                    )
            finally:
                with contextlib.suppress(Exception):
                    log_handle.flush()
                    os.fsync(log_handle.fileno())
                log_handle.close()
            cell["exit_code"] = exit_code
            cell["log_sha256"] = file_sha256(log_path)
            postflight = check_frozen_receipts(snapshot, f"post_process:{cell['cell_id']}")
            cell["protocol_receipt_checks"].append(postflight)
            if postflight["status"] != "passed":
                _latch_drift(ledger, cell, postflight)
                drift_latched = True
                _mark_failed(cell, "protocol_drift", "Source/data drift detected after subprocess")
            elif interrupted is not None:
                _mark_failed(
                    cell,
                    "launcher_exception",
                    f"{type(interrupted).__name__}: {interrupted}",
                    exit_code=exit_code,
                )
            elif exit_code != 0:
                _mark_failed(
                    cell,
                    "nonzero_exit",
                    f"train_v2.py exited with code {exit_code}",
                    exit_code=exit_code,
                )
            else:
                try:
                    cell["artifacts"] = validator(cell)
                except Exception as exc:
                    _mark_failed(
                        cell,
                        "artifact_validation_failed",
                        f"{type(exc).__name__}: {exc}",
                        exit_code=exit_code,
                    )
                else:
                    cell.update(
                        {
                            "status": "completed",
                            "terminal_at_utc": utc_now(),
                            "failure": None,
                        }
                    )
            save_ledger(ledger_path, ledger)
            if interrupted is not None and isinstance(interrupted, (KeyboardInterrupt, SystemExit)):
                raise interrupted
        if all(cell["status"] in TERMINAL_STATUSES for cell in ledger["cells"]):
            ledger["final_integrity"] = _final_integrity(ledger, snapshot)
            if ledger["final_integrity"]["status"] != "passed":
                ledger["protocol_drift"]["latched"] = True
            save_ledger(ledger_path, ledger)
    except BaseException:
        invocation["status"] = "interrupted"
        invocation["finished_at_utc"] = utc_now()
        save_ledger(ledger_path, ledger)
        raise
    invocation["status"] = "finished"
    invocation["finished_at_utc"] = utc_now()
    save_ledger(ledger_path, ledger)
    return ledger_summary(ledger)


def run_campaign(
    project_root: Path,
    snapshot: dict[str, Any],
    *,
    resume_pending: bool,
    launcher: Callable[[list[str], Path, Any, Callable[[int], None]], int] = launch_subprocess,
    validator: Callable[[dict[str, Any]], dict[str, Any]] = validate_cell_artifacts,
) -> dict[str, Any]:
    checkpoint_root = Path(snapshot["checkpoint_root"])
    if checkpoint_root.is_symlink():
        raise CampaignError(f"Checkpoint root may not be a symlink: {checkpoint_root}")
    with campaign_lock(checkpoint_root):
        ledger_path, ledger, created = initialize_or_load_ledger(
            checkpoint_root, snapshot, resume_pending=resume_pending
        )
        if not created:
            seal_stale_running_cells(ledger_path, ledger, snapshot)
        return execute_cells(
            project_root.resolve(),
            ledger_path,
            ledger,
            snapshot,
            launcher=launcher,
            validator=validator,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen 24-cell multi-asset event-time training ledger once"
        )
    )
    parser.add_argument(
        "--resume-pending",
        action="store_true",
        help="Run only never-started cells in an existing identical ledger; never retry a started cell.",
    )
    parser.add_argument(
        "--print-snapshot",
        action="store_true",
        help="Validate and print the frozen snapshot without creating a ledger or launching training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        snapshot = build_campaign_snapshot(REPO_ROOT)
        if not snapshot["environment"]["mps_available"]:
            raise CampaignError(
                "The registered event-time study requires MPS, but torch reports "
                "it unavailable"
            )
        if args.print_snapshot:
            print(json.dumps(snapshot, indent=2, sort_keys=True), flush=True)
            return 0
        summary = run_campaign(
            REPO_ROOT, snapshot, resume_pending=args.resume_pending
        )
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

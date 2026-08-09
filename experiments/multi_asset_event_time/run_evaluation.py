#!/usr/bin/env python3
"""Run the frozen multi-asset event-time evaluation with an auditable ledger.

Preparation is gated on a terminal, successful 24-cell event-time
training ledger.  Every ``final.pt`` is then loaded on CPU and checked for
epoch 50 and exactly 27,400 Adam updates before an evaluation plan can exist.
The immutable plan fixes the four assets, the 3x3 paired seed grid, all child
commands, and source/data/checkpoint hashes.  Execution may resume only stages
that have never started; an attempted stage is never retried.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import signal
import shlex
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # ``torch.load(..., weights_only=False)`` may need project modules while
    # deserializing a checkpoint config.  This is part of the outer gate.
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = 1
TRAINING_CAMPAIGN_ID = "p4_public_itch_event_time_20260807_v1"
EVALUATION_CAMPAIGN_ID = "p4_public_itch_event_time_20260807_v1_evaluation"
ASSETS = (
    ("AAPL", "PUBLIC_ITCH_2019_SAMPLES_AAPL_event"),
    ("AMZN", "PUBLIC_ITCH_2019_SAMPLES_AMZN_event"),
    ("GOOG", "PUBLIC_ITCH_2019_SAMPLES_GOOG_event"),
    ("INTC", "PUBLIC_ITCH_2019_SAMPLES_INTC_event"),
)
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = (0, 1, 2)
REPRESENTATIONS = ("legal_lob", "matched_lob")
METHODS = (
    "legal_lob",
    "matched_posthoc_pava",
    "matched_repeated_pava",
)
IMMUTABLE_STAGE_FIELDS = (
    "stage_id",
    "symbol",
    "kind",
    "command",
    "command_shell",
    "command_sha256",
    "dependencies",
    "declared_outputs",
    "accepted_exit_codes",
)
CHECKPOINT_NAME = "final.pt"
TARGET_EPOCH = 50
TARGET_OPTIMIZER_STEPS = 27_400
N_BATCHES = 5
BATCH_SIZE = 128
N_HISTORIES = N_BATCHES * BATCH_SIZE
HORIZON = 100
DDIM_STEPS = 20
HIST_LEN = 50
PRED_LEN = 10
CHUNK_SIZE = 10

DEFAULT_CHECKPOINT_ROOT = Path(
    "checkpoints_training_seeds/p4_public_itch_event_time_20260807_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/p4_public_itch_event_time_20260807_v1/evaluation"
)
TRAINING_LEDGER_NAME = "p4_training_ledger.json"
PLAN_NAME = "p4_evaluation_plan.json"
LEDGER_NAME = "p4_evaluation_ledger.json"
LOCK_NAME = ".p4_evaluation.lock"
TRAINING_LOCK_NAME = ".p4_event_training.lock"

SOURCE_FILES = (
    "experiments/multi_asset_event_time/protocol.md",
    "experiments/multi_asset_event_time/metric_definition.md",
    "experiments/multi_asset_event_time/run_evaluation.py",
    "scripts/run_all_method_rollout_panel.py",
    "scripts/build_matched_rollout_reference.py",
    "scripts/run_intermediate_consumer_panel.py",
    "scripts/summarize_intermediate_consumer.py",
    "scripts/summarize_intermediate_consumer_outcome.py",
    "experiments/multi_asset_event_time/summarize_results.py",
    "scripts/snowball_test.py",
    "scripts/summarize_financial_sensitivity.py",
    "genlob/models/genlob_v2.py",
    "genlob/models/sampling.py",
    "genlob/models/sde.py",
    "genlob/data/nasdaq.py",
    "genlob/data/normalize.py",
    "genlob/eval/metrics.py",
    "genlob/utils/device.py",
    "genlob/utils/legal_lob.py",
    "genlob/utils/physics.py",
    "tests/test_event_time_evaluation.py",
    "tests/test_build_matched_rollout_reference.py",
    "tests/test_intermediate_consumer.py",
    "tests/test_event_time_summary.py",
)

SCIENCE_ONLY_CELL_CHECKS = frozenset(
    {
        "legal_lob_endpoint_delivered_acceptance",
        "matched_posthoc_pava_endpoint_delivered_acceptance",
        "matched_repeated_pava_endpoint_delivered_acceptance",
        "repeated_anytime_pava_output_acceptance",
    }
)


class EvaluationError(RuntimeError):
    """The frozen event-time evaluation encountered a protocol error."""


Executor = Callable[[list[str], Path, TextIO, Callable[[int], None]], int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


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


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required JSON file missing: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def file_receipt(path: Path, repo_root: Path) -> dict[str, Any]:
    require(path.is_file(), f"required file missing: {path}")
    stat = path.stat()
    return {
        "path": display_path(path, repo_root),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _write_json_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _create_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise EvaluationError(f"refusing to overwrite immutable file: {path}") from error


def _optimizer_steps(checkpoint: dict[str, Any], path: Path) -> list[int]:
    optimizer = checkpoint.get("optimizer_state_dict")
    require(isinstance(optimizer, dict), f"optimizer state missing: {path}")
    states = optimizer.get("state")
    require(isinstance(states, dict) and states, f"Adam parameter states missing: {path}")
    output: list[int] = []
    for state in states.values():
        require(isinstance(state, dict) and "step" in state, f"Adam step missing: {path}")
        raw = state["step"]
        number = float(raw.item() if hasattr(raw, "item") else raw)
        require(math.isfinite(number) and number.is_integer(), f"invalid Adam step: {path}")
        output.append(int(number))
    return output


def validate_final_checkpoint(
    path: Path,
    *,
    repo_root: Path,
    symbol: str,
    dataset: str,
    train_seed: int,
    representation: str,
) -> dict[str, Any]:
    """Independently prove that a checkpoint is the frozen fixed-step final."""
    require(path.name == CHECKPOINT_NAME, f"evaluation checkpoint is not {CHECKPOINT_NAME}: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise EvaluationError(f"cannot load checkpoint {path}: {error}") from error
    require(isinstance(checkpoint, dict), f"checkpoint is not a mapping: {path}")
    require(checkpoint.get("epoch") == TARGET_EPOCH, f"checkpoint epoch is not 50: {path}")
    require(isinstance(checkpoint.get("ema_state_dict"), dict), f"EMA state missing: {path}")
    args = checkpoint.get("args")
    require(isinstance(args, dict), f"checkpoint args missing: {path}")
    expected_args = {
        "dataset": dataset,
        "seed": train_seed,
        "representation": representation,
        "epochs": TARGET_EPOCH,
        "max_train_batches": 548,
        "max_val_batches": 183,
        "batch_size": BATCH_SIZE,
        "hist_len": HIST_LEN,
        "pred_len": PRED_LEN,
        "action_conditioning": "none",
        "price_reference": "history_last_mid",
        "p4_training_entrypoint": (
            "experiments/multi_asset_event_time/train_fixed_steps.py"
        ),
    }
    for key, expected in expected_args.items():
        require(
            args.get(key) == expected,
            f"checkpoint args[{key!r}] mismatch for {symbol}/seed{train_seed}/{representation}",
        )
    steps = _optimizer_steps(checkpoint, path)
    require(
        sorted(set(steps)) == [TARGET_OPTIMIZER_STEPS],
        f"checkpoint Adam steps are not exactly {TARGET_OPTIMIZER_STEPS}: {path}",
    )
    stats = checkpoint.get("representation_stats")
    require(isinstance(stats, dict), f"representation statistics missing: {path}")
    for key in ("mean", "std"):
        tensor = torch.as_tensor(stats.get(key))
        require(
            tuple(tensor.shape) == (1, 2, 20, 1)
            and bool(torch.isfinite(tensor).all()),
            f"invalid representation {key}: {path}",
        )
    require(bool((torch.as_tensor(stats["std"]) > 0).all()), f"non-positive scale: {path}")
    receipt = file_receipt(path, repo_root)
    receipt.update(
        {
            "symbol": symbol,
            "dataset": dataset,
            "train_seed": train_seed,
            "representation": representation,
            "epoch": TARGET_EPOCH,
            "optimizer_step_min": min(steps),
            "optimizer_step_max": max(steps),
            "optimizer_parameter_states": len(steps),
            "args_sha256": object_sha256(args),
        }
    )
    del checkpoint
    gc.collect()
    return receipt


def _canonical_training_cells() -> list[tuple[str, str, int, str]]:
    return [
        (symbol, dataset, seed, representation)
        for symbol, dataset in ASSETS
        for seed in TRAIN_SEEDS
        for representation in REPRESENTATIONS
    ]


def validate_training_gate(
    *, repo_root: Path, data_root: Path, checkpoint_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate the terminal ledger plus every final checkpoint independently."""
    ledger_path = checkpoint_root / TRAINING_LEDGER_NAME
    ledger = read_json(ledger_path)
    require(ledger.get("campaign_id") == TRAINING_CAMPAIGN_ID, "training campaign id mismatch")
    snapshot = ledger.get("snapshot")
    require(isinstance(snapshot, dict), "training snapshot missing")
    require(
        ledger.get("snapshot_sha256") == object_sha256(snapshot),
        "training snapshot hash mismatch",
    )
    protocol = snapshot.get("protocol") or {}
    require(protocol.get("assets") == [item[0] for item in ASSETS], "training asset set mismatch")
    require(protocol.get("datasets") == [item[1] for item in ASSETS], "training dataset set mismatch")
    require(protocol.get("train_seeds") == list(TRAIN_SEEDS), "training seed grid mismatch")
    require(protocol.get("representations") == list(REPRESENTATIONS), "representation grid mismatch")
    require(protocol.get("checkpoint_for_evaluation") == CHECKPOINT_NAME, "training checkpoint contract mismatch")
    require(protocol.get("target_optimizer_steps") == TARGET_OPTIMIZER_STEPS, "optimizer budget mismatch")
    summary = ledger.get("summary") or {}
    require(
        summary.get("status") == "completed"
        and summary.get("expected") == 24
        and summary.get("completed") == 24
        and summary.get("failed") == 0
        and summary.get("terminal") == 24,
        "training ledger is not terminal successful 24/24",
    )
    require(not (ledger.get("protocol_drift") or {}).get("latched"), "training protocol drift latched")
    require((ledger.get("final_integrity") or {}).get("status") == "passed", "training final integrity did not pass")
    cells = ledger.get("cells")
    require(isinstance(cells, list) and len(cells) == 24, "training cell denominator is not 24")
    expected = _canonical_training_cells()
    observed = [
        (cell.get("symbol"), cell.get("dataset"), cell.get("train_seed"), cell.get("representation"))
        for cell in cells
    ]
    require(observed == expected, "training cells are not the canonical four-asset matrix")

    checkpoints: dict[str, dict[str, Any]] = {}
    for cell in cells:
        require(
            cell.get("status") == "completed"
            and cell.get("attempt_count") == 1
            and cell.get("subprocess_launch_count") == 1
            and cell.get("exit_code") == 0,
            f"training cell did not complete exactly once: {cell.get('cell_id')}",
        )
        symbol = str(cell["symbol"])
        dataset = str(cell["dataset"])
        seed = int(cell["train_seed"])
        representation = str(cell["representation"])
        expected_path = (
            checkpoint_root
            / symbol
            / f"train_seed{seed}"
            / representation
            / CHECKPOINT_NAME
        ).resolve()
        require(Path(cell.get("save_dir", "")).resolve() == expected_path.parent, "training save path mismatch")
        receipt = validate_final_checkpoint(
            expected_path,
            repo_root=repo_root,
            symbol=symbol,
            dataset=dataset,
            train_seed=seed,
            representation=representation,
        )
        frozen = (((cell.get("artifacts") or {}).get("files") or {}).get(CHECKPOINT_NAME))
        require(isinstance(frozen, dict), f"training final.pt receipt missing: {cell['cell_id']}")
        for key in ("path", "size_bytes", "mtime_ns", "sha256"):
            expected_value = (
                str(expected_path)
                if key == "path"
                else receipt[key]
            )
            require(frozen.get(key) == expected_value, f"training receipt drift for {cell['cell_id']} {key}")
        checkpoints[f"{symbol}/train_seed{seed}/{representation}"] = receipt

    data_receipts = snapshot.get("data_files")
    require(isinstance(data_receipts, dict) and data_receipts, "training data receipts missing")
    # Full data hashes are checked once at preparation and once at terminal
    # freezing.  Per-stage checks use size/mtime to avoid re-reading ~12 GiB.
    for name, frozen in data_receipts.items():
        path = Path(frozen.get("path", ""))
        symbol = name.split("/", 1)[0]
        dataset = dict(ASSETS).get(symbol)
        require(dataset is not None, f"unexpected training data receipt key: {name}")
        expected_parent = (data_root / "processed" / "snapshot_l2" / dataset).resolve()
        require(path.resolve().parent == expected_parent, f"training data root mismatch: {name}")
        require(path.is_file(), f"training data file missing: {name}")
        stat = path.stat()
        require(
            int(frozen.get("size_bytes", -1)) == stat.st_size
            and int(frozen.get("mtime_ns", -1)) == stat.st_mtime_ns
            and frozen.get("sha256") == sha256_file(path),
            f"training data receipt drift: {name}",
        )
    ledger_receipt = file_receipt(ledger_path, repo_root)
    return ledger_receipt, checkpoints, data_receipts


def _stage(
    stage_id: str,
    *,
    symbol: str | None,
    kind: str,
    command: list[str],
    dependencies: list[str],
    outputs: list[Path],
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "symbol": symbol,
        "kind": kind,
        "command": command,
        "command_shell": shlex.join(command),
        "command_sha256": object_sha256(command),
        "dependencies": dependencies,
        "declared_outputs": [str(path.resolve()) for path in outputs],
        "accepted_exit_codes": [0],
    }


def build_stage_plans(
    *,
    repo_root: Path,
    data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    python: str,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    consumer_stage_ids: list[str] = []
    for symbol, dataset in ASSETS:
        slug = symbol.lower()
        asset_root = output_root / symbol
        asset_checkpoint_root = checkpoint_root / symbol
        reference_root = asset_root / "reference"
        reference_stage_ids: list[str] = []
        reference_eval_roots: list[Path] = []
        for eval_seed in EVAL_SEEDS:
            stage_id = f"{slug}__reference_eval_seed{eval_seed}"
            reference_stage_ids.append(stage_id)
            result_root = reference_root / f"eval_seed{eval_seed}"
            reference_eval_roots.append(result_root)
            command = [
                python,
                "scripts/run_all_method_rollout_panel.py",
                "--dataset", dataset,
                "--data-dir", display_path(data_root, repo_root),
                "--checkpoint-root", display_path(asset_checkpoint_root, repo_root),
                "--checkpoint-name", CHECKPOINT_NAME,
                "--output-root", display_path(result_root, repo_root),
                "--train-seeds", "0", "1", "2",
                "--eval-seed", str(eval_seed),
                "--n-batches", str(N_BATCHES),
                "--batch-size", str(BATCH_SIZE),
                "--horizon", str(HORIZON),
                "--ddim-steps", str(DDIM_STEPS),
                "--device", "mps",
                "--python", python,
                "--methods", "matched_posthoc_pava",
            ]
            stages.append(
                _stage(
                    stage_id,
                    symbol=symbol,
                    kind="reference_eval",
                    command=command,
                    dependencies=[],
                    outputs=[result_root],
                )
            )

        reference_csv = reference_root / "matched_posthoc_reference.csv"
        reference_validation = reference_root / "matched_posthoc_reference_validation.json"
        builder_id = f"{slug}__reference_builder"
        builder_command = [
            python,
            "scripts/build_matched_rollout_reference.py",
            "--results-roots",
            *[display_path(path, repo_root) for path in reference_eval_roots],
            "--dataset", dataset,
            "--data-dir", display_path(data_root, repo_root),
            "--checkpoint-root", display_path(asset_checkpoint_root, repo_root),
            "--checkpoint-name", CHECKPOINT_NAME,
            "--output-csv", display_path(reference_csv, repo_root),
            "--validation-json", display_path(reference_validation, repo_root),
            "--train-seeds", "0", "1", "2",
            "--eval-seeds", "0", "1", "2",
            "--hist-len", str(HIST_LEN),
            "--pred-len", str(PRED_LEN),
            "--chunk-size", str(CHUNK_SIZE),
            "--horizon", str(HORIZON),
            "--ddim-steps", str(DDIM_STEPS),
            "--batch-size", str(BATCH_SIZE),
            "--n-batches", str(N_BATCHES),
            "--device", "mps",
        ]
        stages.append(
            _stage(
                builder_id,
                symbol=symbol,
                kind="reference_builder",
                command=builder_command,
                dependencies=reference_stage_ids,
                outputs=[reference_csv, reference_validation],
            )
        )

        consumer_root = asset_root / "consumer"
        consumer_id = f"{slug}__consumer"
        consumer_stage_ids.append(consumer_id)
        consumer_command = [
            python,
            "scripts/run_intermediate_consumer_panel.py",
            "--dataset", dataset,
            "--data-dir", display_path(data_root, repo_root),
            "--checkpoint-root", display_path(asset_checkpoint_root, repo_root),
            "--checkpoint-name", CHECKPOINT_NAME,
            "--output-root", display_path(consumer_root, repo_root),
            "--train-seeds", "0", "1", "2",
            "--eval-seeds", "0", "1", "2",
            "--hist-len", str(HIST_LEN),
            "--pred-len", str(PRED_LEN),
            "--chunk-size", str(CHUNK_SIZE),
            "--horizon", str(HORIZON),
            "--ddim-steps", str(DDIM_STEPS),
            "--batch-size", str(BATCH_SIZE),
            "--n-batches", str(N_BATCHES),
            "--device", "mps",
            "--threshold-max-train-snapshots", "100000",
            "--reference-rollout-csv", display_path(reference_csv, repo_root),
        ]
        stages.append(
            _stage(
                consumer_id,
                symbol=symbol,
                kind="consumer",
                command=consumer_command,
                dependencies=[builder_id],
                outputs=[consumer_root],
            )
        )

    panel_root = output_root / "panel_c"
    stages.append(
        _stage(
            "cross_asset__panel_c",
            symbol=None,
            kind="panel_c",
            command=[
                python,
                "experiments/multi_asset_event_time/summarize_results.py",
                "--evaluation-root", display_path(output_root, repo_root),
                "--output-dir", display_path(panel_root, repo_root),
            ],
            dependencies=consumer_stage_ids,
            outputs=[panel_root],
        )
    )
    require(
        len(stages) == 21,
        "The event-time evaluation must contain exactly 21 stages",
    )
    require(len({stage["stage_id"] for stage in stages}) == len(stages), "duplicate stage ids")
    return stages


def _assert_new_outputs(stages: list[dict[str, Any]]) -> None:
    declared: set[Path] = set()
    for stage in stages:
        for raw in stage["declared_outputs"]:
            path = Path(raw)
            require(path not in declared, f"output declared by multiple stages: {path}")
            declared.add(path)
            require(not path.exists(), f"declared new output already exists: {path}")


def prepare_evaluation(
    *,
    repo_root: Path,
    data_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    python: str = sys.executable,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    data_root = data_root.resolve()
    checkpoint_root = checkpoint_root.resolve()
    output_root = output_root.resolve()
    require(not output_root.exists(), f"refusing to overwrite evaluation root: {output_root}")
    training_ledger, checkpoints, data_receipts = validate_training_gate(
        repo_root=repo_root, data_root=data_root, checkpoint_root=checkpoint_root
    )
    sources = {
        relative: file_receipt(repo_root / relative, repo_root)
        for relative in SOURCE_FILES
    }
    stages = build_stage_plans(
        repo_root=repo_root,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        python=python,
    )
    _assert_new_outputs(stages)
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": EVALUATION_CAMPAIGN_ID,
        "prepared_at_utc": utc_now(),
        "research_boundary": (
            "Separately trained event-time sampled Public-ITCH Level-2 snapshots; "
            "not message-level generation, replay, queue/fill, causal control, "
            "unseen-asset transfer, or cross-venue generalization."
        ),
        "protocol": {
            "assets": [symbol for symbol, _ in ASSETS],
            "datasets": [dataset for _, dataset in ASSETS],
            "train_seeds": list(TRAIN_SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "methods": list(METHODS),
            "checkpoint_name": CHECKPOINT_NAME,
            "weights": "ema",
            "hist_len": HIST_LEN,
            "pred_len": PRED_LEN,
            "chunk_size": CHUNK_SIZE,
            "horizon": HORIZON,
            "ddim_steps": DDIM_STEPS,
            "batch_size": BATCH_SIZE,
            "n_batches": N_BATCHES,
            "n_histories_per_cell": N_HISTORIES,
            "action_mode": "hold",
            "pairing": (
                "same train/eval cell histories, real futures, post-load RNG state, "
                "and initial DDIM noise across all three methods"
            ),
            "reference_reproduction": {
                "non_endpoint_relative_tolerance": 1e-5,
                "non_endpoint_absolute_tolerance": 5e-7,
                "endpoint_invalidity_absolute_tolerance": 5e-5,
                "endpoint_frames_per_cell": N_HISTORIES * HORIZON,
                "endpoint_tolerance_equivalent_frames_per_cell": 3.2,
                "rule": (
                    "Only endpoint pre-invalidity receives the declared 5e-5 "
                    "diagnostic parity tolerance; every other reference error "
                    "remains disallowed."
                ),
            },
            "panel_reporting": {
                "metrics": [
                    {"key": "pre_projection_invalidity", "direction": "down"},
                    {
                        "key": "intermediate_consumer_rejection",
                        "direction": "down",
                    },
                    {"key": "endpoint_invalidity", "direction": "down"},
                    {"key": "projection_calls_per_path", "direction": "down"},
                    {
                        "key": "correction_per_scheduled_projection_call",
                        "direction": "down",
                        "zero_call_value": "N/A",
                    },
                    {"key": "cumulative_repair", "direction": "down"},
                    {
                        "key": "market_order_slippage_w1_ticks",
                        "direction": "down",
                    },
                ],
                "formatting": {
                    "direction_marker": "down_arrow",
                    "best": "bold",
                    "second_best": "underline",
                    "ties": "dense_rank",
                },
            },
            "retry_policy": "never retry any attempted stage; explicit resume may launch never-started stages only",
        },
        "denominators": {
            "training_checkpoints": 24,
            "assets": 4,
            "train_eval_cells": 36,
            "method_rows": 108,
            "histories_per_cell": N_HISTORIES,
            "reference_rows": 36,
            "execution_stages": 21,
        },
        "paths": {
            "repo_root": str(repo_root),
            "data_root": str(data_root),
            "checkpoint_root": str(checkpoint_root),
            "output_root": str(output_root),
        },
        "provenance": {
            "training_ledger": training_ledger,
            "sources": sources,
            "source_bundle_sha256": object_sha256(sources),
            "data": data_receipts,
            "data_bundle_sha256": object_sha256(data_receipts),
            "checkpoints": checkpoints,
            "checkpoint_bundle_sha256": object_sha256(checkpoints),
            "command_bundle_sha256": object_sha256(
                [
                    {"stage_id": stage["stage_id"], "sha256": stage["command_sha256"]}
                    for stage in stages
                ]
            ),
            "environment": {
                "python_command": python,
                "python_runtime": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "evaluation_device": "mps",
                "checkpoint_validation_device": "cpu",
            },
        },
        "stages": stages,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.prepare-", dir=output_root.parent)
    )
    try:
        orchestrator = temporary / "_orchestrator"
        orchestrator.mkdir()
        _create_json(orchestrator / PLAN_NAME, plan)
        plan_hash = sha256_file(orchestrator / PLAN_NAME)
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": EVALUATION_CAMPAIGN_ID,
            "status": "prepared",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "revision": 0,
            "previous_revision_sha256": None,
            "plan_path": str((output_root / "_orchestrator" / PLAN_NAME).resolve()),
            "plan_sha256": plan_hash,
            "invocations": [],
            "terminal_integrity": None,
            "stages": [
                {
                    **stage,
                    "status": "never_started",
                    "attempt_count": 0,
                    "subprocess_launch_count": 0,
                    "started_at_utc": None,
                    "terminal_at_utc": None,
                    "pid": None,
                    "process_group_id": None,
                    "exit_code": None,
                    "log_path": str(
                        (
                            output_root
                            / "_orchestrator"
                            / "logs"
                            / f"{stage['stage_id']}.log"
                        ).resolve()
                    ),
                    "log_sha256": None,
                    "artifacts": None,
                    "failure": None,
                }
                for stage in stages
            ],
        }
        ledger["summary"] = ledger_summary(ledger)
        _create_json(orchestrator / LEDGER_NAME, ledger)
        os.replace(temporary, output_root)
    finally:
        if temporary.exists():
            with contextlib.suppress(OSError):
                temporary.rmdir()
    return plan


def _verify_receipt(
    receipt: dict[str, Any], repo_root: Path, *, full_hash: bool
) -> None:
    path = resolve_path(receipt.get("path", ""), repo_root)
    require(path.is_file(), f"frozen input missing: {path}")
    stat = path.stat()
    require(
        int(receipt.get("size_bytes", -1)) == stat.st_size
        and int(receipt.get("mtime_ns", -1)) == stat.st_mtime_ns,
        f"frozen input size/mtime drift: {path}",
    )
    if full_hash:
        require(receipt.get("sha256") == sha256_file(path), f"frozen input hash drift: {path}")


def verify_frozen_plan(
    plan: dict[str, Any],
    repo_root: Path,
    *,
    full_data_hash: bool = False,
    full_checkpoint_hash: bool = False,
) -> None:
    require(plan.get("campaign_id") == EVALUATION_CAMPAIGN_ID, "evaluation campaign id mismatch")
    protocol = plan.get("protocol") or {}
    require(protocol.get("assets") == [item[0] for item in ASSETS], "frozen asset drift")
    require(protocol.get("datasets") == [item[1] for item in ASSETS], "frozen dataset drift")
    require(protocol.get("checkpoint_name") == CHECKPOINT_NAME, "checkpoint name drift")
    require(protocol.get("train_seeds") == list(TRAIN_SEEDS), "training seed drift")
    require(protocol.get("eval_seeds") == list(EVAL_SEEDS), "evaluation seed drift")
    require(protocol.get("methods") == list(METHODS), "method grid drift")
    require(
        (protocol.get("reference_reproduction") or {}).get(
            "endpoint_invalidity_absolute_tolerance"
        )
        == 5e-5,
        "endpoint-only reference tolerance drift",
    )
    provenance = plan.get("provenance") or {}
    sources = provenance.get("sources") or {}
    data = provenance.get("data") or {}
    checkpoints = provenance.get("checkpoints") or {}
    require(object_sha256(sources) == provenance.get("source_bundle_sha256"), "source bundle hash mismatch")
    require(object_sha256(data) == provenance.get("data_bundle_sha256"), "data bundle hash mismatch")
    require(object_sha256(checkpoints) == provenance.get("checkpoint_bundle_sha256"), "checkpoint bundle hash mismatch")
    _verify_receipt(provenance.get("training_ledger") or {}, repo_root, full_hash=True)
    for receipt in sources.values():
        _verify_receipt(receipt, repo_root, full_hash=True)
    for receipt in checkpoints.values():
        _verify_receipt(
            receipt, repo_root, full_hash=full_checkpoint_hash
        )
        require(
            receipt.get("epoch") == TARGET_EPOCH
            and receipt.get("optimizer_step_min") == TARGET_OPTIMIZER_STEPS
            and receipt.get("optimizer_step_max") == TARGET_OPTIMIZER_STEPS,
            f"frozen checkpoint no longer proves fixed-step final: {receipt.get('path')}",
        )
    for receipt in data.values():
        _verify_receipt(receipt, repo_root, full_hash=full_data_hash)
    stages = plan.get("stages") or []
    require(len(stages) == 21, "stage denominator drift")
    for stage in stages:
        require(object_sha256(stage["command"]) == stage.get("command_sha256"), f"command hash drift: {stage.get('stage_id')}")
    expected_bundle = object_sha256(
        [{"stage_id": stage["stage_id"], "sha256": stage["command_sha256"]} for stage in stages]
    )
    require(expected_bundle == provenance.get("command_bundle_sha256"), "command bundle hash mismatch")


def verify_ledger_plan_binding(
    plan: dict[str, Any], ledger: dict[str, Any]
) -> None:
    """Bind mutable execution state to every immutable hashed plan field."""
    plan_stages = plan.get("stages") or []
    ledger_stages = ledger.get("stages") or []
    require(len(plan_stages) == 21, "plan stage denominator drift")
    require(len(ledger_stages) == len(plan_stages), "plan/ledger stage denominator mismatch")
    output_root = Path(str((plan.get("paths") or {}).get("output_root", ""))).resolve()
    for index, (plan_stage, ledger_stage) in enumerate(
        zip(plan_stages, ledger_stages)
    ):
        stage_id = plan_stage.get("stage_id")
        for field in IMMUTABLE_STAGE_FIELDS:
            require(
                ledger_stage.get(field) == plan_stage.get(field),
                f"plan/ledger immutable field drift at stage {index} "
                f"({stage_id}): {field}",
            )
        expected_log = (
            output_root / "_orchestrator" / "logs" / f"{stage_id}.log"
        ).resolve()
        require(
            Path(str(ledger_stage.get("log_path", ""))).resolve() == expected_log,
            f"plan/ledger log path drift at stage {index} ({stage_id})",
        )


def _tree_receipts(path: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    require(path.exists(), f"declared stage output missing: {path}")
    if path.is_file():
        return {display_path(path, repo_root): file_receipt(path, repo_root)}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    require(files, f"declared output directory is empty: {path}")
    return {display_path(item, repo_root): file_receipt(item, repo_root) for item in files}


def _declared_output_receipts(
    stage: dict[str, Any], repo_root: Path
) -> dict[str, dict[str, Any]]:
    """Hash the exact current file set for every output declared by a stage."""
    receipts: dict[str, dict[str, Any]] = {}
    for raw in stage.get("declared_outputs") or []:
        for key, receipt in _tree_receipts(Path(raw), repo_root).items():
            require(key not in receipts, f"overlapping declared stage output: {key}")
            receipts[key] = receipt
    require(receipts, f"completed stage has no declared artifact files: {stage.get('stage_id')}")
    return receipts


def verify_completed_stage_receipts(
    stage: dict[str, Any], repo_root: Path
) -> None:
    """Fail closed if any completed-stage output or log changed after validation."""
    stage_id = stage.get("stage_id")
    require(stage.get("status") == "completed", f"stage is not completed: {stage_id}")
    artifacts = stage.get("artifacts") or {}
    expected = artifacts.get("files")
    require(
        isinstance(expected, dict) and expected,
        f"completed stage artifact receipts missing: {stage_id}",
    )
    current = _declared_output_receipts(stage, repo_root)
    require(
        set(current) == set(expected),
        f"completed stage artifact file set drift: {stage_id}",
    )
    for key in sorted(expected):
        require(
            expected[key] == current[key],
            f"completed stage artifact receipt drift: {stage_id}: {key}",
        )
    log_path = Path(str(stage.get("log_path", "")))
    expected_log_hash = stage.get("log_sha256")
    require(
        log_path.is_file()
        and isinstance(expected_log_hash, str)
        and expected_log_hash == sha256_file(log_path),
        f"completed stage log receipt drift: {stage_id}",
    )


def _validate_reference_eval(
    stage: dict[str, Any], plan: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    root = Path(stage["declared_outputs"][0])
    manifest = read_json(root / "run_manifest.json")
    symbol = str(stage["symbol"])
    dataset = dict(ASSETS)[symbol]
    eval_seed = int(stage["stage_id"].removeprefix(f"{symbol.lower()}__reference_eval_seed"))
    require(manifest.get("dataset") == dataset, "reference dataset mismatch")
    require(manifest.get("checkpoint_name") == CHECKPOINT_NAME, "reference did not use final.pt")
    require(manifest.get("train_seeds") == list(TRAIN_SEEDS), "reference train seed mismatch")
    require(manifest.get("eval_seed") == eval_seed, "reference eval seed mismatch")
    require(manifest.get("methods") == ["matched_posthoc_pava"], "reference method mismatch")
    runs = manifest.get("runs") or []
    require(len(runs) == 3, "reference run denominator mismatch")
    for train_seed, run in zip(TRAIN_SEEDS, runs):
        require(
            run.get("train_seed") == train_seed
            and run.get("method") == "matched_posthoc_pava"
            and run.get("status") == "completed",
            "reference run is incomplete or reordered",
        )
        checkpoint = resolve_path(run.get("checkpoint", ""), repo_root)
        require(checkpoint.name == CHECKPOINT_NAME, "reference run checkpoint is not final.pt")
        frozen_checkpoint = (
            ((plan.get("provenance") or {}).get("checkpoints") or {}).get(
                f"{symbol}/train_seed{train_seed}/matched_lob"
            )
            or {}
        )
        require(
            checkpoint == resolve_path(frozen_checkpoint.get("path", ""), repo_root)
            and sha256_file(checkpoint) == frozen_checkpoint.get("sha256"),
            "reference run is not bound to the frozen Matched final.pt",
        )
        require((resolve_path(run.get("output_dir", ""), repo_root) / "snowball_stats.json").is_file(), "reference statistics missing")
    expected_files = {root / "run_manifest.json"}
    for train_seed in TRAIN_SEEDS:
        method_root = root / f"train_seed{train_seed}" / "matched_posthoc_pava"
        expected_files.update(
            {method_root / "command.txt", method_root / "snowball_stats.json"}
        )
    observed_files = {path for path in root.rglob("*") if path.is_file()}
    require(observed_files == expected_files, "reference output has missing/orphan files")
    return {
        "reference_runs": 3,
        "eval_seed": eval_seed,
        "files": _tree_receipts(root, repo_root),
    }


def _validate_reference_builder(stage: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    csv_path, validation_path = map(Path, stage["declared_outputs"])
    validation = read_json(validation_path)
    require(validation.get("status") == "validated", "reference builder did not validate")
    expected = validation.get("expected_protocol") or {}
    require(expected.get("checkpoint_name") == CHECKPOINT_NAME, "builder did not bind final.pt")
    completeness = validation.get("completeness") or {}
    require(
        completeness.get("expected_cells") == 9
        and completeness.get("observed_cells") == 9
        and completeness.get("all_cells_complete") is True,
        "reference builder denominator mismatch",
    )
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 9, "reference CSV must contain exactly nine rows")
    require(all(Path(row["checkpoint_path"]).name == CHECKPOINT_NAME for row in rows), "reference CSV contains non-final checkpoint")
    return {
        "reference_rows": 9,
        "files": {
            display_path(path, repo_root): file_receipt(path, repo_root)
            for path in (csv_path, validation_path)
        },
    }


def _validate_consumer(
    stage: dict[str, Any], plan: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    root = Path(stage["declared_outputs"][0])
    manifest = read_json(root / "run_manifest.json")
    config = manifest.get("config") or {}
    symbol = str(stage["symbol"])
    require(config.get("dataset") == dict(ASSETS)[symbol], "consumer dataset mismatch")
    require(config.get("checkpoint_name") == CHECKPOINT_NAME, "consumer did not use final.pt")
    require(config.get("train_seeds") == list(TRAIN_SEEDS), "consumer train seed mismatch")
    require(config.get("eval_seeds") == list(EVAL_SEEDS), "consumer eval seed mismatch")
    require(config.get("n_histories") == N_HISTORIES, "consumer history denominator mismatch")
    require(
        manifest.get("status") == "completed"
        and manifest.get("completed_cells") == 9
        and manifest.get("failed_cells") == 0,
        "consumer manifest did not complete 9/9",
    )
    entries = manifest.get("cells") or []
    expected_cells = {(train, evaluate) for train in TRAIN_SEEDS for evaluate in EVAL_SEEDS}
    observed_cells: set[tuple[int, int]] = set()
    method_rows = 0
    expected_starts_hash = (
        ((manifest.get("provenance") or {}).get("data") or {}).get(
            "start_indices_sha256"
        )
    )
    require(isinstance(expected_starts_hash, str), "manifest history-start hash missing")
    real_future_hashes: set[str] = set()
    noise_by_eval: dict[int, set[tuple[str, ...]]] = {
        eval_seed: set() for eval_seed in EVAL_SEEDS
    }
    post_load_rng_by_eval: dict[int, set[str]] = {
        eval_seed: set() for eval_seed in EVAL_SEEDS
    }
    for entry in entries:
        key = (int(entry["train_seed"]), int(entry["eval_seed"]))
        observed_cells.add(key)
        require(entry.get("status") == "completed", f"consumer cell incomplete: {key}")
        path = resolve_path(entry.get("path", ""), repo_root)
        require(entry.get("sha256") == sha256_file(path), f"consumer cell hash mismatch: {key}")
        payload = read_json(path)
        require(payload.get("status") == "completed", f"consumer payload incomplete: {key}")
        methods = payload.get("methods") or {}
        require(set(methods) == set(METHODS), f"consumer method set mismatch: {key}")
        method_rows += len(methods)
        require(
            payload.get("history_starts_sha256") == expected_starts_hash,
            f"history starts are not bound to the manifest: {key}",
        )
        real_hash = payload.get("real_futures_sha256")
        require(isinstance(real_hash, str) and len(real_hash) == 64, f"real-future hash missing: {key}")
        real_future_hashes.add(real_hash)
        checkpoints = payload.get("checkpoints") or {}
        for representation in REPRESENTATIONS:
            observed_checkpoint = checkpoints.get(representation) or {}
            frozen_checkpoint = (
                ((plan.get("provenance") or {}).get("checkpoints") or {}).get(
                    f"{symbol}/train_seed{key[0]}/{representation}"
                )
                or {}
            )
            require(
                resolve_path(observed_checkpoint.get("path", ""), repo_root)
                == resolve_path(frozen_checkpoint.get("path", ""), repo_root)
                and observed_checkpoint.get("sha256")
                == frozen_checkpoint.get("sha256"),
                f"consumer checkpoint is not frozen final.pt: {key}/{representation}",
            )
        post_load = payload.get("post_model_load_rng_sha256") or {}
        require(set(post_load) == set(METHODS) and len(set(post_load.values())) == 1, f"post-load RNG is not paired: {key}")
        require(
            all(isinstance(value, str) and len(value) == 64 for value in post_load.values()),
            f"post-load RNG hash is malformed: {key}",
        )
        post_load_rng_by_eval[key[1]].add(next(iter(post_load.values())))
        noise = payload.get("paired_initial_noise_sha256")
        require(
            isinstance(noise, list)
            and len(noise) == N_BATCHES * math.ceil(HORIZON / CHUNK_SIZE)
            and all(isinstance(value, str) and len(value) == 64 for value in noise),
            f"paired initial-noise hashes missing or malformed: {key}",
        )
        noise_by_eval[key[1]].add(tuple(noise))
        implementation_checks = (
            (payload.get("validation") or {}).get("implementation_checks") or {}
        )
        require(isinstance(implementation_checks, dict), f"implementation checks missing: {key}")
        integrity_failures = sorted(
            name
            for name, passed in implementation_checks.items()
            if passed is not True and name not in SCIENCE_ONLY_CELL_CHECKS
        )
        require(not integrity_failures, f"consumer implementation gate failed: {key}: {integrity_failures}")
        for method in methods.values():
            require(method.get("n_histories") == N_HISTORIES, f"method history denominator mismatch: {key}")
    require(observed_cells == expected_cells and len(entries) == 9, "consumer cell set mismatch")
    require(method_rows == 27, "consumer method-row denominator mismatch")
    require(len(real_future_hashes) == 1, "real futures differ across train/evaluation seed cells")
    require(
        all(len(values) == 1 for values in noise_by_eval.values()),
        "initial DDIM noise differs across training checkpoints for an eval seed",
    )
    require(
        all(len(values) == 1 for values in post_load_rng_by_eval.values()),
        "post-load RNG differs across training checkpoints for an eval seed",
    )
    expected_files = {root / "run_manifest.json"} | {
        root / f"train_seed{train_seed}" / f"eval_seed{eval_seed}" / "consumer_stats.json"
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
    }
    observed_files = {path for path in root.rglob("*") if path.is_file()}
    require(observed_files == expected_files, "consumer output has missing/orphan files")
    return {
        "train_eval_cells": 9,
        "method_rows": 27,
        "paired_history_contract": True,
        "history_starts_sha256": expected_starts_hash,
        "real_futures_sha256": next(iter(real_future_hashes)),
        "paired_rng_and_initial_noise": True,
        "cross_training_seed_pairing_by_eval_seed": {
            str(eval_seed): {
                "initial_noise_sha256": list(next(iter(noise_by_eval[eval_seed]))),
                "post_load_rng_sha256": next(iter(post_load_rng_by_eval[eval_seed])),
                "training_checkpoint_count": len(TRAIN_SEEDS),
            }
            for eval_seed in EVAL_SEEDS
        },
        "files": _tree_receipts(root, repo_root),
    }


def _validate_strict_summary(stage: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    root = Path(stage["declared_outputs"][0])
    validation = read_json(root / "intermediate_consumer_validation.json")
    require(validation.get("implementation_verified") is True, "strict summary integrity failed")
    require(
        validation.get("expected_cells") == 9
        and validation.get("completed_cells") == 9,
        "strict summary cell denominator mismatch",
    )
    return {"files": _tree_receipts(root, repo_root)}


def _validate_outcome_summary(stage: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    outputs = [Path(raw) for raw in stage["declared_outputs"]]
    require(all(path.is_file() for path in outputs), "outcome summary output set is incomplete")
    root = outputs[0].parent
    validation = read_json(root / "intermediate_consumer_outcome_validation.json")
    require(validation.get("experiment_integrity_verified") is True, "outcome summary integrity failed")
    return {
        "strict_go": bool(validation.get("pre_registered_strict_go")),
        "files": {
            display_path(path, repo_root): file_receipt(path, repo_root)
            for path in outputs
        },
    }


def validate_stage_outputs(
    stage: dict[str, Any], plan: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    kind = stage["kind"]
    if kind == "reference_eval":
        return _validate_reference_eval(stage, plan, repo_root)
    if kind == "reference_builder":
        return _validate_reference_builder(stage, repo_root)
    if kind == "consumer":
        return _validate_consumer(stage, plan, repo_root)
    if kind == "panel_c":
        root = Path(stage["declared_outputs"][0])
        expected_names = {
            "p4_event_robustness_summary.json",
            "p4_event_robustness_summary.csv",
            "p4_event_robustness_panel_c.md",
            "p4_event_robustness_panel_c.tex",
            "p4_event_robustness_validation.json",
            "p4_event_robustness_receipt.json",
        }
        observed_names = {
            path.name for path in root.iterdir() if path.is_file()
        } if root.is_dir() else set()
        require(observed_names == expected_names, "Panel C exact output-file set mismatch")
        validation = read_json(root / "p4_event_robustness_validation.json")
        require(
            validation.get("implementation_integrity_verified") is True,
            "Panel C implementation integrity failed",
        )
        require(
            validation.get("campaign_result_class")
            in {"verified_go", "verified_scientific_no_go"},
            "Panel C did not produce a verified scientific classification",
        )
        summary = read_json(root / "p4_event_robustness_summary.json")
        require(len(summary.get("panel_rows") or []) == 12, "Panel C row denominator is not 12")
        expected_grid = {
            (symbol, method)
            for symbol, _ in ASSETS
            for method in METHODS
        }
        observed_grid = {
            (row.get("asset"), row.get("method_id"))
            for row in summary.get("panel_rows") or []
        }
        require(observed_grid == expected_grid, "Panel C fixed asset/method grid mismatch")
        with (root / "p4_event_robustness_summary.csv").open(newline="") as handle:
            require(len(list(csv.DictReader(handle))) == 12, "Panel C CSV row denominator is not 12")
        receipt = read_json(root / "p4_event_robustness_receipt.json")
        output_hashes = receipt.get("output_hashes_before_receipt") or {}
        require(
            set(output_hashes) == expected_names - {"p4_event_robustness_receipt.json"},
            "Panel C receipt output set mismatch",
        )
        for name, digest in output_hashes.items():
            require(
                digest == sha256_file(root / name),
                f"Panel C receipt hash mismatch: {name}",
            )
        clarification = (
            (plan.get("provenance") or {}).get("sources") or {}
        ).get("experiments/multi_asset_event_time/metric_definition.md") or {}
        require(
            clarification.get("sha256")
            == "52f378090c996abf49fce85154724d28bfbb3dc2fe72c65f8868d32948b64b4d",
            "metric clarification hash is not bound by the evaluation plan",
        )
        return {
            "result_class": validation["campaign_result_class"],
            "files": _tree_receipts(root, repo_root),
        }
    raise EvaluationError(f"unknown evaluation stage kind: {kind}")


def ledger_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    stages = ledger.get("stages") or []
    require(len(stages) == 21, "execution ledger stage denominator is not 21")
    counts = {status: 0 for status in ("never_started", "running", "completed", "failed")}
    for stage in stages:
        require(stage.get("status") in counts, f"unknown stage status: {stage.get('status')}")
        counts[stage["status"]] += 1
        require(stage.get("attempt_count") in (0, 1), "invalid stage attempt count")
        require(stage.get("subprocess_launch_count") in (0, 1), "invalid launch count")
    terminal = counts["completed"] + counts["failed"]
    eval_cells = sum(
        int((stage.get("artifacts") or {}).get("train_eval_cells", 0))
        for stage in stages
        if stage.get("status") == "completed"
    )
    method_rows = sum(
        int((stage.get("artifacts") or {}).get("method_rows", 0))
        for stage in stages
        if stage.get("status") == "completed"
    )
    reference_rows = sum(
        int((stage.get("artifacts") or {}).get("reference_rows", 0))
        for stage in stages
        if stage.get("status") == "completed"
    )
    terminal_integrity = ledger.get("terminal_integrity")
    terminal_integrity_status = (
        terminal_integrity.get("status")
        if isinstance(terminal_integrity, dict)
        else None
    )
    if counts["completed"] == 21:
        # A fully executed stage grid is not a successful campaign until the
        # independent terminal re-hash has passed.  This also prevents a
        # transient success classification between the last stage receipt and
        # the terminal-integrity ledger revision.
        status = (
            "completed"
            if terminal_integrity_status == "passed"
            else "in_progress"
            if terminal_integrity is None
            else "terminal_with_declared_failures"
        )
    elif terminal == 21:
        status = "terminal_with_declared_failures"
    elif counts["running"] or terminal:
        status = "in_progress"
    else:
        status = "prepared"
    panel_stage = next(
        (stage for stage in stages if stage.get("kind") == "panel_c"), None
    )
    verified_result_class = (
        ((panel_stage or {}).get("artifacts") or {}).get("result_class")
        if (panel_stage or {}).get("status") == "completed"
        else None
    )
    campaign_result_class = (
        verified_result_class
        if status == "completed"
        else "terminal_with_declared_failures"
        if status == "terminal_with_declared_failures"
        else None
    )
    return {
        "status": status,
        "expected_stages": 21,
        **counts,
        "terminal_stages": terminal,
        "terminal_integrity_status": terminal_integrity_status,
        "attempted_once": sum(stage.get("attempt_count") == 1 for stage in stages),
        "launched_once": sum(stage.get("subprocess_launch_count") == 1 for stage in stages),
        "campaign_result_class": campaign_result_class,
        "exact_denominators": {
            "training_checkpoints": 24,
            "train_eval_cells_expected": 36,
            "train_eval_cells_observed": eval_cells,
            "method_rows_expected": 108,
            "method_rows_observed": method_rows,
            "reference_rows_expected": 36,
            "reference_rows_observed": reference_rows,
        },
    }


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    previous = sha256_file(path) if path.exists() else None
    ledger["previous_revision_sha256"] = previous
    ledger["revision"] = int(ledger.get("revision", 0)) + 1
    ledger["updated_at_utc"] = utc_now()
    ledger["summary"] = ledger_summary(ledger)
    ledger["status"] = ledger["summary"]["status"]
    _write_json_replace(path, ledger)


def _default_executor(
    command: list[str], cwd: Path, log: TextIO, on_started: Callable[[int], None]
) -> int:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(cwd) if not existing else f"{cwd}{os.pathsep}{existing}"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    )
    on_started(process.pid)
    try:
        return int(process.wait())
    except BaseException:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def _terminal_integrity(
    plan: dict[str, Any], ledger: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        verify_frozen_plan(
            plan,
            repo_root,
            full_data_hash=True,
            full_checkpoint_hash=True,
        )
    except Exception as error:
        errors.append(f"input_receipts: {type(error).__name__}: {error}")
    try:
        verify_ledger_plan_binding(plan, ledger)
    except Exception as error:
        errors.append(f"plan_ledger_binding: {type(error).__name__}: {error}")
    for stage in ledger.get("stages") or []:
        if stage.get("status") != "completed":
            continue
        try:
            verify_completed_stage_receipts(stage, repo_root)
        except Exception as error:
            errors.append(
                f"stage_receipts[{stage.get('stage_id')}]: "
                f"{type(error).__name__}: {error}"
            )
    summary = ledger_summary(ledger)
    passed_terminal_summary = ledger_summary(
        {**ledger, "terminal_integrity": {"status": "passed"}}
    )
    exact = summary["exact_denominators"]
    if not (
        summary["completed"] == 21
        and exact["train_eval_cells_observed"] == 36
        and exact["method_rows_observed"] == 108
        and exact["reference_rows_observed"] == 36
    ):
        errors.append(f"terminal denominator mismatch: {exact}")
    plan_by_id = {
        stage["stage_id"]: stage for stage in plan.get("stages") or []
    }
    if summary["completed"] == 21:
        execution_receipt_errors = [
            stage.get("stage_id")
            for stage in ledger.get("stages") or []
            if not (
                stage.get("attempt_count") == 1
                and stage.get("subprocess_launch_count") == 1
                and stage.get("exit_code")
                in (plan_by_id.get(stage.get("stage_id"), {}).get("accepted_exit_codes") or [])
            )
        ]
        if execution_receipt_errors:
            errors.append(
                "terminal one-attempt/one-launch/accepted-exit gate failed: "
                f"{execution_receipt_errors}"
            )
    return {
        "checked_at_utc": utc_now(),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "denominators": exact,
        "campaign_result_class": (
            passed_terminal_summary.get("campaign_result_class")
            if not errors
            else "terminal_with_declared_failures"
        ),
    }


@contextlib.contextmanager
def _execution_locks(output_root: Path, checkpoint_root: Path):
    orchestrator = output_root / "_orchestrator"
    lock_paths = [orchestrator / LOCK_NAME, checkpoint_root / TRAINING_LOCK_NAME]
    handles = []
    try:
        for path in lock_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise EvaluationError(f"campaign lock is held: {path}") from error
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _mark_interrupted_running(ledger: dict[str, Any]) -> bool:
    changed = False
    for stage in ledger["stages"]:
        if stage["status"] == "running":
            stage.update(
                {
                    "status": "failed",
                    "terminal_at_utc": utc_now(),
                    "failure": {
                        "kind": "interrupted_process",
                        "message": "prior evaluator ended without a terminal stage receipt; retry forbidden",
                    },
                }
            )
            changed = True
    return changed


def _process_group_exists(process_group_id: int | None) -> bool:
    if not isinstance(process_group_id, int) or process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def execute_prepared(
    output_root: Path,
    *,
    resume_pending: bool,
    executor: Executor = _default_executor,
    stage_validator: Callable[[dict[str, Any], dict[str, Any], Path], dict[str, Any]] = validate_stage_outputs,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    orchestrator = output_root / "_orchestrator"
    plan_path = orchestrator / PLAN_NAME
    ledger_path = orchestrator / LEDGER_NAME
    require(
        plan_path.is_file() and ledger_path.is_file(),
        "The prepared event-time evaluation plan is missing",
    )
    plan = read_json(plan_path)
    repo_root = Path(plan["paths"]["repo_root"]).resolve()
    checkpoint_root = Path(plan["paths"]["checkpoint_root"]).resolve()
    with _execution_locks(output_root, checkpoint_root):
        ledger = read_json(ledger_path)
        require(ledger.get("plan_sha256") == sha256_file(plan_path), "immutable plan hash mismatch")
        require(ledger.get("campaign_id") == EVALUATION_CAMPAIGN_ID, "ledger campaign id mismatch")
        verify_ledger_plan_binding(plan, ledger)
        live_orphan_groups = [
            stage.get("process_group_id")
            for stage in ledger["stages"]
            if stage.get("status") == "running"
            and _process_group_exists(stage.get("process_group_id"))
        ]
        require(
            not live_orphan_groups,
            "refusing resume while prior stage process groups remain alive: "
            f"{live_orphan_groups}",
        )
        interrupted = _mark_interrupted_running(ledger)
        attempted = any(stage.get("attempt_count") == 1 for stage in ledger["stages"])
        if interrupted:
            save_ledger(ledger_path, ledger)
        if attempted and not resume_pending:
            raise EvaluationError(
                "evaluation already contains attempted stages; use --resume-pending to launch only never-started stages"
            )
        # Re-hash every final.pt once at invocation start. Per-stage guards use
        # size/mtime and the terminal freeze re-hashes every checkpoint again.
        verify_frozen_plan(
            plan,
            repo_root,
            full_data_hash=False,
            full_checkpoint_hash=True,
        )
        by_id = {stage["stage_id"]: stage for stage in ledger["stages"]}
        invocation = {
            "invocation_id": f"{utc_now()}__pid{os.getpid()}",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "resume_pending": bool(resume_pending),
            "selected_stage_ids": [
                stage["stage_id"] for stage in ledger["stages"] if stage["status"] == "never_started"
            ],
            "status": "running",
        }
        ledger["invocations"].append(invocation)
        save_ledger(ledger_path, ledger)
        logs = orchestrator / "logs"
        logs.mkdir(exist_ok=True)

        try:
            for index, stage in enumerate(ledger["stages"]):
                if stage["status"] != "never_started":
                    continue
                plan_stage = plan["stages"][index]
                failed_dependencies = [
                    dependency
                    for dependency in plan_stage["dependencies"]
                    if by_id[dependency]["status"] != "completed"
                ]
                if failed_dependencies:
                    stage.update(
                        {
                            "status": "failed",
                            "terminal_at_utc": utc_now(),
                            "failure": {
                                "kind": "upstream_stage_failed",
                                "message": f"dependencies not completed: {failed_dependencies}",
                            },
                        }
                    )
                    save_ledger(ledger_path, ledger)
                    continue
                try:
                    verify_frozen_plan(plan, repo_root, full_data_hash=False)
                    verify_ledger_plan_binding(plan, ledger)
                    for dependency in plan_stage["dependencies"]:
                        verify_completed_stage_receipts(by_id[dependency], repo_root)
                    for raw in plan_stage["declared_outputs"]:
                        require(not Path(raw).exists(), f"stage output exists before one allowed attempt: {raw}")
                except Exception as error:
                    stage.update(
                        {
                            "status": "failed",
                            "terminal_at_utc": utc_now(),
                            "failure": {
                                "kind": "prelaunch_input_drift",
                                "message": f"{type(error).__name__}: {error}",
                            },
                        }
                    )
                    save_ledger(ledger_path, ledger)
                    continue

                stage.update(
                    {
                        "status": "running",
                        "attempt_count": 1,
                        "started_at_utc": utc_now(),
                    }
                )
                save_ledger(ledger_path, ledger)
                log_path = Path(stage["log_path"])
                exit_code: int | None = None

                def on_started(pid: int) -> None:
                    require(stage["subprocess_launch_count"] == 0, "stage process already launched")
                    stage["pid"] = int(pid)
                    # The production executor starts every stage in a new
                    # session, so its PID is also the process-group ID.  This
                    # lets a resumed orchestrator fail closed if a hard-killed
                    # parent left an MPS-using descendant alive.
                    stage["process_group_id"] = int(pid)
                    stage["subprocess_launch_count"] = 1
                    save_ledger(ledger_path, ledger)

                try:
                    with log_path.open("x", encoding="utf-8") as log:
                        log.write(
                            f"campaign_id={EVALUATION_CAMPAIGN_ID}\n"
                            f"stage_id={stage['stage_id']}\n"
                            f"command={plan_stage['command_shell']}\n\n"
                        )
                        log.flush()
                        exit_code = executor(list(plan_stage["command"]), repo_root, log, on_started)
                    stage["exit_code"] = int(exit_code)
                    stage["log_sha256"] = sha256_file(log_path)
                    require(stage["subprocess_launch_count"] == 1, "executor did not record one process")
                    require(exit_code in plan_stage["accepted_exit_codes"], f"stage exited {exit_code}")
                    verify_frozen_plan(plan, repo_root, full_data_hash=False)
                    stage["artifacts"] = stage_validator(plan_stage, plan, repo_root)
                    stage.update(
                        {
                            "status": "completed",
                            "terminal_at_utc": utc_now(),
                            "failure": None,
                        }
                    )
                except BaseException as error:
                    if log_path.exists():
                        stage["log_sha256"] = sha256_file(log_path)
                    stage.update(
                        {
                            "exit_code": exit_code,
                            "status": "failed",
                            "terminal_at_utc": utc_now(),
                            "failure": {
                                "kind": "stage_failed",
                                "message": f"{type(error).__name__}: {error}",
                                "traceback": "".join(
                                    traceback.format_exception(
                                        type(error), error, error.__traceback__
                                    )
                                ),
                            },
                        }
                    )
                    save_ledger(ledger_path, ledger)
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                save_ledger(ledger_path, ledger)

            if all(stage["status"] in {"completed", "failed"} for stage in ledger["stages"]):
                ledger["terminal_integrity"] = _terminal_integrity(plan, ledger, repo_root)
                save_ledger(ledger_path, ledger)
        except BaseException:
            invocation["status"] = "interrupted"
            invocation["finished_at_utc"] = utc_now()
            save_ledger(ledger_path, ledger)
            raise
        invocation["status"] = "finished"
        invocation["finished_at_utc"] = utc_now()
        save_ledger(ledger_path, ledger)
        return ledger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    parser.add_argument(
        "--resume-pending",
        action="store_true",
        help="Launch only never-started stages; attempted stages remain terminal and are never retried.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require(not (args.prepare_only and args.execute_prepared), "prepare/execute flags are mutually exclusive")
    repo_root = REPO_ROOT
    output_root = resolve_path(args.output_root, repo_root)
    if args.execute_prepared:
        ledger = execute_prepared(output_root, resume_pending=args.resume_pending)
        print(json.dumps(ledger["summary"], indent=2, sort_keys=True))
        return 0 if ledger["summary"]["status"] == "completed" else 2
    require(not args.resume_pending, "--resume-pending requires --execute-prepared")
    prepare_evaluation(
        repo_root=repo_root,
        data_root=resolve_path(args.data_root, repo_root),
        checkpoint_root=resolve_path(args.checkpoint_root, repo_root),
        output_root=output_root,
        python=args.python,
    )
    if args.prepare_only:
        print(f"Prepared event-time evaluation plan at: {output_root}")
        return 0
    ledger = execute_prepared(output_root, resume_pending=False)
    print(json.dumps(ledger["summary"], indent=2, sort_keys=True))
    return 0 if ledger["summary"]["status"] == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

import json
from pathlib import Path

import pytest
import torch

from scripts import build_matched_rollout_reference as reference_builder
from scripts import run_all_method_rollout_panel as rollout_runner
from scripts import run_intermediate_consumer_panel as consumer_runner
from experiments.multi_asset_event_time import run_evaluation as p4


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _absolute_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": p4.sha256_file(path),
    }


def _checkpoint(
    path: Path,
    *,
    dataset: str,
    seed: int,
    representation: str,
    optimizer_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 50,
            "ema_state_dict": {"fixture": torch.tensor([1.0])},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(float(optimizer_step))}}
            },
            "args": {
                "dataset": dataset,
                "seed": seed,
                "representation": representation,
                "epochs": 50,
                "max_train_batches": 548,
                "max_val_batches": 183,
                "batch_size": 128,
                "hist_len": 50,
                "pred_len": 10,
                "action_conditioning": "none",
                "price_reference": "history_last_mid",
                "p4_training_entrypoint": (
                    "experiments/multi_asset_event_time/train_fixed_steps.py"
                ),
            },
            "representation_stats": {
                "mean": torch.zeros(1, 2, 20, 1),
                "std": torch.ones(1, 2, 20, 1),
            },
        },
        path,
    )


def _fixture(tmp_path: Path, *, wrong_step_cell: str | None = None):
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in p4.SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n")

    data_root = repo / "data"
    data_receipts = {}
    for symbol, dataset in p4.ASSETS:
        path = data_root / "processed" / "snapshot_l2" / dataset / "metadata.json"
        _json(path, {"symbol": symbol, "dataset_id": dataset})
        data_receipts[f"{symbol}/metadata.json"] = _absolute_record(path)

    checkpoint_root = repo / "checkpoints"
    cells = []
    for slot, (symbol, dataset, seed, representation) in enumerate(
        p4._canonical_training_cells(), start=1
    ):
        cell_id = f"{symbol.lower()}__train_seed{seed}__{representation}"
        path = (
            checkpoint_root
            / symbol
            / f"train_seed{seed}"
            / representation
            / p4.CHECKPOINT_NAME
        )
        _checkpoint(
            path,
            dataset=dataset,
            seed=seed,
            representation=representation,
            optimizer_step=(
                p4.TARGET_OPTIMIZER_STEPS - 1
                if wrong_step_cell == cell_id
                else p4.TARGET_OPTIMIZER_STEPS
            ),
        )
        cells.append(
            {
                "cell_id": cell_id,
                "denominator_slot": slot,
                "symbol": symbol,
                "dataset": dataset,
                "train_seed": seed,
                "representation": representation,
                "save_dir": str(path.parent.resolve()),
                "status": "completed",
                "attempt_count": 1,
                "subprocess_launch_count": 1,
                "exit_code": 0,
                "artifacts": {"files": {"final.pt": _absolute_record(path)}},
            }
        )
    snapshot = {
        "campaign_id": p4.TRAINING_CAMPAIGN_ID,
        "protocol": {
            "assets": [symbol for symbol, _ in p4.ASSETS],
            "datasets": [dataset for _, dataset in p4.ASSETS],
            "train_seeds": list(p4.TRAIN_SEEDS),
            "representations": list(p4.REPRESENTATIONS),
            "checkpoint_for_evaluation": "final.pt",
            "target_optimizer_steps": p4.TARGET_OPTIMIZER_STEPS,
        },
        "data_files": data_receipts,
    }
    ledger = {
        "schema_version": 1,
        "campaign_id": p4.TRAINING_CAMPAIGN_ID,
        "snapshot": snapshot,
        "snapshot_sha256": p4.object_sha256(snapshot),
        "summary": {
            "status": "completed",
            "expected": 24,
            "completed": 24,
            "failed": 0,
            "terminal": 24,
        },
        "protocol_drift": {"latched": False},
        "final_integrity": {"status": "passed"},
        "cells": cells,
    }
    _json(checkpoint_root / p4.TRAINING_LEDGER_NAME, ledger)
    return repo, data_root, checkpoint_root


def _prepare(tmp_path: Path):
    repo, data_root, checkpoint_root = _fixture(tmp_path)
    output_root = repo / "results" / "evaluation"
    plan = p4.prepare_evaluation(
        repo_root=repo,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        python="fixture-python",
    )
    return repo, output_root, plan


def test_checkpoint_name_flags_are_backward_compatible_and_path_safe():
    assert rollout_runner.parse_args([]).checkpoint_name == "best.pt"
    assert consumer_runner.parse_args([]).checkpoint_name == "best.pt"
    assert reference_builder.parse_args(
        [
            "--results-roots", "fixture",
            "--dataset", "fixture",
            "--checkpoint-root", "fixture",
            "--output-csv", "fixture.csv",
            "--validation-json", "fixture.json",
        ]
    ).checkpoint_name == "best.pt"
    assert rollout_runner.parse_args(["--checkpoint-name", "final.pt"]).checkpoint_name == "final.pt"
    assert consumer_runner.parse_args(["--checkpoint-name", "final.pt"]).checkpoint_name == "final.pt"
    with pytest.raises(SystemExit):
        rollout_runner.parse_args(["--checkpoint-name", "../final.pt"])


def test_prepare_binds_final_checkpoint_semantics_and_exact_denominators(tmp_path):
    _, output_root, plan = _prepare(tmp_path)

    assert plan["protocol"]["assets"] == ["AAPL", "AMZN", "GOOG", "INTC"]
    assert plan["protocol"]["checkpoint_name"] == "final.pt"
    assert plan["protocol"]["reference_reproduction"] == {
        "non_endpoint_relative_tolerance": 1e-5,
        "non_endpoint_absolute_tolerance": 5e-7,
        "endpoint_invalidity_absolute_tolerance": 5e-5,
        "endpoint_frames_per_cell": 64_000,
        "endpoint_tolerance_equivalent_frames_per_cell": 3.2,
        "rule": (
            "Only endpoint pre-invalidity receives the declared 5e-5 "
            "diagnostic parity tolerance; every other reference error "
            "remains disallowed."
        ),
    }
    assert plan["denominators"] == {
        "training_checkpoints": 24,
        "assets": 4,
        "train_eval_cells": 36,
        "method_rows": 108,
        "histories_per_cell": 640,
        "reference_rows": 36,
        "execution_stages": 21,
    }
    assert len(plan["provenance"]["checkpoints"]) == 24
    assert all(
        receipt["epoch"] == 50
        and receipt["optimizer_step_min"] == 27_400
        and receipt["optimizer_step_max"] == 27_400
        for receipt in plan["provenance"]["checkpoints"].values()
    )
    assert len(plan["stages"]) == 21
    checkpoint_commands = [
        stage["command"]
        for stage in plan["stages"]
        if stage["kind"] in {"reference_eval", "reference_builder", "consumer"}
    ]
    assert len(checkpoint_commands) == 20
    assert all(
        command[command.index("--checkpoint-name") + 1] == "final.pt"
        for command in checkpoint_commands
    )
    stored_plan = output_root / "_orchestrator" / p4.PLAN_NAME
    ledger = p4.read_json(output_root / "_orchestrator" / p4.LEDGER_NAME)
    assert ledger["plan_sha256"] == p4.sha256_file(stored_plan)
    assert ledger["summary"]["expected_stages"] == 21
    assert ledger["summary"]["never_started"] == 21


def test_prepare_rejects_filename_only_final_with_wrong_optimizer_budget(tmp_path):
    bad_cell = "aapl__train_seed0__legal_lob"
    repo, data_root, checkpoint_root = _fixture(
        tmp_path, wrong_step_cell=bad_cell
    )

    with pytest.raises(p4.EvaluationError, match="Adam steps"):
        p4.prepare_evaluation(
            repo_root=repo,
            data_root=data_root,
            checkpoint_root=checkpoint_root,
            output_root=repo / "results" / "evaluation",
            python="fixture-python",
        )


def _consumer_artifact_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "results" / "AAPL" / "consumer"
    root.mkdir(parents=True)
    checkpoint_receipts = {}
    checkpoint_payloads = {}
    for seed in p4.TRAIN_SEEDS:
        checkpoint_payloads[seed] = {}
        for representation in p4.REPRESENTATIONS:
            checkpoint = (
                repo
                / "checkpoints"
                / "AAPL"
                / f"train_seed{seed}"
                / representation
                / "final.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{seed}:{representation}".encode())
            receipt = p4.file_receipt(checkpoint, repo)
            checkpoint_receipts[f"AAPL/train_seed{seed}/{representation}"] = receipt
            checkpoint_payloads[seed][representation] = {
                "path": str(checkpoint.resolve()),
                "sha256": receipt["sha256"],
            }
    starts_hash = p4.object_sha256({"starts": "fixed"})
    real_hash = p4.object_sha256({"real": "fixed"})
    manifest = {
        "status": "completed",
        "completed_cells": 9,
        "failed_cells": 0,
        "config": {
            "dataset": "PUBLIC_ITCH_2019_SAMPLES_AAPL_event",
            "checkpoint_name": "final.pt",
            "train_seeds": [0, 1, 2],
            "eval_seeds": [0, 1, 2],
            "n_histories": 640,
        },
        "provenance": {"data": {"start_indices_sha256": starts_hash}},
        "cells": [],
    }
    for train_seed in p4.TRAIN_SEEDS:
        for eval_seed in p4.EVAL_SEEDS:
            cell_path = (
                root
                / f"train_seed{train_seed}"
                / f"eval_seed{eval_seed}"
                / "consumer_stats.json"
            )
            post_hash = p4.object_sha256({"eval_seed": eval_seed, "post": True})
            noise = [
                p4.object_sha256({"eval_seed": eval_seed, "draw": index})
                for index in range(50)
            ]
            payload = {
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "status": "completed",
                "history_starts_sha256": starts_hash,
                "real_futures_sha256": real_hash,
                "checkpoints": checkpoint_payloads[train_seed],
                "post_model_load_rng_sha256": {
                    method: post_hash for method in p4.METHODS
                },
                "paired_initial_noise_sha256": noise,
                "methods": {
                    method: {"n_histories": 640} for method in p4.METHODS
                },
                "validation": {
                    "implementation_checks": {
                        "legal_lob_projection_calls": True,
                        "matched_posthoc_pava_projection_calls": True,
                        "matched_repeated_pava_projection_calls": True,
                        # A scientific negative result must not invalidate the
                        # execution receipt.
                        "legal_lob_endpoint_delivered_acceptance": False,
                    }
                },
            }
            _json(cell_path, payload)
            manifest["cells"].append(
                {
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "status": "completed",
                    "path": str(cell_path.resolve()),
                    "sha256": p4.sha256_file(cell_path),
                }
            )
    _json(root / "run_manifest.json", manifest)
    plan = {"provenance": {"checkpoints": checkpoint_receipts}}
    stage = {
        "stage_id": "aapl__consumer",
        "symbol": "AAPL",
        "kind": "consumer",
        "declared_outputs": [str(root.resolve())],
    }
    return repo, root, stage, plan


def test_consumer_gate_separates_science_and_proves_cross_checkpoint_pairing(tmp_path):
    repo, root, stage, plan = _consumer_artifact_fixture(tmp_path)

    receipt = p4._validate_consumer(stage, plan, repo)

    assert receipt["train_eval_cells"] == 9
    assert receipt["method_rows"] == 27
    assert set(receipt["cross_training_seed_pairing_by_eval_seed"]) == {
        "0", "1", "2"
    }

    manifest_path = root / "run_manifest.json"
    manifest = p4.read_json(manifest_path)
    target_entry = next(
        entry
        for entry in manifest["cells"]
        if entry["train_seed"] == 2 and entry["eval_seed"] == 1
    )
    target = Path(target_entry["path"])
    payload = p4.read_json(target)
    payload["paired_initial_noise_sha256"][0] = "f" * 64
    _json(target, payload)
    target_entry["sha256"] = p4.sha256_file(target)
    _json(manifest_path, manifest)

    with pytest.raises(p4.EvaluationError, match="noise differs across training"):
        p4._validate_consumer(stage, plan, repo)


def test_interrupted_stage_is_not_retried_but_pending_independent_assets_resume(tmp_path):
    repo, output_root, _ = _prepare(tmp_path)
    calls = []

    def interrupt_first(command, cwd, log, on_started):
        calls.append(tuple(command))
        on_started(101)
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        p4.execute_prepared(
            output_root,
            resume_pending=False,
            executor=interrupt_first,
            stage_validator=lambda stage, plan, root: {},
        )
    ledger_path = output_root / "_orchestrator" / p4.LEDGER_NAME
    after_interrupt = p4.read_json(ledger_path)
    first = after_interrupt["stages"][0]
    assert first["status"] == "failed"
    assert first["attempt_count"] == 1
    assert first["subprocess_launch_count"] == 1

    with pytest.raises(p4.EvaluationError, match="--resume-pending"):
        p4.execute_prepared(
            output_root,
            resume_pending=False,
            executor=interrupt_first,
            stage_validator=lambda stage, plan, root: {},
        )

    resumed_calls = []

    def complete(command, cwd, log, on_started):
        resumed_calls.append(tuple(command))
        on_started(202)
        return 0

    def validate(stage, plan, root):
        if stage["kind"] == "reference_builder":
            return {"reference_rows": 9}
        if stage["kind"] == "consumer":
            return {"train_eval_cells": 9, "method_rows": 27}
        return {}

    final = p4.execute_prepared(
        output_root,
        resume_pending=True,
        executor=complete,
        stage_validator=validate,
    )
    assert calls[0] not in resumed_calls
    assert final["stages"][0]["attempt_count"] == 1
    assert final["stages"][0]["subprocess_launch_count"] == 1
    assert final["summary"]["status"] == "terminal_with_declared_failures"
    assert final["summary"]["failed"] > 0
    assert final["summary"]["never_started"] == 0


def _complete_stage_with_fixture_artifact(stage: dict, repo: Path) -> Path:
    output = Path(stage["declared_outputs"][0])
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "fixture.json"
    _json(artifact, {"stage_id": stage["stage_id"], "value": "frozen"})
    log_path = Path(stage["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"stage_id={stage['stage_id']}\n")
    stage.update(
        {
            "status": "completed",
            "attempt_count": 1,
            "subprocess_launch_count": 1,
            "exit_code": 0,
            "log_sha256": p4.sha256_file(log_path),
            "artifacts": {
                "files": p4._declared_output_receipts(stage, repo),
            },
        }
    )
    return artifact


def test_completed_stage_artifact_tamper_fails_downstream_prelaunch(tmp_path):
    repo, output_root, _ = _prepare(tmp_path)
    ledger_path = output_root / "_orchestrator" / p4.LEDGER_NAME
    ledger = p4.read_json(ledger_path)
    aapl_references = [
        stage
        for stage in ledger["stages"]
        if stage["symbol"] == "AAPL" and stage["kind"] == "reference_eval"
    ]
    assert len(aapl_references) == 3
    artifacts = [
        _complete_stage_with_fixture_artifact(stage, repo)
        for stage in aapl_references
    ]
    artifacts[0].write_text('{"tampered": true}\n')
    p4.save_ledger(ledger_path, ledger)

    def interrupt_next_independent_stage(command, cwd, log, on_started):
        on_started(777777)
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        p4.execute_prepared(
            output_root,
            resume_pending=True,
            executor=interrupt_next_independent_stage,
            stage_validator=lambda stage, plan, root: {},
        )

    observed = p4.read_json(ledger_path)
    builder = next(
        stage
        for stage in observed["stages"]
        if stage["symbol"] == "AAPL" and stage["kind"] == "reference_builder"
    )
    assert builder["status"] == "failed"
    assert builder["attempt_count"] == 0
    assert builder["failure"]["kind"] == "prelaunch_input_drift"
    assert "artifact receipt drift" in builder["failure"]["message"]


def test_terminal_integrity_rehashes_completed_stage_artifacts(tmp_path):
    repo, output_root, _ = _prepare(tmp_path)
    plan = p4.read_json(output_root / "_orchestrator" / p4.PLAN_NAME)
    ledger = p4.read_json(output_root / "_orchestrator" / p4.LEDGER_NAME)
    stage = ledger["stages"][0]
    artifact = _complete_stage_with_fixture_artifact(stage, repo)
    artifact.write_text('{"tampered": true}\n')

    receipt = p4._terminal_integrity(plan, ledger, repo)

    assert receipt["status"] == "failed"
    assert any(
        f"stage_receipts[{stage['stage_id']}]" in error
        and "artifact receipt drift" in error
        for error in receipt["errors"]
    )


def _mark_all_stages_completed_for_summary(ledger: dict) -> None:
    for stage in ledger["stages"]:
        artifacts = {}
        if stage["kind"] == "reference_builder":
            artifacts["reference_rows"] = 9
        elif stage["kind"] == "consumer":
            artifacts.update({"train_eval_cells": 9, "method_rows": 27})
        elif stage["kind"] == "panel_c":
            artifacts["result_class"] = "verified_go"
        stage.update(
            {
                "status": "completed",
                "attempt_count": 1,
                "subprocess_launch_count": 1,
                "artifacts": artifacts,
            }
        )


def test_terminal_integrity_failure_overrides_completed_stage_grid_and_cli(
    tmp_path, monkeypatch
):
    _, output_root, _ = _prepare(tmp_path)
    ledger = p4.read_json(output_root / "_orchestrator" / p4.LEDGER_NAME)
    _mark_all_stages_completed_for_summary(ledger)

    ledger["terminal_integrity"] = None
    assert p4.ledger_summary(ledger)["status"] == "in_progress"

    ledger["terminal_integrity"] = {"status": "failed", "errors": ["tamper"]}
    ledger["summary"] = p4.ledger_summary(ledger)
    assert ledger["summary"]["status"] == "terminal_with_declared_failures"
    assert ledger["summary"]["campaign_result_class"] == "terminal_with_declared_failures"
    assert ledger["summary"]["terminal_integrity_status"] == "failed"

    monkeypatch.setattr(
        p4,
        "execute_prepared",
        lambda output_root, resume_pending: ledger,
    )
    assert p4.main(["--execute-prepared", "--output-root", str(output_root)]) == 2

    ledger["terminal_integrity"] = {"status": "passed", "errors": []}
    passed = p4.ledger_summary(ledger)
    assert passed["status"] == "completed"
    assert passed["campaign_result_class"] == "verified_go"


def test_mutated_ledger_command_is_rejected_before_any_stage_launch(tmp_path):
    _, output_root, _ = _prepare(tmp_path)
    ledger_path = output_root / "_orchestrator" / p4.LEDGER_NAME
    ledger = p4.read_json(ledger_path)
    ledger["stages"][0]["command"].append("--protocol-drift")
    _json(ledger_path, ledger)
    launches = []

    def must_not_launch(command, cwd, log, on_started):
        launches.append(command)
        raise AssertionError("mutated command was launched")

    with pytest.raises(p4.EvaluationError, match="immutable field drift.*command"):
        p4.execute_prepared(
            output_root,
            resume_pending=False,
            executor=must_not_launch,
            stage_validator=lambda stage, plan, root: {},
        )
    assert launches == []


def test_ledger_log_path_is_derived_from_hashed_plan_output_root(tmp_path):
    _, output_root, _ = _prepare(tmp_path)
    plan = p4.read_json(output_root / "_orchestrator" / p4.PLAN_NAME)
    ledger = p4.read_json(output_root / "_orchestrator" / p4.LEDGER_NAME)
    ledger["stages"][0]["log_path"] = str((tmp_path / "redirected.log").resolve())

    with pytest.raises(p4.EvaluationError, match="log path drift"):
        p4.verify_ledger_plan_binding(plan, ledger)


def test_terminal_integrity_requires_one_attempt_launch_and_accepted_exit(tmp_path):
    repo, output_root, _ = _prepare(tmp_path)
    plan = p4.read_json(output_root / "_orchestrator" / p4.PLAN_NAME)
    ledger = p4.read_json(output_root / "_orchestrator" / p4.LEDGER_NAME)
    _mark_all_stages_completed_for_summary(ledger)
    ledger["stages"][0]["attempt_count"] = 0
    ledger["stages"][1]["subprocess_launch_count"] = 0
    ledger["stages"][2]["exit_code"] = 9

    receipt = p4._terminal_integrity(plan, ledger, repo)

    assert receipt["status"] == "failed"
    assert any(
        "one-attempt/one-launch/accepted-exit gate failed" in error
        for error in receipt["errors"]
    )

import csv
import json
from pathlib import Path

import pytest

from scripts.build_matched_rollout_reference import (
    collect_reference,
    sha256_file,
    write_reference,
)


DATASET = "PUBLIC_ITCH_2019_SAMPLES_TEST_1s"


def _stats(checkpoint: Path, *, ddim_steps: int = 20) -> dict:
    snapshot = {
        "violations_raw": {
            "any_invalid_snapshot_rate": 0.0,
            "crossed_book_rate": 0.0,
            "ask_ladder_violation": 0.0,
            "bid_ladder_violation": 0.0,
            "negative_volume_rate": 0.0,
        },
        "projection_distance": {"total_l2_distance": 0.0},
        "spread_mean": 1.1,
        "spread_real_mean": 1.0,
        "volume_mean": 0.9,
        "volume_real_mean": 1.0,
        "mid_return_std": 0.8,
        "mid_return_real_std": 1.0,
        "temporal_corr": 0.25,
        "w1_mid_return": 0.1,
        "w1_spread": 0.2,
        "w1_volume": 0.3,
    }
    posthoc = {
        "n_batches": 5,
        "n_trajectories": 640,
        "n_projection_calls_per_trajectory": 10,
        "n_projection_kernel_calls": 50,
        "pre_projection_any_invalid_snapshot_rate_mean": 0.4,
        "cumulative_projection_l2_mean": 1.5,
    }
    return {
        "checkpoint": {"path": str(checkpoint), "epoch": 50, "val_loss": 0.1},
        "config": {
            "dataset": DATASET,
            "data_dir": "data",
            "hist_len": 50,
            "pred_len": 10,
            "chunk_size": 10,
            "horizon": 100,
            "ddim_steps": ddim_steps,
            "batch_size": 128,
            "n_histories": 640,
            "action_mode": "hold",
            "projection_mode": "exact",
            "sampler_projection_mode": "none",
            "compute_projection_distance": True,
            "representation": "matched_lob",
            "price_reference": "history_last_mid",
            "weights": "ema",
        },
        "generated": {"step_100": snapshot},
        "rollout_projection_stats": {
            "posthoc_chunk_projection": posthoc,
            "by_horizon": {"step_100": {"posthoc_chunk_projection": posthoc}},
        },
        "runtime": {"rollout_seconds": 50.0, "trajectories_per_second": 12.8},
    }


def _command(checkpoint: Path, output_dir: Path, eval_seed: int) -> str:
    return " ".join(
        [
            "python",
            "scripts/snowball_test.py",
            "--genlob-ckpt",
            str(checkpoint),
            "--weights ema",
            f"--dataset {DATASET}",
            "--data-dir data",
            "--hist-len 50",
            "--pred-len 10",
            "--chunk-size 10",
            "--horizon 100",
            "--ddim-steps 20",
            "--representation auto",
            "--projection-mode exact",
            "--sampler-projection-mode none",
            "--compute-projection-distance",
            "--skip-plots",
            "--action-mode hold",
            "--batch-size 128",
            "--n-batches 5",
            f"--output-dir {output_dir}",
            f"--seed {eval_seed}",
            "--device mps",
        ]
    )


def _fixture(
    tmp_path: Path, *, checkpoint_name: str = "best.pt"
) -> tuple[list[Path], Path, Path]:
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "processed" / "snapshot_l2" / DATASET
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.yaml").write_text("dataset: fixture\n")
    (dataset_dir / "normalization.json").write_text("{}\n")
    checkpoint_root = tmp_path / "checkpoints"
    for train_seed in range(3):
        checkpoint = (
            checkpoint_root
            / f"train_seed{train_seed}"
            / "matched_lob"
            / checkpoint_name
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{train_seed}".encode())

    roots = []
    for eval_seed in range(3):
        root = tmp_path / "results" / f"eval_seed{eval_seed}"
        roots.append(root)
        runs = []
        for train_seed in range(3):
            checkpoint = (
                checkpoint_root
                / f"train_seed{train_seed}"
                / "matched_lob"
                / checkpoint_name
            )
            output_dir = root / f"train_seed{train_seed}" / "matched_posthoc_pava"
            output_dir.mkdir(parents=True)
            command = _command(checkpoint, output_dir, eval_seed)
            (output_dir / "command.txt").write_text(command + "\n")
            (output_dir / "snowball_stats.json").write_text(
                json.dumps(_stats(checkpoint))
            )
            runs.append(
                {
                    "train_seed": train_seed,
                    "method": "matched_posthoc_pava",
                    "checkpoint": str(checkpoint),
                    "output_dir": str(output_dir),
                    "command": command,
                    "status": "completed",
                }
            )
        manifest = {
            "dataset": DATASET,
            "checkpoint_root": str(checkpoint_root),
            "train_seeds": [0, 1, 2],
            "eval_seed": eval_seed,
            "n_batches": 5,
            "batch_size": 128,
            "horizon": 100,
            "ddim_steps": 20,
            "methods": ["matched_posthoc_pava"],
            "runs": runs,
        }
        # Old manifests omitted this field and implied best.pt.  Preserve that
        # fixture as the backward-compatibility case.
        if checkpoint_name != "best.pt":
            manifest["checkpoint_name"] = checkpoint_name
        (root / "run_manifest.json").write_text(json.dumps(manifest))
    return roots, data_dir, checkpoint_root


def _collect(tmp_path: Path):
    roots, data_dir, checkpoint_root = _fixture(tmp_path)
    rows, validation = collect_reference(
        results_roots=roots,
        dataset=DATASET,
        data_dir=data_dir,
        checkpoint_root=checkpoint_root,
        train_seeds=[0, 1, 2],
        eval_seeds=[0, 1, 2],
        hist_len=50,
        pred_len=10,
        chunk_size=10,
        horizon=100,
        ddim_steps=20,
        batch_size=128,
        n_batches=5,
        device="mps",
    )
    return roots, rows, validation


def test_collects_complete_three_by_three_reference_with_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, rows, validation = _collect(tmp_path)
    output_csv = tmp_path / "reference.csv"
    validation_json = tmp_path / "reference_validation.json"
    write_reference(
        rows,
        validation,
        output_csv=output_csv,
        validation_json=validation_json,
    )

    exported = list(csv.DictReader(output_csv.open()))
    receipt = json.loads(validation_json.read_text())
    assert len(exported) == 9
    assert {(int(row["train_seed"]), int(row["eval_seed"])) for row in exported} == {
        (train_seed, eval_seed)
        for train_seed in range(3)
        for eval_seed in range(3)
    }
    assert all(row["method_id"] == "matched_posthoc_pava" for row in exported)
    assert receipt["status"] == "validated"
    assert receipt["completeness"] == {
        "expected_cells": 9,
        "observed_cells": 9,
        "all_cells_complete": True,
    }
    assert receipt["reference_csv"]["sha256"] == sha256_file(output_csv)
    assert len(receipt["cells"]) == 9
    assert all(len(cell["checkpoint_sha256"]) == 64 for cell in receipt["cells"])


def test_collects_explicit_final_checkpoint_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roots, data_dir, checkpoint_root = _fixture(
        tmp_path, checkpoint_name="final.pt"
    )

    rows, validation = collect_reference(
        results_roots=roots,
        dataset=DATASET,
        data_dir=data_dir,
        checkpoint_root=checkpoint_root,
        checkpoint_name="final.pt",
        train_seeds=[0, 1, 2],
        eval_seeds=[0, 1, 2],
        hist_len=50,
        pred_len=10,
        chunk_size=10,
        horizon=100,
        ddim_steps=20,
        batch_size=128,
        n_batches=5,
        device="mps",
    )

    assert len(rows) == 9
    assert validation["expected_protocol"]["checkpoint_name"] == "final.pt"
    assert all(Path(row["checkpoint_path"]).name == "final.pt" for row in rows)


def test_rejects_missing_manifest_cell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roots, data_dir, checkpoint_root = _fixture(tmp_path)
    manifest_path = roots[1] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runs"].pop()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest run count"):
        collect_reference(
            results_roots=roots,
            dataset=DATASET,
            data_dir=data_dir,
            checkpoint_root=checkpoint_root,
            train_seeds=[0, 1, 2],
            eval_seeds=[0, 1, 2],
            hist_len=50,
            pred_len=10,
            chunk_size=10,
            horizon=100,
            ddim_steps=20,
            batch_size=128,
            n_batches=5,
            device="mps",
        )


def test_rejects_stats_protocol_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roots, data_dir, checkpoint_root = _fixture(tmp_path)
    stats_path = roots[2] / "train_seed1" / "matched_posthoc_pava" / "snowball_stats.json"
    stats = json.loads(stats_path.read_text())
    stats["config"]["ddim_steps"] = 19
    stats_path.write_text(json.dumps(stats))

    with pytest.raises(ValueError, match="stats.config.ddim_steps"):
        collect_reference(
            results_roots=roots,
            dataset=DATASET,
            data_dir=data_dir,
            checkpoint_root=checkpoint_root,
            train_seeds=[0, 1, 2],
            eval_seeds=[0, 1, 2],
            hist_len=50,
            pred_len=10,
            chunk_size=10,
            horizon=100,
            ddim_steps=20,
            batch_size=128,
            n_batches=5,
            device="mps",
        )


def test_refuses_to_overwrite_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, rows, validation = _collect(tmp_path)
    output_csv = tmp_path / "reference.csv"
    validation_json = tmp_path / "reference_validation.json"
    output_csv.write_text("existing\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_reference(
            rows,
            validation,
            output_csv=output_csv,
            validation_json=validation_json,
        )
    assert output_csv.read_text() == "existing\n"
    assert not validation_json.exists()

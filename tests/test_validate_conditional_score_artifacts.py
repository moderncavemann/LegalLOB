import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.validate_conditional_score_artifacts import (
    CAMPAIGN_FROZEN_DATA_FILES,
    CAMPAIGN_REQUIRED_SOURCES,
    EXPECTED_ASSETS,
    EXPECTED_SAMPLING_SEEDS,
    EXPECTED_SHAPE,
    EXPECTED_TRAIN_SEEDS,
    METHOD_CONTRACT,
    ValidationError,
    _checkpoint_receipt,
    _ensemble_pairwise_rmse,
    _pairwise_rmse,
    _require_aligned,
    _structural_audit,
    bind_campaign_checkpoint_hash,
    parse_asset_roots,
    sha256_file,
    validate_campaign_manifest,
    validate_manifest_protocol,
    validate_method_artifact,
    write_validation_receipt,
)


def _manifest(repo_root: Path, result_root: Path, *, status: str = "completed"):
    checkpoint_root = repo_root / "checkpoints" / "AAPL"
    runs = []
    for train_seed in EXPECTED_TRAIN_SEEDS:
        for sampling_seed in EXPECTED_SAMPLING_SEEDS:
            for method, contract in METHOD_CONTRACT.items():
                checkpoint = (
                    checkpoint_root
                    / f"train_seed{train_seed}"
                    / contract["representation"]
                    / "best.pt"
                )
                output = (
                    result_root
                    / f"train_seed{train_seed}"
                    / f"sampling_seed{sampling_seed}"
                    / method
                )
                runs.append(
                    {
                        "train_seed": train_seed,
                        "sampling_seed": sampling_seed,
                        "method": method,
                        "checkpoint": str(checkpoint),
                        "output_dir": str(output),
                        "command": "python scripts/evaluate_v2.py",
                        "status": status,
                    }
                )
    return {
        "dataset": "PUBLIC_ITCH_2019_SAMPLES_AAPL_1s",
        "checkpoint_root": str(checkpoint_root),
        "train_seeds": list(EXPECTED_TRAIN_SEEDS),
        "sampling_seeds": list(EXPECTED_SAMPLING_SEEDS),
        "n_batches": 4,
        "batch_size": 128,
        "n_histories": 512,
        "ddim_steps": 20,
        "methods": sorted(METHOD_CONTRACT),
        "runs": runs,
    }


def _valid_book(price_shift: float = 0.0) -> np.ndarray:
    result = np.ones(EXPECTED_SHAPE, dtype=np.float32)
    ask = 1.0 + 0.1 * np.arange(10, dtype=np.float32) + price_shift
    bid = 0.5 - 0.1 * np.arange(10, dtype=np.float32) + price_shift
    result[:, 0, :10, :] = ask[None, :, None]
    result[:, 0, 10:, :] = bid[None, :, None]
    return result


def _structural_zero():
    return {
        "any_invalid_snapshot_rate": 0.0,
        "any_invalid_trajectory_rate": 0.0,
        "max_spread_shortfall": 0.0,
        "max_ask_gap_shortfall": 0.0,
        "max_bid_gap_shortfall": 0.0,
        "max_negative_volume_magnitude": 0.0,
    }


def _write_method_cell(
    repo_root: Path,
    result_root: Path,
    *,
    method: str,
    shift: float,
):
    train_seed = 0
    sampling_seed = 0
    dataset = "PUBLIC_ITCH_2019_SAMPLES_AAPL_1s"
    contract = METHOD_CONTRACT[method]
    checkpoint = (
        repo_root
        / "checkpoints"
        / "AAPL"
        / "train_seed0"
        / contract["representation"]
        / "best.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint-fixture")
    output_dir = result_root / "train_seed0" / "sampling_seed0" / method
    output_dir.mkdir(parents=True)
    command = " ".join(
        [
            "python",
            "scripts/evaluate_v2.py",
            "--checkpoint",
            str(checkpoint),
            "--weights",
            "ema",
            "--dataset",
            dataset,
            "--data-dir",
            str(repo_root / "data"),
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
            "--n-batches",
            "4",
            "--batch-size",
            "128",
            "--ddim-steps",
            "20",
            "--projection-mode",
            contract["projection_mode"],
            "--sampler-projection-mode",
            "none",
            "--representation",
            "auto",
            "--action-conditioning",
            "none",
            "--seed",
            "0",
            "--skip-plots",
        ]
    )
    (output_dir / "command.txt").write_text(command + "\n")
    (output_dir / "run.log").write_text("completed\n")
    structural_key = contract["feasibility_metrics_key"]
    metrics = {
        "representation": contract["representation"],
        "projection_mode": contract["projection_mode"],
        "sampler_projection_mode": "none",
        "min_spread": 0.1,
        "min_gap": 0.1,
        "evaluation_protocol": {
            "sampling_mode": "uniform",
            "selected_windows": 512,
            "action_conditioning": "none",
        },
        "mse_raw": 1.0,
        "mse_projected": 1.0,
        "mae_raw": 1.0,
        "mae_projected": 1.0,
        structural_key: _structural_zero(),
        "undefined_diagnostic": None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics))
    indices = np.arange(512, dtype=np.int64)
    np.savez_compressed(
        output_dir / "per_sample_metrics.npz",
        dataset_index=indices,
        start_indices=indices + 50,
        day_id=np.full(512, 20191030, dtype=np.int64),
        segment_id=np.zeros(512, dtype=np.int64),
        mse=np.ones(512),
    )
    evaluated = _valid_book(shift)
    raw = evaluated.copy()
    if method != "legal_lob":
        raw[:, 1] = -1.0  # Raw may be invalid; the evaluated PAVA output may not.
    np.save(output_dir / "generated_raw.npy", raw)
    np.save(output_dir / "generated_projected.npy", evaluated)
    np.save(output_dir / "real_future.npy", _valid_book(10.0))
    np.save(output_dir / "persistence.npy", _valid_book(9.0))
    run = {
        "checkpoint": str(checkpoint),
        "command": command,
    }
    return validate_method_artifact(
        run,
        repo_root=repo_root,
        result_root=result_root,
        dataset=dataset,
        train_seed=train_seed,
        sampling_seed=sampling_seed,
        method=method,
    )


def test_manifest_requires_exact_288_completed_cells(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    manifest = _manifest(tmp_path, result_root)
    indexed = validate_manifest_protocol(
        manifest,
        asset="AAPL",
        result_root=result_root,
        repo_root=tmp_path,
    )
    assert len(indexed) == 288

    manifest["runs"][0]["status"] = "existing"
    with pytest.raises(ValidationError, match="not completed"):
        validate_manifest_protocol(
            manifest,
            asset="AAPL",
            result_root=result_root,
            repo_root=tmp_path,
        )


def test_three_method_cell_is_aligned_feasible_finite_and_distinct(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results"
    outputs = {
        method: _write_method_cell(tmp_path, result_root, method=method, shift=shift)
        for method, shift in (
            ("legal_lob", 0.0),
            ("raw_posthoc_pava", 1.0),
            ("matched_posthoc_pava", 2.0),
        )
    }
    _require_aligned(outputs, "fixture")
    pairwise = _pairwise_rmse(
        {method: output["evaluated"] for method, output in outputs.items()}
    )
    assert len(pairwise) == 3
    assert min(pairwise.values()) > 1e-6
    for output in outputs.values():
        receipt = output["receipt"]
        assert receipt["structural_feasibility"]["feasible"] is True
        assert receipt["metrics_scan"]["numeric_leaves"] > 0
        assert set(receipt["result_files"]) == {
            "command.txt",
            "run.log",
            "metrics.json",
            "per_sample_metrics.npz",
            "generated_raw.npy",
            "generated_projected.npy",
            "real_future.npy",
            "persistence.npy",
        }
        assert all(
            len(item["sha256"]) == 64 for item in receipt["result_files"].values()
        )


def test_alignment_and_pairwise_guards_fail_closed() -> None:
    base = np.zeros((2,), dtype=np.int64)
    array = np.zeros((2, 2), dtype=np.float32)
    outputs = {
        "legal_lob": {
            "alignment": {"dataset_index": base},
            "real_future": array,
            "persistence": array,
        },
        "raw_posthoc_pava": {
            "alignment": {"dataset_index": np.array([0, 9])},
            "real_future": array,
            "persistence": array,
        },
    }
    with pytest.raises(ValidationError, match="dataset_index mismatch"):
        _require_aligned(outputs, "fixture")
    with pytest.raises(ValidationError, match="pairwise RMSE"):
        _pairwise_rmse({"a": array, "b": array.copy()})
    ensemble = [np.full((2, 1, 1, 1), index, dtype=np.float32) for index in range(32)]
    assert _ensemble_pairwise_rmse(ensemble, "fixture") > 1e-6
    with pytest.raises(ValidationError, match="ensemble pairwise RMSE"):
        _ensemble_pairwise_rmse(
            [np.zeros((2, 1, 1, 1), dtype=np.float32) for _ in range(32)],
            "collapsed",
        )


def test_structural_audit_checks_raw_legal_or_projected_baseline() -> None:
    valid = _valid_book()
    assert _structural_audit(valid, min_spread=0.1, min_gap=0.1)["feasible"]
    invalid = valid.copy()
    invalid[:, 1, 0, 0] = -1.0
    audit = _structural_audit(invalid, min_spread=0.1, min_gap=0.1)
    assert audit["feasible"] is False
    assert audit["negative_volume_count"] == 512


def test_checkpoint_receipt_hashes_config_and_enforces_contract(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    args = {
        "dataset": "PUBLIC_ITCH_2019_SAMPLES_AAPL_1s",
        "seed": 0,
        "representation": "legal_lob",
        "hist_len": 50,
        "pred_len": 10,
        "action_conditioning": "none",
        "price_reference": "history_last_mid",
        "min_spread": 0.1,
        "min_gap": 0.1,
    }
    torch.save({"args": args, "config": {"input_dim": 2, "n_features": 20}}, path)
    receipt = _checkpoint_receipt(
        path,
        repo_root=tmp_path,
        dataset=args["dataset"],
        train_seed=0,
        representation="legal_lob",
    )
    assert len(receipt["sha256"]) == 64
    assert len(receipt["config_sha256"]) == 64
    assert len(receipt["args_sha256"]) == 64
    campaign = {"frozen_checkpoints_sha256": {receipt["path"]: receipt["sha256"]}}
    bound = bind_campaign_checkpoint_hash(receipt, campaign)
    assert bound["matches_launch_preflight"] is True
    campaign["frozen_checkpoints_sha256"][receipt["path"]] = "0" * 64
    with pytest.raises(ValidationError, match="checkpoint hash mismatch"):
        bind_campaign_checkpoint_hash(receipt, campaign)


def test_validation_json_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    write_validation_receipt({"status": "VERIFIED"}, path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_validation_receipt({"status": "CHANGED"}, path)
    assert path.read_bytes() == original


def _campaign_fixture(tmp_path: Path) -> Path:
    sources = {}
    for relative in CAMPAIGN_REQUIRED_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")
        sources[relative] = sha256_file(path)
    datasets = {}
    for asset in EXPECTED_ASSETS:
        dataset = f"PUBLIC_ITCH_2019_SAMPLES_{asset}_1s"
        directory = tmp_path / "data" / "processed" / "snapshot_l2" / dataset
        directory.mkdir(parents=True)
        datasets[dataset] = {}
        for name in CAMPAIGN_FROZEN_DATA_FILES:
            path = directory / name
            path.write_text(f"{dataset}/{name}\n")
            datasets[dataset][name] = sha256_file(path)
    campaign = {
        "schema_version": 1,
        "campaign_id": "fixture",
        "status": "running",
        "sources_sha256": sources,
        "datasets": datasets,
        "frozen_checkpoints_sha256": {},
    }
    path = tmp_path / "campaign_manifest.json"
    path.write_text(json.dumps(campaign))
    return path


def test_campaign_manifest_matches_launch_source_and_dataset_hashes(
    tmp_path: Path,
) -> None:
    path = _campaign_fixture(tmp_path)
    _, evidence = validate_campaign_manifest(
        path, repo_root=tmp_path, data_root=tmp_path / "data"
    )
    assert evidence["campaign_id"] == "fixture"
    assert all(
        item["matches"]
        for item in evidence["source_hash_checks"].values()
    )

    changed = tmp_path / CAMPAIGN_REQUIRED_SOURCES[0]
    changed.write_text("changed after launch\n")
    with pytest.raises(ValidationError, match="source hash mismatch"):
        validate_campaign_manifest(
            path, repo_root=tmp_path, data_root=tmp_path / "data"
        )


def test_asset_cli_parser_rejects_duplicates() -> None:
    assert parse_asset_roots(["AAPL=/a", "AMZN=/b", "GOOG=/c"])["AAPL"] == Path("/a")
    with pytest.raises(Exception, match="duplicate asset root"):
        parse_asset_roots(["AAPL=/a", "AAPL=/b"])

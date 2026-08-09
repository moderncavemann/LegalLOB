import hashlib
import json
from pathlib import Path

import pytest

from scripts import summarize_intermediate_consumer_cross_asset as cross


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(frame_rejections: int) -> dict:
    block_rejections = min(frame_rejections, 1_152_000)
    trajectory_rejections = min(frame_rejections, 5_760)
    item = {
        "anytime_frame_requests": 11_520_000,
        "anytime_frame_rejections": frame_rejections,
        "anytime_block_requests": 1_152_000,
        "anytime_block_rejections": block_rejections,
        "endpoint_frame_requests": 576_000,
        "endpoint_frame_rejections": 0,
        "trajectories": 5_760,
        "trajectories_with_any_anytime_rejection": trajectory_rejections,
        "rejection_reason_counts": {reason: 0 for reason in cross.REJECTION_REASONS},
    }
    item["anytime_frame_acceptance_rate"] = 1 - frame_rejections / 11_520_000
    item["anytime_block_acceptance_rate"] = 1 - block_rejections / 1_152_000
    item["completed_without_anytime_rejection_rate"] = (
        1 - trajectory_rejections / 5_760
    )
    return item


def _fake_asset_result(asset: str, strict_go: bool = True) -> dict:
    rows = []
    for method_index, method_id in enumerate(cross.METHODS):
        row = {
            "asset": asset,
            "method_id": method_id,
            "method": cross.METHOD_LABELS[method_id],
        }
        for metric in cross.ALL_METRICS:
            if metric.direction == "up":
                value = (1.0, 0.5, 0.9)[method_index]
            else:
                value = (0.0, 1.0, 0.5)[method_index]
            row[f"{metric.key}_mean"] = value
            row[f"{metric.key}_std"] = 0.01
        rows.append(row)
    counts = {
        "legal_lob": _counts(0),
        "matched_posthoc_pava": _counts(100),
        "matched_repeated_pava": _counts(1),
    }
    return {
        "asset": asset,
        "dataset": cross.DATASETS[asset],
        "root": f"/fake/{asset}",
        "passed": True,
        "errors": [],
        "cells": 9,
        "input_hashes": {},
        "method_rows": rows,
        "counts": counts,
        "counts_verified": True,
        "outcome": {
            "legacy_experiment_integrity_verified": True,
            "reclassified_base_integrity_verified": True,
            "reclassified_experiment_integrity_verified": True,
            "pre_registered_strict_go": strict_go,
            "registered_scientific_result_class": (
                "strict_go" if strict_go else "validated_strict_no_go"
            ),
            "scientific_result_class": (
                "strict_go" if strict_go else "validated_strict_no_go"
            ),
            "failed_science_checks": [] if strict_go else ["registered_failure"],
        },
    }


def test_dense_ranks_use_direction_and_dense_ties():
    metric_up = cross.Metric("score", "Score", "up")
    metric_down = cross.Metric("cost", "Cost", "down")
    rows = [
        {"method_id": "a", "score_mean": 1.0, "cost_mean": 0.0},
        {"method_id": "b", "score_mean": 1.0 + 5e-10, "cost_mean": 2.0},
        {"method_id": "c", "score_mean": 0.5, "cost_mean": 1.0},
    ]

    assert cross.dense_ranks(rows, metric_up) == {"a": 1, "b": 1, "c": 2}
    assert cross.dense_ranks(rows, metric_down) == {"a": 1, "c": 2, "b": 3}


def test_validate_counts_requires_exact_fixed_denominators_and_recomputation():
    stored = {method: _counts(index) for index, method in enumerate(cross.METHODS)}
    errors = []
    cross._validate_counts("AAPL", stored, json.loads(json.dumps(stored)), errors)
    assert errors == []

    altered = json.loads(json.dumps(stored))
    altered["legal_lob"]["anytime_frame_requests"] -= 1
    errors = []
    cross._validate_counts("AAPL", altered, stored, errors)

    assert any("differ from cell recomputation" in error for error in errors)
    assert any("anytime_frame_requests" in error for error in errors)


def test_registered_strict_no_go_is_allowed_only_when_integrity_is_verified():
    counts = {"matched_repeated_pava": _counts(3)}
    validation = {
        "experiment_integrity_verified": True,
        "pre_registered_strict_go": False,
        "result_class": "validated_strict_no_go",
        "reason_for_strict_no_go": "registered performance failure",
        "reference_rollout_reproduced": True,
        "non_acceptance_reference_errors": [],
        "non_performance_cell_failures": [],
        "repeated_anytime_frame_rejections": 3,
        "repeated_anytime_frame_requests": 11_520_000,
        "repeated_trajectories_with_rejection": 3,
        "repeated_trajectories": 5_760,
        "base_pre_registered_validation": {
            "manifest_status": "completed",
            "expected_cells": 9,
            "completed_cells": 9,
            "method_rows": 27,
            "manifest_integrity_errors": [],
            "rng_pairing_failures": [],
            "denominator_failures": [],
            "source_hash_failures": [],
            "summary_completeness_failures": [],
            "cell_implementation_gate_failures": [],
            "science_checks": {"registered_check": False},
            "science_checks_pass": False,
        },
    }
    errors = []

    outcome = cross._validate_outcome_gate(
        "AAPL", validation, counts, [], errors
    )

    assert errors == []
    assert outcome["registered_scientific_result_class"] == (
        "validated_strict_no_go"
    )
    assert outcome["reclassified_base_integrity_verified"] is True

    validation["experiment_integrity_verified"] = False
    errors = []
    outcome = cross._validate_outcome_gate("AAPL", validation, counts, [], errors)
    assert errors == []
    assert outcome["legacy_experiment_integrity_verified"] is False
    assert outcome["reclassified_base_integrity_verified"] is True

    validation["reference_rollout_reproduced"] = False
    errors = []
    outcome = cross._validate_outcome_gate("AAPL", validation, counts, [], errors)
    assert outcome["reclassified_base_integrity_verified"] is False
    assert errors == [
        "AAPL: reclassified implementation integrity failed: "
        "reference_rollout_reproduced is not true"
    ]


def test_method_acceptance_failures_are_science_not_implementation_integrity():
    failures = [
        "train=0 eval=0 legal_lob_endpoint_delivered_acceptance",
        "train=0 eval=0 repeated_anytime_pava_output_acceptance",
    ]
    validation = {
        "experiment_integrity_verified": False,
        "pre_registered_strict_go": False,
        "result_class": "validated_strict_no_go",
        "reason_for_strict_no_go": "legacy reason",
        "reference_rollout_reproduced": True,
        "non_performance_cell_failures": [failures[0]],
        "repeated_anytime_frame_rejections": 3,
        "repeated_anytime_frame_requests": 11_520_000,
        "repeated_trajectories_with_rejection": 3,
        "repeated_trajectories": 5_760,
        "base_pre_registered_validation": {
            "manifest_status": "completed",
            "expected_cells": 9,
            "completed_cells": 9,
            "method_rows": 27,
            "manifest_integrity_errors": [],
            "cell_implementation_gate_failures": failures,
            "rng_pairing_failures": [],
            "denominator_failures": [],
            "source_hash_failures": [],
            "summary_completeness_failures": [],
            "science_checks": {
                "legal_anytime_delivered_acceptance": False,
                "repeated_anytime_delivered_acceptance": False,
            },
            "science_checks_pass": False,
        },
    }
    errors = []

    outcome = cross._validate_outcome_gate(
        "AMZN",
        validation,
        {"matched_repeated_pava": _counts(3)},
        failures,
        errors,
    )

    assert errors == []
    assert outcome["legacy_experiment_integrity_verified"] is False
    assert outcome["reclassified_base_integrity_verified"] is True
    assert outcome["registered_scientific_result_class"] == (
        "validated_strict_no_go"
    )
    assert outcome["method_acceptance_performance_cell_failures"] == failures
    assert outcome["failed_science_checks"] == [
        "legal_anytime_delivered_acceptance",
        "repeated_anytime_delivered_acceptance",
    ]


def test_load_cells_fails_closed_on_noncompleted_and_missing_grid_cell(tmp_path):
    root = tmp_path / "AAPL"
    root.mkdir()
    entries = []
    for train_seed in cross.TRAIN_SEEDS:
        for eval_seed in cross.EVAL_SEEDS:
            path = root / f"train_seed{train_seed}" / f"eval_seed{eval_seed}"
            path.mkdir(parents=True)
            cell_path = path / "consumer_stats.json"
            payload = {
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "status": "completed",
                "methods": {method: {} for method in cross.METHODS},
            }
            cell_path.write_text(json.dumps(payload))
            entries.append(
                {
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "status": "completed",
                    "path": str(cell_path),
                    "sha256": _sha256(cell_path),
                }
            )
    entries[-1]["status"] = "failed"
    entries.pop(0)
    manifest = {"cells": entries}
    errors = []

    loaded = cross._load_cell_payloads("AAPL", root, manifest, errors)

    assert len(loaded) == 7
    assert any("status=failed" in error for error in errors)
    assert any("cell set mismatch" in error for error in errors)
    assert any("manifest has 8 cells" in error for error in errors)
    assert any("orphan consumer files" in error for error in errors)


def test_success_outputs_are_descriptive_ranked_hashed_and_never_overwritten(
    tmp_path, monkeypatch
):
    input_root = tmp_path / "input"
    for asset in cross.ASSETS:
        (input_root / asset).mkdir(parents=True)
    fake = {
        "AAPL": _fake_asset_result("AAPL", True),
        "AMZN": _fake_asset_result("AMZN", False),
        "GOOG": _fake_asset_result("GOOG", True),
    }
    monkeypatch.setattr(
        cross,
        "validate_asset",
        lambda asset, root: fake[asset],
    )
    output_root = tmp_path / "output"

    assert cross.write_outputs(input_root, output_root) == 0
    summary = json.loads((output_root / "cross_asset_summary.json").read_text())
    validation = json.loads((output_root / "cross_asset_validation.json").read_text())
    receipt = json.loads((output_root / "cross_asset_receipt.json").read_text())
    markdown = (output_root / "cross_asset_summary.md").read_text()

    assert summary["result_class"] == "integrity_verified_with_strict_no_go_assets"
    assert summary["strict_no_go_assets"] == ["AMZN"]
    assert validation["contract"]["total_cells"] == 27
    assert validation["contract"]["p_values_computed"] is False
    assert validation["contract"]["assets_are_not_random_replicates"] is True
    assert summary["pooled_exact_counts"]["legal_lob"]["anytime_frame_requests"] == 34_560_000
    assert "Anytime pre frame ↑" in markdown
    assert "Projection calls/path ↓" in markdown
    assert "**100.000 +/- 1.000**" in markdown
    assert "<u>90.000 +/- 1.000</u>" in markdown
    assert "No cross-asset p-values" in markdown
    assert set(receipt["output_file_sha256"]) == {
        "cross_asset_validation.json",
        "cross_asset_summary.json",
        "cross_asset_per_asset_method.csv",
        "cross_asset_exact_counts.csv",
        "cross_asset_pooled_exact_counts.csv",
        "cross_asset_summary.md",
    }
    assert receipt["integrity_reclassification"]["rule_sha256"] == (
        cross.sha256_json(cross.INTEGRITY_RECLASSIFICATION_RULE)
    )
    assert receipt["integrity_reclassification"]["asset_flags"]["AMZN"] == {
        "legacy_experiment_integrity_verified": True,
        "reclassified_experiment_integrity_verified": True,
    }

    before = _sha256(output_root / "cross_asset_summary.json")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cross.write_outputs(input_root, output_root)
    assert _sha256(output_root / "cross_asset_summary.json") == before


def test_failed_asset_integrity_withholds_summary_but_writes_failure_receipt(
    tmp_path, monkeypatch
):
    input_root = tmp_path / "input"
    for asset in cross.ASSETS:
        (input_root / asset).mkdir(parents=True)
    fake = {asset: _fake_asset_result(asset) for asset in cross.ASSETS}
    fake["GOOG"]["passed"] = False
    fake["GOOG"]["errors"] = ["GOOG: missing cell"]
    fake["GOOG"]["outcome"][
        "reclassified_experiment_integrity_verified"
    ] = False
    monkeypatch.setattr(cross, "validate_asset", lambda asset, root: fake[asset])
    output_root = tmp_path / "failed-output"

    assert cross.write_outputs(input_root, output_root) == 1
    validation = json.loads((output_root / "cross_asset_validation.json").read_text())
    receipt = json.loads((output_root / "cross_asset_receipt.json").read_text())

    assert validation["status"] == "failed"
    assert validation["experiment_integrity_verified"] is False
    assert validation["scientific_summary_status"] == "withheld_integrity_failure"
    assert validation["asset_accounting"] == {
        "expected_assets": ["AAPL", "AMZN", "GOOG"],
        "observed_asset_directories": ["AAPL", "AMZN", "GOOG"],
        "integrity_passed_assets": ["AAPL", "AMZN"],
        "integrity_failed_assets": ["GOOG"],
        "expected_asset_count": 3,
        "observed_asset_directory_count": 3,
        "failed_asset_count": 1,
        "expected_cells": 27,
        "observed_completed_cells": 27,
    }
    assert set(validation["validation_only_exact_counts"]) == {
        "AAPL",
        "AMZN",
        "GOOG",
    }
    assert receipt["outputs_withheld"] is True
    assert receipt["publication"] == "validation_only"
    assert receipt["integrity_reclassification"]["asset_flags"]["GOOG"] == {
        "legacy_experiment_integrity_verified": True,
        "reclassified_experiment_integrity_verified": False,
    }
    assert not (output_root / "cross_asset_summary.json").exists()
    assert not (output_root / "cross_asset_summary.md").exists()
    assert sorted(path.name for path in output_root.iterdir()) == [
        "cross_asset_receipt.json",
        "cross_asset_validation.json",
    ]
    assert not list(tmp_path.glob(".failed-output.staging-*"))

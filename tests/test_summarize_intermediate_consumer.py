import csv
import json

from scripts.summarize_intermediate_consumer import (
    METHOD_LABELS,
    hierarchical_aggregate,
    load_manifest_cells,
    sha256_file,
    validate,
)


def test_hierarchical_aggregate_averages_eval_seeds_before_training_sd():
    rows = [
        {"train_seed": 0, "eval_seed": 0, "method_id": "m", "value": 1.0},
        {"train_seed": 0, "eval_seed": 1, "method_id": "m", "value": 3.0},
        {"train_seed": 1, "eval_seed": 0, "method_id": "m", "value": 5.0},
        {"train_seed": 1, "eval_seed": 1, "method_id": "m", "value": 7.0},
    ]

    result = hierarchical_aggregate(rows, group_fields=("method_id",))[0]

    assert result["value_mean"] == 4.0
    assert abs(result["value_std"] - 2.8284271247461903) < 1e-12
    assert result["n_training_seeds"] == 2
    assert result["n_eval_seeds_per_training_seed"] == [2, 2]
    assert result["value_n_training_seeds"] == 2
    assert result["value_n_eval_seeds_per_training_seed"] == [2, 2]
    assert result["value_missing_cells"] == 0


def test_hierarchical_aggregate_does_not_report_zero_sd_for_one_seed():
    result = hierarchical_aggregate(
        [{"train_seed": 0, "eval_seed": 0, "method_id": "m", "value": 1.0}],
        group_fields=("method_id",),
    )[0]

    assert result["value_std"] is None
    assert result["value_n_training_seeds"] == 1


def test_manifest_loader_rejects_orphan_cell(tmp_path):
    root = tmp_path / "results"
    cell_path = root / "train_seed0" / "eval_seed0" / "consumer_stats.json"
    cell_path.parent.mkdir(parents=True)
    payload = {
        "train_seed": 0,
        "eval_seed": 0,
        "status": "completed",
        "methods": {name: {} for name in METHOD_LABELS},
    }
    cell_path.write_text(json.dumps(payload))
    manifest = {
        "config": {"train_seeds": [0], "eval_seeds": [0]},
        "cells": [
            {
                "train_seed": 0,
                "eval_seed": 0,
                "status": "completed",
                "path": str(cell_path),
                "sha256": sha256_file(cell_path),
            }
        ],
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest))

    _, cells, errors = load_manifest_cells(root)
    assert len(cells) == 1
    assert errors == []

    orphan = root / "train_seed9" / "eval_seed9" / "consumer_stats.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text(json.dumps(payload))
    _, _, errors = load_manifest_cells(root)
    assert any("orphan consumer files" in error for error in errors)


def test_science_negative_does_not_invalidate_implementation(tmp_path):
    reference_path = tmp_path / "reference.csv"
    reference_fields = [
        "train_seed",
        "eval_seed",
        "method_id",
        "horizon",
        "path_cumulative_projection_l2",
        "path_projection_calls",
        "spread_relative_abs_error",
        "volume_relative_abs_error",
        "return_std_relative_abs_error",
        "temporal_correlation",
        "path_pre_projection_invalid_rate",
    ]
    with reference_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reference_fields)
        writer.writeheader()
        writer.writerow(
            {
                "train_seed": 0,
                "eval_seed": 0,
                "method_id": "matched_posthoc_pava",
                "horizon": 10,
                "path_cumulative_projection_l2": 0.1,
                "path_projection_calls": 1.0,
                "spread_relative_abs_error": 0.2,
                "volume_relative_abs_error": 0.3,
                "return_std_relative_abs_error": 0.4,
                "temporal_correlation": 0.05,
                "path_pre_projection_invalid_rate": 0.25,
            }
        )
    config = {
        "train_seeds": [0],
        "eval_seeds": [0],
        "horizon": 10,
        "chunk_size": 10,
        "ddim_steps": 2,
        "pred_len": 10,
        "n_histories": 1,
    }
    manifest = {
        "status": "completed",
        "config": config,
        "provenance": {
            "sources_sha256": {},
            "reference_rollout": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
        },
    }
    common = {
        "train_seed": 0,
        "eval_seed": 0,
        "horizon": 10,
        "n_histories": 1,
        "anytime_pre_frame_requests": 20,
        "anytime_pre_block_requests": 2,
        "anytime_delivered_frame_requests": 20,
        "anytime_delivered_block_requests": 2,
        "endpoint_pre_frame_requests": 10,
        "endpoint_pre_block_requests": 1,
        "endpoint_delivered_frame_requests": 10,
        "endpoint_delivered_block_requests": 1,
        "anytime_pre_frame_acceptance": 1.0,
        "anytime_pre_block_acceptance": 1.0,
        "anytime_delivered_frame_acceptance": 1.0,
        "endpoint_pre_frame_acceptance": 0.75,
        "endpoint_delivered_frame_acceptance": 1.0,
        "state_changing_call_rate": 0.5,
        "projection_l2_cumulative": 0.0,
        "market_order_slippage_w1_ticks": 0.1,
        "liquidity_5tick_w1_target_units": 0.2,
        "risk_alarm_rate_abs_error": 0.01,
        "spread_relative_abs_error": 0.2,
        "volume_relative_abs_error": 0.3,
        "return_std_relative_abs_error": 0.4,
        "temporal_correlation": 0.05,
    }
    rows = []
    for method_id, calls in (
        ("legal_lob", 0.0),
        ("matched_posthoc_pava", 1.0),
        ("matched_repeated_pava", 2.0),
    ):
        row = dict(common)
        row.update(
            {
                "method_id": method_id,
                "method": METHOD_LABELS[method_id],
                "projection_calls_per_trajectory": calls,
            }
        )
        if method_id == "matched_posthoc_pava":
            row["projection_l2_cumulative"] = 0.1
        rows.append(row)
    summary = hierarchical_aggregate(rows, ("method_id", "method"))
    cell = {
        "train_seed": 0,
        "eval_seed": 0,
        "post_model_load_rng_sha256": {
            "legal_lob": "same",
            "matched_posthoc_pava": "same",
            "matched_repeated_pava": "same",
        },
        "validation": {"implementation_checks": {"fixture": True}},
    }

    result = validate(
        manifest,
        [(tmp_path / "cell.json", cell)],
        [],
        rows,
        summary,
        [{"horizon": 10}],
        [{"comparison": "matched_repeated_minus_post"}],
    )

    assert result["implementation_verified"] is True
    assert result["science_checks_pass"] is False
    assert result["narrow_scientific_go"] is False

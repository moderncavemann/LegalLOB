import csv
import json
from pathlib import Path

import pytest

from experiments.multi_asset_event_time import summarize_results as p4


def _acceptance(rejections: int, requests: int) -> float:
    return (requests - rejections) / requests


def _method_rows(
    asset: str,
    *,
    legal_pre_rejections: int = 0,
    legal_block_rejections: int = 0,
    matched_block_rejections: int = 1_280,
) -> list[dict]:
    del asset  # Rows deliberately contain no hard-coded P0 asset field.
    rows = []
    settings = {
        "legal_lob": {
            "pre_rejections": legal_pre_rejections,
            "block_rejections": legal_block_rejections,
            "endpoint_rejections": 0,
            "delivered_rejections": 0,
            "calls": 0.0,
            "correction": 0.0,
            "cumulative": 0.0,
            "slippage": 0.8,
            "changing": 0.0,
        },
        "matched_posthoc_pava": {
            "pre_rejections": 12_800,
            "block_rejections": matched_block_rejections,
            "endpoint_rejections": 0,
            "delivered_rejections": 0,
            "calls": 10.0,
            "correction": 0.20,
            "cumulative": 2.0,
            "slippage": 1.2,
            "changing": 0.9,
        },
        "matched_repeated_pava": {
            "pre_rejections": 12_800,
            "block_rejections": matched_block_rejections,
            "endpoint_rejections": 0,
            "delivered_rejections": 0,
            "calls": 200.0,
            "correction": 0.05,
            "cumulative": 10.0,
            "slippage": 1.0,
            "changing": 0.4,
        },
    }
    for train_seed in p4.TRAIN_SEEDS:
        for eval_seed in p4.EVAL_SEEDS:
            for method_id in p4.METHODS:
                values = settings[method_id]
                # Training-seed movement exercises hierarchical SD without
                # changing any exact count or registered mechanism check.
                offset = train_seed * 0.01
                rows.append(
                    {
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "method_id": method_id,
                        "method": p4.METHOD_LABELS[method_id],
                        "anytime_pre_frame_requests": 1_280_000,
                        "anytime_pre_frame_acceptance": _acceptance(
                            values["pre_rejections"], 1_280_000
                        ),
                        "anytime_pre_block_requests": 128_000,
                        "anytime_pre_block_acceptance": _acceptance(
                            values["block_rejections"], 128_000
                        ),
                        "endpoint_delivered_frame_requests": 64_000,
                        "endpoint_delivered_frame_acceptance": _acceptance(
                            values["endpoint_rejections"], 64_000
                        ),
                        "anytime_delivered_frame_requests": 1_280_000,
                        "anytime_delivered_frame_acceptance": _acceptance(
                            values["delivered_rejections"], 1_280_000
                        ),
                        "projection_calls_per_trajectory": values["calls"],
                        "projection_l2_per_call": values["correction"] + offset,
                        "projection_l2_cumulative": values["cumulative"] + offset,
                        "market_order_slippage_w1_ticks": values["slippage"] + offset,
                        "state_changing_call_rate": values["changing"],
                    }
                )
    return rows


def _asset_result(asset: str, **row_kwargs) -> dict:
    result = p4.build_asset_summary(
        asset,
        _method_rows(asset, **row_kwargs),
        input_root=Path("/tmp") / asset,
    )
    result["integrity_verified"] = True
    result["integrity_errors"] = []
    result["reference_parity"] = {
        "verified": True,
        "endpoint_max_abs_difference": 0.0,
        "policy": p4.REFERENCE_PARITY_POLICY,
    }
    result["scientific_outcome"]["classification"] = (
        "verified_go"
        if result["scientific_outcome"]["mechanism_go"]
        else "verified_scientific_no_go"
    )
    return result


def test_exact_rejections_rejects_materially_rounded_rates():
    assert p4._exact_rejections(
        {"acceptance": 0.75, "requests": 4}, "acceptance", "requests"
    ) == (1, 4)

    with pytest.raises(ValueError, match="does not encode an exact count"):
        p4._exact_rejections(
            {"acceptance": 0.333, "requests": 100}, "acceptance", "requests"
        )


def test_asset_summary_has_exact_denominators_na_and_registered_go():
    result = _asset_result("AAPL")
    rows = {row["method_id"]: row for row in result["method_rows"]}

    assert rows["legal_lob"]["pre_projection_invalidity"] == {
        "invalid_or_rejected": 0,
        "requests": 11_520_000,
        "rate": 0.0,
        "by_training_seed": {
            str(seed): {
                "invalid_or_rejected": 0,
                "requests": 3_840_000,
                "rate": 0.0,
            }
            for seed in p4.TRAIN_SEEDS
        },
    }
    assert (
        rows["matched_posthoc_pava"]["intermediate_consumer_rejection"][
            "invalid_or_rejected"
        ]
        == 11_520
    )
    assert rows["legal_lob"]["exact_projection_calls"] == 0
    assert rows["matched_posthoc_pava"]["exact_projection_calls"] == 57_600
    assert rows["matched_repeated_pava"]["exact_projection_calls"] == 1_152_000
    assert rows["legal_lob"]["correction_per_call"] is None
    assert result["scientific_outcome"]["mechanism_go"] is True
    assert result["scientific_outcome"]["fidelity_warning"] is False


def test_dense_ranking_honours_direction_ties_and_na():
    result = _asset_result("AAPL")
    rows = result["method_rows"]

    calls = p4.dense_ranks(rows, p4.METRIC_BY_KEY["projection_calls_per_path"])
    correction = p4.dense_ranks(rows, p4.METRIC_BY_KEY["correction_per_call"])
    pre = p4.dense_ranks(rows, p4.METRIC_BY_KEY["pre_projection_invalidity"])

    assert calls == {
        "legal_lob": 1,
        "matched_posthoc_pava": 2,
        "matched_repeated_pava": 3,
    }
    assert correction == {
        "legal_lob": None,
        "matched_posthoc_pava": 2,
        "matched_repeated_pava": 1,
    }
    assert pre == {
        "legal_lob": 1,
        "matched_posthoc_pava": 2,
        "matched_repeated_pava": 2,
    }


def test_latex_has_directions_best_second_and_zero_call_na():
    summary = p4.build_campaign_summary(
        [_asset_result(asset) for asset in p4.P4_ASSETS]
    )
    latex = p4.render_latex(summary)
    markdown = p4.render_markdown(summary)

    assert "Panel C: Multi-asset event-time replication" in latex
    assert "Correction/call $\\downarrow$" in latex
    assert "\\textbf{0/11{,}520{,}000}" in latex
    assert "\\underline{115{,}200/11{,}520{,}000}" in latex
    assert "\\textbf{N/A}" not in latex
    assert " & N/A & " in latex
    assert "**0 / 11,520,000 (0.00000%)**" in markdown
    assert "<u>115,200 / 11,520,000" in markdown
    assert summary["validation"]["fidelity_warning_assets"] == []
    assert (
        summary["validation"][
            "absence_of_fidelity_degradation_language_allowed"
        ]
        is False
    )
    assert "descriptive guardrail only" in summary["validation"][
        "fidelity_language_policy"
    ]


def test_verified_scientific_no_go_is_not_integrity_failure():
    results = [_asset_result(asset) for asset in p4.P4_ASSETS]
    failed = _asset_result("INTC", matched_block_rejections=0)
    results[-1] = failed

    summary = p4.build_campaign_summary(results)

    assert summary["validation"]["implementation_integrity_verified"] is True
    assert summary["validation"]["four_asset_mechanism_go"] is False
    assert summary["validation"]["campaign_result_class"] == (
        "verified_scientific_no_go"
    )
    assert summary["validation"]["asset_classifications"]["INTC"] == {
        "integrity_verified": True,
        "integrity_errors": [],
        "scientific_classification": "verified_scientific_no_go",
        "mechanism_go": False,
        "fidelity_warning": False,
        "reference_parity_verified": True,
        "reference_endpoint_max_abs_difference": 0.0,
    }


def test_reference_reproduction_tolerance_is_endpoint_only_and_outcome_blind():
    base = {
        "passed": False,
        "comparisons": 9,
        "tolerance": {"relative": 1e-5, "absolute": 5e-7},
        "max_abs_difference": {"endpoint_pre_invalid_rate": 4e-5},
        "errors": ["(0, 0) endpoint_pre_invalid_rate: current=0.0 reference=0.00004"],
    }

    classified = p4.classify_reference_reproduction(base)
    assert classified["verified"] is True
    assert classified["base_passed"] is False
    assert classified["endpoint_max_abs_difference"] == 4e-5
    assert classified["non_endpoint_errors"] == []

    base["max_abs_difference"]["endpoint_pre_invalid_rate"] = 5.1e-5
    assert p4.classify_reference_reproduction(base)["verified"] is False

    base["max_abs_difference"]["endpoint_pre_invalid_rate"] = 4e-5
    base["errors"].append("(0, 0) projection_l2_cumulative: current=1.0 reference=1.1")
    classified = p4.classify_reference_reproduction(base)
    assert classified["verified"] is False
    assert classified["non_endpoint_errors"] == [
        "(0, 0) projection_l2_cumulative: current=1.0 reference=1.1"
    ]

    base["errors"] = []
    base["passed"] = False
    classified = p4.classify_reference_reproduction(base)
    assert classified["verified"] is False
    assert classified["base_pass_flag_consistent_with_errors"] is False


def test_validate_asset_uses_endpoint_only_reference_classification(
    tmp_path, monkeypatch
):
    root = tmp_path / "consumer"
    root.mkdir()
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text("{}")
    cell_path = root / "consumer_stats.json"
    cell_path.write_text("{}")
    method_rows = _method_rows("AAPL")
    base = {
        "manifest_status": "completed",
        "completed_cells": 9,
        "method_rows": 27,
        "manifest_integrity_errors": [],
        "rng_pairing_failures": [],
        "denominator_failures": [],
        "source_hash_failures": [],
        "summary_completeness_failures": [],
        "cell_implementation_gate_failures": [],
        "reference_reproduction": {
            "passed": False,
            "comparisons": 9,
            "tolerance": {"relative": 1e-5, "absolute": 5e-7},
            "max_abs_difference": {"endpoint_pre_invalid_rate": 4e-5},
            "errors": ["(0, 0) endpoint_pre_invalid_rate: current=0 reference=0.00004"],
        },
    }
    monkeypatch.setattr(
        p4,
        "load_manifest_cells",
        lambda consumer_root: ({}, [(cell_path, {})], []),
    )
    monkeypatch.setattr(p4, "_validate_config", lambda asset, manifest: [])
    monkeypatch.setattr(p4, "_validate_final_checkpoints", lambda asset, cells: [])
    monkeypatch.setattr(
        p4, "_validate_pairing_receipts", lambda asset, manifest, cells: []
    )
    monkeypatch.setattr(p4, "collect_rows", lambda cells: (method_rows, [], []))
    monkeypatch.setattr(p4, "hierarchical_aggregate", lambda *args, **kwargs: [])
    monkeypatch.setattr(p4, "validate", lambda *args, **kwargs: base)

    result = p4.validate_asset("AAPL", root)
    assert result["integrity_verified"] is True
    assert result["reference_parity"]["verified"] is True
    assert result["reference_parity"]["base_passed"] is False

    base["reference_reproduction"]["max_abs_difference"][
        "endpoint_pre_invalid_rate"
    ] = 5.1e-5
    result = p4.validate_asset("AAPL", root)
    assert result["integrity_verified"] is False
    assert result["reference_parity"]["verified"] is False

    base["reference_reproduction"]["max_abs_difference"][
        "endpoint_pre_invalid_rate"
    ] = 4e-5
    base["reference_reproduction"]["errors"].append(
        "(0, 0) temporal_correlation: current=0 reference=0.1"
    )
    result = p4.validate_asset("AAPL", root)
    assert result["integrity_verified"] is False
    assert result["reference_parity"]["non_endpoint_errors"]


def test_metric_clarification_is_bound_by_frozen_hash():
    binding = p4.validate_metric_clarification()

    assert binding["verified"] is True
    assert binding["actual_sha256"] == p4.METRIC_CLARIFICATION_SHA256


def test_pairing_receipts_require_same_histories_futures_and_noise_by_eval():
    starts = "a" * 64
    futures = "b" * 64
    manifest = {"provenance": {"data": {"start_indices_sha256": starts}}}
    cells = []
    for train_seed in p4.TRAIN_SEEDS:
        for eval_seed in p4.EVAL_SEEDS:
            payload = {
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "history_starts_sha256": starts,
                "real_futures_sha256": futures,
                "real_futures_shape": [640, 2, 20, 100],
                "real_futures_dtype": "float32",
                "post_model_load_rng_sha256": {
                    method: "c" * 64 for method in p4.METHODS
                },
                "paired_initial_noise_sha256": [
                    ("d", "e", "f")[eval_seed] * 64 for _ in range(50)
                ],
            }
            cells.append((Path(f"cell-{train_seed}-{eval_seed}.json"), payload))

    assert p4._validate_pairing_receipts("AAPL", manifest, cells) == []

    cells[-1][1]["real_futures_sha256"] = "0" * 64
    cells[-1][1]["paired_initial_noise_sha256"] = ["1" * 64] * 50
    errors = p4._validate_pairing_receipts("AAPL", manifest, cells)
    assert any("real_futures_sha256 differs" in error for error in errors)
    assert any("initial noise differs across checkpoints" in error for error in errors)


def test_evaluation_binding_requires_frozen_plan_sources_and_dependencies(tmp_path):
    root = tmp_path / "evaluation"
    orchestrator = root / "_orchestrator"
    orchestrator.mkdir(parents=True)
    sources = {
        "experiments/multi_asset_event_time/metric_definition.md": {
            "sha256": p4.METRIC_CLARIFICATION_SHA256,
        },
        "experiments/multi_asset_event_time/summarize_results.py": {
            "sha256": p4.sha256_file(Path(p4.__file__).resolve()),
        },
    }
    plan = {
        "campaign_id": "p4_public_itch_event_time_20260807_v1_evaluation",
        "protocol": {
            "assets": list(p4.P4_ASSETS),
            "datasets": [p4.P4_DATASETS[asset] for asset in p4.P4_ASSETS],
            "train_seeds": list(p4.TRAIN_SEEDS),
            "eval_seeds": list(p4.EVAL_SEEDS),
            "methods": list(p4.METHODS),
            "checkpoint_name": "final.pt",
            "n_histories_per_cell": 640,
            "horizon": 100,
            "ddim_steps": 20,
            "reference_reproduction": p4.REFERENCE_PARITY_POLICY,
        },
        "provenance": {"sources": sources},
    }
    plan_path = orchestrator / "p4_evaluation_plan.json"
    plan_path.write_text(json.dumps(plan))
    dependencies = [f"consumer-{index}" for index in range(4)]
    stages = [
        {
            "stage_id": stage_id,
            "kind": "consumer",
            "status": "completed",
        }
        for stage_id in dependencies
    ]
    stages.extend(
        {
            "stage_id": f"other-{index}",
            "kind": "reference_eval",
            "status": "completed",
        }
        for index in range(p4.EXPECTED_EVALUATION_STAGES - len(stages) - 1)
    )
    stages.append(
        {
            "stage_id": "panel",
            "kind": "panel_c",
            "status": "running",
            "dependencies": dependencies,
        }
    )
    ledger_path = orchestrator / "p4_evaluation_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "campaign_id": plan["campaign_id"],
                "plan_sha256": p4.sha256_file(plan_path),
                "revision": 21,
                "status": "in_progress",
                "stages": stages,
            }
        )
    )

    assert p4.validate_evaluation_binding(root)["verified"] is True

    stages[0]["status"] = "failed"
    ledger = json.loads(ledger_path.read_text())
    ledger["stages"] = stages
    ledger_path.write_text(json.dumps(ledger))
    binding = p4.validate_evaluation_binding(root)
    assert binding["verified"] is False
    assert any("dependencies are not completed" in error for error in binding["errors"])


def test_write_outputs_is_atomic_complete_and_never_overwrites(tmp_path, monkeypatch):
    fake = {asset: _asset_result(asset) for asset in p4.P4_ASSETS}
    monkeypatch.setattr(
        p4,
        "validate_asset",
        lambda asset, root: fake[asset],
    )
    monkeypatch.setattr(
        p4,
        "validate_evaluation_binding",
        lambda root: {
            "verified": True,
            "errors": [],
            "plan_path": "/frozen/plan.json",
            "plan_sha256": "a" * 64,
            "ledger_observation": {"revision": 28},
        },
    )
    output = tmp_path / "panel_c"

    assert p4.write_outputs(tmp_path / "evaluation", output) == 0
    assert {path.name for path in output.iterdir()} == {
        "p4_event_robustness_panel_c.md",
        "p4_event_robustness_panel_c.tex",
        "p4_event_robustness_receipt.json",
        "p4_event_robustness_summary.csv",
        "p4_event_robustness_summary.json",
        "p4_event_robustness_validation.json",
    }
    summary = json.loads((output / "p4_event_robustness_summary.json").read_text())
    receipt = json.loads((output / "p4_event_robustness_receipt.json").read_text())
    with (output / "p4_event_robustness_summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(summary["panel_rows"]) == 12
    assert len(csv_rows) == 12
    assert receipt["metric_clarification_binding"]["verified"] is True
    assert receipt["reference_parity_policy"] == p4.REFERENCE_PARITY_POLICY
    assert receipt["reference_parity_policy_sha256"] == p4._sha256_json(
        p4.REFERENCE_PARITY_POLICY
    )
    assert set(receipt["reference_parity_by_asset"]) == set(p4.P4_ASSETS)
    assert all(
        item["verified"] is True
        for item in receipt["reference_parity_by_asset"].values()
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        p4.write_outputs(tmp_path / "evaluation", output)


def test_integrity_failure_writes_receipt_and_validation_but_no_panel(
    tmp_path, monkeypatch
):
    fake = {asset: _asset_result(asset) for asset in p4.P4_ASSETS}
    fake["GOOG"]["integrity_verified"] = False
    fake["GOOG"]["integrity_errors"] = ["declared hash mismatch"]
    monkeypatch.setattr(
        p4,
        "validate_asset",
        lambda asset, root: fake[asset],
    )
    monkeypatch.setattr(
        p4,
        "validate_evaluation_binding",
        lambda root: {
            "verified": True,
            "errors": [],
            "plan_path": "/frozen/plan.json",
            "plan_sha256": "a" * 64,
            "ledger_observation": {"revision": 28},
        },
    )
    output = tmp_path / "failed_panel"

    assert p4.write_outputs(tmp_path / "evaluation", output) == 1
    assert {path.name for path in output.iterdir()} == {
        "p4_event_robustness_receipt.json",
        "p4_event_robustness_validation.json",
    }
    validation = json.loads(
        (output / "p4_event_robustness_validation.json").read_text()
    )
    assert validation["campaign_result_class"] == "terminal_with_declared_failures"

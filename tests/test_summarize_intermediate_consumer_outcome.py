from scripts.summarize_intermediate_consumer_outcome import (
    outcome_validation,
    verdict_lines,
)


def test_method_performance_failure_is_not_data_integrity_failure():
    base = {
        "manifest_status": "completed",
        "completed_cells": 9,
        "expected_cells": 9,
        "manifest_integrity_errors": [],
        "cell_implementation_gate_failures": [
            "train=0 eval=0 repeated_anytime_pava_output_acceptance"
        ],
        "rng_pairing_failures": [],
        "denominator_failures": [],
        "source_hash_failures": [],
        "summary_completeness_failures": [],
        "reference_reproduction": {
            "errors": [
                "(0, 0) endpoint_pre_invalid_rate: current=0.5 reference=0.49998"
            ],
            "max_abs_difference": {"endpoint_pre_invalid_rate": 2e-5},
        },
        "science_checks_pass": False,
    }
    counts = {
        "matched_repeated_pava": {
            "anytime_frame_rejections": 1,
            "anytime_frame_requests": 100,
            "trajectories_with_any_anytime_rejection": 1,
            "trajectories": 10,
        }
    }

    result = outcome_validation(base, counts)

    assert result["experiment_integrity_verified"] is True
    assert result["pre_registered_strict_go"] is False
    assert result["result_class"] == "validated_strict_no_go"


def test_nonperformance_integrity_failure_remains_fatal():
    base = {
        "manifest_status": "completed",
        "completed_cells": 9,
        "expected_cells": 9,
        "manifest_integrity_errors": ["hash mismatch"],
        "cell_implementation_gate_failures": [],
        "rng_pairing_failures": [],
        "denominator_failures": [],
        "source_hash_failures": [],
        "summary_completeness_failures": [],
        "reference_reproduction": {
            "errors": [],
            "max_abs_difference": {"endpoint_pre_invalid_rate": 0.0},
        },
        "science_checks_pass": True,
    }
    counts = {
        "matched_repeated_pava": {
            "anytime_frame_rejections": 0,
            "anytime_frame_requests": 100,
            "trajectories_with_any_anytime_rejection": 0,
            "trajectories": 10,
        }
    }

    result = outcome_validation(base, counts)

    assert result["experiment_integrity_verified"] is False


def test_no_go_reason_uses_current_failed_checks_not_prior_asset_narrative():
    base = {
        "manifest_status": "completed",
        "completed_cells": 9,
        "expected_cells": 9,
        "manifest_integrity_errors": [],
        "cell_implementation_gate_failures": [],
        "rng_pairing_failures": [],
        "denominator_failures": [],
        "source_hash_failures": [],
        "summary_completeness_failures": [],
        "reference_reproduction": {
            "errors": [],
            "max_abs_difference": {"endpoint_pre_invalid_rate": 0.0},
        },
        "science_checks": {
            "legal_anytime_delivered_acceptance": False,
            "repeated_anytime_delivered_acceptance": True,
        },
        "science_checks_pass": False,
    }
    counts = {
        "matched_repeated_pava": {
            "anytime_frame_rejections": 0,
            "anytime_frame_requests": 100,
            "trajectories_with_any_anytime_rejection": 0,
            "trajectories": 10,
        }
    }

    result = outcome_validation(base, counts)

    assert result["failed_science_checks"] == [
        "legal_anytime_delivered_acceptance"
    ]
    assert result["reason_for_strict_no_go"] == (
        "Failed pre-registered science checks: "
        "legal_anytime_delivered_acceptance"
    )
    assert "Repeated PAVA left rare" not in result["reason_for_strict_no_go"]


def test_verdict_is_derived_from_current_counts_and_checks():
    rows = {
        method: {"projection_calls_per_trajectory_mean": calls}
        for method, calls in {
            "legal_lob": 0.0,
            "matched_posthoc_pava": 10.0,
            "matched_repeated_pava": 200.0,
        }.items()
    }
    counts = {
        method: {
            "anytime_frame_rejections": rejected,
            "anytime_frame_requests": 1000,
            "anytime_block_rejections": rejected,
            "anytime_block_requests": 100,
            "trajectories_with_any_anytime_rejection": int(rejected > 0),
            "trajectories": 10,
            "endpoint_frame_rejections": 0,
        }
        for method, rejected in {
            "legal_lob": 2,
            "matched_posthoc_pava": 900,
            "matched_repeated_pava": 0,
        }.items()
    }
    validation = {
        "pre_registered_strict_go": False,
        "failed_science_checks": ["legal_anytime_delivered_acceptance"],
    }

    text = "\n".join(verdict_lines(rows, counts, validation))

    assert "LegalLOB recorded 2/1,000 anytime frame rejections" in text
    assert "Repeated PAVA left rare" not in text
    assert "failed checks: legal_anytime_delivered_acceptance" in text

#!/usr/bin/env python3
"""Summarize a completed consumer panel, including strict-GO failures.

Experiment integrity is kept separate from the pre-registered scientific GO
criterion. The verdict is derived from the current panel's checks and exact
counts; it must not inherit an asset-specific narrative from an earlier run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarize_intermediate_consumer import (
    METHOD_LABELS,
    _fmt,
    _fmt_percent,
    _write_csv,
    collect_rows,
    hierarchical_aggregate,
    load_manifest_cells,
    sha256_file,
    validate,
)


REPEATED_PAVA_PERFORMANCE_CHECK = "repeated_anytime_pava_output_acceptance"


def exact_outcome_counts(
    cells: list[tuple[Path, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for method_id in METHOD_LABELS:
        totals = {
            "anytime_frame_requests": 0,
            "anytime_frame_rejections": 0,
            "anytime_block_requests": 0,
            "anytime_block_rejections": 0,
            "endpoint_frame_requests": 0,
            "endpoint_frame_rejections": 0,
            "trajectories": 0,
            "trajectories_with_any_anytime_rejection": 0,
            "rejection_reason_counts": {
                "nonfinite": 0,
                "spread": 0,
                "ask_chain": 0,
                "bid_chain": 0,
                "negative_depth": 0,
            },
        }
        for _, cell in cells:
            method = cell["methods"][method_id]
            anytime = method["anytime_consumer"]
            delivered = anytime["delivered"]
            endpoint = method["endpoint_consumer"]["delivered"]
            totals["anytime_frame_requests"] += int(delivered["frame_requests"])
            totals["anytime_frame_rejections"] += int(
                delivered["frame_requests"]
                - round(
                    delivered["frame_acceptance_rate"]
                    * delivered["frame_requests"]
                )
            )
            totals["anytime_block_requests"] += int(delivered["block_requests"])
            totals["anytime_block_rejections"] += int(
                delivered["block_requests"]
                - round(
                    delivered["block_acceptance_rate"]
                    * delivered["block_requests"]
                )
            )
            totals["endpoint_frame_requests"] += int(endpoint["frame_requests"])
            totals["endpoint_frame_rejections"] += int(
                endpoint["frame_requests"]
                - round(
                    endpoint["frame_acceptance_rate"]
                    * endpoint["frame_requests"]
                )
            )
            trajectories = int(anytime["trajectories"])
            totals["trajectories"] += trajectories
            totals["trajectories_with_any_anytime_rejection"] += int(
                trajectories
                - round(
                    anytime["delivered_completed_without_rejection_rate"]
                    * trajectories
                )
            )
            for reason, count in delivered["rejection_reason_counts"].items():
                totals["rejection_reason_counts"][reason] += int(count)
        totals["anytime_frame_acceptance_rate"] = 1.0 - (
            totals["anytime_frame_rejections"]
            / totals["anytime_frame_requests"]
        )
        totals["anytime_block_acceptance_rate"] = 1.0 - (
            totals["anytime_block_rejections"]
            / totals["anytime_block_requests"]
        )
        totals["completed_without_anytime_rejection_rate"] = 1.0 - (
            totals["trajectories_with_any_anytime_rejection"]
            / totals["trajectories"]
        )
        output[method_id] = totals
    return output


def outcome_validation(
    base: dict[str, Any], counts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    non_performance_cell_failures = [
        failure
        for failure in base["cell_implementation_gate_failures"]
        if REPEATED_PAVA_PERFORMANCE_CHECK not in failure
    ]
    reference = base["reference_reproduction"]
    non_acceptance_reference_errors = [
        error
        for error in reference["errors"]
        if "endpoint_pre_invalid_rate" not in error
    ]
    # Torch float32 and NumPy float64 comparison thresholds can differ by one
    # or two frames out of 64,000 while the generated paths and all rollout
    # metrics are byte-for-byte/numerically identical.  This declared
    # diagnostic tolerance is 3.2 frames per cell and is not applied to any
    # scientific acceptance result.
    endpoint_diagnostic_tolerance = 5e-5
    reference_rollout_reproduced = (
        not non_acceptance_reference_errors
        and reference["max_abs_difference"]["endpoint_pre_invalid_rate"]
        <= endpoint_diagnostic_tolerance
    )
    integrity_verified = (
        base["manifest_status"] == "completed"
        and base["completed_cells"] == base["expected_cells"]
        and not base["manifest_integrity_errors"]
        and not non_performance_cell_failures
        and not base["rng_pairing_failures"]
        and not base["denominator_failures"]
        and not base["source_hash_failures"]
        and not base["summary_completeness_failures"]
        and reference_rollout_reproduced
    )
    strict_go = bool(base["science_checks_pass"])
    failed_science_checks = sorted(
        name
        for name, passed in (base.get("science_checks") or {}).items()
        if not passed
    )
    repeated = counts["matched_repeated_pava"]
    return {
        "experiment_integrity_verified": integrity_verified,
        "pre_registered_strict_go": strict_go,
        "result_class": (
            "strict_go" if strict_go else "validated_strict_no_go"
        ),
        "reason_for_strict_no_go": (
            None
            if strict_go
            else (
                "Failed pre-registered science checks: "
                + ", ".join(failed_science_checks)
                if failed_science_checks
                else "One or more pre-registered science checks failed."
            )
        ),
        "failed_science_checks": failed_science_checks,
        "reference_rollout_reproduced": reference_rollout_reproduced,
        "reference_endpoint_diagnostic_tolerance": endpoint_diagnostic_tolerance,
        "non_acceptance_reference_errors": non_acceptance_reference_errors,
        "non_performance_cell_failures": non_performance_cell_failures,
        "repeated_anytime_frame_rejections": repeated[
            "anytime_frame_rejections"
        ],
        "repeated_anytime_frame_requests": repeated["anytime_frame_requests"],
        "repeated_trajectories_with_rejection": repeated[
            "trajectories_with_any_anytime_rejection"
        ],
        "repeated_trajectories": repeated["trajectories"],
        "base_pre_registered_validation": base,
    }


def verdict_lines(
    rows: dict[str, dict[str, Any]],
    counts: dict[str, dict[str, Any]],
    validation: dict[str, Any],
) -> list[str]:
    """Render a panel-specific verdict from validated counts and checks."""

    if validation["pre_registered_strict_go"]:
        go_sentence = "The panel passed every pre-registered strict-GO check."
    else:
        failed = validation.get("failed_science_checks") or []
        details = ", ".join(failed) if failed else "unspecified science checks"
        go_sentence = f"The panel did not meet strict GO; failed checks: {details}."

    method_sentences: list[str] = []
    for method_id, label in METHOD_LABELS.items():
        item = counts[method_id]
        method_sentences.append(
            f"{label} recorded {item['anytime_frame_rejections']:,}/"
            f"{item['anytime_frame_requests']:,} anytime frame rejections, "
            f"{item['anytime_block_rejections']:,}/"
            f"{item['anytime_block_requests']:,} block rejections, "
            f"{item['trajectories_with_any_anytime_rejection']:,}/"
            f"{item['trajectories']:,} trajectories with any rejection, and "
            f"{item['endpoint_frame_rejections']:,} endpoint frame rejections."
        )

    call_parts = []
    for method_id, label in METHOD_LABELS.items():
        calls = rows[method_id].get("projection_calls_per_trajectory_mean")
        call_parts.append(
            f"{label}={calls:.1f}" if calls is not None else f"{label}=--"
        )

    return [
        go_sentence,
        *method_sentences,
        "Mean projection calls per trajectory: " + ", ".join(call_parts) + ".",
        "Endpoint risk and fidelity quantities are descriptive guardrails. This",
        "experiment does not establish superior risk accuracy, trading performance,",
        "or causal control.",
    ]


def render_markdown(
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    delta: list[dict[str, Any]],
    counts: dict[str, dict[str, Any]],
    validation: dict[str, Any],
) -> str:
    rows = {row["method_id"]: row for row in summary}
    lines = [
        "# Intermediate Consumer Outcome",
        "",
        f"Experiment integrity verified: **{validation['experiment_integrity_verified']}**.  "
        f"Pre-registered strict GO: **{validation['pre_registered_strict_go']}**.",
        "",
        "The audit preserves every predicted-clean request. These requests are",
        "correlated points on a reverse path, not independent market samples.",
        "Risk outputs are computed only after the feasibility gate; generation",
        "continues in audit mode so rejected requests remain in the denominator.",
        "",
        "| Method | Anytime pre frame % | Anytime pre block % | Delivered anytime frame % | No-rejection trajectories % | Endpoint frame % | PAVA calls/path | Cumulative L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in METHOD_LABELS:
        row = rows[method_id]
        lines.append(
            f"| {row['method']} | "
            f"{_fmt_percent(row, 'anytime_pre_frame_acceptance', 3)} | "
            f"{_fmt_percent(row, 'anytime_pre_block_acceptance', 3)} | "
            f"{_fmt_percent(row, 'anytime_delivered_frame_acceptance', 5)} | "
            f"{_fmt_percent(row, 'anytime_delivered_completed_trajectory_rate', 3)} | "
            f"{_fmt_percent(row, 'endpoint_delivered_frame_acceptance', 3)} | "
            f"{_fmt(row, 'projection_calls_per_trajectory', 1)} | "
            f"{_fmt(row, 'projection_l2_cumulative', 3)} |"
        )
    lines.extend(
        [
            "",
            "Exact pooled counts:",
            "",
            "| Method | Anytime frame rejects / requests | Block rejects / requests | Trajectories with any reject / total | Endpoint frame rejects |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method_id, label in METHOD_LABELS.items():
        item = counts[method_id]
        lines.append(
            f"| {label} | {item['anytime_frame_rejections']} / "
            f"{item['anytime_frame_requests']} | "
            f"{item['anytime_block_rejections']} / "
            f"{item['anytime_block_requests']} | "
            f"{item['trajectories_with_any_anytime_rejection']} / "
            f"{item['trajectories']} | {item['endpoint_frame_rejections']} |"
        )
    lines.extend(
        [
            "",
            "## Endpoint risk/fidelity guardrails",
            "",
            "| Method | Slippage W1 | Liq@5 W1 | Alarm error % | Spread rel. error % | Volume rel. error % | Return-vol. error % | Temporal corr. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method_id in METHOD_LABELS:
        row = rows[method_id]
        lines.append(
            f"| {row['method']} | {_fmt(row, 'market_order_slippage_w1_ticks', 3)} | "
            f"{_fmt(row, 'liquidity_5tick_w1_target_units', 3)} | "
            f"{_fmt_percent(row, 'risk_alarm_rate_abs_error', 2)} | "
            f"{_fmt_percent(row, 'spread_relative_abs_error', 2)} | "
            f"{_fmt_percent(row, 'volume_relative_abs_error', 2)} | "
            f"{_fmt_percent(row, 'return_std_relative_abs_error', 2)} | "
            f"{_fmt(row, 'temporal_correlation', 3)} |"
        )
    lines.extend(
        [
            "",
            "## Same-checkpoint Matched path divergence",
            "",
            "| Horizon | Trajectory L2 | Risk-alarm disagreement % |",
            "|---:|---:|---:|",
        ]
    )
    for row in sorted(paired, key=lambda item: item["horizon"]):
        lines.append(
            f"| {row['horizon']} | {_fmt(row, 'trajectory_l2_mean', 3)} | "
            f"{_fmt_percent(row, 'risk_alarm_disagreement_rate', 2)} |"
        )
    signed = delta[0]
    lines.extend(
        [
            "",
            "Matched+Repeated minus Matched+Post at horizon 100: "
            f"calls/path {_fmt(signed, 'projection_calls_per_trajectory_delta', 1)}, "
            f"cumulative L2 {_fmt(signed, 'projection_l2_cumulative_delta', 3)}, "
            f"slippage W1 {_fmt(signed, 'market_order_slippage_w1_ticks_delta', 3)}, "
            f"Liq@5 W1 {_fmt(signed, 'liquidity_5tick_w1_target_units_delta', 3)}, "
            f"and alarm error {_fmt_percent(signed, 'risk_alarm_rate_abs_error_delta', 2)}.",
            "",
            "## Verdict",
            "",
            *verdict_lines(rows, counts, validation),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    manifest, cells, manifest_errors = load_manifest_cells(args.results_root)
    rows, paired_rows, delta_rows = collect_rows(cells)
    summary = hierarchical_aggregate(rows, ("method_id", "method"))
    paired = hierarchical_aggregate(paired_rows, ("horizon",))
    delta = hierarchical_aggregate(delta_rows, ("comparison",))
    base = validate(
        manifest,
        cells,
        manifest_errors,
        rows,
        summary,
        paired,
        delta,
    )
    counts = exact_outcome_counts(cells)
    validation = outcome_validation(base, counts)
    root = args.results_root
    _write_csv(root / "intermediate_consumer_outcome_per_run.csv", rows)
    _write_csv(root / "intermediate_consumer_outcome_summary.csv", summary)
    _write_csv(root / "intermediate_consumer_outcome_paired_summary.csv", paired)
    _write_csv(root / "intermediate_consumer_outcome_matched_delta.csv", delta)
    (root / "intermediate_consumer_outcome_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (root / "intermediate_consumer_outcome_paired_summary.json").write_text(
        json.dumps(paired, indent=2) + "\n"
    )
    (root / "intermediate_consumer_outcome_matched_delta.json").write_text(
        json.dumps(delta, indent=2) + "\n"
    )
    (root / "intermediate_consumer_outcome_counts.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    (root / "intermediate_consumer_outcome_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    (root / "intermediate_consumer_outcome.md").write_text(
        render_markdown(summary, paired, delta, counts, validation)
    )
    receipt = {
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "base_summarizer_sha256": sha256_file(
            Path("scripts/summarize_intermediate_consumer.py")
        ),
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "validation_sha256": sha256_file(
            root / "intermediate_consumer_outcome_validation.json"
        ),
        "summary_sha256": sha256_file(
            root / "intermediate_consumer_outcome_summary.json"
        ),
    }
    (root / "intermediate_consumer_outcome_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print((root / "intermediate_consumer_outcome.md").read_text())
    print(json.dumps(validation, indent=2))
    return 0 if validation["experiment_integrity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare frozen softplus LegalLOB and boundary-aware LegalLOB on AMZN.

The go/no-go rule is intentionally narrow: boundary calibration must improve by
at least 25%, while overall MSE, price MSE, and conditional interior-gap W1 may
degrade by at most 5%. Structural invalidity must remain zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.eval.metrics import wasserstein_1d
from scripts.audit_saved_generation import continuous_boundary_stats
from scripts.summarize_financial_sensitivity import admissible_snapshot_mask


def _gap_values(lob: np.ndarray) -> np.ndarray:
    prices = np.asarray(lob[:, 0], dtype=np.float64)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    return np.concatenate(
        [(ask[:, 1:] - ask[:, :-1]).reshape(-1), (bid[:, :-1] - bid[:, 1:]).reshape(-1)]
    )


def summarize_method(
    generated: np.ndarray,
    real: np.ndarray,
    normalization: dict[str, Any],
) -> dict[str, float]:
    min_spread = float(normalization["min_spread_normalized"])
    min_gap = float(normalization["min_level_gap_normalized"])
    generated64 = np.asarray(generated, dtype=np.float64)
    real64 = np.asarray(real, dtype=np.float64)
    diff = generated64 - real64
    real_boundary = continuous_boundary_stats(real64, min_spread, min_gap)
    generated_boundary = continuous_boundary_stats(generated64, min_spread, min_gap)
    tolerance = float(real_boundary["boundary_tolerance"])
    real_gap = _gap_values(real64)
    generated_gap = _gap_values(generated64)
    real_interior = real_gap[real_gap > min_gap + tolerance] - min_gap
    generated_interior = generated_gap[generated_gap > min_gap + tolerance] - min_gap
    real_boundary_rate = 0.5 * (
        real_boundary["ask_gap_at_minimum_rate"]
        + real_boundary["bid_gap_at_minimum_rate"]
    )
    generated_boundary_rate = 0.5 * (
        generated_boundary["ask_gap_at_minimum_rate"]
        + generated_boundary["bid_gap_at_minimum_rate"]
    )
    return {
        "samples": int(generated64.shape[0]),
        "overall_mse": float(np.mean(np.square(diff))),
        "price_mse": float(np.mean(np.square(diff[:, 0]))),
        "volume_mse": float(np.mean(np.square(diff[:, 1]))),
        "invalid_snapshot_rate": float(
            1.0 - admissible_snapshot_mask(generated64, normalization).mean()
        ),
        "real_gap_boundary_rate": float(real_boundary_rate),
        "generated_gap_boundary_rate": float(generated_boundary_rate),
        "gap_boundary_calibration_abs_error": float(
            abs(generated_boundary_rate - real_boundary_rate)
        ),
        "real_interior_gap_count": int(real_interior.size),
        "generated_interior_gap_count": int(generated_interior.size),
        "conditional_interior_gap_w1": wasserstein_1d(
            generated_interior, real_interior
        ),
    }


def load_paired(method_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    generated = np.load(method_dir / "generated_raw.npy", mmap_mode="r")
    real = np.load(method_dir / "real_future.npy", mmap_mode="r")
    if generated.shape != real.shape:
        raise ValueError(f"Shape mismatch in {method_dir}: {generated.shape} vs {real.shape}")
    return generated, real


def _relative_ratio(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if candidate == 0 else float("inf")
    return candidate / baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/boundary_aware_amzn_seed0/summary"),
    )
    parser.add_argument("--minimum-boundary-improvement", type=float, default=0.25)
    parser.add_argument("--maximum-fidelity-ratio", type=float, default=1.05)
    args = parser.parse_args()

    normalization = json.loads(args.normalization.read_text())
    baseline_generated, baseline_real = load_paired(args.baseline_dir)
    candidate_generated, candidate_real = load_paired(args.candidate_dir)
    if not np.array_equal(baseline_real, candidate_real):
        raise ValueError("Baseline and boundary-aware runs do not use identical real futures")

    baseline = summarize_method(baseline_generated, baseline_real, normalization)
    candidate = summarize_method(candidate_generated, candidate_real, normalization)
    baseline_error = baseline["gap_boundary_calibration_abs_error"]
    candidate_error = candidate["gap_boundary_calibration_abs_error"]
    boundary_improvement = (
        (baseline_error - candidate_error) / baseline_error
        if baseline_error > 0
        else (0.0 if candidate_error == 0 else -math.inf)
    )
    ratios = {
        key: _relative_ratio(candidate[key], baseline[key])
        for key in ("overall_mse", "price_mse", "conditional_interior_gap_w1")
    }
    gates = {
        "zero_candidate_invalidity": candidate["invalid_snapshot_rate"] == 0.0,
        "boundary_calibration_improves_by_required_fraction": (
            boundary_improvement >= args.minimum_boundary_improvement
        ),
        "overall_mse_not_materially_worse": (
            ratios["overall_mse"] <= args.maximum_fidelity_ratio
        ),
        "price_mse_not_materially_worse": (
            ratios["price_mse"] <= args.maximum_fidelity_ratio
        ),
        "interior_gap_w1_not_materially_worse": (
            ratios["conditional_interior_gap_w1"] <= args.maximum_fidelity_ratio
        ),
    }
    report = {
        "comparison": "AMZN-1Hz train-seed-0 fixed-evaluation-seed-0",
        "baseline": {"representation": "legal_lob", **baseline},
        "candidate": {"representation": "legal_lob_ba", **candidate},
        "boundary_calibration_relative_improvement": boundary_improvement,
        "candidate_to_baseline_ratios": ratios,
        "frozen_before_evaluation_thresholds": {
            "minimum_boundary_improvement": args.minimum_boundary_improvement,
            "maximum_fidelity_ratio": args.maximum_fidelity_ratio,
        },
        "gates": gates,
        "scale_up_warranted": all(gates.values()),
        "scope": (
            "Boundary-aware parameterization diagnostic only; not causal, "
            "message-level, or exchange-level evidence."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "boundary_aware_pilot_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Boundary-Aware LegalLOB Pilot",
        "",
        "| Method | MSE | Price MSE | Invalidity | Gap boundary | Boundary error | Interior-gap W1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (report["baseline"], report["candidate"]):
        lines.append(
            f"| {row['representation']} | {row['overall_mse']:.6g} | "
            f"{row['price_mse']:.6g} | {row['invalid_snapshot_rate']:.6g} | "
            f"{row['generated_gap_boundary_rate']:.6g} | "
            f"{row['gap_boundary_calibration_abs_error']:.6g} | "
            f"{row['conditional_interior_gap_w1']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Boundary-calibration relative improvement: `{boundary_improvement:.3%}`.",
            f"Scale-up warranted by all frozen-before-evaluation gates: **{all(gates.values())}**.",
            "",
            "This is a representation diagnostic, not causal or message-level evidence.",
        ]
    )
    (args.output_dir / "boundary_aware_pilot_summary.md").write_text(
        "\n".join(lines) + "\n"
    )
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    print((args.output_dir / "boundary_aware_pilot_summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

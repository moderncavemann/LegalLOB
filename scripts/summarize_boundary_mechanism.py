#!/usr/bin/env python3
"""Summarize feasibility-boundary mass and LegalLOB latent tails."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = (
    ("AAPL_PUBLIC_ITCH_1S", "AAPL-1Hz", Path("results/submission_audits/AAPL/saved_generation_audit.json")),
    ("AMZN_PUBLIC_ITCH_1S", "AMZN-1Hz", Path("results/submission_audits/AMZN/saved_generation_audit.json")),
    ("GOOG_DATABENTO_RTH_V2", "GOOG-Event", Path("results/submission_audits/GOOG_databento_RTH_v2/saved_generation_audit.json")),
    ("GOOG_PUBLIC_ITCH_1S", "GOOG-1Hz", Path("results/submission_audits/GOOG/saved_generation_audit.json")),
)


def _method_lookup(summary_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(summary_path.read_text())
    return {
        (row["dataset_id"], row["method_id"]): row
        for row in payload["summary"]
    }


def collect_rows(summary_path: Path) -> list[dict[str, Any]]:
    metrics = _method_lookup(summary_path)
    rows = []
    for dataset_id, label, audit_path in DATASETS:
        audit = json.loads(audit_path.read_text())
        boundary = audit["real_boundary_prevalence"]
        latent_runs = audit["legal_latent_audit"]

        def latent_mean(group: str, field: str) -> float:
            values = [float(run["real"][group][field]) for run in latent_runs]
            return float(np.mean(values))

        matched = metrics[(dataset_id, "matched_no_projection")]
        legal = metrics[(dataset_id, "legal_no_projection")]
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": label,
                "spread_boundary_rate": float(boundary["spread_at_minimum_rate"]),
                "ask_gap_boundary_rate": float(boundary["ask_gap_at_minimum_rate"]),
                "bid_gap_boundary_rate": float(boundary["bid_gap_at_minimum_rate"]),
                "mean_gap_boundary_rate": 0.5
                * (
                    float(boundary["ask_gap_at_minimum_rate"])
                    + float(boundary["bid_gap_at_minimum_rate"])
                ),
                "ask_gap_latent_below_minus20_rate": latent_mean(
                    "ask_gap_raw", "fraction_below_minus_20"
                ),
                "bid_gap_latent_below_minus20_rate": latent_mean(
                    "bid_gap_raw", "fraction_below_minus_20"
                ),
                "mean_gap_latent_below_minus20_rate": 0.5
                * (
                    latent_mean("ask_gap_raw", "fraction_below_minus_20")
                    + latent_mean("bid_gap_raw", "fraction_below_minus_20")
                ),
                "matched_mse": float(matched["mse_output_mean"]),
                "legal_mse": float(legal["mse_output_mean"]),
                "legal_minus_matched_mse": float(legal["mse_output_mean"])
                - float(matched["mse_output_mean"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feasibility-summary",
        default="paper_data/cross_asset_feasibility_summary.json",
    )
    parser.add_argument(
        "--output-csv", default="paper_data/boundary_mechanism_summary.csv"
    )
    parser.add_argument(
        "--output-json", default="paper_data/boundary_mechanism_summary.json"
    )
    parser.add_argument(
        "--table-path", default="paper_tables/table_boundary_mechanism.md"
    )
    args = parser.parse_args()

    rows = collect_rows(Path(args.feasibility_summary))
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_json).write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Boundary Mass and Positive-Link Latent Tails",
        "",
        "Boundary and latent-tail rates use the frozen real test windows. The",
        "MSE delta is LegalLOB minus the matched identity-link control.",
        "",
        "| Dataset | Mean gap boundary % | Mean gap latent < -20 % | Matched MSE | LegalLOB MSE | Legal - Matched |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {boundary:.2f} | {tail:.2f} | {matched:.4f} | {legal:.4f} | {delta:+.4f} |".format(
                dataset=row["dataset"],
                boundary=100 * row["mean_gap_boundary_rate"],
                tail=100 * row["mean_gap_latent_below_minus20_rate"],
                matched=row["matched_mse"],
                legal=row["legal_mse"],
                delta=row["legal_minus_matched_mse"],
            )
        )
    lines.extend(
        [
            "",
            "Boundary mass deterministically creates a negative inverse-softplus tail,",
            "but it is not sufficient to explain fidelity loss: AAPL has the largest",
            "boundary mass while LegalLOB remains close to Matched. AMZN combines a",
            "nontrivial boundary atom with a broad interior gap distribution and has the",
            "largest LegalLOB penalty; the current evidence does not isolate a single cause.",
        ]
    )
    table_path = Path(args.table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

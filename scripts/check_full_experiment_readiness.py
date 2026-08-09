#!/usr/bin/env python3
"""Machine-readable gate for the current full snapshot experiment protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.utils.device import device_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GOOG_databento_20190130_mbp10_rth_v2")
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--fi2010-dir", default="data/processed/fi2010/noauction_zscore_full")
    parser.add_argument("--public-panel-root", default="data/processed/snapshot_l2")
    parser.add_argument("--public-panel-prefix", default="PUBLIC_ITCH_2019_SAMPLES")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="FULL_EXPERIMENT_READINESS.json")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    cfg = manifest["datasets"][args.dataset]
    dataset_dir = Path(cfg["state_path"]).parent
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    normalization = json.loads((dataset_dir / "normalization.json").read_text())
    schema = json.loads((dataset_dir / "schema_validation.json").read_text())

    checks = {
        "schema_has_no_errors": schema.get("status") in {"pass", "pass_with_warnings"} and not schema.get("errors"),
        "regular_hours_only": metadata.get("session_filter") == "regular_hours_0930_1600",
        "true_timestamps": bool(metadata.get("has_true_timestamps")),
        "true_exchange_date": (
            bool(metadata.get("has_true_day_id"))
            and metadata.get("day_id_status") == "exchange_local_date_from_databento_ts_event"
        ),
        "continuity_segments": bool(metadata.get("has_continuity_segments")) and (dataset_dir / "segment_id.npy").exists(),
        "source_rows_preserved": bool(metadata.get("source_row_index_preserved")) and (dataset_dir / "source_row_idx.npy").exists(),
        "anchor_relative_tick_scaled": normalization.get("price_transform") == "history_last_mid_tick_scaled_dynamic",
        "train_only_normalization": normalization.get("fit_split") == "train_only",
        "positive_spread_constraint": float(normalization.get("min_spread_normalized", 0)) > 0,
        "positive_level_gap_constraint": float(normalization.get("min_level_gap_normalized", 0)) > 0,
        "snapshot_arrays_present": all(
            (dataset_dir / name).exists()
            for name in ["states.npy", "states_raw.npy", "actions.npy", "timestamps.npy", "day_id.npy", "split.npy"]
        ),
    }
    if checks["snapshot_arrays_present"]:
        states = np.load(dataset_dir / "states.npy", mmap_mode="r")
        actions = np.load(dataset_dir / "actions.npy", mmap_mode="r")
        checks["state_action_alignment"] = len(states) == len(actions) + 1

    fi_dir = Path(args.fi2010_dir)
    fi_metadata_path = fi_dir / "metadata.json"
    if fi_metadata_path.exists():
        fi_metadata = json.loads(fi_metadata_path.read_text())
        fi_checks = {
            "all_nine_folds_processed": fi_metadata.get("folds") == list(range(1, 10)),
            "anchored_cv_declared_complete": bool(fi_metadata.get("anchored_cross_validation_complete")),
            "all_fold_arrays_present": all(
                (fi_dir / f"{kind}_cf{fold}_{suffix}.npy").exists()
                for fold in range(1, 10)
                for kind in ["train", "test"]
                for suffix in ["lob_features", "labels"]
            ),
        }
    else:
        fi_checks = {"metadata_present": False}

    repository_checks = {
        "matched_coordinate_baseline": (
            Path("genlob/utils/legal_lob.py").exists()
            and "class MatchedLOBTransform" in Path("genlob/utils/legal_lob.py").read_text()
            and "--include-matched-baseline" in Path("scripts/run_snapshot_ablation.py").read_text()
        ),
        "persistence_and_empirical_block_baselines": (
            Path("scripts/evaluate_snapshot_baseline.py").exists()
            and "moving_block_bootstrap" in Path("scripts/evaluate_snapshot_baseline.py").read_text()
        ),
        "temporal_distribution_metrics": all(
            token in Path("genlob/eval/metrics.py").read_text()
            for token in ["stylized_facts", "horizon_fidelity", "ks_2sample", "rbf_mmd2_biased"]
        ),
        "discrete_validity_diagnostics": (
            "discrete_lob_validity" in Path("genlob/eval/metrics.py").read_text()
        ),
        "continuous_claim_boundary_documented": (
            "continuous hard structural legality" in Path("FULL_EXPERIMENT_PROTOCOL.md").read_text()
        ),
    }

    public_root = Path(args.public_panel_root)
    public_datasets = [
        f"{args.public_panel_prefix}_{symbol}_{mode}"
        for symbol in ["AAPL", "AMZN", "GOOG", "INTC"]
        for mode in ["event", "1s"]
    ]
    public_panel_checks = {}
    for dataset_id in public_datasets:
        panel_dir = public_root / dataset_id
        required = [
            "states.npy", "states_raw.npy", "actions.npy", "timestamps.npy",
            "day_id.npy", "segment_id.npy", "split.npy", "normalization.json",
            "metadata.json", "build_report.json",
        ]
        public_panel_checks[dataset_id] = all((panel_dir / name).exists() for name in required)

    public_panel_ready = all(public_panel_checks.values()) and all(repository_checks.values())
    public_panel_status = (
        "The six-day multi-stock public ITCH panel is ready for the confirmatory snapshot experiment. "
        if public_panel_ready
        else "The multi-stock public panel is ready only after all eight event/1s processed datasets exist. "
    )
    payload = {
        "dataset": args.dataset,
        "device": device_summary(args.device),
        "snapshot_protocol_checks": checks,
        "fi2010_checks": fi_checks,
        "repository_protocol_checks": repository_checks,
        "public_multistock_panel_checks": public_panel_checks,
        "ready_for_full_snapshot_projection_experiment": all(checks.values()),
        "ready_for_full_fi2010_anchored_benchmark": all(fi_checks.values()),
        "ready_for_multistock_public_panel_experiment": public_panel_ready,
        "still_out_of_scope": [
            "consecutive-day GOOG_9day or LOBSTER replication",
            "message-level exact replay",
            "causal Buy/Sell action effects",
            "queue position and fill calibration",
            "full LOB-Bench",
        ],
        "claim_boundary": (
            "Ready for the single-day RTH snapshot-level projection ablation and FI-2010 prediction run. "
            + public_panel_status +
            "LegalLOB guarantees continuous structural legality, not tick-grid or integer-volume legality, "
            "and is not ready for a full message-level LOB world-model claim."
        ),
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready_for_full_snapshot_projection_experiment"] and payload["ready_for_full_fi2010_anchored_benchmark"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

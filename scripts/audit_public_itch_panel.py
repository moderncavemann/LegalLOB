#!/usr/bin/env python3
"""Audit the canonical public ITCH panel and optionally export DiffLOB views."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_dataset_schema import validate_snapshot_dataset


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--prefix", default="PUBLIC_ITCH_2019_SAMPLES")
    parser.add_argument("--symbols", default="AAPL,AMZN,GOOG,INTC")
    parser.add_argument("--modes", default="event,1s")
    parser.add_argument("--output", default="PUBLIC_ITCH_PANEL_AUDIT.json")
    parser.add_argument("--reconstruction-batches", type=int, default=2)
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--export-difflob", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    configured = manifest.get("datasets", {})
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    rows = []
    failures = []

    for symbol in symbols:
        for mode in modes:
            dataset_id = f"{args.prefix}_{symbol}_{mode}"
            cfg = configured.get(dataset_id)
            if cfg is None:
                failures.append(f"missing manifest dataset: {dataset_id}")
                rows.append({"dataset": dataset_id, "status": "missing"})
                continue
            report = validate_snapshot_dataset(dataset_id, cfg, manifest_path, tolerance=1e-8)
            dataset_dir = Path(cfg["state_path"]).parent
            schema_path = dataset_dir / "schema_validation.json"
            if not args.dry_run:
                schema_path.write_text(json.dumps(report, indent=2) + "\n")
            state_shape = list(np.load(cfg["state_path"], mmap_mode="r").shape)
            row = {
                "dataset": dataset_id,
                "mode": mode,
                "status": report["status"],
                "errors": report["errors"],
                "warnings": report["warnings"],
                "state_shape": state_shape,
                "split_counts": report.get("metrics", {}).get("split_counts"),
                "normalized_legality": report.get("metrics", {}).get("legality"),
                "raw_exchange_validity": report.get("metrics", {}).get("raw_exchange_validity"),
                "timestamp_reversal_rate_within_day": report.get("metrics", {}).get(
                    "timestamp_reversal_rate_within_day"
                ),
            }
            rows.append(row)
            if report["errors"]:
                failures.extend(f"{dataset_id}: {message}" for message in report["errors"])
                continue

            if not args.skip_reconstruction:
                _run(
                    [
                        sys.executable,
                        "scripts/check_legal_lob_reconstruction.py",
                        "--dataset",
                        dataset_id,
                        "--data-dir",
                        args.data_dir,
                        "--n-batches",
                        str(args.reconstruction_batches),
                        "--json-out",
                        str(dataset_dir / "legal_lob_reconstruction.json"),
                    ],
                    args.dry_run,
                )
            if args.export_difflob and mode == "1s":
                _run(
                    [
                        sys.executable,
                        "scripts/export_difflob_dataset.py",
                        "--dataset",
                        dataset_id,
                        "--data-dir",
                        args.data_dir,
                    ],
                    args.dry_run,
                )

    payload = {
        "panel": args.prefix,
        "symbols": symbols,
        "modes": modes,
        "datasets_expected": len(symbols) * len(modes),
        "datasets_found": sum(row.get("status") != "missing" for row in rows),
        "all_schema_checks_pass": not failures,
        "failures": failures,
        "datasets": rows,
        "scope": (
            "six non-consecutive public NASDAQ sample days; snapshot-level continuous "
            "legality and DiffLOB-compatible 1 Hz data, not LOBSTER or message-level replay"
        ),
    }
    if not args.dry_run:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backfill exact int64 Databento event timestamps without rebuilding states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    config = manifest.get("datasets", {}).get(args.dataset)
    if not config:
        raise SystemExit(f"Unknown dataset: {args.dataset}")

    metadata_path = Path(config["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    dbn_path = Path(metadata["dbn_file"])
    source_rows = np.load(config["source_row_idx_path"], mmap_mode="r")

    frame = db.DBNStore.from_file(dbn_path).to_df()
    event_ns = pd.to_datetime(frame["ts_event"], utc=True).astype("int64").to_numpy(
        dtype=np.int64
    )
    if source_rows.min(initial=0) < 0 or source_rows.max(initial=-1) >= len(event_ns):
        raise ValueError("source_row_idx contains rows outside the Databento source")
    retained_ns = event_ns[np.asarray(source_rows)]

    output = metadata_path.parent / "timestamps_ns.npy"
    np.save(output, retained_ns)
    metadata.update(
        {
            "timestamp_status": (
                "exact_databento_ts_event_int64_ns_with_float_seconds_compatibility"
            ),
            "timestamp_precision": "nanoseconds_int64",
            "timestamp_ns_path": str(output),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    config["timestamp_ns_path"] = str(output)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    dataset_manifest = metadata_path.parent / "manifest.yaml"
    if dataset_manifest.exists():
        payload = yaml.safe_load(dataset_manifest.read_text()) or {}
        payload.setdefault("files", {})["timestamps_ns"] = "timestamps_ns.npy"
        dataset_manifest.write_text(yaml.safe_dump(payload, sort_keys=False))

    print(
        json.dumps(
            {
                "status": "pass",
                "dataset": args.dataset,
                "rows": int(len(retained_ns)),
                "dtype": str(retained_ns.dtype),
                "output": str(output),
                "monotone_non_decreasing": bool(np.all(np.diff(retained_ns) >= 0)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

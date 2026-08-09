import json
import sqlite3
from pathlib import Path

import numpy as np
import yaml

from scripts.export_snapshot_l2_sqlite import export_dataset


def test_snapshot_export_creates_constrained_rows(tmp_path: Path):
    dataset_dir = tmp_path / "data/processed/snapshot_l2/SYNTH"
    dataset_dir.mkdir(parents=True)
    states = np.zeros((3, 2, 20), dtype=np.float64)
    states[:, 0, :10] = 101 + np.arange(10)
    states[:, 0, 10:] = 99 - np.arange(10)
    states[:, 1] = 10
    np.save(dataset_dir / "states_raw.npy", states)
    np.save(dataset_dir / "timestamps_ns.npy", np.asarray([1, 2, 3], dtype=np.int64))
    np.save(dataset_dir / "timestamps.npy", np.asarray([1e-9, 2e-9, 3e-9]))
    np.save(dataset_dir / "day_id.npy", np.asarray([20200101] * 3, dtype=np.int32))
    np.save(dataset_dir / "segment_id.npy", np.zeros(3, dtype=np.int32))
    np.save(dataset_dir / "source_row_idx.npy", np.arange(3, dtype=np.int64))
    np.save(dataset_dir / "split.npy", np.asarray([0, 1, 2], dtype=np.int8))
    (dataset_dir / "normalization.json").write_text(
        json.dumps({"fit_split": "train_only", "price_transform": "none", "volume_transform": "none"})
    )
    (dataset_dir / "metadata.json").write_text("{}")

    manifest = {
        "datasets": {
            "SYNTH": {
                "raw_state_path": str(dataset_dir.relative_to(tmp_path) / "states_raw.npy"),
                "timestamps_path": str(dataset_dir.relative_to(tmp_path) / "timestamps.npy"),
                "timestamp_ns_path": str(dataset_dir.relative_to(tmp_path) / "timestamps_ns.npy"),
                "day_id_path": str(dataset_dir.relative_to(tmp_path) / "day_id.npy"),
                "segment_id_path": str(dataset_dir.relative_to(tmp_path) / "segment_id.npy"),
                "source_row_idx_path": str(dataset_dir.relative_to(tmp_path) / "source_row_idx.npy"),
                "split_path": str(dataset_dir.relative_to(tmp_path) / "split.npy"),
                "normalization_path": str(dataset_dir.relative_to(tmp_path) / "normalization.json"),
                "metadata_path": str(dataset_dir.relative_to(tmp_path) / "metadata.json"),
            }
        }
    }
    manifest_path = tmp_path / "datasets.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    output = tmp_path / "warehouse.sqlite"
    schema = Path(__file__).resolve().parents[1] / "data/catalog/CANONICAL_MARKET_SCHEMA.sql"
    report = export_dataset(tmp_path, manifest_path, "SYNTH", output, schema)

    assert report["status"] == "pass"
    assert report["stored_rows"] == 3
    assert report["timestamp_precision"] == "exact_int64_ns"
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT COUNT(*) FROM lob_snapshot_l2").fetchone()[0] == 3

#!/usr/bin/env python3
"""Export one processed snapshot_l2 dataset to the local SQLite warehouse."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml


SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _rows(
    dataset_id: str,
    states: np.ndarray,
    timestamps_ns: np.ndarray,
    day_id: np.ndarray,
    segment_id: np.ndarray,
    split: np.ndarray,
    source_row_idx: np.ndarray,
    start: int,
    stop: int,
) -> Iterator[tuple[Any, ...]]:
    for idx in range(start, stop):
        prices = states[idx, 0]
        sizes = states[idx, 1]
        values: list[Any] = [
            dataset_id,
            idx,
            int(timestamps_ns[idx]),
            int(day_id[idx]),
            int(segment_id[idx]),
            SPLIT_NAMES[int(split[idx])],
            int(source_row_idx[idx]),
        ]
        for level in range(10):
            values.extend((float(prices[level]), float(sizes[level])))
        for level in range(10):
            values.extend((float(prices[10 + level]), float(sizes[10 + level])))
        yield tuple(values)


def export_dataset(
    root: Path,
    manifest_path: Path,
    dataset_id: str,
    output: Path,
    schema_path: Path,
    chunk_size: int = 10_000,
    replace_dataset: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    config = manifest.get("datasets", {}).get(dataset_id)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_id}")

    def resolve(key: str) -> Path:
        value = config.get(key)
        if not value:
            raise ValueError(f"{dataset_id} does not define {key}")
        return root / str(value)

    states_key = "raw_state_path" if config.get("raw_state_path") else "state_path"
    states = np.load(resolve(states_key), mmap_mode="r")
    split = np.load(resolve("split_path"), mmap_mode="r")
    day_id = np.load(resolve("day_id_path"), mmap_mode="r")

    dataset_dir = resolve("metadata_path").parent
    segment_path = root / str(config.get("segment_id_path", dataset_dir / "segment_id.npy"))
    source_path = root / str(
        config.get("source_row_idx_path", dataset_dir / "source_row_idx.npy")
    )
    segment_id = (
        np.load(segment_path, mmap_mode="r")
        if segment_path.exists()
        else np.zeros(len(states), dtype=np.int32)
    )
    source_row_idx = (
        np.load(source_path, mmap_mode="r")
        if source_path.exists()
        else np.arange(len(states), dtype=np.int64)
    )

    timestamp_ns_value = config.get("timestamp_ns_path")
    timestamp_ns_path = root / str(timestamp_ns_value) if timestamp_ns_value else None
    if timestamp_ns_path and timestamp_ns_path.exists():
        timestamps_ns = np.load(timestamp_ns_path, mmap_mode="r")
        timestamp_precision = "exact_int64_ns"
    else:
        timestamps = np.load(resolve("timestamps_path"), mmap_mode="r")
        timestamps_ns = np.rint(np.asarray(timestamps) * 1e9).astype(np.int64)
        timestamp_precision = "reconstructed_from_float_seconds"

    lengths = {
        len(states),
        len(timestamps_ns),
        len(day_id),
        len(segment_id),
        len(split),
        len(source_row_idx),
    }
    if len(lengths) != 1:
        raise ValueError(f"Dataset arrays are not row-aligned: lengths={sorted(lengths)}")
    if states.ndim != 3 or states.shape[1:] != (2, 20):
        raise ValueError(f"Expected states [T,2,20], got {states.shape}")

    row_count = len(states) if max_rows is None else min(len(states), max_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(schema_path.read_text())
        existing = connection.execute(
            "SELECT COUNT(*) FROM lob_snapshot_l2 WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()[0]
        if existing and not replace_dataset:
            raise ValueError(
                f"{dataset_id} already has {existing} rows; pass --replace-dataset"
            )
        if existing:
            connection.execute("DELETE FROM lob_snapshot_l2 WHERE dataset_id = ?", (dataset_id,))

        placeholders = ",".join("?" for _ in range(47))
        insert = f"INSERT INTO lob_snapshot_l2 VALUES ({placeholders})"
        for start in range(0, row_count, chunk_size):
            stop = min(start + chunk_size, row_count)
            connection.executemany(
                insert,
                _rows(
                    dataset_id,
                    states,
                    timestamps_ns,
                    day_id,
                    segment_id,
                    split,
                    source_row_idx,
                    start,
                    stop,
                ),
            )
        normalization = _load_json(resolve("normalization_path"))
        connection.execute(
            "INSERT OR REPLACE INTO dataset_transform VALUES (?,?,?,?,?,?,?,?)",
            (
                dataset_id,
                str(normalization.get("fit_split", "unknown")),
                None,
                None,
                str(normalization.get("price_transform", "unknown")),
                str(normalization.get("volume_transform", "unknown")),
                json.dumps(normalization, sort_keys=True),
                None,
            ),
        )
        connection.commit()
        stored = connection.execute(
            "SELECT COUNT(*) FROM lob_snapshot_l2 WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]

    return {
        "status": "pass" if stored == row_count and integrity == "ok" else "fail",
        "dataset_id": dataset_id,
        "source_rows": int(len(states)),
        "stored_rows": int(stored),
        "timestamp_precision": timestamp_precision,
        "database": str(output),
        "database_size_bytes": output.stat().st_size,
        "sqlite_integrity": integrity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/datasets.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/warehouse/genlob_market.sqlite"))
    parser.add_argument(
        "--schema", type=Path, default=Path("data/catalog/CANONICAL_MARKET_SCHEMA.sql")
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--replace-dataset", action="store_true")
    args = parser.parse_args()
    report = export_dataset(
        args.root.resolve(),
        args.manifest.resolve(),
        args.dataset,
        args.output.resolve(),
        args.schema.resolve(),
        args.chunk_size,
        args.replace_dataset,
        args.max_rows,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

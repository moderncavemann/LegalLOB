#!/usr/bin/env python3
"""Prepare FI-2010 no-auction benchmark arrays.

This adapter intentionally produces benchmark feature/label arrays, not a
GenLOB ``snapshot_l2`` dataset. FI-2010 is normalized and label-oriented, so it
is useful for DeepLOB-style public benchmark comparability but not for market
replay, queue simulation, or LegalLOB projection ablations.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def _fold_files(folds: list[int]) -> dict[str, str]:
    files = {}
    for fold in folds:
        files[f"train_cf{fold}"] = f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
        files[f"test_cf{fold}"] = f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
    return files


def _load_matrix(path: Path, max_samples: int | None = None) -> np.ndarray:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".txt")]
            if len(members) != 1:
                raise ValueError(f"{path}: expected exactly one text matrix, got {members}")
            with archive.open(members[0]) as stream:
                matrix = np.loadtxt(stream, dtype=np.float32)
    else:
        matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{path}: expected 2D FI-2010 matrix, got {matrix.shape}")
    if matrix.shape[0] != 149:
        raise ValueError(f"{path}: expected 149 rows, got {matrix.shape[0]}")
    if max_samples is not None:
        matrix = matrix[:, :max_samples]
    return matrix


def _find_source(raw_dir: Path, filename: str) -> Path:
    plain = raw_dir / filename
    zipped = raw_dir / f"{filename}.zip"
    if plain.exists():
        return plain
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Missing {plain} or {zipped}")


def _label_counts(labels: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for col in range(labels.shape[1]):
        values, counts = np.unique(labels[:, col], return_counts=True)
        out[f"label_{col}"] = {str(int(v)): int(c) for v, c in zip(values, counts)}
    return out


def _write_dataset_card(output_dir: Path, metadata: dict[str, Any]) -> None:
    card = f"""# FI-2010 NoAuction ZScore Anchored-Fold Benchmark Adapter

Status: processed benchmark arrays.

This directory was generated from FI-2010 no-auction z-score text matrices. It
is a public benchmark adapter for DeepLOB-style prediction experiments, not a
replayable LOB dataset and not a `snapshot_l2` LegalLOB projection-ablation
dataset.

## Source

- Raw directory: `{metadata['raw_dir']}`
- Variant: no-auction, z-score normalization
- Feature rows retained:
  - `all_features`: first 144 FI-2010 feature rows
  - `lob_features`: first 40 FI-2010 LOB rows, commonly used by DeepLOB-style baselines
- Label rows: last 5 official FI-2010 label rows

## Output Files

For every requested anchored fold, outputs use `train_cfN` and `test_cfN` prefixes:

- `{{split}}_all_features.npy`: `[N, 144]` when the adapter is not run with `--lob-only`
- `{{split}}_lob_features.npy`: `[N, 40]`
- `{{split}}_labels.npy`: `[N, 5]`

## Scope

Supports:

- FI-2010 public benchmark loaders
- DeepLOB-style windowed prediction experiments
- pipeline and label protocol checks

Does not support:

- historical LOB replay
- queue/fill calibration
- event-level action causality
- LegalLOB structural projection claims
- realistic trading backtests
"""
    output_dir.joinpath("DATASET_CARD.md").write_text(card)


def _write_validation(output_dir: Path, metadata: dict[str, Any]) -> None:
    errors = []
    warnings = [
        "FI-2010 is normalized benchmark data, not replayable exchange data.",
        "Adapter output is benchmark feature/label arrays, not snapshot_l2.",
        "Labels are official FI-2010 labels and are not action-causality labels.",
    ]
    for split, info in metadata["splits"].items():
        if info["all_features_shape"][1] != 144:
            errors.append(f"{split}: all_features must have 144 columns")
        if info["lob_features_shape"][1] != 40:
            errors.append(f"{split}: lob_features must have 40 columns")
        if info["labels_shape"][1] != 5:
            errors.append(f"{split}: labels must have 5 columns")
        label_values = set()
        for counts in info["label_counts"].values():
            label_values.update(counts.keys())
        if not label_values.issubset({"1", "2", "3"}):
            errors.append(f"{split}: labels outside expected 1/2/3 values: {sorted(label_values)}")
    report = {
        "status": "pass_with_warnings" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
    }
    output_dir.joinpath("schema_validation.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="data/raw/fi2010/noauction_zscore/extracted",
        help="Directory containing FI-2010 no-auction z-score text files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/fi2010/noauction_zscore_cf7",
        help="Output directory for processed benchmark arrays.",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional smoke cap per split.")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--lob-only", action="store_true",
                        help="Skip redundant 144-feature arrays when only DeepLOB LOB rows are needed.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    if not folds or any(fold < 1 or fold > 9 for fold in folds):
        raise ValueError("--folds must contain values from 1 through 9")

    metadata: dict[str, Any] = {
        "dataset_id": output_dir.name,
        "dataset_type": "fi2010_benchmark",
        "variant": "noauction_zscore",
        "raw_dir": str(raw_dir),
        "max_samples": args.max_samples,
        "lob_only": args.lob_only,
        "folds": folds,
        "anchored_cross_validation_complete": folds == list(range(1, 10)),
        "feature_rows": 144,
        "lob_feature_rows": 40,
        "label_rows": 5,
        "splits": {},
        "limitations": [
            "normalized_benchmark_not_replay",
            "not_snapshot_l2",
            "not_action_causality",
            "not_queue_fill_calibration",
        ],
    }

    for split, filename in _fold_files(folds).items():
        path = _find_source(raw_dir, filename)
        print(f"Loading {path}...")
        matrix = _load_matrix(path, max_samples=args.max_samples)
        all_features = np.ascontiguousarray(matrix[:144].T)
        lob_features = np.ascontiguousarray(matrix[:40].T)
        labels = np.ascontiguousarray(matrix[144:].T.astype(np.int64))

        outputs = {
            f"{split}_lob_features.npy": lob_features,
            f"{split}_labels.npy": labels,
        }
        if not args.lob_only:
            outputs[f"{split}_all_features.npy"] = all_features
        for name, array in outputs.items():
            out_path = output_dir / name
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"{out_path} exists; pass --overwrite to replace")
            np.save(out_path, array)

        metadata["splits"][split] = {
            "source_file": str(path),
            "matrix_shape": list(matrix.shape),
            "all_features_shape": list(all_features.shape),
            "lob_features_shape": list(lob_features.shape),
            "labels_shape": list(labels.shape),
            "label_counts": _label_counts(labels),
        }
        print(
            f"  {split}: all={all_features.shape} lob={lob_features.shape} "
            f"labels={labels.shape}"
        )

    output_dir.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_validation(output_dir, metadata)
    _write_dataset_card(output_dir, metadata)
    print(f"Wrote FI-2010 benchmark adapter output to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

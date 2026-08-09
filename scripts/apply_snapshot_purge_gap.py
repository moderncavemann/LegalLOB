#!/usr/bin/env python3
"""Apply split-boundary purge gaps to a processed snapshot_l2 dataset.

This keeps train/val/test as contiguous blocks, removes rows around split
boundaries, recomputes train-only normalization from retained raw states, and
rebuilds action rows so actions remain aligned to state transitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_public_itch_snapshot_l2 import _fit_normalization, _write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _find_boundaries(split: np.ndarray) -> tuple[int, int]:
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]
    if val_idx.size == 0 or test_idx.size == 0:
        raise ValueError("split.npy must contain contiguous train/val/test labels 0, 1, 2")
    return int(val_idx[0]), int(test_idx[0])


def _rebuild_actions(old_actions: np.ndarray, kept_idx: np.ndarray, action_dim: int) -> np.ndarray:
    new_actions = np.zeros((len(kept_idx) - 1, action_dim), dtype=np.float32)
    contiguous = kept_idx[1:] == kept_idx[:-1] + 1
    src = kept_idx[:-1][contiguous]
    valid = src < len(old_actions)
    if valid.any():
        positions = np.where(contiguous)[0][valid]
        new_actions[positions] = old_actions[src[valid]]
    return new_actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--purge-gap", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    build_report_path = dataset_dir / "build_report.json"
    metadata = _load_json(metadata_path)
    if metadata.get("split_purge_gap") is not None and not args.force:
        raise SystemExit(
            f"{dataset_dir} already has split_purge_gap={metadata.get('split_purge_gap')}; "
            "use --force to reapply."
        )

    raw_states = np.load(dataset_dir / "states_raw.npy")
    old_actions = np.load(dataset_dir / "actions.npy")
    timestamps = np.load(dataset_dir / "timestamps.npy")
    day_id = np.load(dataset_dir / "day_id.npy")
    split = np.load(dataset_dir / "split.npy")

    if len(raw_states) != len(split):
        raise ValueError("states_raw.npy and split.npy lengths do not match")
    if len(timestamps) != len(split) or len(day_id) != len(split):
        raise ValueError("timestamps/day_id lengths must match split length")
    if args.purge_gap < 0:
        raise ValueError("--purge-gap must be non-negative")

    b1, b2 = _find_boundaries(split)
    gap = args.purge_gap
    keep = np.ones(len(split), dtype=bool)
    keep[max(0, b1 - gap):min(len(split), b1 + gap)] = False
    keep[max(0, b2 - gap):min(len(split), b2 + gap)] = False
    kept_idx = np.where(keep)[0]
    if len(kept_idx) < 2:
        raise ValueError("purge removed too many rows")

    raw_kept = raw_states[kept_idx].astype(np.float32, copy=False)
    split_kept = split[kept_idx].astype(np.int8, copy=False)
    states_kept, normalization = _fit_normalization(raw_kept, split_kept)
    actions_kept = _rebuild_actions(old_actions, kept_idx, old_actions.shape[1])
    timestamps_kept = timestamps[kept_idx]
    day_id_kept = day_id[kept_idx]

    np.save(dataset_dir / "states.npy", states_kept)
    np.save(dataset_dir / "states_raw.npy", raw_kept)
    np.save(dataset_dir / "states_merged.npy", raw_kept)
    np.save(dataset_dir / "actions.npy", actions_kept)
    np.save(dataset_dir / "timestamps.npy", timestamps_kept)
    np.save(dataset_dir / "day_id.npy", day_id_kept)
    np.save(dataset_dir / "split.npy", split_kept)
    _write_json(dataset_dir / "normalization.json", normalization)

    report = {
        "dataset_dir": str(dataset_dir),
        "purge_gap": gap,
        "old_rows": int(len(split)),
        "new_rows": int(len(split_kept)),
        "removed_rows": int(len(split) - len(split_kept)),
        "old_boundaries": {"train_val": b1, "val_test": b2},
        "kept_split_counts": {
            "train": int((split_kept == 0).sum()),
            "val": int((split_kept == 1).sum()),
            "test": int((split_kept == 2).sum()),
        },
        "actions_shape": list(actions_kept.shape),
        "normalization": normalization,
        "boundary_transition_actions_zeroed": 2 if gap > 0 else 0,
    }
    _write_json(dataset_dir / "purge_report.json", report)

    metadata.update({
        "has_message_data": False,
        "has_raw_itch_source": True,
        "processed_event_l3_available": False,
        "action_label_status": "itch_event_level_diagnostic_not_causal",
        "split_purge_gap": gap,
        "split_policy": "contiguous_time_split_with_boundary_purge_gap",
        "state_shape": list(states_kept.shape),
        "action_shape": list(actions_kept.shape),
    })
    _write_json(metadata_path, metadata)

    build_report = _load_json(build_report_path)
    build_report.update({
        "states_shape": list(states_kept.shape),
        "actions_shape": list(actions_kept.shape),
        "normalization": normalization,
        "split_purge_gap": gap,
        "purge_report": report,
    })
    _write_json(build_report_path, build_report)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

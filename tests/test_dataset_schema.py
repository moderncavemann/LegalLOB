import json

import numpy as np

from scripts.validate_dataset_schema import validate_snapshot_dataset


def test_validator_accepts_audited_purge_rows_between_ordered_splits(tmp_path):
    total = 20
    states = np.zeros((total, 2, 20), dtype=np.float32)
    states[:, 0, :10] = np.arange(1, 11, dtype=np.float32) * 0.1
    states[:, 0, 10:] = -np.arange(0, 10, dtype=np.float32) * 0.1
    states[:, 1] = 1.0
    raw_states = states.astype(np.float64)
    raw_states[:, 0] += 100.0
    actions = np.zeros((total - 1, 5), dtype=np.float32)
    timestamps = np.arange(total, dtype=np.float64)
    day_id = np.full(total, 20190130, dtype=np.int32)
    split = np.array([0] * 5 + [-1] * 2 + [1] * 5 + [-1] * 2 + [2] * 6, dtype=np.int8)

    np.save(tmp_path / "states.npy", states)
    np.save(tmp_path / "states_raw.npy", raw_states)
    np.save(tmp_path / "actions.npy", actions)
    np.save(tmp_path / "timestamps.npy", timestamps)
    np.save(tmp_path / "day_id.npy", day_id)
    np.save(tmp_path / "split.npy", split)
    (tmp_path / "normalization.json").write_text(json.dumps({
        "fit_split": "train_only",
        "min_spread_normalized": 0.1,
        "min_level_gap_normalized": 0.1,
        "tick_size_raw": 0.1,
    }))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "split_policy": {"type": "single_day_contiguous_with_purge"},
        "day_id_status": "exchange_local_date",
    }))
    cfg = {
        "state_path": str(tmp_path / "states.npy"),
        "raw_state_path": str(tmp_path / "states_raw.npy"),
        "action_path": str(tmp_path / "actions.npy"),
        "timestamps_path": str(tmp_path / "timestamps.npy"),
        "day_id_path": str(tmp_path / "day_id.npy"),
        "split_path": str(tmp_path / "split.npy"),
        "normalization_path": str(tmp_path / "normalization.json"),
        "metadata_path": str(tmp_path / "metadata.json"),
        "has_true_timestamps": True,
        "has_message_data": False,
        "action_label_status": "diagnostic_proxy",
    }

    report = validate_snapshot_dataset("TEST", cfg, tmp_path / "manifest.yaml", 1e-8)
    assert report["errors"] == []
    assert report["metrics"]["split_sequence"] == ["train", "val", "test"]
    assert report["metrics"]["split_counts"]["purged"] == 4
    assert report["metrics"]["raw_exchange_validity"]["off_tick_price_rate"] == 0.0
    assert report["metrics"]["raw_exchange_validity"]["noninteger_volume_rate"] == 0.0

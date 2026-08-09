from pathlib import Path

import numpy as np

from scripts.prepare_public_itch_multisymbol import (
    ChunkWriter,
    LastObservationPerSecond,
    ParseResult,
    RTH_START_SECOND,
    _load_completed_parse,
    _source_identity,
    assign_splits,
    infer_day_id,
)


def test_infer_day_id():
    assert infer_day_id(Path("01302019.NASDAQ_ITCH50.gz")) == 20190130


def test_multiday_split_uses_last_two_days_for_validation_and_test():
    days = np.asarray([20190130, 20190130, 20190327, 20190730, 20190830], dtype=np.int32)
    split, report = assign_splits(days)
    assert split.tolist() == [0, 0, 0, 1, 2]
    assert report["policy"] == "default_last_two_days_val_test"


def test_single_day_split_marks_purge_rows():
    days = np.full(1000, 20190130, dtype=np.int32)
    split, report = assign_splits(days, purge_gap=10)
    assert report["policy"] == "single_day_chronological_with_purge"
    assert (split == -1).sum() == 40
    assert set(np.unique(split)) == {-1, 0, 1, 2}


def test_last_observation_sampler_carries_forward_missing_second(tmp_path: Path):
    writer = ChunkWriter(tmp_path, chunk_size=100)
    sampler = LastObservationPerSecond(writer, day_id=20190130)
    state_a = np.ones((2, 20), dtype=np.float64)
    state_b = np.full((2, 20), 2.0, dtype=np.float64)
    sampler.update((RTH_START_SECOND * 1_000_000_000) + 1, state_a, [1.0] * 5, 1)
    sampler.update(((RTH_START_SECOND + 2) * 1_000_000_000) + 1, state_b, [2.0] * 5, 2)
    sampler.finish(fill_to_close=False)
    writer.flush()
    states = np.load(tmp_path / "states_000000.npy")
    actions = np.load(tmp_path / "actions_000000.npy")
    assert len(states) == 3
    np.testing.assert_array_equal(states[0], state_a)
    np.testing.assert_array_equal(states[1], state_a)
    np.testing.assert_array_equal(states[2], state_b)
    np.testing.assert_array_equal(actions[1], np.zeros(5))


def test_completed_parse_marker_is_reused_only_for_matching_inputs(tmp_path: Path):
    source = tmp_path / "01302019.NASDAQ_ITCH50.gz"
    source.write_bytes(b"fixture")
    source.with_name(source.name + ".sha256").write_text("abc123  fixture\n")
    identity = _source_identity(source, ("AAPL", "GOOG"), ("event", "1s"), None)
    report = ParseResult(
        day_id=20190130,
        messages=42,
        truncated=False,
        counters={"AAPL": {"type_A": 1}, "GOOG": {}},
    )
    marker = tmp_path / "_parse_complete.json"
    import json

    marker.write_text(json.dumps({"identity": identity, "report": report.__dict__}))
    loaded = _load_completed_parse(marker, identity)
    assert loaded == report

    changed = dict(identity, symbols=["GOOG"])
    assert _load_completed_parse(marker, changed) is None

import numpy as np

from genlob.data.nasdaq import (
    NASDAQMultiDayDataset,
    normalize_anchor_relative_sequence,
    parse_snapshot_dataset_id,
    resolve_lob_constraints,
    select_evaluation_indices,
)


def test_parse_snapshot_dataset_id_accepts_canonical_and_legacy_alias():
    assert parse_snapshot_dataset_id("INTC_9day") == ("INTC", "9day")
    assert parse_snapshot_dataset_id("nasdaq_INTC_9day") == ("INTC", "9day")


def test_multiday_dataset_filters_cross_day_windows_when_day_ids_are_real():
    data = np.zeros((8, 2, 20), dtype=np.float32)
    actions = np.zeros((8, 5), dtype=np.float32)
    day_id = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)

    ds = NASDAQMultiDayDataset(
        data,
        actions,
        hist_len=2,
        pred_len=2,
        stride=1,
        day_id=day_id,
    )

    assert ds.starts.tolist() == [0, 4]


def test_multiday_dataset_keeps_windows_when_day_ids_are_placeholders():
    data = np.zeros((8, 2, 20), dtype=np.float32)
    actions = np.zeros((8, 5), dtype=np.float32)
    day_id = np.full(8, -1, dtype=np.int16)

    ds = NASDAQMultiDayDataset(
        data,
        actions,
        hist_len=2,
        pred_len=2,
        stride=1,
        day_id=day_id,
    )

    assert ds.starts.tolist() == [0, 1, 2, 3, 4]


def test_multiday_dataset_does_not_cross_continuity_segments():
    data = np.zeros((10, 2, 20), dtype=np.float32)
    actions = np.zeros((10, 5), dtype=np.float32)
    segment_id = np.asarray([0] * 5 + [1] * 5, dtype=np.int32)
    ds = NASDAQMultiDayDataset(
        data,
        actions,
        hist_len=2,
        pred_len=2,
        segment_id=segment_id,
    )
    assert ds.starts.tolist() == [0, 1, 5, 6]


def test_anchor_relative_normalization_preserves_future_mid_path():
    raw = np.zeros((5, 2, 20), dtype=np.float32)
    for t in range(5):
        mid = 100.0 + t
        raw[t, 0, :10] = mid + np.arange(1, 11, dtype=np.float32)
        raw[t, 0, 10:] = mid - np.arange(1, 11, dtype=np.float32)
        raw[t, 1] = 4.0
    normalization = {
        "price_scale": 1.0,
        "price_global_std": 1.0,
        "volume_clip": 100.0,
        "volume_scale": 1.0,
    }
    normalized = normalize_anchor_relative_sequence(raw, hist_len=3, normalization=normalization)
    mid = (normalized[:, 0, 0] + normalized[:, 0, 10]) / 2
    assert np.allclose(mid.numpy(), [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_constraints_resolve_from_dataset_metadata():
    normalization = {
        "min_spread_normalized": 0.02,
        "min_level_gap_normalized": 0.01,
    }
    assert resolve_lob_constraints(normalization) == (0.02, 0.01)


def test_uniform_evaluation_selection_spans_the_split():
    selected = select_evaluation_indices(1000, n_batches=2, batch_size=10, sampling_mode="uniform")
    assert len(selected) == 20
    assert selected[0] == 0
    assert selected[-1] == 999

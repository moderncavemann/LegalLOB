import numpy as np

from scripts.export_difflob_dataset import (
    build_regime_arrays,
    resolve_dataset_dir,
    to_difflob_tensor,
    valid_window_starts,
)


def test_difflob_level_orientation():
    raw = np.zeros((1, 2, 20), dtype=np.float64)
    raw[0, 0] = np.arange(20)
    raw[0, 1] = np.arange(100, 120)
    converted = to_difflob_tensor(raw)
    np.testing.assert_array_equal(converted[0, :10, 0], np.arange(9, -1, -1))
    np.testing.assert_array_equal(converted[0, 10:, 0], np.arange(10, 20))
    np.testing.assert_array_equal(converted[0, :10, 2], np.zeros(10))
    np.testing.assert_array_equal(converted[0, 10:, 2], np.ones(10))


def test_window_starts_do_not_cross_segments():
    split = np.zeros(10, dtype=np.int8)
    day = np.ones(10, dtype=np.int32)
    segment = np.asarray([0] * 5 + [1] * 5, dtype=np.int32)
    starts = valid_window_starts(split, day, segment, total_window=4, step=1)
    assert starts.tolist() == [0, 1, 5, 6]


def test_regime_arrays_match_difflob_shape():
    raw = np.zeros((8, 2, 20), dtype=np.float64)
    for t in range(8):
        raw[t, 0, :10] = 101 + t
        raw[t, 0, 10:] = 99 + t
        raw[t, 1] = 10 + t
    lob = to_difflob_tensor(raw)
    result = build_regime_arrays(lob, np.asarray([0]), past_window=4, predict_window=4)
    assert result["trend"].shape == (1,)
    assert result["future_liquidity"].shape == (1, 4)
    assert result["future_imbalance"].shape == (1, 4)


def test_resolve_dataset_dir_accepts_direct_root(tmp_path):
    dataset = tmp_path / "EXAMPLE"
    dataset.mkdir()
    (dataset / "states_raw.npy").touch()
    (dataset / "metadata.json").write_text("{}")
    assert resolve_dataset_dir(tmp_path, "EXAMPLE") == dataset

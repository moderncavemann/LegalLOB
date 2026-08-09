import numpy as np

from scripts.summarize_boundary_aware_pilot import summarize_method


def _legal_fixture() -> np.ndarray:
    ask = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float32)
    bid = np.array([0, -1, -3, -4, -5, -6, -7, -8, -9, -10], dtype=np.float32)
    prices = np.concatenate([ask, bid])
    state = np.stack([prices, np.ones(20, dtype=np.float32)], axis=0)
    return np.broadcast_to(state[None, :, :, None], (2, 2, 20, 2)).copy()


def test_summarize_method_reports_exact_fixture_match():
    real = _legal_fixture()
    result = summarize_method(
        real.copy(),
        real,
        {"min_spread_normalized": 1.0, "min_level_gap_normalized": 1.0},
    )
    assert result["overall_mse"] == 0.0
    assert result["price_mse"] == 0.0
    assert result["invalid_snapshot_rate"] == 0.0
    assert result["gap_boundary_calibration_abs_error"] == 0.0
    assert result["conditional_interior_gap_w1"] == 0.0

import numpy as np

from scripts.audit_saved_generation import (
    continuous_boundary_stats,
    encode_legal_numpy,
    error_breakdown,
    repair_distribution,
)


def _legal_fixture() -> np.ndarray:
    lob = np.zeros((2, 2, 20, 3), dtype=np.float32)
    lob[:, 0, :10] = np.arange(1, 11, dtype=np.float32)[None, :, None]
    lob[:, 0, 10:] = -np.arange(0, 10, dtype=np.float32)[None, :, None]
    lob[:, 1] = 1.0
    return lob


def test_error_breakdown_separates_price_and_volume() -> None:
    real = _legal_fixture()
    generated = real.copy()
    generated[:, 0] += 0.2
    generated[:, 1] += 0.5
    stats = error_breakdown(
        generated,
        real,
        {
            "min_level_gap_normalized": 0.1,
            "model_price_unit_raw": 0.01,
            "volume_scale": 2.0,
        },
    )
    assert np.isclose(stats["price"]["mse"], 0.04)
    assert np.isclose(stats["volume"]["mse"], 0.25)
    assert np.isclose(stats["price"]["mae_ticks"], 2.0)
    assert np.isclose(stats["price_semantics"]["mid"]["mae_ticks"], 2.0)
    assert np.isclose(stats["price_semantics"]["spread"]["mae_ticks"], 0.0, atol=1e-6)
    assert len(stats["by_horizon"]) == 3


def test_repair_distribution_reports_modified_tick_quantiles() -> None:
    raw = _legal_fixture()
    projected = raw.copy()
    projected[:, 0, 0, 0] += 0.2
    stats = repair_distribution(
        raw,
        projected,
        {"min_level_gap_normalized": 0.1},
        tolerance=1e-8,
    )
    assert stats["any_snapshot_modified_rate"] == 1.0 / 3.0
    assert np.isclose(stats["ticks_moved_modified_price_levels"]["p50"], 2.0)
    for threshold in ("0.1", "0.5", "1"):
        assert np.isclose(
            stats["tick_threshold_exceedance"][threshold]["snapshot_rate"],
            1.0 / 3.0,
        )
        assert np.isclose(
            stats["tick_threshold_exceedance"][threshold]["price_level_rate"],
            1.0 / 60.0,
        )
    assert stats["volume_level_modified_rate"] == 0.0


def test_legal_encoding_and_boundary_audit_are_finite() -> None:
    lob = _legal_fixture()
    encoded = encode_legal_numpy(lob, min_spread=0.1, min_gap=0.1)
    boundary = continuous_boundary_stats(lob, min_spread=0.1, min_gap=0.1)
    assert encoded.shape == lob.shape
    assert np.isfinite(encoded).all()
    assert boundary["spread_at_minimum_rate"] == 0.0
    assert boundary["zero_volume_rate"] == 0.0

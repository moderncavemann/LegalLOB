import numpy as np
import pytest

from scripts.summarize_financial_sensitivity import (
    admissible_snapshot_mask,
    functional_vectors,
    summarize_output,
    summarize_paired_repair,
)


NORMALIZATION = {
    "ticks_per_price_unit": 10.0,
    "volume_scale": 15.0,
    "min_spread_normalized": 0.1,
    "min_level_gap_normalized": 0.1,
}


def make_book() -> np.ndarray:
    book = np.zeros((1, 2, 20, 1), dtype=np.float32)
    book[0, 0, :10, 0] = 0.05 + 0.1 * np.arange(10)
    book[0, 0, 10:, 0] = -0.05 - 0.1 * np.arange(10)
    # Inverse-transforms to one clipped share at every level.
    book[0, 1, :, 0] = 1.0 / 15.0
    return book


def test_identical_legal_books_have_zero_functional_distance():
    book = make_book()
    result, _, _, target = summarize_output(book, book, NORMALIZATION)

    assert target == pytest.approx(1.0)
    assert result["admissible_snapshot_rate"] == 1.0
    assert result["microprice_offset_w1_ticks"] == 0.0
    assert result["imbalance_l10_w1"] == 0.0
    assert result["market_order_slippage_w1_ticks"] == 0.0
    assert result["liquidity_5tick_w1_target_units"] == 0.0


def test_inverted_ask_level_is_inadmissible_and_repair_changes_slippage():
    projected = make_book()
    raw = make_book()
    raw[0, 0, 1, 0] = raw[0, 0, 0, 0] - 0.1

    assert not admissible_snapshot_mask(raw, NORMALIZATION)[0]
    assert admissible_snapshot_mask(projected, NORMALIZATION)[0]

    raw_vectors, _ = functional_vectors(raw, NORMALIZATION, target_size=1.5)
    assert np.nanmin(raw_vectors["market_order_slippage"]) < 0

    repair = summarize_paired_repair(
        raw, projected, NORMALIZATION, target_size=1.5
    )
    assert repair["repair_market_order_slippage_abs_shift_ticks"] > 0
    assert repair["repair_raw_negative_slippage_rate"] > 0
    assert repair["repair_final_negative_slippage_rate"] == 0

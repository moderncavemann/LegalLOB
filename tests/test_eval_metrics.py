import numpy as np
import pytest
import torch

from genlob.eval.metrics import (
    discrete_lob_validity,
    horizon_fidelity,
    histogram_distances,
    ks_2sample,
    per_sample_metrics,
    projection_repair_details,
    structural_validity_details,
)
from genlob.utils.physics import project_to_valid_lob
from scripts.summarize_ensemble_generation import _ensemble_metrics


def _valid_tick_grid_book(batch: int = 2, horizon: int = 4) -> torch.Tensor:
    book = torch.zeros(batch, 2, 20, horizon, dtype=torch.float64)
    ask = torch.arange(1, 11, dtype=torch.float64) * 0.1
    bid = -torch.arange(0, 10, dtype=torch.float64) * 0.1
    book[:, 0, :10, :] = ask[None, :, None]
    book[:, 0, 10:, :] = bid[None, :, None]
    book[:, 1, :, :] = 2.0 / 15.0  # inverse sqrt transform is exactly 4 shares
    return book


def _normalization() -> dict:
    return {
        "tick_size_raw": 0.01,
        "model_price_unit_raw": 0.1,
        "volume_transform": "sqrt_clip_train_quantile_scale",
        "volume_scale": 15.0,
    }


def test_distribution_distances_identical_and_shifted():
    values = np.array([-1.0, 0.0, 1.0, 2.0])
    assert ks_2sample(values, values) == pytest.approx(0.0)
    assert histogram_distances(values, values)["js"] == pytest.approx(0.0)
    assert ks_2sample(values, values + 10.0) == pytest.approx(1.0)


def test_horizon_fidelity_reports_persistence_skill():
    real = torch.zeros(2, 2, 20, 3)
    persistence = torch.ones_like(real)
    result_equal = horizon_fidelity(real, persistence, persistence)
    assert result_equal["persistence_skill_score"] == pytest.approx([0.0, 0.0, 0.0])

    better = torch.full_like(real, 0.5)
    result_better = horizon_fidelity(real, better, persistence)
    assert result_better["persistence_skill_score"] == pytest.approx([0.75, 0.75, 0.75])


def test_discrete_validity_distinguishes_grid_and_continuous_outputs():
    book = _valid_tick_grid_book()
    valid = discrete_lob_validity(book, _normalization())
    assert valid["available"] is True
    assert valid["off_tick_spread_rate"] == 0.0
    assert valid["off_tick_gap_rate"] == 0.0
    assert valid["off_tick_internal_price_rate"] == 0.0
    assert valid["sub_tick_spread_rate"] == 0.0
    assert valid["sub_tick_gap_rate"] == 0.0
    assert valid["noninteger_volume_rate"] == 0.0

    continuous = book.clone()
    continuous[0, 0, 1, 0] += 0.02
    continuous[0, 1, 0, 0] += 0.001
    invalid = discrete_lob_validity(continuous, _normalization())
    assert invalid["off_tick_gap_rate"] > 0.0
    assert invalid["off_tick_internal_price_rate"] > 0.0
    assert invalid["sub_tick_gap_rate"] > 0.0
    assert invalid["noninteger_volume_rate"] > 0.0


def test_discrete_validity_is_explicitly_unavailable_without_scale_metadata():
    result = discrete_lob_validity(_valid_tick_grid_book(), {})
    assert result["available"] is False


def test_structural_and_projection_repair_details_are_snapshot_aware():
    book = _valid_tick_grid_book()
    book[0, 0, 0, 0] = -0.1
    details = structural_validity_details(book, min_spread=0.1, min_gap=0.1)
    assert details["any_invalid_snapshot_rate"] == pytest.approx(1 / 8)
    assert details["any_invalid_trajectory_rate"] == pytest.approx(1 / 2)
    assert details["max_spread_shortfall"] > 0

    projected = project_to_valid_lob(book, min_spread=0.1, min_gap=0.1, use_gpu_approx=False)
    repair = projection_repair_details(book, projected, _normalization())
    assert repair["any_snapshot_modified_rate"] == pytest.approx(1 / 8)
    assert repair["price_level_modified_rate"] > 0
    assert repair["mean_ticks_moved_modified_levels"] > 0


def test_per_sample_legality_uses_same_floating_tolerance_as_aggregate_metrics():
    book = _valid_tick_grid_book().float()
    # Simulate harmless roundoff from an exact projection/decode operation.
    rounded = book.clone()
    rounded[:, 0, 1:, :] -= 5e-7

    sample = per_sample_metrics(
        real=book,
        raw=rounded,
        projected=rounded,
        min_spread=0.1,
        min_gap=0.1,
    )

    assert np.all(sample["crossed_book_rate"] == 0.0)
    assert np.all(sample["ask_ladder_violation"] == 0.0)
    assert np.all(sample["bid_ladder_violation"] == 0.0)
    assert np.all(sample["negative_volume_rate"] == 0.0)


def test_sampling_ensemble_scores_perfect_forecast():
    real = np.zeros((3, 2, 20, 4), dtype=np.float32)
    samples = np.stack([real, real, real])
    persistence = np.ones_like(real)
    summary, per_sample = _ensemble_metrics(samples, real, persistence)
    assert summary["ensemble_mean_mse"] == 0.0
    assert summary["energy_score"] == 0.0
    assert summary["scalar_crps"] == 0.0
    assert summary["minmax_element_coverage"] == 1.0
    assert summary["ensemble_mean_persistence_skill"] == 1.0
    assert np.all(per_sample["ensemble_mean_mse"] == 0.0)

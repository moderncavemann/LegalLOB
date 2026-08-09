import numpy as np

from scripts.summarize_conditional_scores import (
    interval_statistics,
    state_energy_and_crps,
    trajectory_features,
    variogram_score,
)


def test_energy_score_is_zero_for_perfect_deterministic_forecast() -> None:
    target = np.zeros((3, 2, 20, 10), dtype=np.float32)
    samples = target[None]
    scores = state_energy_and_crps(samples, target)
    assert np.allclose(scores["energy_score"], 0.0)
    assert np.allclose(scores["marginal_crps"], 0.0)


def test_interval_statistics_reports_nominal_fixture_coverage() -> None:
    samples = np.asarray([0.0, 1.0, 2.0, 3.0])[:, None, None, None, None]
    target = np.asarray([1.5])[None, None, None, :]
    result = interval_statistics(samples, target)
    assert result["coverage_50"] == 1.0
    assert result["coverage_90"] == 1.0


def test_temporal_feature_shape_and_perfect_variogram() -> None:
    target = np.zeros((4, 2, 20, 10), dtype=np.float32)
    target[:, 0, :10] = 1.0
    target[:, 0, 10:] = -1.0
    target[:, 1] = 1.0
    samples = target[None]
    assert trajectory_features(target).shape == (4, 29)
    assert np.allclose(variogram_score(samples, target), 0.0)

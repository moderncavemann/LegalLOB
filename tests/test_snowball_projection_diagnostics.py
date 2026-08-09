from scripts.snowball_test import merge_rollout_projection_diagnostics


MEAN_FIELDS = (
    "per_call_projection_l2_mean",
    "cumulative_projection_l2_mean",
    "pre_projection_crossed_book_rate_mean",
    "pre_projection_ask_ladder_violation_mean",
    "pre_projection_bid_ladder_violation_mean",
    "pre_projection_negative_volume_rate_mean",
    "pre_projection_any_invalid_snapshot_rate_mean",
)
MAX_FIELDS = (
    "per_call_projection_l2_max",
    "cumulative_projection_l2_max",
    "pre_projection_crossed_book_rate_max",
    "pre_projection_ask_ladder_violation_max",
    "pre_projection_bid_ladder_violation_max",
    "pre_projection_negative_volume_rate_max",
    "pre_projection_any_invalid_snapshot_rate_max",
)


def _path(calls: int, mean: float, maximum: float) -> dict:
    item = {"n_projection_calls": calls}
    item.update({field: mean for field in MEAN_FIELDS})
    item.update({field: maximum for field in MAX_FIELDS})
    return item


def test_merge_rollout_projection_diagnostics_preserves_horizon_prefixes():
    parts = [
        {
            "n_trajectories": 2,
            "sampler_projection": _path(20, 2.0, 4.0),
            "posthoc_chunk_projection": None,
            "by_horizon": {
                "step_10": {
                    "sampler_projection": _path(2, 0.5, 1.0),
                    "posthoc_chunk_projection": None,
                },
                "step_20": {
                    "sampler_projection": _path(4, 1.0, 2.0),
                    "posthoc_chunk_projection": None,
                },
            },
        },
        {
            "n_trajectories": 1,
            "sampler_projection": _path(20, 5.0, 8.0),
            "posthoc_chunk_projection": None,
            "by_horizon": {
                "step_10": {
                    "sampler_projection": _path(2, 2.0, 3.0),
                    "posthoc_chunk_projection": None,
                },
                "step_20": {
                    "sampler_projection": _path(4, 4.0, 6.0),
                    "posthoc_chunk_projection": None,
                },
            },
        },
    ]

    merged = merge_rollout_projection_diagnostics(parts)

    assert merged["sampler_projection"]["n_projection_calls_per_trajectory"] == 20
    assert merged["sampler_projection"]["per_call_projection_l2_mean"] == 3.0
    assert merged["sampler_projection"]["per_call_projection_l2_max"] == 8.0
    assert list(merged["by_horizon"]) == ["step_10", "step_20"]
    step_10 = merged["by_horizon"]["step_10"]["sampler_projection"]
    step_20 = merged["by_horizon"]["step_20"]["sampler_projection"]
    assert step_10["n_projection_calls_per_trajectory"] == 2
    assert step_20["n_projection_calls_per_trajectory"] == 4
    assert step_10["per_call_projection_l2_mean"] == 1.0
    assert step_20["per_call_projection_l2_mean"] == 2.0

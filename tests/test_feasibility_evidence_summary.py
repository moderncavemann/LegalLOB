import numpy as np

from scripts.summarize_feasibility_evidence import (
    acf_integrated_absolute_error,
    depth_repair_severity,
    hierarchical_aggregate,
)


def test_acf_integrated_absolute_error_uses_only_finite_saved_lags() -> None:
    facts = {
        "acf": {
            "spread": {
                "1": {"real": 0.8, "generated": 0.5},
                "2": {"real": 0.6, "generated": 0.4},
                "5": {"real": -0.1, "generated": 0.0},
                "10": {"real": None, "generated": None},
            }
        }
    }
    assert np.isclose(acf_integrated_absolute_error(facts, "spread"), 0.6)


def test_hierarchical_aggregate_does_not_treat_eval_seeds_as_training_seeds() -> None:
    rows = []
    for train_seed, center in enumerate((1.0, 2.0, 4.0)):
        for eval_seed, offset in enumerate((-0.1, 0.0, 0.1)):
            rows.append(
                {
                    "dataset": "fixture",
                    "method": "legal",
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "metric": center + offset,
                }
            )

    result = hierarchical_aggregate(
        rows,
        fields=("metric",),
        group_fields=("dataset", "method"),
    )[0]

    expected_centers = np.asarray([1.0, 2.0, 4.0])
    assert result["n_training_seeds"] == 3
    assert result["n_eval_seeds_per_training_seed"] == [3]
    assert np.isclose(result["metric_mean"], expected_centers.mean())
    assert np.isclose(result["metric_std"], expected_centers.std(ddof=1))


def test_depth_repair_severity_is_frame_and_level_aware() -> None:
    raw = np.zeros((1, 2, 20, 2), dtype=np.float32)
    projected = raw.copy()
    raw[0, 1, 0, 0] = -0.2
    raw[0, 1, 1, 0] = -0.4

    result = depth_repair_severity(
        raw,
        projected,
        tolerance=1e-8,
        volume_scale=10.0,
    )

    assert np.isclose(result["repair_depth_invalid_snapshot_rate"], 0.5)
    assert np.isclose(result["repair_depth_level_clamped_rate"], 2 / 40)
    assert np.isclose(result["repair_depth_norm_p50"], 0.3)
    assert np.isclose(result["repair_depth_share_equiv_p95"], 15.4)

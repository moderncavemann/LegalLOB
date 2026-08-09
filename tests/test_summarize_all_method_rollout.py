import json

from scripts.summarize_all_method_rollout import METHOD_LABELS, collect_rows


def test_collect_rows_reports_generated_lob_steps_per_second(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({"eval_seed": 2}))
    snapshot = {
        "violations_raw": {
            "any_invalid_snapshot_rate": 0.0,
            "crossed_book_rate": 0.0,
            "ask_ladder_violation": 0.0,
            "bid_ladder_violation": 0.0,
            "negative_volume_rate": 0.0,
        },
        "projection_distance": {"total_l2_distance": 0.0},
        "spread_mean": 1.0,
        "spread_real_mean": 1.0,
        "volume_mean": 1.0,
        "volume_real_mean": 1.0,
        "mid_return_std": 1.0,
        "mid_return_real_std": 1.0,
        "temporal_corr": 0.0,
    }
    stats = {
        "config": {"horizon": 100, "n_histories": 640},
        "runtime": {"rollout_seconds": 50.0, "trajectories_per_second": 12.8},
        "generated": {f"step_{h}": snapshot for h in (10, 20, 50, 100)},
    }
    for method_id in METHOD_LABELS:
        output = tmp_path / "train_seed0" / method_id
        output.mkdir(parents=True)
        (output / "snowball_stats.json").write_text(json.dumps(stats))

    rows = collect_rows(tmp_path)

    assert len(rows) == 24
    assert all(row["eval_seed"] == 2 for row in rows)
    assert all(row["lob_steps_per_second"] == 1280.0 for row in rows)

import json
import os

from scripts.summarize_runtime_cost import (
    collect_data_build_span,
    collect_evaluation_rows,
    collect_snowball_rows,
    collect_training_rows,
)


def _write_json(path, payload, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_collect_training_rows_uses_direct_epoch_timers(tmp_path):
    root = tmp_path / "checkpoints"
    method_dir = root / "AAPL" / "train_seed0" / "raw"
    _write_json(
        method_dir / "history.json",
        {"epochs": [{"elapsed_seconds": 10.0}, {"elapsed_seconds": 14.0}]},
    )
    _write_json(method_dir / "completion.json", {"status": "complete"})

    rows, warnings = collect_training_rows(root, methods=["raw"])

    assert warnings == []
    assert len(rows) == 1
    assert rows[0]["total_epoch_seconds"] == 24.0
    assert rows[0]["mean_epoch_seconds"] == 12.0
    assert rows[0]["measurement_source"] == "direct_train_epoch_timer"


def test_collect_evaluation_rows_recovers_sequential_artifact_intervals(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    results = tmp_path / "results"
    completion = checkpoints / "AAPL" / "train_seed0" / "raw" / "completion.json"
    _write_json(completion, {"status": "complete"}, mtime=100.0)

    first = results / "AAPL" / "train_seed0" / "raw_no_projection" / "metrics.json"
    second = results / "AAPL" / "train_seed0" / "raw_posthoc_pava" / "metrics.json"
    payload = {"evaluation_protocol": {"selected_windows": 4}}
    _write_json(first, payload, mtime=110.0)
    _write_json(second, payload, mtime=122.0)

    rows, warnings = collect_evaluation_rows(
        results,
        checkpoints,
        eval_seeds=[0],
        methods=["raw_no_projection", "raw_posthoc_pava"],
        frames_per_trajectory=10,
    )

    assert any("missing baseline artifact" in warning for warning in warnings)
    assert [row["seconds"] for row in rows] == [10.0, 12.0]
    assert rows[0]["milliseconds_per_trajectory"] == 2500.0
    assert rows[0]["snapshots"] == 40


def test_collect_snowball_rows_uses_last_bootstrap_as_start_boundary(tmp_path):
    results = tmp_path / "results"
    seed_dir = results / "AAPL" / "train_seed0"
    _write_json(seed_dir / "block_bootstrap_raw_vs_legal.json", {}, mtime=200.0)
    stats = seed_dir / "legal_snowball_100_replay" / "snowball_stats.json"
    _write_json(
        stats,
        {"config": {"n_histories": 8, "horizon": 100}},
        mtime=260.0,
    )

    rows, warnings = collect_snowball_rows(results, snowball_seeds=[0])

    assert warnings == []
    assert rows[0]["seconds"] == 60.0
    assert rows[0]["snapshots"] == 800
    assert rows[0]["milliseconds_per_trajectory"] == 7500.0


def test_collect_data_build_span_uses_last_canonical_restart(tmp_path):
    log = tmp_path / "build.log"
    log.write_text(
        "[2026-07-15T13:00:00+10:00] Restarting canonical panel.\n"
        "[2026-07-15T14:16:18+10:00] Restarting canonical panel as final build.\n"
    )
    artifact = tmp_path / "build_report.json"
    _write_json(artifact, {}, mtime=1784091873.0)

    result = collect_data_build_span(log, artifact)

    assert result["started_at_utc"] == "2026-07-15T04:16:18+00:00"
    assert result["finished_at_utc"] == "2026-07-15T05:04:33+00:00"
    assert result["seconds"] > 0
    assert "January parse reused" in result["scope"]

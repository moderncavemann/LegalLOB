import json
from pathlib import Path

import numpy as np
import pytest

from scripts.summarize_public_itch_by_day import (
    IntegrityError,
    collect_rows,
    hierarchical_aggregate,
    worst_day_aggregate,
    write_outputs,
)


DAYS = (20200101, 20200102)
NORMALIZATION = {
    "ticks_per_price_unit": 10.0,
    "volume_scale": 15.0,
    "min_spread_normalized": 0.1,
    "min_level_gap_normalized": 0.1,
}


def _book(depth: float = 2.0 / 15.0) -> np.ndarray:
    value = np.zeros((2, 20), dtype=np.float32)
    value[0, :10] = 0.05 + 0.1 * np.arange(10)
    value[0, 10:] = -0.05 - 0.1 * np.arange(10)
    value[1] = depth
    return value


def _result_dir(root: Path, train_seed: int, eval_seed: int, method: str) -> Path:
    base = root / "AAPL" / f"train_seed{train_seed}"
    if eval_seed:
        base = base / "seeds" / f"seed{eval_seed}"
    return base / method


def _make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    results_root = tmp_path / "results"
    data_root = tmp_path / "data"
    dataset = data_root / "PUBLIC_ITCH_2019_SAMPLES_AAPL_1s"
    dataset.mkdir(parents=True)
    states = np.stack([_book() for _ in range(8)])
    np.save(dataset / "states.npy", states)
    np.save(dataset / "split.npy", np.full(8, 2, dtype=np.int8))
    np.save(dataset / "day_id.npy", np.asarray([DAYS[0]] * 4 + [DAYS[1]] * 4))
    np.save(dataset / "segment_id.npy", np.asarray([0] * 4 + [1] * 4))
    (dataset / "normalization.json").write_text(json.dumps(NORMALIZATION))

    real = np.stack([_book(), _book()])[..., None]
    starts = np.asarray([0, 4], dtype=np.int64)
    sample_common = {
        "dataset_index": np.asarray([0, 1], dtype=np.int64),
        "start_indices": starts,
        "day_id": np.asarray(DAYS, dtype=np.int64),
        "segment_id": np.asarray([0, 1], dtype=np.int64),
    }
    contracts = {
        "legal_no_projection": ("legal_lob", "none"),
        "matched_no_projection": ("matched_lob", "none"),
        "matched_posthoc_pava": ("matched_lob", "exact"),
    }
    for train_seed in (0, 1):
        for eval_seed in (0, 1):
            for method, (representation, projection_mode) in contracts.items():
                path = _result_dir(results_root, train_seed, eval_seed, method)
                path.mkdir(parents=True)
                delta = 0.001 * (1 + train_seed + eval_seed)
                raw = real.copy()
                raw[:, 1] += delta
                projected = real.copy() if method.startswith("matched") else raw.copy()
                np.save(path / "generated_raw.npy", raw)
                np.save(path / "generated_projected.npy", projected)
                np.save(path / "real_future.npy", real)
                np.savez_compressed(
                    path / "per_sample_metrics.npz",
                    **sample_common,
                    projection_l2=np.full(2, delta, dtype=np.float64),
                )
                (path / "metrics.json").write_text(
                    json.dumps(
                        {
                            "representation": representation,
                            "projection_mode": projection_mode,
                        }
                    )
                )
    return results_root, data_root


def _collect(tmp_path: Path):
    results_root, data_root = _make_fixture(tmp_path)
    values = collect_rows(
        results_root,
        data_root,
        ("AAPL",),
        DAYS,
        ("legal_no_projection", "matched_posthoc_pava"),
        (0, 1),
        (0, 1),
        1,
        1,
    )
    return results_root, data_root, values


def test_collect_rows_freezes_full_test_target_and_aggregates_eval_first(tmp_path):
    _, _, (rows, targets, integrity, source_files) = _collect(tmp_path)

    assert targets["AAPL"] == pytest.approx(4.0)
    assert len(rows) == 16
    assert integrity["status"] == "passed"
    assert integrity["expected_input_cells"] == 8
    assert source_files

    summary = hierarchical_aggregate(rows, (0, 1), (0, 1))
    assert len(summary) == 4
    legal_day = next(
        row
        for row in summary
        if row["method_id"] == "legal_no_projection" and row["day_id"] == DAYS[0]
    )
    per_train = []
    for train_seed in (0, 1):
        values = [
            row["output_mse"]
            for row in rows
            if row["method_id"] == "legal_no_projection"
            and row["day_id"] == DAYS[0]
            and row["train_seed"] == train_seed
        ]
        per_train.append(np.mean(values))
    assert legal_day["output_mse_mean"] == pytest.approx(np.mean(per_train))
    assert legal_day["output_mse_std"] == pytest.approx(np.std(per_train, ddof=1))

    worst = worst_day_aggregate(rows, (0, 1), (0, 1))
    assert len(worst) == 2
    assert all("output_mse_worst_mean" in row for row in worst)


def test_collect_rows_rejects_unpaired_history_metadata(tmp_path):
    results_root, data_root = _make_fixture(tmp_path)
    broken = (
        results_root
        / "AAPL/train_seed0/matched_posthoc_pava/per_sample_metrics.npz"
    )
    with np.load(broken, allow_pickle=False) as payload:
        values = {key: np.asarray(payload[key]) for key in payload.files}
    values["start_indices"] = np.asarray([1, 4], dtype=np.int64)
    np.savez_compressed(broken, **values)

    with pytest.raises(IntegrityError, match="paired history metadata mismatch"):
        collect_rows(
            results_root,
            data_root,
            ("AAPL",),
            DAYS,
            ("legal_no_projection", "matched_posthoc_pava"),
            (0, 1),
            (0, 1),
            1,
            1,
        )


def test_matched_no_projection_uses_unrepaired_raw_output(tmp_path):
    results_root, data_root = _make_fixture(tmp_path)
    rows, _, integrity, _ = collect_rows(
        results_root,
        data_root,
        ("AAPL",),
        DAYS,
        ("matched_no_projection",),
        (0, 1),
        (0, 1),
        1,
        1,
    )

    assert len(rows) == 8
    assert integrity["expected_input_cells"] == 4
    assert all(row["output_mse"] > 0.0 for row in rows)


def test_collect_rows_accepts_legacy_singular_start_index_when_paired(tmp_path):
    results_root, data_root = _make_fixture(tmp_path)
    legacy = results_root / "AAPL/train_seed0/legal_no_projection/per_sample_metrics.npz"
    with np.load(legacy, allow_pickle=False) as payload:
        values = {key: np.asarray(payload[key]) for key in payload.files}
    values["start_index"] = values.pop("start_indices")
    np.savez_compressed(legacy, **values)

    rows, _, integrity, _ = collect_rows(
        results_root,
        data_root,
        ("AAPL",),
        DAYS,
        ("legal_no_projection", "matched_posthoc_pava"),
        (0, 1),
        (0, 1),
        1,
        1,
    )

    assert len(rows) == 16
    assert integrity["status"] == "passed"


def test_collect_rows_normalizes_integral_pairing_dtypes(tmp_path):
    results_root, data_root = _make_fixture(tmp_path)
    path = results_root / "AAPL/train_seed0/legal_no_projection/per_sample_metrics.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = {key: np.asarray(payload[key]) for key in payload.files}
    values["day_id"] = values["day_id"].astype(np.int32)
    values["segment_id"] = values["segment_id"].astype(np.int32)
    np.savez_compressed(path, **values)

    rows, _, integrity, _ = collect_rows(
        results_root,
        data_root,
        ("AAPL",),
        DAYS,
        ("legal_no_projection", "matched_posthoc_pava"),
        (0, 1),
        (0, 1),
        1,
        1,
    )

    assert len(rows) == 16
    assert integrity["status"] == "passed"


def test_write_outputs_emits_schema_and_refuses_overwrite(tmp_path):
    _, _, (rows, targets, integrity, source_files) = _collect(tmp_path)
    day_summary = hierarchical_aggregate(rows, (0, 1), (0, 1))
    worst_summary = worst_day_aggregate(rows, (0, 1), (0, 1))
    output = tmp_path / "audit"
    write_outputs(
        output,
        rows,
        day_summary,
        worst_summary,
        targets,
        integrity,
        source_files,
        {"fixture": True},
        "fixture command",
    )
    assert {path.name for path in output.iterdir()} == {
        "run_manifest.json",
        "integrity_report.json",
        "per_cell_day_metrics.csv",
        "hierarchical_day_summary.csv",
        "worst_day_summary.csv",
        "summary.json",
        "summary.md",
    }
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_outputs(
            output,
            rows,
            day_summary,
            worst_summary,
            targets,
            integrity,
            source_files,
            {"fixture": True},
            "fixture command",
        )

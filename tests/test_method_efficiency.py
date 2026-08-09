import pytest

from scripts.benchmark_method_efficiency import derive_comparisons, summarize_records


def _row(method, total_ms, repair_ms=None):
    return {
        "method": method,
        "preprocess_ms": 1.0,
        "sampling_ms": total_ms - 2.0 - (repair_ms or 0.0),
        "decode_transfer_ms": 1.0,
        "repair_ms": repair_ms,
        "total_ms": total_ms,
        "projection_location": "post_hoc" if repair_ms is not None else "none",
        "projection_calls": 1 if repair_ms is not None else 0,
    }


def test_summarize_records_reports_paired_projection_overhead_and_throughput():
    records = [
        _row("raw_no_projection", 100.0),
        _row("raw_no_projection", 120.0),
        _row("raw_posthoc_pava", 110.0, repair_ms=10.0),
        _row("raw_posthoc_pava", 132.0, repair_ms=12.0),
        _row("legal_no_projection", 115.0),
        _row("legal_no_projection", 117.0),
    ]

    summary = summarize_records(records, batch_size=10, frames_per_trajectory=5)
    by_method = {row["method"]: row for row in summary}

    assert by_method["raw_no_projection"]["mean_total_ms_per_batch"] == 110.0
    assert by_method["raw_posthoc_pava"]["mean_repair_ms_per_batch"] == 11.0
    assert by_method["raw_posthoc_pava"]["relative_to_raw"] == 1.1
    assert by_method["raw_no_projection"]["trajectories_per_second"] == 1000 / 11
    assert by_method["legal_no_projection"]["milliseconds_per_trajectory"] == 11.6


def test_derive_comparisons_uses_protocol_snapshot_count():
    summary = [
        {"method": "raw_no_projection", "mean_total_ms_per_batch": 100.0},
        {
            "method": "raw_posthoc_pava",
            "mean_total_ms_per_batch": 110.0,
            "mean_repair_ms_per_batch": 10.0,
        },
        {"method": "raw_inloop_pava", "mean_total_ms_per_batch": 180.0},
        {"method": "legal_no_projection", "mean_total_ms_per_batch": 101.0},
    ]

    result = derive_comparisons(summary, batch_size=10, frames_per_trajectory=5)

    assert result["legal_overhead_vs_raw_percent"] == pytest.approx(1.0)
    assert result["posthoc_overhead_vs_raw_percent"] == pytest.approx(10.0)
    assert result["posthoc_repair_microseconds_per_snapshot"] == 200.0

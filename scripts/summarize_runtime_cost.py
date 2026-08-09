#!/usr/bin/env python3
"""Summarize measured runtime and cost proxies for the frozen snapshot panel.

The script keeps three evidence classes separate:

1. direct epoch timers stored by ``train_v2.py``;
2. end-to-end artifact-boundary intervals from the sequential ablation runner;
3. direct ``perf_counter`` CPU benchmarks for the exact PAVA kernel.

Artifact-boundary intervals use filesystem timestamps and include process
startup, model/data loading, generation, metrics, plots, and writes.
They are useful end-to-end engineering measurements, but are not online p99
latency measurements. Monetary and energy costs are emitted only when explicit
rates are supplied by the caller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRAIN_METHODS = ("raw", "matched_lob", "legal_lob", "raw_physics")
EVAL_METHODS = (
    "raw_no_projection",
    "raw_posthoc_pava",
    "raw_inloop_pava",
    "legal_no_projection",
    "matched_no_projection",
    "matched_posthoc_pava",
    "raw_physics_no_projection",
)
METHOD_LABELS = {
    "raw": "Raw",
    "matched_lob": "Matched-coordinate",
    "legal_lob": "LegalLOB",
    "raw_physics": "Raw + soft penalty",
    "raw_no_projection": "Raw",
    "raw_posthoc_pava": "Raw + post-hoc PAVA",
    "raw_inloop_pava": "Raw + in-loop PAVA",
    "legal_no_projection": "LegalLOB",
    "matched_no_projection": "Matched-coordinate",
    "matched_posthoc_pava": "Matched-coordinate + post-hoc PAVA",
    "raw_physics_no_projection": "Raw + soft penalty",
    "persistence": "Persistence",
    "moving_block_bootstrap": "Moving-block bootstrap",
    "legal_snowball_100_replay": "LegalLOB 100-step rollout",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"n": 0}
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": _sample_std(numbers),
        "p50": statistics.median(numbers),
        "p95": _percentile(numbers, 0.95),
        "min": min(numbers),
        "max": max(numbers),
        "sum": sum(numbers),
    }


def collect_training_rows(
    checkpoint_root: Path,
    methods: Iterable[str] = TRAIN_METHODS,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    methods = tuple(methods)
    if not checkpoint_root.exists():
        return rows, [f"missing checkpoint root: {checkpoint_root}"]

    for asset_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        for seed_dir in sorted(asset_dir.glob("train_seed*")):
            try:
                train_seed = int(seed_dir.name.removeprefix("train_seed"))
            except ValueError:
                continue
            for method in methods:
                history_path = seed_dir / method / "history.json"
                completion_path = seed_dir / method / "completion.json"
                if not history_path.exists() or not completion_path.exists():
                    warnings.append(f"missing training artifact: {history_path}")
                    continue
                history = _read_json(history_path).get("epochs", [])
                completion = _read_json(completion_path)
                elapsed = [float(epoch["elapsed_seconds"]) for epoch in history]
                if not elapsed:
                    warnings.append(f"empty training history: {history_path}")
                    continue
                if completion.get("status") != "complete":
                    warnings.append(f"incomplete training marker: {completion_path}")
                    continue
                rows.append(
                    {
                        "asset": asset_dir.name,
                        "train_seed": train_seed,
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "epochs": len(elapsed),
                        "total_epoch_seconds": sum(elapsed),
                        "mean_epoch_seconds": statistics.fmean(elapsed),
                        "p50_epoch_seconds": statistics.median(elapsed),
                        "p95_epoch_seconds": _percentile(elapsed, 0.95),
                        "min_epoch_seconds": min(elapsed),
                        "max_epoch_seconds": max(elapsed),
                        "device_hours": sum(elapsed) / 3600.0,
                        "measurement_source": "direct_train_epoch_timer",
                        "scope": "training_plus_validation; excludes setup and checkpoint writes",
                        "history_path": str(history_path),
                    }
                )
    return rows, warnings


def _eval_result_dir(seed_dir: Path, eval_seed: int, method: str) -> Path:
    if eval_seed == 0:
        return seed_dir / method
    return seed_dir / "seeds" / f"seed{eval_seed}" / method


def _selected_windows(metrics: dict[str, Any]) -> int:
    protocol = metrics.get("evaluation_protocol") or {}
    return int(protocol.get("selected_windows") or 0)


def collect_evaluation_rows(
    results_root: Path,
    checkpoint_root: Path,
    eval_seeds: Iterable[int] = (0, 1, 2),
    methods: Iterable[str] = EVAL_METHODS,
    frames_per_trajectory: int = 10,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Recover sequential end-to-end intervals from immutable artifact mtimes."""

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    methods = tuple(methods)
    eval_seeds = tuple(eval_seeds)
    if not results_root.exists():
        return rows, [f"missing results root: {results_root}"]

    for asset_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        checkpoint_asset = checkpoint_root / asset_dir.name
        for seed_dir in sorted(asset_dir.glob("train_seed*")):
            try:
                train_seed = int(seed_dir.name.removeprefix("train_seed"))
            except ValueError:
                continue
            completion_paths = [
                checkpoint_asset / seed_dir.name / method / "completion.json"
                for method in TRAIN_METHODS
            ]
            existing_completions = [path for path in completion_paths if path.exists()]
            if not existing_completions:
                warnings.append(f"missing completion markers for {asset_dir.name}/{seed_dir.name}")
                continue
            previous_time = max(path.stat().st_mtime for path in existing_completions)

            for eval_seed in eval_seeds:
                for method in methods:
                    metrics_path = _eval_result_dir(seed_dir, eval_seed, method) / "metrics.json"
                    if not metrics_path.exists():
                        warnings.append(f"missing evaluation artifact: {metrics_path}")
                        continue
                    end_time = metrics_path.stat().st_mtime
                    seconds = end_time - previous_time
                    if seconds <= 0:
                        warnings.append(f"non-positive artifact interval: {metrics_path}")
                        previous_time = max(previous_time, end_time)
                        continue
                    metrics = _read_json(metrics_path)
                    trajectories = _selected_windows(metrics)
                    snapshots = trajectories * frames_per_trajectory
                    rows.append(
                        {
                            "asset": asset_dir.name,
                            "train_seed": train_seed,
                            "eval_seed": eval_seed,
                            "method": method,
                            "method_label": METHOD_LABELS.get(method, method),
                            "seconds": seconds,
                            "trajectories": trajectories,
                            "frames_per_trajectory": frames_per_trajectory,
                            "snapshots": snapshots,
                            "milliseconds_per_trajectory": (
                                seconds * 1000.0 / trajectories if trajectories else None
                            ),
                            "microseconds_per_snapshot": (
                                seconds * 1e6 / snapshots if snapshots else None
                            ),
                            "measurement_source": "sequential_artifact_boundary",
                            "scope": (
                                "process startup + model/data load + generation + metrics + plots + writes; "
                                "artifact timestamps are not an in-process synchronized profiler"
                            ),
                            "started_at_utc": _iso_utc(previous_time),
                            "finished_at_utc": _iso_utc(end_time),
                            "metrics_path": str(metrics_path),
                        }
                    )
                    previous_time = end_time

            # Deterministic and empirical baselines run after all diffusion evaluations.
            baseline_specs = [("persistence", 0)] + [
                ("moving_block_bootstrap", seed) for seed in eval_seeds
            ]
            for method, eval_seed in baseline_specs:
                metrics_path = _eval_result_dir(seed_dir, eval_seed, method) / "metrics.json"
                if not metrics_path.exists():
                    warnings.append(f"missing baseline artifact: {metrics_path}")
                    continue
                end_time = metrics_path.stat().st_mtime
                seconds = end_time - previous_time
                if seconds <= 0:
                    warnings.append(f"non-positive baseline interval: {metrics_path}")
                    previous_time = max(previous_time, end_time)
                    continue
                metrics = _read_json(metrics_path)
                trajectories = _selected_windows(metrics)
                snapshots = trajectories * frames_per_trajectory
                rows.append(
                    {
                        "asset": asset_dir.name,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "seconds": seconds,
                        "trajectories": trajectories,
                        "frames_per_trajectory": frames_per_trajectory,
                        "snapshots": snapshots,
                        "milliseconds_per_trajectory": (
                            seconds * 1000.0 / trajectories if trajectories else None
                        ),
                        "microseconds_per_snapshot": (
                            seconds * 1e6 / snapshots if snapshots else None
                        ),
                        "measurement_source": "sequential_artifact_boundary",
                        "scope": "data load + baseline generation + metrics + writes",
                        "started_at_utc": _iso_utc(previous_time),
                        "finished_at_utc": _iso_utc(end_time),
                        "metrics_path": str(metrics_path),
                    }
                )
                previous_time = end_time
    return rows, warnings


def collect_snowball_rows(
    results_root: Path,
    snowball_seeds: Iterable[int] = (0, 1, 2),
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for asset_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        for seed_dir in sorted(asset_dir.glob("train_seed*")):
            try:
                train_seed = int(seed_dir.name.removeprefix("train_seed"))
            except ValueError:
                continue
            boundary = seed_dir / "block_bootstrap_raw_vs_legal.json"
            if not boundary.exists():
                warnings.append(f"missing pre-snowball boundary: {boundary}")
                continue
            previous_time = boundary.stat().st_mtime
            for snowball_seed in snowball_seeds:
                result_dir = _eval_result_dir(
                    seed_dir, snowball_seed, "legal_snowball_100_replay"
                )
                stats_path = result_dir / "snowball_stats.json"
                if not stats_path.exists():
                    warnings.append(f"missing snowball artifact: {stats_path}")
                    continue
                end_time = stats_path.stat().st_mtime
                seconds = end_time - previous_time
                if seconds <= 0:
                    warnings.append(f"non-positive snowball interval: {stats_path}")
                    previous_time = max(previous_time, end_time)
                    continue
                payload = _read_json(stats_path)
                config = payload.get("config") or {}
                trajectories = int(config.get("n_histories") or 0)
                horizon = int(config.get("horizon") or 0)
                snapshots = trajectories * horizon
                rows.append(
                    {
                        "asset": asset_dir.name,
                        "train_seed": train_seed,
                        "eval_seed": snowball_seed,
                        "method": "legal_snowball_100_replay",
                        "method_label": "LegalLOB 100-step rollout",
                        "seconds": seconds,
                        "trajectories": trajectories,
                        "horizon": horizon,
                        "snapshots": snapshots,
                        "milliseconds_per_trajectory": (
                            seconds * 1000.0 / trajectories if trajectories else None
                        ),
                        "microseconds_per_snapshot": (
                            seconds * 1e6 / snapshots if snapshots else None
                        ),
                        "measurement_source": "sequential_artifact_boundary",
                        "scope": "model/data load + 10 chunk rollouts + metrics + plot + writes",
                        "started_at_utc": _iso_utc(previous_time),
                        "finished_at_utc": _iso_utc(end_time),
                        "stats_path": str(stats_path),
                    }
                )
                previous_time = end_time
    return rows, warnings


def collect_projection_rows(audit_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not audit_root.exists():
        return rows, [f"missing projection audit root: {audit_root}"]
    for path in sorted(audit_root.glob("*/saved_generation_audit.json")):
        payload = _read_json(path)
        runtime = payload.get("projection_runtime")
        if not runtime:
            warnings.append(f"missing projection runtime: {path}")
            continue
        seconds = runtime.get("seconds") or {}
        rows.append(
            {
                "dataset": path.parent.name,
                "device": runtime.get("device"),
                "implementation": runtime.get("implementation"),
                "n_trajectories": runtime.get("n_trajectories"),
                "frames_per_trajectory": runtime.get("frames_per_trajectory"),
                "repeats": runtime.get("repeats"),
                "mean_seconds": seconds.get("mean"),
                "std_seconds": seconds.get("std"),
                "p95_seconds": seconds.get("p95"),
                "milliseconds_per_trajectory": runtime.get("milliseconds_per_trajectory"),
                "microseconds_per_snapshot": runtime.get("microseconds_per_snapshot"),
                "measurement_source": "direct_perf_counter",
                "scope": runtime.get("scope"),
                "audit_path": str(path),
            }
        )
    return rows, warnings


def collect_data_build_span(build_log: Path, build_artifact: Path) -> dict[str, Any]:
    """Recover the bounded canonical rebuild span from a timestamped restart log.

    This is intentionally not labelled as a cold-start preprocessing time: the
    restart line states that the January parse was reused, and the historical
    download log does not contain a reliable start timestamp.
    """

    if not build_log.exists():
        raise FileNotFoundError(f"missing data build log: {build_log}")
    if not build_artifact.exists():
        raise FileNotFoundError(f"missing data build artifact: {build_artifact}")

    pattern = re.compile(r"^\[([^\]]+)\]\s+Restarting canonical panel\b")
    starts: list[datetime] = []
    for line in build_log.read_text().splitlines():
        match = pattern.search(line)
        if match:
            starts.append(datetime.fromisoformat(match.group(1)))
    if not starts:
        raise ValueError(f"no canonical restart timestamp found in {build_log}")

    start = starts[-1].timestamp()
    end = build_artifact.stat().st_mtime
    seconds = end - start
    if seconds <= 0:
        raise ValueError(
            f"non-positive data build interval: start={_iso_utc(start)}, end={_iso_utc(end)}"
        )
    return {
        "measurement_source": "timestamped_log_to_artifact_boundary",
        "started_at_utc": _iso_utc(start),
        "finished_at_utc": _iso_utc(end),
        "seconds": seconds,
        "minutes": seconds / 60.0,
        "scope": (
            "bounded canonical panel rebuild with January parse reused; "
            "not a cold-start parse or historical download time"
        ),
        "build_log": str(build_log),
        "terminal_artifact": str(build_artifact),
    }


def _aggregate_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries: list[dict[str, Any]] = []
    for method, method_rows in grouped.items():
        seconds = _stats(row["seconds"] for row in method_rows)
        total_trajectories = sum(int(row.get("trajectories") or 0) for row in method_rows)
        total_snapshots = sum(int(row.get("snapshots") or 0) for row in method_rows)
        total_seconds = float(seconds.get("sum", 0.0))
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "runs": seconds["n"],
                "mean_seconds": seconds.get("mean"),
                "std_seconds": seconds.get("std"),
                "p50_seconds": seconds.get("p50"),
                "p95_seconds": seconds.get("p95"),
                "total_device_hours": total_seconds / 3600.0,
                "milliseconds_per_trajectory": (
                    total_seconds * 1000.0 / total_trajectories
                    if total_trajectories else None
                ),
                "microseconds_per_snapshot": (
                    total_seconds * 1e6 / total_snapshots if total_snapshots else None
                ),
            }
        )
    return sorted(summaries, key=lambda row: row["method"])


def _aggregate_training(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries = []
    for method, method_rows in grouped.items():
        per_epoch = _stats(row["mean_epoch_seconds"] for row in method_rows)
        per_run = _stats(row["total_epoch_seconds"] for row in method_rows)
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "runs": len(method_rows),
                "mean_epoch_seconds": per_epoch.get("mean"),
                "std_epoch_seconds": per_epoch.get("std"),
                "mean_run_hours": float(per_run.get("mean", 0.0)) / 3600.0,
                "std_run_hours": float(per_run.get("std", 0.0)) / 3600.0,
                "total_device_hours": float(per_run.get("sum", 0.0)) / 3600.0,
            }
        )
    return sorted(summaries, key=lambda row: row["method"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
) -> None:
    training = summary["training_summary"]
    evaluation = summary["evaluation_summary"]
    projection = summary["projection_kernel"]
    snowball = summary["snowball_summary"]
    derived = summary["derived_comparisons"]
    data_build = summary["data_preparation"]["canonical_rebuild"]
    raw_train = next((row for row in training if row["method"] == "raw"), None)
    raw_eval = next((row for row in evaluation if row["method"] == "raw_no_projection"), None)

    lines = [
        "# Table: Panel Runtime Budget",
        "",
        "Hardware: Apple M3 Ultra, PyTorch 2.8.0, MPS. All panel jobs ran sequentially.",
        "",
        "This table reconstructs resource use for the frozen runner. It is not the horizontal method-efficiency benchmark because evaluation rows include loading, diagnostics, plots, and writes. Use `paper_tables/table_method_efficiency.md` for synchronized Raw/PAVA/LegalLOB comparisons.",
        "",
        "## Training",
        "",
        "Direct timers cover each epoch's training and fixed-validation passes. They exclude setup and checkpoint writes.",
        "",
        "| Method | Runs | Seconds/epoch | Hours/50-epoch run | Relative to Raw | Total device-hours |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in training:
        relative = (
            row["mean_epoch_seconds"] / raw_train["mean_epoch_seconds"]
            if raw_train and raw_train.get("mean_epoch_seconds") else None
        )
        lines.append(
            f"| {row['method_label']} | {row['runs']} | "
            f"{_fmt(row['mean_epoch_seconds'])} +/- {_fmt(row['std_epoch_seconds'])} | "
            f"{_fmt(row['mean_run_hours'])} +/- {_fmt(row['std_run_hours'])} | "
            f"{_fmt(relative, 3)}x | {_fmt(row['total_device_hours'])} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-Checkpoint Evaluation",
            "",
            "Each run evaluates 6,400 trajectories x 10 future snapshots with DDIM-20. "
            "Times are sequential artifact-boundary wall-clock intervals and include process startup, loading, metrics, plots, and writes.",
            "",
            "| Method | Runs | Seconds/run | ms/trajectory | Relative to Raw | Device-hours |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    display_order = list(EVAL_METHODS) + ["persistence", "moving_block_bootstrap"]
    by_method = {row["method"]: row for row in evaluation}
    for method in display_order:
        row = by_method.get(method)
        if not row:
            continue
        relative = (
            row["mean_seconds"] / raw_eval["mean_seconds"]
            if raw_eval and raw_eval.get("mean_seconds") else None
        )
        lines.append(
            f"| {row['method_label']} | {row['runs']} | "
            f"{_fmt(row['mean_seconds'])} +/- {_fmt(row['std_seconds'])} | "
            f"{_fmt(row['milliseconds_per_trajectory'], 3)} | {_fmt(relative, 3)}x | "
            f"{_fmt(row['total_device_hours'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Rollout and Projection",
            "",
            "| Operation | Runs | Runtime | Normalized runtime | Scope |",
            "|---|---:|---:|---:|---|",
        ]
    )
    if snowball:
        row = snowball[0]
        lines.append(
            f"| LegalLOB 100-step rollout | {row['runs']} | "
            f"{_fmt(row['mean_seconds'])} +/- {_fmt(row['std_seconds'])} s | "
            f"{_fmt(row['milliseconds_per_trajectory'], 3)} ms/trajectory | "
            "6,400 trajectories; 10 DDIM chunks; end-to-end artifact boundary |"
        )
    if projection:
        micros = _stats(row["microseconds_per_snapshot"] for row in projection)
        lines.append(
            f"| Exact post-hoc PAVA kernel | {micros['n']} datasets | "
            f"{_fmt(micros['mean'], 3)} +/- {_fmt(micros['std'], 3)} us/snapshot | "
            f"{_fmt(micros['mean'] * 10 / 1000.0, 3)} ms/10-frame trajectory | "
            "CPU kernel only; excludes denoising, I/O, and transfer |"
        )

    lines.extend(
        [
            "",
            "## Artifact-Boundary Sanity Comparisons",
            "",
            f"- LegalLOB training overhead versus Raw: `{derived['legal_training_overhead_percent']:.2f}%`.",
            f"- LegalLOB fixed-checkpoint end-to-end evaluation overhead versus Raw: "
            f"`{derived['legal_evaluation_overhead_percent']:.2f}%`.",
            f"- Current in-loop exact-PAVA evaluation cost versus Raw: "
            f"`{derived['inloop_evaluation_multiplier']:.3f}x`.",
            f"- Standalone exact post-hoc PAVA cost: "
            f"`{derived['posthoc_projection_ms_per_trajectory']:.3f}` ms per 10-frame trajectory, "
            f"equivalent to `{derived['posthoc_projection_fraction_of_raw_percent']:.2f}%` of the "
            "measured Raw end-to-end evaluation time.",
            f"- A 100-step LegalLOB rollout costs `{derived['snowball_vs_10_step_multiplier']:.2f}x` "
            "a standard 10-frame LegalLOB evaluation per trajectory.",
            "",
            "The canonical no-projection evaluator still runs exact PAVA once to compute diagnostic "
            "projection distance. Therefore, the near-equal Raw and Raw + post-hoc wall times do not "
            "isolate projection overhead; the standalone kernel benchmark is the incremental repair-cost measurement.",
            "The in-loop result reflects the current implementation, including per-step projection, device transfer, "
            "and path diagnostics; it is not a universal lower bound for optimized constrained samplers.",
        ]
    )

    panel = summary["panel_budget"]
    cost = summary["cost_model"]
    lines.extend(
        [
            "",
            "## Full Frozen Panel Budget",
            "",
            f"- Artifact wall span: `{panel['wall_hours']:.3f}` hours.",
            f"- Directly timed epoch compute: `{panel['training_device_hours']:.3f}` MPS-hours.",
            f"- Training share of panel wall time: `{derived['training_share_of_panel_percent']:.2f}%`.",
            f"- Artifact-boundary fixed-checkpoint evaluation: `{panel['evaluation_device_hours']:.3f}` MPS-hours.",
            f"- Artifact-boundary 100-step rollout: `{panel['snowball_device_hours']:.3f}` MPS-hours.",
            f"- Residual orchestration/setup/reporting time: `{panel['unattributed_hours']:.3f}` hours.",
            "",
            "## Data Preparation Timing",
            "",
            f"- Bounded canonical panel rebuild: `{data_build['minutes']:.2f}` minutes.",
            f"- Scope: {data_build['scope']}.",
            "- Historical NASDAQ downloads are marked unmeasured because the retained download log has no reliable start timestamps.",
            "",
            "## Cost Interpretation",
            "",
        ]
    )
    if cost.get("device_cost") is None:
        lines.append(
            "- Monetary compute cost is not reported: the panel ran on a local MPS machine and no hourly device rate was supplied."
        )
    else:
        lines.append(
            f"- Estimated device cost: `{cost['device_cost']:.4f}` at "
            f"`{cost['hourly_device_rate']:.4f}` currency units/hour."
        )
    if cost.get("energy_kwh") is None:
        lines.append(
            "- Energy cost is not reported: average system power and electricity tariff were not measured."
        )
    else:
        lines.append(
            f"- Scenario energy: `{cost['energy_kwh']:.4f}` kWh; electricity cost "
            f"`{cost['electricity_cost']:.4f}` currency units."
        )
    lines.extend(
        [
            "",
            "These are engineering measurements for this machine and frozen protocol, not hardware-normalized p99 deployment claims.",
            "Artifact timestamps provide end-to-end run boundaries, not an in-process synchronized profiler; use a dedicated latency harness for production p50/p95/p99.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_summary(
    checkpoint_root: Path,
    results_root: Path,
    audit_root: Path,
    panel_config: Path,
    panel_completion: Path,
    build_log: Path,
    build_artifact: Path,
    hourly_device_rate: float | None = None,
    average_system_power_watts: float | None = None,
    electricity_rate_per_kwh: float | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    training_rows, training_warnings = collect_training_rows(checkpoint_root)
    evaluation_rows, evaluation_warnings = collect_evaluation_rows(
        results_root, checkpoint_root
    )
    snowball_rows, snowball_warnings = collect_snowball_rows(results_root)
    projection_rows, projection_warnings = collect_projection_rows(audit_root)

    training_summary = _aggregate_training(training_rows)
    evaluation_summary = _aggregate_by_method(evaluation_rows)
    snowball_summary = _aggregate_by_method(snowball_rows)
    data_build_span = collect_data_build_span(build_log, build_artifact)

    training_by_method = {row["method"]: row for row in training_summary}
    evaluation_by_method = {row["method"]: row for row in evaluation_summary}
    raw_train = training_by_method["raw"]
    legal_train = training_by_method["legal_lob"]
    raw_eval = evaluation_by_method["raw_no_projection"]
    legal_eval = evaluation_by_method["legal_no_projection"]
    inloop_eval = evaluation_by_method["raw_inloop_pava"]
    snowball_eval = snowball_summary[0]
    projection_microseconds = _stats(
        row["microseconds_per_snapshot"] for row in projection_rows
    )
    projection_frames = int(projection_rows[0]["frames_per_trajectory"])
    posthoc_projection_ms = (
        float(projection_microseconds["mean"]) * projection_frames / 1000.0
    )

    start = panel_config.stat().st_mtime
    end = panel_completion.stat().st_mtime
    wall_seconds = end - start
    training_seconds = sum(row["total_epoch_seconds"] for row in training_rows)
    evaluation_seconds = sum(row["seconds"] for row in evaluation_rows)
    snowball_seconds = sum(row["seconds"] for row in snowball_rows)
    unattributed_seconds = wall_seconds - training_seconds - evaluation_seconds - snowball_seconds

    wall_hours = wall_seconds / 3600.0
    device_cost = wall_hours * hourly_device_rate if hourly_device_rate is not None else None
    energy_kwh = (
        wall_hours * average_system_power_watts / 1000.0
        if average_system_power_watts is not None else None
    )
    electricity_cost = (
        energy_kwh * electricity_rate_per_kwh
        if energy_kwh is not None and electricity_rate_per_kwh is not None else None
    )
    derived_comparisons = {
        "legal_training_overhead_percent": 100.0 * (
            legal_train["mean_epoch_seconds"] / raw_train["mean_epoch_seconds"] - 1.0
        ),
        "legal_evaluation_overhead_percent": 100.0 * (
            legal_eval["milliseconds_per_trajectory"]
            / raw_eval["milliseconds_per_trajectory"]
            - 1.0
        ),
        "inloop_evaluation_multiplier": (
            inloop_eval["milliseconds_per_trajectory"]
            / raw_eval["milliseconds_per_trajectory"]
        ),
        "posthoc_projection_ms_per_trajectory": posthoc_projection_ms,
        "posthoc_projection_fraction_of_raw_percent": (
            100.0 * posthoc_projection_ms / raw_eval["milliseconds_per_trajectory"]
        ),
        "snowball_vs_10_step_multiplier": (
            snowball_eval["milliseconds_per_trajectory"]
            / legal_eval["milliseconds_per_trajectory"]
        ),
        "training_share_of_panel_percent": 100.0 * training_seconds / wall_seconds,
    }

    summary = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "platform": "macOS arm64",
            "accelerator": "Apple M3 Ultra MPS",
            "torch": "2.8.0",
            "execution": "sequential",
        },
        "measurement_contract": {
            "training": "direct epoch timer; training plus fixed validation",
            "evaluation": (
                "sequential artifact-boundary interval; includes process startup, data/model load, "
                "generation, diagnostics, plots, and writes"
            ),
            "snowball": (
                "sequential artifact-boundary interval; includes load, chunk rollout, diagnostics, "
                "plot, and writes"
            ),
            "projection": "direct perf_counter CPU exact-PAVA kernel benchmark",
            "data_preparation": "timestamped canonical restart log to terminal build artifact",
            "monetary_cost": "reported only when explicit rates are supplied",
        },
        "training_summary": training_summary,
        "evaluation_summary": evaluation_summary,
        "snowball_summary": snowball_summary,
        "projection_kernel": projection_rows,
        "derived_comparisons": derived_comparisons,
        "data_preparation": {
            "canonical_rebuild": data_build_span,
            "historical_download": {
                "status": "unmeasured",
                "reason": "retained resumable download log has no reliable start timestamps",
            },
        },
        "panel_budget": {
            "started_at_utc": _iso_utc(start),
            "finished_at_utc": _iso_utc(end),
            "wall_seconds": wall_seconds,
            "wall_hours": wall_hours,
            "training_device_hours": training_seconds / 3600.0,
            "evaluation_device_hours": evaluation_seconds / 3600.0,
            "snowball_device_hours": snowball_seconds / 3600.0,
            "unattributed_hours": unattributed_seconds / 3600.0,
        },
        "cost_model": {
            "hourly_device_rate": hourly_device_rate,
            "device_cost": device_cost,
            "average_system_power_watts": average_system_power_watts,
            "electricity_rate_per_kwh": electricity_rate_per_kwh,
            "energy_kwh": energy_kwh,
            "electricity_cost": electricity_cost,
        },
        "warnings": training_warnings + evaluation_warnings + snowball_warnings + projection_warnings,
    }
    detail = {
        "training": training_rows,
        "evaluation": evaluation_rows,
        "snowball": snowball_rows,
        "projection": projection_rows,
    }
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints_training_seeds/public_itch_1s_panel_50ep",
    )
    parser.add_argument("--results-root", default="results/public_itch_1s_panel_full")
    parser.add_argument("--audit-root", default="results/submission_audits")
    parser.add_argument(
        "--panel-config",
        default="results/public_itch_1s_panel_full/panel_run_config.json",
    )
    parser.add_argument(
        "--panel-completion",
        default="results/public_itch_1s_panel_full/panel_completion.json",
    )
    parser.add_argument(
        "--build-log",
        default="logs/public_itch_multisymbol_build.log",
    )
    parser.add_argument(
        "--build-artifact",
        default=(
            "data/processed/snapshot_l2/"
            "PUBLIC_ITCH_2019_SAMPLES_INTC_1s/build_report.json"
        ),
    )
    parser.add_argument("--output-dir", default="paper_data")
    parser.add_argument("--table", default="paper_tables/table_runtime.md")
    parser.add_argument("--hourly-device-rate", type=float, default=None)
    parser.add_argument("--average-system-power-watts", type=float, default=None)
    parser.add_argument("--electricity-rate-per-kwh", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, detail = build_summary(
        Path(args.checkpoint_root),
        Path(args.results_root),
        Path(args.audit_root),
        Path(args.panel_config),
        Path(args.panel_completion),
        Path(args.build_log),
        Path(args.build_artifact),
        hourly_device_rate=args.hourly_device_rate,
        average_system_power_watts=args.average_system_power_watts,
        electricity_rate_per_kwh=args.electricity_rate_per_kwh,
    )

    (output_dir / "runtime_cost_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    _write_csv(output_dir / "runtime_training_per_run.csv", detail["training"])
    _write_csv(output_dir / "runtime_evaluation_per_run.csv", detail["evaluation"])
    _write_csv(output_dir / "runtime_snowball_per_run.csv", detail["snowball"])
    _write_csv(output_dir / "runtime_projection_kernel.csv", detail["projection"])
    _write_markdown(Path(args.table), summary)

    print(json.dumps({
        "status": "complete",
        "summary": str(output_dir / "runtime_cost_summary.json"),
        "table": args.table,
        "warnings": len(summary["warnings"]),
        "panel_wall_hours": summary["panel_budget"]["wall_hours"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

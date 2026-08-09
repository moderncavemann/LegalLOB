#!/usr/bin/env python3
"""Direct synchronized efficiency benchmark for frozen snapshot methods.

Unlike the full evaluation runner, this benchmark excludes model/data loading,
metrics, plots, sequence serialization, and diagnostic projections. It compares
the actual generation path on the same batch shape and DDIM schedule:

* Raw diffusion;
* Raw + exact post-hoc PAVA;
* Raw + exact in-loop PAVA;
* matched-coordinate diffusion, with and without post-hoc PAVA;
* LegalLOB;
* Raw + the evaluated soft-penalty checkpoint.

Post-hoc rows are paired with their unprojected row: one generated batch is
timed once, then exact CPU PAVA is timed on that same output. In-loop PAVA uses
the current exact CPU projection path at every DDIM step, without evaluation
diagnostics. Accelerator synchronization brackets every measured stage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.data.nasdaq import (  # noqa: E402
    NASDAQMultiDayDataset,
    load_nasdaq_multiday,
    parse_snapshot_dataset_id,
    resolve_lob_constraints,
    resolve_price_reference,
    select_evaluation_indices,
)
from genlob.models.sampling import ddim_sample  # noqa: E402
from genlob.models.sde import VPSDE  # noqa: E402
from genlob.utils.device import device_summary, select_device  # noqa: E402
from genlob.utils.legal_lob import make_lob_transform  # noqa: E402
from genlob.utils.physics import compute_violation_rate, project_to_valid_lob  # noqa: E402
from scripts.evaluate_v2 import (  # noqa: E402
    load_model,
    representation_stats_to_device,
    resolve_representation,
)


METHOD_ORDER = (
    "raw_no_projection",
    "raw_posthoc_pava",
    "raw_inloop_pava",
    "matched_no_projection",
    "matched_posthoc_pava",
    "legal_no_projection",
    "raw_physics_no_projection",
)
METHOD_LABELS = {
    "raw_no_projection": "Raw",
    "raw_posthoc_pava": "Raw + post-hoc PAVA",
    "raw_inloop_pava": "Raw + in-loop PAVA",
    "matched_no_projection": "Matched-coordinate",
    "matched_posthoc_pava": "Matched-coordinate + post-hoc PAVA",
    "legal_no_projection": "LegalLOB",
    "raw_physics_no_projection": "Raw + soft penalty",
}
CHECKPOINT_METHODS = {
    "raw": "raw",
    "matched": "matched_lob",
    "legal": "legal_lob",
    "raw_physics": "raw_physics",
}


@dataclass
class ModelSpec:
    name: str
    model: torch.nn.Module
    representation: str
    representation_stats: dict[str, torch.Tensor] | None
    transform: Any
    sde: VPSDE
    action_dim: int
    checkpoint: str


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter_ns()
    result = fn()
    _sync(device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return result, elapsed_ms


def _percentile(values: list[float], q: float) -> float:
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


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _load_batch(
    dataset_id: str,
    data_dir: str,
    batch_size: int,
    price_reference: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[int]]:
    stock, nday = parse_snapshot_dataset_id(dataset_id)
    raw = load_nasdaq_multiday(data_dir, stock, nday)
    resolved_reference = resolve_price_reference(
        price_reference,
        bool(raw.get("has_raw_states", False)),
    )
    dataset = NASDAQMultiDayDataset(
        raw["test"],
        raw["test_actions"],
        hist_len=50,
        pred_len=10,
        day_id=raw.get("test_day_id"),
        segment_id=raw.get("test_segment_id"),
        raw_data=raw.get("test_raw"),
        normalization=raw.get("normalization"),
        price_reference=resolved_reference,
    )
    selected = select_evaluation_indices(
        len(dataset),
        n_batches=1,
        batch_size=batch_size,
        sampling_mode="uniform",
    )
    loader = DataLoader(
        Subset(dataset, selected.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    history, future, _ = next(iter(loader))
    return history, future, raw, selected.tolist()


def _load_specs(
    checkpoint_root: Path,
    asset: str,
    train_seed: int,
    device: torch.device,
    min_spread: float,
    min_gap: float,
) -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for name, directory in CHECKPOINT_METHODS.items():
        checkpoint = checkpoint_root / asset / f"train_seed{train_seed}" / directory / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing benchmark checkpoint: {checkpoint}")
        model, config, payload = load_model(checkpoint, device, weights="ema")
        representation = resolve_representation("auto", payload)
        stats = representation_stats_to_device(payload, device)
        transform = (
            make_lob_transform(
                representation,
                min_spread=min_spread,
                gap_eps=min_gap,
            )
            if representation != "raw"
            else None
        )
        specs[name] = ModelSpec(
            name=name,
            model=model,
            representation=representation,
            representation_stats=stats,
            transform=transform,
            sde=VPSDE(N=config.num_steps),
            action_dim=config.action_dim,
            checkpoint=str(checkpoint),
        )
    return specs


def _prepare_history(spec: ModelSpec, history_lob: torch.Tensor) -> torch.Tensor:
    if spec.representation == "raw":
        return history_lob
    encoded = spec.transform.encode(history_lob)
    if spec.representation_stats is not None:
        encoded = (
            encoded - spec.representation_stats["mean"]
        ) / spec.representation_stats["std"]
    return encoded


def _decode_output(spec: ModelSpec, generated_model: torch.Tensor) -> torch.Tensor:
    generated = generated_model
    if spec.representation != "raw":
        if spec.representation_stats is not None:
            generated = (
                generated * spec.representation_stats["std"]
                + spec.representation_stats["mean"]
            )
        generated = spec.transform.decode(generated)
    return generated.detach().cpu()


def _exact_inloop_projector(min_spread: float, min_gap: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def project(z: torch.Tensor) -> torch.Tensor:
        projected = project_to_valid_lob(
            z.detach().cpu(),
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=False,
        )
        return projected.to(z.device)

    return project


@torch.inference_mode()
def _benchmark_generation(
    spec: ModelSpec,
    history_lob: torch.Tensor,
    pred_shape: tuple[int, ...],
    device: torch.device,
    ddim_steps: int,
    seed: int,
    method: str,
    min_spread: float,
    min_gap: float,
    include_posthoc: bool = False,
    inloop: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    history_model, preprocess_ms = _timed(
        device, lambda: _prepare_history(spec, history_lob)
    )
    action = torch.zeros(pred_shape[0], spec.action_dim, device=device)
    project_fn = _exact_inloop_projector(min_spread, min_gap) if inloop else None
    _set_seed(seed)
    generated_model, sampling_ms = _timed(
        device,
        lambda: ddim_sample(
            spec.model,
            spec.sde,
            pred_shape,
            history_model,
            action,
            ddim_steps=ddim_steps,
            device=device,
            project_fn=project_fn,
        ),
    )
    generated_cpu, output_ms = _timed(
        device, lambda: _decode_output(spec, generated_model)
    )
    total_ms = preprocess_ms + sampling_ms + output_ms
    base_record = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "preprocess_ms": preprocess_ms,
        "sampling_ms": sampling_ms,
        "decode_transfer_ms": output_ms,
        "repair_ms": None,
        "total_ms": total_ms,
        "projection_location": "in_loop" if inloop else "none",
        "projection_calls": ddim_steps if inloop else 0,
    }
    records = [base_record]
    outputs = {method: generated_cpu}

    if include_posthoc:
        projected, repair_ms = _timed(
            torch.device("cpu"),
            lambda: project_to_valid_lob(
                generated_cpu,
                min_spread=min_spread,
                min_gap=min_gap,
                use_gpu_approx=False,
            ),
        )
        posthoc_method = method.replace("no_projection", "posthoc_pava")
        records.append(
            {
                **base_record,
                "method": posthoc_method,
                "method_label": METHOD_LABELS[posthoc_method],
                "repair_ms": repair_ms,
                "total_ms": total_ms + repair_ms,
                "projection_location": "post_hoc",
                "projection_calls": 1,
            }
        )
        outputs[posthoc_method] = projected
    return records, outputs


def summarize_records(
    records: list[dict[str, Any]],
    batch_size: int,
    frames_per_trajectory: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)

    raw_total = statistics.fmean(
        row["total_ms"] for row in grouped["raw_no_projection"]
    )
    summaries = []
    for method in METHOD_ORDER:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        total = _stats([row["total_ms"] for row in method_rows])
        repair_values = [
            float(row["repair_ms"])
            for row in method_rows
            if row["repair_ms"] is not None
        ]
        mean_ms = total["mean"]
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "measurements": len(method_rows),
                "mean_preprocess_ms_per_batch": statistics.fmean(
                    row["preprocess_ms"] for row in method_rows
                ),
                "mean_sampling_ms_per_batch": statistics.fmean(
                    row["sampling_ms"] for row in method_rows
                ),
                "mean_decode_transfer_ms_per_batch": statistics.fmean(
                    row["decode_transfer_ms"] for row in method_rows
                ),
                "mean_repair_ms_per_batch": (
                    statistics.fmean(repair_values) if repair_values else None
                ),
                "mean_total_ms_per_batch": mean_ms,
                "std_total_ms_per_batch": total["std"],
                "p50_total_ms_per_batch": total["p50"],
                "p95_total_ms_per_batch": total["p95"],
                "milliseconds_per_trajectory": mean_ms / batch_size,
                "microseconds_per_snapshot": (
                    mean_ms * 1000.0 / (batch_size * frames_per_trajectory)
                ),
                "trajectories_per_second": batch_size * 1000.0 / mean_ms,
                "snapshots_per_second": (
                    batch_size * frames_per_trajectory * 1000.0 / mean_ms
                ),
                "relative_to_raw": mean_ms / raw_total,
                "projection_location": method_rows[0]["projection_location"],
                "projection_calls": method_rows[0]["projection_calls"],
            }
        )
    return summaries


def _per_asset_summary(
    records: list[dict[str, Any]],
    batch_size: int,
    frames_per_trajectory: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["asset"]].append(row)
    output = []
    for asset, asset_rows in sorted(grouped.items()):
        for row in summarize_records(asset_rows, batch_size, frames_per_trajectory):
            output.append({"asset": asset, **row})
    return output


def derive_comparisons(
    summary: list[dict[str, Any]],
    batch_size: int,
    frames_per_trajectory: int,
) -> dict[str, float]:
    by_method = {row["method"]: row for row in summary}
    raw = by_method["raw_no_projection"]["mean_total_ms_per_batch"]
    posthoc = by_method["raw_posthoc_pava"]["mean_total_ms_per_batch"]
    inloop = by_method["raw_inloop_pava"]["mean_total_ms_per_batch"]
    legal = by_method["legal_no_projection"]["mean_total_ms_per_batch"]
    repair = by_method["raw_posthoc_pava"]["mean_repair_ms_per_batch"]
    snapshots = float(batch_size * frames_per_trajectory)
    return {
        "legal_overhead_vs_raw_percent": 100.0 * (legal / raw - 1.0),
        "posthoc_overhead_vs_raw_percent": 100.0 * (posthoc / raw - 1.0),
        "inloop_overhead_vs_raw_percent": 100.0 * (inloop / raw - 1.0),
        "legal_latency_reduction_vs_posthoc_percent": 100.0 * (1.0 - legal / posthoc),
        "legal_latency_reduction_vs_inloop_percent": 100.0 * (1.0 - legal / inloop),
        "legal_throughput_gain_vs_posthoc_percent": 100.0 * (posthoc / legal - 1.0),
        "posthoc_repair_microseconds_per_snapshot": repair * 1000.0 / snapshots,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "integrated"
    return f"{float(value):.{digits}f}"


def _load_training_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("training_summary", [])


def _write_markdown(
    path: Path,
    payload: dict[str, Any],
    training_summary: list[dict[str, Any]],
) -> None:
    protocol = payload["protocol"]
    rows = payload["summary"]
    comparisons = payload["comparisons"]
    lines = [
        "# Table: Direct Method Efficiency",
        "",
        f"Hardware: {payload['hardware']['selected']} on {payload['hardware']['platform']}; "
        f"PyTorch {payload['hardware']['torch']}.",
        "",
        f"Protocol: {len(protocol['assets'])} assets x {protocol['repeats']} measured repeats; "
        f"batch={protocol['batch_size']}; future={protocol['frames_per_trajectory']} frames; "
        f"DDIM={protocol['ddim_steps']}; {protocol['warmup']} warmups per operation.",
        "Model/data loading, metrics, plots, sequence serialization, and diagnostic projection are excluded. "
        "Every measured stage is synchronized on the selected accelerator.",
        "",
        "One generated step is one future Level-2 snapshot; it is not one DDIM reverse update. "
        f"Each trajectory contains {protocol['frames_per_trajectory']} generated steps, and each timed "
        f"batch contains {protocol['batch_size'] * protocol['frames_per_trajectory']} generated steps.",
        "",
        "| Method | Preprocess ms/batch | Sampling ms/batch | Decode/transfer ms/batch | Repair ms/batch | Time s/batch | p95 ms/batch | LOB steps/s | Relative to Raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["projection_location"] == "in_loop":
            repair_display = f"integrated ({row['projection_calls']}x)"
        elif row["mean_repair_ms_per_batch"] is None:
            repair_display = "-"
        else:
            repair_display = _fmt(row["mean_repair_ms_per_batch"])
        lines.append(
            f"| {row['method_label']} | "
            f"{_fmt(row['mean_preprocess_ms_per_batch'])} | "
            f"{_fmt(row['mean_sampling_ms_per_batch'])} | "
            f"{_fmt(row['mean_decode_transfer_ms_per_batch'])} | "
            f"{repair_display} | "
            f"{_fmt(row['mean_total_ms_per_batch'] / 1000.0, 4)} +/- "
            f"{_fmt(row['std_total_ms_per_batch'] / 1000.0, 4)} | "
            f"{_fmt(row['p95_total_ms_per_batch'])} | "
            f"{_fmt(row['snapshots_per_second'], 1)} | "
            f"{_fmt(row['relative_to_raw'])}x |"
        )

    lines.extend(
        [
            "",
            "## Direct Comparisons",
            "",
            f"- LegalLOB overhead versus Raw: `{comparisons['legal_overhead_vs_raw_percent']:.2f}%`.",
            f"- Exact post-hoc PAVA overhead versus Raw: `{comparisons['posthoc_overhead_vs_raw_percent']:.2f}%`; "
            f"isolated repair cost `{comparisons['posthoc_repair_microseconds_per_snapshot']:.2f}` us/snapshot "
            f"(`{1e6 / comparisons['posthoc_repair_microseconds_per_snapshot']:.1f}` projection steps/s).",
            f"- Exact in-loop PAVA overhead versus Raw: `{comparisons['inloop_overhead_vs_raw_percent']:.2f}%`.",
            f"- LegalLOB latency reduction versus post-hoc/in-loop PAVA: "
            f"`{comparisons['legal_latency_reduction_vs_posthoc_percent']:.2f}%` / "
            f"`{comparisons['legal_latency_reduction_vs_inloop_percent']:.2f}%`.",
            f"- LegalLOB throughput gain versus post-hoc PAVA: "
            f"`{comparisons['legal_throughput_gain_vs_posthoc_percent']:.2f}%`.",
        ]
    )

    if training_summary:
        raw = next(row for row in training_summary if row["method"] == "raw")
        lines.extend(
            [
                "",
                "## Training Efficiency",
                "",
                "Direct epoch timers cover training plus fixed validation across nine runs per method.",
                "",
                "| Method | Seconds/epoch | Hours/50 epochs | Relative to Raw |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in ("raw", "matched_lob", "legal_lob", "raw_physics"):
            row = next(item for item in training_summary if item["method"] == method)
            lines.append(
                f"| {row['method_label']} | "
                f"{row['mean_epoch_seconds']:.2f} +/- {row['std_epoch_seconds']:.2f} | "
                f"{row['mean_run_hours']:.3f} +/- {row['std_run_hours']:.3f} | "
                f"{row['mean_epoch_seconds'] / raw['mean_epoch_seconds']:.3f}x |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Post-hoc PAVA reuses the exact same generated batch as its unprojected row; its repair column is the isolated incremental CPU exact-PAVA cost.",
            "- In-loop PAVA projects predicted x0 at every DDIM step. Its projection cost is integrated into sampling because projection and denoising alternate.",
            "- LegalLOB includes representation encoding, latent de-normalization, legality-preserving decode, and output transfer, but no repair operator.",
            "- LOB steps/s counts generated future snapshots: batch size times future frames divided by synchronized wall time. It does not count DDIM reverse updates.",
            "- Results are batched throughput measurements, not single-request p50/p95/p99 latency and not a CUDA-versus-MPS comparison.",
            "- Exact PAVA is the current CPU implementation; the result must not be generalized to every possible optimized projection implementation.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="AAPL,AMZN,GOOG")
    parser.add_argument(
        "--dataset-template",
        default="PUBLIC_ITCH_2019_SAMPLES_{asset}_1s",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints_training_seeds/public_itch_1s_panel_50ep",
    )
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-json",
        default="paper_data/method_efficiency_benchmark.json",
    )
    parser.add_argument(
        "--output-csv",
        default="paper_data/method_efficiency_per_repeat.csv",
    )
    parser.add_argument(
        "--table",
        default="paper_tables/table_method_efficiency.md",
    )
    parser.add_argument(
        "--training-summary",
        default="paper_data/runtime_cost_summary.json",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.ddim_steps <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("batch size, DDIM steps, and repeats must be positive; warmup cannot be negative")

    assets = [part.strip() for part in args.assets.split(",") if part.strip()]
    device = select_device(args.device)
    checkpoint_root = Path(args.checkpoint_root)
    records: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    selected_indices: dict[str, list[int]] = {}
    benchmark_started = time.perf_counter()

    operations = (
        "raw_pair",
        "raw_inloop",
        "matched_pair",
        "legal",
        "raw_physics",
    )

    for asset_index, asset in enumerate(assets):
        dataset_id = args.dataset_template.format(asset=asset)
        raw_checkpoint = (
            checkpoint_root / asset / f"train_seed{args.train_seed}" / "raw" / "best.pt"
        )
        raw_payload = torch.load(raw_checkpoint, map_location="cpu", weights_only=False)
        raw_args = raw_payload.get("args") or {}
        history, future, raw_data, indices = _load_batch(
            dataset_id,
            args.data_dir,
            args.batch_size,
            raw_args.get("price_reference", "dataset"),
        )
        min_spread, min_gap = resolve_lob_constraints(
            raw_data.get("normalization"),
            min_spread=raw_args.get("min_spread"),
            min_gap=raw_args.get("min_gap"),
        )
        selected_indices[asset] = indices
        history_lob = history.to(device)
        pred_shape = tuple(future.shape)
        specs = _load_specs(
            checkpoint_root,
            asset,
            args.train_seed,
            device,
            min_spread,
            min_gap,
        )

        def run_operation(operation: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
            if operation == "raw_pair":
                return _benchmark_generation(
                    specs["raw"], history_lob, pred_shape, device, args.ddim_steps,
                    seed, "raw_no_projection", min_spread, min_gap,
                    include_posthoc=True,
                )
            if operation == "raw_inloop":
                return _benchmark_generation(
                    specs["raw"], history_lob, pred_shape, device, args.ddim_steps,
                    seed, "raw_inloop_pava", min_spread, min_gap, inloop=True,
                )
            if operation == "matched_pair":
                return _benchmark_generation(
                    specs["matched"], history_lob, pred_shape, device, args.ddim_steps,
                    seed, "matched_no_projection", min_spread, min_gap,
                    include_posthoc=True,
                )
            if operation == "legal":
                return _benchmark_generation(
                    specs["legal"], history_lob, pred_shape, device, args.ddim_steps,
                    seed, "legal_no_projection", min_spread, min_gap,
                )
            if operation == "raw_physics":
                return _benchmark_generation(
                    specs["raw_physics"], history_lob, pred_shape, device, args.ddim_steps,
                    seed, "raw_physics_no_projection", min_spread, min_gap,
                )
            raise ValueError(operation)

        for warmup_index in range(args.warmup):
            for operation in operations:
                run_operation(operation, args.seed + 900_000 + warmup_index)

        for repeat in range(args.repeats):
            order = list(operations)
            random.Random(args.seed + asset_index * 10_000 + repeat).shuffle(order)
            for operation in order:
                operation_records, outputs = run_operation(
                    operation,
                    args.seed + repeat,
                )
                for row in operation_records:
                    row.update(
                        {
                            "asset": asset,
                            "dataset": dataset_id,
                            "train_seed": args.train_seed,
                            "repeat": repeat,
                            "batch_size": args.batch_size,
                            "frames_per_trajectory": pred_shape[-1],
                            "ddim_steps": args.ddim_steps,
                            "device": str(device),
                        }
                    )
                    records.append(row)
                if repeat == 0:
                    for method, output in outputs.items():
                        validation[f"{asset}/{method}"] = compute_violation_rate(
                            output,
                            min_spread=min_spread,
                            min_gap=min_gap,
                        )

        del specs, history_lob, history, future, raw_payload
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    frames_per_trajectory = int(records[0]["frames_per_trajectory"])
    summary = summarize_records(records, args.batch_size, frames_per_trajectory)
    per_asset = _per_asset_summary(records, args.batch_size, frames_per_trajectory)
    comparisons = derive_comparisons(summary, args.batch_size, frames_per_trajectory)
    benchmark_wall_seconds = time.perf_counter() - benchmark_started
    hardware = {
        **device_summary(device),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "hardware": hardware,
        "protocol": {
            "assets": assets,
            "dataset_template": args.dataset_template,
            "train_seed": args.train_seed,
            "batch_size": args.batch_size,
            "frames_per_trajectory": frames_per_trajectory,
            "ddim_steps": args.ddim_steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "paired_noise_seed_base": args.seed,
            "selected_dataset_indices": selected_indices,
            "model_and_data_loading_timed": False,
            "metrics_plots_and_serialization_timed": False,
            "diagnostic_projection_timed": False,
            "posthoc_projection": "exact CPU PAVA on the paired generated output",
            "inloop_projection": "exact CPU PAVA with CPU-device transfer at each DDIM step; no path diagnostics",
            "synchronization": "before and after every measured accelerator stage",
        },
        "benchmark_wall_seconds": benchmark_wall_seconds,
        "summary": summary,
        "comparisons": comparisons,
        "per_asset_summary": per_asset,
        "validation": validation,
        "records": records,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_csv(Path(args.output_csv), records)
    training_summary = _load_training_summary(Path(args.training_summary))
    _write_markdown(Path(args.table), payload, training_summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "device": str(device),
                "measurements": len(records),
                "benchmark_wall_seconds": benchmark_wall_seconds,
                "output_json": args.output_json,
                "output_csv": args.output_csv,
                "table": args.table,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

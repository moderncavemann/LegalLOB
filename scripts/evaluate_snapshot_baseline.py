#!/usr/bin/env python3
"""Evaluate deterministic snapshot baselines on the shared GenLOB protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.data.nasdaq import (
    NASDAQMultiDayDataset,
    load_nasdaq_multiday,
    normalized_price_unit,
    parse_snapshot_dataset_id,
    resolve_lob_constraints,
    resolve_price_reference,
    select_evaluation_indices,
)
from genlob.eval.metrics import (
    discrete_lob_validity,
    horizon_fidelity,
    json_safe,
    per_sample_metrics,
    projection_repair_details,
    safe_corr,
    structural_validity_details,
    stylized_facts,
)
from genlob.utils.physics import (
    compute_projection_distance,
    compute_violation_rate,
    project_to_valid_lob,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--baseline",
        choices=["persistence", "moving_block_bootstrap"],
        default="persistence",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--sampling-mode", choices=["uniform", "head", "full"], default="uniform")
    parser.add_argument("--price-reference", choices=["auto", "dataset", "history_last_mid"], default="auto")
    parser.add_argument("--min-spread", type=float, default=None)
    parser.add_argument("--min-gap", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save-sequences", dest="save_sequences", action="store_false")
    parser.set_defaults(save_sequences=True)
    args = parser.parse_args()

    stock, nday = parse_snapshot_dataset_id(args.dataset)
    data = load_nasdaq_multiday(args.data_dir, stock, nday)
    args.price_reference = resolve_price_reference(
        args.price_reference,
        bool(data.get("has_raw_states", False)),
    )
    args.min_spread, args.min_gap = resolve_lob_constraints(
        data.get("normalization"),
        min_spread=args.min_spread,
        min_gap=args.min_gap,
    )
    dataset = NASDAQMultiDayDataset(
        data[args.split],
        data[f"{args.split}_actions"],
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        day_id=data.get(f"{args.split}_day_id"),
        segment_id=data.get(f"{args.split}_segment_id"),
        raw_data=data.get(f"{args.split}_raw"),
        normalization=data.get("normalization"),
        price_reference=args.price_reference,
    )
    selected = select_evaluation_indices(
        len(dataset),
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        sampling_mode=args.sampling_mode,
    )
    loader = DataLoader(
        Subset(dataset, selected.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    bootstrap_indices = None
    bootstrap_loader = None
    if args.baseline == "moving_block_bootstrap":
        train_dataset = NASDAQMultiDayDataset(
            data["train"],
            data["train_actions"],
            hist_len=args.hist_len,
            pred_len=args.pred_len,
            day_id=data.get("train_day_id"),
            segment_id=data.get("train_segment_id"),
            raw_data=data.get("train_raw"),
            normalization=data.get("normalization"),
            price_reference=args.price_reference,
        )
        if not len(train_dataset):
            raise ValueError("No training windows available for moving-block bootstrap")
        rng = np.random.default_rng(args.seed)
        bootstrap_indices = rng.integers(0, len(train_dataset), size=len(selected), dtype=np.int64)
        bootstrap_loader = DataLoader(
            Subset(train_dataset, bootstrap_indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        bootstrap_iterator = iter(bootstrap_loader)

    real_parts = []
    generated_parts = []
    persistence_parts = []
    for history, future, _ in loader:
        persistence = history[..., -1:].expand(-1, -1, -1, args.pred_len).clone()
        if args.baseline == "persistence":
            generated = persistence
        else:
            _, generated, _ = next(bootstrap_iterator)
        real_parts.append(future.float())
        generated_parts.append(generated.float())
        persistence_parts.append(persistence.float())
    if not real_parts:
        raise ValueError("No evaluation windows selected")
    real = torch.cat(real_parts)
    generated = torch.cat(generated_parts)
    persistence = torch.cat(persistence_parts)
    projected = project_to_valid_lob(
        generated,
        min_spread=args.min_spread,
        min_gap=args.min_gap,
        use_gpu_approx=False,
    )
    price_unit = normalized_price_unit(data.get("normalization")) or 1.0
    projection_distance = compute_projection_distance(generated, projected)
    spread_real = real[:, 0, 0] - real[:, 0, 10]
    spread_generated = generated[:, 0, 0] - generated[:, 0, 10]
    results = {
        "method": args.baseline,
        "representation": "deterministic_baseline",
        "projection_mode": "none",
        "sampler_projection_mode": "none",
        "price_reference": args.price_reference,
        "min_spread": args.min_spread,
        "min_gap": args.min_gap,
        "mse_raw": float(((generated - real) ** 2).mean()),
        "mse_projected": float(((projected - real) ** 2).mean()),
        "mae_raw": float((generated - real).abs().mean()),
        "mae_projected": float((projected - real).abs().mean()),
        "corr_price_raw": safe_corr(real[:, 0].numpy(), generated[:, 0].numpy()),
        "corr_price_projected": safe_corr(real[:, 0].numpy(), projected[:, 0].numpy()),
        "corr_volume_raw": safe_corr(real[:, 1].numpy(), generated[:, 1].numpy()),
        "corr_volume_projected": safe_corr(real[:, 1].numpy(), projected[:, 1].numpy()),
        "violations_raw": compute_violation_rate(
            generated, min_spread=args.min_spread, min_gap=args.min_gap,
        ),
        "violations_proj": compute_violation_rate(
            projected, min_spread=args.min_spread, min_gap=args.min_gap,
        ),
        "violations_real": compute_violation_rate(
            real, min_spread=args.min_spread, min_gap=args.min_gap,
        ),
        "structural_validity_real": structural_validity_details(
            real, args.min_spread, args.min_gap
        ),
        "structural_validity_raw": structural_validity_details(
            generated, args.min_spread, args.min_gap
        ),
        "structural_validity_projected": structural_validity_details(
            projected, args.min_spread, args.min_gap
        ),
        "diagnostic_projection_repair": projection_repair_details(
            generated, projected, data.get("normalization")
        ),
        "output_projection_repair": projection_repair_details(
            generated, generated, data.get("normalization")
        ),
        "projection_distance": projection_distance,
        "spread_mean_raw": float(spread_generated.mean()),
        "spread_mean_proj": float((projected[:, 0, 0] - projected[:, 0, 10]).mean()),
        "spread_mean_real": float(spread_real.mean()),
        "spread_shift_raw": float(spread_generated.mean() - spread_real.mean()),
        "spread_shift_projected": float(
            (projected[:, 0, 0] - projected[:, 0, 10]).mean() - spread_real.mean()
        ),
        "stylized_facts_raw": stylized_facts(real, generated, price_unit=price_unit),
        "stylized_facts_projected": stylized_facts(real, projected, price_unit=price_unit),
        "horizon_fidelity_raw": horizon_fidelity(real, generated, persistence),
        "horizon_fidelity_projected": horizon_fidelity(real, projected, persistence),
        "horizon_fidelity_persistence": horizon_fidelity(real, persistence),
        "discrete_validity_real": discrete_lob_validity(real, data.get("normalization")),
        "discrete_validity_raw": discrete_lob_validity(generated, data.get("normalization")),
        "discrete_validity_projected": discrete_lob_validity(
            projected, data.get("normalization")
        ),
        "evaluation_protocol": {
            "sampling_mode": args.sampling_mode,
            "dataset_windows": len(dataset),
            "selected_windows": len(selected),
            "baseline_seed": args.seed,
            "bootstrap_source": "train_only_contiguous_future_blocks_with_replacement"
            if args.baseline == "moving_block_bootstrap"
            else None,
            "bootstrap_training_windows": len(train_dataset)
            if args.baseline == "moving_block_bootstrap"
            else None,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_results = json_safe(results)
    (output_dir / "metrics.json").write_text(
        json.dumps(safe_results, indent=2, allow_nan=False) + "\n"
    )
    sample = per_sample_metrics(
        real,
        generated,
        projected,
        min_spread=args.min_spread,
        min_gap=args.min_gap,
    )
    sample["dataset_index"] = selected[:len(real)]
    selected_starts = dataset.starts[selected]
    sample["start_index"] = selected_starts
    if dataset.day_id is not None:
        sample["day_id"] = dataset.day_id[selected_starts]
    if dataset.segment_id is not None:
        sample["segment_id"] = dataset.segment_id[selected_starts]
    if bootstrap_indices is not None:
        sample["bootstrap_train_index"] = bootstrap_indices[:len(real)]
    np.savez_compressed(output_dir / "per_sample_metrics.npz", **sample)
    if args.save_sequences:
        np.save(output_dir / "real_future.npy", real.numpy().astype(np.float32, copy=False))
        np.save(output_dir / "generated_raw.npy", generated.numpy().astype(np.float32, copy=False))
        np.save(
            output_dir / "generated_projected.npy",
            projected.numpy().astype(np.float32, copy=False),
        )
        np.save(output_dir / "persistence.npy", persistence.numpy().astype(np.float32, copy=False))
    print(json.dumps(safe_results, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

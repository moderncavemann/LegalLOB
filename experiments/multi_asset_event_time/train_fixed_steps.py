#!/usr/bin/env python3
"""Fixed-step event-time training with uniform split-only subsampling.

The general ``train_v2.py`` path retains its legacy defaults for all previous
experiments. This wrapper changes two deterministic data views for this study:
representation-coordinate statistics span the full training-window index, and
the capped validation loader spans the full validation-window index.  Neither
view can access validation/test rows outside its declared split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_v2 as base


_BASE_PARSE_ARGS = base.parse_args


DATASETS = frozenset(
    {
        "PUBLIC_ITCH_2019_SAMPLES_AAPL_event",
        "PUBLIC_ITCH_2019_SAMPLES_AMZN_event",
        "PUBLIC_ITCH_2019_SAMPLES_GOOG_event",
        "PUBLIC_ITCH_2019_SAMPLES_INTC_event",
    }
)


def deterministic_uniform_indices(length: int, count: int) -> list[int]:
    """Return unique ordered indices that deterministically span a dataset."""
    if length < 0:
        raise ValueError("Dataset length must be non-negative")
    if count <= 0:
        raise ValueError("Uniform sample count must be positive")
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return [(index * (length - 1)) // (count - 1) for index in range(count)]


def parse_and_validate_args():
    args = _BASE_PARSE_ARGS()
    expected = {
        "data_dir": "data",
        "hist_len": 50,
        "pred_len": 10,
        "base_channels": 128,
        "num_layers": 16,
        "action_dim": 5,
        "action_conditioning": "none",
        "representation_stat_batches": 548,
        "beta_min": 0.1,
        "beta_max": 20.0,
        "num_scales": 100,
        "continuous": False,
        "epochs": 50,
        "batch_size": 128,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "clip_grad": 1.0,
        "ema_rate": 0.999,
        "cfg_drop": 0.1,
        "physics_loss_weight": 0.0,
        "action_contrastive_weight": 0.0,
        "action_margin": 0.0,
        "allow_diagnostic_action_loss": False,
        "save_every": 1000,
        "resume": None,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "max_train_batches": 548,
        "max_val_batches": 183,
        "fixed_train_order": False,
        "device": "mps",
        "num_workers": 0,
        "validation_seed": 10000,
    }
    if args.dataset not in DATASETS:
        raise ValueError(
            f"Dataset is not in the frozen event-time asset set: {args.dataset}"
        )
    if args.representation not in {"legal_lob", "matched_lob"}:
        raise ValueError(
            f"Representation is not registered for this study: {args.representation}"
        )
    if args.seed not in {0, 1, 2}:
        raise ValueError(f"Training seed is not registered for this study: {args.seed}")
    if args.price_reference != "auto":
        raise ValueError(
            "The study command must resolve price reference from frozen metadata"
        )
    for key, required in expected.items():
        actual = getattr(args, key)
        if actual != required:
            raise ValueError(
                f"Fixed protocol mismatch for {key}: expected {required!r}, "
                f"got {actual!r}"
            )
    # These fields make the wrapper-only behavior explicit in every checkpoint.
    args.p4_uniform_representation_stats = True
    args.p4_uniform_validation = True
    args.p4_training_entrypoint = (
        "experiments/multi_asset_event_time/train_fixed_steps.py"
    )
    return args


def get_data_loaders(args):
    from genlob.data.nasdaq import (
        NASDAQMultiDayDataset,
        load_nasdaq_multiday,
        parse_snapshot_dataset_id,
        resolve_lob_constraints,
        resolve_price_reference,
    )

    stock, nday = parse_snapshot_dataset_id(args.dataset)
    raw = load_nasdaq_multiday(args.data_dir or "data", stock, nday)
    args.price_reference = resolve_price_reference(
        args.price_reference, bool(raw.get("has_raw_states", False))
    )
    args.min_spread, args.min_gap = resolve_lob_constraints(
        raw.get("normalization"),
        min_spread=args.min_spread,
        min_gap=args.min_gap,
    )
    loaders = []
    for split in ("train", "val", "test"):
        dataset = NASDAQMultiDayDataset(
            raw[split],
            raw[f"{split}_actions"],
            args.hist_len,
            args.pred_len,
            day_id=raw.get(f"{split}_day_id"),
            segment_id=raw.get(f"{split}_segment_id"),
            raw_data=raw.get(f"{split}_raw"),
            normalization=raw.get("normalization"),
            price_reference=args.price_reference,
        )
        loader_dataset = dataset
        if split == "val":
            sample_count = min(
                len(dataset), args.max_val_batches * args.batch_size
            )
            loader_dataset = Subset(
                dataset,
                deterministic_uniform_indices(len(dataset), sample_count),
            )
        loaders.append(
            DataLoader(
                loader_dataset,
                batch_size=args.batch_size,
                shuffle=(split == "train"),
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return tuple(loaders)


@torch.no_grad()
def compute_representation_stats(train_loader, args, transform):
    if args.representation == "raw":
        return None
    max_batches = args.representation_stat_batches
    sample_count = min(
        len(train_loader.dataset), max_batches * train_loader.batch_size
    )
    stats_dataset = Subset(
        train_loader.dataset,
        deterministic_uniform_indices(len(train_loader.dataset), sample_count),
    )
    stats_loader = DataLoader(
        stats_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    total = None
    total_sq = None
    count = 0
    for batch_idx, batch in enumerate(stats_loader):
        if batch_idx >= max_batches:
            break
        history, future = batch[0].float(), batch[1].float()
        encoded = torch.cat(
            [transform.encode(history), transform.encode(future)], dim=-1
        )
        batch_sum = encoded.sum(dim=(0, 3), keepdim=True)
        batch_sq = (encoded ** 2).sum(dim=(0, 3), keepdim=True)
        batch_count = encoded.shape[0] * encoded.shape[-1]
        total = batch_sum if total is None else total + batch_sum
        total_sq = batch_sq if total_sq is None else total_sq + batch_sq
        count += batch_count
    if count == 0:
        raise ValueError("Event-time representation-stat subset is empty")
    mean = total / count
    variance = torch.clamp(
        total_sq / count - mean ** 2, min=args.representation_eps ** 2
    )
    return {"mean": mean.cpu(), "std": torch.sqrt(variance).cpu()}


def main() -> None:
    base.parse_args = parse_and_validate_args
    base.get_data_loaders = get_data_loaders
    base.compute_representation_stats = compute_representation_stats
    base.main()


if __name__ == "__main__":
    main()

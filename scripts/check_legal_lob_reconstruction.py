#!/usr/bin/env python3
"""Deterministic encode/decode checks for legality-preserving representations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.data.nasdaq import (
    load_nasdaq_multiday,
    NASDAQMultiDayDataset,
    parse_snapshot_dataset_id,
    resolve_lob_constraints,
    resolve_price_reference,
)
from genlob.utils.legal_lob import make_lob_transform
from genlob.utils.physics import compute_projection_distance, compute_violation_rate


def get_loader(args):
    data_dir = args.data_dir or "data"
    stock, nday = parse_snapshot_dataset_id(args.dataset)
    raw = load_nasdaq_multiday(data_dir, stock, nday)
    args.price_reference = resolve_price_reference(
        args.price_reference,
        bool(raw.get("has_raw_states", False)),
    )
    args.min_spread, args.min_gap = resolve_lob_constraints(
        raw.get("normalization"),
        min_spread=args.min_spread,
        min_gap=args.min_gap,
    )
    ds = NASDAQMultiDayDataset(
        raw["test"], raw["test_actions"],
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        day_id=raw.get("test_day_id"),
        segment_id=raw.get("test_segment_id"),
        raw_data=raw.get("test_raw"),
        normalization=raw.get("normalization"),
        price_reference=args.price_reference,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)


def _gap_values(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prices = x[:, 0]
    ask = prices[:, :10]
    bid = prices[:, 10:]
    return ask[:, 1:] - ask[:, :-1], bid[:, :-1] - bid[:, 1:]


def _boundary_stats(x: torch.Tensor, min_gap: float) -> dict[str, float]:
    ask_gap, bid_gap = _gap_values(x)
    tolerance = max(1e-7, abs(min_gap) * 2e-5)
    combined = torch.cat([ask_gap.flatten(), bid_gap.flatten()])
    boundary = (combined - min_gap).abs() <= tolerance
    shortfall = combined < min_gap - tolerance
    interior = combined > min_gap + tolerance
    return {
        "boundary_tolerance": tolerance,
        "gap_boundary_rate": float(boundary.float().mean()),
        "gap_shortfall_rate": float(shortfall.float().mean()),
        "gap_interior_rate": float(interior.float().mean()),
        "gap_min": float(combined.min()),
        "gap_mean": float(combined.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="INTC_9day")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--price-reference", choices=["auto", "dataset", "history_last_mid"], default="auto")
    parser.add_argument("--min-spread", type=float, default=None)
    parser.add_argument("--min-gap", type=float, default=None)
    parser.add_argument(
        "--representation",
        choices=["legal_lob", "legal_lob_ba"],
        default="legal_lob",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    loader = get_loader(args)
    transform = make_lob_transform(
        args.representation,
        min_spread=args.min_spread,
        gap_eps=args.min_gap,
    )
    xs = []
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        future = batch[1]
        xs.append(future.float())
    if not xs:
        raise ValueError("No batches loaded")

    x = torch.cat(xs, dim=0)
    z = transform.encode(x)
    x_rec = transform.decode(z)
    diff = x_rec - x

    spread = x[:, 0, 0, :] - x[:, 0, 10, :]
    spread_rec = x_rec[:, 0, 0, :] - x_rec[:, 0, 10, :]
    original_boundary = _boundary_stats(x, args.min_gap)
    reconstructed_boundary = _boundary_stats(x_rec, args.min_gap)

    report = {
        "dataset": args.dataset,
        "representation": args.representation,
        "price_reference": args.price_reference,
        "min_spread": args.min_spread,
        "min_gap": args.min_gap,
        "rows": int(x.shape[0]),
        "shape": list(x.shape),
        "mse": float((diff ** 2).mean()),
        "mae": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "finite_latent_rate": float(torch.isfinite(z).float().mean()),
        "finite_reconstruction_rate": float(torch.isfinite(x_rec).float().mean()),
        "projection_distance": compute_projection_distance(x, x_rec),
        "violations_reconstructed": compute_violation_rate(
            x_rec, min_spread=args.min_spread, min_gap=args.min_gap,
        ),
        "violations_original": compute_violation_rate(
            x, min_spread=args.min_spread, min_gap=args.min_gap,
        ),
        "spread_mean_original": float(spread.mean()),
        "spread_mean_reconstructed": float(spread_rec.mean()),
        "spread_mae": float((spread_rec - spread).abs().mean()),
        "boundary_original": original_boundary,
        "boundary_reconstructed": reconstructed_boundary,
        "gap_boundary_calibration_abs_error": abs(
            reconstructed_boundary["gap_boundary_rate"]
            - original_boundary["gap_boundary_rate"]
        ),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a 1 Hz GenLOB snapshot dataset in DiffLOB's level orientation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resolve_dataset_dir(data_dir: Path, dataset: str) -> Path:
    candidates = (
        data_dir / dataset,
        data_dir / "processed" / "snapshot_l2" / dataset,
        data_dir / "snapshot_l2" / dataset,
    )
    for candidate in candidates:
        if (candidate / "states_raw.npy").exists() and (candidate / "metadata.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not find {dataset} under {data_dir}")


def to_difflob_tensor(raw_states: np.ndarray) -> np.ndarray:
    """Convert [T,2,20] ask1..10,bid1..10 to [T,20,3] ask10..1,bid1..10."""
    raw_states = np.asarray(raw_states)
    if raw_states.ndim != 3 or raw_states.shape[1:] != (2, 20):
        raise ValueError(f"Expected [T,2,20], got {raw_states.shape}")
    ask_prices = raw_states[:, 0, :10][:, ::-1]
    bid_prices = raw_states[:, 0, 10:]
    ask_sizes = raw_states[:, 1, :10][:, ::-1]
    bid_sizes = raw_states[:, 1, 10:]
    prices = np.concatenate([ask_prices, bid_prices], axis=1)
    sizes = np.concatenate([ask_sizes, bid_sizes], axis=1)
    sides = np.concatenate(
        [np.zeros((len(raw_states), 10)), np.ones((len(raw_states), 10))],
        axis=1,
    )
    return np.stack([prices, sizes, sides], axis=-1).astype(np.float32)


def valid_window_starts(
    split: np.ndarray,
    day_id: np.ndarray,
    segment_id: np.ndarray,
    total_window: int,
    step: int,
) -> np.ndarray:
    starts = []
    for start in range(0, len(split) - total_window + 1, step):
        stop = start + total_window
        if split[start] < 0 or segment_id[start] < 0:
            continue
        if (
            np.all(split[start:stop] == split[start])
            and np.all(day_id[start:stop] == day_id[start])
            and np.all(segment_id[start:stop] == segment_id[start])
        ):
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def build_regime_arrays(
    lob: np.ndarray,
    starts: np.ndarray,
    past_window: int,
    predict_window: int,
) -> dict[str, np.ndarray]:
    trend = np.empty(len(starts), dtype=np.float32)
    volatility = np.empty(len(starts), dtype=np.float32)
    liquidity = np.empty((len(starts), predict_window), dtype=np.float32)
    imbalance = np.empty((len(starts), predict_window), dtype=np.float32)
    for row, start in enumerate(starts):
        future_start = start + past_window
        future = lob[future_start:future_start + predict_window]
        last_past = lob[future_start - 1]
        mids = np.concatenate(
            [
                np.asarray([(last_past[9, 0] + last_past[10, 0]) * 0.5]),
                (future[:, 9, 0] + future[:, 10, 0]) * 0.5,
            ]
        )
        differences = np.diff(mids)
        trend[row] = differences.sum()
        volatility[row] = differences.std(ddof=1) if len(differences) > 1 else 0.0
        ask_liquidity = future[:, :10, 1].sum(axis=1)
        bid_liquidity = future[:, 10:, 1].sum(axis=1)
        liquidity[row] = ask_liquidity + bid_liquidity
        denominator = np.maximum(ask_liquidity + bid_liquidity, 1e-12)
        imbalance[row] = (ask_liquidity - bid_liquidity) / denominator
    return {
        "window_start": starts,
        "trend": trend,
        "volatility": volatility,
        "future_liquidity": liquidity,
        "future_imbalance": imbalance,
    }


def _write_csv(
    path: Path,
    raw: np.ndarray,
    timestamps: np.ndarray,
    day_id: np.ndarray,
) -> None:
    import pandas as pd

    dates = pd.to_datetime(np.asarray(day_id).astype(str), format="%Y%m%d")
    time_dt = dates + pd.to_timedelta(np.asarray(timestamps), unit="s")
    payload: dict[str, object] = {"Time_dt": time_dt, "Time": timestamps}
    for level in range(1, 11):
        payload[f"Ask_Price_{level}"] = raw[:, 0, level - 1]
        payload[f"Ask_Size_{level}"] = raw[:, 1, level - 1]
        payload[f"Bid_Price_{level}"] = raw[:, 0, 10 + level - 1]
        payload[f"Bid_Size_{level}"] = raw[:, 1, 10 + level - 1]
    pd.DataFrame(payload).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--past-window", type=int, default=32)
    parser.add_argument("--predict-window", type=int, default=32)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(Path(args.data_dir), args.dataset)
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    if metadata.get("sampling") != "1s":
        raise ValueError("DiffLOB export requires a dataset built with sampling=1s")
    raw = np.load(dataset_dir / "states_raw.npy", mmap_mode="r")
    split = np.load(dataset_dir / "split.npy")
    day_id = np.load(dataset_dir / "day_id.npy")
    segment_id = np.load(dataset_dir / "segment_id.npy")
    timestamps = np.load(dataset_dir / "timestamps.npy")
    lob = to_difflob_tensor(raw)
    starts = valid_window_starts(
        split,
        day_id,
        segment_id,
        args.past_window + args.predict_window,
        args.step,
    )
    regimes = build_regime_arrays(lob, starts, args.past_window, args.predict_window)

    output_dir = Path(args.output_dir) if args.output_dir else Path("data/processed/difflob") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "lob_tensor.npy", lob)
    np.save(output_dir / "timestamps.npy", timestamps)
    np.save(output_dir / "day_id.npy", day_id)
    np.save(output_dir / "split.npy", split)
    np.save(output_dir / "segment_id.npy", segment_id)
    np.savez_compressed(output_dir / "regime_windows.npz", **regimes)
    if args.write_csv:
        for split_value, split_name in ((0, "train"), (1, "val"), (2, "test")):
            mask = split == split_value
            _write_csv(
                output_dir / f"{split_name}.csv",
                raw[mask],
                timestamps[mask],
                day_id[mask],
            )
    export_metadata = {
        "source_dataset": args.dataset,
        "format": "DiffLOB [T,20,3] ask10..ask1,bid1..bid10 with price,size,side",
        "past_window": args.past_window,
        "predict_window": args.predict_window,
        "step": args.step,
        "windows": int(len(starts)),
        "future_regime_labels": ["trend", "volatility", "liquidity_path", "imbalance_path"],
        "comparison_policy": "future regime labels are oracle counterfactual conditions",
        "official_code_status": "compatibility wrapper still required; official preprocessing package is missing",
    }
    (output_dir / "metadata.json").write_text(json.dumps(export_metadata, indent=2) + "\n")
    print(json.dumps(export_metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

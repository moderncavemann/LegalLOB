#!/usr/bin/env python3
"""Parse public Nasdaq ITCH 5.0 samples into GenLOB snapshot_l2 datasets.

This parser targets research/data-preparation use for a small set of symbols.
It streams the gzip file, filters selected symbols by stock locate, reconstructs
displayed add/cancel/delete/execute/replace order book events, and writes
top-10 L2 snapshots.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


EVENT_CODE = {
    "none": 0.0,
    "submission": 0.25,
    "cancel_delete": 0.5,
    "execution": 1.0,
    "replace": 0.75,
    "trade": 1.0,
}


def _u48(buf: bytes, offset: int) -> int:
    return int.from_bytes(buf[offset:offset + 6], "big")


def _stock(buf: bytes) -> str:
    return buf.decode("ascii", errors="ignore").strip()


def _side_value(side: str) -> float:
    return 1.0 if side == "B" else -1.0 if side == "S" else 0.0


@dataclass
class Order:
    side: str
    shares: int
    price: int


class Book:
    def __init__(self) -> None:
        self.orders: dict[int, Order] = {}
        self.bid: dict[int, int] = defaultdict(int)
        self.ask: dict[int, int] = defaultdict(int)

    def _levels(self, side: str) -> dict[int, int]:
        return self.bid if side == "B" else self.ask

    def add(self, order_ref: int, side: str, shares: int, price: int) -> None:
        if order_ref in self.orders:
            self.delete(order_ref)
        self.orders[order_ref] = Order(side=side, shares=shares, price=price)
        self._levels(side)[price] += shares

    def reduce(self, order_ref: int, shares: int) -> tuple[str, int] | None:
        order = self.orders.get(order_ref)
        if order is None:
            return None
        removed = min(order.shares, shares)
        levels = self._levels(order.side)
        levels[order.price] -= removed
        if levels[order.price] <= 0:
            levels.pop(order.price, None)
        order.shares -= removed
        if order.shares <= 0:
            self.orders.pop(order_ref, None)
        return order.side, removed

    def delete(self, order_ref: int) -> tuple[str, int] | None:
        order = self.orders.pop(order_ref, None)
        if order is None:
            return None
        levels = self._levels(order.side)
        levels[order.price] -= order.shares
        if levels[order.price] <= 0:
            levels.pop(order.price, None)
        return order.side, order.shares

    def replace(self, old_ref: int, new_ref: int, shares: int, price: int) -> tuple[str, int] | None:
        old = self.orders.get(old_ref)
        if old is None:
            return None
        side = old.side
        self.delete(old_ref)
        self.add(new_ref, side, shares, price)
        return side, shares

    def top10(self) -> np.ndarray | None:
        if len(self.ask) < 10 or len(self.bid) < 10:
            return None
        ask_prices = sorted(self.ask)[:10]
        bid_prices = sorted(self.bid, reverse=True)[:10]
        prices = [p / 10_000.0 for p in ask_prices] + [p / 10_000.0 for p in bid_prices]
        volumes = [self.ask[p] for p in ask_prices] + [self.bid[p] for p in bid_prices]
        return np.stack([np.asarray(prices, dtype=np.float32), np.asarray(volumes, dtype=np.float32)], axis=0)


def _iter_itch_messages(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        while True:
            length_bytes = f.read(2)
            if not length_bytes:
                break
            if len(length_bytes) != 2:
                break
            length = int.from_bytes(length_bytes, "big")
            payload = f.read(length)
            if len(payload) != length:
                break
            yield payload


def _fit_normalization(raw_states: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train = raw_states[split == 0]
    train_prices = train[:, 0, :] * 100.0
    price_std = float(np.std(train_prices))
    if price_std <= 0:
        raise ValueError("Invalid train price std")
    volume_clip = float(max(np.quantile(train[:, 1, :], 0.99), 1.0))
    prices = raw_states[:, 0, :]
    mid = ((prices[:, 0] + prices[:, 10]) / 2.0)[:, None]
    norm_prices = ((prices - mid) * 100.0 / price_std).astype(np.float32)
    norm_volumes = (np.sqrt(np.clip(raw_states[:, 1, :], 0, volume_clip)) / 15.0).astype(np.float32)
    states = np.stack([norm_prices, norm_volumes], axis=1).astype(np.float32)
    spreads = train[:, 0, 0] - train[:, 0, 10]
    pos = spreads[spreads > 0]
    min_spread_raw = float(pos.min()) if len(pos) else 0.01
    train_ask = train[:, 0, :10]
    train_bid = train[:, 0, 10:]
    level_gaps = np.concatenate(
        [
            (train_ask[:, 1:] - train_ask[:, :-1]).reshape(-1),
            (train_bid[:, :-1] - train_bid[:, 1:]).reshape(-1),
        ]
    )
    positive_gaps = level_gaps[level_gaps > 0]
    min_level_gap_raw = float(positive_gaps.min()) if len(positive_gaps) else min_spread_raw
    normalization = {
        "price_transform": "mid_relative_train_price_std",
        "volume_transform": "sqrt_clip_train_quantile_scale",
        "fit_split": "train_only",
        "price_scale": 100.0,
        "price_global_std": price_std,
        "volume_clip_quantile": 0.99,
        "volume_clip": volume_clip,
        "volume_scale": 15.0,
        "min_spread_raw": min_spread_raw,
        "min_spread_normalized": float(max(min_spread_raw * 100.0 / price_std, 1e-8)),
        "min_level_gap_raw": min_level_gap_raw,
        "min_level_gap_normalized": float(max(min_level_gap_raw * 100.0 / price_std, 1e-8)),
        "levels": 10,
        "state_shape": [2, 20],
    }
    return states, normalization


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _update_manifest(
    manifest_path: Path,
    dataset_id: str,
    symbol: str,
    output_dir: Path,
    has_true_day_boundaries: bool,
) -> None:
    manifest = {"datasets": {}}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or manifest
    manifest.setdefault("datasets", {})
    manifest["datasets"][dataset_id] = {
        "status": "prepared_snapshot_l2_from_public_nasdaq_itch",
        "symbol": symbol,
        "source": "public_nasdaq_itch_sample",
        "level": 10,
        "state_path": str(output_dir / "states.npy"),
        "merged_state_path": str(output_dir / "states_merged.npy"),
        "raw_state_path": str(output_dir / "states_raw.npy"),
        "action_path": str(output_dir / "actions.npy"),
        "timestamps_path": str(output_dir / "timestamps.npy"),
        "day_id_path": str(output_dir / "day_id.npy"),
        "split_path": str(output_dir / "split.npy"),
        "normalization_path": str(output_dir / "normalization.json"),
        "metadata_path": str(output_dir / "metadata.json"),
        "dataset_card_path": str(output_dir / "DATASET_CARD.md"),
        "has_message_data": False,
        "has_raw_itch_source": True,
        "processed_event_l3_available": False,
        "has_true_timestamps": True,
        "has_true_day_boundaries": has_true_day_boundaries,
        "action_label_status": "itch_event_level_diagnostic_not_causal",
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def parse_symbol(
    path: Path,
    symbol: str,
    max_events: int | None = None,
    progress_every: int = 5_000_000,
) -> dict[str, Any]:
    symbol = symbol.upper()
    locate_to_symbol: dict[int, str] = {}
    target_locates: set[int] = set()
    book = Book()
    states: list[np.ndarray] = []
    timestamps: list[float] = []
    action_rows: list[list[float]] = []
    action_sizes: list[float] = []
    counters: dict[str, int] = defaultdict(int)

    def append_snapshot(ts_ns: int, action: list[float]) -> None:
        snap = book.top10()
        if snap is None:
            return
        states.append(snap)
        timestamps.append(ts_ns / 1e9)
        action_rows.append(action)
        action_sizes.append(action[3])

    for i, msg in enumerate(_iter_itch_messages(path), start=1):
        if not msg:
            continue
        mtype = chr(msg[0])
        counters[f"type_{mtype}"] += 1
        if mtype == "R":
            locate = struct.unpack_from(">H", msg, 1)[0]
            stock = _stock(msg[11:19])
            locate_to_symbol[locate] = stock
            if stock == symbol:
                target_locates.add(locate)
            continue

        if len(msg) < 11:
            continue
        locate = struct.unpack_from(">H", msg, 1)[0]
        if locate not in target_locates:
            continue
        ts = _u48(msg, 5)
        counters["target_messages"] += 1

        try:
            if mtype in {"A", "F"}:
                order_ref = struct.unpack_from(">Q", msg, 11)[0]
                side = chr(msg[19])
                shares = struct.unpack_from(">I", msg, 20)[0]
                price = struct.unpack_from(">I", msg, 32)[0]
                book.add(order_ref, side, shares, price)
                append_snapshot(ts, [0.0, _side_value(side), 0.0, float(shares), EVENT_CODE["submission"]])
            elif mtype == "E":
                order_ref = struct.unpack_from(">Q", msg, 11)[0]
                shares = struct.unpack_from(">I", msg, 19)[0]
                side_removed = book.reduce(order_ref, shares)
                aggressor = -_side_value(side_removed[0]) if side_removed else 0.0
                append_snapshot(ts, [aggressor, 0.0, 0.0, float(shares), EVENT_CODE["execution"]])
            elif mtype == "C":
                order_ref = struct.unpack_from(">Q", msg, 11)[0]
                shares = struct.unpack_from(">I", msg, 19)[0]
                side_removed = book.reduce(order_ref, shares)
                aggressor = -_side_value(side_removed[0]) if side_removed else 0.0
                append_snapshot(ts, [aggressor, 0.0, 0.0, float(shares), EVENT_CODE["execution"]])
            elif mtype == "X":
                order_ref = struct.unpack_from(">Q", msg, 11)[0]
                shares = struct.unpack_from(">I", msg, 19)[0]
                side_removed = book.reduce(order_ref, shares)
                cancel_side = _side_value(side_removed[0]) if side_removed else 0.0
                append_snapshot(ts, [0.0, 0.0, cancel_side, float(shares), EVENT_CODE["cancel_delete"]])
            elif mtype == "D":
                order_ref = struct.unpack_from(">Q", msg, 11)[0]
                side_removed = book.delete(order_ref)
                cancel_side = _side_value(side_removed[0]) if side_removed else 0.0
                size = float(side_removed[1]) if side_removed else 0.0
                append_snapshot(ts, [0.0, 0.0, cancel_side, size, EVENT_CODE["cancel_delete"]])
            elif mtype == "U":
                old_ref = struct.unpack_from(">Q", msg, 11)[0]
                new_ref = struct.unpack_from(">Q", msg, 19)[0]
                shares = struct.unpack_from(">I", msg, 27)[0]
                price = struct.unpack_from(">I", msg, 31)[0]
                side_replaced = book.replace(old_ref, new_ref, shares, price)
                passive_side = _side_value(side_replaced[0]) if side_replaced else 0.0
                append_snapshot(ts, [0.0, passive_side, 0.0, float(shares), EVENT_CODE["replace"]])
        except (struct.error, IndexError):
            counters["parse_errors"] += 1

        if max_events and len(states) >= max_events:
            break
        if progress_every and i % progress_every == 0:
            print(
                f"parsed_messages={i} target_messages={counters.get('target_messages', 0)} "
                f"snapshots={len(states)}",
                flush=True,
            )

    if len(states) < 2:
        raise ValueError(f"Not enough reconstructed snapshots for {symbol}: {len(states)}")

    raw_states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(action_rows[1:], dtype=np.float32)
    size_scale = float(max(np.quantile(np.asarray(action_sizes, dtype=np.float64), 0.99), 1.0))
    actions[:, 3] = np.clip(actions[:, 3] / size_scale, 0.0, 1.0)
    return {
        "raw_states": raw_states,
        "actions": actions,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "size_scale": size_scale,
        "counters": dict(counters),
        "target_locates": sorted(target_locates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itch-file", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("data/processed/snapshot_l2") / args.dataset_id
    parsed = parse_symbol(
        Path(args.itch_file),
        args.symbol,
        max_events=args.max_events,
        progress_every=args.progress_every,
    )
    raw_states = parsed["raw_states"]
    split = np.zeros(len(raw_states), dtype=np.int8)
    n = len(raw_states)
    split[int(n * 0.7):int(n * 0.85)] = 1
    split[int(n * 0.85):] = 2
    states, normalization = _fit_normalization(raw_states, split)
    day_id = np.full(len(raw_states), -1, dtype=np.int16)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "states_raw.npy", raw_states)
    np.save(output_dir / "states_merged.npy", raw_states)
    np.save(output_dir / "actions.npy", parsed["actions"])
    np.save(output_dir / "timestamps.npy", parsed["timestamps"])
    np.save(output_dir / "day_id.npy", day_id)
    np.save(output_dir / "split.npy", split)
    _write_json(output_dir / "normalization.json", normalization)
    metadata = {
        "dataset_id": args.dataset_id,
        "dataset_status": "prepared_snapshot_l2_from_public_nasdaq_itch",
        "symbol": args.symbol.upper(),
        "source": "public_nasdaq_itch_sample",
        "itch_file": args.itch_file,
        "state_shape": list(states.shape),
        "action_shape": list(parsed["actions"].shape),
        "has_message_data": False,
        "has_raw_itch_source": True,
        "processed_event_l3_available": False,
        "has_true_timestamps": True,
        "has_true_day_boundaries": False,
        "day_id_status": "single_public_sample_file_not_day_aware_split",
        "action_label_status": "itch_event_level_diagnostic_not_causal",
        "action_size_scale": parsed["size_scale"],
        "target_locates": parsed["target_locates"],
        "counters": parsed["counters"],
        "limitations": [
            "Public ITCH sample file, not a 9-day LOBSTER benchmark.",
            "Single-file split is contiguous, not true multi-day train/val/test.",
        ],
    }
    _write_json(output_dir / "metadata.json", metadata)
    (output_dir / "DATASET_CARD.md").write_text(
        f"# {args.dataset_id} Dataset Card\n\n"
        "Status: `prepared_snapshot_l2_from_public_nasdaq_itch`.\n\n"
        f"Source file: `{args.itch_file}`.\n\n"
        "This is a public Nasdaq ITCH sample-derived snapshot dataset, not a LOBSTER 9-day benchmark.\n"
    )
    (output_dir / "manifest.yaml").write_text(
        f"dataset_id: {args.dataset_id}\nfiles:\n"
        "  states: states.npy\n"
        "  states_raw: states_raw.npy\n"
        "  states_merged: states_merged.npy\n"
        "  actions: actions.npy\n"
        "  timestamps: timestamps.npy\n"
        "  day_id: day_id.npy\n"
        "  split: split.npy\n"
        "  normalization: normalization.json\n"
        "  metadata: metadata.json\n"
        "  dataset_card: DATASET_CARD.md\n"
    )
    _update_manifest(
        Path(args.manifest),
        args.dataset_id,
        args.symbol.upper(),
        output_dir,
        has_true_day_boundaries=False,
    )
    report = {
        "dataset_id": args.dataset_id,
        "output_dir": str(output_dir),
        "states_shape": list(states.shape),
        "actions_shape": list(parsed["actions"].shape),
        "normalization": normalization,
        "counters": parsed["counters"],
    }
    _write_json(output_dir / "build_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a processed snapshot_l2 dataset from LOBSTER message/orderbook files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.constraints import check_lobster_orderbook_invariants, validate_lobster_messages
from genlob.data.lobster import (
    LOBSTER_ASK_DUMMY_PRICE,
    LOBSTER_BID_DUMMY_PRICE,
    infer_orderbook_depth,
    load_lobster_pair,
    orderbook_to_numpy,
)
from genlob.data.lobster_actions import derive_lobster_action_labels, lobster_actions_to_array


@dataclass(frozen=True)
class LobsterPair:
    date_id: str
    message_path: Path
    orderbook_path: Path


def _extract_date(text: str) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else None


def _pair_sort_key(pair: LobsterPair) -> tuple[str, str]:
    return pair.date_id, str(pair.message_path)


def discover_lobster_pairs(raw_dir: Path) -> list[LobsterPair]:
    """Find message/orderbook CSV pairs in a LOBSTER raw directory."""
    pairs: list[LobsterPair] = []
    seen: set[tuple[Path, Path]] = set()

    for directory in sorted([p for p in raw_dir.rglob("*") if p.is_dir()] + [raw_dir]):
        direct_message = directory / "message.csv"
        direct_orderbook = directory / "orderbook_10.csv"
        if direct_message.exists() and direct_orderbook.exists():
            key = (direct_message.resolve(), direct_orderbook.resolve())
            if key not in seen:
                seen.add(key)
                pairs.append(
                    LobsterPair(
                        date_id=_extract_date(str(directory)) or directory.name,
                        message_path=direct_message,
                        orderbook_path=direct_orderbook,
                    )
                )

    for message_path in sorted(raw_dir.rglob("*message*.csv")):
        candidates = sorted(message_path.parent.glob("*orderbook*.csv"))
        if len(candidates) != 1:
            continue
        orderbook_path = candidates[0]
        key = (message_path.resolve(), orderbook_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            LobsterPair(
                date_id=_extract_date(message_path.name) or _extract_date(str(message_path.parent)) or message_path.parent.name,
                message_path=message_path,
                orderbook_path=orderbook_path,
            )
        )

    return sorted(pairs, key=_pair_sort_key)


def _orderbook_to_raw_states(orderbook_values: np.ndarray, depth: int) -> np.ndarray:
    values = orderbook_values.reshape(orderbook_values.shape[0], depth, 4)
    ask_price = values[:, :, 0].astype(np.float32) / 10_000.0
    ask_size = values[:, :, 1].astype(np.float32)
    bid_price = values[:, :, 2].astype(np.float32) / 10_000.0
    bid_size = values[:, :, 3].astype(np.float32)
    prices = np.concatenate([ask_price, bid_price], axis=1)
    volumes = np.concatenate([ask_size, bid_size], axis=1)
    return np.stack([prices, volumes], axis=1).astype(np.float32)


def _valid_orderbook_rows(orderbook_values: np.ndarray, depth: int) -> np.ndarray:
    values = orderbook_values.reshape(orderbook_values.shape[0], depth, 4)
    ask_price = values[:, :, 0]
    ask_size = values[:, :, 1]
    bid_price = values[:, :, 2]
    bid_size = values[:, :, 3]
    no_dummy = (ask_price < LOBSTER_ASK_DUMMY_PRICE).all(axis=1) & (bid_price > LOBSTER_BID_DUMMY_PRICE).all(axis=1)
    no_negative = (ask_size >= 0).all(axis=1) & (bid_size >= 0).all(axis=1)
    ask_ok = (ask_price[:, :-1] <= ask_price[:, 1:]).all(axis=1)
    bid_ok = (bid_price[:, :-1] >= bid_price[:, 1:]).all(axis=1)
    spread_ok = ask_price[:, 0] > bid_price[:, 0]
    return no_dummy & no_negative & ask_ok & bid_ok & spread_ok


def _day_split_values(n_days: int, train_ratio: float, val_ratio: float) -> np.ndarray:
    if n_days <= 0:
        raise ValueError("Need at least one day")
    if n_days == 1:
        return np.asarray([0], dtype=np.int8)
    if n_days == 2:
        return np.asarray([0, 2], dtype=np.int8)
    n_train = max(1, int(round(n_days * train_ratio)))
    n_val = max(1, int(round(n_days * val_ratio)))
    if n_train + n_val >= n_days:
        n_train = max(1, n_days - 2)
        n_val = 1
    n_test = n_days - n_train - n_val
    return np.asarray([0] * n_train + [1] * n_val + [2] * n_test, dtype=np.int8)


def _compute_normalized_states(
    raw_states: np.ndarray,
    split: np.ndarray,
    price_scale: float,
    volume_scale: float,
    volume_clip_quantile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = raw_states[split == 0]
    if train.size == 0:
        raise ValueError("Train split is empty; cannot fit normalization")

    train_prices_scaled = train[:, 0, :] * price_scale
    price_global_std = float(np.std(train_prices_scaled))
    if not np.isfinite(price_global_std) or price_global_std <= 0:
        raise ValueError(f"Invalid train price std: {price_global_std}")

    train_volumes = train[:, 1, :]
    volume_clip = float(np.quantile(train_volumes, volume_clip_quantile))
    volume_clip = max(volume_clip, 1.0)

    prices = raw_states[:, 0, :]
    ask1 = prices[:, 0]
    bid1 = prices[:, 10]
    mid = ((ask1 + bid1) / 2.0)[:, None]
    norm_prices = ((prices - mid) * price_scale / price_global_std).astype(np.float32)
    norm_volumes = (np.sqrt(np.clip(raw_states[:, 1, :], 0.0, volume_clip)) / volume_scale).astype(np.float32)
    states = np.stack([norm_prices, norm_volumes], axis=1).astype(np.float32)

    train_spreads = train[:, 0, 0] - train[:, 0, 10]
    positive_spreads = train_spreads[train_spreads > 0]
    min_spread_raw = float(positive_spreads.min()) if positive_spreads.size else 0.01
    min_spread_normalized = float(max(min_spread_raw * price_scale / price_global_std, 1e-8))

    normalization = {
        "price_transform": "mid_relative_train_price_std",
        "volume_transform": "sqrt_clip_train_quantile_scale",
        "fit_split": "train_only",
        "price_scale": price_scale,
        "price_global_std": price_global_std,
        "volume_clip_quantile": volume_clip_quantile,
        "volume_clip": volume_clip,
        "volume_scale": volume_scale,
        "min_spread_raw": min_spread_raw,
        "min_spread_normalized": min_spread_normalized,
        "levels": 10,
        "state_shape": [2, 20],
    }
    return states, normalization


def _zero_action(action_dim: int = 5) -> np.ndarray:
    return np.zeros(action_dim, dtype=np.float32)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_dataset_card(path: Path, metadata: dict[str, Any]) -> None:
    text = f"""# {metadata['dataset_id']} Dataset Card

Status: `{metadata['dataset_status']}`.

Symbol: `{metadata['symbol']}`.

Source: LOBSTER message/orderbook files under `{metadata['raw_dir']}`.

Rows: `{metadata['state_shape'][0]}`.

Levels: `{metadata['level']}`.

This dataset uses the GenLOB `snapshot_l2` schema:

- `states.npy`: `[T, 2, 20]`, normalized Level-2 snapshots.
- `states_raw.npy` and `states_merged.npy`: raw dollar prices and share volumes.
- `actions.npy`: `[T-1, 5]`, event-derived diagnostic action labels aligned to snapshot transitions.
- `timestamps.npy`: LOBSTER message times in seconds after midnight.
- `day_id.npy`: integer trading-day ids.
- `split.npy`: day-aware train/val/test split.
- `normalization.json`: train-only normalization parameters.

Action labels are event-derived from LOBSTER messages but should not be treated
as causal counterfactual labels without the event-level evaluation protocol.
"""
    path.write_text(text)


def _write_manifest_yaml(path: Path, dataset_id: str, files: dict[str, str]) -> None:
    lines = [
        f"dataset_id: {dataset_id}",
        "files:",
    ]
    for key, value in files.items():
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def _update_dataset_manifest(
    manifest_path: Path,
    dataset_id: str,
    symbol: str,
    output_dir: Path,
    has_true_timestamps: bool,
) -> None:
    manifest = {"datasets": {}}
    if manifest_path.exists():
        with open(manifest_path) as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            manifest = loaded
    manifest.setdefault("datasets", {})
    manifest["datasets"][dataset_id] = {
        "status": "prepared_snapshot_l2_from_lobster",
        "symbol": symbol,
        "source": "lobster_message_orderbook",
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
        "has_message_data": True,
        "has_true_timestamps": has_true_timestamps,
        "has_true_day_boundaries": True,
        "action_label_status": "lobster_event_level_diagnostic",
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    pairs = discover_lobster_pairs(raw_dir)
    if args.max_days:
        pairs = pairs[: args.max_days]
    if args.dry_run:
        return {
            "symbol": args.symbol,
            "raw_dir": str(raw_dir),
            "n_pairs": len(pairs),
            "pairs": [
                {"date_id": pair.date_id, "message": str(pair.message_path), "orderbook": str(pair.orderbook_path)}
                for pair in pairs
            ],
        }
    if not pairs:
        raise FileNotFoundError(
            f"No LOBSTER pairs found under {raw_dir}. Expected date directories with "
            "message.csv/orderbook_10.csv or files matching *message*.csv and *orderbook*.csv."
        )

    raw_states_parts: list[np.ndarray] = []
    timestamps_parts: list[np.ndarray] = []
    day_id_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    labels_parts = []
    day_row_counts: list[int] = []
    pair_reports: list[dict[str, Any]] = []

    day_split = _day_split_values(len(pairs), args.train_ratio, args.val_ratio)
    for day_idx, pair in enumerate(pairs):
        messages, orderbook = load_lobster_pair(
            pair.message_path,
            pair.orderbook_path,
            depth=args.depth,
            has_header=args.has_header,
        )
        orderbook_values = orderbook_to_numpy(orderbook)
        depth = infer_orderbook_depth(orderbook_values.shape[1])
        if depth < 10:
            raise ValueError(f"{pair.orderbook_path}: need at least depth 10, got {depth}")
        if depth != 10:
            orderbook_values = orderbook_values[:, :40]
            depth = 10

        grammar = validate_lobster_messages(messages)
        invariant = check_lobster_orderbook_invariants(orderbook_values, depth=depth)
        valid_rows = _valid_orderbook_rows(orderbook_values, depth=depth)
        invalid_count = int((~valid_rows).sum())
        if invalid_count and not args.drop_invalid_rows:
            raise ValueError(
                f"{pair.date_id}: {invalid_count} invalid/dummy/crossed rows found. "
                "Pass --drop-invalid-rows to remove them, or inspect the raw day."
            )
        if invalid_count:
            messages = messages.loc[valid_rows].reset_index(drop=True)
            orderbook_values = orderbook_values[valid_rows]

        raw_states = _orderbook_to_raw_states(orderbook_values, depth=depth)
        labels = derive_lobster_action_labels(messages)
        raw_states_parts.append(raw_states)
        timestamps_parts.append(messages["Time"].to_numpy(dtype=np.float64))
        day_id_parts.append(np.full(len(raw_states), day_idx, dtype=np.int16))
        split_parts.append(np.full(len(raw_states), int(day_split[day_idx]), dtype=np.int8))
        labels_parts.append(labels)
        day_row_counts.append(len(raw_states))
        pair_reports.append(
            {
                "date_id": pair.date_id,
                "message": str(pair.message_path),
                "orderbook": str(pair.orderbook_path),
                "rows": int(len(raw_states)),
                "invalid_rows_dropped": invalid_count,
                "grammar_ok": grammar.ok,
                "grammar_violations": grammar.total_violations,
                "raw_invariants_ok_before_filter": invariant.ok,
                "raw_invariants_before_filter": invariant.as_dict(),
            }
        )

    raw_states = np.concatenate(raw_states_parts, axis=0)
    timestamps = np.concatenate(timestamps_parts, axis=0)
    day_id = np.concatenate(day_id_parts, axis=0)
    split = np.concatenate(split_parts, axis=0)

    size_values = np.concatenate([labels["Size"].to_numpy(dtype=np.float64) for labels in labels_parts])
    action_size_scale = float(max(np.quantile(size_values, 0.99), 1.0))
    actions_by_row: list[np.ndarray] = []
    for day_idx, labels in enumerate(labels_parts):
        actions_day = lobster_actions_to_array(labels, size_scale=action_size_scale)
        n = len(actions_day)
        if n > 1:
            actions_by_row.extend(actions_day[1:])
        if day_idx < len(labels_parts) - 1:
            actions_by_row.append(_zero_action(actions_day.shape[1] if actions_day.ndim == 2 else 5))
    actions = np.asarray(actions_by_row, dtype=np.float32)
    if len(actions) != len(raw_states) - 1:
        raise RuntimeError(f"Action/state alignment failed: actions={len(actions)} states={len(raw_states)}")

    states, normalization = _compute_normalized_states(
        raw_states,
        split,
        price_scale=args.price_scale,
        volume_scale=args.volume_scale,
        volume_clip_quantile=args.volume_clip_quantile,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("data/processed/snapshot_l2") / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "states_raw.npy", raw_states)
    np.save(output_dir / "states_merged.npy", raw_states)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "timestamps.npy", timestamps)
    np.save(output_dir / "day_id.npy", day_id)
    np.save(output_dir / "split.npy", split)

    metadata = {
        "dataset_id": args.dataset_id,
        "dataset_status": "prepared_snapshot_l2_from_lobster",
        "symbol": args.symbol,
        "source": "lobster_message_orderbook",
        "raw_dir": str(raw_dir),
        "source_files_available": True,
        "has_message_data": True,
        "has_true_timestamps": True,
        "has_true_day_boundaries": True,
        "timestamp_status": "lobster_seconds_after_midnight",
        "day_id_status": "true_trading_day_boundaries_from_raw_files",
        "level": 10,
        "state_shape": list(states.shape),
        "raw_state_shape": list(raw_states.shape),
        "action_shape": list(actions.shape),
        "state_schema": {
            "shape": "[T, 2, 20]",
            "channel_0": "normalized prices: ask L1-L10, bid L1-L10",
            "channel_1": "normalized volumes: ask L1-L10, bid L1-L10",
        },
        "action_schema": {
            "shape": "[T-1, 5]",
            "columns": ["aggressor_side", "passive_side", "cancel_side", "normalized_size", "event_code"],
            "status": "LOBSTER event-derived diagnostic action; formal causality still requires event-level protocol",
        },
        "normalization": normalization,
        "split_policy": {
            "type": "day_aware_contiguous_by_file_order",
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "day_split_values": day_split.astype(int).tolist(),
            "day_row_counts": day_row_counts,
        },
        "action_size_scale": action_size_scale,
        "pairs": pair_reports,
        "limitations": [
            "Snapshot-level dataset; message-level baselines still need event_l3 export.",
            "Action vectors are event-derived diagnostics, not observed counterfactual outcomes.",
        ],
    }
    _write_json(output_dir / "normalization.json", normalization)
    _write_json(output_dir / "metadata.json", metadata)
    _write_dataset_card(output_dir / "DATASET_CARD.md", metadata)
    _write_manifest_yaml(
        output_dir / "manifest.yaml",
        args.dataset_id,
        {
            "states": "states.npy",
            "states_raw": "states_raw.npy",
            "states_merged": "states_merged.npy",
            "actions": "actions.npy",
            "timestamps": "timestamps.npy",
            "day_id": "day_id.npy",
            "split": "split.npy",
            "normalization": "normalization.json",
            "metadata": "metadata.json",
            "dataset_card": "DATASET_CARD.md",
        },
    )
    _update_dataset_manifest(
        Path(args.manifest),
        args.dataset_id,
        args.symbol,
        output_dir,
        has_true_timestamps=True,
    )

    report = {
        "dataset_id": args.dataset_id,
        "symbol": args.symbol,
        "output_dir": str(output_dir),
        "n_days": len(pairs),
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "normalization": normalization,
        "action_size_scale": action_size_scale,
        "split_counts": {
            "train": int((split[:-1] == 0).sum()),
            "val": int((split[:-1] == 1).sum()),
            "test": int((split[:-1] == 2).sum()),
        },
        "pair_reports": pair_reports,
    }
    _write_json(output_dir / "build_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--price-scale", type=float, default=100.0)
    parser.add_argument("--volume-scale", type=float, default=15.0)
    parser.add_argument("--volume-clip-quantile", type=float, default=0.99)
    parser.add_argument("--drop-invalid-rows", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_dataset(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

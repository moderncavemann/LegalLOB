#!/usr/bin/env python3
"""Build multi-symbol RTH snapshot panels from public NASDAQ ITCH samples."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_public_itch_snapshot_l2 import (
    EVENT_CODE,
    Book,
    _iter_itch_messages,
    _side_value,
    _stock,
    _u48,
)


RTH_START_SECOND = 9 * 60 * 60 + 30 * 60
RTH_END_SECOND = 16 * 60 * 60
PARSER_VERSION = 2


def infer_day_id(path: Path) -> int:
    match = re.search(r"(\d{2})(\d{2})(\d{4})", path.name)
    if not match:
        raise ValueError(f"Cannot infer MMDDYYYY date from {path.name}")
    month, day, year = match.groups()
    return int(f"{year}{month}{day}")


def _top10_float64(book: Book) -> np.ndarray | None:
    if len(book.ask) < 10 or len(book.bid) < 10:
        return None
    ask_prices = sorted(book.ask)[:10]
    bid_prices = sorted(book.bid, reverse=True)[:10]
    prices = np.asarray(ask_prices + bid_prices, dtype=np.float64) / 10_000.0
    volumes = np.asarray(
        [book.ask[p] for p in ask_prices] + [book.bid[p] for p in bid_prices],
        dtype=np.float64,
    )
    return np.stack([prices, volumes], axis=0)


def _valid_snapshot(state: np.ndarray) -> bool:
    ask = state[0, :10]
    bid = state[0, 10:]
    volume = state[1]
    return bool(
        np.isfinite(state).all()
        and (ask > 0).all()
        and (bid > 0).all()
        and (volume >= 0).all()
        and np.all(ask[:-1] <= ask[1:])
        and np.all(bid[:-1] >= bid[1:])
        and ask[0] > bid[0]
    )


class ChunkWriter:
    """Write bounded in-memory chunks so full-day panels do not fill RAM."""

    def __init__(self, root: Path, chunk_size: int, max_snapshots: int | None = None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.max_snapshots = max_snapshots
        self.count = 0
        self.chunk_index = 0
        self.states: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.actions: list[list[float]] = []
        self.days: list[int] = []
        self.source_indices: list[int] = []

    @property
    def capped(self) -> bool:
        return self.max_snapshots is not None and self.count >= self.max_snapshots

    def append(
        self,
        state: np.ndarray,
        timestamp_seconds: float,
        action: list[float],
        day_id: int,
        source_index: int,
    ) -> None:
        if self.capped:
            return
        self.states.append(np.asarray(state, dtype=np.float64))
        self.timestamps.append(float(timestamp_seconds))
        self.actions.append(action)
        self.days.append(day_id)
        self.source_indices.append(source_index)
        self.count += 1
        if len(self.states) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.states:
            return
        suffix = f"{self.chunk_index:06d}.npy"
        np.save(self.root / f"states_{suffix}", np.asarray(self.states, dtype=np.float64))
        np.save(self.root / f"timestamps_{suffix}", np.asarray(self.timestamps, dtype=np.float64))
        np.save(self.root / f"actions_{suffix}", np.asarray(self.actions, dtype=np.float32))
        np.save(self.root / f"days_{suffix}", np.asarray(self.days, dtype=np.int32))
        np.save(self.root / f"source_{suffix}", np.asarray(self.source_indices, dtype=np.int64))
        self.states.clear()
        self.timestamps.clear()
        self.actions.clear()
        self.days.clear()
        self.source_indices.clear()
        self.chunk_index += 1


class LastObservationPerSecond:
    """Emit the last valid book in each RTH second with carry-forward gaps."""

    def __init__(self, writer: ChunkWriter, day_id: int):
        self.writer = writer
        self.day_id = day_id
        self.second: int | None = None
        self.state: np.ndarray | None = None
        self.action: list[float] | None = None
        self.source_index = -1

    def _emit(self, second: int, action: list[float]) -> None:
        if self.state is None:
            return
        self.writer.append(self.state, float(second + 1), action, self.day_id, self.source_index)

    def update(self, timestamp_ns: int, state: np.ndarray, action: list[float], source_index: int) -> None:
        second = timestamp_ns // 1_000_000_000
        if not RTH_START_SECOND <= second < RTH_END_SECOND:
            return
        if self.second is None:
            self.second = int(second)
        elif second > self.second:
            self._emit(self.second, self.action or [0.0] * 5)
            for missing_second in range(self.second + 1, int(second)):
                self._emit(missing_second, [0.0] * 5)
                if self.writer.capped:
                    return
            self.second = int(second)
        self.state = np.asarray(state, dtype=np.float64).copy()
        self.action = list(action)
        self.source_index = source_index

    def finish(self, fill_to_close: bool = True) -> None:
        if self.second is None or self.state is None or self.writer.capped:
            return
        self._emit(self.second, self.action or [0.0] * 5)
        if fill_to_close:
            for second in range(self.second + 1, RTH_END_SECOND):
                self._emit(second, [0.0] * 5)
                if self.writer.capped:
                    break


def _apply_order_event(msg: bytes, book: Book) -> list[float] | None:
    mtype = chr(msg[0])
    try:
        if mtype in {"A", "F"}:
            order_ref = struct.unpack_from(">Q", msg, 11)[0]
            side = chr(msg[19])
            shares = struct.unpack_from(">I", msg, 20)[0]
            price = struct.unpack_from(">I", msg, 32)[0]
            book.add(order_ref, side, shares, price)
            return [0.0, _side_value(side), 0.0, float(shares), EVENT_CODE["submission"]]
        if mtype in {"E", "C"}:
            order_ref = struct.unpack_from(">Q", msg, 11)[0]
            shares = struct.unpack_from(">I", msg, 19)[0]
            removed = book.reduce(order_ref, shares)
            aggressor = -_side_value(removed[0]) if removed else 0.0
            return [aggressor, 0.0, 0.0, float(shares), EVENT_CODE["execution"]]
        if mtype == "X":
            order_ref = struct.unpack_from(">Q", msg, 11)[0]
            shares = struct.unpack_from(">I", msg, 19)[0]
            removed = book.reduce(order_ref, shares)
            side = _side_value(removed[0]) if removed else 0.0
            return [0.0, 0.0, side, float(shares), EVENT_CODE["cancel_delete"]]
        if mtype == "D":
            order_ref = struct.unpack_from(">Q", msg, 11)[0]
            removed = book.delete(order_ref)
            side = _side_value(removed[0]) if removed else 0.0
            size = float(removed[1]) if removed else 0.0
            return [0.0, 0.0, side, size, EVENT_CODE["cancel_delete"]]
        if mtype == "U":
            old_ref = struct.unpack_from(">Q", msg, 11)[0]
            new_ref = struct.unpack_from(">Q", msg, 19)[0]
            shares = struct.unpack_from(">I", msg, 27)[0]
            price = struct.unpack_from(">I", msg, 31)[0]
            replaced = book.replace(old_ref, new_ref, shares, price)
            side = _side_value(replaced[0]) if replaced else 0.0
            return [0.0, side, 0.0, float(shares), EVENT_CODE["replace"]]
    except (IndexError, struct.error):
        return None
    return None


@dataclass
class ParseResult:
    day_id: int
    messages: int
    truncated: bool
    counters: dict[str, dict[str, int]]


def parse_file(
    path: Path,
    symbols: tuple[str, ...],
    modes: tuple[str, ...],
    staging_root: Path,
    chunk_size: int,
    max_snapshots: int | None,
    progress_every: int,
) -> ParseResult:
    day_id = infer_day_id(path)
    books = {symbol: Book() for symbol in symbols}
    counters: dict[str, dict[str, int]] = {symbol: defaultdict(int) for symbol in symbols}
    source_sequence = {symbol: 0 for symbol in symbols}
    locate_to_symbol: dict[int, str] = {}
    writers: dict[tuple[str, str], ChunkWriter] = {}
    samplers: dict[str, LastObservationPerSecond] = {}
    for symbol in symbols:
        for mode in modes:
            writer = ChunkWriter(
                staging_root / str(day_id) / symbol / mode,
                chunk_size=chunk_size,
                max_snapshots=max_snapshots,
            )
            writers[(symbol, mode)] = writer
            if mode == "1s":
                samplers[symbol] = LastObservationPerSecond(writer, day_id)

    truncated = False
    message_count = 0
    for message_count, msg in enumerate(_iter_itch_messages(path), start=1):
        if not msg:
            continue
        mtype = chr(msg[0])
        if mtype == "R":
            locate = struct.unpack_from(">H", msg, 1)[0]
            symbol = _stock(msg[11:19]).upper()
            if symbol in books:
                locate_to_symbol[locate] = symbol
            continue
        if len(msg) < 11:
            continue
        timestamp_ns = _u48(msg, 5)
        if timestamp_ns // 1_000_000_000 >= RTH_END_SECOND and len(set(locate_to_symbol.values())) == len(symbols):
            break
        locate = struct.unpack_from(">H", msg, 1)[0]
        symbol = locate_to_symbol.get(locate)
        if symbol is None:
            continue
        action = _apply_order_event(msg, books[symbol])
        if action is None:
            continue
        source_sequence[symbol] += 1
        counters[symbol][f"type_{mtype}"] += 1
        state = _top10_float64(books[symbol])
        if state is None:
            counters[symbol]["incomplete_top10"] += 1
            continue
        if not _valid_snapshot(state):
            counters[symbol]["invalid_snapshot_dropped"] += 1
            continue
        second = timestamp_ns // 1_000_000_000
        if not RTH_START_SECOND <= second < RTH_END_SECOND:
            continue
        if "event" in modes:
            writers[(symbol, "event")].append(
                state,
                timestamp_ns / 1e9,
                action,
                day_id,
                source_sequence[symbol],
            )
        if "1s" in modes:
            samplers[symbol].update(timestamp_ns, state, action, source_sequence[symbol])

        if max_snapshots is not None:
            all_capped = all(writers[(symbol, mode)].capped for symbol in symbols for mode in modes)
            if all_capped:
                truncated = True
                break
        if progress_every and message_count % progress_every == 0:
            summary = ", ".join(
                f"{symbol}:" + "/".join(f"{mode}={writers[(symbol, mode)].count}" for mode in modes)
                for symbol in symbols
            )
            print(f"{path.name}: messages={message_count:,}; {summary}", flush=True)

    for sampler in samplers.values():
        sampler.finish(fill_to_close=not truncated)
    for writer in writers.values():
        writer.flush()
    return ParseResult(
        day_id=day_id,
        messages=message_count,
        truncated=truncated,
        counters={symbol: dict(values) for symbol, values in counters.items()},
    )


def _chunk_files(staging_root: Path, symbol: str, mode: str, prefix: str) -> list[Path]:
    return sorted(staging_root.glob(f"*/{symbol}/{mode}/{prefix}_*.npy"))


def _parse_date_list(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(item.strip().replace("-", "")) for item in value.split(",") if item.strip()}


def assign_splits(
    day_id: np.ndarray,
    train_dates: set[int] | None = None,
    val_dates: set[int] | None = None,
    test_dates: set[int] | None = None,
    purge_gap: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    unique_days = sorted(int(day) for day in np.unique(day_id))
    explicit = bool(train_dates or val_dates or test_dates)
    if explicit:
        mapping = {
            **{day: 0 for day in train_dates or set()},
            **{day: 1 for day in val_dates or set()},
            **{day: 2 for day in test_dates or set()},
        }
        missing = set(unique_days) - set(mapping)
        if missing:
            raise ValueError(f"Explicit split does not assign days: {sorted(missing)}")
        split = np.asarray([mapping[int(day)] for day in day_id], dtype=np.int8)
        policy = "explicit_day_split"
    elif len(unique_days) >= 3:
        mapping = {day: 0 for day in unique_days[:-2]}
        mapping[unique_days[-2]] = 1
        mapping[unique_days[-1]] = 2
        split = np.asarray([mapping[int(day)] for day in day_id], dtype=np.int8)
        policy = "default_last_two_days_val_test"
    elif len(unique_days) == 1:
        n = len(day_id)
        b1 = int(n * 0.7)
        b2 = int(n * 0.85)
        split = np.zeros(n, dtype=np.int8)
        split[b1:b2] = 1
        split[b2:] = 2
        split[max(0, b1 - purge_gap):min(n, b1 + purge_gap)] = -1
        split[max(0, b2 - purge_gap):min(n, b2 + purge_gap)] = -1
        mapping = {unique_days[0]: "chronological_70_15_15"}
        policy = "single_day_chronological_with_purge"
    else:
        raise ValueError("Two-day panels require explicit train/val/test date assignments")
    return split, {"policy": policy, "day_mapping": mapping, "purge_gap": purge_gap}


def _sample_rows(indices: np.ndarray, max_rows: int) -> np.ndarray:
    if len(indices) <= max_rows:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_rows, dtype=np.int64)
    return indices[positions]


def _hardlink_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _source_identity(
    path: Path,
    symbols: tuple[str, ...],
    modes: tuple[str, ...],
    max_snapshots: int | None,
) -> dict[str, Any]:
    """Return the inputs that must match before a completed day can be reused."""
    checksum_path = path.with_name(path.name + ".sha256")
    checksum = None
    if checksum_path.exists():
        tokens = checksum_path.read_text().strip().split()
        checksum = tokens[0] if tokens else None
    return {
        "parser_version": PARSER_VERSION,
        "source_path": str(path.resolve()),
        "source_size": path.stat().st_size,
        "source_sha256": checksum,
        "symbols": list(symbols),
        "modes": list(modes),
        "max_snapshots_per_symbol": max_snapshots,
    }


def _load_completed_parse(
    marker: Path,
    identity: dict[str, Any],
) -> ParseResult | None:
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text())
        if payload.get("identity") != identity:
            return None
        report = payload["report"]
        return ParseResult(
            day_id=int(report["day_id"]),
            messages=int(report["messages"]),
            truncated=bool(report["truncated"]),
            counters=report["counters"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def finalize_dataset(
    staging_root: Path,
    output_root: Path,
    dataset_prefix: str,
    symbol: str,
    mode: str,
    train_dates: set[int],
    val_dates: set[int],
    test_dates: set[int],
    purge_gap: int,
    max_timestamp_gap: float,
    tick_size: float,
    ticks_per_price_unit: float,
    normalization_sample_rows: int,
    truncated: bool,
) -> dict[str, Any]:
    state_files = _chunk_files(staging_root, symbol, mode, "states")
    if not state_files:
        raise ValueError(f"No chunks for {symbol} {mode}")
    total = sum(len(np.load(path, mmap_mode="r")) for path in state_files)
    dataset_id = f"{dataset_prefix}_{symbol}_{mode}"
    output_dir = output_root / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "states_raw.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float64, shape=(total, 2, 20))
    timestamps = np.lib.format.open_memmap(output_dir / "timestamps.npy", mode="w+", dtype=np.float64, shape=(total,))
    day_id = np.lib.format.open_memmap(output_dir / "day_id.npy", mode="w+", dtype=np.int32, shape=(total,))
    source_idx = np.lib.format.open_memmap(output_dir / "source_row_idx.npy", mode="w+", dtype=np.int64, shape=(total,))
    state_actions_path = output_dir / ".state_actions.npy"
    state_actions = np.lib.format.open_memmap(state_actions_path, mode="w+", dtype=np.float32, shape=(total, 5))

    offset = 0
    for state_path in state_files:
        suffix = state_path.name.removeprefix("states_")
        base = state_path.parent
        states_chunk = np.load(state_path, mmap_mode="r")
        n = len(states_chunk)
        raw[offset:offset + n] = states_chunk
        timestamps[offset:offset + n] = np.load(base / f"timestamps_{suffix}", mmap_mode="r")
        day_id[offset:offset + n] = np.load(base / f"days_{suffix}", mmap_mode="r")
        source_idx[offset:offset + n] = np.load(base / f"source_{suffix}", mmap_mode="r")
        state_actions[offset:offset + n] = np.load(base / f"actions_{suffix}", mmap_mode="r")
        offset += n
    raw.flush()
    timestamps.flush()
    day_id.flush()
    source_idx.flush()
    state_actions.flush()

    split, split_policy = assign_splits(
        np.asarray(day_id),
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        purge_gap=purge_gap,
    )
    np.save(output_dir / "split.npy", split)
    valid = split >= 0
    breaks = np.ones(total, dtype=bool)
    if total > 1:
        breaks[1:] = (
            (day_id[1:] != day_id[:-1])
            | (split[1:] != split[:-1])
            | (timestamps[1:] < timestamps[:-1])
            | ((timestamps[1:] - timestamps[:-1]) > max_timestamp_gap)
        )
        if mode == "event":
            breaks[1:] |= source_idx[1:] != source_idx[:-1] + 1
    breaks |= ~valid
    raw_segment = np.cumsum(breaks, dtype=np.int64) - 1
    segment_id = raw_segment.astype(np.int32)
    segment_id[~valid] = -1
    np.save(output_dir / "segment_id.npy", segment_id)

    train_idx = np.flatnonzero(split == 0)
    if not len(train_idx):
        raise ValueError(f"No training rows in {dataset_id}")
    sampled = _sample_rows(train_idx, normalization_sample_rows)
    volume_sample = np.asarray(raw[sampled, 1, :], dtype=np.float64).reshape(-1)
    volume_clip = float(max(np.quantile(volume_sample, 0.99), 1.0))
    action_sample = np.asarray(state_actions[sampled, 3], dtype=np.float64)
    action_size_scale = float(max(np.quantile(action_sample, 0.99), 1.0))

    model_price_unit = tick_size * ticks_per_price_unit
    states = np.lib.format.open_memmap(output_dir / "states.npy", mode="w+", dtype=np.float32, shape=(total, 2, 20))
    block = 100_000
    for start in range(0, total, block):
        stop = min(total, start + block)
        raw_block = np.asarray(raw[start:stop])
        prices = raw_block[:, 0]
        mid = ((prices[:, 0] + prices[:, 10]) * 0.5)[:, None]
        states[start:stop, 0] = ((prices - mid) / model_price_unit).astype(np.float32)
        states[start:stop, 1] = (
            np.sqrt(np.clip(raw_block[:, 1], 0.0, volume_clip)) / 15.0
        ).astype(np.float32)
    states.flush()

    actions = np.lib.format.open_memmap(output_dir / "actions.npy", mode="w+", dtype=np.float32, shape=(total - 1, 5))
    actions[:] = state_actions[1:]
    actions[:, 3] = np.clip(actions[:, 3] / action_size_scale, 0.0, 1.0)
    invalid_transition = (segment_id[:-1] < 0) | (segment_id[1:] < 0) | (segment_id[:-1] != segment_id[1:])
    actions[invalid_transition] = 0.0
    actions.flush()
    del actions, state_actions
    state_actions_path.unlink()
    _hardlink_or_copy(raw_path, output_dir / "states_merged.npy")

    normalization = {
        "price_transform": "history_last_mid_tick_scaled_dynamic",
        "stored_state_price_transform": "per_frame_mid_tick_scaled_compatibility",
        "price_reference_protocol": "history_last_mid",
        "volume_transform": "sqrt_clip_train_quantile_scale",
        "fit_split": "train_only",
        "volume_clip_quantile": 0.99,
        "volume_clip": volume_clip,
        "volume_scale": 15.0,
        "normalization_fit_rows": int(len(sampled)),
        "normalization_fit_method": "deterministic_uniform_sample_of_train_rows",
        "tick_size_raw": tick_size,
        "ticks_per_price_unit": ticks_per_price_unit,
        "model_price_unit_raw": model_price_unit,
        "min_spread_raw": tick_size,
        "min_spread_normalized": 1.0 / ticks_per_price_unit,
        "min_level_gap_raw": tick_size,
        "min_level_gap_normalized": 1.0 / ticks_per_price_unit,
        "levels": 10,
        "state_shape": [2, 20],
    }
    _write_json(output_dir / "normalization.json", normalization)
    metadata = {
        "dataset_id": dataset_id,
        "dataset_status": "smoke_truncated" if truncated else "prepared_public_itch_multiday_rth",
        "symbol": symbol,
        "source": "nasdaq_public_itch_samples",
        "sampling": mode,
        "session": "regular_hours_0930_1600_America_New_York",
        "state_shape": [total, 2, 20],
        "action_shape": [total - 1, 5],
        "days": sorted(int(value) for value in np.unique(day_id)),
        "split_policy": split_policy,
        "segment_count": int(len(np.unique(segment_id[segment_id >= 0]))),
        "has_true_timestamps": True,
        "has_true_day_id": True,
        "has_message_data": False,
        "has_raw_itch_source": True,
        "processed_event_l3_available": False,
        "action_label_status": "itch_event_proxy_diagnostic_not_causal",
        "normalization": normalization,
        "scope_notes": [
            "Public sample days are non-consecutive.",
            "Snapshot action proxies do not support causal Buy/Sell claims.",
            "Message replay and queue/fill evaluation require an event_l3 export.",
        ],
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "build_report.json", {
        "dataset_id": dataset_id,
        "states": total,
        "split_counts": {name: int((split == value).sum()) for name, value in (("train", 0), ("val", 1), ("test", 2), ("purged", -1))},
        "volume_clip": volume_clip,
        "action_size_scale": action_size_scale,
        "truncated": truncated,
    })
    (output_dir / "manifest.yaml").write_text(
        f"dataset_id: {dataset_id}\nfiles:\n"
        "  states: states.npy\n"
        "  states_raw: states_raw.npy\n"
        "  states_merged: states_merged.npy\n"
        "  actions: actions.npy\n"
        "  timestamps: timestamps.npy\n"
        "  day_id: day_id.npy\n"
        "  segment_id: segment_id.npy\n"
        "  source_row_idx: source_row_idx.npy\n"
        "  split: split.npy\n"
        "  normalization: normalization.json\n"
        "  metadata: metadata.json\n"
    )
    (output_dir / "DATASET_CARD.md").write_text(
        f"# {dataset_id}\n\n"
        f"Public NASDAQ ITCH RTH `{mode}` snapshot panel for `{symbol}`.\n\n"
        "The source dates are public non-consecutive sample days. The snapshot panel "
        "supports structural generation and consumer-acceptance evaluation.\n"
    )
    return {"dataset_id": dataset_id, "output_dir": str(output_dir), "metadata": metadata}


def _update_manifest(path: Path, datasets: Iterable[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {"datasets": {}}
    if path.exists():
        loaded = yaml.safe_load(path.read_text()) or {}
        if isinstance(loaded, dict):
            payload = loaded
    payload.setdefault("datasets", {})
    for result in datasets:
        metadata = result["metadata"]
        output = Path(result["output_dir"])
        dataset_id = result["dataset_id"]
        payload["datasets"][dataset_id] = {
            "status": metadata["dataset_status"],
            "symbol": metadata["symbol"],
            "source": metadata["source"],
            "sampling": metadata["sampling"],
            "level": 10,
            "state_path": str(output / "states.npy"),
            "raw_state_path": str(output / "states_raw.npy"),
            "merged_state_path": str(output / "states_merged.npy"),
            "action_path": str(output / "actions.npy"),
            "timestamps_path": str(output / "timestamps.npy"),
            "day_id_path": str(output / "day_id.npy"),
            "segment_id_path": str(output / "segment_id.npy"),
            "source_row_idx_path": str(output / "source_row_idx.npy"),
            "split_path": str(output / "split.npy"),
            "normalization_path": str(output / "normalization.json"),
            "metadata_path": str(output / "metadata.json"),
            "has_true_timestamps": True,
            "has_true_day_id": True,
            "has_message_data": False,
            "has_raw_itch_source": True,
            "action_label_status": metadata["action_label_status"],
            "scope_notes": metadata["scope_notes"],
        }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itch-file", action="append", required=True, help="Repeat for each public sample day.")
    parser.add_argument("--symbols", default="AAPL,AMZN,GOOG,INTC")
    parser.add_argument("--sampling", choices=("event", "1s", "both"), default="both")
    parser.add_argument("--dataset-prefix", default="PUBLIC_ITCH_2019_SAMPLES")
    parser.add_argument("--output-root", default="data/processed/snapshot_l2")
    parser.add_argument("--staging-root", default="data/processed/.public_itch_staging")
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--train-dates", default=None)
    parser.add_argument("--val-dates", default=None)
    parser.add_argument("--test-dates", default=None)
    parser.add_argument("--purge-gap", type=int, default=100)
    parser.add_argument("--max-timestamp-gap", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--normalization-sample-rows", type=int, default=250_000)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--ticks-per-price-unit", type=float, default=10.0)
    parser.add_argument("--max-snapshots-per-symbol", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse fully parsed day staging marked by _parse_complete.json.",
    )
    args = parser.parse_args()

    symbols = tuple(dict.fromkeys(item.strip().upper() for item in args.symbols.split(",") if item.strip()))
    modes = ("event", "1s") if args.sampling == "both" else (args.sampling,)
    itch_files = sorted((Path(value) for value in args.itch_file), key=infer_day_id)
    staging_root = Path(args.staging_root) / args.dataset_prefix
    if staging_root.exists() and not args.resume:
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    parse_reports = []
    for path in itch_files:
        if not path.exists():
            raise FileNotFoundError(path)
        day_id = infer_day_id(path)
        day_root = staging_root / str(day_id)
        marker = day_root / "_parse_complete.json"
        identity = _source_identity(path, symbols, modes, args.max_snapshots_per_symbol)
        completed = _load_completed_parse(marker, identity) if args.resume else None
        if completed is not None:
            parse_reports.append(completed)
            print(f"Reusing completed parse for {path.name}: {marker}", flush=True)
            continue

        # A missing or mismatched marker means this day's chunks may be partial.
        # Remove only that day and rebuild it; completed days remain reusable.
        if day_root.exists():
            shutil.rmtree(day_root)
        report = parse_file(
            path,
            symbols,
            modes,
            staging_root,
            args.chunk_size,
            args.max_snapshots_per_symbol,
            args.progress_every,
        )
        parse_reports.append(report)
        marker.parent.mkdir(parents=True, exist_ok=True)
        _write_json(marker, {"identity": identity, "report": report.__dict__})
        print(json.dumps(report.__dict__, indent=2))

    truncated = any(report.truncated for report in parse_reports)
    train_dates = _parse_date_list(args.train_dates)
    val_dates = _parse_date_list(args.val_dates)
    test_dates = _parse_date_list(args.test_dates)
    outputs = []
    for symbol in symbols:
        for mode in modes:
            outputs.append(finalize_dataset(
                staging_root=staging_root,
                output_root=Path(args.output_root),
                dataset_prefix=args.dataset_prefix,
                symbol=symbol,
                mode=mode,
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
                purge_gap=args.purge_gap,
                max_timestamp_gap=args.max_timestamp_gap,
                tick_size=args.tick_size,
                ticks_per_price_unit=args.ticks_per_price_unit,
                normalization_sample_rows=args.normalization_sample_rows,
                truncated=truncated,
            ))
    if args.update_manifest and not truncated:
        _update_manifest(Path(args.manifest), outputs)
    elif args.update_manifest and truncated:
        print("Manifest not updated because the build is truncated smoke data.", file=sys.stderr)
    if not args.keep_staging:
        shutil.rmtree(staging_root)
    print(json.dumps({"outputs": [item["dataset_id"] for item in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

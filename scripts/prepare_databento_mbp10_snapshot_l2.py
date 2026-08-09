#!/usr/bin/env python3
"""Build a GenLOB snapshot_l2 dataset from Databento MBP-10 DBN files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import databento as db
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_public_itch_snapshot_l2 import _fit_normalization


EVENT_CODE = {
    "none": 0.0,
    "add": 0.25,
    "cancel": 0.50,
    "modify": 0.75,
    "trade": 1.0,
    "clear": -1.0,
}


def _side_value(side: Any) -> float:
    text = str(side)
    if text == "B":
        return 1.0
    if text in {"A", "S"}:
        return -1.0
    return 0.0


def _event_code(action: Any) -> float:
    text = str(action)
    if text == "A":
        return EVENT_CODE["add"]
    if text in {"C", "D"}:
        return EVENT_CODE["cancel"]
    if text == "M":
        return EVENT_CODE["modify"]
    if text == "T":
        return EVENT_CODE["trade"]
    if text == "R":
        return EVENT_CODE["clear"]
    return EVENT_CODE["none"]


def _action_row(row: pd.Series) -> list[float]:
    action = str(row.get("action", ""))
    side = _side_value(row.get("side", "N"))
    size = float(row.get("size", 0.0) or 0.0)
    if action == "T":
        # Databento MBP action side is used only as a diagnostic proxy here.
        # Formal aggressor labels require event-level MBO processing.
        return [-side, 0.0, 0.0, size, _event_code(action)]
    if action == "A":
        return [0.0, side, 0.0, size, _event_code(action)]
    if action in {"C", "D"}:
        return [0.0, 0.0, side, size, _event_code(action)]
    if action == "M":
        return [0.0, side, 0.0, size, _event_code(action)]
    return [0.0, 0.0, 0.0, size, _event_code(action)]


def _valid_rows(raw_states: np.ndarray, tolerance: float = 0.0, max_relative_depth: float = 0.5) -> np.ndarray:
    prices = raw_states[:, 0, :]
    volumes = raw_states[:, 1, :]
    ask = prices[:, :10]
    bid = prices[:, 10:]
    mid = (ask[:, 0] + bid[:, 0]) / 2.0
    finite = np.isfinite(raw_states).all(axis=(1, 2))
    positive_prices = (ask > 0).all(axis=1) & (bid > 0).all(axis=1)
    nonnegative_volumes = (volumes >= -tolerance).all(axis=1)
    ask_ok = (ask[:, :-1] <= ask[:, 1:] + tolerance).all(axis=1)
    bid_ok = (bid[:, :-1] >= bid[:, 1:] - tolerance).all(axis=1)
    spread_ok = ask[:, 0] > bid[:, 0] + tolerance
    finite_mid = np.isfinite(mid) & (mid > 0)
    rel_ask = (ask - mid[:, None]) / mid[:, None]
    rel_bid = (mid[:, None] - bid) / mid[:, None]
    depth_ok = (rel_ask <= max_relative_depth).all(axis=1) & (rel_bid <= max_relative_depth).all(axis=1)
    return finite & positive_prices & nonnegative_volumes & ask_ok & bid_ok & spread_ok & finite_mid & depth_ok


def _split_and_purge(n: int, train_ratio: float, val_ratio: float, purge_gap: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n < 3:
        raise ValueError("Need at least three rows to split")
    split = np.zeros(n, dtype=np.int8)
    b1 = int(n * train_ratio)
    b2 = int(n * (train_ratio + val_ratio))
    b1 = min(max(b1, 1), n - 2)
    b2 = min(max(b2, b1 + 1), n - 1)
    split[b1:b2] = 1
    split[b2:] = 2

    keep = np.ones(n, dtype=bool)
    if purge_gap > 0:
        keep[max(0, b1 - purge_gap):min(n, b1 + purge_gap)] = False
        keep[max(0, b2 - purge_gap):min(n, b2 + purge_gap)] = False
    kept_idx = np.where(keep)[0]
    if len(kept_idx) < 2:
        raise ValueError("Purge removed too many rows")
    report = {
        "type": "single_day_contiguous_time_split_with_boundary_purge_gap",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "purge_gap": purge_gap,
        "original_rows": n,
        "kept_rows": int(len(kept_idx)),
        "removed_rows": int(n - len(kept_idx)),
        "boundaries": {"train_val": b1, "val_test": b2},
        "kept_split_counts": {
            "train": int((split[kept_idx] == 0).sum()),
            "val": int((split[kept_idx] == 1).sum()),
            "test": int((split[kept_idx] == 2).sum()),
        },
    }
    return split, kept_idx, report


def _build_actions(
    action_rows: np.ndarray,
    kept_idx: np.ndarray,
    source_rows: np.ndarray,
) -> np.ndarray:
    actions = np.zeros((len(kept_idx) - 1, action_rows.shape[1]), dtype=np.float32)
    kept_source_rows = source_rows[kept_idx]
    contiguous = kept_source_rows[1:] == kept_source_rows[:-1] + 1
    dst_positions = np.where(contiguous)[0]
    if dst_positions.size:
        src_rows = kept_idx[1:][contiguous]
        actions[dst_positions] = action_rows[src_rows]
    return actions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _update_dataset_manifest(manifest_path: Path, dataset_id: str, symbol: str, output_dir: Path) -> None:
    manifest = {"datasets": {}}
    if manifest_path.exists():
        with open(manifest_path) as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            manifest = loaded
    manifest.setdefault("datasets", {})
    manifest["datasets"][dataset_id] = {
        "status": "prepared_snapshot_l2_from_databento_mbp10",
        "symbol": symbol,
        "source": "databento_xnas_itch_mbp10",
        "level": 10,
        "state_path": str(output_dir / "states.npy"),
        "merged_state_path": str(output_dir / "states_merged.npy"),
        "raw_state_path": str(output_dir / "states_raw.npy"),
        "action_path": str(output_dir / "actions.npy"),
        "timestamps_path": str(output_dir / "timestamps.npy"),
        "timestamp_ns_path": str(output_dir / "timestamps_ns.npy"),
        "day_id_path": str(output_dir / "day_id.npy"),
        "segment_id_path": str(output_dir / "segment_id.npy"),
        "source_row_idx_path": str(output_dir / "source_row_idx.npy"),
        "split_path": str(output_dir / "split.npy"),
        "normalization_path": str(output_dir / "normalization.json"),
        "metadata_path": str(output_dir / "metadata.json"),
        "dataset_card_path": str(output_dir / "DATASET_CARD.md"),
        "has_message_data": False,
        "has_databento_mbp10_source": True,
        "processed_event_l3_available": False,
        "has_true_timestamps": True,
        "has_true_day_id": True,
        "has_true_day_boundaries": False,
        "action_label_status": "databento_mbp_action_side_diagnostic_not_causal",
        "limitations": [
            "single_day_rth_contiguous_split",
            "not_lobster",
            "not_message_level_action_causality",
            "not_queue_fill_calibration",
        ],
    }
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def _write_manifest_yaml(path: Path, dataset_id: str) -> None:
    path.write_text(
        f"dataset_id: {dataset_id}\n"
        "files:\n"
        "  states: states.npy\n"
        "  states_raw: states_raw.npy\n"
        "  states_merged: states_merged.npy\n"
        "  actions: actions.npy\n"
        "  timestamps: timestamps.npy\n"
        "  timestamps_ns: timestamps_ns.npy\n"
        "  day_id: day_id.npy\n"
        "  segment_id: segment_id.npy\n"
        "  source_row_idx: source_row_idx.npy\n"
        "  split: split.npy\n"
        "  normalization: normalization.json\n"
        "  metadata: metadata.json\n"
        "  dataset_card: DATASET_CARD.md\n"
    )


def _write_dataset_card(path: Path, metadata: dict[str, Any]) -> None:
    text = f"""# {metadata['dataset_id']} Dataset Card

Status: `{metadata['dataset_status']}`.

Symbol: `{metadata['symbol']}`.

Source: Databento `{metadata['databento_dataset']}` `{metadata['schema']}` DBN file:

```text
{metadata['dbn_file']}
```

Rows: `{metadata['state_shape'][0]}`.

This dataset uses the GenLOB `snapshot_l2` schema:

- `states.npy`: `[T, 2, 20]`, per-frame midpoint-centered, tick-scaled
  compatibility snapshots used for schema inspection.
- `states_raw.npy` and `states_merged.npy`: raw dollar prices and share volumes.
- `actions.npy`: `[T-1, 5]`, diagnostic MBP action labels aligned to snapshot transitions.
- `timestamps.npy`: Databento `ts_event` timestamps in Unix seconds for loader compatibility.
- `timestamps_ns.npy`: exact Databento `ts_event` timestamps as `int64` Unix nanoseconds.
- `day_id.npy`: true exchange-local trading date derived from `ts_event`.
- `segment_id.npy`: continuity segments that prevent windows crossing dropped rows,
  split purge gaps, or large timestamp gaps.
- `source_row_idx.npy`: original DBN dataframe row for every retained snapshot.
- `split.npy`: contiguous train/val/test split with a boundary purge gap.
- `normalization.json`: train-only normalization parameters.

Training and evaluation use `states_raw.npy` to normalize each complete
history/future window around the last history midpoint. One model price unit is
ten exchange ticks, so the future midpoint path is preserved.

The action labels are derived from Databento MBP action/side fields for
diagnostics only. They are not true causal Buy/Sell or queue/fill labels.
"""
    path.write_text(text)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dbn_file = Path(args.dbn_file)
    if not dbn_file.exists():
        raise FileNotFoundError(dbn_file)

    store = db.DBNStore.from_file(dbn_file)
    df = store.to_df()
    if len(df) < 2:
        raise ValueError("Databento file has too few rows")
    original_source_rows = len(df)

    source_row_all = np.arange(len(df), dtype=np.int64)
    event_time_all = pd.to_datetime(df["ts_event"], utc=True)
    if args.regular_hours_only:
        local_time = event_time_all.dt.tz_convert(args.exchange_timezone)
        minutes = local_time.dt.hour * 60 + local_time.dt.minute
        rth_mask = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
        df = df.loc[rth_mask].copy()
        source_row_all = source_row_all[np.asarray(rth_mask)]
        event_time_all = event_time_all.loc[rth_mask]
    if len(df) < 2:
        raise ValueError("Databento file has too few rows after session filtering")

    ask_px_cols = [f"ask_px_{i:02d}" for i in range(10)]
    bid_px_cols = [f"bid_px_{i:02d}" for i in range(10)]
    ask_sz_cols = [f"ask_sz_{i:02d}" for i in range(10)]
    bid_sz_cols = [f"bid_sz_{i:02d}" for i in range(10)]
    required = ask_px_cols + bid_px_cols + ask_sz_cols + bid_sz_cols + ["ts_event", "action", "side", "size"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing Databento MBP-10 columns: {missing}")

    prices = np.concatenate(
        [
            df[ask_px_cols].to_numpy(dtype=np.float64),
            df[bid_px_cols].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    volumes = np.concatenate(
        [
            df[ask_sz_cols].to_numpy(dtype=np.float64),
            df[bid_sz_cols].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    raw_all = np.stack([prices, volumes], axis=1)
    action_all = np.asarray([_action_row(row) for _, row in df.iterrows()], dtype=np.float32)
    timestamps_ns_all = event_time_all.astype("int64").to_numpy(dtype=np.int64)
    timestamps_all = timestamps_ns_all.astype(np.float64) / 1e9

    valid = _valid_rows(raw_all, max_relative_depth=args.max_relative_depth)
    if args.drop_invalid_rows:
        valid_idx = np.where(valid)[0]
    else:
        invalid_count = int((~valid).sum())
        if invalid_count:
            raise ValueError(f"{invalid_count} invalid/incomplete MBP rows found; pass --drop-invalid-rows")
        valid_idx = np.arange(len(raw_all))

    raw_valid = raw_all[valid_idx]
    action_valid = action_all[valid_idx]
    timestamps_valid = timestamps_all[valid_idx]
    timestamps_ns_valid = timestamps_ns_all[valid_idx]
    source_row_valid = source_row_all[valid_idx]
    split_valid, kept_rel_idx, split_report = _split_and_purge(
        len(raw_valid),
        args.train_ratio,
        args.val_ratio,
        args.purge_gap,
    )
    raw_states = raw_valid[kept_rel_idx].astype(np.float64, copy=False)
    timestamps = timestamps_valid[kept_rel_idx]
    timestamps_ns = timestamps_ns_valid[kept_rel_idx]
    source_row_idx = source_row_valid[kept_rel_idx]
    split = split_valid[kept_rel_idx].astype(np.int8, copy=False)
    local_dates = pd.to_datetime(timestamps_ns, unit="ns", utc=True).tz_convert(args.exchange_timezone)
    day_id = np.asarray(local_dates.strftime("%Y%m%d"), dtype=np.int32)
    segment_break = np.ones(len(raw_states), dtype=bool)
    if len(raw_states) > 1:
        segment_break[1:] = (
            (np.diff(source_row_idx) != 1)
            | (np.diff(split) != 0)
            | (np.diff(timestamps_ns) < 0)
            | (np.diff(timestamps_ns) > int(args.max_timestamp_gap_seconds * 1e9))
            | (np.diff(day_id) != 0)
        )
    segment_id = np.cumsum(segment_break).astype(np.int32) - 1
    actions = _build_actions(action_valid, kept_rel_idx, source_row_valid)

    size_scale = float(max(np.quantile(action_valid[:, 3], 0.99), 1.0))
    actions[:, 3] = np.clip(actions[:, 3] / size_scale, 0.0, 1.0)

    states, normalization = _fit_normalization(raw_states, split)
    model_price_unit_raw = float(args.tick_size * args.ticks_per_price_unit)
    raw_prices = raw_states[:, 0]
    frame_mid = ((raw_prices[:, 0] + raw_prices[:, 10]) / 2.0)[:, None]
    states[:, 0] = ((raw_prices - frame_mid) / model_price_unit_raw).astype(np.float32)
    tick_normalized = float(1.0 / args.ticks_per_price_unit)
    normalization.update({
        "price_transform": "history_last_mid_tick_scaled_dynamic",
        "stored_state_price_transform": "per_frame_mid_tick_scaled_compatibility",
        "price_reference_protocol": "history_last_mid",
        "tick_size_raw": float(args.tick_size),
        "ticks_per_price_unit": float(args.ticks_per_price_unit),
        "model_price_unit_raw": model_price_unit_raw,
        "min_spread_raw": float(args.tick_size),
        "min_spread_normalized": tick_normalized,
        "min_level_gap_raw": float(args.tick_size),
        "min_level_gap_normalized": tick_normalized,
    })
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/processed/snapshot_l2") / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "states_raw.npy", raw_states)
    np.save(output_dir / "states_merged.npy", raw_states)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "timestamps.npy", timestamps)
    np.save(output_dir / "timestamps_ns.npy", timestamps_ns)
    np.save(output_dir / "day_id.npy", day_id)
    np.save(output_dir / "segment_id.npy", segment_id)
    np.save(output_dir / "source_row_idx.npy", source_row_idx)
    np.save(output_dir / "split.npy", split)
    _write_json(output_dir / "normalization.json", normalization)

    metadata = {
        "dataset_id": args.dataset_id,
        "dataset_status": "prepared_snapshot_l2_from_databento_mbp10",
        "symbol": args.symbol.upper(),
        "source": "databento",
        "databento_dataset": str(store.dataset),
        "schema": str(store.schema),
        "dbn_file": str(dbn_file),
        "source_start": str(store.start),
        "source_end": str(store.end),
        "source_symbols": list(store.symbols or []),
        "source_rows": int(original_source_rows),
        "session_rows": int(len(source_row_all)),
        "session_filter": "regular_hours_0930_1600" if args.regular_hours_only else "all_available_hours",
        "exchange_timezone": args.exchange_timezone,
        "valid_rows_before_purge": int(len(raw_valid)),
        "invalid_rows_dropped": int((~valid).sum()),
        "max_relative_depth": args.max_relative_depth,
        "state_shape": list(states.shape),
        "raw_state_shape": list(raw_states.shape),
        "action_shape": list(actions.shape),
        "level": 10,
        "has_message_data": False,
        "has_databento_mbp10_source": True,
        "processed_event_l3_available": False,
        "has_true_timestamps": True,
        "timestamp_status": "exact_databento_ts_event_int64_ns_with_float_seconds_compatibility",
        "timestamp_precision": "nanoseconds_int64",
        "has_true_day_id": True,
        "has_true_day_boundaries": False,
        "day_id_status": "exchange_local_date_from_databento_ts_event",
        "has_continuity_segments": True,
        "segment_count": int(segment_id.max() + 1) if len(segment_id) else 0,
        "source_row_index_preserved": True,
        "action_label_status": "databento_mbp_action_side_diagnostic_not_causal",
        "action_size_scale": size_scale,
        "action_schema": {
            "shape": "[T-1, 5]",
            "columns": ["aggressor_side_proxy", "passive_side_proxy", "cancel_side_proxy", "normalized_size", "event_code"],
            "status": "diagnostic only; formal action causality requires MBO event_l3",
        },
        "split_policy": split_report,
        "normalization": normalization,
        "limitations": [
            "Single-day Databento sample; not a multi-day benchmark.",
            "Single-day chronological split; not multi-day cross-validation.",
            "MBP action labels are diagnostic proxies, not true counterfactual actions.",
            "Queue/fill calibration requires MBO event_l3 and a queue simulator.",
        ],
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_dataset_card(output_dir / "DATASET_CARD.md", metadata)
    _write_manifest_yaml(output_dir / "manifest.yaml", args.dataset_id)
    _update_dataset_manifest(Path(args.manifest), args.dataset_id, args.symbol.upper(), output_dir)

    report = {
        "dataset_id": args.dataset_id,
        "symbol": args.symbol.upper(),
        "output_dir": str(output_dir),
        "source_rows": int(original_source_rows),
        "session_rows": int(len(source_row_all)),
        "invalid_rows_dropped": int((~valid).sum()),
        "valid_rows_before_purge": int(len(raw_valid)),
        "max_relative_depth": args.max_relative_depth,
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "split_counts": {
            "train": int((split == 0).sum()),
            "val": int((split == 1).sum()),
            "test": int((split == 2).sum()),
        },
        "normalization": normalization,
        "action_size_scale": size_scale,
        "split_policy": split_report,
        "session_filter": "regular_hours_0930_1600" if args.regular_hours_only else "all_available_hours",
        "segment_count": int(segment_id.max() + 1) if len(segment_id) else 0,
    }
    _write_json(output_dir / "build_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbn-file", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--purge-gap", type=int, default=100)
    parser.add_argument("--max-relative-depth", type=float, default=0.5,
                        help="Drop rows where any top-10 level is farther than this fraction from mid.")
    parser.add_argument("--drop-invalid-rows", action="store_true")
    parser.add_argument("--regular-hours-only", action="store_true",
                        help="Keep 09:30 <= exchange-local time < 16:00 only.")
    parser.add_argument("--exchange-timezone", default="America/New_York")
    parser.add_argument("--max-timestamp-gap-seconds", type=float, default=60.0,
                        help="Start a new continuity segment after a larger event gap.")
    parser.add_argument("--tick-size", type=float, default=0.01,
                        help="Exchange tick size in raw dollar price units.")
    parser.add_argument("--ticks-per-price-unit", type=float, default=10.0,
                        help="Price normalization scale; 10 means one model unit equals ten ticks.")
    args = parser.parse_args()

    report = build_dataset(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit local datasets before model training/evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.constraints import (
    check_lobster_orderbook_invariants,
    check_snapshot_invariants,
    validate_lobster_messages,
)
from genlob.data.action_templates import summarize_action_templates
from genlob.data.lobster import infer_orderbook_depth, load_lobster_pair, orderbook_to_numpy
from genlob.data.nasdaq import (
    load_nasdaq_multiday,
    load_nasdaq_stock,
    parse_snapshot_dataset_id,
)
from genlob.matcher import compare_replay_to_orderbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GenLOB datasets")
    parser.add_argument("--dataset", required=True,
                        help="INTC_9day, nasdaq_INTC_9day, nasdaq_STOCK, or lobster")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--json-out", default=None)

    # LOBSTER-only args
    parser.add_argument("--message", default=None)
    parser.add_argument("--orderbook", default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--bootstrap-first-row", action="store_true")
    parser.add_argument("--aggregate-fallback", action="store_true")
    parser.add_argument("--max-mismatches", type=int, default=10)
    return parser.parse_args()


def _sequence_count(n_rows: int, hist_len: int, pred_len: int, stride: int) -> int:
    total = hist_len + pred_len
    return max(0, (n_rows - total) // stride + 1)


def _action_summary(actions: np.ndarray) -> dict[str, Any]:
    if actions.size == 0:
        return {"n": 0}
    direction = actions[:, 0]
    buy = int((direction > 0.1).sum())
    sell = int((direction < -0.1).sum())
    hold = int(len(direction) - buy - sell)
    return {
        "n": int(len(actions)),
        "shape": list(actions.shape),
        "buy": buy,
        "sell": sell,
        "hold": hold,
        "buy_pct": buy / max(len(actions), 1),
        "sell_pct": sell / max(len(actions), 1),
        "hold_pct": hold / max(len(actions), 1),
        "mean": actions.mean(axis=0).astype(float).tolist(),
        "std": actions.std(axis=0).astype(float).tolist(),
        "empirical_templates": summarize_action_templates(actions),
        "nan_rate": float(np.isnan(actions).mean()),
        "inf_rate": float(np.isinf(actions).mean()),
    }


def _action_response_summary(
    data: np.ndarray,
    actions: np.ndarray,
    horizons: list[int],
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Diagnostic association between proxy action channel and future mid move.

    This is not a causal market-impact estimate unless the action channel comes
    from validated message-level aggressor-side labels and the evaluation
    protocol explicitly controls histories/seeds.
    """
    if data.size == 0 or actions.size == 0:
        return {}

    n = min(len(data), len(actions))
    mid = (data[:n, 0, 0] + data[:n, 0, 10]) / 2
    direction = actions[:n, 0]
    out: dict[str, Any] = {}
    for h in horizons:
        if h <= 0 or h >= n:
            continue
        dmid = mid[h:] - mid[:-h]
        action_dir = direction[:-h]
        buy_mask = action_dir > threshold
        sell_mask = action_dir < -threshold
        row: dict[str, Any] = {
            "horizon": int(h),
            "buy_count": int(buy_mask.sum()),
            "sell_count": int(sell_mask.sum()),
        }
        if buy_mask.any():
            row["mean_dmid_buy"] = float(dmid[buy_mask].mean())
            row["p_up_buy"] = float((dmid[buy_mask] > 0).mean())
        if sell_mask.any():
            row["mean_dmid_sell"] = float(dmid[sell_mask].mean())
            row["p_up_sell"] = float((dmid[sell_mask] > 0).mean())
        if buy_mask.any() and sell_mask.any():
            row["buy_minus_sell_dmid"] = float(dmid[buy_mask].mean() - dmid[sell_mask].mean())
        out[f"h{h}"] = row
    return out


def _snapshot_split_report(data: np.ndarray, hist_len: int, pred_len: int, stride: int) -> dict[str, Any]:
    inv = check_snapshot_invariants(data)
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sequence_count": _sequence_count(len(data), hist_len, pred_len, stride),
        "invariants": inv.as_dict(),
    }


def audit_nasdaq(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir or "data")
    if args.dataset.startswith("nasdaq_") and len(args.dataset.split("_")) == 2:
        parts = args.dataset.split("_")
        stock = parts[1].upper()
        nday = None
    else:
        stock, nday = parse_snapshot_dataset_id(args.dataset)

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "hist_len": args.hist_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
    }

    if nday:
        raw = load_nasdaq_multiday(str(data_dir), stock, nday)
        dataset_dir = Path(raw.get("dataset_dir", data_dir))
        merged_path = dataset_dir / "states_merged.npy"
        raw_response: dict[str, np.ndarray] | None = None
        if merged_path.exists():
            merged = np.load(merged_path, mmap_mode="r")
            split_path = dataset_dir / "split.npy"
            if split_path.exists():
                split = np.load(split_path)
                merged_aligned = merged[:-1] if len(merged) == len(split) else merged
                split_aligned = split[:-1] if len(split) == len(merged) else split
                raw_response = {
                    "train": np.asarray(merged_aligned[split_aligned == 0]),
                    "val": np.asarray(merged_aligned[split_aligned == 1]),
                    "test": np.asarray(merged_aligned[split_aligned == 2]),
                }
            else:
                merged_aligned = merged[:-1]
                n_total = len(merged_aligned)
                n_train = int(n_total * 0.7)
                n_val = int(n_total * 0.15)
                raw_response = {
                    "train": np.asarray(merged_aligned[:n_train]),
                    "val": np.asarray(merged_aligned[n_train:n_train + n_val]),
                    "test": np.asarray(merged_aligned[n_train + n_val:]),
                }
        report["source"] = {
            "dataset_dir": str(dataset_dir),
            "states_path": str(dataset_dir / "states.npy"),
            "actions_path": str(dataset_dir / "actions.npy"),
            "normalization_path": str(dataset_dir / "normalization.json"),
            "merged_path": str(merged_path) if merged_path.exists() else None,
        }
        report["row_alignment"] = {
            "norm_rows_total_after_drop": int(sum(len(raw[split]) for split in ["train", "val", "test"])),
            "actions_rows_total": int(sum(len(raw[f"{split}_actions"]) for split in ["train", "val", "test"])),
            "ok": int(sum(len(raw[split]) for split in ["train", "val", "test"]))
            == int(sum(len(raw[f"{split}_actions"]) for split in ["train", "val", "test"])),
        }
        report["action_response_scope"] = (
            "diagnostic association only; not causal Buy/Sell market impact "
            "without validated message-level aggressor-side labels"
        )
        report["splits"] = {}
        for split in ["train", "val", "test"]:
            report["splits"][split] = _snapshot_split_report(
                raw[split],
                args.hist_len,
                args.pred_len,
                args.stride,
            )
            report["splits"][split]["actions"] = _action_summary(raw[f"{split}_actions"])
            report["splits"][split]["action_response"] = _action_response_summary(
                raw_response[split] if raw_response is not None else raw[split],
                raw[f"{split}_actions"],
                horizons=[1, args.pred_len],
            )

        if merged_path.exists():
            merged = np.load(merged_path, mmap_mode="r")
            report["merged_invariants"] = _snapshot_split_report(
                np.asarray(merged),
                args.hist_len,
                args.pred_len,
                args.stride,
            )
    else:
        raw = load_nasdaq_stock(str(data_dir), stock)
        report["source"] = {"lob_path": str(data_dir / f"{stock}_lob.npy")}
        report["splits"] = {
            split: _snapshot_split_report(raw[split], args.hist_len, args.pred_len, args.stride)
            for split in ["train", "val", "test"]
        }

    return report


def audit_lobster(args: argparse.Namespace) -> dict[str, Any]:
    if not args.message or not args.orderbook:
        raise ValueError("LOBSTER audit requires --message and --orderbook")
    messages, orderbook = load_lobster_pair(
        args.message,
        args.orderbook,
        depth=args.depth,
        has_header=args.has_header,
    )
    orderbook_values = orderbook_to_numpy(orderbook)
    depth = args.depth or infer_orderbook_depth(orderbook_values.shape[1])
    grammar = validate_lobster_messages(messages)
    official_inv = check_lobster_orderbook_invariants(orderbook_values, depth=depth)
    replay = compare_replay_to_orderbook(
        messages,
        orderbook_values,
        depth=depth,
        bootstrap_first_row=args.bootstrap_first_row,
        aggregate_fallback=args.aggregate_fallback,
        max_mismatches=args.max_mismatches,
    )
    replay_inv = check_lobster_orderbook_invariants(replay.replay.states, depth=depth)
    return {
        "dataset": "lobster",
        "message": args.message,
        "orderbook": args.orderbook,
        "rows": int(len(messages)),
        "depth": int(depth),
        "grammar": grammar.__dict__ | {"ok": grammar.ok, "total_violations": grammar.total_violations},
        "official_invariants": official_inv.as_dict() | {"ok": official_inv.ok},
        "replay_invariants": replay_inv.as_dict() | {"ok": replay_inv.ok},
        "replay": {
            "exact": replay.exact,
            "match_rate": replay.match_rate,
            "matched_rows": replay.matched_rows,
            "total_rows": replay.total_rows,
            "issues": [issue.__dict__ for issue in replay.replay.issues[: args.max_mismatches]],
            "first_mismatches": replay.first_mismatches,
        },
    }


def main() -> int:
    args = parse_args()
    if args.dataset == "lobster":
        report = audit_lobster(args)
    else:
        report = audit_nasdaq(args)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

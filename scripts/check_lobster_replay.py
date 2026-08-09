#!/usr/bin/env python3
"""Run LOBSTER grammar, invariant, and exact-replay checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.constraints import check_lobster_orderbook_invariants, validate_lobster_messages
from genlob.data.lobster import infer_orderbook_depth, load_lobster_pair, orderbook_to_numpy
from genlob.matcher import compare_replay_to_orderbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LOBSTER message/orderbook replay consistency")
    parser.add_argument("--message", required=True, help="LOBSTER message CSV path")
    parser.add_argument("--orderbook", required=True, help="LOBSTER orderbook CSV path")
    parser.add_argument("--depth", type=int, default=None, help="LOB depth. Inferred when omitted")
    parser.add_argument("--has-header", action="store_true", help="CSV files include headers")
    parser.add_argument(
        "--bootstrap-first-row",
        action="store_true",
        help="Initialize aggregate book from official row 0 and replay from message row 1",
    )
    parser.add_argument(
        "--aggregate-fallback",
        action="store_true",
        help="When an order id is missing, reduce aggregate level by message side/price/size",
    )
    parser.add_argument("--max-mismatches", type=int, default=20)
    parser.add_argument("--json-out", type=str, default=None, help="Optional path for JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    messages, orderbook = load_lobster_pair(
        args.message,
        args.orderbook,
        depth=args.depth,
        has_header=args.has_header,
    )
    orderbook_values = orderbook_to_numpy(orderbook)
    depth = args.depth or infer_orderbook_depth(orderbook_values.shape[1])

    grammar = validate_lobster_messages(messages)
    expected_invariants = check_lobster_orderbook_invariants(orderbook_values, depth=depth)
    comparison = compare_replay_to_orderbook(
        messages,
        orderbook_values,
        depth=depth,
        bootstrap_first_row=args.bootstrap_first_row,
        aggregate_fallback=args.aggregate_fallback,
        max_mismatches=args.max_mismatches,
    )
    replay_invariants = check_lobster_orderbook_invariants(comparison.replay.states, depth=depth)

    report = {
        "rows": int(len(messages)),
        "depth": int(depth),
        "grammar": grammar.__dict__ | {"ok": grammar.ok, "total_violations": grammar.total_violations},
        "expected_invariants": expected_invariants.as_dict() | {"ok": expected_invariants.ok},
        "replay_invariants": replay_invariants.as_dict() | {"ok": replay_invariants.ok},
        "replay": {
            "exact": comparison.exact,
            "match_rate": comparison.match_rate,
            "matched_rows": comparison.matched_rows,
            "total_rows": comparison.total_rows,
            "bootstrap_first_row": comparison.bootstrap_first_row,
            "counts": comparison.replay.counts,
            "issues": [issue.__dict__ for issue in comparison.replay.issues[: args.max_mismatches]],
            "first_mismatches": comparison.first_mismatches,
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")

    return 0 if grammar.ok and expected_invariants.ok and replay_invariants.ok and comparison.exact else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
LOBSTER message replay engine.

The matcher keeps visible order-id state and aggregate price levels. It is meant
for engineering checks: exact replay, missing-reference diagnostics, and
generated event validation before more specialized engines such as JAX-LOB.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from genlob.data.lobster import (
    DIRECTION_BUY,
    DIRECTION_SELL,
    EVENT_CANCELLATION,
    EVENT_CROSS_TRADE,
    EVENT_DELETION,
    EVENT_EXECUTION_HIDDEN,
    EVENT_EXECUTION_VISIBLE,
    EVENT_SUBMISSION,
    EVENT_TRADING_HALT,
    LOBSTER_ASK_DUMMY_PRICE,
    LOBSTER_BID_DUMMY_PRICE,
    infer_orderbook_depth,
    iter_orderbook_levels,
    orderbook_to_numpy,
)


@dataclass(frozen=True)
class ReplayIssue:
    """A non-fatal replay issue with row context."""

    row: int
    code: str
    message: str
    order_id: int | None = None


@dataclass
class ReplayResult:
    """Replay output and diagnostics."""

    states: np.ndarray
    issues: list[ReplayIssue] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass
class ReplayComparison:
    """Replay-vs-official-orderbook comparison result."""

    replay: ReplayResult
    expected: np.ndarray
    row_matches: np.ndarray
    first_mismatches: list[dict[str, Any]]
    bootstrap_first_row: bool = False

    @property
    def matched_rows(self) -> int:
        return int(self.row_matches.sum())

    @property
    def total_rows(self) -> int:
        return int(len(self.row_matches))

    @property
    def match_rate(self) -> float:
        return self.matched_rows / max(self.total_rows, 1)

    @property
    def exact(self) -> bool:
        return self.total_rows > 0 and self.matched_rows == self.total_rows


class LobsterOrderBook:
    """Visible order-id book plus aggregate price levels."""

    def __init__(self) -> None:
        self.orders: dict[int, tuple[int, int, int]] = {}
        self.bids: defaultdict[int, int] = defaultdict(int)
        self.asks: defaultdict[int, int] = defaultdict(int)

    @classmethod
    def from_orderbook_row(cls, row: Iterable[int], depth: int) -> "LobsterOrderBook":
        """
        Build an aggregate book from one LOBSTER snapshot row.

        Order IDs cannot be recovered from a snapshot. This mode is useful for
        approximate bootstrapping only; later cancel/delete by real IDs will be
        reported as missing references unless aggregate fallback is enabled.
        """
        book = cls()
        next_id = -1
        for ask_price, ask_size, bid_price, bid_size in iter_orderbook_levels(row, depth):
            if ask_price < LOBSTER_ASK_DUMMY_PRICE and ask_size > 0:
                book.add_order(next_id, DIRECTION_SELL, int(ask_size), int(ask_price))
                next_id -= 1
            if bid_price > LOBSTER_BID_DUMMY_PRICE and bid_size > 0:
                book.add_order(next_id, DIRECTION_BUY, int(bid_size), int(bid_price))
                next_id -= 1
        return book

    def clone(self) -> "LobsterOrderBook":
        copied = LobsterOrderBook()
        copied.orders = dict(self.orders)
        copied.bids = defaultdict(int, self.bids)
        copied.asks = defaultdict(int, self.asks)
        return copied

    def _book_for_side(self, side: int) -> defaultdict[int, int]:
        if side == DIRECTION_BUY:
            return self.bids
        if side == DIRECTION_SELL:
            return self.asks
        raise ValueError(f"Invalid LOBSTER direction: {side}")

    def _add_level_size(self, side: int, price: int, size: int) -> None:
        if size <= 0:
            return
        book = self._book_for_side(side)
        book[int(price)] += int(size)

    def _reduce_level_size(self, side: int, price: int, size: int) -> None:
        if size <= 0:
            return
        book = self._book_for_side(side)
        price = int(price)
        book[price] -= int(size)
        if book[price] <= 0:
            book.pop(price, None)

    def add_order(self, order_id: int, side: int, size: int, price: int) -> ReplayIssue | None:
        issue = None
        if order_id in self.orders:
            old_side, old_price, old_size = self.orders[order_id]
            self._reduce_level_size(old_side, old_price, old_size)
            issue = ReplayIssue(-1, "duplicate_order_id", "Replacing duplicate order id", order_id)
        self.orders[int(order_id)] = (int(side), int(price), int(size))
        self._add_level_size(side, price, size)
        return issue

    def reduce_order(
        self,
        order_id: int,
        size: int | None,
        row: int,
        code: str,
        fallback_side: int | None = None,
        fallback_price: int | None = None,
        aggregate_fallback: bool = False,
    ) -> ReplayIssue | None:
        """Reduce or delete an order by id, with optional aggregate fallback."""
        order_id = int(order_id)
        if order_id not in self.orders:
            if aggregate_fallback and fallback_side in (DIRECTION_BUY, DIRECTION_SELL) and fallback_price is not None:
                fallback_size = int(size or 0)
                self._reduce_level_size(int(fallback_side), int(fallback_price), fallback_size)
                return ReplayIssue(row, "aggregate_fallback", "Missing order id; reduced aggregate level", order_id)
            return ReplayIssue(row, "missing_order_id", "Order id not found in visible book", order_id)

        side, price, current_size = self.orders[order_id]
        reduce_size = current_size if size is None else int(size)
        if reduce_size >= current_size:
            self._reduce_level_size(side, price, current_size)
            self.orders.pop(order_id, None)
            if reduce_size > current_size:
                return ReplayIssue(row, "oversized_reduction", "Reduction exceeded visible order size", order_id)
            return None

        self.orders[order_id] = (side, price, current_size - reduce_size)
        self._reduce_level_size(side, price, reduce_size)
        return None

    def apply_message(
        self,
        message: pd.Series | dict[str, Any],
        row: int,
        aggregate_fallback: bool = False,
    ) -> list[ReplayIssue]:
        """Apply one LOBSTER message row."""
        event_type = int(message["EventType"])
        order_id = int(message["OrderID"])
        size = int(message["Size"])
        price = int(message["Price"])
        direction = int(message["Direction"])
        issues: list[ReplayIssue] = []

        if event_type == EVENT_SUBMISSION:
            if direction not in (DIRECTION_BUY, DIRECTION_SELL):
                issues.append(ReplayIssue(row, "invalid_direction", "Submission has invalid direction", order_id))
            else:
                issue = self.add_order(order_id, direction, size, price)
                if issue is not None:
                    issues.append(ReplayIssue(row, issue.code, issue.message, order_id))

        elif event_type == EVENT_CANCELLATION:
            issue = self.reduce_order(order_id, size, row, "cancel", direction, price, aggregate_fallback)
            if issue is not None:
                issues.append(issue)

        elif event_type == EVENT_DELETION:
            issue = self.reduce_order(order_id, None, row, "delete", direction, price, aggregate_fallback)
            if issue is not None:
                issues.append(issue)

        elif event_type == EVENT_EXECUTION_VISIBLE:
            issue = self.reduce_order(order_id, size, row, "execute_visible", direction, price, aggregate_fallback)
            if issue is not None:
                issues.append(issue)

        elif event_type in (EVENT_EXECUTION_HIDDEN, EVENT_CROSS_TRADE, EVENT_TRADING_HALT):
            # Hidden executions, cross trades, and halts do not identify a
            # displayed resting order to update in the visible book.
            pass

        else:
            issues.append(ReplayIssue(row, "unknown_event_type", f"Unsupported event type {event_type}", order_id))

        return issues

    def to_orderbook_row(self, depth: int) -> np.ndarray:
        """Return one LOBSTER-format orderbook row."""
        asks = sorted((p, s) for p, s in self.asks.items() if s > 0)[:depth]
        bids = sorted(((p, s) for p, s in self.bids.items() if s > 0), key=lambda x: -x[0])[:depth]

        row: list[int] = []
        for level in range(depth):
            if level < len(asks):
                ask_price, ask_size = asks[level]
            else:
                ask_price, ask_size = LOBSTER_ASK_DUMMY_PRICE, 0
            if level < len(bids):
                bid_price, bid_size = bids[level]
            else:
                bid_price, bid_size = LOBSTER_BID_DUMMY_PRICE, 0
            row.extend([int(ask_price), int(ask_size), int(bid_price), int(bid_size)])
        return np.asarray(row, dtype=np.int64)


def _count_event(counts: dict[str, int], event_type: int) -> None:
    counts["events"] = counts.get("events", 0) + 1
    counts[f"event_type_{event_type}"] = counts.get(f"event_type_{event_type}", 0) + 1


def replay_lobster_messages(
    messages: pd.DataFrame,
    depth: int,
    initial_book: LobsterOrderBook | None = None,
    initial_orderbook_row: Iterable[int] | None = None,
    aggregate_fallback: bool = False,
    row_offset: int = 0,
) -> ReplayResult:
    """
    Replay LOBSTER messages and emit LOBSTER-format book rows.

    `initial_orderbook_row` is aggregate-only bootstrapping. It cannot recover
    original order IDs, so exact replay should prefer a true initial book or a
    message stream that starts before all referenced visible orders.
    """
    if initial_book is not None and initial_orderbook_row is not None:
        raise ValueError("Pass either initial_book or initial_orderbook_row, not both")
    if initial_book is not None:
        book = initial_book.clone()
    elif initial_orderbook_row is not None:
        book = LobsterOrderBook.from_orderbook_row(initial_orderbook_row, depth)
    else:
        book = LobsterOrderBook()

    states = np.zeros((len(messages), 4 * depth), dtype=np.int64)
    issues: list[ReplayIssue] = []
    counts: dict[str, int] = {}

    for out_idx, (_, message) in enumerate(messages.iterrows()):
        row = row_offset + out_idx
        event_type = int(message["EventType"])
        _count_event(counts, event_type)
        row_issues = book.apply_message(message, row=row, aggregate_fallback=aggregate_fallback)
        issues.extend(row_issues)
        for issue in row_issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        states[out_idx] = book.to_orderbook_row(depth)

    return ReplayResult(states=states, issues=issues, counts=counts)


def _first_mismatches(
    replay_states: np.ndarray,
    expected: np.ndarray,
    max_mismatches: int,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    bad_rows = np.flatnonzero(np.any(replay_states != expected, axis=1))
    for row in bad_rows[:max_mismatches]:
        diff_cols = np.flatnonzero(replay_states[row] != expected[row])
        mismatches.append(
            {
                "row": int(row),
                "columns": diff_cols[:20].astype(int).tolist(),
                "replay": replay_states[row, diff_cols[:20]].astype(int).tolist(),
                "expected": expected[row, diff_cols[:20]].astype(int).tolist(),
            }
        )
    return mismatches


def compare_replay_to_orderbook(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame | np.ndarray,
    depth: int | None = None,
    bootstrap_first_row: bool = False,
    aggregate_fallback: bool = False,
    max_mismatches: int = 20,
) -> ReplayComparison:
    """
    Replay messages and compare row-by-row to the official LOBSTER orderbook.

    If `bootstrap_first_row=True`, row 0 of the official orderbook initializes
    aggregate levels and replay starts from message row 1. This is approximate
    because LOBSTER snapshots do not carry order IDs.
    """
    expected_all = orderbook_to_numpy(orderbook, depth=depth)
    if depth is None:
        depth = infer_orderbook_depth(expected_all.shape[1])
    if len(messages) != len(expected_all):
        raise ValueError(f"Message/orderbook row mismatch: {len(messages)} vs {len(expected_all)}")

    if bootstrap_first_row:
        replay = replay_lobster_messages(
            messages.iloc[1:].reset_index(drop=True),
            depth=depth,
            initial_orderbook_row=expected_all[0],
            aggregate_fallback=aggregate_fallback,
            row_offset=1,
        )
        expected = expected_all[1:]
    else:
        replay = replay_lobster_messages(
            messages,
            depth=depth,
            aggregate_fallback=aggregate_fallback,
        )
        expected = expected_all

    row_matches = np.all(replay.states == expected, axis=1)
    return ReplayComparison(
        replay=replay,
        expected=expected,
        row_matches=row_matches,
        first_mismatches=_first_mismatches(replay.states, expected, max_mismatches),
        bootstrap_first_row=bootstrap_first_row,
    )

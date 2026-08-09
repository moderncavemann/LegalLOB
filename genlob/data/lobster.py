"""
LOBSTER message/orderbook CSV helpers.

LOBSTER emits one message file and one orderbook file per trading day. The kth
message row causes the transition to the kth orderbook row. Prices are integer
prices scaled by 10000.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MESSAGE_COLUMNS = ["Time", "EventType", "OrderID", "Size", "Price", "Direction"]

EVENT_SUBMISSION = 1
EVENT_CANCELLATION = 2
EVENT_DELETION = 3
EVENT_EXECUTION_VISIBLE = 4
EVENT_EXECUTION_HIDDEN = 5
EVENT_CROSS_TRADE = 6
EVENT_TRADING_HALT = 7
EVENT_TYPES = {
    EVENT_SUBMISSION,
    EVENT_CANCELLATION,
    EVENT_DELETION,
    EVENT_EXECUTION_VISIBLE,
    EVENT_EXECUTION_HIDDEN,
    EVENT_CROSS_TRADE,
    EVENT_TRADING_HALT,
}

DIRECTION_BUY = 1
DIRECTION_SELL = -1
DIRECTIONS = {DIRECTION_BUY, DIRECTION_SELL}

LOBSTER_ASK_DUMMY_PRICE = 9_999_999_999
LOBSTER_BID_DUMMY_PRICE = -9_999_999_999


def orderbook_column_names(depth: int) -> list[str]:
    """Return LOBSTER orderbook column names for a given depth."""
    columns: list[str] = []
    for level in range(1, depth + 1):
        columns.extend(
            [
                f"AskPrice{level}",
                f"AskSize{level}",
                f"BidPrice{level}",
                f"BidSize{level}",
            ]
        )
    return columns


def infer_orderbook_depth(n_columns: int) -> int:
    """Infer LOBSTER depth from an orderbook column count."""
    if n_columns % 4 != 0:
        raise ValueError(f"LOBSTER orderbook column count must be 4 * depth, got {n_columns}")
    return n_columns // 4


def _coerce_message_dtypes(messages: pd.DataFrame) -> pd.DataFrame:
    messages = messages.copy()
    int_cols = ["EventType", "OrderID", "Size", "Price", "Direction"]
    for col in int_cols:
        messages[col] = pd.to_numeric(messages[col], errors="raise").astype(np.int64)
    messages["Time"] = pd.to_numeric(messages["Time"], errors="raise").astype(float)
    return messages


def read_message_csv(path: str | Path, has_header: bool = False) -> pd.DataFrame:
    """
    Read a LOBSTER message CSV.

    Official LOBSTER files usually do not include a header. Set `has_header=True`
    for hand-written fixtures or exported files with column names.
    """
    header = 0 if has_header else None
    messages = pd.read_csv(path, header=header)
    if not has_header:
        if messages.shape[1] != len(MESSAGE_COLUMNS):
            raise ValueError(
                f"Expected {len(MESSAGE_COLUMNS)} LOBSTER message columns, got {messages.shape[1]}"
            )
        messages.columns = MESSAGE_COLUMNS
    else:
        missing = [col for col in MESSAGE_COLUMNS if col not in messages.columns]
        if missing:
            raise ValueError(f"Missing LOBSTER message columns: {missing}")
        messages = messages[MESSAGE_COLUMNS]
    return _coerce_message_dtypes(messages)


def read_orderbook_csv(
    path: str | Path,
    depth: int | None = None,
    has_header: bool = False,
) -> pd.DataFrame:
    """Read a LOBSTER orderbook CSV and assign canonical column names."""
    header = 0 if has_header else None
    orderbook = pd.read_csv(path, header=header)
    if depth is None:
        depth = infer_orderbook_depth(orderbook.shape[1])

    expected = 4 * depth
    if orderbook.shape[1] != expected:
        raise ValueError(f"Expected {expected} orderbook columns for depth={depth}, got {orderbook.shape[1]}")

    if not has_header:
        orderbook.columns = orderbook_column_names(depth)
    else:
        missing = [col for col in orderbook_column_names(depth) if col not in orderbook.columns]
        if missing:
            raise ValueError(f"Missing LOBSTER orderbook columns: {missing}")
        orderbook = orderbook[orderbook_column_names(depth)]

    return orderbook.astype(np.int64)


def orderbook_to_numpy(orderbook: pd.DataFrame | np.ndarray, depth: int | None = None) -> np.ndarray:
    """Convert a LOBSTER orderbook DataFrame/array to an int64 numpy matrix."""
    values = orderbook.to_numpy() if isinstance(orderbook, pd.DataFrame) else np.asarray(orderbook)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D orderbook matrix, got shape {values.shape}")
    if depth is not None and values.shape[1] != 4 * depth:
        raise ValueError(f"Expected {4 * depth} columns for depth={depth}, got {values.shape[1]}")
    infer_orderbook_depth(values.shape[1])
    return values.astype(np.int64, copy=False)


def load_lobster_pair(
    message_path: str | Path,
    orderbook_path: str | Path,
    depth: int | None = None,
    has_header: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a LOBSTER message/orderbook pair and check row alignment."""
    messages = read_message_csv(message_path, has_header=has_header)
    orderbook = read_orderbook_csv(orderbook_path, depth=depth, has_header=has_header)
    if len(messages) != len(orderbook):
        raise ValueError(f"Message/orderbook row mismatch: {len(messages)} vs {len(orderbook)}")
    return messages, orderbook


def is_dummy_ask(price: int | np.ndarray) -> np.ndarray | bool:
    """Return whether ask price(s) are LOBSTER dummy levels."""
    return np.asarray(price) >= LOBSTER_ASK_DUMMY_PRICE


def is_dummy_bid(price: int | np.ndarray) -> np.ndarray | bool:
    """Return whether bid price(s) are LOBSTER dummy levels."""
    return np.asarray(price) <= LOBSTER_BID_DUMMY_PRICE


def iter_orderbook_levels(row: Iterable[int], depth: int) -> Iterable[tuple[int, int, int, int]]:
    """Yield `(ask_price, ask_size, bid_price, bid_size)` tuples from one row."""
    values = list(row)
    if len(values) != 4 * depth:
        raise ValueError(f"Expected {4 * depth} values for depth={depth}, got {len(values)}")
    for level in range(depth):
        offset = 4 * level
        yield values[offset], values[offset + 1], values[offset + 2], values[offset + 3]

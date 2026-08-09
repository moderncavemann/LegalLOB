"""Numpy invariant checks for LOBSTER-format orderbook matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from genlob.data.lobster import (
    LOBSTER_ASK_DUMMY_PRICE,
    LOBSTER_BID_DUMMY_PRICE,
    infer_orderbook_depth,
    orderbook_to_numpy,
)


@dataclass(frozen=True)
class LobInvariantReport:
    """Violation rates for LOBSTER-format orderbook rows."""

    n_rows: int
    depth: int
    crossed_book_rate: float
    ask_ladder_violation_rate: float
    bid_ladder_violation_rate: float
    negative_volume_rate: float

    @property
    def ok(self) -> bool:
        return (
            self.crossed_book_rate == 0.0
            and self.ask_ladder_violation_rate == 0.0
            and self.bid_ladder_violation_rate == 0.0
            and self.negative_volume_rate == 0.0
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_rows": self.n_rows,
            "depth": self.depth,
            "crossed_book_rate": self.crossed_book_rate,
            "ask_ladder_violation_rate": self.ask_ladder_violation_rate,
            "bid_ladder_violation_rate": self.bid_ladder_violation_rate,
            "negative_volume_rate": self.negative_volume_rate,
        }


def check_lobster_orderbook_invariants(
    orderbook: pd.DataFrame | np.ndarray,
    depth: int | None = None,
) -> LobInvariantReport:
    """Check crossed-book, monotonic ladder, and non-negative volume rates."""
    values = orderbook_to_numpy(orderbook, depth=depth)
    if depth is None:
        depth = infer_orderbook_depth(values.shape[1])

    reshaped = values.reshape(values.shape[0], depth, 4)
    ask_prices = reshaped[:, :, 0]
    ask_sizes = reshaped[:, :, 1]
    bid_prices = reshaped[:, :, 2]
    bid_sizes = reshaped[:, :, 3]

    active_ask = ask_prices < LOBSTER_ASK_DUMMY_PRICE
    active_bid = bid_prices > LOBSTER_BID_DUMMY_PRICE

    crossed_mask = active_ask[:, 0] & active_bid[:, 0] & (ask_prices[:, 0] <= bid_prices[:, 0])
    crossed = float(crossed_mask.mean()) if len(values) else 0.0

    ask_pairs_active = active_ask[:, :-1] & active_ask[:, 1:]
    bid_pairs_active = active_bid[:, :-1] & active_bid[:, 1:]
    ask_viol = ask_pairs_active & (ask_prices[:, :-1] > ask_prices[:, 1:])
    bid_viol = bid_pairs_active & (bid_prices[:, :-1] < bid_prices[:, 1:])

    ask_den = max(int(ask_pairs_active.sum()), 1)
    bid_den = max(int(bid_pairs_active.sum()), 1)
    vol_values = np.concatenate([ask_sizes.reshape(-1), bid_sizes.reshape(-1)])

    return LobInvariantReport(
        n_rows=int(values.shape[0]),
        depth=int(depth),
        crossed_book_rate=crossed,
        ask_ladder_violation_rate=float(ask_viol.sum() / ask_den),
        bid_ladder_violation_rate=float(bid_viol.sum() / bid_den),
        negative_volume_rate=float((vol_values < 0).mean()) if vol_values.size else 0.0,
    )

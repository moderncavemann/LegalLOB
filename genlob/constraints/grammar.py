"""Grammar checks and projections for LOBSTER-style event rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from genlob.data.lobster import DIRECTIONS, EVENT_TYPES, MESSAGE_COLUMNS


@dataclass(frozen=True)
class GrammarReport:
    """Summary of LOBSTER message grammar violations."""

    n_rows: int
    invalid_event_type: int
    invalid_direction: int
    invalid_size: int
    invalid_price: int
    invalid_order_id: int

    @property
    def total_violations(self) -> int:
        return (
            self.invalid_event_type
            + self.invalid_direction
            + self.invalid_size
            + self.invalid_price
            + self.invalid_order_id
        )

    @property
    def ok(self) -> bool:
        return self.total_violations == 0


def _require_columns(messages: pd.DataFrame) -> None:
    missing = [col for col in MESSAGE_COLUMNS if col not in messages.columns]
    if missing:
        raise ValueError(f"Missing LOBSTER message columns: {missing}")


def validate_lobster_messages(messages: pd.DataFrame) -> GrammarReport:
    """Validate field-level LOBSTER message grammar."""
    _require_columns(messages)
    event_type = messages["EventType"].astype(np.int64)
    direction = messages["Direction"].astype(np.int64)
    size = messages["Size"].astype(np.int64)
    price = messages["Price"].astype(np.int64)
    order_id = messages["OrderID"].astype(np.int64)

    direction_required = event_type.isin([1, 2, 3, 4])
    price_required = event_type.isin([1, 2, 3, 4])
    order_id_required = event_type.isin([1, 2, 3, 4])
    size_required = event_type.isin([1, 2, 4, 5, 6])

    return GrammarReport(
        n_rows=int(len(messages)),
        invalid_event_type=int((~event_type.isin(EVENT_TYPES)).sum()),
        invalid_direction=int((direction_required & ~direction.isin(DIRECTIONS)).sum()),
        invalid_size=int((size_required & (size <= 0)).sum()),
        invalid_price=int((price_required & (price <= 0)).sum()),
        invalid_order_id=int((order_id_required & (order_id <= 0)).sum()),
    )


def project_lobster_messages(
    messages: pd.DataFrame,
    tick_size: int = 1,
    min_size: int = 1,
) -> pd.DataFrame:
    """
    Project raw/generated fields into a LOBSTER-compatible value domain.

    This only fixes field grammar. Semantic checks such as missing references are
    left to the matcher because they depend on book state.
    """
    _require_columns(messages)
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    if min_size <= 0:
        raise ValueError("min_size must be positive")

    projected = messages.copy()
    projected["EventType"] = (
        pd.to_numeric(projected["EventType"], errors="coerce")
        .fillna(1)
        .round()
        .clip(1, 7)
        .astype(np.int64)
    )

    direction = pd.to_numeric(projected["Direction"], errors="coerce").fillna(1)
    projected["Direction"] = np.where(direction >= 0, 1, -1).astype(np.int64)

    size = pd.to_numeric(projected["Size"], errors="coerce").fillna(min_size)
    projected["Size"] = size.round().clip(lower=min_size).astype(np.int64)

    price = pd.to_numeric(projected["Price"], errors="coerce").fillna(tick_size)
    price = price.round().clip(lower=tick_size).astype(np.int64)
    projected["Price"] = ((price + tick_size // 2) // tick_size * tick_size).astype(np.int64)

    order_id = pd.to_numeric(projected["OrderID"], errors="coerce").fillna(1)
    projected["OrderID"] = order_id.round().clip(lower=1).astype(np.int64)

    projected["Time"] = pd.to_numeric(projected["Time"], errors="coerce").fillna(0.0).astype(float)
    return projected[MESSAGE_COLUMNS]

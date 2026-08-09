"""LOBSTER event-level action labels.

LOBSTER `Direction` is the resting limit-order side. For executions this means
the trade aggressor is the opposite side: a sell limit order executed implies a
buyer-initiated trade, and a buy limit order executed implies a seller-initiated
trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .lobster import (
    EVENT_CANCELLATION,
    EVENT_CROSS_TRADE,
    EVENT_DELETION,
    EVENT_EXECUTION_HIDDEN,
    EVENT_EXECUTION_VISIBLE,
    EVENT_SUBMISSION,
    EVENT_TRADING_HALT,
    MESSAGE_COLUMNS,
)


EVENT_CODE = {
    "none": 0.0,
    "submission": 0.25,
    "cancel_delete": 0.5,
    "execution": 1.0,
    "cross_trade": -0.5,
    "halt": -1.0,
}


def derive_lobster_action_labels(messages: pd.DataFrame) -> pd.DataFrame:
    """Return event-level action labels aligned 1:1 with LOBSTER messages."""
    missing = [col for col in MESSAGE_COLUMNS if col not in messages.columns]
    if missing:
        raise ValueError(f"Missing LOBSTER message columns: {missing}")

    labels = pd.DataFrame(index=messages.index)
    event_type = messages["EventType"].astype(int)
    direction = messages["Direction"].astype(float)

    is_execution = event_type.isin([EVENT_EXECUTION_VISIBLE, EVENT_EXECUTION_HIDDEN])
    is_submission = event_type == EVENT_SUBMISSION
    is_cancel = event_type.isin([EVENT_CANCELLATION, EVENT_DELETION])
    is_cross = event_type == EVENT_CROSS_TRADE
    is_halt = event_type == EVENT_TRADING_HALT

    labels["Time"] = messages["Time"].astype(float)
    labels["EventType"] = event_type
    labels["Size"] = messages["Size"].astype(float)
    labels["Price"] = messages["Price"].astype(float)
    labels["resting_side"] = direction
    labels["aggressor_side"] = np.where(is_execution, -direction, 0.0)
    labels["passive_side"] = np.where(is_submission, direction, 0.0)
    labels["cancel_side"] = np.where(is_cancel, direction, 0.0)
    labels["event_class"] = "none"
    labels.loc[is_submission, "event_class"] = "submission"
    labels.loc[is_cancel, "event_class"] = "cancel_delete"
    labels.loc[is_execution, "event_class"] = "execution"
    labels.loc[is_cross, "event_class"] = "cross_trade"
    labels.loc[is_halt, "event_class"] = "halt"
    labels["event_code"] = labels["event_class"].map(EVENT_CODE).astype(float)
    return labels


def lobster_actions_to_array(labels: pd.DataFrame, size_scale: float | None = None) -> np.ndarray:
    """Convert derived labels to a compact 5D model action array.

    Columns are:
      [aggressor_side, passive_side, cancel_side, normalized_size, event_code]
    """
    if size_scale is None:
        size_scale = float(max(labels["Size"].quantile(0.99), 1.0))
    size = np.clip(labels["Size"].to_numpy(dtype=np.float32) / size_scale, 0.0, 1.0)
    return np.stack(
        [
            labels["aggressor_side"].to_numpy(dtype=np.float32),
            labels["passive_side"].to_numpy(dtype=np.float32),
            labels["cancel_side"].to_numpy(dtype=np.float32),
            size,
            labels["event_code"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )

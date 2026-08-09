"""Invariant checks for GenLOB snapshot tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SnapshotInvariantReport:
    """Violation rates for `[N, 2, 20]` LOB snapshots."""

    n_frames: int
    crossed_book_rate: float
    ask_ladder_violation_rate: float
    bid_ladder_violation_rate: float
    negative_volume_rate: float
    nan_rate: float
    inf_rate: float
    avg_spread: float

    @property
    def ok(self) -> bool:
        return (
            self.crossed_book_rate == 0.0
            and self.ask_ladder_violation_rate == 0.0
            and self.bid_ladder_violation_rate == 0.0
            and self.negative_volume_rate == 0.0
            and self.nan_rate == 0.0
            and self.inf_rate == 0.0
        )

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "n_frames": self.n_frames,
            "crossed_book_rate": self.crossed_book_rate,
            "ask_ladder_violation_rate": self.ask_ladder_violation_rate,
            "bid_ladder_violation_rate": self.bid_ladder_violation_rate,
            "negative_volume_rate": self.negative_volume_rate,
            "nan_rate": self.nan_rate,
            "inf_rate": self.inf_rate,
            "avg_spread": self.avg_spread,
            "ok": self.ok,
        }


def flatten_snapshot_tensor(data: np.ndarray) -> np.ndarray:
    """
    Convert common GenLOB snapshot layouts to `[N_frames, 2, 20]`.

    Supported layouts:
    - `[N, 2, 20]`
    - `[N, T, 2, 20]`
    - `[B, 2, 20, T]`
    """
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[1:] == (2, 20):
        return arr
    if arr.ndim == 4 and arr.shape[2:] == (2, 20):
        return arr.reshape(-1, 2, 20)
    if arr.ndim == 4 and arr.shape[1:3] == (2, 20):
        return arr.transpose(0, 3, 1, 2).reshape(-1, 2, 20)
    raise ValueError(f"Unsupported snapshot layout: {arr.shape}")


def check_snapshot_invariants(data: np.ndarray) -> SnapshotInvariantReport:
    """Check snapshot-level LOB constraints for GenLOB `[N, 2, 20]` data."""
    frames = flatten_snapshot_tensor(data).astype(np.float64, copy=False)
    if frames.size == 0:
        return SnapshotInvariantReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    nan_rate = float(np.isnan(frames).mean())
    inf_rate = float(np.isinf(frames).mean())
    finite = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    prices = finite[:, 0, :]
    volumes = finite[:, 1, :]
    ask = prices[:, :10]
    bid = prices[:, 10:]

    crossed = float((ask[:, 0] <= bid[:, 0]).mean())
    ask_ladder = float((ask[:, :-1] > ask[:, 1:]).mean())
    bid_ladder = float((bid[:, :-1] < bid[:, 1:]).mean())
    neg_vol = float((volumes < 0).mean())
    avg_spread = float((ask[:, 0] - bid[:, 0]).mean())

    return SnapshotInvariantReport(
        n_frames=int(frames.shape[0]),
        crossed_book_rate=crossed,
        ask_ladder_violation_rate=ask_ladder,
        bid_ladder_violation_rate=bid_ladder,
        negative_volume_rate=neg_vol,
        nan_rate=nan_rate,
        inf_rate=inf_rate,
        avg_spread=avg_spread,
    )

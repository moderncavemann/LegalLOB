#!/usr/bin/env python3
"""Run the LegalLOB intermediate-state snapshot-risk consumer experiment.

The experiment has two consumer points:

1. every DDIM predicted-clean estimate ("anytime"), and
2. every delivered chunk before it is appended to recursive history
   ("endpoint").

It compares LegalLOB, Matched with endpoint-only PAVA, and Matched with exact
PAVA at every predicted-clean DDIM state.  The consumer is fail-closed: it
computes snapshot risk proxies only after the paper's continuous feasibility
contract passes.  This is not a queue-aware execution or trading experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.data.nasdaq import (
    load_nasdaq_multiday,
    normalized_price_unit,
    parse_snapshot_dataset_id,
    resolve_lob_constraints,
    resolve_price_reference,
)
from genlob.eval.metrics import wasserstein_1d
from genlob.utils.device import select_device
from genlob.utils.legal_lob import make_lob_transform
from genlob.utils.physics import project_to_valid_lob
from scripts.snowball_test import (
    decode_from_model,
    encode_for_model,
    load_genlob,
    make_seed_batches_nasdaq,
    representation_stats_to_device,
    resolve_representation,
    summarize_rollout,
)
from scripts.summarize_financial_sensitivity import (
    _imbalance,
    _liquidity_within_ticks,
    _market_order_slippage,
    _to_economic_units,
)


VARIANTS = (
    "legal_lob",
    "matched_posthoc_pava",
    "matched_repeated_pava",
)


def checkpoint_filename(value: str) -> str:
    """Accept one checkpoint filename, never a path traversal."""
    path = Path(value)
    if not value or value in {".", ".."} or path.name != value:
        raise argparse.ArgumentTypeError(
            "--checkpoint-name must be a single filename (for example best.pt)"
        )
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_rng_sha256() -> str:
    return sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes())


def load_variant_for_legacy_stream(
    checkpoint_path: Path,
    device: torch.device,
    eval_seed: int,
) -> tuple[torch.nn.Module, Any, Any, dict[str, Any], str]:
    """Match snowball_test's seed -> model construction -> sampling order."""
    set_seed(eval_seed)
    model, sde, config, checkpoint = load_genlob(
        checkpoint_path, device, "ema"
    )
    return model, sde, config, checkpoint, torch_rng_sha256()


def _config_signature(config: Any) -> str:
    payload = vars(config) if hasattr(config, "__dict__") else config
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, default=str).encode()
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _flatten_lob(lob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(lob)
    if array.ndim != 4 or array.shape[1:3] != (2, 20):
        raise ValueError(f"Expected [N, 2, 20, T], got {array.shape}")
    prices = np.transpose(array[:, 0], (0, 2, 1)).reshape(-1, 20)
    volumes = np.transpose(array[:, 1], (0, 2, 1)).reshape(-1, 20)
    return prices.astype(np.float64, copy=False), volumes.astype(np.float64, copy=False)


def _select_frames(lob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select flattened [batch, time] snapshots and restore a singleton time axis."""
    array = np.asarray(lob)
    snapshots = np.transpose(array, (0, 3, 1, 2)).reshape(-1, 2, 20)
    selected = snapshots[np.asarray(mask, dtype=bool).reshape(-1)]
    return selected[..., None]


def strict_admissibility(
    lob: np.ndarray,
    normalization: dict[str, Any],
    zero_tolerance: bool = False,
) -> dict[str, np.ndarray]:
    """Return fail-closed per-snapshot masks for the paper's continuous contract."""
    prices, volumes = _flatten_lob(lob)
    ask = prices[:, :10]
    bid = prices[:, 10:]
    min_spread = float(normalization["min_spread_normalized"])
    min_gap = float(normalization["min_level_gap_normalized"])
    tolerance = (
        0.0
        if zero_tolerance
        else max(1e-8, min(min_spread, max(min_gap, 1e-12)) * 2e-5)
    )
    finite = np.isfinite(prices).all(axis=1) & np.isfinite(volumes).all(axis=1)
    spread_ok = ask[:, 0] - bid[:, 0] >= min_spread - tolerance
    ask_ok = np.all(np.diff(ask, axis=1) >= min_gap - tolerance, axis=1)
    bid_ok = np.all(-np.diff(bid, axis=1) >= min_gap - tolerance, axis=1)
    depth_ok = np.all(volumes >= -tolerance, axis=1)
    accepted = finite & spread_ok & ask_ok & bid_ok & depth_ok
    return {
        "accepted": accepted,
        "nonfinite": ~finite,
        "spread": finite & ~spread_ok,
        "ask_chain": finite & ~ask_ok,
        "bid_chain": finite & ~bid_ok,
        "negative_depth": finite & ~depth_ok,
    }


@dataclass(frozen=True)
class RiskContract:
    target_size_shares: float
    slippage_alarm_ticks: float
    liquidity_alarm_target_units: float
    threshold_train_snapshots: int
    threshold_index_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_size_shares": self.target_size_shares,
            "slippage_alarm_ticks": self.slippage_alarm_ticks,
            "liquidity_alarm_target_units": self.liquidity_alarm_target_units,
            "threshold_train_snapshots": self.threshold_train_snapshots,
            "threshold_index_sha256": self.threshold_index_sha256,
            "threshold_source": "training_split_only",
        }


def _algebraic_risk_arrays(
    lob: np.ndarray,
    normalization: dict[str, Any],
    contract: RiskContract,
) -> dict[str, np.ndarray]:
    prices, volumes, _ = _to_economic_units(lob, normalization)
    ask_price = prices[:, :10]
    bid_price = prices[:, 10:]
    ask_volume = volumes[:, :10]
    bid_volume = volumes[:, 10:]
    buy_slippage, buy_full = _market_order_slippage(
        prices, volumes, contract.target_size_shares, buy=True
    )
    sell_slippage, sell_full = _market_order_slippage(
        prices, volumes, contract.target_size_shares, buy=False
    )
    underfill = ~(buy_full & sell_full)
    max_slippage = np.fmax(buy_slippage, sell_slippage)
    max_slippage[underfill] = np.inf
    liquidity = _liquidity_within_ticks(
        ask_price,
        bid_price,
        ask_volume,
        bid_volume,
        ticks=5.0,
        target_size=contract.target_size_shares,
    )
    mid = 0.5 * (ask_price[:, 0] + bid_price[:, 0])
    best_total = ask_volume[:, 0] + bid_volume[:, 0]
    microprice = np.divide(
        ask_price[:, 0] * bid_volume[:, 0]
        + bid_price[:, 0] * ask_volume[:, 0],
        best_total,
        out=mid.copy(),
        where=best_total > 0,
    )
    alarm = (
        underfill
        | (max_slippage > contract.slippage_alarm_ticks)
        | (liquidity < contract.liquidity_alarm_target_units)
    )
    return {
        "max_slippage": max_slippage,
        "liquidity_5tick": liquidity,
        "underfill": underfill,
        "alarm": alarm,
        "microprice_offset": microprice - mid,
        "imbalance_l10": _imbalance(ask_volume, bid_volume, levels=10),
        "spread": ask_price[:, 0] - bid_price[:, 0],
    }


def freeze_risk_contract(
    train_lob: np.ndarray,
    normalization: dict[str, Any],
    max_threshold_snapshots: int,
) -> RiskContract:
    volume_scale = float(normalization["volume_scale"])
    best_depth = np.concatenate(
        [
            np.square(np.maximum(train_lob[:, 1, 0], 0.0) * volume_scale),
            np.square(np.maximum(train_lob[:, 1, 10], 0.0) * volume_scale),
        ]
    )
    positive = best_depth[np.isfinite(best_depth) & (best_depth > 0)]
    if positive.size == 0:
        raise ValueError("Training split has no positive best-level depth")
    target_size = float(np.median(positive))

    sample_count = min(int(max_threshold_snapshots), len(train_lob))
    indices = np.linspace(0, len(train_lob) - 1, sample_count, dtype=np.int64)
    sample = train_lob[indices, :, :, None]
    provisional = RiskContract(
        target_size_shares=target_size,
        slippage_alarm_ticks=math.inf,
        liquidity_alarm_target_units=-math.inf,
        threshold_train_snapshots=sample_count,
        threshold_index_sha256=sha256_bytes(indices.tobytes()),
    )
    arrays = _algebraic_risk_arrays(sample, normalization, provisional)
    finite_slippage = arrays["max_slippage"][np.isfinite(arrays["max_slippage"])]
    finite_liquidity = arrays["liquidity_5tick"][
        np.isfinite(arrays["liquidity_5tick"])
    ]
    if finite_slippage.size == 0 or finite_liquidity.size == 0:
        raise ValueError("Training split cannot define finite risk thresholds")
    return RiskContract(
        target_size_shares=target_size,
        slippage_alarm_ticks=float(np.quantile(finite_slippage, 0.90)),
        liquidity_alarm_target_units=float(np.quantile(finite_liquidity, 0.10)),
        threshold_train_snapshots=sample_count,
        threshold_index_sha256=sha256_bytes(indices.tobytes()),
    )


@dataclass
class StageTotals:
    frame_requests: int = 0
    frame_accepts: int = 0
    zero_tolerance_frame_accepts: int = 0
    block_requests: int = 0
    block_accepts: int = 0
    nonfinite: int = 0
    spread_rejects: int = 0
    ask_chain_rejects: int = 0
    bid_chain_rejects: int = 0
    negative_depth_rejects: int = 0
    accepted_risk_snapshots: int = 0
    underfills: int = 0
    alarms: int = 0
    finite_slippage_sum: float = 0.0
    finite_slippage_count: int = 0
    liquidity_sum: float = 0.0
    liquidity_count: int = 0

    def update(
        self,
        lob: np.ndarray,
        normalization: dict[str, Any],
        contract: RiskContract,
    ) -> tuple[np.ndarray, np.ndarray]:
        masks = strict_admissibility(lob, normalization)
        zero_masks = strict_admissibility(
            lob, normalization, zero_tolerance=True
        )
        accepted = masks["accepted"]
        batch, _, _, frames = lob.shape
        frame_matrix = accepted.reshape(batch, frames)
        block_accept = frame_matrix.all(axis=1)
        self.frame_requests += int(accepted.size)
        self.frame_accepts += int(accepted.sum())
        self.zero_tolerance_frame_accepts += int(
            zero_masks["accepted"].sum()
        )
        self.block_requests += int(batch)
        self.block_accepts += int(block_accept.sum())
        self.nonfinite += int(masks["nonfinite"].sum())
        self.spread_rejects += int(masks["spread"].sum())
        self.ask_chain_rejects += int(masks["ask_chain"].sum())
        self.bid_chain_rejects += int(masks["bid_chain"].sum())
        self.negative_depth_rejects += int(masks["negative_depth"].sum())

        self.accepted_risk_snapshots += int(accepted.sum())
        if accepted.any():
            # Fail closed: rejected/non-finite frames never enter the risk
            # algebra.  Repair-shift calculations on raw invalid states live
            # in the separately labelled algebraic diagnostic below.
            accepted_lob = _select_frames(lob, accepted)
            valid_arrays = _algebraic_risk_arrays(
                accepted_lob, normalization, contract
            )
            self.underfills += int(valid_arrays["underfill"].sum())
            self.alarms += int(valid_arrays["alarm"].sum())
            finite_slippage = valid_arrays["max_slippage"][
                np.isfinite(valid_arrays["max_slippage"])
            ]
            finite_liquidity = valid_arrays["liquidity_5tick"][
                np.isfinite(valid_arrays["liquidity_5tick"])
            ]
            self.finite_slippage_sum += float(finite_slippage.sum())
            self.finite_slippage_count += int(finite_slippage.size)
            self.liquidity_sum += float(finite_liquidity.sum())
            self.liquidity_count += int(finite_liquidity.size)
        return frame_matrix, block_accept

    def summary(self) -> dict[str, Any]:
        return {
            "frame_requests": self.frame_requests,
            "frame_acceptance_rate": self.frame_accepts
            / max(self.frame_requests, 1),
            "zero_tolerance_frame_acceptance_rate": (
                self.zero_tolerance_frame_accepts
                / max(self.frame_requests, 1)
            ),
            "block_requests": self.block_requests,
            "block_acceptance_rate": self.block_accepts
            / max(self.block_requests, 1),
            "rejection_reason_counts": {
                "nonfinite": self.nonfinite,
                "spread": self.spread_rejects,
                "ask_chain": self.ask_chain_rejects,
                "bid_chain": self.bid_chain_rejects,
                "negative_depth": self.negative_depth_rejects,
            },
            "accepted_risk_snapshots": self.accepted_risk_snapshots,
            "underfill_rate_among_accepted": (
                self.underfills / self.accepted_risk_snapshots
                if self.accepted_risk_snapshots
                else None
            ),
            "risk_alarm_rate_among_accepted": (
                self.alarms / self.accepted_risk_snapshots
                if self.accepted_risk_snapshots
                else None
            ),
            "finite_max_slippage_mean_ticks": (
                self.finite_slippage_sum / self.finite_slippage_count
                if self.finite_slippage_count
                else None
            ),
            "liquidity_5tick_mean_target_units": (
                self.liquidity_sum / self.liquidity_count
                if self.liquidity_count
                else None
            ),
        }


@dataclass
class PairedRiskShiftTotals:
    snapshots: int = 0
    slippage_abs_shift_sum: float = 0.0
    slippage_abs_shift_count: int = 0
    liquidity_abs_shift_sum: float = 0.0
    liquidity_abs_shift_count: int = 0
    alarm_disagreements: int = 0
    alarm_pairs: int = 0
    raw_negative_slippage: int = 0
    raw_finite_slippage: int = 0
    delivered_negative_slippage: int = 0
    delivered_finite_slippage: int = 0

    def update(
        self,
        pre: np.ndarray,
        delivered: np.ndarray,
        normalization: dict[str, Any],
        contract: RiskContract,
    ) -> None:
        pre_arrays = _algebraic_risk_arrays(pre, normalization, contract)
        delivered_arrays = _algebraic_risk_arrays(delivered, normalization, contract)
        self.snapshots += int(pre_arrays["alarm"].size)
        self.alarm_disagreements += int(
            np.sum(pre_arrays["alarm"] != delivered_arrays["alarm"])
        )
        self.alarm_pairs += int(pre_arrays["alarm"].size)

        pre_slip = pre_arrays["max_slippage"]
        delivered_slip = delivered_arrays["max_slippage"]
        paired_finite = np.isfinite(pre_slip) & np.isfinite(delivered_slip)
        self.slippage_abs_shift_sum += float(
            np.abs(pre_slip[paired_finite] - delivered_slip[paired_finite]).sum()
        )
        self.slippage_abs_shift_count += int(paired_finite.sum())
        raw_finite = np.isfinite(pre_slip)
        delivered_finite = np.isfinite(delivered_slip)
        self.raw_negative_slippage += int(np.sum(pre_slip[raw_finite] < -1e-8))
        self.raw_finite_slippage += int(raw_finite.sum())
        self.delivered_negative_slippage += int(
            np.sum(delivered_slip[delivered_finite] < -1e-8)
        )
        self.delivered_finite_slippage += int(delivered_finite.sum())

        pre_liq = pre_arrays["liquidity_5tick"]
        delivered_liq = delivered_arrays["liquidity_5tick"]
        finite_liq = np.isfinite(pre_liq) & np.isfinite(delivered_liq)
        self.liquidity_abs_shift_sum += float(
            np.abs(pre_liq[finite_liq] - delivered_liq[finite_liq]).sum()
        )
        self.liquidity_abs_shift_count += int(finite_liq.sum())

    def summary(self) -> dict[str, Any]:
        return {
            "paired_snapshots": self.snapshots,
            "market_order_slippage_abs_shift_mean_ticks": self.slippage_abs_shift_sum
            / max(self.slippage_abs_shift_count, 1),
            "liquidity_5tick_abs_shift_mean_target_units": self.liquidity_abs_shift_sum
            / max(self.liquidity_abs_shift_count, 1),
            "risk_alarm_disagreement_rate": self.alarm_disagreements
            / max(self.alarm_pairs, 1),
            "raw_negative_slippage_rate": self.raw_negative_slippage
            / max(self.raw_finite_slippage, 1),
            "delivered_negative_slippage_rate": self.delivered_negative_slippage
            / max(self.delivered_finite_slippage, 1),
        }


@dataclass
class ConsumerTrace:
    normalization: dict[str, Any]
    contract: RiskContract
    pre: StageTotals = field(default_factory=StageTotals)
    delivered: StageTotals = field(default_factory=StageTotals)
    paired_shift: PairedRiskShiftTotals = field(default_factory=PairedRiskShiftTotals)
    trajectory_pre_ok: list[np.ndarray] = field(default_factory=list)
    trajectory_delivered_ok: list[np.ndarray] = field(default_factory=list)
    first_pre_reject: list[np.ndarray] = field(default_factory=list)
    first_delivered_reject: list[np.ndarray] = field(default_factory=list)
    _batch_pre_ok: np.ndarray | None = None
    _batch_delivered_ok: np.ndarray | None = None
    _batch_first_pre: np.ndarray | None = None
    _batch_first_delivered: np.ndarray | None = None

    def start_batch(self, batch_size: int) -> None:
        self._batch_pre_ok = np.ones(batch_size, dtype=bool)
        self._batch_delivered_ok = np.ones(batch_size, dtype=bool)
        self._batch_first_pre = np.full(batch_size, -1, dtype=np.int64)
        self._batch_first_delivered = np.full(batch_size, -1, dtype=np.int64)

    def observe(self, pre: np.ndarray, delivered: np.ndarray, call_index: int) -> None:
        if self._batch_pre_ok is None:
            raise RuntimeError("start_batch must be called before observe")
        _, pre_block = self.pre.update(pre, self.normalization, self.contract)
        _, delivered_block = self.delivered.update(
            delivered, self.normalization, self.contract
        )
        self.paired_shift.update(pre, delivered, self.normalization, self.contract)
        pre_reject = ~pre_block
        delivered_reject = ~delivered_block
        unset_pre = (self._batch_first_pre < 0) & pre_reject
        unset_delivered = (self._batch_first_delivered < 0) & delivered_reject
        self._batch_first_pre[unset_pre] = call_index
        self._batch_first_delivered[unset_delivered] = call_index
        self._batch_pre_ok &= pre_block
        self._batch_delivered_ok &= delivered_block

    def end_batch(self) -> None:
        if self._batch_pre_ok is None:
            raise RuntimeError("No active batch")
        self.trajectory_pre_ok.append(self._batch_pre_ok.copy())
        self.trajectory_delivered_ok.append(self._batch_delivered_ok.copy())
        self.first_pre_reject.append(self._batch_first_pre.copy())
        self.first_delivered_reject.append(self._batch_first_delivered.copy())
        self._batch_pre_ok = None
        self._batch_delivered_ok = None
        self._batch_first_pre = None
        self._batch_first_delivered = None

    @staticmethod
    def _first_reject_summary(values: np.ndarray) -> dict[str, Any]:
        rejected = values[values >= 0]
        return {
            "rejected_trajectories": int(rejected.size),
            "first_reject_call_mean": float(np.mean(rejected))
            if rejected.size
            else None,
            "first_reject_call_median": float(np.median(rejected))
            if rejected.size
            else None,
        }

    def summary(self) -> dict[str, Any]:
        pre_ok = np.concatenate(self.trajectory_pre_ok)
        delivered_ok = np.concatenate(self.trajectory_delivered_ok)
        first_pre = np.concatenate(self.first_pre_reject)
        first_delivered = np.concatenate(self.first_delivered_reject)
        return {
            "pre": self.pre.summary(),
            "delivered": self.delivered.summary(),
            "algebraic_preprojection_shift_diagnostic": self.paired_shift.summary(),
            "trajectories": int(pre_ok.size),
            "pre_completed_without_rejection_rate": float(pre_ok.mean()),
            "delivered_completed_without_rejection_rate": float(
                delivered_ok.mean()
            ),
            "pre_first_rejection": self._first_reject_summary(first_pre),
            "delivered_first_rejection": self._first_reject_summary(
                first_delivered
            ),
        }


@dataclass
class ProjectionTotals:
    path_count: int = 0
    kernel_calls: int = 0
    trajectory_call_pairs: int = 0
    correction_sum: float = 0.0
    correction_max: float = 0.0
    nonzero_repairs: int = 0

    def register_paths(self, count: int) -> None:
        self.path_count += int(count)

    def observe(self, pre: torch.Tensor, projected: torch.Tensor) -> None:
        correction = (
            ((pre - projected) ** 2).sum(dim=(1, 2, 3)).sqrt().cpu().numpy()
        )
        self.kernel_calls += 1
        self.trajectory_call_pairs += int(correction.size)
        self.correction_sum += float(correction.sum())
        self.correction_max = max(self.correction_max, float(correction.max()))
        self.nonzero_repairs += int(np.sum(correction > 1e-8))

    def summary(self) -> dict[str, Any]:
        return {
            "path_count": self.path_count,
            "kernel_calls": self.kernel_calls,
            "calls_per_trajectory": self.trajectory_call_pairs
            / max(self.path_count, 1),
            "state_changing_call_rate": self.nonzero_repairs
            / max(self.trajectory_call_pairs, 1),
            "mean_correction_per_call": self.correction_sum
            / max(self.trajectory_call_pairs, 1),
            "mean_cumulative_correction_per_trajectory": self.correction_sum
            / max(self.path_count, 1),
            "max_correction": self.correction_max,
        }


@dataclass
class FeedbackRoundtripTotals:
    calls: int = 0
    states: int = 0
    nonzero_states: int = 0
    l2_sum: float = 0.0
    l2_max: float = 0.0

    def observe(self, exact_lob: torch.Tensor, roundtrip_lob: torch.Tensor) -> None:
        error = (
            ((exact_lob - roundtrip_lob) ** 2)
            .sum(dim=(1, 2, 3))
            .sqrt()
            .detach()
            .cpu()
            .numpy()
        )
        self.calls += 1
        self.states += int(error.size)
        self.nonzero_states += int(np.sum(error > 1e-8))
        self.l2_sum += float(error.sum())
        self.l2_max = max(self.l2_max, float(error.max()))

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "states": self.states,
            "nonzero_state_rate": self.nonzero_states / max(self.states, 1),
            "mean_l2_per_state": self.l2_sum / max(self.states, 1),
            "max_l2": self.l2_max,
        }


@dataclass(frozen=True)
class DecodedProjectionFeedback:
    pre_lob: torch.Tensor
    exact_projected_lob: torch.Tensor
    feedback_state: torch.Tensor
    roundtrip_lob: torch.Tensor


def exact_decoded_projection_feedback(
    model_state: torch.Tensor,
    representation: str,
    transform: Any,
    representation_stats: dict[str, torch.Tensor] | None,
    min_spread: float,
    min_gap: float,
    device: torch.device,
) -> DecodedProjectionFeedback:
    """Project in decoded LOB space; keep the exact output separate from feedback."""
    pre_lob = decode_from_model(
        model_state, representation, transform, representation_stats
    ).detach().cpu()
    exact_projected_lob = project_to_valid_lob(
        pre_lob,
        min_spread=min_spread,
        min_gap=min_gap,
        use_gpu_approx=False,
    )
    feedback_state = encode_for_model(
        exact_projected_lob.to(device),
        representation,
        transform,
        representation_stats,
    )
    roundtrip_lob = decode_from_model(
        feedback_state,
        representation,
        transform,
        representation_stats,
    ).detach().cpu()
    return DecodedProjectionFeedback(
        pre_lob=pre_lob,
        exact_projected_lob=exact_projected_lob,
        feedback_state=feedback_state,
        roundtrip_lob=roundtrip_lob,
    )


@torch.no_grad()
def observed_ddim_sample(
    model: torch.nn.Module,
    sde: Any,
    shape: tuple[int, int, int, int],
    history: torch.Tensor,
    action: torch.Tensor | None,
    ddim_steps: int,
    device: torch.device,
    project_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    observer_fn: Callable[[torch.Tensor, torch.Tensor, int, int], None],
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """DDIM-0 with paired pre/delivered predicted-clean observation."""
    model.eval()
    batch_size = shape[0]
    step_size = sde.N // ddim_steps
    time_indices = list(reversed(list(range(0, sde.N, step_size))[:ddim_steps]))
    alphas_cumprod = sde.alphas_cumprod.to(device)
    x = sde.prior_sampling(shape).to(device)
    noise_hash = sha256_bytes(x.detach().cpu().numpy().tobytes())
    last_pre = x

    for reverse_index, timestep in enumerate(time_indices):
        labels = torch.full(
            (batch_size,), timestep, device=device, dtype=torch.long
        )
        noise_pred = model(x, labels, history, action)
        alpha_bar = alphas_cumprod[timestep]
        pre_x0 = (
            x - torch.sqrt(1 - alpha_bar) * noise_pred
        ) / torch.sqrt(alpha_bar)
        pre_x0 = torch.clamp(pre_x0, -10, 10)
        delivered_x0 = project_fn(pre_x0) if project_fn is not None else pre_x0
        observer_fn(pre_x0, delivered_x0, reverse_index, timestep)
        last_pre = pre_x0

        if reverse_index < len(time_indices) - 1:
            previous_timestep = time_indices[reverse_index + 1]
            alpha_bar_previous = alphas_cumprod[previous_timestep]
            direction = torch.sqrt(1 - alpha_bar_previous) * noise_pred
            x = torch.sqrt(alpha_bar_previous) * delivered_x0 + direction
        else:
            x = delivered_x0
    return x, last_pre, noise_hash


def _decode_with_anchor(
    model_state: torch.Tensor,
    representation: str,
    transform: Any,
    representation_stats: dict[str, torch.Tensor] | None,
    anchor_mid: torch.Tensor | None,
) -> torch.Tensor:
    lob = decode_from_model(
        model_state, representation, transform, representation_stats
    )
    if anchor_mid is not None:
        lob = lob.clone()
        lob[:, 0] += anchor_mid[:, None, None]
    return lob


def _add_anchor(
    lob: torch.Tensor, anchor_mid: torch.Tensor | None
) -> torch.Tensor:
    if anchor_mid is None:
        return lob
    anchored = lob.clone()
    anchored[:, 0] += anchor_mid[:, None, None]
    return anchored


def _risk_fidelity(
    generated: np.ndarray,
    real: np.ndarray,
    normalization: dict[str, Any],
    contract: RiskContract,
) -> dict[str, Any]:
    generated_arrays = _algebraic_risk_arrays(generated, normalization, contract)
    real_arrays = _algebraic_risk_arrays(real, normalization, contract)
    generated_valid = strict_admissibility(generated, normalization)["accepted"]
    real_valid = strict_admissibility(real, normalization)["accepted"]
    generated_slip = generated_arrays["max_slippage"]
    real_slip = real_arrays["max_slippage"]
    generated_slip = generated_slip[np.isfinite(generated_slip)]
    real_slip = real_slip[np.isfinite(real_slip)]
    return {
        "generated_admissible_rate": float(generated_valid.mean()),
        "real_admissible_rate": float(real_valid.mean()),
        "market_order_slippage_w1_ticks": wasserstein_1d(
            generated_slip, real_slip
        ),
        "liquidity_5tick_w1_target_units": wasserstein_1d(
            generated_arrays["liquidity_5tick"],
            real_arrays["liquidity_5tick"],
        ),
        "risk_alarm_rate_abs_error": abs(
            float(generated_arrays["alarm"].mean())
            - float(real_arrays["alarm"].mean())
        ),
        "underfill_rate_abs_error": abs(
            float(generated_arrays["underfill"].mean())
            - float(real_arrays["underfill"].mean())
        ),
    }


def _paired_path_divergence(
    left: torch.Tensor,
    right: torch.Tensor,
    normalization: dict[str, Any],
    contract: RiskContract,
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"Paired path shape mismatch: {left.shape} vs {right.shape}")
    output: dict[str, Any] = {}
    for horizon in (10, 20, 50, left.shape[-1]):
        if horizon > left.shape[-1] or f"step_{horizon}" in output:
            continue
        difference = left[..., :horizon] - right[..., :horizon]
        trajectory_l2 = (difference**2).sum(dim=(1, 2, 3)).sqrt().numpy()
        left_risk = _algebraic_risk_arrays(
            left[..., :horizon].numpy(), normalization, contract
        )
        right_risk = _algebraic_risk_arrays(
            right[..., :horizon].numpy(), normalization, contract
        )
        output[f"step_{horizon}"] = {
            "trajectory_l2_mean": float(trajectory_l2.mean()),
            "trajectory_l2_p95": float(np.quantile(trajectory_l2, 0.95)),
            "risk_alarm_disagreement_rate": float(
                np.mean(left_risk["alarm"] != right_risk["alarm"])
            ),
        }
    return output


def run_variant(
    variant: str,
    model: torch.nn.Module,
    sde: Any,
    representation: str,
    transform: Any,
    representation_stats: dict[str, torch.Tensor] | None,
    seed_batches: list[dict[str, torch.Tensor]],
    horizon: int,
    pred_len: int,
    chunk_size: int,
    hist_len: int,
    ddim_steps: int,
    device: torch.device,
    price_reference: str,
    min_spread: float,
    min_gap: float,
    normalization: dict[str, Any],
    contract: RiskContract,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[str]]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    anytime_trace = ConsumerTrace(normalization, contract)
    endpoint_trace = ConsumerTrace(normalization, contract)
    projection_totals = ProjectionTotals()
    feedback_roundtrip_totals = FeedbackRoundtripTotals()
    generated_parts: list[torch.Tensor] = []
    real_parts: list[torch.Tensor] = []
    initial_noise_hashes: list[str] = []
    started = time.perf_counter()

    for batch_index, batch in enumerate(seed_batches):
        history_lob = batch["history"].to(device)
        action_chunks = batch["action_chunks"].to(device)
        batch_size = history_lob.shape[0]
        cur_hist_lob = history_lob.clone()
        collected: list[torch.Tensor] = []
        remaining = horizon
        anytime_trace.start_batch(batch_size)
        endpoint_trace.start_batch(batch_size)
        projection_totals.register_paths(batch_size)

        for chunk_index in range(action_chunks.shape[0]):
            if remaining <= 0:
                break
            action = action_chunks[chunk_index]
            if price_reference == "history_last_mid":
                anchor_mid = (
                    cur_hist_lob[:, 0, 0, -1]
                    + cur_hist_lob[:, 0, 10, -1]
                ) / 2
                history_for_model = cur_hist_lob.clone()
                history_for_model[:, 0] -= anchor_mid[:, None, None]
            else:
                anchor_mid = None
                history_for_model = cur_hist_lob
            history_model = encode_for_model(
                history_for_model,
                representation,
                transform,
                representation_stats,
            )

            latest_exact_projected_lob: torch.Tensor | None = None

            def project_model_state(model_state: torch.Tensor) -> torch.Tensor:
                nonlocal latest_exact_projected_lob
                feedback = exact_decoded_projection_feedback(
                    model_state,
                    representation,
                    transform,
                    representation_stats,
                    min_spread,
                    min_gap,
                    device,
                )
                feedback_roundtrip_totals.observe(
                    feedback.exact_projected_lob, feedback.roundtrip_lob
                )
                projection_totals.observe(
                    feedback.pre_lob, feedback.exact_projected_lob
                )
                latest_exact_projected_lob = (
                    feedback.exact_projected_lob.to(device)
                )
                return feedback.feedback_state

            def observe_model_states(
                pre_state: torch.Tensor,
                delivered_state: torch.Tensor,
                reverse_index: int,
                _timestep: int,
            ) -> None:
                pre_lob = _decode_with_anchor(
                    pre_state,
                    representation,
                    transform,
                    representation_stats,
                    anchor_mid,
                )
                if variant == "matched_repeated_pava":
                    if latest_exact_projected_lob is None:
                        raise RuntimeError("Missing exact PAVA consumer payload")
                    delivered_lob = _add_anchor(
                        latest_exact_projected_lob, anchor_mid
                    )
                else:
                    delivered_lob = _decode_with_anchor(
                        delivered_state,
                        representation,
                        transform,
                        representation_stats,
                        anchor_mid,
                    )
                call_index = chunk_index * ddim_steps + reverse_index
                anytime_trace.observe(
                    pre_lob.detach().cpu().numpy(),
                    delivered_lob.detach().cpu().numpy(),
                    call_index,
                )

            project_fn = (
                project_model_state
                if variant == "matched_repeated_pava"
                else None
            )
            block_model, last_pre_model, noise_hash = observed_ddim_sample(
                model,
                sde,
                (batch_size, 2, 20, pred_len),
                history_model,
                action,
                ddim_steps,
                device,
                project_fn,
                observe_model_states,
            )
            initial_noise_hashes.append(noise_hash)
            if variant == "matched_repeated_pava":
                if latest_exact_projected_lob is None:
                    raise RuntimeError("Missing final exact PAVA block")
                block_lob = _add_anchor(latest_exact_projected_lob, anchor_mid)
            else:
                block_lob = _decode_with_anchor(
                    block_model,
                    representation,
                    transform,
                    representation_stats,
                    anchor_mid,
                )
            last_pre_lob = _decode_with_anchor(
                last_pre_model,
                representation,
                transform,
                representation_stats,
                anchor_mid,
            )
            take = min(chunk_size, remaining)
            block_take = block_lob[..., :take]
            pre_endpoint = last_pre_lob[..., :take]

            if variant == "matched_posthoc_pava":
                pre_cpu = block_take.detach().cpu()
                delivered_cpu = project_to_valid_lob(
                    pre_cpu,
                    min_spread=min_spread,
                    min_gap=min_gap,
                    use_gpu_approx=False,
                )
                projection_totals.observe(pre_cpu, delivered_cpu)
                delivered_endpoint = delivered_cpu.to(device)
            else:
                delivered_endpoint = block_take

            endpoint_trace.observe(
                pre_endpoint.detach().cpu().numpy(),
                delivered_endpoint.detach().cpu().numpy(),
                chunk_index,
            )
            collected.append(delivered_endpoint.detach().cpu())
            cur_hist_lob = torch.cat(
                [cur_hist_lob[..., take:], delivered_endpoint], dim=-1
            )
            if cur_hist_lob.shape[-1] > hist_len:
                cur_hist_lob = cur_hist_lob[..., -hist_len:]
            remaining -= take

        anytime_trace.end_batch()
        endpoint_trace.end_batch()
        generated_parts.append(torch.cat(collected, dim=-1))
        real_parts.append(batch["real_future"].cpu())
        print(
            f"  {variant}: batch {batch_index + 1}/{len(seed_batches)} complete",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    generated = torch.cat(generated_parts, dim=0)
    real = torch.cat(real_parts, dim=0)
    price_unit = normalized_price_unit(normalization) or 1.0
    horizons = []
    for item in (10, 20, 50, horizon):
        if item <= horizon and item not in horizons:
            horizons.append(item)
    rollout_summary = summarize_rollout(
        generated,
        real,
        horizons,
        min_spread=min_spread,
        min_gap=min_gap,
        compute_projection=True,
        price_unit=price_unit,
    )
    summary = {
        "variant": variant,
        "n_histories": int(generated.shape[0]),
        "anytime_consumer": anytime_trace.summary(),
        "endpoint_consumer": endpoint_trace.summary(),
        "projection": projection_totals.summary(),
        "feedback_roundtrip_diagnostic": feedback_roundtrip_totals.summary(),
        "risk_fidelity": _risk_fidelity(
            generated.numpy(), real.numpy(), normalization, contract
        ),
        "rollout": rollout_summary,
        "runtime": {
            "seconds": elapsed,
            "trajectories_per_second": generated.shape[0] / max(elapsed, 1e-12),
            "generated_frames_per_second": generated.numel()
            / (2 * 20)
            / max(elapsed, 1e-12),
        },
    }
    return summary, generated, real, initial_noise_hashes


def _checkpoint_contract(
    checkpoint: dict[str, Any],
    raw: dict[str, Any],
    requested_representation: str,
    device: torch.device,
) -> tuple[
    str,
    str,
    float,
    float,
    Any,
    dict[str, torch.Tensor] | None,
]:
    representation = resolve_representation(requested_representation, checkpoint)
    checkpoint_args = checkpoint.get("args") or {}
    price_reference = resolve_price_reference(
        checkpoint_args.get("price_reference", "dataset"),
        bool(raw.get("has_raw_states", False)),
    )
    min_spread, min_gap = resolve_lob_constraints(
        raw.get("normalization"),
        min_spread=checkpoint_args.get("min_spread"),
        min_gap=checkpoint_args.get("min_gap"),
    )
    transform = make_lob_transform(
        representation,
        min_spread=min_spread,
        gap_eps=min_gap,
    )
    stats = representation_stats_to_device(checkpoint, device)
    return (
        representation,
        price_reference,
        min_spread,
        min_gap,
        transform,
        stats,
    )


def _git_receipt(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        )
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "status_sha256": sha256_bytes(status.encode()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_sha256": None}


def _source_receipt(root: Path) -> dict[str, str]:
    relative_paths = (
        "scripts/run_intermediate_consumer_panel.py",
        "scripts/summarize_intermediate_consumer.py",
        "scripts/snowball_test.py",
        "scripts/summarize_financial_sensitivity.py",
        "tests/test_intermediate_consumer.py",
        "tests/test_summarize_intermediate_consumer.py",
        "genlob/models/sampling.py",
        "genlob/utils/legal_lob.py",
        "genlob/utils/physics.py",
    )
    return {
        relative: sha256_file(root / relative)
        for relative in relative_paths
        if (root / relative).exists()
    }


def _validate_cell(
    methods: dict[str, dict[str, Any]],
    horizon: int,
    chunk_size: int,
    ddim_steps: int,
) -> dict[str, Any]:
    chunks = math.ceil(horizon / chunk_size)
    expected_calls = {
        "legal_lob": 0,
        "matched_posthoc_pava": chunks,
        "matched_repeated_pava": chunks * ddim_steps,
    }
    implementation_checks: dict[str, bool] = {}
    for variant, expected in expected_calls.items():
        observed = methods[variant]["projection"]["calls_per_trajectory"]
        implementation_checks[f"{variant}_projection_calls"] = math.isclose(
            observed, expected, abs_tol=1e-12
        )
        delivered_acceptance = methods[variant]["endpoint_consumer"][
            "delivered"
        ]["frame_acceptance_rate"]
        implementation_checks[
            f"{variant}_endpoint_delivered_acceptance"
        ] = math.isclose(
            delivered_acceptance, 1.0, abs_tol=1e-12
        )
    implementation_checks["repeated_anytime_pava_output_acceptance"] = math.isclose(
        methods["matched_repeated_pava"]["anytime_consumer"]["delivered"][
            "frame_acceptance_rate"
        ],
        1.0,
        abs_tol=1e-12,
    )
    science_checks = {}
    science_checks["legal_anytime_delivered_acceptance"] = math.isclose(
        methods["legal_lob"]["anytime_consumer"]["delivered"][
            "frame_acceptance_rate"
        ],
        1.0,
        abs_tol=1e-12,
    )
    science_checks["matched_post_anytime_gap_over_1pct"] = (
        methods["matched_posthoc_pava"]["anytime_consumer"]["pre"][
            "block_acceptance_rate"
        ]
        < 0.99
    )
    science_checks["repeated_nonzero_repair_over_1pct"] = (
        methods["matched_repeated_pava"]["projection"][
            "state_changing_call_rate"
        ]
        > 0.01
    )
    return {
        "implementation_checks": implementation_checks,
        "science_checks": science_checks,
        "all_implementation_gates_pass": all(implementation_checks.values()),
        "cell_level_mechanism_go": (
            all(implementation_checks.values()) and all(science_checks.values())
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="GOOG_databento_20190130_mbp10_rth_v2"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints_training_seeds/goog_databento_rth_v2_50ep"),
    )
    parser.add_argument(
        "--checkpoint-name",
        type=checkpoint_filename,
        default="best.pt",
        help="Checkpoint filename within each seed/representation directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/intermediate_consumer_goog_event"),
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--hist-len", type=int, default=50)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold-max-train-snapshots", type=int, default=100000)
    parser.add_argument(
        "--reference-rollout-csv",
        type=Path,
        default=Path(
            "paper_data/all_method_rollout_goog_event_640_3eval_per_run.csv"
        ),
    )
    parser.add_argument("--supersedes-root", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.chunk_size > args.pred_len:
        raise ValueError("--chunk-size must be <= --pred-len")
    root = Path(__file__).resolve().parent.parent
    device = select_device(args.device)
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output root: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    stock, nday = parse_snapshot_dataset_id(args.dataset)
    raw = load_nasdaq_multiday(args.data_dir, stock, nday)
    normalization = raw["normalization"]
    contract = freeze_risk_contract(
        raw["train"], normalization, args.threshold_max_train_snapshots
    )
    seed_batches = make_seed_batches_nasdaq(
        raw,
        hist_len=args.hist_len,
        horizon=args.horizon,
        chunk_size=args.chunk_size,
        action_mode="hold",
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        n_histories=None,
        price_reference="history_last_mid",
    )
    starts = np.concatenate(
        [np.asarray(batch["starts"], dtype=np.int64) for batch in seed_batches]
    )
    source_receipt = _source_receipt(root)
    dataset_dir = Path(raw["dataset_dir"])
    data_hash_names = (
        "states.npy",
        "states_raw.npy",
        "split.npy",
        "segment_id.npy",
        "actions.npy",
        "source_row_idx.npy",
    )
    supersedes = None
    if args.supersedes_root is not None:
        superseded_manifest = args.supersedes_root / "run_manifest.json"
        superseded_invalidation = args.supersedes_root / "INVALIDATION.md"
        if not superseded_manifest.exists() or not superseded_invalidation.exists():
            raise FileNotFoundError(
                "--supersedes-root must contain run_manifest.json and INVALIDATION.md"
            )
        supersedes = {
            "root": str(args.supersedes_root),
            "manifest_sha256": sha256_file(superseded_manifest),
            "invalidation_sha256": sha256_file(superseded_invalidation),
        }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "command": " ".join(sys.argv),
        "supersedes": supersedes,
        "config": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "checkpoint_root": str(args.checkpoint_root),
            "checkpoint_name": args.checkpoint_name,
            "train_seeds": args.train_seeds,
            "eval_seeds": args.eval_seeds,
            "hist_len": args.hist_len,
            "pred_len": args.pred_len,
            "chunk_size": args.chunk_size,
            "horizon": args.horizon,
            "ddim_steps": args.ddim_steps,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "n_histories": int(len(starts)),
            "action_mode": "hold",
            "weights": "ema",
            "device": str(device),
        },
        "risk_contract": contract.as_dict(),
        "provenance": {
            "git": _git_receipt(root),
            "sources_sha256": source_receipt,
            "data": {
                "dataset_dir": str(dataset_dir),
                "start_indices_sha256": sha256_bytes(starts.tobytes()),
                "normalization_sha256": sha256_file(
                    dataset_dir / "normalization.json"
                ),
                "manifest_sha256": sha256_file(
                    dataset_dir / "manifest.yaml"
                ),
                "arrays_sha256": {
                    name: sha256_file(dataset_dir / name)
                    for name in data_hash_names
                },
            },
            "reference_rollout": {
                "path": str(args.reference_rollout_csv),
                "sha256": sha256_file(args.reference_rollout_csv),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
            },
        },
        "cells": [],
    }
    manifest_path = args.output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")

    for train_seed in args.train_seeds:
        matched_checkpoint_path = (
            args.checkpoint_root
            / f"train_seed{train_seed}"
            / "matched_lob"
            / args.checkpoint_name
        )
        legal_checkpoint_path = (
            args.checkpoint_root
            / f"train_seed{train_seed}"
            / "legal_lob"
            / args.checkpoint_name
        )
        for path in (matched_checkpoint_path, legal_checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(path)
        print(f"Loading train seed {train_seed} checkpoints", flush=True)
        matched_model, matched_sde, matched_config, matched_checkpoint = load_genlob(
            matched_checkpoint_path, device, "ema"
        )
        legal_model, legal_sde, legal_config, legal_checkpoint = load_genlob(
            legal_checkpoint_path, device, "ema"
        )
        matched_config_signature = _config_signature(matched_config)
        legal_config_signature = _config_signature(legal_config)
        if matched_config_signature != legal_config_signature:
            raise ValueError(
                "LegalLOB and Matched backbones differ; cannot both reproduce "
                "the legacy RNG stream and share paired noise"
            )
        matched_contract = _checkpoint_contract(
            matched_checkpoint, raw, "matched_lob", device
        )
        legal_contract = _checkpoint_contract(
            legal_checkpoint, raw, "legal_lob", device
        )
        if matched_contract[1:4] != legal_contract[1:4]:
            raise ValueError(
                "Matched and Legal checkpoint price/constraint contracts differ: "
                f"{matched_contract[1:4]} vs {legal_contract[1:4]}"
            )
        price_reference = matched_contract[1]
        min_spread = matched_contract[2]
        min_gap = matched_contract[3]
        checkpoint_receipt = {
            "matched_lob": {
                "path": str(matched_checkpoint_path.resolve()),
                "sha256": sha256_file(matched_checkpoint_path),
                "epoch": matched_checkpoint.get("epoch"),
                "val_loss": matched_checkpoint.get("val_loss"),
                "config_sha256": matched_config_signature,
            },
            "legal_lob": {
                "path": str(legal_checkpoint_path.resolve()),
                "sha256": sha256_file(legal_checkpoint_path),
                "epoch": legal_checkpoint.get("epoch"),
                "val_loss": legal_checkpoint.get("val_loss"),
                "config_sha256": legal_config_signature,
            },
        }
        del (
            matched_model,
            matched_sde,
            matched_checkpoint,
            legal_model,
            legal_sde,
            legal_checkpoint,
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        for eval_seed in args.eval_seeds:
            cell_dir = (
                args.output_root
                / f"train_seed{train_seed}"
                / f"eval_seed{eval_seed}"
            )
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell: dict[str, Any] = {
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "status": "running",
                "checkpoints": checkpoint_receipt,
                "history_starts_sha256": sha256_bytes(starts.tobytes()),
                "methods": {},
            }
            cell_path = cell_dir / "consumer_stats.json"
            cell_path.write_text(json.dumps(_json_safe(cell), indent=2) + "\n")
            try:
                method_specs = (
                    (
                        "legal_lob",
                        legal_checkpoint_path,
                        legal_contract,
                    ),
                    (
                        "matched_posthoc_pava",
                        matched_checkpoint_path,
                        matched_contract,
                    ),
                    (
                        "matched_repeated_pava",
                        matched_checkpoint_path,
                        matched_contract,
                    ),
                )
                generated_by_method: dict[str, torch.Tensor] = {}
                noise_hashes_by_method: dict[str, list[str]] = {}
                post_load_rng_by_method: dict[str, str] = {}
                real_reference: torch.Tensor | None = None
                for variant, checkpoint_path, method_contract in method_specs:
                    (
                        representation,
                        method_price_reference,
                        method_min_spread,
                        method_min_gap,
                        transform,
                        representation_stats,
                    ) = method_contract
                    if (
                        method_price_reference != price_reference
                        or method_min_spread != min_spread
                        or method_min_gap != min_gap
                    ):
                        raise ValueError("Variant contract mismatch")
                    print(
                        f"RUN train_seed={train_seed} eval_seed={eval_seed} "
                        f"variant={variant}",
                        flush=True,
                    )
                    (
                        model,
                        sde,
                        loaded_config,
                        loaded_checkpoint,
                        post_load_rng_hash,
                    ) = load_variant_for_legacy_stream(
                        checkpoint_path, device, eval_seed
                    )
                    if _config_signature(loaded_config) != matched_config_signature:
                        raise ValueError("Loaded variant backbone config changed")
                    post_load_rng_by_method[variant] = post_load_rng_hash
                    summary, generated, real, noise_hashes = run_variant(
                        variant,
                        model,
                        sde,
                        representation,
                        transform,
                        representation_stats,
                        seed_batches,
                        args.horizon,
                        args.pred_len,
                        args.chunk_size,
                        args.hist_len,
                        args.ddim_steps,
                        device,
                        method_price_reference,
                        method_min_spread,
                        method_min_gap,
                        normalization,
                        contract,
                    )
                    del model, sde, loaded_checkpoint
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    cell["methods"][variant] = summary
                    generated_by_method[variant] = generated
                    noise_hashes_by_method[variant] = noise_hashes
                    if real_reference is None:
                        real_reference = real
                    elif not torch.equal(real_reference, real):
                        raise ValueError("Real rollout reference differs between methods")

                if len({tuple(value) for value in noise_hashes_by_method.values()}) != 1:
                    raise ValueError("Initial DDIM noise hashes are not paired")
                if len(set(post_load_rng_by_method.values())) != 1:
                    raise ValueError(
                        "Post-constructor RNG states differ across variants"
                    )
                cell["post_model_load_rng_sha256"] = post_load_rng_by_method
                cell["paired_initial_noise_sha256"] = noise_hashes_by_method[
                    "legal_lob"
                ]
                if real_reference is None:
                    raise ValueError("Real rollout reference was not produced")
                real_array = (
                    real_reference.detach().cpu().contiguous().numpy()
                )
                cell["real_futures_sha256"] = sha256_bytes(
                    real_array.tobytes()
                )
                cell["real_futures_shape"] = list(real_array.shape)
                cell["real_futures_dtype"] = str(real_array.dtype)
                cell["paired_matched_path_divergence"] = _paired_path_divergence(
                    generated_by_method["matched_posthoc_pava"],
                    generated_by_method["matched_repeated_pava"],
                    normalization,
                    contract,
                )
                cell["validation"] = _validate_cell(
                    cell["methods"], args.horizon, args.chunk_size, args.ddim_steps
                )
                cell["status"] = "completed"
            except Exception as error:
                cell["status"] = "failed"
                cell["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                cell_path.write_text(
                    json.dumps(_json_safe(cell), indent=2) + "\n"
                )
                manifest["cells"].append(
                    {
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "status": "failed",
                        "path": str(cell_path.resolve()),
                        "error": cell["error"],
                    }
                )
                manifest["status"] = "failed"
                manifest_path.write_text(
                    json.dumps(_json_safe(manifest), indent=2) + "\n"
                )
                raise
            cell_path.write_text(json.dumps(_json_safe(cell), indent=2) + "\n")
            manifest["cells"].append(
                {
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "status": "completed",
                    "path": str(cell_path.resolve()),
                    "sha256": sha256_file(cell_path),
                }
            )
            manifest_path.write_text(
                json.dumps(_json_safe(manifest), indent=2) + "\n"
            )

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    manifest["status"] = "completed"
    manifest["completed_cells"] = sum(
        cell["status"] == "completed" for cell in manifest["cells"]
    )
    manifest["failed_cells"] = sum(
        cell["status"] == "failed" for cell in manifest["cells"]
    )
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    print(f"Intermediate consumer panel complete: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

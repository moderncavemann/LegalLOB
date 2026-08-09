#!/usr/bin/env python3
"""Chunk-rollout stability test for raw, matched, and LegalLOB checkpoints."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genlob.data.nasdaq import (
    load_nasdaq_multiday,
    normalize_anchor_relative_sequence,
    normalized_price_unit,
    parse_snapshot_dataset_id,
    resolve_lob_constraints,
    resolve_price_reference,
)
from genlob.models.genlob_v2 import GenLOBv2
from genlob.eval.metrics import json_safe
from genlob.models.sampling import ddim_sample
from genlob.models.sde import VPSDE
from genlob.utils.device import select_device
from genlob.utils.legal_lob import COORDINATE_REPRESENTATIONS, make_lob_transform
from genlob.utils.physics import (
    compute_projection_distance,
    compute_violation_rate,
    project_to_valid_lob,
)

# =============================================================================
# Per-step violation rates
# =============================================================================

def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def violation_rates(x: torch.Tensor) -> dict:
    """
    x: [B, 2, 20, T]  -> returns 1D tensors of length T for each metric.
    Rates are per-step fractions over batch and (for ladder) over adjacent level pairs.
    """
    prices = x[:, 0]                # [B, 20, T]
    volumes = x[:, 1]               # [B, 20, T]
    ask = prices[:, :10]            # [B, 10, T]
    bid = prices[:, 10:]            # [B, 10, T]

    crossed = (ask[:, 0] <= bid[:, 0]).float().mean(dim=0)       # [T]
    ask_lv = (ask[:, :-1] > ask[:, 1:]).float().mean(dim=(0, 1))  # [T]
    bid_lv = (bid[:, :-1] < bid[:, 1:]).float().mean(dim=(0, 1))  # [T]
    neg_vol = (volumes < 0).float().mean(dim=(0, 1))              # [T]
    return {
        "crossed": crossed.cpu(),
        "ask_ladder": ask_lv.cpu(),
        "bid_ladder": bid_lv.cpu(),
        "neg_vol": neg_vol.cpu(),
    }


def any_invalid_snapshot_rate(
    x: torch.Tensor,
    min_spread: float,
    min_gap: float,
) -> float:
    """Fraction of (trajectory, frame) snapshots violating any constraint."""
    prices = x[:, 0]
    volumes = x[:, 1]
    ask = prices[:, :10]
    bid = prices[:, 10:]
    tolerance = max(1e-7, abs(min_spread) * 2e-5, abs(min_gap) * 2e-5)
    invalid = (
        (ask[:, 0] - bid[:, 0] < min_spread - tolerance)
        | (torch.diff(ask, dim=1) < min_gap - tolerance).any(dim=1)
        | (-torch.diff(bid, dim=1) < min_gap - tolerance).any(dim=1)
        | (volumes < -tolerance).any(dim=1)
    )
    return float(invalid.float().mean())


# =============================================================================
# Rollout adapters (all return [B, 2, 20, horizon])
# =============================================================================

@torch.no_grad()
def rollout_genlob(model, sde, history, action, horizon, pred_len, ddim_steps,
                   device, project=True, hist_len=None):
    """
    Autoregressive rollout with GenLOB: sample pred_len at a time, slide window.
    """
    hist_len = hist_len or history.shape[-1]
    proj_fn = project_to_valid_lob if project else None
    cur_hist = history.clone()
    B = history.shape[0]
    collected = []
    remaining = horizon
    while remaining > 0:
        shape = (B, 2, 20, pred_len)
        block = ddim_sample(model, sde, shape, cur_hist, action,
                            ddim_steps=ddim_steps, device=device, project_fn=proj_fn)
        take = min(remaining, pred_len)
        collected.append(block[..., :take])
        cur_hist = torch.cat([cur_hist[..., take:], block[..., :take]], dim=-1)
        if cur_hist.shape[-1] > hist_len:
            cur_hist = cur_hist[..., -hist_len:]
        remaining -= take
    return torch.cat(collected, dim=-1)


# =============================================================================
# Bootstrap CI over histories (batch axis)
# =============================================================================

def bootstrap_curve(per_history_violations: np.ndarray, n_boot=300, seed=0):
    """
    per_history_violations: [B, T] boolean-like per-sample violation at each step.
    Returns (mean[T], low95[T], high95[T]).
    """
    rng = np.random.default_rng(seed)
    B, T = per_history_violations.shape
    boots = np.empty((n_boot, T))
    for i in range(n_boot):
        idx = rng.integers(0, B, size=B)
        boots[i] = per_history_violations[idx].mean(axis=0)
    mean = boots.mean(axis=0)
    low = np.percentile(boots, 2.5, axis=0)
    high = np.percentile(boots, 97.5, axis=0)
    return mean, low, high


def violations_per_history(x: torch.Tensor, metric: str) -> np.ndarray:
    """Return [B, T] float array of per-(history, step) violation indicator."""
    ask = x[:, 0, :10]
    bid = x[:, 0, 10:]
    if metric == "crossed":
        return (ask[:, 0] <= bid[:, 0]).float().cpu().numpy()
    if metric == "ask_ladder":
        # fraction of violating adjacent pairs per (B, T)
        v = (ask[:, :-1] > ask[:, 1:]).float().mean(dim=1)
        return v.cpu().numpy()
    if metric == "bid_ladder":
        v = (bid[:, :-1] < bid[:, 1:]).float().mean(dim=1)
        return v.cpu().numpy()
    raise KeyError(metric)


# =============================================================================
# Empirical floor (real held-out data)
# =============================================================================

def compute_empirical_floor_nasdaq(raw: dict, horizon: int, max_histories: int = 512):
    """
    NASDAQ: test states are (N, 2, 20) per single timestep. Take `horizon`
    consecutive timesteps from the test array as each rollout's floor.
    """
    test = raw["test"]                             # (N, 2, 20) numpy
    N = test.shape[0]
    n_floors = min(max_histories, N - horizon)
    # Stride so each floor uses disjoint horizon windows
    stride = max(1, (N - horizon) // max(n_floors, 1))
    floors = []
    for i in range(n_floors):
        start = i * stride
        chunk = test[start:start + horizon]        # (horizon, 2, 20)
        floors.append(chunk.transpose(1, 2, 0))    # (2, 20, horizon)
    import numpy as np
    real = torch.from_numpy(np.stack(floors)).float()  # (n_floors, 2, 20, horizon)
    return real


# =============================================================================
# Plotting
# =============================================================================

def plot_curves(per_model_results: dict, out_path: Path, metric_key: str,
                title: str, ylabel: str, log_y: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {
        "Generated": ("#1f77b4", "-"),
        "Empirical floor": ("#2ca02c", "--"),
    }

    for name, curves in per_model_results.items():
        if metric_key not in curves:
            continue
        mean, low, high = curves[metric_key]
        color, ls = palette.get(name, ("#333", "-"))
        t = np.arange(1, len(mean) + 1)
        ax.plot(t, mean * 100, color=color, linestyle=ls, linewidth=2.2,
                label=name, marker="o", markersize=3, markevery=max(len(t) // 20, 1))
        ax.fill_between(t, low * 100, high * 100, color=color, alpha=0.15)

    ax.set_xlabel("Rollout step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    if log_y:
        ax.set_yscale("symlog", linthresh=1e-2)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Loaders
# =============================================================================

def load_genlob(ckpt_path, device, weights="ema"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GenLOBv2(cfg).to(device)
    if weights == "ema" and "ema_state_dict" in ckpt:
        state = ckpt["ema_state_dict"]
    else:
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, VPSDE(N=cfg.num_steps), cfg, ckpt


# =============================================================================
# Legal GenLOB rollout and drift metrics
# =============================================================================

def resolve_representation(requested: str, ckpt: dict) -> str:
    if requested != "auto":
        return requested
    ckpt_args = ckpt.get("args") or {}
    return ckpt_args.get("representation", "raw")


def representation_stats_to_device(ckpt: dict, device: torch.device) -> dict | None:
    stats = ckpt.get("representation_stats")
    if not stats:
        return None
    return {
        "mean": stats["mean"].to(device),
        "std": stats["std"].to(device),
    }


def make_seed_batches_nasdaq(raw: dict, hist_len: int, horizon: int,
                             chunk_size: int, action_mode: str,
                             batch_size: int, n_batches: int | None,
                             n_histories: int | None,
                             price_reference: str):
    """Build deterministic rollout seeds from the NASDAQ test split.

    Each item contains:
      history: [B, 2, 20, hist_len]
      real_future: [B, 2, 20, horizon]
      action_chunks: [n_chunks, B, 5]
    """
    test = raw["test"]
    test_raw = raw.get("test_raw")
    actions = raw["test_actions"]
    day_id = raw.get("test_day_id")
    segment_id = raw.get("test_segment_id")
    n_possible_starts = len(test) - hist_len - horizon + 1
    if n_possible_starts <= 0:
        raise ValueError(
            f"Test split too short for hist_len={hist_len}, horizon={horizon}"
        )

    total = n_histories if n_histories is not None else batch_size * (n_batches or 1)
    total = min(total, n_possible_starts)
    candidate_starts = np.arange(n_possible_starts, dtype=np.int64)
    groups = []
    if day_id is not None:
        unique = np.unique(day_id)
        if not (unique.size == 1 and int(unique[0]) < 0):
            groups.append(day_id)
    if segment_id is not None:
        groups.append(segment_id)
    if groups:
        valid = []
        total_window = hist_len + horizon
        for start in candidate_starts:
            if all(
                len(group[start:start + total_window]) == total_window
                and np.all(group[start:start + total_window] >= 0)
                and np.all(group[start:start + total_window] == group[start])
                for group in groups
            ):
                valid.append(int(start))
        candidate_starts = np.asarray(valid, dtype=np.int64)
    if candidate_starts.size == 0:
        raise ValueError("No rollout starts remain after day-aware filtering")
    total = min(total, candidate_starts.size)
    selected_positions = np.linspace(0, candidate_starts.size - 1, total, dtype=np.int64)
    starts = candidate_starts[selected_positions]

    n_chunks = int(np.ceil(horizon / chunk_size))
    batches = []
    for batch_start in range(0, total, batch_size):
        batch_starts = starts[batch_start:batch_start + batch_size]
        histories = []
        futures = []
        action_chunks = []

        for start in batch_starts:
            if price_reference == "history_last_mid":
                window = normalize_anchor_relative_sequence(
                    test_raw[start:start + hist_len + horizon],
                    hist_len=hist_len,
                    normalization=raw["normalization"],
                ).numpy()
                h = window[:hist_len].transpose(1, 2, 0)
                f = window[hist_len:].transpose(1, 2, 0)
            else:
                h = test[start:start + hist_len].transpose(1, 2, 0)
                f = test[start + hist_len:start + hist_len + horizon].transpose(1, 2, 0)
            histories.append(h)
            futures.append(f)

        for chunk_idx in range(n_chunks):
            offset = chunk_idx * chunk_size
            if action_mode == "replay":
                chunk_actions = []
                for start in batch_starts:
                    action_idx = start + hist_len + offset - 1
                    action_idx = min(action_idx, len(actions) - 1)
                    chunk_actions.append(actions[action_idx])
                action_chunks.append(np.stack(chunk_actions).astype(np.float32))
            elif action_mode == "hold":
                action_chunks.append(np.zeros((len(batch_starts), 5), dtype=np.float32))
            elif action_mode in {"buy", "sell"}:
                direction = 1.0 if action_mode == "buy" else -1.0
                template = np.zeros((len(batch_starts), 5), dtype=np.float32)
                template[:, 0] = direction
                template[:, 1] = 0.5
                template[:, 2] = 0.5
                action_chunks.append(template)
            else:
                raise ValueError(f"Unknown action_mode: {action_mode}")

        batches.append({
            "history": torch.from_numpy(np.stack(histories)).float(),
            "real_future": torch.from_numpy(np.stack(futures)).float(),
            "action_chunks": torch.from_numpy(np.stack(action_chunks)).float(),
            "starts": batch_starts.tolist(),
        })
    return batches


def encode_for_model(x: torch.Tensor, representation: str,
                     transform,
                     stats: dict | None) -> torch.Tensor:
    if representation in COORDINATE_REPRESENTATIONS:
        assert transform is not None
        z = transform.encode(x)
        if stats is not None:
            z = (z - stats["mean"]) / stats["std"]
        return z
    if representation == "raw":
        return x
    raise ValueError(f"Unknown representation: {representation}")


def decode_from_model(z: torch.Tensor, representation: str,
                      transform,
                      stats: dict | None) -> torch.Tensor:
    if representation in COORDINATE_REPRESENTATIONS:
        assert transform is not None
        if stats is not None:
            z = z * stats["std"] + stats["mean"]
        return transform.decode(z)
    if representation == "raw":
        return z
    raise ValueError(f"Unknown representation: {representation}")


@torch.no_grad()
def rollout_genlob_chunked(model, sde, history_lob, action_chunks, horizon,
                           pred_len, chunk_size, ddim_steps, device,
                           representation, transform, representation_stats,
                           projection_mode="none", hist_len=None,
                           price_reference="dataset", min_spread=1e-6,
                           min_gap=0.0, sampler_projection_mode="none",
                           return_diagnostics=False):
    """Chunk rollout with real usage semantics: sample pred_len, append chunk_size."""
    if chunk_size > pred_len:
        raise ValueError("--chunk-size must be <= --pred-len")

    hist_len = hist_len or history_lob.shape[-1]
    cur_hist_lob = history_lob.to(device).clone()
    action_chunks = action_chunks.to(device)
    B = cur_hist_lob.shape[0]
    collected = []
    remaining = horizon
    n_chunks = action_chunks.shape[0]
    sampler_l2_values = []
    sampler_pre_violations = []
    posthoc_l2_values = []
    posthoc_pre_violations = []
    projection_prefixes = []

    if sampler_projection_mode != "none" and representation != "raw":
        raise ValueError("sampler_projection_mode is only valid for raw representation")

    def exact_project_with_record(z):
        z_cpu = z.detach().cpu()
        projected_cpu = project_to_valid_lob(
            z_cpu,
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=False,
        )
        sampler_l2_values.append(
            ((z_cpu - projected_cpu) ** 2).sum(dim=(1, 2, 3)).sqrt()
        )
        violations = compute_violation_rate(
            z_cpu, min_spread=min_spread, min_gap=min_gap
        )
        violations["any_invalid_snapshot_rate"] = any_invalid_snapshot_rate(
            z_cpu, min_spread=min_spread, min_gap=min_gap
        )
        sampler_pre_violations.append(violations)
        return projected_cpu.to(z.device)

    def fast_project_with_record(z):
        projected = project_to_valid_lob(
            z,
            min_spread=min_spread,
            min_gap=min_gap,
            use_gpu_approx=True,
        )
        z_cpu = z.detach().cpu()
        projected_cpu = projected.detach().cpu()
        sampler_l2_values.append(
            ((z_cpu - projected_cpu) ** 2).sum(dim=(1, 2, 3)).sqrt()
        )
        violations = compute_violation_rate(
            z_cpu, min_spread=min_spread, min_gap=min_gap
        )
        violations["any_invalid_snapshot_rate"] = any_invalid_snapshot_rate(
            z_cpu, min_spread=min_spread, min_gap=min_gap
        )
        sampler_pre_violations.append(violations)
        return projected

    if sampler_projection_mode == "none":
        sampler_project_fn = None
    elif sampler_projection_mode == "exact":
        sampler_project_fn = exact_project_with_record
    elif sampler_projection_mode == "fast":
        sampler_project_fn = fast_project_with_record
    else:
        raise ValueError(f"Unknown sampler_projection_mode: {sampler_projection_mode}")

    for chunk_idx in range(n_chunks):
        if remaining <= 0:
            break
        action = action_chunks[chunk_idx]
        if price_reference == "history_last_mid":
            anchor_mid = (
                cur_hist_lob[:, 0, 0, -1] + cur_hist_lob[:, 0, 10, -1]
            ) / 2
            hist_for_model = cur_hist_lob.clone()
            hist_for_model[:, 0] -= anchor_mid[:, None, None]
        else:
            anchor_mid = None
            hist_for_model = cur_hist_lob
        hist_model = encode_for_model(
            hist_for_model, representation, transform, representation_stats,
        )
        shape = (B, 2, 20, pred_len)
        block_model = ddim_sample(
            model, sde, shape, hist_model, action,
            ddim_steps=ddim_steps, device=device, project_fn=sampler_project_fn,
        )
        block_lob = decode_from_model(
            block_model, representation, transform, representation_stats,
        )
        if anchor_mid is not None:
            block_lob[:, 0] += anchor_mid[:, None, None]

        take = min(chunk_size, remaining)
        block_take = block_lob[..., :take]
        if projection_mode == "exact":
            pre_projection = block_take.detach().cpu()
            projected = project_to_valid_lob(
                pre_projection, min_spread=min_spread, min_gap=min_gap,
                use_gpu_approx=False,
            )
            posthoc_l2_values.append(
                ((pre_projection - projected) ** 2).sum(dim=(1, 2, 3)).sqrt()
            )
            violations = compute_violation_rate(
                pre_projection, min_spread=min_spread, min_gap=min_gap,
            )
            violations["any_invalid_snapshot_rate"] = any_invalid_snapshot_rate(
                pre_projection, min_spread=min_spread, min_gap=min_gap
            )
            posthoc_pre_violations.append(violations)
            block_take = projected.to(device)
        elif projection_mode == "fast":
            pre_projection = block_take.detach().cpu()
            projected = project_to_valid_lob(
                block_take, min_spread=min_spread, min_gap=min_gap,
                use_gpu_approx=True,
            )
            posthoc_l2_values.append(
                ((pre_projection - projected.detach().cpu()) ** 2)
                .sum(dim=(1, 2, 3)).sqrt()
            )
            violations = compute_violation_rate(
                pre_projection, min_spread=min_spread, min_gap=min_gap,
            )
            violations["any_invalid_snapshot_rate"] = any_invalid_snapshot_rate(
                pre_projection, min_spread=min_spread, min_gap=min_gap
            )
            posthoc_pre_violations.append(violations)
            block_take = projected
        elif projection_mode != "none":
            raise ValueError(f"Unknown projection_mode: {projection_mode}")

        collected.append(block_take.cpu())
        cur_hist_lob = torch.cat([cur_hist_lob[..., take:], block_take], dim=-1)
        if cur_hist_lob.shape[-1] > hist_len:
            cur_hist_lob = cur_hist_lob[..., -hist_len:]
        remaining -= take
        projection_prefixes.append(
            {
                "horizon": int(horizon - remaining),
                "sampler_calls": len(sampler_l2_values),
                "posthoc_calls": len(posthoc_l2_values),
            }
        )

    rollout = torch.cat(collected, dim=-1)
    if not return_diagnostics:
        return rollout

    def summarize_projection_path(l2_values, pre_violations):
        if not l2_values:
            return None
        l2_stack = torch.stack(l2_values)  # [calls, B]
        cumulative = l2_stack.sum(dim=0)
        result = {
            "n_projection_calls": int(l2_stack.shape[0]),
            "per_call_projection_l2_mean": float(l2_stack.mean()),
            "per_call_projection_l2_max": float(l2_stack.max()),
            "cumulative_projection_l2_mean": float(cumulative.mean()),
            "cumulative_projection_l2_max": float(cumulative.max()),
        }
        for metric in (
            "any_invalid_snapshot_rate",
            "crossed_book_rate",
            "ask_ladder_violation",
            "bid_ladder_violation",
            "negative_volume_rate",
        ):
            values = np.asarray([item[metric] for item in pre_violations])
            result[f"pre_projection_{metric}_mean"] = float(values.mean())
            result[f"pre_projection_{metric}_max"] = float(values.max())
        return result

    by_horizon = {}
    for prefix in projection_prefixes:
        by_horizon[f"step_{prefix['horizon']}"] = {
            "sampler_projection": summarize_projection_path(
                sampler_l2_values[: prefix["sampler_calls"]],
                sampler_pre_violations[: prefix["sampler_calls"]],
            ),
            "posthoc_chunk_projection": summarize_projection_path(
                posthoc_l2_values[: prefix["posthoc_calls"]],
                posthoc_pre_violations[: prefix["posthoc_calls"]],
            ),
        }

    return rollout, {
        "n_trajectories": int(B),
        "sampler_projection": summarize_projection_path(
            sampler_l2_values, sampler_pre_violations,
        ),
        "posthoc_chunk_projection": summarize_projection_path(
            posthoc_l2_values, posthoc_pre_violations,
        ),
        "by_horizon": by_horizon,
    }


def merge_rollout_projection_diagnostics(parts: list[dict]) -> dict:
    """Combine per-batch path diagnostics with trajectory weighting."""
    def merge_path(key: str, scoped_parts: list[dict]) -> dict | None:
        entries = [part for part in scoped_parts if part.get(key) is not None]
        if not entries:
            return None
        total_trajectories = sum(part["n_trajectories"] for part in entries)
        first = entries[0][key]
        summary = {
            "n_batches": len(entries),
            "n_trajectories": total_trajectories,
            "n_projection_calls_per_trajectory": first["n_projection_calls"],
            "n_projection_kernel_calls": int(
                sum(part[key]["n_projection_calls"] for part in entries)
            ),
        }
        for field in (
            "per_call_projection_l2_mean",
            "cumulative_projection_l2_mean",
            "pre_projection_crossed_book_rate_mean",
            "pre_projection_ask_ladder_violation_mean",
            "pre_projection_bid_ladder_violation_mean",
            "pre_projection_negative_volume_rate_mean",
            "pre_projection_any_invalid_snapshot_rate_mean",
        ):
            summary[field] = float(
                sum(part[key][field] * part["n_trajectories"] for part in entries)
                / total_trajectories
            )
        for field in (
            "per_call_projection_l2_max",
            "cumulative_projection_l2_max",
            "pre_projection_crossed_book_rate_max",
            "pre_projection_ask_ladder_violation_max",
            "pre_projection_bid_ladder_violation_max",
            "pre_projection_negative_volume_rate_max",
            "pre_projection_any_invalid_snapshot_rate_max",
        ):
            summary[field] = float(max(part[key][field] for part in entries))
        return summary

    output = {
        key: merge_path(key, parts)
        for key in ("sampler_projection", "posthoc_chunk_projection")
    }
    horizon_keys = sorted(
        {
            horizon_key
            for part in parts
            for horizon_key in (part.get("by_horizon") or {})
        },
        key=lambda item: int(item.removeprefix("step_")),
    )
    output["by_horizon"] = {}
    for horizon_key in horizon_keys:
        scoped_parts = [
            {
                "n_trajectories": part["n_trajectories"],
                **((part.get("by_horizon") or {}).get(horizon_key) or {}),
            }
            for part in parts
            if horizon_key in (part.get("by_horizon") or {})
        ]
        output["by_horizon"][horizon_key] = {
            key: merge_path(key, scoped_parts)
            for key in ("sampler_projection", "posthoc_chunk_projection")
        }
    return output


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    if len(a) != n:
        a = np.random.default_rng(0).choice(a, size=n, replace=False)
    if len(b) != n:
        b = np.random.default_rng(1).choice(b, size=n, replace=False)
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def corr_pair(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    corr = corr_pair(a, b)
    return corr if corr is not None else float("nan")


def panel_acf(x: np.ndarray, lags: list[int]) -> dict[str, float]:
    """Mean per-history autocorrelation for panel data shaped [B, T]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    out = {}
    for lag in lags:
        vals = []
        if x.shape[1] <= lag:
            out[f"lag_{lag}"] = float("nan")
            continue
        for row in x:
            a = row[:-lag]
            b = row[lag:]
            corr = corr_pair(a, b)
            if corr is not None:
                vals.append(corr)
        out[f"lag_{lag}"] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def rbf_mmd_1d(a: np.ndarray, b: np.ndarray, max_samples: int = 1024) -> float:
    """Biased deterministic RBF MMD^2 diagnostic for one-dimensional samples."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if a.size > max_samples:
        a = rng.choice(a, size=max_samples, replace=False)
    if b.size > max_samples:
        b = rng.choice(b, size=max_samples, replace=False)
    pooled = np.concatenate([a, b])
    if pooled.size < 2:
        return float("nan")
    bandwidth = np.median(np.abs(pooled[:, None] - pooled[None, :]))
    bandwidth = float(max(bandwidth, 1e-8))
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    k_xx = np.exp(-gamma * (a[:, None] - a[None, :]) ** 2)
    k_yy = np.exp(-gamma * (b[:, None] - b[None, :]) ** 2)
    k_xy = np.exp(-gamma * (a[:, None] - b[None, :]) ** 2)
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def temporal_corr(real: torch.Tensor, gen: torch.Tensor, step: int) -> float:
    vals = []
    r = ((real[:, 0, 0, :step] + real[:, 0, 10, :step]) / 2).numpy()
    g = ((gen[:, 0, 0, :step] + gen[:, 0, 10, :step]) / 2).numpy()
    for i in range(r.shape[0]):
        corr = corr_pair(r[i], g[i])
        if corr is not None:
            vals.append(corr)
    return float(np.nanmean(vals)) if vals else float("nan")


def order_imbalance(x: torch.Tensor) -> torch.Tensor:
    ask_vol = x[:, 1, :10, :].sum(dim=1)
    bid_vol = x[:, 1, 10:, :].sum(dim=1)
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)


def queue_imbalance(x: torch.Tensor) -> torch.Tensor:
    """Best-level depth imbalance proxy shaped [B, T]."""
    ask_best = x[:, 1, 0, :]
    bid_best = x[:, 1, 10, :]
    return (bid_best - ask_best) / (bid_best + ask_best + 1e-8)


def summarize_rollout(gen: torch.Tensor, real: torch.Tensor, steps: list[int],
                      min_spread: float, min_gap: float,
                      compute_projection: bool, price_unit: float = 1.0) -> dict:
    summary = {}
    gen_mid = (gen[:, 0, 0, :] + gen[:, 0, 10, :]) / 2 * price_unit
    real_mid = (real[:, 0, 0, :] + real[:, 0, 10, :]) / 2 * price_unit
    gen_spread = (gen[:, 0, 0, :] - gen[:, 0, 10, :]) * price_unit
    real_spread = (real[:, 0, 0, :] - real[:, 0, 10, :]) * price_unit
    gen_imb = order_imbalance(gen)
    real_imb = order_imbalance(real)
    gen_queue = queue_imbalance(gen)
    real_queue = queue_imbalance(real)
    gen_depth = gen[:, 1].mean(dim=1)
    real_depth = real[:, 1].mean(dim=1)

    for step in steps:
        idx = step - 1
        gen_step = gen[..., idx:idx + 1]
        real_step = real[..., idx:idx + 1]
        viol = compute_violation_rate(gen_step, min_spread=min_spread, min_gap=min_gap)
        real_viol = compute_violation_rate(real_step, min_spread=min_spread, min_gap=min_gap)
        viol["any_invalid_snapshot_rate"] = any_invalid_snapshot_rate(
            gen_step, min_spread=min_spread, min_gap=min_gap
        )
        real_viol["any_invalid_snapshot_rate"] = any_invalid_snapshot_rate(
            real_step, min_spread=min_spread, min_gap=min_gap
        )

        proj_dist = None
        if compute_projection:
            proj = project_to_valid_lob(gen_step.cpu(), min_spread=min_spread, min_gap=min_gap,
                                        use_gpu_approx=False)
            proj_dist = compute_projection_distance(gen_step.cpu(), proj)

        if step > 1:
            gen_ret_panel = torch.diff(gen_mid[:, :step], dim=1).numpy()
            real_ret_panel = torch.diff(real_mid[:, :step], dim=1).numpy()
            gen_ret = gen_ret_panel.reshape(-1)
            real_ret = real_ret_panel.reshape(-1)
        else:
            gen_ret_panel = np.empty((gen.shape[0], 0), dtype=np.float32)
            real_ret_panel = np.empty((real.shape[0], 0), dtype=np.float32)
            gen_ret = np.array([], dtype=np.float32)
            real_ret = np.array([], dtype=np.float32)

        g_sp = gen_spread[:, idx].numpy()
        r_sp = real_spread[:, idx].numpy()
        g_mid = gen_mid[:, idx].numpy()
        r_mid = real_mid[:, idx].numpy()
        g_vol = gen[:, 1, :, idx].numpy().reshape(-1)
        r_vol = real[:, 1, :, idx].numpy().reshape(-1)
        g_imb = gen_imb[:, idx].numpy()
        r_imb = real_imb[:, idx].numpy()
        g_queue = gen_queue[:, idx].numpy()
        r_queue = real_queue[:, idx].numpy()
        acf_lags = [1, 5, 10]

        gen_spread_prefix = gen_spread[:, :step].numpy()
        real_spread_prefix = real_spread[:, :step].numpy()
        gen_depth_prefix = gen_depth[:, :step].numpy()
        real_depth_prefix = real_depth[:, :step].numpy()
        gen_imb_prefix = gen_imb[:, :step].numpy()
        real_imb_prefix = real_imb[:, :step].numpy()
        gen_queue_prefix = gen_queue[:, :step].numpy()
        real_queue_prefix = real_queue[:, :step].numpy()

        summary[f"step_{step}"] = {
            "violations_raw": viol,
            "violations_real": real_viol,
            "projection_distance": proj_dist,
            "spread_mean": float(g_sp.mean()),
            "spread_std": float(g_sp.std()),
            "spread_real_mean": float(r_sp.mean()),
            "spread_real_std": float(r_sp.std()),
            "mid_mean": float(g_mid.mean()),
            "mid_std": float(g_mid.std()),
            "mid_real_mean": float(r_mid.mean()),
            "mid_real_std": float(r_mid.std()),
            "mid_return_mean": float(gen_ret.mean()) if gen_ret.size else float("nan"),
            "mid_return_std": float(gen_ret.std()) if gen_ret.size else float("nan"),
            "mid_return_real_mean": float(real_ret.mean()) if real_ret.size else float("nan"),
            "mid_return_real_std": float(real_ret.std()) if real_ret.size else float("nan"),
            "volume_mean": float(g_vol.mean()),
            "volume_std": float(g_vol.std()),
            "volume_real_mean": float(r_vol.mean()),
            "volume_real_std": float(r_vol.std()),
            "order_imbalance_mean": float(g_imb.mean()),
            "order_imbalance_std": float(g_imb.std()),
            "order_imbalance_real_mean": float(r_imb.mean()),
            "order_imbalance_real_std": float(r_imb.std()),
            "queue_imbalance_mean": float(g_queue.mean()),
            "queue_imbalance_std": float(g_queue.std()),
            "queue_imbalance_real_mean": float(r_queue.mean()),
            "queue_imbalance_real_std": float(r_queue.std()),
            "price_corr": safe_corr(
                real[:, 0, :, :step].numpy(), gen[:, 0, :, :step].numpy(),
            ),
            "volume_corr": safe_corr(
                real[:, 1, :, :step].numpy(), gen[:, 1, :, :step].numpy(),
            ),
            "temporal_corr": temporal_corr(real, gen, step),
            "w1_spread": wasserstein_1d(g_sp, r_sp),
            "w1_mid_return": wasserstein_1d(gen_ret, real_ret) if gen_ret.size else float("nan"),
            "w1_volume": wasserstein_1d(g_vol, r_vol),
            "w1_order_imbalance": wasserstein_1d(g_imb, r_imb),
            "w1_queue_imbalance": wasserstein_1d(g_queue, r_queue),
            "mmd_mid_return": rbf_mmd_1d(gen_ret, real_ret) if gen_ret.size else float("nan"),
            "mmd_spread": rbf_mmd_1d(g_sp, r_sp),
            "mmd_volume": rbf_mmd_1d(g_vol, r_vol),
            "mmd_order_imbalance": rbf_mmd_1d(g_imb, r_imb),
            "mmd_queue_imbalance": rbf_mmd_1d(g_queue, r_queue),
            "acf_mid_return_generated": panel_acf(gen_ret_panel, acf_lags),
            "acf_mid_return_real": panel_acf(real_ret_panel, acf_lags),
            "acf_abs_mid_return_generated": panel_acf(np.abs(gen_ret_panel), acf_lags),
            "acf_abs_mid_return_real": panel_acf(np.abs(real_ret_panel), acf_lags),
            "acf_spread_generated": panel_acf(gen_spread_prefix, acf_lags),
            "acf_spread_real": panel_acf(real_spread_prefix, acf_lags),
            "acf_volume_generated": panel_acf(gen_depth_prefix, acf_lags),
            "acf_volume_real": panel_acf(real_depth_prefix, acf_lags),
            "acf_order_imbalance_generated": panel_acf(gen_imb_prefix, acf_lags),
            "acf_order_imbalance_real": panel_acf(real_imb_prefix, acf_lags),
            "acf_queue_imbalance_generated": panel_acf(gen_queue_prefix, acf_lags),
            "acf_queue_imbalance_real": panel_acf(real_queue_prefix, acf_lags),
        }
    return summary


def empirical_floor_summary(
    real: torch.Tensor,
    steps: list[int],
    min_spread: float,
    min_gap: float,
    price_unit: float,
) -> dict:
    return summarize_rollout(
        real, real, steps, min_spread=min_spread, min_gap=min_gap,
        compute_projection=False, price_unit=price_unit,
    )


def plot_drift_curves(gen: torch.Tensor, real: torch.Tensor, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(1, gen.shape[-1] + 1)
    gen_mid = ((gen[:, 0, 0, :] + gen[:, 0, 10, :]) / 2).numpy()
    real_mid = ((real[:, 0, 0, :] + real[:, 0, 10, :]) / 2).numpy()
    gen_spread = (gen[:, 0, 0, :] - gen[:, 0, 10, :]).numpy()
    real_spread = (real[:, 0, 0, :] - real[:, 0, 10, :]).numpy()
    gen_vol = gen[:, 1].mean(dim=1).numpy()
    real_vol = real[:, 1].mean(dim=1).numpy()

    gen_ret_std = [np.diff(gen_mid[:, :i + 1], axis=1).std() if i > 0 else np.nan for i in range(len(t))]
    real_ret_std = [np.diff(real_mid[:, :i + 1], axis=1).std() if i > 0 else np.nan for i in range(len(t))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], gen_spread.mean(axis=0), real_spread.mean(axis=0), "Spread mean"),
        (axes[0, 1], gen_vol.mean(axis=0), real_vol.mean(axis=0), "Volume mean"),
        (axes[1, 0], np.asarray(gen_ret_std), np.asarray(real_ret_std), "Prefix mid-return std"),
        (axes[1, 1], gen_mid.std(axis=0), real_mid.std(axis=0), "Mid std"),
    ]
    for ax, g, r, title in panels:
        ax.plot(t, r, label="real", linewidth=2)
        ax.plot(t, g, label="generated", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Rollout step")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "drift_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genlob-ckpt", required=True)
    ap.add_argument("--timegrad-ckpt", default=None)
    ap.add_argument("--tabddpm-ckpt", default=None)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--dataset", default="INTC_9day",
                    help="Snapshot dataset id, e.g. INTC_9day; nasdaq_INTC_9day is accepted as an alias")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--hist-len", type=int, default=50)
    ap.add_argument("--pred-len", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--n-histories", type=int, default=128)
    ap.add_argument("--n-batches", type=int, default=None)
    ap.add_argument("--ddim-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output-dir", default="results/snowball")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--representation",
        choices=["auto", "raw", "matched_lob", "legal_lob", "legal_lob_ba"],
        default="auto",
    )
    ap.add_argument("--price-reference", choices=["auto", "dataset", "history_last_mid"], default="auto")
    ap.add_argument("--projection-mode", choices=["none", "exact", "fast"], default="none")
    ap.add_argument(
        "--sampler-projection-mode",
        choices=["none", "exact", "fast"],
        default="none",
        help="Project the predicted clean raw state at every DDIM step",
    )
    ap.add_argument("--compute-projection-distance", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--min-spread", type=float, default=None)
    ap.add_argument("--min-gap", type=float, default=None)
    ap.add_argument("--action-mode", choices=["replay", "hold", "buy", "sell"], default="hold")
    ap.add_argument("--allow-diagnostic-counterfactual-actions", action="store_true")
    # Backward-compatible alias for old CLI.
    ap.add_argument("--action", choices=["hold", "buy", "sell"], default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducible regression runs; set negative to disable")
    args = ap.parse_args()
    set_seed(None if args.seed < 0 else args.seed)
    if args.action is not None and args.action_mode == "hold":
        args.action_mode = args.action
    if args.action_mode in {"buy", "sell"} and not args.allow_diagnostic_counterfactual_actions:
        raise ValueError(
            "buy/sell rollout modes use proxy labels and are not causal. Pass "
            "--allow-diagnostic-counterfactual-actions only for explicitly diagnostic runs."
        )
    args.chunk_size = args.chunk_size or args.pred_len

    device = select_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model first so representation semantics are known before rollout.
    print("Loading GenLOB...")
    genlob, sde, genlob_cfg, ckpt = load_genlob(args.genlob_ckpt, device, args.weights)
    representation = resolve_representation(args.representation, ckpt)
    representation_stats = representation_stats_to_device(ckpt, device)

    # Build rollout seeds and the empirical floor from the same starts.
    data_dir = args.data_dir or "data"
    stock, nday = parse_snapshot_dataset_id(args.dataset)
    raw = load_nasdaq_multiday(data_dir, stock, nday)
    ckpt_args = ckpt.get("args") or {}
    requested_reference = (
        ckpt_args.get("price_reference", "dataset")
        if args.price_reference == "auto"
        else args.price_reference
    )
    args.price_reference = resolve_price_reference(
        requested_reference,
        bool(raw.get("has_raw_states", False)),
    )
    args.min_spread, args.min_gap = resolve_lob_constraints(
        raw.get("normalization"),
        min_spread=args.min_spread if args.min_spread is not None else ckpt_args.get("min_spread"),
        min_gap=args.min_gap if args.min_gap is not None else ckpt_args.get("min_gap"),
    )
    transform = (
        make_lob_transform(
            representation,
            min_spread=args.min_spread,
            gap_eps=args.min_gap,
        )
        if representation != "raw"
        else None
    )
    price_unit = normalized_price_unit(raw.get("normalization")) or 1.0
    print(
        f"Checkpoint epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')} "
        f"representation={representation} weights={args.weights} "
        f"price_reference={args.price_reference} min_spread={args.min_spread:.8g} "
        f"min_gap={args.min_gap:.8g}"
    )
    if representation_stats is not None:
        print("Loaded representation stats")
    if args.sampler_projection_mode != "none" and representation != "raw":
        raise ValueError("--sampler-projection-mode is only valid for raw representation")
    seed_batches = make_seed_batches_nasdaq(
        raw,
        hist_len=args.hist_len,
        horizon=args.horizon,
        chunk_size=args.chunk_size,
        action_mode=args.action_mode,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        n_histories=args.n_histories if args.n_batches is None else None,
        price_reference=args.price_reference,
    )
    n_histories = sum(b["history"].shape[0] for b in seed_batches)
    print(
        f"Rolling out {n_histories} histories, horizon={args.horizon}, "
        f"pred_len={args.pred_len}, chunk_size={args.chunk_size}, "
        f"action_mode={args.action_mode}, projection_mode={args.projection_mode}"
    )

    gen_parts = []
    real_parts = []
    projection_diagnostic_parts = []
    rollout_started = time.perf_counter()
    for batch_idx, batch in enumerate(seed_batches, start=1):
        rollout, projection_diagnostics = rollout_genlob_chunked(
            genlob,
            sde,
            batch["history"],
            batch["action_chunks"],
            horizon=args.horizon,
            pred_len=args.pred_len,
            chunk_size=args.chunk_size,
            ddim_steps=args.ddim_steps,
            device=device,
            representation=representation,
            transform=transform,
            representation_stats=representation_stats,
            projection_mode=args.projection_mode,
            hist_len=args.hist_len,
            price_reference=args.price_reference,
            min_spread=args.min_spread,
            min_gap=args.min_gap,
            sampler_projection_mode=args.sampler_projection_mode,
            return_diagnostics=True,
        )
        gen_parts.append(rollout.cpu())
        real_parts.append(batch["real_future"].cpu())
        projection_diagnostic_parts.append(projection_diagnostics)
        print(f"  batch {batch_idx}/{len(seed_batches)} rollout done")
    rollout_seconds = time.perf_counter() - rollout_started

    gen_rollout = torch.cat(gen_parts, dim=0)
    real_rollout = torch.cat(real_parts, dim=0)
    print(f"Generated rollout shape={tuple(gen_rollout.shape)}")
    print(f"Real floor shape={tuple(real_rollout.shape)}")

    snapshot_steps = []
    for step in [10, 20, 50, args.horizon]:
        if step <= args.horizon and step not in snapshot_steps:
            snapshot_steps.append(step)
    summary = {
        "config": {
            "dataset": args.dataset,
            "data_dir": data_dir,
            "stock": stock,
            "nday": nday,
            "hist_len": args.hist_len,
            "pred_len": args.pred_len,
            "chunk_size": args.chunk_size,
            "horizon": args.horizon,
            "ddim_steps": args.ddim_steps,
            "batch_size": args.batch_size,
            "n_histories": n_histories,
            "action_mode": args.action_mode,
            "projection_mode": args.projection_mode,
            "sampler_projection_mode": args.sampler_projection_mode,
            "compute_projection_distance": args.compute_projection_distance,
            "representation": representation,
            "price_reference": args.price_reference,
            "min_spread": args.min_spread,
            "min_gap": args.min_gap,
            "price_unit_raw_per_normalized": price_unit,
            "weights": args.weights,
        },
        "checkpoint": {
            "path": args.genlob_ckpt,
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
        },
        "runtime": {
            "rollout_seconds": rollout_seconds,
            "trajectories_per_second": n_histories / max(rollout_seconds, 1e-12),
            "generated_frames_per_second": (
                n_histories * args.horizon / max(rollout_seconds, 1e-12)
            ),
        },
        "rollout_projection_stats": merge_rollout_projection_diagnostics(
            projection_diagnostic_parts
        ),
        "generated": summarize_rollout(
            gen_rollout,
            real_rollout,
            snapshot_steps,
            min_spread=args.min_spread,
            min_gap=args.min_gap,
            compute_projection=args.compute_projection_distance,
            price_unit=price_unit,
        ),
        "empirical_floor": empirical_floor_summary(
            real_rollout,
            snapshot_steps,
            min_spread=args.min_spread,
            min_gap=args.min_gap,
            price_unit=price_unit,
        ),
    }

    with open(out_dir / "snowball_stats.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)

    if not args.skip_plots:
        plot_drift_curves(gen_rollout, real_rollout, out_dir)

    print("\nSnapshot summary:")
    for step in snapshot_steps:
        m = summary["generated"][f"step_{step}"]
        v = m["violations_raw"]
        proj = m.get("projection_distance") or {}
        print(
            f"  step {step}: CBR={v['crossed_book_rate']:.6f} "
            f"ask={v['ask_ladder_violation']:.6f} "
            f"bid={v['bid_ladder_violation']:.6f} "
            f"negvol={v['negative_volume_rate']:.6f} "
            f"proj={proj.get('total_l2_distance', float('nan')):.3e} "
            f"spread={m['spread_mean']:.6g}/{m['spread_real_mean']:.6g} "
            f"vol={m['volume_mean']:.6g}/{m['volume_real_mean']:.6g} "
            f"ret_std={m['mid_return_std']:.6g}/{m['mid_return_real_std']:.6g} "
            f"W1spread={m['w1_spread']:.6g}"
        )
    print(f"\nAll outputs -> {out_dir}")
    print(
        f"Rollout runtime: {rollout_seconds:.2f}s "
        f"({n_histories / max(rollout_seconds, 1e-12):.2f} trajectories/s)"
    )


if __name__ == "__main__":
    main()

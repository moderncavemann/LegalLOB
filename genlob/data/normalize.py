"""
Unified LOB Normalization for GenLOB

Preserves microstructural validity (ask > bid, monotonic ladders).

Method (following DiffLOB, Wang & Ventre 2026):
  Price:  Δp = (price - mid_price) * scale_factor
          → ask levels become positive, bid levels become negative
          → spread and ladder structure fully preserved
  Volume: v_norm = sqrt(clip(v, 0, q99)) / c
          → handles heavy tails, preserves relative magnitudes

Input:  raw LOB snapshots [N, 2, 20] with absolute prices/volumes
Output: normalized [N, 2, 20] ready for diffusion model

The normalization is invertible: denormalize() recovers original scale.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json


@dataclass
class NormStats:
    """Statistics needed to denormalize."""
    price_scale: float = 100.0      # multiplier for mid-price relative differences
    volume_clip: float = 10000.0    # 99th percentile volume cap
    volume_scale: float = 15.0      # divisor after sqrt
    price_global_std: float = 1.0   # global std for final scaling (NEW)
    mid_prices: Optional[np.ndarray] = None  # per-timestep mid prices for denorm

    def save(self, path: str):
        d = {
            'price_scale': self.price_scale,
            'volume_clip': self.volume_clip,
            'volume_scale': self.volume_scale,
            'price_global_std': self.price_global_std,
        }
        with open(path, 'w') as f:
            json.dump(d, f)

    @staticmethod
    def load(path: str) -> 'NormStats':
        with open(path) as f:
            d = json.load(f)
        return NormStats(**d)


def compute_norm_stats(data: np.ndarray) -> NormStats:
    """
    Compute normalization statistics from raw LOB data.

    Two-stage price normalization:
    1. Mid-price relative: preserves LOB structure (ask>0, bid<0)
    2. Global std scaling: brings values into [-3, 3] for diffusion

    Args:
        data: [N, 2, 20] raw LOB snapshots

    Returns:
        NormStats with all computed statistics
    """
    prices = data[:, 0, :]   # [N, 20]
    volumes = data[:, 1, :]  # [N, 20]

    # Volume stats
    volume_clip = np.percentile(volumes[volumes > 0], 99)
    volume_scale = 15.0

    # Price stats: compute what the mid-relative prices would be
    ask_p1 = prices[:, 0]
    bid_p1 = prices[:, 10]
    mid = (ask_p1 + bid_p1) / 2
    price_scale = 100.0
    relative_prices = (prices - mid[:, None]) * price_scale

    # Global std of all relative prices → for final scaling
    price_global_std = float(relative_prices.std())

    return NormStats(
        price_scale=price_scale,
        volume_clip=volume_clip,
        volume_scale=volume_scale,
        price_global_std=price_global_std,
    )


def normalize_lob(data: np.ndarray, stats: NormStats = None) -> tuple:
    """
    Normalize raw LOB data using mid-price relative method.

    Price normalization:
        mid = (ask_p1 + bid_p1) / 2
        Δp_i = (p_i - mid) * price_scale
        → ask levels: positive values (spread/2, spread/2 + tick, ...)
        → bid levels: negative values (-spread/2, -spread/2 - tick, ...)

    Volume normalization:
        v_norm = sqrt(clip(v, 0, q99)) / volume_scale

    Args:
        data: [N, 2, 20] raw LOB
        stats: precomputed NormStats (if None, compute from data)

    Returns:
        (normalized_data [N, 2, 20], stats NormStats)
    """
    data = data.copy().astype(np.float64)
    N = data.shape[0]

    if stats is None:
        stats = compute_norm_stats(data)

    prices = data[:, 0, :]   # [N, 20]
    volumes = data[:, 1, :]  # [N, 20]

    # Mid-price = (ask_p1 + bid_p1) / 2
    ask_p1 = prices[:, 0]    # [N]
    bid_p1 = prices[:, 10]   # [N]
    mid_prices = (ask_p1 + bid_p1) / 2  # [N]

    # Stage 1: relative to mid-price, scaled by price_scale
    prices_relative = (prices - mid_prices[:, None]) * stats.price_scale

    # Stage 2: global std scaling → brings into ~[-3, 3] range
    prices_norm = prices_relative / stats.price_global_std

    # Volume: clip + sqrt + scale
    volumes_clipped = np.clip(volumes, 0, stats.volume_clip)
    volumes_norm = np.sqrt(volumes_clipped) / stats.volume_scale

    result = np.stack([prices_norm, volumes_norm], axis=1).astype(np.float32)

    # Store mid_prices for denormalization
    stats.mid_prices = mid_prices

    return result, stats


def denormalize_lob(data_norm: np.ndarray, stats: NormStats,
                    mid_prices: np.ndarray = None) -> np.ndarray:
    """
    Recover original LOB scale from normalized data.

    Args:
        data_norm: [N, 2, 20] or [B, 2, 20, T] normalized data
        stats: NormStats used for normalization
        mid_prices: [N] or [B, T] mid prices (required for price recovery)

    Returns:
        denormalized data in same shape
    """
    data = data_norm.copy().astype(np.float64)

    if data.ndim == 3:  # [N, 2, 20]
        prices_norm = data[:, 0, :]
        volumes_norm = data[:, 1, :]

        # Reverse global scaling, then reverse mid-price relative
        prices_relative = prices_norm * stats.price_global_std
        if mid_prices is not None:
            prices = prices_relative / stats.price_scale + mid_prices[:, None]
        else:
            prices = prices_relative / stats.price_scale

        volumes = (volumes_norm * stats.volume_scale) ** 2

        return np.stack([prices, volumes], axis=1).astype(np.float32)

    elif data.ndim == 4:  # [B, 2, 20, T]
        prices_norm = data[:, 0, :, :]
        volumes_norm = data[:, 1, :, :]

        prices_relative = prices_norm * stats.price_global_std
        if mid_prices is not None:
            prices = prices_relative / stats.price_scale + mid_prices[:, None, :]
        else:
            prices = prices_relative / stats.price_scale

        volumes = (volumes_norm * stats.volume_scale) ** 2

        return np.stack([prices, volumes], axis=1).astype(np.float32)


def normalize_lob_windowed(data: np.ndarray, stats: NormStats = None) -> tuple:
    """
    Normalize pre-windowed data with a per-window mid-price.

    Each window uses its first timestep's mid-price as reference.

    Args:
        data: [N_windows, T, 2, 20] pre-windowed LOB data

    Returns:
        (normalized [N, T, 2, 20], stats)
    """
    N, T = data.shape[0], data.shape[1]

    if stats is None:
        # Compute volume stats from all data
        all_volumes = data[:, :, 1, :].reshape(-1, 20)
        volume_clip = np.percentile(all_volumes[all_volumes > 0], 99)
        stats = NormStats(price_scale=100.0, volume_clip=volume_clip)

    result = np.zeros_like(data, dtype=np.float32)

    for i in range(N):
        window = data[i]  # [T, 2, 20]
        # Use first timestep's mid-price as window reference
        ask_p1_0 = window[0, 0, 0]
        bid_p1_0 = window[0, 0, 10]
        ref_mid = (ask_p1_0 + bid_p1_0) / 2

        prices = window[:, 0, :]  # [T, 20]
        volumes = window[:, 1, :]  # [T, 20]

        prices_norm = (prices - ref_mid) * stats.price_scale
        volumes_norm = np.sqrt(np.clip(volumes, 0, stats.volume_clip)) / stats.volume_scale

        result[i] = np.stack([prices_norm, volumes_norm], axis=1)

    return result, stats


def verify_normalization(data_norm: np.ndarray, label: str = ""):
    """
    Verify that normalized data preserves LOB structure.

    After mid-price relative normalization:
    - All ask prices should be > 0 (above mid)
    - All bid prices should be < 0 (below mid)
    - ask_p1 (smallest ask) should be closest to 0
    - bid_p1 (largest bid) should be closest to 0
    """
    if data_norm.ndim == 3:  # [N, 2, 20]
        prices = data_norm[:, 0, :]
    elif data_norm.ndim == 4:  # [B, 2, 20, T]
        prices = data_norm[:, 0, :, :].reshape(-1, 20)

    ask_prices = prices[:, :10]   # should be > 0
    bid_prices = prices[:, 10:]   # should be < 0

    ask_positive = np.mean(ask_prices > 0)
    bid_negative = np.mean(bid_prices < 0)
    spread = (ask_prices[:, 0] - bid_prices[:, 0])
    spread_positive = np.mean(spread > 0)

    print(f"{'[' + label + '] ' if label else ''}Normalization verification:")
    print(f"  Ask > 0: {ask_positive:.2%}")
    print(f"  Bid < 0: {bid_negative:.2%}")
    print(f"  Spread > 0: {spread_positive:.2%}")
    print(f"  Avg spread: {spread.mean():.4f}")
    print(f"  Ask L1 mean: {ask_prices[:, 0].mean():.4f}")
    print(f"  Bid L1 mean: {bid_prices[:, 0].mean():.4f}")

    return {
        'ask_positive_rate': float(ask_positive),
        'bid_negative_rate': float(bid_negative),
        'spread_positive_rate': float(spread_positive),
    }

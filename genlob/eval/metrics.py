"""Distributional and temporal metrics for L2 snapshot trajectories."""

from __future__ import annotations

import numpy as np
import torch


def json_safe(value):
    """Recursively replace non-finite scalars with JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def wasserstein_1d(a: np.ndarray, b: np.ndarray, quantiles: int = 4096) -> float:
    """Approximate W1 via matched empirical quantiles for unequal samples."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    n = min(max(len(a), len(b)), quantiles)
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def ks_2sample(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample empirical Kolmogorov-Smirnov distance without SciPy."""
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    points = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def histogram_distances(
    a: np.ndarray,
    b: np.ndarray,
    bins: int = 100,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Shared-support histogram TV, KL(real||generated), and JS distances."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return {"total_variation": float("nan"), "kl_real_generated": float("nan"), "js": float("nan")}
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if not hi > lo:
        return {"total_variation": 0.0, "kl_real_generated": 0.0, "js": 0.0}
    p = np.histogram(a, bins=bins, range=(lo, hi))[0].astype(np.float64) + eps
    q = np.histogram(b, bins=bins, range=(lo, hi))[0].astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    return {
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "kl_real_generated": float(np.sum(p * np.log(p / q))),
        "js": float(0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint))),
    }


def rbf_mmd2_biased(
    a: np.ndarray,
    b: np.ndarray,
    max_samples: int = 1024,
    seed: int = 0,
) -> float:
    """Biased non-negative RBF MMD^2 diagnostic with median bandwidth."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(a) > max_samples:
        a = a[rng.choice(len(a), max_samples, replace=False)]
    if len(b) > max_samples:
        b = b[rng.choice(len(b), max_samples, replace=False)]
    pooled = np.concatenate([a, b])
    probe = pooled if len(pooled) <= 2048 else pooled[rng.choice(len(pooled), 2048, replace=False)]
    distances = np.abs(probe[:, None] - probe[None, :])
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth = max(bandwidth, 1e-12)

    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-((x[:, None] - y[None, :]) ** 2) / (2.0 * bandwidth ** 2))

    value = kernel(a, a).mean() + kernel(b, b).mean() - 2.0 * kernel(a, b).mean()
    return float(max(value, 0.0))


def panel_acf(values: np.ndarray, lag: int) -> float:
    """Correlation at a lag without joining independent trajectories."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or lag <= 0 or values.shape[1] <= lag:
        return float("nan")
    left = values[:, :-lag].reshape(-1)
    right = values[:, lag:].reshape(-1)
    return safe_corr(left, right)


def lob_series(x: torch.Tensor, price_unit: float = 1.0) -> dict[str, np.ndarray]:
    """Extract core temporal series from [B, 2, 20, T] snapshots."""
    arr = x.detach().cpu().numpy().astype(np.float64, copy=False)
    prices = arr[:, 0]
    volumes = arr[:, 1]
    ask = prices[:, :10]
    bid = prices[:, 10:]
    ask_vol = volumes[:, :10]
    bid_vol = volumes[:, 10:]
    mid = (ask[:, 0] + bid[:, 0]) * 0.5 * price_unit
    spread = (ask[:, 0] - bid[:, 0]) * price_unit
    mid_return = np.diff(mid, axis=1)
    total_volume = volumes.sum(axis=1)
    denom = np.maximum(ask_vol.sum(axis=1) + bid_vol.sum(axis=1), 1e-12)
    depth_imbalance = (bid_vol.sum(axis=1) - ask_vol.sum(axis=1)) / denom
    queue_denom = np.maximum(bid_vol[:, 0] + ask_vol[:, 0], 1e-12)
    queue_imbalance = (bid_vol[:, 0] - ask_vol[:, 0]) / queue_denom
    return {
        "mid": mid,
        "mid_return": mid_return,
        "abs_mid_return": np.abs(mid_return),
        "spread": spread,
        "total_volume": total_volume,
        "depth_imbalance": depth_imbalance,
        "queue_imbalance": queue_imbalance,
    }


def stylized_facts(
    real: torch.Tensor,
    generated: torch.Tensor,
    price_unit: float = 1.0,
) -> dict[str, object]:
    """Compare marginal distributions and short-horizon temporal dependence."""
    real_s = lob_series(real, price_unit=price_unit)
    gen_s = lob_series(generated, price_unit=price_unit)
    names = ["mid_return", "spread", "total_volume", "depth_imbalance", "queue_imbalance"]
    result: dict[str, object] = {
        "price_unit_raw_per_normalized": float(price_unit),
        "distribution": {},
        "acf": {},
    }
    distribution = result["distribution"]
    assert isinstance(distribution, dict)
    for name in names:
        r = real_s[name].reshape(-1)
        g = gen_s[name].reshape(-1)
        distribution[name] = {
            "real_mean": float(np.mean(r)),
            "real_std": float(np.std(r)),
            "generated_mean": float(np.mean(g)),
            "generated_std": float(np.std(g)),
            "wasserstein_1": wasserstein_1d(r, g),
            "ks": ks_2sample(r, g),
            "mmd2_rbf_biased": rbf_mmd2_biased(r, g),
            **histogram_distances(r, g),
        }
    acf = result["acf"]
    assert isinstance(acf, dict)
    for name in ["mid_return", "abs_mid_return", "spread", "total_volume", "depth_imbalance"]:
        acf[name] = {}
        for lag in [1, 2, 5, 10, 20, 50]:
            acf[name][str(lag)] = {
                "real": panel_acf(real_s[name], lag),
                "generated": panel_acf(gen_s[name], lag),
            }
    real_volume_diff = np.diff(real.detach().cpu().numpy()[:, 1], axis=-1)
    gen_volume_diff = np.diff(generated.detach().cpu().numpy()[:, 1], axis=-1)

    def cross_level_corr(values: np.ndarray) -> np.ndarray:
        observations = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
        if len(observations) < 2:
            return np.full((values.shape[1], values.shape[1]), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.corrcoef(observations, rowvar=False)
        return np.nan_to_num(matrix, nan=0.0)

    real_corr = cross_level_corr(real_volume_diff)
    gen_corr = cross_level_corr(gen_volume_diff)
    result["volume_difference_cross_level_correlation"] = {
        "real": real_corr.tolist(),
        "generated": gen_corr.tolist(),
        "mean_absolute_error": float(np.mean(np.abs(real_corr - gen_corr))),
        "frobenius_error": float(np.linalg.norm(real_corr - gen_corr)),
    }
    return result


def horizon_fidelity(
    real: torch.Tensor,
    generated: torch.Tensor,
    persistence: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    """Pointwise errors by future step, with optional Persistence skill score."""
    error = generated - real
    result = {
        "mse": torch.mean(error.square(), dim=(0, 1, 2)).cpu().tolist(),
        "mae": torch.mean(error.abs(), dim=(0, 1, 2)).cpu().tolist(),
        "price_mse": torch.mean(error[:, 0].square(), dim=(0, 1)).cpu().tolist(),
        "volume_mse": torch.mean(error[:, 1].square(), dim=(0, 1)).cpu().tolist(),
    }
    if persistence is not None:
        reference_mse = torch.mean((persistence - real).square(), dim=(0, 1, 2))
        model_mse = torch.mean(error.square(), dim=(0, 1, 2))
        skill = 1.0 - model_mse / torch.clamp(reference_mse, min=1e-12)
        result["persistence_mse"] = reference_mse.cpu().tolist()
        result["persistence_skill_score"] = skill.cpu().tolist()
    return result


def discrete_lob_validity(
    x: torch.Tensor,
    normalization: dict | None,
    tolerance_ticks: float = 1e-3,
    tolerance_shares: float = 1e-3,
) -> dict[str, object]:
    """Translation-invariant tick-grid and integer-share diagnostics.

    The history-last-mid reference may carry a half-tick phase, so price-grid
    checks use spreads, adjacent gaps, and offsets from best bid rather than an
    absolute normalized price origin.
    """
    normalization = normalization or {}
    tick_size = normalization.get("tick_size_raw")
    model_unit = normalization.get("model_price_unit_raw")
    if not tick_size or not model_unit:
        return {"available": False, "reason": "tick_size_raw/model_price_unit_raw unavailable"}
    tick = float(tick_size) / float(model_unit)
    arr = x.detach().cpu().numpy().astype(np.float64, copy=False)
    prices = arr[:, 0]
    volumes = arr[:, 1]
    ask = prices[:, :10]
    bid = prices[:, 10:]
    spread = ask[:, 0] - bid[:, 0]
    ask_gaps = ask[:, 1:] - ask[:, :-1]
    bid_gaps = bid[:, :-1] - bid[:, 1:]
    gaps = np.concatenate([ask_gaps.reshape(-1), bid_gaps.reshape(-1)])

    def off_grid(values: np.ndarray) -> np.ndarray:
        ticks = np.asarray(values, dtype=np.float64) / tick
        return np.abs(ticks - np.rint(ticks)) > tolerance_ticks

    bid1 = bid[:, 0:1]
    relative_prices = prices - bid1
    negative_volume = volumes < 0
    volume_scale = normalization.get("volume_scale")
    volume_transform = str(normalization.get("volume_transform", ""))
    volume_metrics: dict[str, object]
    if volume_scale and "sqrt" in volume_transform:
        raw_volume = np.square(np.clip(volumes, 0.0, None) * float(volume_scale))
        noninteger = np.abs(raw_volume - np.rint(raw_volume)) > tolerance_shares
        volume_metrics = {
            "integer_share_check_available": True,
            "noninteger_volume_rate": float(np.mean(noninteger | negative_volume)),
        }
    else:
        volume_metrics = {
            "integer_share_check_available": False,
            "noninteger_volume_rate": None,
        }
    return {
        "available": True,
        "scope": "internal tick grid and integer shares; absolute grid phase inherited from the valid history anchor",
        "normalized_tick_size": tick,
        "off_tick_spread_rate": float(np.mean(off_grid(spread))),
        "off_tick_gap_rate": float(np.mean(off_grid(gaps))),
        "off_tick_internal_price_rate": float(np.mean(off_grid(relative_prices))),
        "sub_tick_spread_rate": float(np.mean(spread < tick * (1.0 - tolerance_ticks))),
        "sub_tick_gap_rate": float(np.mean(gaps < tick * (1.0 - tolerance_ticks))),
        **volume_metrics,
    }


def structural_validity_details(
    x: torch.Tensor,
    min_spread: float,
    min_gap: float,
    tolerance: float = 1e-7,
) -> dict[str, float]:
    """Snapshot-level invalidity and maximum constraint shortfalls."""
    arr = x.detach().cpu().numpy().astype(np.float64, copy=False)
    prices = arr[:, 0]
    volumes = arr[:, 1]
    ask = prices[:, :10]
    bid = prices[:, 10:]
    tol = max(tolerance, abs(min_spread) * 2e-5, abs(min_gap) * 2e-5)
    spread_shortfall = np.maximum(min_spread - tol - (ask[:, 0] - bid[:, 0]), 0.0)
    ask_shortfall = np.maximum(min_gap - tol - (ask[:, 1:] - ask[:, :-1]), 0.0)
    bid_shortfall = np.maximum(min_gap - tol - (bid[:, :-1] - bid[:, 1:]), 0.0)
    negative_volume = np.maximum(-tol - volumes, 0.0)
    invalid_snapshot = (
        (spread_shortfall > 0)
        | np.any(ask_shortfall > 0, axis=1)
        | np.any(bid_shortfall > 0, axis=1)
        | np.any(negative_volume > 0, axis=1)
    )
    invalid_trajectory = np.any(invalid_snapshot, axis=1)
    return {
        "any_invalid_snapshot_rate": float(np.mean(invalid_snapshot)),
        "any_invalid_trajectory_rate": float(np.mean(invalid_trajectory)),
        "max_spread_shortfall": float(spread_shortfall.max(initial=0.0)),
        "max_ask_gap_shortfall": float(ask_shortfall.max(initial=0.0)),
        "max_bid_gap_shortfall": float(bid_shortfall.max(initial=0.0)),
        "max_negative_volume_magnitude": float(negative_volume.max(initial=0.0)),
    }


def projection_repair_details(
    original: torch.Tensor,
    projected: torch.Tensor,
    normalization: dict | None,
    tolerance: float = 1e-7,
) -> dict[str, float | None]:
    """How often and how far a projection changes generated snapshots."""
    diff = (projected - original).detach().cpu().numpy().astype(np.float64, copy=False)
    price_abs = np.abs(diff[:, 0])
    volume_abs = np.abs(diff[:, 1])
    tick_size = (normalization or {}).get("tick_size_raw")
    model_unit = (normalization or {}).get("model_price_unit_raw")
    normalized_tick = (
        float(tick_size) / float(model_unit)
        if tick_size and model_unit
        else None
    )
    change_tolerance = max(
        tolerance,
        normalized_tick * 1e-4 if normalized_tick else tolerance,
    )
    price_changed = price_abs > change_tolerance
    volume_changed = volume_abs > change_tolerance
    snapshot_modified = np.any(price_changed | volume_changed, axis=1)
    mean_modified_price = (
        float(price_abs[price_changed].mean()) if np.any(price_changed) else 0.0
    )
    return {
        "any_snapshot_modified_rate": float(np.mean(snapshot_modified)),
        "change_tolerance": float(change_tolerance),
        "price_level_modified_rate": float(np.mean(price_changed)),
        "volume_level_modified_rate": float(np.mean(volume_changed)),
        "mean_abs_price_change_all_levels": float(np.mean(price_abs)),
        "mean_abs_price_change_modified_levels": mean_modified_price,
        "max_abs_price_change": float(price_abs.max(initial=0.0)),
        "mean_ticks_moved_all_levels": (
            float(np.mean(price_abs) / normalized_tick) if normalized_tick else None
        ),
        "mean_ticks_moved_modified_levels": (
            float(mean_modified_price / normalized_tick) if normalized_tick else None
        ),
    }


def per_sample_metrics(
    real: torch.Tensor,
    raw: torch.Tensor,
    projected: torch.Tensor,
    min_spread: float,
    min_gap: float,
    diagnostic_projection: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Return trajectory-level observations for time-block uncertainty."""
    real_np = real.detach().cpu().numpy()
    raw_np = raw.detach().cpu().numpy()
    proj_np = projected.detach().cpu().numpy()
    diagnostic_np = (
        diagnostic_projection.detach().cpu().numpy()
        if diagnostic_projection is not None
        else proj_np
    )
    raw_price = raw_np[:, 0]
    raw_volume = raw_np[:, 1]
    ask = raw_price[:, :10]
    bid = raw_price[:, 10:]
    spread = ask[:, 0] - bid[:, 0]
    ask_gaps = ask[:, 1:] - ask[:, :-1]
    bid_gaps = bid[:, :-1] - bid[:, 1:]
    projection_l2 = np.sqrt(np.sum((raw_np - diagnostic_np) ** 2, axis=(1, 2, 3)))
    tol = max(1e-7, abs(min_spread) * 2e-5, abs(min_gap) * 2e-5)
    invalid_snapshot = (
        (spread < min_spread - tol)
        | np.any(ask_gaps < min_gap - tol, axis=1)
        | np.any(bid_gaps < min_gap - tol, axis=1)
        | np.any(raw_volume < -tol, axis=1)
    )
    return {
        "mse_raw": np.mean((raw_np - real_np) ** 2, axis=(1, 2, 3)),
        "mse_projected": np.mean((proj_np - real_np) ** 2, axis=(1, 2, 3)),
        "mae_raw": np.mean(np.abs(raw_np - real_np), axis=(1, 2, 3)),
        "mae_projected": np.mean(np.abs(proj_np - real_np), axis=(1, 2, 3)),
        "crossed_book_rate": np.mean(spread < min_spread - tol, axis=1),
        "ask_ladder_violation": np.mean(ask_gaps < min_gap - tol, axis=(1, 2)),
        "bid_ladder_violation": np.mean(bid_gaps < min_gap - tol, axis=(1, 2)),
        "negative_volume_rate": np.mean(raw_volume < -tol, axis=(1, 2)),
        "projection_l2": projection_l2,
        "spread_mean_raw": np.mean(spread, axis=1),
        "any_invalid_snapshot_rate": np.mean(invalid_snapshot, axis=1),
        "any_invalid_trajectory": np.any(invalid_snapshot, axis=1).astype(np.float64),
    }

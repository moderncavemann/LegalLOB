"""
Physics-Constrained Manifold Projection for LOB data.

Defines the valid LOB manifold M and the L2 projection operator Π_M.

Valid LOB Manifold:
    M = { x ∈ R^{2×20×T} |
          a_1 ≤ a_2 ≤ ... ≤ a_10,       (monotonic ask ladder)
          b_1 ≥ b_2 ≥ ... ≥ b_10,       (monotonic bid ladder)
          a_1 - b_1 ≥ δ,                 (minimum spread / no crossed book)
          v ≥ 0                           (non-negative volumes)
    }

Projection Operator:
    Π_M(x) = argmin_{y ∈ M} ||x - y||_2^2

Implementation:
    - Prices: a single-chain PAVA over reverse(bid) and ask-min_spread.
      This jointly handles bid precedence, spread, and ask precedence in one
      monotone cone projection.
    - Volume: ReLU clamp (trivially optimal L2 projection onto R+)

The same operator is used either once after sampling (post-hoc repair) or at
every denoising step (in-loop projected diffusion), depending on the caller.

Data format: [B, 2, 20, T]
  Channel 0 (price):  rows 0-9 = ask prices L1-L10, rows 10-19 = bid prices L1-L10
  Channel 1 (volume): rows 0-9 = ask volumes L1-L10, rows 10-19 = bid volumes L1-L10
"""

import torch
import numpy as np


def _isotonic_regression_pava(y: torch.Tensor, increasing: bool = True) -> torch.Tensor:
    """
    Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.

    Solves: argmin_{z monotone} ||z - y||_2^2

    This is the exact L2 projection onto the monotone cone.
    Operates on the last dimension of y.

    Args:
        y: [..., K] tensor to project
        increasing: if True, enforce z_1 ≤ z_2 ≤ ... ≤ z_K
                    if False, enforce z_1 ≥ z_2 ≥ ... ≥ z_K

    Returns:
        z: [..., K] isotonic regression solution
    """
    # For decreasing, flip, solve increasing, flip back
    if not increasing:
        return torch.flip(_isotonic_regression_pava(torch.flip(y, [-1]), increasing=True), [-1])

    shape = y.shape
    K = shape[-1]
    # Flatten all leading dims
    y_flat = y.reshape(-1, K)
    N = y_flat.shape[0]

    # PAVA over independent short chains (K=20 for the current LOB layout).
    result = y_flat.clone()
    for i in range(N):
        vals = result[i].tolist()
        # PAVA: merge adjacent violators by averaging
        blocks = [[vals[0], 1]]  # [sum, count]
        for j in range(1, K):
            blocks.append([vals[j], 1])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                # Merge last two blocks
                blocks[-2][0] += blocks[-1][0]
                blocks[-2][1] += blocks[-1][1]
                blocks.pop()
        # Expand blocks back to values
        idx = 0
        for s, c in blocks:
            mean_val = s / c
            for _ in range(c):
                result[i, idx] = mean_val
                idx += 1

    return result.reshape(shape)


def _isotonic_regression_batch_gpu(y: torch.Tensor, increasing: bool = True,
                                    n_iters: int = 20) -> torch.Tensor:
    """
    GPU-friendly monotone projection via cumulative max/min.

    Guarantees strict monotonicity (zero violations) but is NOT the L2-optimal
    projection — it makes each element at least as large as its left neighbour
    (for increasing case), which is a valid projection onto the monotone cone
    but may not be the argmin. For physics-constrained sampling where zero
    violations is the hard requirement, this is sufficient.

    Args:
        y: [..., K] tensor
        increasing: if True, enforce y_1 ≤ y_2 ≤ ... ≤ y_K via cummax
                    if False, enforce y_1 ≥ y_2 ≥ ... ≥ y_K via cummin

    Returns:
        z: [..., K] strictly monotone along last axis
    """
    if increasing:
        return torch.cummax(y, dim=-1).values
    return torch.cummin(y, dim=-1).values


def _project_prices_single_chain(
    prices: torch.Tensor,
    min_spread: float,
    min_gap: float,
    use_fast_approx: bool,
) -> torch.Tensor:
    """Project prices onto one minimum-increment monotone chain:

    b10 <= ... <= b1 < a1 <= ... <= a10, with an explicit minimum
    adjacent-level gap and minimum spread. Subtracting cumulative required
    increments reduces the problem to ordinary isotonic regression, preserving
    exact L2 optimality.
    """
    ask_prices = prices[:, :10, :]
    bid_prices = prices[:, 10:, :]

    chain = torch.cat([torch.flip(bid_prices, dims=[1]), ask_prices], dim=1)
    increments = torch.tensor(
        [min_gap] * 9 + [min_spread] + [min_gap] * 9,
        dtype=chain.dtype,
        device=chain.device,
    )
    offsets = torch.cat(
        [torch.zeros(1, dtype=chain.dtype, device=chain.device), increments.cumsum(0)]
    )
    shifted = chain - offsets[None, :, None]
    q_perm = shifted.permute(0, 2, 1)
    if use_fast_approx:
        q_proj = _isotonic_regression_batch_gpu(q_perm, increasing=True)
    else:
        q_proj = _isotonic_regression_pava(q_perm, increasing=True)
    q_proj = q_proj.permute(0, 2, 1) + offsets[None, :, None]

    bid_proj = torch.flip(q_proj[:, :10, :], dims=[1])
    ask_proj = q_proj[:, 10:, :]
    return torch.cat([ask_proj, bid_proj], dim=1)


def project_to_valid_lob(x: torch.Tensor, min_spread: float = 1e-6,
                          min_gap: float = 0.0,
                          use_gpu_approx: bool = None) -> torch.Tensor:
    """
    Project generated LOB state onto valid microstructural manifold M.

    Π_M(x) = argmin_{y ∈ M} ||x - y||_2^2

    Decomposed into independent subproblems:
    1. Price validity: single-chain isotonic regression over
       reverse(bid) || (ask - min_spread)
    2. Volume non-negativity: ReLU clamp

    Args:
        x: [B, 2, 20, T] LOB tensor
        min_spread: minimum spread δ in the tensor's current scale. GenLOB
            trains on normalized prices, so this default is deliberately small;
            dataset-specific tick sizes should be passed explicitly for raw
            price-space tensors.
        min_gap: minimum adjacent price-level gap in the current scale.
        use_gpu_approx: if True, use the faster cumulative approximation;
            if False, use exact PAVA. If None, exact PAVA is used.

    Returns:
        x_valid: [B, 2, 20, T] with all constraints satisfied, minimum L2 distance from x
    """
    x = x.clone()

    if use_gpu_approx is None:
        use_gpu_approx = False

    # === Price constraints (channel 0) ===
    prices = x[:, 0, :, :]          # [B, 20, T]
    x[:, 0, :, :] = _project_prices_single_chain(
        prices,
        min_spread=min_spread,
        min_gap=min_gap,
        use_fast_approx=use_gpu_approx,
    )

    # === Volume constraints (channel 1) ===
    # L2 projection onto R+ is simply max(0, v)
    x[:, 1, :, :] = torch.clamp(x[:, 1, :, :], min=0)

    return x


def soft_lob_penalty(
    x: torch.Tensor,
    min_spread: float = 1e-6,
    min_gap: float = 0.0,
) -> torch.Tensor:
    """Differentiable LOB validity penalty for denoised x0 predictions."""
    prices = x[:, 0]
    volumes = x[:, 1]
    ask = prices[:, :10]
    bid = prices[:, 10:]

    crossed = torch.relu(min_spread + bid[:, 0:1, :] - ask[:, 0:1, :]).pow(2).mean()
    ask_ladder = torch.relu(min_gap + ask[:, :-1, :] - ask[:, 1:, :]).pow(2).mean()
    bid_ladder = torch.relu(min_gap + bid[:, 1:, :] - bid[:, :-1, :]).pow(2).mean()
    neg_vol = torch.relu(-volumes).pow(2).mean()
    return crossed + ask_ladder + bid_ladder + neg_vol


def action_direction_penalty(
    x_buy: torch.Tensor,
    x_sell: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Contrastive penalty encouraging Buy rollouts to have higher mid than Sell."""
    mid_buy = (x_buy[:, 0, 0, :] + x_buy[:, 0, 10, :]) / 2
    mid_sell = (x_sell[:, 0, 0, :] + x_sell[:, 0, 10, :]) / 2
    return torch.relu(margin - (mid_buy.mean(dim=1) - mid_sell.mean(dim=1))).mean()


def compute_projection_distance(x_original: torch.Tensor, x_projected: torch.Tensor) -> dict:
    """
    Compute L2 distance and per-component distances from projection.

    Useful for measuring how far generated samples are from the valid manifold.
    """
    diff = x_original - x_projected
    total_l2 = (diff ** 2).sum(dim=(1, 2, 3)).sqrt().mean().item()
    price_l2 = (diff[:, 0] ** 2).sum(dim=(1, 2)).sqrt().mean().item()
    volume_l2 = (diff[:, 1] ** 2).sum(dim=(1, 2)).sqrt().mean().item()

    return {
        "total_l2_distance": total_l2,
        "price_l2_distance": price_l2,
        "volume_l2_distance": volume_l2,
    }


def compute_violation_rate(
    x: torch.Tensor,
    min_spread: float = 1e-6,
    min_gap: float = 0.0,
    constraint_tol: float = 1e-7,
) -> dict:
    """
    Compute physics violation statistics for a batch of LOB states.

    Args:
        x: [B, 2, 20, T] LOB tensor

    Returns:
        dict with violation rates
    """
    prices = x[:, 0, :, :]
    volumes = x[:, 1, :, :]
    ask_prices = prices[:, :10, :]
    bid_prices = prices[:, 10:, :]

    spread = ask_prices[:, 0, :] - bid_prices[:, 0, :]
    tol = max(
        float(constraint_tol),
        abs(float(min_spread)) * 2e-5,
        abs(float(min_gap)) * 2e-5,
    )
    crossed = (spread < (min_spread - tol)).float().mean().item()
    ask_gaps = ask_prices[:, 1:, :] - ask_prices[:, :-1, :]
    bid_gaps = bid_prices[:, :-1, :] - bid_prices[:, 1:, :]
    ask_violations = (ask_gaps < (min_gap - tol)).float().mean().item()
    bid_violations = (bid_gaps < (min_gap - tol)).float().mean().item()
    neg_vol = (volumes < -tol).float().mean().item()
    avg_spread = spread.mean().item()

    return {
        "crossed_book_rate": crossed,
        "ask_ladder_violation": ask_violations,
        "bid_ladder_violation": bid_violations,
        "negative_volume_rate": neg_vol,
        "avg_spread": avg_spread,
        "min_spread": float(min_spread),
        "min_gap": float(min_gap),
        "constraint_tol": float(tol),
    }

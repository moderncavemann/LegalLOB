"""
Sampling methods for GenLOB diffusion model.

Implements:
- Ancestral sampling (DDPM-style, discrete)
- Predictor-Corrector (PC) sampling with Langevin corrector
- DDIM-style deterministic sampling

All samplers support action conditioning and optional physics projection.
"""

import torch
import numpy as np
import functools
from typing import Optional, Tuple

from .sde import VPSDE, VESDE, get_score_fn


# =============================================================================
# Ancestral Sampling (Discrete DDPM)
# =============================================================================

@torch.no_grad()
def ancestral_sample(
    model,
    sde: VPSDE,
    shape: tuple,
    history: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    device: str = "cpu",
    project_fn=None,
) -> torch.Tensor:
    """
    Ancestral sampling for VPSDE (equivalent to DDPM sampling).

    Args:
        model: GenLOBv2 model
        sde: VPSDE instance
        shape: output shape (B, 2, 20, T)
        history: conditioning history [B, 2, 20, T_h]
        action: agent action [B, action_dim] or None
        guidance_scale: classifier-free guidance weight (1.0 = no guidance)
        device: torch device
        project_fn: optional physics projection function

    Returns:
        generated samples [B, 2, 20, T]
    """
    model.eval()
    B = shape[0]

    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(sde.T, 0, sde.N + 1, device=device)

    for i in range(sde.N):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        vec_t = torch.ones(B, device=device) * t
        labels = (vec_t * (sde.N - 1)).long()

        # Model prediction (noise)
        noise_pred = model(x, labels, history, action)

        # Classifier-free guidance
        if guidance_scale != 1.0 and action is not None:
            noise_uncond = model(x, labels, history, None)
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # Ancestral sampling step
        timestep = labels
        beta = sde.discrete_betas.to(device)[timestep]
        alpha = sde.alphas.to(device)[timestep]
        alpha_bar = sde.alphas_cumprod.to(device)[timestep]

        # Predict x_0
        x_0_pred = (x - torch.sqrt(1 - alpha_bar)[:, None, None, None] * noise_pred) / \
                    torch.sqrt(alpha_bar)[:, None, None, None]
        x_0_pred = torch.clamp(x_0_pred, -10, 10)

        # === Projected Diffusion: project x_0_pred onto valid manifold ===
        # This guides the diffusion trajectory along the manifold boundary,
        # rather than post-hoc correcting the final sample.
        if project_fn is not None:
            x_0_pred = project_fn(x_0_pred)

        # Compute x_{t-1} using projected x_0_pred
        if i < sde.N - 1:
            alpha_bar_prev = sde.alphas_cumprod.to(device)[timestep - 1]
            x_mean = torch.sqrt(alpha_bar_prev)[:, None, None, None] * x_0_pred + \
                     torch.sqrt(1 - alpha_bar_prev)[:, None, None, None] * noise_pred
            noise = torch.randn_like(x)
            x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        else:
            x = x_0_pred

    return x


# =============================================================================
# Predictor-Corrector Sampling
# =============================================================================

@torch.no_grad()
def pc_sample(
    model,
    sde: VPSDE,
    shape: tuple,
    history: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    device: str = "cpu",
    project_fn=None,
    denoise: bool = True,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Predictor-Corrector sampling.

    Predictor: ancestral sampling step
    Corrector: Langevin dynamics step

    Args:
        n_corrector_steps: number of Langevin corrector steps per predictor step
        snr: signal-to-noise ratio for Langevin dynamics
    """
    model.eval()
    score_fn = get_score_fn(sde, model, continuous=False)
    B = shape[0]

    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

    for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(B, device=device) * t
        labels = (vec_t * (sde.N - 1)).long()

        # === Corrector (Langevin) ===
        for _ in range(n_corrector_steps):
            score = score_fn(x, vec_t, history, action)

            # Classifier-free guidance on score
            if guidance_scale != 1.0 and action is not None:
                score_uncond = score_fn(x, vec_t, history, None)
                score = score_uncond + guidance_scale * (score - score_uncond)

            noise = torch.randn_like(x)
            grad_norm = torch.norm(score.reshape(B, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(B, -1), dim=-1).mean()

            alpha = sde.alphas.to(device)[labels]
            step_size = (snr * noise_norm / (grad_norm + 1e-8)) ** 2 * 2 * alpha
            x = x + step_size[:, None, None, None] * score + \
                torch.sqrt(step_size * 2)[:, None, None, None] * noise

        # === Predictor (Ancestral) ===
        beta = sde.discrete_betas.to(device)[labels]
        score = score_fn(x, vec_t, history, action)

        if guidance_scale != 1.0 and action is not None:
            score_uncond = score_fn(x, vec_t, history, None)
            score = score_uncond + guidance_scale * (score - score_uncond)

        x_mean = (x + beta[:, None, None, None] * score) / \
                 torch.sqrt(1.0 - beta)[:, None, None, None]

        # === Projected Diffusion: project x_mean (predicted clean state) ===
        if project_fn is not None:
            x_mean = project_fn(x_mean)

        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise

    # Final denoising
    if denoise:
        vec_t = torch.ones(B, device=device) * eps
        score = score_fn(x, vec_t, history, action)
        beta_eps = sde.discrete_betas.to(device)[0]
        x = (x + beta_eps * score) / torch.sqrt(1.0 - beta_eps)

    return x


# =============================================================================
# DDIM Sampling (Deterministic)
# =============================================================================

@torch.no_grad()
def ddim_sample(
    model,
    sde: VPSDE,
    shape: tuple,
    history: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    ddim_steps: int = 50,
    eta: float = 0.0,
    device: str = "cpu",
    project_fn=None,
) -> torch.Tensor:
    """
    DDIM deterministic/stochastic sampling.

    Args:
        ddim_steps: number of DDIM steps (can be << sde.N for faster sampling)
        eta: DDIM stochasticity (0 = deterministic, 1 = DDPM equivalent)
    """
    model.eval()
    B = shape[0]

    # Sub-select timesteps for DDIM
    step_size = sde.N // ddim_steps
    time_indices = list(range(0, sde.N, step_size))[:ddim_steps]
    time_indices = list(reversed(time_indices))

    alphas_cumprod = sde.alphas_cumprod.to(device)

    x = sde.prior_sampling(shape).to(device)

    for i, t_idx in enumerate(time_indices):
        labels = torch.full((B,), t_idx, device=device, dtype=torch.long)

        noise_pred = model(x, labels, history, action)

        if guidance_scale != 1.0 and action is not None:
            noise_uncond = model(x, labels, history, None)
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        alpha_bar = alphas_cumprod[t_idx]

        # Predict x_0
        x_0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        x_0_pred = torch.clamp(x_0_pred, -10, 10)

        # === Projected Diffusion: project x_0_pred onto valid manifold ===
        if project_fn is not None:
            x_0_pred = project_fn(x_0_pred)

        if i < len(time_indices) - 1:
            t_prev_idx = time_indices[i + 1]
            alpha_bar_prev = alphas_cumprod[t_prev_idx]

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt + sigma * noise
        else:
            x = x_0_pred

    return x

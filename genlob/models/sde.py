"""
SDE library for GenLOB diffusion framework.

Adapted from DiffLOB (Wang & Ventre, 2026) which follows
Song et al. (2021) "Score-Based Generative Modeling through SDEs".

Implements:
- VPSDE (Variance Preserving) — default for GenLOB
- VESDE (Variance Exploding)
- DDPM-style discrete loss
- Continuous SDE loss with score matching
"""

import abc
import math
import torch
import torch.nn.functional as F
import numpy as np


class SDE(abc.ABC):
    """SDE abstract class."""

    def __init__(self, N):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G


class VPSDE(SDE):
    """Variance Preserving SDE (equivalent to DDPM in discrete form)."""

    def __init__(self, beta_min=0.1, beta_max=20, N=100):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class VESDE(SDE):
    """Variance Exploding SDE."""

    def __init__(self, sigma_min=0.01, sigma_max=50, N=100):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device)
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max


# =============================================================================
# Loss Functions
# =============================================================================

def get_score_fn(sde, model, continuous=False):
    """Wrap model output to produce score function."""

    if isinstance(sde, (VPSDE,)):
        def score_fn(x, t, cond, action):
            if continuous:
                labels = t * (sde.N - 1)
                score = model(x, labels, cond, action)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                labels = t * (sde.N - 1)
                score = model(x, labels, cond, action)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, t, cond, action):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()
            score = model(x, labels, cond, action)
            return score
    else:
        raise NotImplementedError

    return score_fn


def get_ddpm_loss_fn(vpsde, reduce_mean=True):
    """DDPM noise prediction loss (discrete)."""
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, cond, action):
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = (
            sqrt_alphas_cumprod[labels, None, None, None] * batch
            + sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        )
        score = model(perturbed_data, labels, cond, action)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        return torch.mean(losses)

    return loss_fn


def get_sde_loss_fn(sde, continuous=True, reduce_mean=True, likelihood_weighting=False, eps=1e-5):
    """Continuous SDE score matching loss."""
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, cond, action):
        score_fn = get_score_fn(sde, model, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t, cond, action)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        return torch.mean(losses)

    return loss_fn


def get_loss_fn(sde, continuous=False, reduce_mean=True, likelihood_weighting=False):
    """Get appropriate loss function based on SDE type and training mode."""
    if continuous:
        return get_sde_loss_fn(sde, continuous=True, reduce_mean=reduce_mean,
                               likelihood_weighting=likelihood_weighting)
    else:
        if isinstance(sde, VPSDE):
            return get_ddpm_loss_fn(sde, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} not supported.")

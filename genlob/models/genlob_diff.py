#!/usr/bin/env python3
"""
GenLOB-Diff: Legacy Action-Conditioned Diffusion Prototype for LOB
==================================================================
This legacy module supports future LOB-state generation conditioned on:
1. Historical market states (past H timesteps)
2. Diagnostic action/proxy vectors

Optimized for Apple Silicon M3 Ultra with MPS acceleration.

Architecture: 1D ResNet with FiLM conditioning (WaveNet-inspired)
- Faster than Transformer on M3 for time series
- FiLM (Feature-wise Linear Modulation) for condition injection

Current paper scope note:
Action conditioning in this repository is diagnostic unless the dataset provides
validated message-level aggressor-side labels. Do not use this legacy module's
name or comments as evidence for causal Buy/Sell market-impact claims.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenLOBConfig:
    """Configuration for GenLOB-Diff model."""

    # Data dimensions
    input_dim: int = 40          # LOB features (10 levels × 4)
    seq_len: int = 10            # Future prediction window (K)
    hist_len: int = 90           # History context window (H)

    # Model dimensions
    hidden_dim: int = 128        # Base hidden dimension
    time_dim: int = 64           # Time embedding dimension
    action_dim: int = 5          # Action space: [direction, size, aggressiveness, ...]

    # Architecture
    n_res_blocks: int = 4        # Number of residual blocks per stage
    dropout: float = 0.1

    # Diffusion
    timesteps: int = 1000        # Diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    batch_size: int = 512        # M3 Ultra can handle large batches
    learning_rate: float = 1e-4


# =============================================================================
# Core Components
# =============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for diffusion timestep t.
    Allows the model to know the current noise level.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Modulates features based on conditioning information.

    FiLM(x | c) = gamma(c) * x + beta(c)
    """
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) or (B, C)
        # cond: (B, cond_dim)
        style = self.mlp(cond)

        if x.dim() == 3:
            style = style.unsqueeze(-1)  # (B, 2C, 1)

        gamma, beta = style.chunk(2, dim=1)
        return x * (1 + gamma) + beta


class ResidualBlock1D(nn.Module):
    """
    Residual block with FiLM conditioning.
    Inspired by WaveNet gated activations.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        # Main convolution path
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        # Normalization
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        # FiLM conditioning
        self.film = FiLMLayer(out_channels, cond_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # cond: (B, cond_dim)

        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film(h, cond)  # Apply conditioning via FiLM
        h = F.silu(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)

        return h + residual


class HistoryEncoder(nn.Module):
    """
    Encodes historical LOB states into a fixed-size conditioning vector.
    Uses dilated convolutions to capture multi-scale temporal patterns.
    """
    def __init__(self, input_dim: int, hidden_dim: int, hist_len: int):
        super().__init__()

        self.encoder = nn.Sequential(
            # First layer: capture local patterns
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.MaxPool1d(2),

            # Second layer: medium-range patterns (dilated)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.MaxPool1d(2),

            # Third layer: long-range patterns
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),

            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # Final projection
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (B, input_dim, hist_len)
        return self.encoder(history)


class ActionEncoder(nn.Module):
    """
    Encodes agent actions into conditioning vectors.

    Action space design for LOB:
    - direction: [-1, 0, 1] (sell, hold, buy)
    - size: normalized order size
    - aggressiveness: how much to cross the spread
    - duration: expected holding period
    - urgency: execution urgency
    """
    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        # action: (B, action_dim)
        return self.encoder(action)


# =============================================================================
# Main Model
# =============================================================================

class GenLOBDiffusion(nn.Module):
    """
    GenLOB-Diff legacy diffusion prototype.

    Given:
    - history: Past H timesteps of LOB states
    - action: Diagnostic action/proxy vector at current time
    - x_noisy: Noisy future states (during training)
    - t: Diffusion timestep

    Predicts:
    - epsilon: The noise added to x_0 to get x_t
    """

    def __init__(self, config: GenLOBConfig):
        super().__init__()
        self.config = config

        # Dimension shortcuts
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        time_dim = config.time_dim

        # === Conditioning Encoders ===

        # 1. History encoder
        self.history_encoder = HistoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hist_len=config.hist_len
        )

        # 2. Action encoder
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_dim=hidden_dim
        )

        # 3. Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Combined condition dimension: history + action + time
        cond_dim = hidden_dim + hidden_dim + time_dim

        # === Diffusion Backbone (U-Net style with skip connections) ===

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, hidden_dim, cond_dim, dilation=1),
            ResidualBlock1D(hidden_dim, hidden_dim, cond_dim, dilation=2),
            ResidualBlock1D(hidden_dim, hidden_dim * 2, cond_dim, dilation=4),
            ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim, dilation=8),
        ])

        # Middle
        self.middle_block = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim)

        # Decoder (upsampling path with skip connections)
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim * 4, hidden_dim * 2, cond_dim, dilation=8),  # *4 due to skip
            ResidualBlock1D(hidden_dim * 4, hidden_dim, cond_dim, dilation=4),
            ResidualBlock1D(hidden_dim * 2, hidden_dim, cond_dim, dilation=2),
            ResidualBlock1D(hidden_dim * 2, hidden_dim, cond_dim, dilation=1),
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,      # (B, input_dim, seq_len)
        t: torch.Tensor,             # (B,)
        history: torch.Tensor,       # (B, input_dim, hist_len)
        action: Optional[torch.Tensor] = None  # (B, action_dim)
    ) -> torch.Tensor:
        """
        Forward pass: predict noise epsilon given noisy input and conditions.
        """
        batch_size = x_noisy.shape[0]

        # === Encode conditions ===

        # 1. History conditioning
        hist_emb = self.history_encoder(history)  # (B, hidden_dim)

        # 2. Action conditioning (use zero if not provided)
        if action is None:
            action = torch.zeros(batch_size, self.config.action_dim, device=x_noisy.device)
        action_emb = self.action_encoder(action)  # (B, hidden_dim)

        # 3. Time conditioning
        time_emb = self.time_mlp(t.float())  # (B, time_dim)

        # Concatenate all conditions
        cond = torch.cat([hist_emb, action_emb, time_emb], dim=1)  # (B, cond_dim)

        # === U-Net forward pass ===

        x = self.input_proj(x_noisy)

        # Encoder with skip connections
        skips = []
        for block in self.encoder_blocks:
            x = block(x, cond)
            skips.append(x)

        # Middle
        x = self.middle_block(x, cond)

        # Decoder with skip connections
        for block in self.decoder_blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = block(x, cond)

        # Output
        return self.output_proj(x)

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Diffusion Utilities
# =============================================================================

class DiffusionScheduler:
    """
    Manages the noise schedule for diffusion process.
    Supports linear and cosine schedules.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear"
    ):
        self.timesteps = timesteps

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            # Cosine schedule from "Improved DDPM"
            s = 0.008
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps) / timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "DiffusionScheduler":
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to x_0 to get x_t.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    @torch.no_grad()
    def sample_step(
        self,
        model: GenLOBDiffusion,
        x_t: torch.Tensor,
        t: int,
        history: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        eta: float = 0.0  # DDIM eta parameter (0 = deterministic)
    ) -> torch.Tensor:
        """
        Single sampling step: x_t -> x_{t-1}

        Supports both DDPM (eta=1) and DDIM (eta=0) sampling.
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x_t, t_tensor, history, action)

        # Get schedule values
        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]

        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        x_0_pred = torch.clamp(x_0_pred, -10, 10)  # Clip for stability

        # Compute variance
        sigma = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
        )

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * predicted_noise

        # Compute x_{t-1}
        if t > 0:
            noise = torch.randn_like(x_t) if eta > 0 else 0
            x_prev = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt + sigma * noise
        else:
            x_prev = x_0_pred

        return x_prev


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Test on MPS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create config and model
    config = GenLOBConfig()
    model = GenLOBDiffusion(config).to(device)

    print(f"Model parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size = 4
    x_noisy = torch.randn(batch_size, config.input_dim, config.seq_len).to(device)
    t = torch.randint(0, config.timesteps, (batch_size,)).to(device)
    history = torch.randn(batch_size, config.input_dim, config.hist_len).to(device)
    action = torch.randn(batch_size, config.action_dim).to(device)

    # Forward
    output = model(x_noisy, t, history, action)
    print(f"Input shape: {x_noisy.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")

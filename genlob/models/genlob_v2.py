"""
GenLOB v2: 2D Gated WaveNet Diffusion Model for LOB Snapshots

Architecture based on DiffLOB (Wang & Ventre, 2026) with key additions:
- LegalLOB-compatible snapshot generation path.
- Optional action FiLM conditioning. Current snapshot experiments treat action
  channels as diagnostic conditioning signals only; they are not causal
  Buy/Sell market-impact evidence without event-level aggressor-side labels.
- Projection hooks for diagnostic/safety ablations.

Input format: [B, 2, 20, T]
  Channel 0: prices  (ask_p1..ask_p10, bid_p1..bid_p10)
  Channel 1: volumes (ask_v1..ask_v10, bid_v1..bid_v10)

Conditioning:
  - History: past LOB states [B, 2, 20, T_h]
  - Action:  agent action vector [B, action_dim]
  - t:       diffusion timestep [B]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GenLOBv2Config:
    input_dim: int = 2           # channels (price + volume)
    output_dim: int = 2
    base_channels: int = 128
    emb_dim: int = 128
    num_layers: int = 16
    n_features: int = 20         # 10 ask + 10 bid levels
    num_heads: int = 8
    num_steps: int = 100         # diffusion steps (N)
    action_dim: int = 5
    dropout: float = 0.0
    cfg_drop_prob: float = 0.5   # classifier-free guidance dropout


# =============================================================================
# Core Components
# =============================================================================

class DiffusionEmbedding(nn.Module):
    """Sinusoidal + MLP embedding for diffusion timesteps."""

    def __init__(self, num_steps, embedding_dim=128):
        super().__init__()
        embed = self._build_embedding(num_steps, embedding_dim // 2)
        self.register_buffer("embedding", embed, persistent=False)
        self.proj1 = nn.Linear(embedding_dim, embedding_dim)
        self.proj2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        diffusion_step = diffusion_step.long()
        x = self.embedding[diffusion_step]
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        freqs = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0)
        table = steps * freqs.unsqueeze(0)
        return torch.cat([table.sin(), table.cos()], dim=1)


class TimeFiLM(nn.Module):
    """FiLM layer for diffusion timestep conditioning."""

    def __init__(self, emb_dim, n_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, n_channels * 2),
            nn.SiLU(),
            nn.Linear(n_channels * 2, n_channels * 2),
        )

    def forward(self, x, t_emb):
        gamma, beta = self.linear(t_emb).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class ActionFiLM(nn.Module):
    """
    FiLM layer for action/proxy conditioning.

    In the current snapshot protocol this supports diagnostic conditioning. It
    should not be interpreted as validated interactive simulation or causal
    action response unless the dataset supplies message-level aggressor-side
    labels and the action evaluation protocol is enabled.

    action_emb: [B, emb_dim] → modulates [B, C, F, T]
    """

    def __init__(self, emb_dim, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(emb_dim, n_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(n_channels, 2 * n_channels, kernel_size=1),
        )

    def forward(self, x, action_emb):
        cond = action_emb.unsqueeze(-1).unsqueeze(-1)
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return x * (1 + gamma) + beta


class CondFiLM(nn.Module):
    """FiLM layer for spatiotemporal condition maps (history encoding)."""

    def __init__(self, cond_channels, n_channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(cond_channels, n_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=1),
        )

    def forward(self, x, cond):
        gamma, beta = self.fusion(cond).chunk(2, dim=1)
        return x * (1 + gamma) + beta


class GatedWaveNetBlock(nn.Module):
    """
    2D Gated WaveNet residual block with triple FiLM conditioning.

    Conditioning order: TimeFiLM → ActionFiLM → CondFiLM
    """

    def __init__(self, channels, emb_dim, dilation, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-5)

        self.filter_conv = nn.Conv2d(
            channels, channels, kernel_size=(3, 3),
            padding=(dilation, dilation), dilation=(dilation, dilation),
        )
        self.gate_conv = nn.Conv2d(
            channels, channels, kernel_size=(3, 3),
            padding=(dilation, dilation), dilation=(dilation, dilation),
        )

        self.time_film = TimeFiLM(emb_dim, channels)
        self.action_film = ActionFiLM(emb_dim, channels)
        self.cond_film = CondFiLM(channels, channels)

        self.residual_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cond_map, action_emb, t_emb):
        """
        Args:
            x: [B, C, F, T]
            cond_map: [B, C, F, T] encoded history
            action_emb: [B, emb_dim]
            t_emb: [B, emb_dim]
        """
        x_input = x

        x = self.norm(x)
        x = self.time_film(x, t_emb)
        x = self.action_film(x, action_emb)
        x = self.cond_film(x, cond_map)

        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        h = filter_out * gate_out
        h = self.dropout(h)

        residual = self.residual_proj(h)
        skip = self.skip_proj(h)

        return x_input + residual, skip


class FeatureAttention(nn.Module):
    """Per-level multi-head self-attention along the time axis."""

    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, x):
        B, C, F, T = x.shape
        # Reshape: attend across time for each feature
        x_t = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        attn_out = attn_out.view(B, F, T, C).permute(0, 3, 1, 2).contiguous()
        return attn_out


class ConditionEncoder(nn.Module):
    """
    Encode history LOB states into a spatiotemporal condition map.

    Input: past LOB trajectory [B, 2, 20, T_h]
    Output: cond_map [B, C, 20, T_pred] (projected to prediction window size)
    """

    def __init__(self, base_channels=128):
        super().__init__()
        self.past_in = nn.Conv2d(2, base_channels, kernel_size=1)
        self.past_net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

    def forward(self, history, target_T):
        """
        Args:
            history: [B, 2, 20, T_h]
            target_T: int, temporal size to match (pred_len)
        Returns:
            cond_map: [B, C, 20, target_T]
        """
        h = self.past_net(self.past_in(history))  # [B, C, 20, T_h]
        # Adaptive pool to match target temporal dimension
        cond_map = F.adaptive_avg_pool2d(h, (h.shape[2], target_T))
        return cond_map


class ActionEncoder(nn.Module):
    """Encode agent actions into embedding vector."""

    def __init__(self, action_dim=5, emb_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

    def forward(self, action):
        return self.encoder(action)


# =============================================================================
# Main Model
# =============================================================================

class GenLOBv2(nn.Module):
    """
    GenLOB v2 diffusion model for LOB snapshots.

    Predicts noise ε given noisy LOB state, history, optional diagnostic action
    conditioning, and diffusion timestep.
    """

    def __init__(self, config: GenLOBv2Config = None):
        super().__init__()
        if config is None:
            config = GenLOBv2Config()
        self.config = config
        C = config.base_channels

        # Input projection: [B, 2, 20, T] → [B, C, 20, T]
        self.target_proj = nn.Conv2d(config.input_dim, C, kernel_size=1)

        # Level position embedding
        self.dim_emb = nn.Embedding(config.n_features, C)

        # Feature attention
        self.feat_attn = FeatureAttention(C, num_heads=config.num_heads)

        # Diffusion time embedding
        self.time_embed = DiffusionEmbedding(config.num_steps, config.emb_dim)

        # Condition encoders
        self.cond_encoder = ConditionEncoder(base_channels=C)
        self.action_encoder = ActionEncoder(config.action_dim, config.emb_dim)

        # Gated WaveNet blocks
        base_cycle = [1, 2, 4, 8, 16]
        dilations = [base_cycle[i % len(base_cycle)] for i in range(config.num_layers)]
        self.blocks = nn.ModuleList([
            GatedWaveNetBlock(C, config.emb_dim, d, config.dropout)
            for d in dilations
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(C, config.output_dim, kernel_size=1),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,        # [B, 2, 20, T]
        t: torch.Tensor,              # [B] diffusion timestep labels
        history: torch.Tensor,         # [B, 2, 20, T_h]
        action: torch.Tensor = None,   # [B, action_dim]
    ) -> torch.Tensor:
        """
        Forward pass: predict noise ε.

        Args:
            x_noisy: noisy future LOB state
            t: diffusion timestep (integer labels 0..N-1)
            history: past LOB trajectory for conditioning
            action: agent action vector (None → zero action)

        Returns:
            predicted noise [B, 2, 20, T]
        """
        B = x_noisy.shape[0]
        T_pred = x_noisy.shape[-1]

        # === Input projection + level embeddings ===
        x = self.target_proj(x_noisy.float())  # [B, C, 20, T]

        F_dim = self.config.n_features
        dims = torch.arange(F_dim, device=x.device)
        dims_emb = self.dim_emb(dims)  # [F, C]
        dims_emb = dims_emb.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # [1, C, F, 1]
        x = x + dims_emb

        # Feature attention
        x = self.feat_attn(x)

        # === Condition encoding ===
        t_emb = self.time_embed(t)  # [B, emb_dim]

        # History conditioning
        cond_map = self.cond_encoder(history, T_pred)  # [B, C, 20, T]

        # Action conditioning
        if action is None:
            action = torch.zeros(B, self.config.action_dim, device=x.device)
        action_emb = self.action_encoder(action)  # [B, emb_dim]

        # Classifier-free guidance dropout during training
        if self.training and self.config.cfg_drop_prob > 0:
            mask = torch.rand(B, device=x.device) < self.config.cfg_drop_prob
            if mask.any():
                cond_map[mask] = 0
                action_emb[mask] = 0

        # === WaveNet backbone ===
        skip_total = 0
        for block in self.blocks:
            x, skip = block(x, cond_map, action_emb, t_emb)
            skip_total = skip_total + skip

        # === Output ===
        return self.output_proj(skip_total)  # [B, 2, 20, T]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    config = GenLOBv2Config()
    model = GenLOBv2(config)
    print(f"GenLOB v2 parameters: {model.get_num_params():,}")

    B = 4
    x = torch.randn(B, 2, 20, 10)
    t = torch.randint(0, config.num_steps, (B,))
    hist = torch.randn(B, 2, 20, 50)
    action = torch.randn(B, 5)

    out = model(x, t, hist, action)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print("Forward pass OK!")

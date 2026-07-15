"""Conditional 1D temporal U-Net for action-sequence diffusion.

Standard Diffusion Policy architecture: Conv1d residual blocks with FiLM
conditioning on (diffusion-timestep embedding ++ flattened observation
history), down/up-sampling over the prediction-horizon axis.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out_ch), out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResBlock1d(nn.Module):
    """Two conv blocks with FiLM (scale, bias) conditioning after the first."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel_size)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel_size)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        scale, bias = self.cond_encoder(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + scale) + bias
        h = self.block2(h)
        return h + self.residual(x)


class Downsample1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1d(nn.Module):
    """epsilon_theta(x_tau, tau, obs_cond) over action sequences.

    Input/output: [B, horizon, action_dim]. ``horizon`` must be divisible
    by 2**(len(channels) - 1).
    """

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        channels: tuple[int, ...] = (128, 256, 512),
        diffusion_embedding_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.global_cond_dim = global_cond_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(diffusion_embedding_dim),
            nn.Linear(diffusion_embedding_dim, diffusion_embedding_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_embedding_dim * 4, diffusion_embedding_dim),
        )
        cond_dim = diffusion_embedding_dim + global_cond_dim

        dims = [action_dim, *channels]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList()
        for i, (ci, co) in enumerate(in_out):
            last = i == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1d(ci, co, cond_dim, kernel_size),
                        ConditionalResBlock1d(co, co, cond_dim, kernel_size),
                        Downsample1d(co) if not last else nn.Identity(),
                    ]
                )
            )

        mid_ch = channels[-1]
        self.mid1 = ConditionalResBlock1d(mid_ch, mid_ch, cond_dim, kernel_size)
        self.mid2 = ConditionalResBlock1d(mid_ch, mid_ch, cond_dim, kernel_size)

        # Mirrors Diffusion Policy's ConditionalUnet1D: one up block per
        # down block except the first; every up block upsamples, restoring
        # the original horizon (the first-level skip is unused, as in DP).
        self.ups = nn.ModuleList()
        for ci, co in reversed(in_out[1:]):
            self.ups.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1d(co * 2, ci, cond_dim, kernel_size),
                        ConditionalResBlock1d(ci, ci, cond_dim, kernel_size),
                        Upsample1d(ci),
                    ]
                )
            )

        self.final = nn.Sequential(
            Conv1dBlock(channels[0], channels[0], kernel_size),
            nn.Conv1d(channels[0], action_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """x: [B, H_p, A]; timesteps: [B] int; global_cond: [B, cond]."""
        horizon = x.shape[1]
        divisor = 2 ** (len(self.downs) - 1)
        if horizon % divisor != 0:
            raise ValueError(
                f"prediction horizon {horizon} must be divisible by {divisor} "
                f"for {len(self.downs)} U-Net levels"
            )
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0])
        cond = torch.cat([self.time_mlp(timesteps), global_cond], dim=-1)

        h = x.movedim(-1, -2)  # [B, A, H_p]
        skips = []
        for res1, res2, down in self.downs:
            h = res1(h, cond)
            h = res2(h, cond)
            skips.append(h)
            h = down(h)
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)
        for res1, res2, up in self.ups:
            h = torch.cat([h, skips.pop()], dim=1)
            h = res1(h, cond)
            h = res2(h, cond)
            h = up(h)
        h = self.final(h)
        return h.movedim(-1, -2)  # [B, H_p, A]

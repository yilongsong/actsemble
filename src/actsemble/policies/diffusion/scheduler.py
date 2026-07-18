"""DDPM/DDIM noise scheduler for action diffusion (epsilon prediction).

Self-contained implementation (no diffusers dependency) of:
* forward q(x_tau | x_0) noising for training;
* DDPM ancestral reverse steps;
* DDIM (eta=0, deterministic given the initial noise) reverse steps over a
  strided subset of timesteps.

Conventions: tau=0 is the least-noised timestep; ``num_train_steps`` is the
total diffusion depth used in training.
"""

from __future__ import annotations

import numpy as np
import torch


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """The squaredcos_cap_v2 schedule used by Diffusion Policy."""
    steps = np.arange(num_steps + 1, dtype=np.float64)
    f = np.cos(((steps / num_steps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.from_numpy(np.clip(betas, 0.0, 0.999)).float()


class DiffusionScheduler:
    def __init__(self, num_train_steps: int = 100, schedule: str = "cosine"):
        self.num_train_steps = int(num_train_steps)
        if schedule == "cosine":
            betas = cosine_beta_schedule(self.num_train_steps)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.num_train_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
        self.schedule = schedule
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    # -- training ----------------------------------------------------------
    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """q(x_tau | x_0): sqrt(a_bar) x0 + sqrt(1 - a_bar) eps."""
        a_bar = self.alphas_cumprod.to(x0.device)[timesteps].float()
        while a_bar.dim() < x0.dim():
            a_bar = a_bar.unsqueeze(-1)
        return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise

    def sample_timesteps(
        self, batch_size: int, device, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.randint(
            0, self.num_train_steps, (batch_size,), device=device, generator=generator
        )

    # -- inference ---------------------------------------------------------
    def inference_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Conventional DDIM/DDPM subsequence (diffusers 'leading' spacing):
        integer ``step_ratio = T // N`` gives timesteps ``0, r, 2r, ..., (N-1)r``
        taken in descending order (ending at 0). The exact timesteps are a
        deterministic function of (T, N) and are logged by the eval so the
        sampler spacing is fully recorded."""
        if not 1 <= num_inference_steps <= self.num_train_steps:
            raise ValueError(
                f"num_inference_steps must be in [1, {self.num_train_steps}]"
            )
        step_ratio = self.num_train_steps // num_inference_steps
        ts = (np.arange(num_inference_steps) * step_ratio).astype(np.int64)
        return torch.from_numpy(ts[::-1].copy())

    def _pred_x0(self, x: torch.Tensor, eps: torch.Tensor, tau: int) -> torch.Tensor:
        a_bar = self.alphas_cumprod[tau].to(x.device).float()
        return (x - (1 - a_bar).sqrt() * eps) / a_bar.sqrt().clamp(min=1e-8)

    def ddim_step(
        self, x: torch.Tensor, eps: torch.Tensor, tau: int, tau_prev: int
    ) -> torch.Tensor:
        """Deterministic DDIM step from tau to tau_prev (eta = 0)."""
        x0 = self._pred_x0(x, eps, tau).clamp(-1.0 * _CLIP, _CLIP)
        if tau_prev < 0:
            return x0
        a_bar_prev = self.alphas_cumprod[tau_prev].to(x.device).float()
        return a_bar_prev.sqrt() * x0 + (1 - a_bar_prev).sqrt() * eps

    def ddpm_step(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        tau: int,
        tau_prev: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Stochastic ancestral step from tau to tau_prev.

        Supports strided schedules by treating (tau -> tau_prev) as one
        DDIM step plus the DDPM posterior noise for that stride.
        """
        x0 = self._pred_x0(x, eps, tau).clamp(-1.0 * _CLIP, _CLIP)
        if tau_prev < 0:
            return x0
        a_bar = self.alphas_cumprod[tau].to(x.device).float()
        a_bar_prev = self.alphas_cumprod[tau_prev].to(x.device).float()
        # Posterior variance for the strided step (DDIM eta=1 form).
        sigma2 = ((1 - a_bar_prev) / (1 - a_bar)) * (1 - a_bar / a_bar_prev)
        sigma = sigma2.clamp(min=0.0).sqrt()
        dir_coeff = (1 - a_bar_prev - sigma2).clamp(min=0.0).sqrt()
        noise = torch.empty_like(x).normal_(generator=generator)
        return a_bar_prev.sqrt() * x0 + dir_coeff * eps + sigma * noise


# Predicted x0 is clipped to this multiple of the normalized data range
# during sampling, matching Diffusion Policy's clip_sample behavior.
_CLIP = 1.0

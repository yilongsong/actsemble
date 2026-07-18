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
    def __init__(self, num_train_steps: int = 100, schedule: str = "cosine",
                 timestep_spacing: str = "leading"):
        self.num_train_steps = int(num_train_steps)
        if schedule == "cosine":
            betas = cosine_beta_schedule(self.num_train_steps)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.num_train_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
        if timestep_spacing not in ("leading", "linspace"):
            raise ValueError(f"Unknown timestep_spacing: {timestep_spacing}")
        self.schedule = schedule
        # Which named rule maps num_inference_steps -> the timestep subsequence.
        # This is a *config* choice recorded in the checkpoint, never source-only
        # state, so a checkpoint fully determines its own inference behavior.
        self.timestep_spacing = str(timestep_spacing)
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
        """Descending timestep subsequence, ending at 0, under ``timestep_spacing``:

        * ``leading`` (default, conventional diffusers): integer ``step_ratio =
          T // N`` -> timesteps ``0, r, 2r, ..., (N-1)r`` reversed.
        * ``linspace``: the pre-fix rounded ``T / N`` spacing (kept only to
          reproduce runs recorded before the 'leading' switch).

        The rule is a config/checkpoint field; the exact timesteps are a
        deterministic function of (T, N, spacing) and are recorded in the eval
        output, so the sampler is never source-code-only state."""
        if not 1 <= num_inference_steps <= self.num_train_steps:
            raise ValueError(
                f"num_inference_steps must be in [1, {self.num_train_steps}]"
            )
        if self.timestep_spacing == "leading":
            step_ratio = self.num_train_steps // num_inference_steps
            ts = (np.arange(num_inference_steps) * step_ratio).astype(np.int64)
        else:  # linspace (legacy)
            stride = self.num_train_steps / num_inference_steps
            ts = (np.arange(num_inference_steps) * stride).round().astype(np.int64)
            ts = np.clip(ts, 0, self.num_train_steps - 1)
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
        """Canonical DDPM ancestral posterior step (diffusers ``DDPMScheduler``,
        ``fixed_small`` variance). The posterior mean is formed from the CLIPPED
        x0 and the current sample x — NOT the predicted epsilon — so that clipping
        x0 stays self-consistent. (The DDIM eta=1 form ``sqrt(a_bar_prev) x0 +
        sqrt(1-a_bar_prev-sigma^2) eps`` equals this only when x0 is unclipped;
        once x0 is clipped, eps no longer corresponds to it.) For a strided
        schedule the per-step alpha/beta are taken between the two cumulative
        products; at 100 inference steps this is the exact reference DDPM."""
        x0 = self._pred_x0(x, eps, tau).clamp(-1.0 * _CLIP, _CLIP)
        if tau_prev < 0:
            return x0
        a_bar = self.alphas_cumprod[tau].to(x.device).float()
        a_bar_prev = self.alphas_cumprod[tau_prev].to(x.device).float()
        alpha = a_bar / a_bar_prev  # per-step alpha over the (possibly strided) step
        beta = 1.0 - alpha
        x0_coeff = a_bar_prev.sqrt() * beta / (1 - a_bar)
        x_coeff = alpha.sqrt() * (1 - a_bar_prev) / (1 - a_bar)
        var = ((1 - a_bar_prev) / (1 - a_bar)) * beta  # fixed_small posterior variance
        noise = torch.empty_like(x).normal_(generator=generator)
        return x0_coeff * x0 + x_coeff * x + var.clamp(min=0.0).sqrt() * noise


# Predicted x0 is clipped to this multiple of the normalized data range
# during sampling, matching Diffusion Policy's clip_sample behavior.
_CLIP = 1.0

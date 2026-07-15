"""Reverse-diffusion sampling of action chunks.

A diffusion policy produces SAMPLES from the learned action distribution;
there is no directly available "mean action". Determinism contract: given
the same model, observation, sampler settings, and torch.Generator state,
``sample_chunks`` returns bitwise-identical candidates.
"""

from __future__ import annotations

import torch

from .scheduler import DiffusionScheduler


@torch.no_grad()
def sample_chunks(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    obs_cond: torch.Tensor,
    *,
    num_samples: int,
    prediction_horizon: int,
    action_dim: int,
    sampler: str = "ddim",
    num_inference_steps: int = 10,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample ``num_samples`` normalized action chunks for ONE observation.

    Args:
        obs_cond: [cond_dim] or [1, cond_dim] flattened normalized
            observation history.
        temperature: scales the initial noise x_T (sampling temperature);
            1.0 is the standard sampler.

    Returns:
        [num_samples, prediction_horizon, action_dim], normalized scale.
    """
    if obs_cond.dim() == 1:
        obs_cond = obs_cond.unsqueeze(0)
    if obs_cond.shape[0] != 1:
        raise ValueError("sample_chunks conditions on exactly one observation")
    cond = obs_cond.to(device).float().expand(num_samples, -1)

    x = torch.empty(
        (num_samples, prediction_horizon, action_dim), device=device
    ).normal_(generator=generator)
    x = x * float(temperature)

    timesteps = scheduler.inference_timesteps(num_inference_steps).tolist()
    for i, tau in enumerate(timesteps):
        tau_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        eps = model(x, torch.full((num_samples,), tau, device=device, dtype=torch.long), cond)
        if sampler == "ddim":
            x = scheduler.ddim_step(x, eps, tau, tau_prev)
        elif sampler == "ddpm":
            x = scheduler.ddpm_step(x, eps, tau, tau_prev, generator=generator)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
    return x

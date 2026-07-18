"""Euler ODE sampling for a flow-matching (rectified-flow) policy.

Integrates ``dz/dt = v_theta(z, t, cond)`` from ``t=0`` (Gaussian noise) to
``t=1`` (the action), where ``v_theta`` is the velocity field (the same
conditional U-Net used by the diffusion policy). Deterministic given the model,
observation, step count, and the ``torch.Generator`` that draws the initial
noise. Flow matching integrates well at FEW steps (k=1-8), which is what makes
it cheap enough to replan at high frequency (docs/latency_rule.md).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    obs_cond: torch.Tensor,
    *,
    num_samples: int,
    prediction_horizon: int,
    action_dim: int,
    num_steps: int = 10,
    time_scale: float = 1000.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample ``num_samples`` normalized action chunks for ONE observation.

    ``time_scale`` maps the continuous flow time ``t in [0,1]`` onto the U-Net's
    timestep-embedding frequency range (the net was designed for integer
    diffusion steps); training must use the same scale.

    Returns ``[num_samples, prediction_horizon, action_dim]`` (normalized).
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if obs_cond.dim() == 1:
        obs_cond = obs_cond.unsqueeze(0)
    if obs_cond.shape[0] != 1:
        raise ValueError("euler_sample conditions on exactly one observation")
    cond = obs_cond.to(device).float().expand(num_samples, -1)

    z = torch.empty(
        (num_samples, prediction_horizon, action_dim), device=device
    ).normal_(generator=generator)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i * dt
        t_batch = torch.full((num_samples,), t * time_scale, device=device)
        v = model(z, t_batch, cond)
        z = z + v * dt
    return z

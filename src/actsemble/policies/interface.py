"""Policy interface: any action-chunk-sampling policy usable by the systems."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .meta import PolicyMeta


@runtime_checkable
class ActionChunkPolicy(Protocol):
    """A frozen policy that samples future action chunks.

    Candidate sampling lives HERE, not inside any Actsemble component, so
    the exact same candidate tensor can be given to the multi-sample
    control and to the Actsemble reranker in paired diagnostics.
    """

    meta: PolicyMeta

    @property
    def dataset_hash(self) -> str: ...

    @property
    def checkpoint_hash(self) -> str: ...

    def reset(self) -> None: ...

    def new_generator(self, seed: int) -> torch.Generator: ...

    def sample_action_chunks(
        self,
        observation_history: np.ndarray,
        *,
        num_samples: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample candidate chunks conditioned on an observation history.

        Args:
            observation_history: [obs_horizon, state_dim] raw (unnormalized)
                states, oldest first.
            num_samples: number of independent candidates K.
            generator: torch.Generator controlling all sampling randomness.

        Returns:
            [num_samples, prediction_horizon, action_dim] raw (unnormalized)
            actions, clipped to the action bounds.
        """
        ...


def sampler_provenance(policy) -> dict:
    """Exact inference algorithm and discretization used by a loaded policy."""
    sched = getattr(policy, "scheduler", None)
    if sched is not None and hasattr(policy, "num_inference_steps"):
        n = int(policy.num_inference_steps)
        return {
            "family": "diffusion",
            "sampler": getattr(policy, "sampler", "ddim"),
            "num_train_steps": int(sched.num_train_steps),
            "num_inference_steps": n,
            "timestep_spacing": getattr(sched, "timestep_spacing", "leading"),
            "timesteps": sched.inference_timesteps(n).tolist(),
            "temperature": float(getattr(policy, "temperature", 1.0)),
            "prediction_type": "epsilon",
            "beta_schedule": "squared_cosine_cap_v2",
        }
    if hasattr(policy, "num_inference_steps") and hasattr(policy, "time_scale"):
        return {
            "family": "flow",
            "sampler": "euler",
            "num_steps": int(policy.num_inference_steps),
            "time_scale": float(policy.time_scale),
        }
    return {"family": "deterministic", "latent": "zero"}

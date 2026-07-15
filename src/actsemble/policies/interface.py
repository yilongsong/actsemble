"""Policy interface: any action-chunk-sampling policy usable by the systems."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class ActionChunkPolicy(Protocol):
    """A frozen policy that samples future action chunks.

    Candidate sampling lives HERE, not inside any Actsemble component, so
    the exact same candidate tensor can be given to the multi-sample
    control and to the Actsemble reranker in paired diagnostics.
    """

    @property
    def dataset_hash(self) -> str: ...

    @property
    def checkpoint_hash(self) -> str: ...

    def reset(self) -> None: ...

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

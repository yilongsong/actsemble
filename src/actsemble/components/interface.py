"""Learned-component interfaces.

A component is any model trained on the exact same frozen dataset as the
policy, assembled with the unchanged policy at runtime. Components never
see reward, success labels, failed rollouts, or the simulator during
training.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LearnedComponent(Protocol):
    @property
    def dataset_hash(self) -> str: ...

    @property
    def checkpoint_hash(self) -> str: ...

    def reset(self) -> None: ...


@runtime_checkable
class ActionChunkCompatibilityScorer(LearnedComponent, Protocol):
    """Scores candidate action chunks for compatibility with the dataset.

    A larger score means the chunk is more compatible with successful
    behavior represented in the training data. This is a same-data support
    estimator — NOT a reward model, success oracle, or value function.
    High offline compatibility accuracy does not imply improved closed-loop
    task success; closed-loop evaluation is the decisive test.
    """

    def score(
        self,
        observation_history: torch.Tensor,  # [obs_horizon, feat] raw values
        candidate_chunks: torch.Tensor,  # [K, prediction_horizon, action_dim] raw
    ) -> torch.Tensor:  # [K] scores (higher = more compatible)
        ...

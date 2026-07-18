"""Converts raw ManiSkill observations to the StateObservation contract."""

from __future__ import annotations

import numpy as np
import torch

from ..types import StateObservation


def to_numpy_state(obs) -> np.ndarray:
    """Flatten a ManiSkill obs (torch [1, D] / np [1, D] / [D]) to np [D]."""
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy()
    else:
        arr = np.asarray(obs)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected a single-env observation, got shape {arr.shape}"
            )
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected flat state observation, got shape {arr.shape}")
    return arr


class ObservationAdapter:
    """Builds StateObservation frames, tracking the previously executed action."""

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)
        self.reset()

    def reset(self) -> None:
        self._previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self._step_index = 0

    def observe(self, raw_obs) -> StateObservation:
        return StateObservation(
            state=to_numpy_state(raw_obs),
            previous_action=self._previous_action.copy(),
            step_index=self._step_index,
        )

    def after_step(self, executed_action: np.ndarray) -> None:
        self._previous_action = np.asarray(executed_action, dtype=np.float32).copy()
        self._step_index += 1

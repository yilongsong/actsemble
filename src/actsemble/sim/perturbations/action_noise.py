"""Bounded additive noise on commanded actions."""

from __future__ import annotations

import numpy as np

from ...seed import derive_seed
from ...types import RobotAction
from .base import NoOpPerturbation


class ActionNoisePerturbation(NoOpPerturbation):
    name = "action_noise"

    def __init__(
        self, *, scale: float = 0.1, distribution: str = "gaussian", seed: int = 0
    ):
        if distribution not in ("gaussian", "uniform"):
            raise ValueError(f"Unknown distribution {distribution}")
        self.scale = float(scale)
        self.distribution = distribution
        self.seed = int(seed)
        self._rng = np.random.default_rng(0)

    def reset(self, *, episode_seed: int) -> None:
        self._rng = np.random.default_rng(
            derive_seed(self.seed, self.name, episode_seed)
        )

    def modify_action(self, action: RobotAction, step_index: int) -> RobotAction:
        value = np.asarray(action.value, dtype=np.float32)
        if self.distribution == "gaussian":
            noise = self._rng.normal(0.0, self.scale, size=value.shape)
            noise = np.clip(noise, -3 * self.scale, 3 * self.scale)  # bounded
        else:
            noise = self._rng.uniform(-self.scale, self.scale, size=value.shape)
        return RobotAction(value=(value + noise.astype(np.float32)))

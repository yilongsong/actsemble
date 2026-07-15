"""Small noise on selected state dimensions.

A diagnostic perturbation for state-based Phase 0 — NOT a claim of
realistic sensing.
"""

from __future__ import annotations

import numpy as np

from ...seed import derive_seed
from ...types import StateObservation
from .base import NoOpPerturbation


class ObservationNoisePerturbation(NoOpPerturbation):
    name = "observation_noise"

    def __init__(self, *, sigma: float = 0.005, dims: list[int] | None = None, seed: int = 0):
        self.sigma = float(sigma)
        self.dims = dims  # None = all state dimensions
        self.seed = int(seed)
        self._rng = np.random.default_rng(0)

    def reset(self, *, episode_seed: int) -> None:
        self._rng = np.random.default_rng(derive_seed(self.seed, self.name, episode_seed))

    def modify_observation(self, observation: StateObservation, step_index: int) -> StateObservation:
        state = observation.state.copy()
        idx = np.arange(state.shape[0]) if self.dims is None else np.asarray(self.dims)
        state[idx] = state[idx] + self._rng.normal(0.0, self.sigma, size=idx.shape[0]).astype(
            np.float32
        )
        return StateObservation(
            state=state,
            previous_action=observation.previous_action,
            step_index=observation.step_index,
        )

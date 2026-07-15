"""Core data contracts shared across the Actsemble codebase.

Phase 0 note: `StateObservation.state` is a flat low-dimensional simulator
state vector that may contain privileged fields (object pose, goal pose,
...). Positive state-based results establish the system-level mechanism,
not real-world applicability. Vision-based validation is a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateObservation:
    """A single low-dimensional observation at one control step.

    Attributes:
        state: Flat float32 state vector (privileged sim state allowed in
            Phase 0). Must never contain reward, success, failure, dense
            task progress, future state, or termination reason.
        previous_action: The action executed at the previous control step
            (zeros at episode start).
        step_index: 0-based control step index within the episode.
    """

    state: np.ndarray
    previous_action: np.ndarray
    step_index: int


@dataclass
class RobotAction:
    """A single continuous robot action in the environment's action space."""

    value: np.ndarray


@dataclass
class EpisodeRecord:
    """One successful demonstration episode, aligned as (s_t, a_t, s_{t+1}).

    All arrays have leading dimension T (number of transitions). ``state[t]``
    is the observation *before* ``action[t]`` is applied and ``next_state[t]``
    is the observation after. The final state s_T has no associated action;
    it is stored only as ``next_state[T-1]``.

    ``previous_action[t] == action[t-1]`` with ``previous_action[0] == 0``.
    """

    episode_id: str
    state: np.ndarray  # [T, state_dim] float32
    previous_action: np.ndarray  # [T, action_dim] float32
    action: np.ndarray  # [T, action_dim] float32
    next_state: np.ndarray  # [T, state_dim] float32
    step_index: np.ndarray  # [T] int64

    def __len__(self) -> int:
        return int(self.state.shape[0])


@dataclass
class RolloutResult:
    """Outcome of one closed-loop episode."""

    episode_seed: int
    num_steps: int
    success_once: bool
    success_at_end: bool
    timed_out: bool
    exception: str | None = None
    diagnostics: dict = field(default_factory=dict)

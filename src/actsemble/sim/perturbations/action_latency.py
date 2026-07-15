"""Action latency: commands take effect ``delay_steps`` control steps late.

The first ``delay_steps`` executed actions are zeros (a zero delta holds
the arm still under delta-pose control).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ...types import RobotAction
from .base import NoOpPerturbation


class ActionLatencyPerturbation(NoOpPerturbation):
    name = "action_latency"

    def __init__(self, *, delay_steps: int = 1):
        if delay_steps < 1:
            raise ValueError("delay_steps must be >= 1")
        self.delay_steps = int(delay_steps)
        self._queue: deque = deque()

    def reset(self, *, episode_seed: int) -> None:
        self._queue = deque()

    def modify_action(self, action: RobotAction, step_index: int) -> RobotAction:
        self._queue.append(np.asarray(action.value, dtype=np.float32))
        if len(self._queue) <= self.delay_steps:
            return RobotAction(value=np.zeros_like(self._queue[0]))
        return RobotAction(value=self._queue.popleft())

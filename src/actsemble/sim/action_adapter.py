"""Converts RobotAction to the environment's action space, with clip stats."""

from __future__ import annotations

import numpy as np

from ..types import RobotAction


class ActionAdapter:
    def __init__(self, action_low: np.ndarray, action_high: np.ndarray):
        self.low = np.asarray(action_low, dtype=np.float32)
        self.high = np.asarray(action_high, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.total_actions = 0
        self.clipped_actions = 0
        self.max_clip_magnitude = 0.0

    def adapt(self, action: RobotAction) -> np.ndarray:
        value = np.asarray(action.value, dtype=np.float32).reshape(-1)
        if value.shape != self.low.shape:
            raise ValueError(f"Action shape {value.shape} != expected {self.low.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite action: {value}")
        clipped = np.clip(value, self.low, self.high)
        self.total_actions += 1
        overflow = float(np.max(np.abs(value - clipped)))
        if overflow > 0:
            self.clipped_actions += 1
            self.max_clip_magnitude = max(self.max_clip_magnitude, overflow)
        return clipped

    @property
    def clip_rate(self) -> float:
        return self.clipped_actions / self.total_actions if self.total_actions else 0.0

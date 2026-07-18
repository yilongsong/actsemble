"""Perturbation protocol and composition.

Perturbations model execution/dynamics disturbances (Phase 0 focus), not
visual disturbances. They are seeded per episode; the runtime SYSTEM never
observes that a perturbation exists — it only sees the resulting
observations and outcomes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...types import RobotAction, StateObservation


@runtime_checkable
class Perturbation(Protocol):
    name: str

    def reset(self, *, episode_seed: int) -> None: ...

    def modify_observation(
        self, observation: StateObservation, step_index: int
    ) -> StateObservation: ...

    def modify_action(self, action: RobotAction, step_index: int) -> RobotAction: ...

    def before_step(self, env: object, step_index: int) -> None: ...

    def after_step(self, env: object, step_index: int) -> None: ...


class NoOpPerturbation:
    """Base class implementing the identity for every hook."""

    name = "noop"

    def reset(self, *, episode_seed: int) -> None:
        pass

    def modify_observation(
        self, observation: StateObservation, step_index: int
    ) -> StateObservation:
        return observation

    def modify_action(self, action: RobotAction, step_index: int) -> RobotAction:
        return action

    def before_step(self, env: object, step_index: int) -> None:
        pass

    def after_step(self, env: object, step_index: int) -> None:
        pass


def build_perturbations(specs: list[dict]) -> list:
    """Instantiate perturbations from evaluation-config dicts."""
    from .action_latency import ActionLatencyPerturbation
    from .action_noise import ActionNoisePerturbation
    from .object_nudge import ObjectNudgePerturbation
    from .observation_noise import ObservationNoisePerturbation

    registry = {
        "action_noise": ActionNoisePerturbation,
        "action_latency": ActionLatencyPerturbation,
        "observation_noise": ObservationNoisePerturbation,
        "object_nudge": ObjectNudgePerturbation,
    }
    out = []
    for spec in specs or []:
        spec = dict(spec)
        kind = spec.pop("type")
        if kind not in registry:
            raise ValueError(f"Unknown perturbation type: {kind!r}")
        out.append(registry[kind](**spec))
    return out

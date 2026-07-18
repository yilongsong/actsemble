"""Policy provenance + contract metadata, shared by every policy family.

``PolicyMeta`` is policy-architecture-agnostic (diffusion, ACT, …): it records
the dataset/split/normalization identity and the task/controller/backend +
horizon contract that the fairness safeguards check. It is stored in every
policy checkpoint and consumed by the systems/evaluation layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyMeta:
    """Provenance and contract metadata stored in every policy checkpoint."""

    dataset_hash: str
    split_hash: str
    normalization: dict
    task_id: str
    controller: str
    simulation_backend: str
    state_dim: int
    action_dim: int
    action_low: list[float]
    action_high: list[float]
    obs_horizon: int
    prediction_horizon: int
    action_horizon: int
    include_previous_action: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyMeta":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

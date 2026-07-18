"""Actsemble frozen-dataset schema (HDF5).

Layout::

    metadata/                     (group; fields stored as attributes)
        project_name              "Actsemble"
        schema_version            "0.1"
        simulator                 e.g. "ManiSkill3"
        simulator_version
        task_id                   e.g. "PushT-v1"
        robot
        observation_mode          e.g. "state"
        state_dimension
        state_layout              JSON string (field name -> slice) when known
        controller                control mode string
        action_dimension
        action_definition         JSON string (semantics, frame, units, bounds)
        control_frequency
        simulation_backend        e.g. "physx_cpu"
        source_dataset            provenance of raw demonstrations
        generation_or_replay_seed
        dataset_hash              content hash over all episode arrays + key metadata

    episodes/<episode_id>/
        state            [T, state_dim] float32   s_t
        previous_action  [T, action_dim] float32  a_{t-1} (zeros at t=0)
        action           [T, action_dim] float32  a_t
        next_state       [T, state_dim] float32   s_{t+1}
        step_index       [T] int64

Alignment is exactly (s_t, a_t, s_{t+1}): ``state[t]`` is observed, then
``action[t]`` is executed, producing ``next_state[t]``. The terminal state
s_T has no action and appears only as ``next_state[T-1]``.

The dataset contains ONLY successful episodes. Reward, per-step success,
and termination reasons are never stored in episode arrays; success-only
filtering provenance lives in a separate private sidecar JSON written by
the preparation script (see ``writer.write_private_provenance``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "0.1"
PROJECT_NAME = "Actsemble"

EPISODE_ARRAY_KEYS = ("state", "previous_action", "action", "next_state", "step_index")

REQUIRED_METADATA_KEYS = (
    "project_name",
    "schema_version",
    "simulator",
    "simulator_version",
    "task_id",
    "robot",
    "observation_mode",
    "state_dimension",
    "state_layout",
    "controller",
    "action_dimension",
    "action_definition",
    "control_frequency",
    "simulation_backend",
    "source_dataset",
    "generation_or_replay_seed",
    "dataset_hash",
)

# Keys that must NEVER appear as episode arrays (leakage guards).
FORBIDDEN_EPISODE_KEYS = (
    "reward",
    "success",
    "fail",
    "failure",
    "done",
    "terminated",
    "truncated",
    "progress",
    "return",
)


@dataclass
class DatasetMetadata:
    """Typed view of the metadata group."""

    project_name: str = PROJECT_NAME
    schema_version: str = SCHEMA_VERSION
    simulator: str = ""
    simulator_version: str = ""
    task_id: str = ""
    robot: str = ""
    observation_mode: str = ""
    state_dimension: int = 0
    state_layout: str = "{}"  # JSON: field -> [start, end) when known
    controller: str = ""
    action_dimension: int = 0
    action_definition: str = "{}"  # JSON: semantics/frame/units/bounds/scaling/clipping
    control_frequency: float = 0.0
    simulation_backend: str = ""
    source_dataset: str = ""
    generation_or_replay_seed: int = 0
    dataset_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in REQUIRED_METADATA_KEYS}
        d.update(self.extra)
        return d

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]) -> "DatasetMetadata":
        known = {k: attrs[k] for k in REQUIRED_METADATA_KEYS if k in attrs}
        extra = {k: v for k, v in attrs.items() if k not in REQUIRED_METADATA_KEYS}
        meta = cls(**known)  # type: ignore[arg-type]
        meta.state_dimension = int(meta.state_dimension)
        meta.action_dimension = int(meta.action_dimension)
        meta.control_frequency = float(meta.control_frequency)
        meta.generation_or_replay_seed = int(meta.generation_or_replay_seed)
        meta.extra = extra
        return meta

"""Shared fixtures: fabricated episodes and datasets (no simulator needed).

Most tests run against small fabricated datasets so the suite is fast and
simulator-independent. Tests that need a live ManiSkill env are marked
``sim`` and skipped unless ``-m sim`` (or ``-m "sim or not sim"``) is
requested; the end-to-end sim path is exercised by scripts/smoke_test.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from actsemble.data.schema import DatasetMetadata  # noqa: E402
from actsemble.data.writer import write_dataset, write_private_provenance  # noqa: E402
from actsemble.types import EpisodeRecord  # noqa: E402

STATE_DIM = 8
ACTION_DIM = 3


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: needs a live ManiSkill environment (slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="sim tests run only with -m sim (see smoke_test.py)")
    for item in items:
        if item.get_closest_marker("sim") is not None:
            item.add_marker(skip)


def make_episode(episode_id: str, T: int, seed: int) -> EpisodeRecord:
    """A smooth fabricated episode obeying all schema invariants."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=STATE_DIM)
    t = np.arange(T + 1)[:, None]
    states = np.sin(0.05 * t + phase)[:, :STATE_DIM].astype(np.float32)
    states += 0.01 * rng.standard_normal((T + 1, STATE_DIM)).astype(np.float32)
    actions = np.clip(
        np.diff(states[:, :ACTION_DIM], axis=0) * 5.0
        + 0.05 * rng.standard_normal((T, ACTION_DIM)),
        -1.0,
        1.0,
    ).astype(np.float32)
    previous = np.concatenate([np.zeros((1, ACTION_DIM), np.float32), actions[:-1]])
    return EpisodeRecord(
        episode_id=episode_id,
        state=states[:-1].copy(),
        previous_action=previous,
        action=actions,
        next_state=states[1:].copy(),
        step_index=np.arange(T, dtype=np.int64),
    )


def make_episodes(n: int, T: int = 40, seed: int = 0) -> list[EpisodeRecord]:
    return [make_episode(f"ep_{i:05d}", T, seed * 1000 + i) for i in range(n)]


def make_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        simulator="FakeSim",
        simulator_version="0.0",
        task_id="FakeTask-v1",
        robot="fake_arm",
        observation_mode="state",
        state_dimension=STATE_DIM,
        state_layout=json.dumps({"all": [0, STATE_DIM]}),
        controller="fake_delta",
        action_dimension=ACTION_DIM,
        action_definition=json.dumps(
            {
                "semantics": "fake deltas",
                "frame": "fake",
                "units": "normalized",
                "bounds": [[-1.0] * ACTION_DIM, [1.0] * ACTION_DIM],
                "scaling": "none",
                "clipping_rules": "clip to bounds",
            }
        ),
        control_frequency=20.0,
        simulation_backend="fake_cpu",
        source_dataset="fabricated",
        generation_or_replay_seed=0,
    )


@pytest.fixture
def episodes() -> list[EpisodeRecord]:
    return make_episodes(6, T=40)


@pytest.fixture
def dataset_path(tmp_path, episodes) -> Path:
    path = tmp_path / "fake.h5"
    write_dataset(path, episodes, make_metadata())
    write_private_provenance(
        path,
        {
            "success_only": True,
            "exported_episodes": [
                {"episode_id": ep.episode_id, "source_success": True} for ep in episodes
            ],
            "rejected_episodes": [],
            "rejected_count": 0,
            "conversion_failures": [],
            "conversion_failure_count": 0,
        },
    )
    return path


TINY_POLICY_CFG = {
    "name": "tiny_test_policy",
    "observation": {"mode": "state", "history": 2, "include_previous_action": False},
    "action": {"prediction_horizon": 8, "execution_horizon": 4},
    "model": {"channels": [32, 64], "diffusion_embedding_dim": 32, "kernel_size": 5},
    "diffusion": {
        "training_steps": 20,
        "beta_schedule": "cosine",
        "inference_sampler": "ddim",
        "inference_steps": 5,
        "prediction_type": "epsilon",
        "temperature": 1.0,
    },
    "training": {
        "seed": 0,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "ema_decay": 0.99,
        "gradient_clip_norm": 1.0,
        "val_fraction": 0.34,
        "split_seed": 0,
        "mask_padded_actions": False,
        "max_steps": 10,
        "log_every": 5,
        "eval_every": 5,
    },
}

TINY_COMPONENT_CFG = {
    "name": "tiny_test_component",
    "observation": {"mode": "state", "history": 2, "include_previous_action": False},
    "action": {"prediction_horizon": 8, "execution_horizon": 4},
    "model": {"hidden": [32, 32], "dropout": 0.0},
    "negatives": {
        "enabled_types": [
            "additive_noise",
            "action_scaling",
            "temporal_shift",
            "other_trajectory",
            "partial_reversal",
            "discontinuity_injection",
        ],
        "position_dims": [0, 1],
        "rotation_dims": [2],
        "temporal_shift_min": 2,
        "temporal_shift_max": 6,
        "incompatible_min_distance": 16,
    },
    "training": {
        "seed": 0,
        "objective": "binary_classification",
        "negatives_per_positive": 2,
        "negative_seed": 99,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "gradient_clip_norm": 1.0,
        "val_fraction": 0.34,
        "split_seed": 0,
        "max_steps": 10,
        "log_every": 5,
        "eval_every": 5,
    },
}


@pytest.fixture
def tiny_policy_cfg() -> dict:
    import copy

    return copy.deepcopy(TINY_POLICY_CFG)


@pytest.fixture
def tiny_component_cfg() -> dict:
    import copy

    return copy.deepcopy(TINY_COMPONENT_CFG)


@pytest.fixture(scope="session")
def trained_checkpoints(tmp_path_factory):
    """Train tiny policy + component once per session on a fabricated dataset."""
    import copy

    from actsemble.training.train_component import train_component
    from actsemble.training.train_diffusion_policy import train_diffusion_policy

    root = tmp_path_factory.mktemp("ckpts")
    dataset = root / "fake.h5"
    eps = make_episodes(6, T=40)
    write_dataset(dataset, eps, make_metadata())
    policy_out = train_diffusion_policy(
        policy_cfg=copy.deepcopy(TINY_POLICY_CFG),
        dataset_path=dataset,
        output_dir=root / "policy",
        device="cpu",
    )
    component_out = train_component(
        component_cfg=copy.deepcopy(TINY_COMPONENT_CFG),
        dataset_path=dataset,
        output_dir=root / "component",
        device="cpu",
    )
    return {
        "dataset": dataset,
        "policy_best": policy_out["checkpoints"]["best_ema"],
        "policy_final": policy_out["checkpoints"]["final"],
        "component_best": component_out["checkpoints"]["best"],
        "policy_summary": policy_out,
        "component_summary": component_out,
    }

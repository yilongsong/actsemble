"""Data subsystem: action-window alignment and its execution-offset contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from actsemble.data.windows import enumerate_window_indices, extract_window
from actsemble.policies.meta import PolicyMeta
from actsemble.systems.factory import build_system
from actsemble.systems.interface import check_same_data
from actsemble.systems.standalone import StandaloneDiffusionSystem
from actsemble.systems.temporal_ensemble import TemporalEnsembleSystem
from actsemble.types import StateObservation


class Episode:
    def __init__(self, length, action_dim=2, state_dim=3):
        self.state = np.arange(length * state_dim, dtype=np.float32).reshape(
            length, state_dim
        )
        self.next_state = self.state
        self.previous_action = np.zeros((length, action_dim), np.float32)
        self.action = np.arange(length * action_dim, dtype=np.float32).reshape(
            length, action_dim
        )
        self.episode_id = "e"

    def __len__(self):
        return len(self.state)


def test_window_future_only_starts_at_t():
    episode = Episode(10)
    window = extract_window(
        episode, 3, obs_horizon=2, prediction_horizon=4, alignment="future_only"
    )
    assert np.array_equal(window.action_chunk[0], episode.action[3])
    assert window.action_mask.all()


def test_window_diffusion_policy_alignment():
    episode, obs_horizon = Episode(10), 2
    window = extract_window(
        episode,
        3,
        obs_horizon=obs_horizon,
        prediction_horizon=4,
        alignment="diffusion_policy",
    )
    assert np.array_equal(window.action_chunk[0], episode.action[3 - obs_horizon + 1])
    assert np.array_equal(window.action_chunk[obs_horizon - 1], episode.action[3])


def test_window_dp_alignment_pads_the_start():
    window = extract_window(
        Episode(10),
        0,
        obs_horizon=2,
        prediction_horizon=4,
        alignment="diffusion_policy",
    )
    assert not window.action_mask[0] and window.action_mask[1]


def test_window_unknown_alignment_raises():
    with pytest.raises(ValueError, match="alignment"):
        extract_window(
            Episode(5), 2, obs_horizon=2, prediction_horizon=3, alignment="bogus"
        )


def test_diffusion_policy_window_range_caps_terminal_padding():
    episodes = [Episode(20)]
    obs_horizon, prediction_horizon, action_horizon = 2, 16, 8
    assert len(enumerate_window_indices(episodes)) == 20
    capped = enumerate_window_indices(
        episodes,
        alignment="diffusion_policy",
        obs_horizon=obs_horizon,
        prediction_horizon=prediction_horizon,
        action_horizon=action_horizon,
    )
    assert [t for _, t in capped] == list(range(13))
    assert (
        len(
            enumerate_window_indices(
                episodes,
                alignment="diffusion_policy",
                obs_horizon=obs_horizon,
                prediction_horizon=prediction_horizon,
            )
        )
        == 20
    )


def _policy_meta(alignment):
    return PolicyMeta(
        dataset_hash="d",
        split_hash="s",
        normalization={"method": "minmax_to_unit_range"},
        task_id="PushT-v1",
        controller="c",
        simulation_backend="b",
        state_dim=3,
        action_dim=2,
        action_low=[-1, -1],
        action_high=[1, 1],
        obs_horizon=2,
        prediction_horizon=8,
        action_horizon=4,
        include_previous_action=False,
        extra={"window_alignment": alignment},
    )


class FakePolicy:
    def __init__(self, alignment="future_only"):
        self.meta = _policy_meta(alignment)
        self.checkpoint_hash = "h"

    @property
    def dataset_hash(self):
        return self.meta.dataset_hash

    def reset(self):
        pass

    def new_generator(self, seed):
        return np.random.default_rng(int(seed) % (2**32))

    def sample_action_chunks(self, obs, *, num_samples, generator):
        del obs, generator
        chunks = np.tile(
            np.arange(8, dtype=np.float32)[None, :, None], (num_samples, 1, 2)
        )
        return torch.from_numpy(chunks)


def _observation():
    return StateObservation(
        state=np.zeros(3, np.float32),
        previous_action=np.zeros(2, np.float32),
        step_index=0,
    )


def _standalone(policy, execution=None):
    config = {
        "policy": {"num_candidates": 1},
        "selection": {"type": "candidate_zero"},
        "execution": execution or {},
    }
    return build_system(config, policy, [])


def test_factory_defaults_offset_from_window_alignment():
    assert _standalone(FakePolicy("diffusion_policy")).execution_offset == 1
    assert _standalone(FakePolicy("future_only")).execution_offset == 0
    explicit = _standalone(FakePolicy("diffusion_policy"), {"action_offset": 0})
    assert explicit.execution_offset == 0


def test_dp_aligned_standalone_executes_action_for_t():
    system = _standalone(FakePolicy("diffusion_policy"))
    system.reset(episode_seed=1)
    assert np.array_equal(system.act(_observation()).value, [1, 1])


def test_temporal_ensemble_respects_execution_offset():
    system = TemporalEnsembleSystem(
        FakePolicy("diffusion_policy"), aggregation="latest"
    )
    system.set_execution_offset(2)
    system.reset(episode_seed=1)
    assert np.array_equal(system.act(_observation()).value, [2, 2])


def test_execution_offset_shifts_the_executed_slice():
    system = StandaloneDiffusionSystem(FakePolicy(), num_candidates=1)
    system.reset(episode_seed=1)
    assert np.array_equal(system.act(_observation()).value, [0, 0])
    system = StandaloneDiffusionSystem(FakePolicy(), num_candidates=1)
    system.set_execution_offset(2)
    system.reset(episode_seed=1)
    assert np.array_equal(system.act(_observation()).value, [2, 2])


def test_execution_offset_validation():
    system = StandaloneDiffusionSystem(FakePolicy(), num_candidates=1)
    with pytest.raises(ValueError, match="must fit"):
        system.set_execution_offset(5)
    with pytest.raises(ValueError, match="must fit"):
        system.set_execution_offset(-1)


def test_check_same_data_rejects_alignment_mismatch():
    policy = FakePolicy("diffusion_policy")

    class Component:
        dataset_hash = "d"
        checkpoint_hash = "c"
        meta = {**policy.meta.to_dict(), "extra": {"window_alignment": "future_only"}}

    with pytest.raises(ValueError, match="window_alignment"):
        check_same_data(policy, [Component()])

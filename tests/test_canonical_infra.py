"""Phase-0 shared infra for the authoritative policies: standardize
normalization (ACT), the Diffusion-Policy window alignment, and the system
execution offset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from actsemble.data.normalization import (
    MINMAX,
    STANDARDIZE,
    NormalizationStats,
    Normalizer,
)
from actsemble.data.windows import extract_window
from actsemble.systems.standalone import StandaloneDiffusionSystem
from actsemble.types import StateObservation


# ---- standardize normalization (ACT convention) ---------------------------
def test_standardize_roundtrip_and_zero_mean():
    stats = NormalizationStats(
        method=STANDARDIZE,
        state_mean=np.array([1.0, 2.0, 3.0], np.float32),
        state_std=np.array([2.0, 0.5, 1.0], np.float32),
        action_mean=np.array([0.0, -1.0], np.float32),
        action_std=np.array([1.0, 2.0], np.float32),
    )
    n = Normalizer(stats)
    x = np.array([[1.0, 2.0, 3.0], [3.0, 2.5, 4.0]], np.float32)
    z = n.normalize_state(x)
    assert np.allclose(z[0], [0, 0, 0], atol=1e-6)  # the mean maps to 0
    assert np.allclose(n.unnormalize_state(z), x, atol=1e-5)
    a = np.array([[0.0, -1.0], [2.0, 3.0]], np.float32)
    assert np.allclose(n.unnormalize_action(n.normalize_action(a)), a, atol=1e-5)


def test_standardize_dict_roundtrip():
    stats = NormalizationStats(
        method=STANDARDIZE, state_mean=np.zeros(2, np.float32), state_std=np.ones(2, np.float32),
        action_mean=np.zeros(2, np.float32), action_std=np.ones(2, np.float32),
    )
    d = stats.to_dict()
    assert d["method"] == STANDARDIZE and "state_mean" in d
    r = NormalizationStats.from_dict(d)
    assert r.method == STANDARDIZE and np.allclose(r.action_std, stats.action_std)


def test_minmax_is_still_the_default_and_backward_compatible():
    stats = NormalizationStats(
        state_min=-np.ones(2, np.float32), state_max=np.ones(2, np.float32),
        action_min=-np.ones(2, np.float32), action_max=np.ones(2, np.float32),
    )
    assert stats.method == MINMAX
    d = stats.to_dict()
    assert d["method"] == MINMAX and "state_min" in d
    NormalizationStats.from_dict(d)  # old-format checkpoints still load


# ---- window alignment ------------------------------------------------------
class _Ep:
    def __init__(self, T, A=2, S=3):
        self.state = np.arange(T * S, dtype=np.float32).reshape(T, S)
        self.previous_action = np.zeros((T, A), np.float32)
        self.action = np.arange(T * A, dtype=np.float32).reshape(T, A)  # action[i]=[2i,2i+1]
        self.episode_id = "e"

    def __len__(self):
        return len(self.state)


def test_window_future_only_starts_at_t():
    ep = _Ep(10)
    w = extract_window(ep, 3, obs_horizon=2, prediction_horizon=4, alignment="future_only")
    assert np.array_equal(w.action_chunk[0], ep.action[3]) and w.action_mask.all()


def test_window_diffusion_policy_alignment():
    ep, Ho = _Ep(10), 2
    w = extract_window(ep, 3, obs_horizon=Ho, prediction_horizon=4, alignment="diffusion_policy")
    assert np.array_equal(w.action_chunk[0], ep.action[3 - Ho + 1])  # starts at t-Ho+1
    assert np.array_equal(w.action_chunk[Ho - 1], ep.action[3])       # index Ho-1 = action for t


def test_window_dp_alignment_pads_the_start():
    ep = _Ep(10)
    w = extract_window(ep, 0, obs_horizon=2, prediction_horizon=4, alignment="diffusion_policy")
    assert not w.action_mask[0] and w.action_mask[1]  # act_start=-1 -> first is padding


def test_window_unknown_alignment_raises():
    with pytest.raises(ValueError, match="alignment"):
        extract_window(_Ep(5), 2, obs_horizon=2, prediction_horizon=3, alignment="bogus")


# ---- execution offset ------------------------------------------------------
class _Meta:
    obs_horizon = 2
    prediction_horizon = 8
    action_horizon = 4
    action_dim = 2
    state_dim = 3
    include_previous_action = False


class _FakePolicy:
    def __init__(self):
        self.meta = _Meta()
        self.checkpoint_hash = "h"

    def reset(self):
        pass

    def new_generator(self, seed):
        return np.random.default_rng(int(seed) % (2**32))

    def sample_action_chunks(self, obs, *, num_samples, generator):
        # chunk[k, i, :] = i, so the executed value reveals which index ran
        arr = np.tile(np.arange(8, dtype=np.float32)[None, :, None], (num_samples, 1, 2))
        return torch.from_numpy(arr)


def _first_action(offset):
    system = StandaloneDiffusionSystem(_FakePolicy(), num_candidates=1)
    if offset:
        system.set_execution_offset(offset)
    system.reset(episode_seed=1)
    obs = StateObservation(state=np.zeros(3, np.float32),
                           previous_action=np.zeros(2, np.float32), step_index=0)
    return system.act(obs).value


def test_execution_offset_shifts_the_executed_slice():
    assert np.array_equal(_first_action(0), [0, 0])   # default: chunk[0]
    assert np.array_equal(_first_action(2), [2, 2])   # DP-aligned: chunk[offset]


def test_execution_offset_validation():
    system = StandaloneDiffusionSystem(_FakePolicy(), num_candidates=1)  # H_p=8, H_a=4
    with pytest.raises(ValueError, match="must fit"):
        system.set_execution_offset(5)  # 5 + 4 > 8
    with pytest.raises(ValueError, match="must fit"):
        system.set_execution_offset(-1)


# ---- dataset-adaptive training budget --------------------------------------
def test_resolve_total_steps_precedence_and_epoch_conversion():
    from actsemble.training.train_diffusion_policy import resolve_total_steps

    # max_epochs -> steps is dataset-adaptive: epochs * (windows // batch)
    assert resolve_total_steps({"max_epochs": 10}, 1000, 100) == 100
    assert resolve_total_steps({"max_epochs": 10}, 2000, 100) == 200  # bigger dataset -> more steps
    # explicit max_steps overrides max_epochs; --max-steps override wins over all
    assert resolve_total_steps({"max_epochs": 10, "max_steps": 500}, 1000, 100) == 500
    assert resolve_total_steps({"max_epochs": 10, "max_steps": 500}, 1000, 100, 42) == 42
    assert resolve_total_steps({}, 1000, 100) == 10000  # fallback

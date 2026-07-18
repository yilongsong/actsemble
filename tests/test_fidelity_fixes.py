"""Reference-fidelity fixes raised in review: window range (SequenceSampler),
execution-offset safety, temporal-ensemble offset, alignment contract,
standardize std clip, DP FiLM, prediction_type guard, DETR ACT, episode sampling.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from actsemble.data.normalization import STANDARDIZE, NormalizationStats, Normalizer
from actsemble.data.torch_dataset import ACTEpisodeDataset
from actsemble.data.windows import enumerate_window_indices
from actsemble.policies.act.model import ACTModel
from actsemble.policies.diffusion.policy import build_model, build_scheduler
from actsemble.policies.meta import PolicyMeta
from actsemble.systems.factory import build_system
from actsemble.systems.interface import check_same_data
from actsemble.systems.temporal_ensemble import TemporalEnsembleSystem
from actsemble.types import StateObservation


class _Ep:
    def __init__(self, T, A=2, S=3):
        self.state = np.arange(T * S, dtype=np.float32).reshape(T, S)
        self.next_state = self.state
        self.previous_action = np.zeros((T, A), np.float32)
        self.action = np.arange(T * A, dtype=np.float32).reshape(T, A)
        self.episode_id = "e"

    def __len__(self):
        return len(self.state)


# ---- window range: reference SequenceSampler terminal cap --------------------
def test_diffusion_policy_window_range_caps_terminal_padding():
    eps = [_Ep(20)]
    Ho, Hp, Ha = 2, 16, 8
    assert len(enumerate_window_indices(eps)) == 20  # future_only: every timestep
    capped = enumerate_window_indices(
        eps, alignment="diffusion_policy", obs_horizon=Ho,
        prediction_horizon=Hp, action_horizon=Ha,
    )
    # pad_after = H_a-1: last t = T + H_o + H_a - H_p - 2 = 12 -> 13 windows (= T-7)
    assert [t for _, t in capped] == list(range(13))
    # no cap without action_horizon (backward compatible)
    assert len(enumerate_window_indices(
        eps, alignment="diffusion_policy", obs_horizon=Ho, prediction_horizon=Hp)) == 20


# ---- fakes for system-level tests -------------------------------------------
def _meta(alignment):
    return PolicyMeta(
        dataset_hash="d", split_hash="s", normalization={"method": "minmax_to_unit_range"},
        task_id="PushT-v1", controller="c", simulation_backend="b",
        state_dim=3, action_dim=2, action_low=[-1, -1], action_high=[1, 1],
        obs_horizon=2, prediction_horizon=8, action_horizon=4,
        include_previous_action=False, extra={"window_alignment": alignment},
    )


class _FakePolicy:
    def __init__(self, alignment="future_only"):
        self.meta = _meta(alignment)
        self.checkpoint_hash = "h"

    @property
    def dataset_hash(self):
        return self.meta.dataset_hash

    def reset(self):
        pass

    def new_generator(self, seed):
        return np.random.default_rng(int(seed) % (2**32))

    def sample_action_chunks(self, obs, *, num_samples, generator):
        arr = np.tile(np.arange(8, dtype=np.float32)[None, :, None], (num_samples, 1, 2))
        return torch.from_numpy(arr)  # chunk[k, i, :] = i


def _standalone(policy, execution=None):
    cfg = {"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"},
           "execution": execution or {}}
    return build_system(cfg, policy, [])


def _obs():
    return StateObservation(state=np.zeros(3, np.float32),
                            previous_action=np.zeros(2, np.float32), step_index=0)


# ---- execution-offset safety: derived from the policy's alignment ------------
def test_factory_defaults_offset_from_window_alignment():
    assert _standalone(_FakePolicy("diffusion_policy")).execution_offset == 1  # H_o - 1
    assert _standalone(_FakePolicy("future_only")).execution_offset == 0
    # explicit action_offset wins over the alignment default
    assert _standalone(_FakePolicy("diffusion_policy"), {"action_offset": 0}).execution_offset == 0


def test_dp_aligned_standalone_executes_action_for_t():
    dp = _standalone(_FakePolicy("diffusion_policy"))
    dp.reset(episode_seed=1)
    assert np.array_equal(dp.act(_obs()).value, [1, 1])  # chunk[H_o-1], not chunk[0]


# ---- temporal ensemble emits chunk[offset + age] -----------------------------
def test_temporal_ensemble_respects_execution_offset():
    sysm = TemporalEnsembleSystem(_FakePolicy("diffusion_policy"), aggregation="latest")
    sysm.set_execution_offset(2)
    sysm.reset(episode_seed=1)
    assert np.array_equal(sysm.act(_obs()).value, [2, 2])  # age 0 -> idx = offset


# ---- alignment contract ------------------------------------------------------
def test_check_same_data_rejects_alignment_mismatch():
    policy = _FakePolicy("diffusion_policy")

    class _Comp:
        dataset_hash = "d"
        checkpoint_hash = "c"
        meta = {**policy.meta.to_dict(), "extra": {"window_alignment": "future_only"}}

    with pytest.raises(ValueError, match="window_alignment"):
        check_same_data(policy, [_Comp()])


# ---- standardize std clip (reference ACT: >= 0.01) ---------------------------
def test_standardize_clips_near_constant_dims():
    s, _ = Normalizer._standardize(
        np.array([0.0, 5.0], np.float32), np.array([0.001, 2.0], np.float32)
    )
    assert np.isclose(s[0], 100.0)  # 1/max(0.001, 0.01), not 1/0.001
    assert np.isclose(s[1], 0.5)


# ---- DP FiLM (cond_predict_scale) + prediction_type guard --------------------
def test_build_model_threads_cond_predict_scale():
    meta = _meta("diffusion_policy")
    on = build_model({"model": {"channels": [32, 64], "cond_predict_scale": True}}, meta)
    off = build_model({"model": {"channels": [32, 64]}}, meta)
    assert on.downs[0][0].cond_predict_scale is True
    assert off.downs[0][0].cond_predict_scale is False


def test_build_scheduler_rejects_non_epsilon():
    with pytest.raises(NotImplementedError, match="epsilon"):
        build_scheduler({"diffusion": {"prediction_type": "sample"}})
    build_scheduler({"diffusion": {}})  # default epsilon OK


# ---- DETR ACT ----------------------------------------------------------------
def test_act_arch_validates_and_detr_is_deterministic():
    with pytest.raises(ValueError, match="arch"):
        ACTModel(obs_feature_dim=6, action_dim=2, obs_horizon=1, prediction_horizon=8, arch="bad")
    m = ACTModel(
        obs_feature_dim=6, action_dim=2, obs_horizon=1, prediction_horizon=8,
        hidden_dim=32, n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
        dim_feedforward=64, pos_embedding="sinusoidal", arch="detr",
    ).eval()
    obs, z = torch.randn(2, 1, 6), torch.zeros(2, m.latent_dim)
    with torch.no_grad():
        a, b = m.decode(obs, z), m.decode(obs, z)
    assert torch.allclose(a, b) and torch.isfinite(a).all() and a.shape == (2, 8, 2)


# ---- episode-weighted ACT sampling -------------------------------------------
def test_compat_dataset_positive_chunk_respects_alignment():
    from actsemble.components.action_chunk_compatibility import NegativeConfig, NegativeGenerator
    from actsemble.training.train_component import CompatibilityDataset

    ep = _Ep(10)  # action[i] = [2i, 2i+1]
    norm = Normalizer(NormalizationStats(
        method=STANDARDIZE, state_mean=np.zeros(3, np.float32), state_std=np.ones(3, np.float32),
        action_mean=np.zeros(2, np.float32), action_std=np.ones(2, np.float32)))  # identity
    neg = NegativeGenerator(NegativeConfig.from_dict({}), action_low=np.array([-99.0, -99.0]),
                            action_high=np.array([99.0, 99.0]), prediction_horizon=8, obs_horizon=2)
    ds = CompatibilityDataset(
        [ep], norm, neg, obs_horizon=2, prediction_horizon=8, include_previous_action=False,
        negatives_per_positive=1, negative_seed=0, alignment="diffusion_policy", action_horizon=4)
    pos = ds[3]["positive_chunk"].numpy()  # index list is [(0,0)..(0,6)]; item 3 -> t=3
    assert np.array_equal(pos[1], ep.action[3])  # DP-aligned index H_o-1 = a_t = [6,7]
    assert np.array_equal(pos[0], ep.action[2])  # index 0 = a_{t-1} = [4,5]


def test_ddim_uses_conventional_step_ratio_spacing():
    from actsemble.policies.diffusion.scheduler import DiffusionScheduler
    ts = DiffusionScheduler(num_train_steps=100).inference_timesteps(16).tolist()
    assert ts == [90, 84, 78, 72, 66, 60, 54, 48, 42, 36, 30, 24, 18, 12, 6, 0]


def test_euler_sample_rejects_zero_steps():
    from actsemble.policies.flow.sampling import euler_sample
    with pytest.raises(ValueError, match="num_steps"):
        euler_sample(None, torch.zeros(1, 4), num_samples=1, prediction_horizon=8,
                     action_dim=2, num_steps=0)


def test_act_episode_dataset_len_and_reproducible():
    eps = [_Ep(12), _Ep(20), _Ep(8)]
    norm = Normalizer(NormalizationStats(
        method=STANDARDIZE, state_mean=np.zeros(3, np.float32), state_std=np.ones(3, np.float32),
        action_mean=np.zeros(2, np.float32), action_std=np.ones(2, np.float32),
    ))
    kw = dict(obs_horizon=1, prediction_horizon=8, start_seed=7)
    d1 = ACTEpisodeDataset(eps, norm, **kw)
    d2 = ACTEpisodeDataset(eps, norm, **kw)
    assert len(d1) == 3
    assert [d1[i]["t"].item() for i in range(3)] == [d2[i]["t"].item() for i in range(3)]
    assert d1[0]["action_chunk"].shape == (8, 2)

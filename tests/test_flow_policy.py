"""Unit tests for the flow-matching policy (CPU, tiny U-Net, no dataset/GPU).

Covers the Euler ODE sampler (determinism, few-step, shape/clip), the CFM loss
direction, ActionChunkPolicy compliance, checkpoint round-trip, and loader
dispatch of the flow checkpoint kind.
"""

from __future__ import annotations

import numpy as np
import torch

from actsemble.data.normalization import NormalizationStats, Normalizer
from actsemble.policies.diffusion.model import ConditionalUnet1d
from actsemble.policies.flow.policy import FlowMatchingPolicy
from actsemble.policies.flow.sampling import euler_sample
from actsemble.policies.interface import ActionChunkPolicy
from actsemble.policies.loader import load_policy
from actsemble.policies.meta import PolicyMeta

SD, AD, HO, HP = 3, 2, 2, 4


def _model():
    return ConditionalUnet1d(
        action_dim=AD, global_cond_dim=HO * SD, channels=(16, 32),
        diffusion_embedding_dim=16, kernel_size=3,
    )


def _meta():
    stats = NormalizationStats(
        state_min=-np.ones(SD, np.float32), state_max=np.ones(SD, np.float32),
        action_min=-np.ones(AD, np.float32), action_max=np.ones(AD, np.float32),
    )
    meta = PolicyMeta(
        dataset_hash="d", split_hash="s", normalization=stats.to_dict(),
        task_id="PushT-v1", controller="c", simulation_backend="b",
        state_dim=SD, action_dim=AD, action_low=[-1.0] * AD, action_high=[1.0] * AD,
        obs_horizon=HO, prediction_horizon=HP, action_horizon=2,
    )
    return meta, Normalizer(stats)


def _policy(model=None):
    meta, norm = _meta()
    return FlowMatchingPolicy(model or _model(), norm, meta,
                              config={"flow": {"inference_steps": 8}}, device=torch.device("cpu"))


# ---- Euler ODE sampler -----------------------------------------------------
def test_euler_sample_shape_and_determinism():
    m = _model().eval()
    cond = torch.randn(1, HO * SD)
    a = euler_sample(m, cond, num_samples=3, prediction_horizon=HP, action_dim=AD,
                     num_steps=8, generator=torch.Generator().manual_seed(0))
    b = euler_sample(m, cond, num_samples=3, prediction_horizon=HP, action_dim=AD,
                     num_steps=8, generator=torch.Generator().manual_seed(0))
    assert a.shape == (3, HP, AD) and torch.equal(a, b)  # deterministic given the seed


def test_euler_one_step_is_z0_plus_v():
    m = _model().eval()
    cond = torch.randn(1, HO * SD)
    gen = torch.Generator().manual_seed(1)
    z0 = torch.empty(1, HP, AD).normal_(generator=torch.Generator().manual_seed(1))
    out = euler_sample(m, cond, num_samples=1, prediction_horizon=HP, action_dim=AD,
                       num_steps=1, generator=gen)
    with torch.no_grad():  # 1-step Euler from t=0: z1 = z0 + v(z0, 0)*1
        v = m(z0, torch.zeros(1), cond)
    assert torch.allclose(out, z0 + v, atol=1e-5)


# ---- policy contract -------------------------------------------------------
def test_is_action_chunk_policy():
    assert isinstance(_policy(), ActionChunkPolicy)


def test_sample_shape_clip_and_determinism():
    policy = _policy()
    obs = np.random.randn(HO, SD).astype(np.float32)
    a = policy.sample_action_chunks(obs, num_samples=4, generator=policy.new_generator(0))
    b = policy.sample_action_chunks(obs, num_samples=4, generator=policy.new_generator(0))
    assert a.shape == (4, HP, AD) and torch.equal(a, b)
    assert (a >= -1.0 - 1e-6).all() and (a <= 1.0 + 1e-6).all()


# ---- CFM loss direction ----------------------------------------------------
def test_cfm_target_is_straight_path_velocity():
    x1 = torch.randn(5, HP, AD)
    z0 = torch.randn(5, HP, AD)
    t = torch.rand(5)[:, None, None]
    z_t = (1 - t) * z0 + t * x1
    # the interpolant is the straight path z0 -> x1, whose constant velocity
    # (the CFM regression target) is (x1 - z0), independent of t
    assert torch.allclose(z_t, z0 + t * (x1 - z0), atol=1e-6)


# ---- checkpoint + loader ---------------------------------------------------
def test_checkpoint_roundtrip_and_loader(tmp_path):
    meta, _ = _meta()
    model = _model()
    path = tmp_path / "flow.pt"
    FlowMatchingPolicy.save_checkpoint(
        path, config={"model": {"channels": [16, 32], "diffusion_embedding_dim": 16,
                                "kernel_size": 3}, "flow": {"inference_steps": 8}},
        meta=meta, model_state=model.state_dict(), ema_state=model.state_dict(),
    )
    loaded = load_policy(path, device="cpu")
    assert type(loaded).__name__ == "FlowMatchingPolicy" and loaded.checkpoint_hash
    obs = np.random.randn(HO, SD).astype(np.float32)
    ref = FlowMatchingPolicy(model, Normalizer(NormalizationStats.from_dict(meta.normalization)),
                             meta, config={"flow": {"inference_steps": 8}}, device=torch.device("cpu"))
    assert torch.allclose(
        loaded.sample_action_chunks(obs, num_samples=1, generator=loaded.new_generator(0)),
        ref.sample_action_chunks(obs, num_samples=1, generator=ref.new_generator(0)), atol=1e-6,
    )

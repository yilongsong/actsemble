"""Diffusion model/scheduler tensor shapes and normalization roundtrips."""

import numpy as np
import pytest
import torch

from actsemble.data.normalization import NormalizationStats, Normalizer
from actsemble.policies.diffusion.model import ConditionalUnet1d
from actsemble.policies.diffusion.scheduler import DiffusionScheduler


def test_unet_output_shape():
    model = ConditionalUnet1d(action_dim=3, global_cond_dim=16, channels=(32, 64))
    x = torch.randn(5, 8, 3)
    out = model(x, torch.randint(0, 20, (5,)), torch.randn(5, 16))
    assert out.shape == (5, 8, 3)


def test_unet_three_level_shape():
    model = ConditionalUnet1d(action_dim=6, global_cond_dim=62, channels=(32, 64, 128))
    x = torch.randn(2, 16, 6)
    out = model(x, torch.randint(0, 100, (2,)), torch.randn(2, 62))
    assert out.shape == (2, 16, 6)


def test_forward_diffusion_interpolates():
    sched = DiffusionScheduler(num_train_steps=50)
    x0 = torch.ones(4, 8, 3)
    noise = torch.zeros_like(x0)
    # tau=0 is nearly clean; the deepest tau is nearly pure noise-scale.
    x_low = sched.add_noise(x0, noise, torch.zeros(4, dtype=torch.long))
    assert torch.allclose(x_low, x0 * sched.alphas_cumprod[0].sqrt(), atol=1e-5)
    a_bar_last = sched.alphas_cumprod[-1]
    assert a_bar_last < 0.05  # cosine schedule ends heavily noised
    x_high = sched.add_noise(x0, torch.randn_like(x0), torch.full((4,), 49, dtype=torch.long))
    assert x_high.shape == x0.shape


def test_inference_timesteps_descend_to_zero():
    sched = DiffusionScheduler(num_train_steps=100)
    ts = sched.inference_timesteps(10)
    assert len(ts) == 10
    assert ts[-1].item() == 0
    assert (ts[:-1] > ts[1:]).all()
    with pytest.raises(ValueError):
        sched.inference_timesteps(101)


def test_ddim_step_shapes_and_final_x0():
    sched = DiffusionScheduler(num_train_steps=20)
    x = torch.randn(3, 8, 2)
    eps = torch.randn_like(x)
    out = sched.ddim_step(x, eps, tau=10, tau_prev=5)
    assert out.shape == x.shape
    final = sched.ddim_step(x, eps, tau=5, tau_prev=-1)
    assert final.abs().max() <= 1.0 + 1e-6  # x0 clipped to normalized range


def test_normalizer_roundtrip():
    rng = np.random.default_rng(0)
    stats = NormalizationStats(
        state_min=np.array([-2.0, 0.0, 5.0], np.float32),
        state_max=np.array([2.0, 1.0, 5.0], np.float32),  # third dim constant
        action_min=np.array([-1.0], np.float32),
        action_max=np.array([1.0], np.float32),
    )
    norm = Normalizer(stats)
    x = rng.uniform(-2, 2, size=(10, 3)).astype(np.float32)
    x[:, 1] = rng.uniform(0, 1, size=10)
    x[:, 2] = 5.0
    z = norm.normalize_state(x)
    assert np.abs(z[:, :2]).max() <= 1.0 + 1e-6
    assert np.allclose(z[:, 2], 0.0)  # constant dim maps to 0
    back = norm.unnormalize_state(z)
    np.testing.assert_allclose(back[:, :2], x[:, :2], atol=1e-5)
    # torch path agrees with numpy path
    zt = norm.normalize_state(torch.from_numpy(x))
    np.testing.assert_allclose(zt.numpy(), z, atol=1e-6)


def test_horizon_divisibility_matters():
    model = ConditionalUnet1d(action_dim=3, global_cond_dim=8, channels=(16, 32))
    with pytest.raises(Exception):
        # horizon 7 is not divisible by 2; up/down path cannot restore it
        model(torch.randn(1, 7, 3), torch.zeros(1, dtype=torch.long), torch.randn(1, 8))

"""Policy subsystem: diffusion model and scheduler numerical contracts."""

import numpy as np
import pytest
import torch

from actsemble.data.normalization import NormalizationStats, Normalizer
from actsemble.policies.diffusion.model import ConditionalUnet1d
from actsemble.policies.diffusion.policy import build_model, build_scheduler
from actsemble.policies.diffusion.scheduler import DiffusionScheduler
from actsemble.policies.meta import PolicyMeta


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
    x_high = sched.add_noise(
        x0, torch.randn_like(x0), torch.full((4,), 49, dtype=torch.long)
    )
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
    np.testing.assert_allclose(back, x, atol=1e-5)
    # torch path agrees with numpy path
    zt = norm.normalize_state(torch.from_numpy(x))
    np.testing.assert_allclose(zt.numpy(), z, atol=1e-6)
    torch_back = norm.unnormalize_state(zt)
    np.testing.assert_allclose(torch_back.numpy(), x, atol=1e-5)


def test_horizon_divisibility_matters():
    model = ConditionalUnet1d(action_dim=3, global_cond_dim=8, channels=(16, 32))
    with pytest.raises(Exception):
        # horizon 7 is not divisible by 2; up/down path cannot restore it
        model(torch.randn(1, 7, 3), torch.zeros(1, dtype=torch.long), torch.randn(1, 8))


def _diffusion_meta():
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
    )


def test_build_model_threads_cond_predict_scale():
    enabled = build_model(
        {"model": {"channels": [32, 64], "cond_predict_scale": True}},
        _diffusion_meta(),
    )
    disabled = build_model({"model": {"channels": [32, 64]}}, _diffusion_meta())
    assert enabled.downs[0][0].cond_predict_scale is True
    assert disabled.downs[0][0].cond_predict_scale is False


def test_build_scheduler_rejects_non_epsilon():
    with pytest.raises(NotImplementedError, match="epsilon"):
        build_scheduler({"diffusion": {"prediction_type": "sample"}})
    build_scheduler({"diffusion": {}})


def test_ddim_uses_conventional_step_ratio_spacing():
    timesteps = DiffusionScheduler(num_train_steps=100).inference_timesteps(16).tolist()
    assert timesteps == [90, 84, 78, 72, 66, 60, 54, 48, 42, 36, 30, 24, 18, 12, 6, 0]


def test_ddpm_step_is_canonical_posterior_not_ddim_eta1():
    scheduler = DiffusionScheduler(num_train_steps=100)
    timestep, previous_timestep = 60, 54
    torch.manual_seed(0)
    sample = 0.02 * torch.randn(2, 4, 2)
    epsilon = 0.02 * torch.randn(2, 4, 2)
    output = scheduler.ddpm_step(
        sample,
        epsilon,
        timestep,
        previous_timestep,
        generator=torch.Generator().manual_seed(7),
    )
    alpha_bar = scheduler.alphas_cumprod[timestep].float()
    alpha_bar_previous = scheduler.alphas_cumprod[previous_timestep].float()
    predicted_x0 = (sample - (1 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()
    assert predicted_x0.abs().max() <= 1.0
    alpha = alpha_bar / alpha_bar_previous
    beta = 1 - alpha
    x0_coefficient = alpha_bar_previous.sqrt() * beta / (1 - alpha_bar)
    sample_coefficient = alpha.sqrt() * (1 - alpha_bar_previous) / (1 - alpha_bar)
    variance = (1 - alpha_bar_previous) / (1 - alpha_bar) * beta
    noise = torch.empty_like(sample).normal_(generator=torch.Generator().manual_seed(7))
    expected = (
        x0_coefficient * predicted_x0
        + sample_coefficient * sample
        + variance.clamp(min=0).sqrt() * noise
    )
    assert torch.allclose(output, expected, atol=1e-5)
    assert torch.allclose(
        scheduler.ddpm_step(sample, epsilon, timestep, -1), predicted_x0, atol=1e-6
    )


def test_scheduler_timestep_spacing_is_a_named_config_rule():
    leading = DiffusionScheduler(
        num_train_steps=100, timestep_spacing="leading"
    ).inference_timesteps(16)
    linspace = DiffusionScheduler(
        num_train_steps=100, timestep_spacing="linspace"
    ).inference_timesteps(16)
    assert leading.tolist() == [
        90,
        84,
        78,
        72,
        66,
        60,
        54,
        48,
        42,
        36,
        30,
        24,
        18,
        12,
        6,
        0,
    ]
    assert linspace.tolist() == [
        94,
        88,
        81,
        75,
        69,
        62,
        56,
        50,
        44,
        38,
        31,
        25,
        19,
        12,
        6,
        0,
    ]
    with pytest.raises(ValueError, match="timestep_spacing"):
        DiffusionScheduler(timestep_spacing="bogus")

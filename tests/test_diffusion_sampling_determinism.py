"""Sampling determinism, EMA loading, clipping, receding-horizon extraction."""

import numpy as np
import torch

from actsemble.policies.diffusion.policy import DiffusionPolicy


def _policy(trained_checkpoints, **kwargs) -> DiffusionPolicy:
    return DiffusionPolicy.from_checkpoint(
        trained_checkpoints["policy_best"], device="cpu", **kwargs
    )


def _obs(policy, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(policy.meta.obs_horizon, policy.meta.state_dim)).astype(
        np.float32
    )


def test_same_seed_same_samples(trained_checkpoints):
    policy = _policy(trained_checkpoints)
    obs = _obs(policy)
    a = policy.sample_action_chunks(obs, num_samples=4, generator=policy.new_generator(3))
    b = policy.sample_action_chunks(obs, num_samples=4, generator=policy.new_generator(3))
    assert torch.equal(a, b)


def test_different_seeds_differ(trained_checkpoints):
    policy = _policy(trained_checkpoints)
    obs = _obs(policy)
    a = policy.sample_action_chunks(obs, num_samples=4, generator=policy.new_generator(3))
    b = policy.sample_action_chunks(obs, num_samples=4, generator=policy.new_generator(4))
    assert not torch.equal(a, b)


def test_candidates_within_one_call_are_independent(trained_checkpoints):
    policy = _policy(trained_checkpoints)
    chunks = policy.sample_action_chunks(
        _obs(policy), num_samples=8, generator=policy.new_generator(0)
    )
    assert chunks.shape[0] == 8
    flat = chunks.flatten(1)
    assert not any(torch.equal(flat[i], flat[j]) for i in range(8) for j in range(i + 1, 8))


def test_output_shape_and_bounds(trained_checkpoints):
    policy = _policy(trained_checkpoints)
    chunks = policy.sample_action_chunks(
        _obs(policy), num_samples=5, generator=policy.new_generator(1)
    )
    assert chunks.shape == (5, policy.meta.prediction_horizon, policy.meta.action_dim)
    low = torch.tensor(policy.meta.action_low)
    high = torch.tensor(policy.meta.action_high)
    assert (chunks >= low - 1e-6).all() and (chunks <= high + 1e-6).all()
    assert torch.isfinite(chunks).all()


def test_ddpm_sampler_also_deterministic(trained_checkpoints):
    policy = _policy(
        trained_checkpoints, sampler_overrides={"inference_sampler": "ddpm", "inference_steps": 5}
    )
    obs = _obs(policy)
    a = policy.sample_action_chunks(obs, num_samples=2, generator=policy.new_generator(9))
    b = policy.sample_action_chunks(obs, num_samples=2, generator=policy.new_generator(9))
    assert torch.equal(a, b)


def test_temperature_scales_initial_noise_spread(trained_checkpoints):
    cold = _policy(trained_checkpoints, sampler_overrides={"temperature": 0.0})
    obs = _obs(cold)
    a = cold.sample_action_chunks(obs, num_samples=3, generator=cold.new_generator(0))
    # temperature 0 => identical initial noise (all zeros) => samples collapse.
    # allclose, not equal: conv kernels are not batch-position bitwise
    # invariant (ULP-level differences across batch rows are expected).
    assert torch.allclose(a[0], a[1], atol=1e-6)
    assert torch.allclose(a[1], a[2], atol=1e-6)
    warm = _policy(trained_checkpoints, sampler_overrides={"temperature": 1.0})
    b = warm.sample_action_chunks(obs, num_samples=3, generator=warm.new_generator(0))
    assert not torch.allclose(b[0], b[1], atol=1e-3)  # spread returns at T=1


def test_ema_and_raw_weights_differ(trained_checkpoints):
    ema = _policy(trained_checkpoints, use_ema=True)
    raw = _policy(trained_checkpoints, use_ema=False)
    assert ema.weights_kind == "ema" and raw.weights_kind == "raw"
    obs = _obs(ema)
    a = ema.sample_action_chunks(obs, num_samples=1, generator=ema.new_generator(0))
    b = raw.sample_action_chunks(obs, num_samples=1, generator=raw.new_generator(0))
    assert not torch.equal(a, b)


def test_receding_horizon_extraction(trained_checkpoints):
    from actsemble.systems.standalone import StandaloneDiffusionSystem
    from actsemble.types import StateObservation

    policy = _policy(trained_checkpoints)
    system = StandaloneDiffusionSystem(policy, candidate_root_seed=0)
    system.reset(episode_seed=5)
    actions = []
    rng = np.random.default_rng(0)
    for step in range(system.action_horizon + 1):
        obs = StateObservation(
            state=rng.uniform(-1, 1, policy.meta.state_dim).astype(np.float32),
            previous_action=np.zeros(policy.meta.action_dim, np.float32),
            step_index=step,
        )
        actions.append(system.act(obs).value)
    diag = system.diagnostics()
    # one replan for the first H_a steps, a second when the queue empties
    assert diag["num_replans"] == 2
    assert len(actions) == system.action_horizon + 1

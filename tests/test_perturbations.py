"""Perturbation determinism and semantics (sim-free where possible)."""

import numpy as np
import pytest

from actsemble.sim.perturbations.action_latency import ActionLatencyPerturbation
from actsemble.sim.perturbations.action_noise import ActionNoisePerturbation
from actsemble.sim.perturbations.base import build_perturbations
from actsemble.sim.perturbations.object_nudge import ObjectNudgePerturbation
from actsemble.sim.perturbations.observation_noise import ObservationNoisePerturbation
from actsemble.types import RobotAction, StateObservation


def _action(v=0.5, dim=3):
    return RobotAction(value=np.full(dim, v, np.float32))


def _obs(dim=6, step=0):
    return StateObservation(
        state=np.linspace(-1, 1, dim).astype(np.float32),
        previous_action=np.zeros(3, np.float32),
        step_index=step,
    )


def test_action_noise_seeded_and_bounded():
    p1 = ActionNoisePerturbation(scale=0.1, seed=5)
    p2 = ActionNoisePerturbation(scale=0.1, seed=5)
    p1.reset(episode_seed=3)
    p2.reset(episode_seed=3)
    a1 = p1.modify_action(_action(), 0).value
    a2 = p2.modify_action(_action(), 0).value
    np.testing.assert_array_equal(a1, a2)
    p3 = ActionNoisePerturbation(scale=0.1, seed=5)
    p3.reset(episode_seed=4)  # different episode seed => different noise
    a3 = p3.modify_action(_action(), 0).value
    assert not np.array_equal(a1, a3)
    assert np.abs(a1 - 0.5).max() <= 0.3 + 1e-6  # gaussian clipped at 3 sigma


def test_action_latency_delays_by_k_steps():
    p = ActionLatencyPerturbation(delay_steps=2)
    p.reset(episode_seed=0)
    outs = []
    for i in range(5):
        outs.append(p.modify_action(_action(float(i)), i).value[0])
    # first two are zeros, then the delayed stream begins
    assert outs[0] == 0.0 and outs[1] == 0.0
    assert outs[2:] == [0.0, 1.0, 2.0]


def test_observation_noise_only_touches_selected_dims():
    p = ObservationNoisePerturbation(sigma=0.1, dims=[0, 2], seed=1)
    p.reset(episode_seed=0)
    before = _obs()
    after = p.modify_observation(before, 0)
    changed = before.state != after.state
    assert changed[0] and changed[2]
    assert not changed[1] and not changed[3:].any()
    # original untouched (no aliasing)
    assert before.state[0] == np.float32(-1.0)


def test_object_nudge_trigger_deterministic_per_seed():
    p1 = ObjectNudgePerturbation(step_window=(5, 15), seed=2)
    p2 = ObjectNudgePerturbation(step_window=(5, 15), seed=2)
    p1.reset(episode_seed=11)
    p2.reset(episode_seed=11)
    assert p1.trigger_step == p2.trigger_step
    assert 5 <= p1.trigger_step <= 15
    p2.reset(episode_seed=12)
    triggers = set()
    for s in range(20):
        p2.reset(episode_seed=s)
        triggers.add(p2.trigger_step)
    assert len(triggers) > 3  # varies across episodes


def test_build_perturbations_from_specs():
    ps = build_perturbations(
        [
            {"type": "action_noise", "scale": 0.05},
            {"type": "action_latency", "delay_steps": 1},
        ]
    )
    assert [p.name for p in ps] == ["action_noise", "action_latency"]
    with pytest.raises(ValueError, match="Unknown perturbation"):
        build_perturbations([{"type": "wind"}])


@pytest.mark.sim
def test_object_nudge_moves_the_tee_in_sim():
    import torch

    from actsemble.sim.env_factory import make_env

    env = make_env(
        task_id="PushT-v1",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cuda",
        render_mode=None,
    )
    try:
        p = ObjectNudgePerturbation(step_window=(0, 0), max_translation=0.03, seed=0)
        p.reset(episode_seed=0)
        env.reset(seed=0)
        before = env.unwrapped.get_state_dict()["actors"]["Tee"].clone()
        p.before_step(env, 0)
        after = env.unwrapped.get_state_dict()["actors"]["Tee"]
        moved = torch.linalg.norm(after[0, :2] - before[0, :2]).item()
        assert 0.5 * 0.03 * 0.5 < moved <= 0.03 + 1e-6
        # applied exactly once
        p.before_step(env, 0)
        after2 = env.unwrapped.get_state_dict()["actors"]["Tee"]
        assert torch.equal(after, after2)
    finally:
        env.close()

"""Recovery-track v0 primitives (CPU, synthetic states, no sim)."""

from __future__ import annotations

import numpy as np

from actsemble.recovery import QPOS, ReturnController, SupportIndex, quat_yaw, wrap_angle


def _state(qpos7, block_xy, yaw, tcp=(0.0, 0.0, 0.02)):
    s = np.zeros(31)
    s[QPOS] = qpos7
    s[14:17] = tcp
    s[24:26] = block_xy
    s[26] = 0.02
    s[27] = np.cos(yaw / 2.0)
    s[30] = np.sin(yaw / 2.0)
    return s


def _demo_bank(n_ep=20, T=30, seed=0):
    """Synthetic manifold: block along a line, robot qpos a smooth function of
    block position (the conditional structure the estimator should learn)."""
    rng = np.random.default_rng(seed)
    states, eps = [], []
    for e in range(n_ep):
        x0 = rng.uniform(-0.1, 0.1)
        for t in range(T):
            bx = x0 + 0.002 * t
            by = 0.05 * np.sin(bx * 10)
            yaw = 0.5 * bx
            q = np.concatenate([[bx * 2.0, by * 2.0], np.zeros(5)])
            states.append(_state(q, (bx, by), yaw))
            eps.append(e)
    return np.asarray(states), np.asarray(eps)


def test_quat_yaw_roundtrip():
    for yaw in (-2.0, -0.3, 0.0, 1.1, 3.0):
        q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        assert np.isclose(wrap_angle(quat_yaw(q) - yaw), 0.0, atol=1e-9)


def test_stats_directions():
    states, eps = _demo_bank()
    idx = SupportIndex(states, eps)
    on = _state(np.concatenate([[0.0, 0.0], np.zeros(5)]), (0.0, 0.0), 0.0)
    c_on, e_on = idx.stats(on)

    robot_off = on.copy()
    robot_off[0] += 1.0  # robot moved, scene unchanged -> c up, e unchanged
    c_r, e_r = idx.stats(robot_off)
    assert c_r > c_on + 0.5 and np.isclose(e_r, e_on)

    scene_off = on.copy()
    scene_off[24] += 1.0  # block far off-support -> e up
    _c_s, e_s = idx.stats(scene_off)
    assert e_s > e_on + 1.0


def test_calibrate_and_candidates():
    states, eps = _demo_bank()
    idx = SupportIndex(states, eps)
    kc, ke = idx.calibrate(percentile=99, subsample=200)
    assert kc > 0 and ke > 0
    # on-manifold states sit below the 99th-pct thresholds
    c, e = idx.stats(states[5], exclude_ep=eps[5])
    assert c <= kc * 1.5 and e <= ke * 1.5
    cands = idx.candidates(states[5], m=3)
    assert len(cands) == 3
    assert all(q.shape == (7,) and t.shape == (3,) for q, t in cands)


def test_return_controller_reaches_target():
    target = np.array([0.05, -0.03, 0.02])
    ctl = ReturnController(target, lift_dz=0.05, pos_scale=0.1, tol=0.01, max_steps=30)
    tip = np.array([-0.05, 0.06, 0.02])
    # kinematic sandbox: tip moves by the commanded (unnormalized) delta
    for _ in range(30):
        if ctl.done(tip):
            break
        a = ctl.action(tip)
        tip = tip + a[:3] * ctl.pos_scale
    assert ctl.done(tip) and np.linalg.norm(tip - target) < 0.011
    # the path lifted before translating (collision awareness)
    assert ctl._z_safe is not None and ctl._z_safe > 0.05


def test_return_controller_budget():
    ctl = ReturnController(np.array([1.0, 1.0, 0.02]), max_steps=4)
    tip = np.zeros(3)
    n = 0
    while not ctl.done(tip):
        ctl.action(tip)
        n += 1
        assert n < 10
    assert ctl.steps == 4  # budget exhausted, not reached

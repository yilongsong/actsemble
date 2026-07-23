"""Recovery-track v0 primitives, PickCube-v1 instantiation (CPU, synthetic
states, no sim). Mirrors test_recovery.py's structure and coverage."""

from __future__ import annotations

import numpy as np

from actsemble.recovery import (
    PC_GOAL_POS,
    PC_GRASPED,
    PC_OBJ_POS,
    PC_QPOS,
    PC_TCP_POS,
    PickCubeReturnController,
    PickCubeSupportIndex,
)


def _state(qpos9, obj_xyz, tcp=(0.0, 0.0, 0.05), grasped=False):
    s = np.zeros(42)
    s[PC_QPOS] = qpos9
    s[PC_TCP_POS] = tcp
    s[PC_GOAL_POS] = (0.0, 0.0, 0.05)
    s[PC_OBJ_POS] = obj_xyz
    s[PC_GRASPED] = 1.0 if grasped else 0.0
    return s


def _demo_bank(n_ep=20, T=20, seed=0):
    """Synthetic manifold: cube along a line, arm qpos a smooth function of
    cube xy (the conditional structure the estimator should learn); the
    gripper closes (grasped) only in the back half of each episode, at which
    point the cube LIFTS (z rises) — matching real PickCube physics and
    keeping the two grasp phases separated in cube-position space, so the
    kNN-by-cube-position neighbor set doesn't conflate them (a table-height
    cube cannot be confused with a lifted one)."""
    rng = np.random.default_rng(seed)
    states, eps = [], []
    for e in range(n_ep):
        x0 = rng.uniform(-0.05, 0.05)
        for t in range(T):
            cx = x0 + 0.001 * t
            cy = 0.03 * np.sin(cx * 10)
            arm = np.concatenate([[cx * 2.0, cy * 2.0], np.zeros(5)])
            grasped = t >= T // 2
            cz = 0.07 if grasped else 0.02  # lifted vs on-table
            grip = np.array([0.019, 0.019]) if grasped else np.array([0.04, 0.04])
            q = np.concatenate([arm, grip])
            states.append(_state(q, (cx, cy, cz), tcp=(cx, cy, cz + 0.01), grasped=grasped))
            eps.append(e)
    return np.asarray(states), np.asarray(eps)


def test_stats_directions():
    states, eps = _demo_bank()
    idx = PickCubeSupportIndex(states, eps)
    on = states[5]
    c_on, e_on = idx.stats(on, exclude_ep=eps[5])

    robot_off = on.copy()
    robot_off[0] += 1.0  # arm moved, cube unchanged -> c up, e unchanged
    c_r, e_r = idx.stats(robot_off, exclude_ep=eps[5])
    assert c_r > c_on + 0.5 and np.isclose(e_r, e_on)

    scene_off = on.copy()
    scene_off[PC_OBJ_POS.start] += 1.0  # cube far off-support -> e up
    _c_s, e_s = idx.stats(scene_off, exclude_ep=eps[5])
    assert e_s > e_on + 1.0


def test_gripper_state_registers_as_surprisal():
    """Same arm+cube position, only the FINGER qpos flips open<->closed:
    c must rise (a wrongly-open/closed gripper at an otherwise-correct
    pose is surprising) — the reason c uses full 9-dim qpos, not just arm."""
    states, eps = _demo_bank()
    idx = PickCubeSupportIndex(states, eps)
    grasped_state = states[np.where(np.asarray([s[PC_GRASPED][0] for s in states]) > 0.5)[0][0]]
    c_correct, _ = idx.stats(grasped_state)
    flipped = grasped_state.copy()
    flipped[7:9] = 0.04  # command: open, contradicting the grasped-region pose
    c_flipped, _ = idx.stats(flipped)
    assert c_flipped > c_correct + 0.01


def test_calibrate_and_candidates():
    states, eps = _demo_bank()
    idx = PickCubeSupportIndex(states, eps)
    kc, ke = idx.calibrate(percentile=99, subsample=200)
    assert kc > 0 and ke > 0
    c, e = idx.stats(states[5], exclude_ep=eps[5])
    assert c <= kc * 1.5 and e <= ke * 1.5
    cands = idx.candidates(states[5], m=3)
    assert len(cands) == 3
    for q, t, g in cands:
        assert q.shape == (9,) and t.shape == (3,) and isinstance(g, bool)


def test_return_controller_reaches_target_and_sets_gripper():
    target = np.array([0.05, -0.03, 0.03])
    ctl = PickCubeReturnController(
        target, target_grasped=True, lift_dz=0.05, pos_scale=0.1, tol=0.01, max_steps=40
    )
    tip = np.array([-0.05, 0.06, 0.03])
    grip_cmd = 0.0
    for _ in range(40):
        if ctl.done(tip):
            break
        a = ctl.action(tip)
        tip = tip + a[:3] * ctl.pos_scale
        grip_cmd = a[6]
    assert ctl.done(tip) and np.linalg.norm(tip - target) < 0.011
    assert ctl._z_safe is not None and ctl._z_safe > 0.05
    assert grip_cmd < 0  # commanded to CLOSE (negative), matching target_grasped=True


def test_return_controller_commands_open_when_target_not_grasped():
    target = np.array([0.02, 0.02, 0.03])
    ctl = PickCubeReturnController(target, target_grasped=False, pos_scale=0.1, tol=0.01, max_steps=40)
    tip = target.copy()  # start already at the target xy/z
    a = ctl.action(tip)
    assert a[6] > 0  # commanded to OPEN (positive)


def test_return_controller_budget():
    ctl = PickCubeReturnController(np.array([1.0, 1.0, 0.03]), target_grasped=False, max_steps=4)
    tip = np.zeros(3)
    n = 0
    while not ctl.done(tip):
        ctl.action(tip)
        n += 1
        assert n < 10
    assert ctl.steps == 4  # budget exhausted, not reached


# ---------------------------------------------------------------------------
# Grasp segmentation must NOT depend on the recorded is_grasped column.
#
# Regression: dim 18 is identically 0.0 in every frame of the real PickCube
# bundles, so `states[:, PC_GRASPED] > 0.5` selected nothing and every
# "pre-grasp" bank silently kept the whole episode -- ~84% carry-phase frames.
# The tests above missed it because they SYNTHESIZE states with the flag set.
# These pin the behaviour under the condition the real data actually presents.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

from actsemble.recovery import (  # noqa: E402
    PC_CUBE_REST_Z,
    pickcube_approach_len,
    pickcube_height_bank_report,
    pickcube_lifted,
)


def _episode_with_dead_grasp_flag(T=20, lift_at=6):
    """An episode exactly as the real bundles record it: the cube demonstrably
    leaves the table, but dim 18 is never set."""
    s = np.zeros((T, 42))
    s[:, PC_OBJ_POS.start + 2] = PC_CUBE_REST_Z
    s[lift_at:, PC_OBJ_POS.start + 2] = np.linspace(0.05, 0.25, T - lift_at)
    assert not (s[:, PC_GRASPED] > 0.5).any(), "fixture must leave the flag dead"
    return s


def test_approach_len_ignores_dead_grasp_flag():
    s = _episode_with_dead_grasp_flag(T=20, lift_at=6)
    assert pickcube_approach_len(s) == 6
    # the old flag-based rule degenerates to the whole episode
    g = np.where(s[:, PC_GRASPED.start] > 0.5)[0]
    assert (int(g[0]) if len(g) else len(s)) == 20


def test_lifted_is_per_frame_and_tracks_cube_height():
    s = _episode_with_dead_grasp_flag(T=10, lift_at=4)
    lifted = pickcube_lifted(s)
    assert lifted.shape == (10,)
    assert not lifted[:4].any() and lifted[4:].all()


def test_approach_len_handles_cube_never_lifting():
    s = np.zeros((12, 42))
    s[:, PC_OBJ_POS.start + 2] = PC_CUBE_REST_Z
    assert pickcube_approach_len(s) == 12


def test_support_index_grasped_derived_not_read_from_dead_column():
    """A bank whose cubes are lifted must report grasped candidates even
    though the recorded flag is zero everywhere."""
    s = _episode_with_dead_grasp_flag(T=20, lift_at=6)
    idx = PickCubeSupportIndex(s, np.zeros(20, dtype=int))
    assert idx.grasped[:6].sum() == 0
    assert idx.grasped[6:].all()


def test_height_bank_report_rejects_uncovered_height():
    tcp_z = np.full(400, 0.05)
    cube = np.tile([0.0, 0.0, PC_CUBE_REST_Z], (400, 1))
    ok = pickcube_height_bank_report(tcp_z, cube, [0.05])
    assert ok[0]["n_frames"] == 400
    with pytest.raises(ValueError, match="does not cover"):
        pickcube_height_bank_report(tcp_z, cube, [0.05, 0.30])

"""Failure detection + recovery, v0 (train-nothing) — support statistics and
transport primitives for the E/F track.

Population objects and the full design: docs/recovery_scheme_and_oracle.md.
This module implements the nonparametric estimators (Part I §6.1) and the
return controller (Part I §4.2), shared by the deployable prototype and the
ceiling oracle (scripts/recovery_oracle.py). Diagnostic tier.

Two task instantiations, same kNN-conditional design, different features:

* ``SupportIndex`` / ``ReturnController`` — PushT-v1. State layout (31-dim):
  qpos 0:7 | qvel 7:14 | tcp_pose 14:21 | goal 21:24 | obj_pose 24:31 (pos 3
  + wxyz quat 4). "Environment" = block pose (xy + yaw); "robot given
  environment" = qpos (7-dim, stick end-effector, no gripper).
* ``PickCubeSupportIndex`` / ``PickCubeReturnController`` — PickCube-v1.
  State layout (42-dim, `scripts/failure_taxonomy.py::PickCubeAdapter`
  probed): qpos 0:9 (7 arm + 2 finger) | qvel 9:18 | is_grasped 18:19 |
  tcp_pose 19:26 | goal_pos 26:29 | obj_pose 29:36 | tcp_to_obj 36:39 |
  obj_to_goal 39:42. "Environment" = cube position (xyz; no orientation
  term — top-down grasps on a small cube are not measurably
  orientation-sensitive at this task's scale); "robot given environment" =
  the full 9-dim qpos, so gripper aperture is part of the surprisal (an
  open gripper at the right arm pose is a different situation from a
  closed one). Candidates carry ``is_grasped`` so the return controller
  knows whether to re-close on arrival. Gripper action sign (checked
  against the RL demo bundle): positive = open, negative = close; finger
  qpos ~0.04 open, ~0.019 closed/grasping.
"""

from __future__ import annotations

import numpy as np

QPOS = slice(0, 7)
TCP_POS = slice(14, 17)
OBJ_POS = slice(24, 27)
OBJ_QUAT = slice(27, 31)


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw (rotation about z) of wxyz quaternions [..., 4] -> [...]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class SupportIndex:
    """Nonparametric estimators of the three population objects (v0):

    * ``e`` — environment familiarity: mean distance to the ``k_env`` nearest
      demo BLOCK poses under the environment metric
      ``d_E = ||xy - xy'|| / L + lam * |wrap(yaw - yaw')|``.
    * ``c`` — conditional robot surprisal: distance of the current qpos to the
      demo qpos among the ``m_cond`` nearest-by-``d_E`` demo states.
    * candidate generation — the same neighbors' full joint configurations
      (distinct episodes), i.e. sampling the empirical conditional.

    Thresholds come from ``calibrate`` (percentile of the same statistics on
    the demo states themselves, with same-episode neighbors excluded).
    """

    def __init__(
        self,
        states: np.ndarray,
        episode_index: np.ndarray,
        *,
        k_env: int = 5,
        m_cond: int = 25,
        lam: float | None = None,
    ):
        states = np.asarray(states, dtype=np.float64)
        self.block_xy = states[:, OBJ_POS][:, :2]
        self.block_yaw = quat_yaw(states[:, OBJ_QUAT])
        self.qpos = states[:, QPOS]
        self.tcp_pos = states[:, TCP_POS]
        self.episode_index = np.asarray(episode_index)
        self.k_env = int(k_env)
        self.m_cond = int(m_cond)
        # workspace scale L: diameter of demonstrated block positions
        self.L = float(
            np.linalg.norm(self.block_xy.max(0) - self.block_xy.min(0))
        )
        self.lam = float(self.L / np.pi) if lam is None else float(lam)

    def _d_env(self, xy: np.ndarray, yaw: float, exclude_ep=None) -> np.ndarray:
        """Environment metric (normalized): ||xy-xy'||/L + (lam/L)*|dyaw|."""
        d = np.linalg.norm(self.block_xy - xy, axis=1) / self.L + (
            self.lam / self.L
        ) * np.abs(wrap_angle(self.block_yaw - yaw))
        if exclude_ep is not None:
            d = np.where(self.episode_index == exclude_ep, np.inf, d)
        return d

    def stats(self, state: np.ndarray, *, exclude_ep=None) -> tuple[float, float]:
        """(c, e) for one 31-dim state."""
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        d = self._d_env(s[OBJ_POS][:2], float(quat_yaw(s[OBJ_QUAT])), exclude_ep)
        order = np.argsort(d)
        e = float(d[order[: self.k_env]].mean())
        cond = order[: self.m_cond]
        c = float(np.linalg.norm(self.qpos[cond] - s[QPOS], axis=1).min())
        return c, e

    def calibrate(self, percentile: float = 99.0, subsample: int = 2000, seed: int = 0):
        """(kappa_c, kappa_e): percentile of demo-state statistics with
        same-episode exclusion (else every demo state is its own neighbor)."""
        rng = np.random.default_rng(seed)
        n = self.qpos.shape[0]
        idx = rng.choice(n, size=min(subsample, n), replace=False)
        cs, es = [], []
        for i in idx:
            # reconstruct the fields stats() reads (qpos, block xy, yaw quat)
            state = np.zeros(31)
            state[QPOS] = self.qpos[i]
            state[24:26] = self.block_xy[i]
            yaw = self.block_yaw[i]
            state[27] = np.cos(yaw / 2.0)
            state[30] = np.sin(yaw / 2.0)
            c, e = self.stats(state, exclude_ep=self.episode_index[i])
            cs.append(c)
            es.append(e)
        return float(np.percentile(cs, percentile)), float(np.percentile(es, percentile))

    def candidates(self, state: np.ndarray, m: int = 3):
        """m nearest-by-d_E demo states from DISTINCT episodes ->
        list of (qpos[7], tcp_pos[3])."""
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        d = self._d_env(s[OBJ_POS][:2], float(quat_yaw(s[OBJ_QUAT])))
        out, seen = [], set()
        for i in np.argsort(d):
            ep = self.episode_index[i]
            if ep in seen:
                continue
            seen.add(ep)
            out.append((self.qpos[i].copy(), self.tcp_pos[i].copy()))
            if len(out) == m:
                break
        return out


class ReturnController:
    """Scripted collision-aware return in delta-EE space (Part I §4.2):
    lift -> translate above the target -> descend. Emits normalized
    ``pd_ee_delta_pose`` actions (6-dim: dpos then drot, drot = 0)."""

    def __init__(
        self,
        target_tip: np.ndarray,
        *,
        lift_dz: float = 0.08,
        pos_scale: float = 0.1,
        tol: float = 0.01,
        max_steps: int = 16,
    ):
        self.target = np.asarray(target_tip, dtype=np.float64).reshape(3)
        self.lift_dz = float(lift_dz)
        self.pos_scale = float(pos_scale)
        self.tol = float(tol)
        self.max_steps = int(max_steps)
        self.steps = 0
        self._z_safe: float | None = None

    def done(self, tip: np.ndarray) -> bool:
        return (
            self.steps >= self.max_steps
            or float(np.linalg.norm(np.asarray(tip).reshape(3) - self.target)) < self.tol
        )

    def action(self, tip: np.ndarray) -> np.ndarray:
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
        if self._z_safe is None:
            self._z_safe = max(tip[2], self.target[2]) + self.lift_dz
        wp = np.array(self.target)
        if np.linalg.norm(tip[:2] - self.target[:2]) > self.tol:
            # not yet above the target: lift first, then translate at height
            if tip[2] < self._z_safe - self.tol:
                wp = np.array([tip[0], tip[1], self._z_safe])
            else:
                wp = np.array([self.target[0], self.target[1], self._z_safe])
        delta = np.clip((wp - tip) / self.pos_scale, -1.0, 1.0)
        self.steps += 1
        return np.concatenate([delta, np.zeros(3)]).astype(np.float32)


# =============================================================================
# PickCube-v1
# =============================================================================
PC_QPOS = slice(0, 9)         # 7 arm joints + 2 finger joints
PC_GRIP_QPOS = slice(7, 9)    # finger joints only
PC_GRASPED = slice(18, 19)    # DEAD COLUMN in the recorded bundles -- see below
PC_TCP_POS = slice(19, 22)
PC_GOAL_POS = slice(26, 29)
PC_OBJ_POS = slice(29, 32)
PC_GRIP_OPEN_QPOS = 0.04      # finger joint value, fully open
PC_GRIP_CLOSED_QPOS = 0.019   # finger joint value, closed on/near an object

# Cube centre height, metres. Resting on the table the cube centre sits at
# PC_CUBE_REST_Z; PC_LIFT_Z clears settling jitter without waiting for a full
# lift, so `cube z > PC_LIFT_Z` means "the cube has left the table".
PC_CUBE_REST_Z = 0.020
PC_LIFT_Z = 0.035


def pickcube_lifted(states: np.ndarray) -> np.ndarray:
    """Per-frame "the cube is off the table", derived from cube HEIGHT.

    Why not the recorded ``is_grasped`` column: in the PickCube bundles we
    train on, dim 18 is identically 0.0 across every frame of every episode
    (measured: 0 nonzero of 4967 frames, 100/100 episodes), even though the
    cube demonstrably reaches z=0.319 -- so the demos do grasp, the flag just
    was never populated. Any code that segmented "pre-grasp" with that column
    therefore kept the WHOLE episode, leaving a bank that is ~84% carry-phase
    frames (cube already in the gripper, in the air). Deriving the signal from
    geometry is robust to that column being absent, dead, or renamed.
    """
    return np.asarray(states, dtype=np.float64)[:, PC_OBJ_POS][:, 2] > PC_LIFT_Z


def pickcube_approach_len(states: np.ndarray) -> int:
    """Number of leading frames of an episode that are APPROACH (cube still on
    the table). Returns len(states) if the cube never leaves the table.

    This is the correct upper bound for a "reset target" bank: a carry-phase
    config holds fingers closed around a cube that, in the scene being
    recovered, is still on the table -- resetting to one commands a pose whose
    whole point is an object that is not there.
    """
    lifted = np.where(pickcube_lifted(states))[0]
    return int(lifted[0]) if len(lifted) else len(states)


def pickcube_height_bank_report(tcp_z, cube_xyz, heights, *, tol=0.025, min_frames=50):
    """Coverage of a reset-target bank at each requested height.

    Returns one row per height with the frame count and the WORST-CASE best
    achievable cube match (metres) over a grid of plausible cube positions.
    A height whose bin is empty, or whose best match is far, cannot support a
    reset target: retrieval will silently hand back a config belonging to a
    scene with the cube somewhere else entirely. Raises if any height falls
    under ``min_frames`` so a sweep cannot quietly report a number for a
    height its data does not cover.
    """
    tcp_z = np.asarray(tcp_z); cube_xyz = np.asarray(cube_xyz)
    probes = np.array([[cx, cy, PC_CUBE_REST_Z]
                       for cx in np.linspace(-0.1, 0.1, 7)
                       for cy in np.linspace(-0.1, 0.1, 7)])
    rows, thin = [], []
    for h in heights:
        m = np.abs(tcp_z - h) < tol
        n = int(m.sum())
        if n:
            sub = cube_xyz[m]
            worst = float(max(np.linalg.norm(sub - p, axis=1).min() for p in probes))
        else:
            worst = float("inf")
        rows.append({"height": float(h), "n_frames": n, "worst_cube_match": worst})
        if n < min_frames:
            thin.append(f"z={h:.3f} has {n} frames (need >= {min_frames})")
    if thin:
        raise ValueError(
            "reset-target bank does not cover these heights: " + "; ".join(thin)
            + ". The demo approach phase spans a limited height range -- sweeping "
              "above it retrieves carry-phase configs matched to a distant cube."
        )
    return rows


class PickCubeSupportIndex:
    """PickCube analog of ``SupportIndex``: same kNN-conditional design,
    different features.

    * ``e`` — environment familiarity: mean distance to the ``k_env``
      nearest demo CUBE positions (xyz, no orientation term).
    * ``c`` — conditional robot surprisal: distance of the current 9-dim
      qpos (arm + gripper) to the demo qpos among the ``m_cond``
      nearest-by-cube-position demo states — so a correctly-positioned but
      wrongly-open/closed gripper registers as surprising.
    * candidates — the same neighbors' (qpos[9], tcp_pos[3], grasped[bool])
      from distinct episodes.
    """

    def __init__(
        self,
        states: np.ndarray,
        episode_index: np.ndarray,
        *,
        k_env: int = 5,
        m_cond: int = 25,
    ):
        states = np.asarray(states, dtype=np.float64)
        self.obj_xyz = states[:, PC_OBJ_POS]
        self.qpos = states[:, PC_QPOS]
        self.tcp_pos = states[:, PC_TCP_POS]
        # NOT states[:, PC_GRASPED] -- that column is identically zero in the
        # recorded bundles, which silently labelled every candidate "not
        # grasped" (and therefore had every scripted return command OPEN jaws,
        # including at carry-phase targets). See pickcube_lifted().
        self.grasped = pickcube_lifted(states)
        self.episode_index = np.asarray(episode_index)
        self.k_env = int(k_env)
        self.m_cond = int(m_cond)
        self.L = float(
            np.linalg.norm(self.obj_xyz.max(0) - self.obj_xyz.min(0))
        ) or 1.0

    def _d_env(self, xyz: np.ndarray, exclude_ep=None) -> np.ndarray:
        d = np.linalg.norm(self.obj_xyz - xyz, axis=1) / self.L
        if exclude_ep is not None:
            d = np.where(self.episode_index == exclude_ep, np.inf, d)
        return d

    def stats(self, state: np.ndarray, *, exclude_ep=None) -> tuple[float, float]:
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        d = self._d_env(s[PC_OBJ_POS], exclude_ep)
        order = np.argsort(d)
        e = float(d[order[: self.k_env]].mean())
        cond = order[: self.m_cond]
        c = float(np.linalg.norm(self.qpos[cond] - s[PC_QPOS], axis=1).min())
        return c, e

    def calibrate(self, percentile: float = 99.0, subsample: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = self.qpos.shape[0]
        idx = rng.choice(n, size=min(subsample, n), replace=False)
        cs, es = [], []
        for i in idx:
            state = np.zeros(42)
            state[PC_QPOS] = self.qpos[i]
            state[PC_OBJ_POS] = self.obj_xyz[i]
            c, e = self.stats(state, exclude_ep=self.episode_index[i])
            cs.append(c)
            es.append(e)
        return float(np.percentile(cs, percentile)), float(np.percentile(es, percentile))

    def candidates(self, state: np.ndarray, m: int = 3):
        """m nearest-by-cube-position demo states from DISTINCT episodes ->
        list of (qpos[9], tcp_pos[3], grasped[bool])."""
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        d = self._d_env(s[PC_OBJ_POS])
        out, seen = [], set()
        for i in np.argsort(d):
            ep = self.episode_index[i]
            if ep in seen:
                continue
            seen.add(ep)
            out.append((self.qpos[i].copy(), self.tcp_pos[i].copy(), bool(self.grasped[i])))
            if len(out) == m:
                break
        return out


class PickCubeReturnController:
    """Scripted collision-aware return for PickCube (Part I §4.2 analog):
    lift -> translate above the target -> descend -> set gripper to the
    candidate's grasp state and hold. Emits normalized ``pd_ee_delta_pose``
    actions (7-dim: dpos 3, drot 3 = 0, gripper 1)."""

    def __init__(
        self,
        target_tip: np.ndarray,
        target_grasped: bool,
        *,
        lift_dz: float = 0.08,
        pos_scale: float = 0.1,
        tol: float = 0.01,
        grip_hold_steps: int = 4,
        max_steps: int = 20,
    ):
        self.target = np.asarray(target_tip, dtype=np.float64).reshape(3)
        self.target_grasped = bool(target_grasped)
        self.lift_dz = float(lift_dz)
        self.pos_scale = float(pos_scale)
        self.tol = float(tol)
        self.grip_hold_steps = int(grip_hold_steps)
        self.max_steps = int(max_steps)
        self.steps = 0
        self._grip_steps = 0
        self._z_safe: float | None = None

    def done(self, tip: np.ndarray) -> bool:
        at_target = float(np.linalg.norm(np.asarray(tip).reshape(3) - self.target)) < self.tol
        return (
            self.steps >= self.max_steps
            or (at_target and self._grip_steps >= self.grip_hold_steps)
        )

    def action(self, tip: np.ndarray) -> np.ndarray:
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
        if self._z_safe is None:
            self._z_safe = max(tip[2], self.target[2]) + self.lift_dz
        at_target_xy = np.linalg.norm(tip[:2] - self.target[:2]) <= self.tol
        at_target = at_target_xy and abs(tip[2] - self.target[2]) <= self.tol
        wp = np.array(self.target)
        if not at_target_xy:
            # not yet above the target: lift first, then translate at height
            if tip[2] < self._z_safe - self.tol:
                wp = np.array([tip[0], tip[1], self._z_safe])
            else:
                wp = np.array([self.target[0], self.target[1], self._z_safe])
        delta = np.clip((wp - tip) / self.pos_scale, -1.0, 1.0)
        grip = 0.0
        if at_target:
            self._grip_steps += 1
            grip = -1.0 if self.target_grasped else 1.0
        self.steps += 1
        return np.concatenate([delta, np.zeros(3), [grip]]).astype(np.float32)

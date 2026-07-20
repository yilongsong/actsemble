"""Failure detection + recovery, v0 (train-nothing) — support statistics and
transport primitives for the E/F track.

Population objects and the full design: docs/recovery_scheme_and_oracle.md.
This module implements the nonparametric estimators (Part I §6.1) and the
return controller (Part I §4.2), shared by the deployable prototype and the
ceiling oracle (scripts/recovery_oracle.py). Diagnostic tier.

State layout (31-dim): qpos 0:7 | qvel 7:14 | tcp_pose 14:21 | goal 21:24 |
obj_pose 24:31 (pos 3 + wxyz quat 4).
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

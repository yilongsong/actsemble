#!/usr/bin/env python
"""Does a PERFECTLY specified recovery input remove the U-shape?

The policy's input is obs_horizon=2 FULL frames. "Perfect injection" therefore
means both frames correct, not just the current one's velocity. Three arms per
reset height, so the U-shape can be decomposed:

  (a) zero-vel       : teleport qpos, velocity zeroed, history = duplicated frame
                       (what produced the original U-shape / dead zone)
  (b) matched-vel    : teleport qpos+qvel, history STILL a duplicated frame
                       (half-measure: current frame right, previous frame a lie)
  (c) perfect        : teleport qpos+qvel AND inject the demo's genuine previous
                       frame (scene dims spliced to the CURRENT scene so the pair
                       is internally consistent) -> the policy's whole input is
                       coherent and in-distribution

Prediction under test (user): with (c) the dead zone vanishes and success becomes
MONOTONIC in distance-from-grasp (closer = better, less task left to fail).

    python scripts/recovery_perfect_injection.py --n 60 --k 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from recovery_oracle_pickcube import MAX_STEPS, STANDALONE, Runner, _success  # noqa: E402

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.panels import make_panel, panel_episodes  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import (  # noqa: E402
    PC_OBJ_POS,
    PC_QPOS,
    PC_TCP_POS,
    pickcube_approach_len,
    pickcube_height_bank_report,
)
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402

HEIGHTS = [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.19]
GOAL = slice(26, 29); OBJ = slice(29, 36); T2O = slice(36, 39); O2G = slice(39, 42)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "outputs/pickcube/recovery_oracle/perfect_injection.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # flat bank of demo PRE-GRASP frames with episode/frame indexing
    reader = DatasetReader(args.dataset)
    S, EP, FR = [], [], []
    for j, eid in enumerate(reader.episode_ids):
        s = reader.episode(eid).state
        gf = pickcube_approach_len(s)
        S.append(s[:gf]); EP.append(np.full(gf, j)); FR.append(np.arange(gf))
    S = np.concatenate(S); EP = np.concatenate(EP); FR = np.concatenate(FR)
    CUBE = S[:, PC_OBJ_POS]; TZ = S[:, PC_TCP_POS][:, 2]
    print("[bank] reset-target coverage (approach frames only):")
    for r in pickcube_height_bank_report(TZ, CUBE, HEIGHTS):
        print(f"  z={r['height']:.2f}  {r['n_frames']:5d} frames  worst cube-match "
              f"{100 * r['worst_cube_match']:.1f} cm", flush=True)

    def retrieve(cube, h, tol=0.025):
        m = np.abs(TZ - h) < tol
        if not m.any():
            m = np.abs(TZ - h) < 0.05
        idx = np.where(m)[0]
        return int(idx[int(np.argmin(np.linalg.norm(CUBE[idx] - cube, axis=1)))])

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode=None, max_episode_steps=MAX_STEPS)
    rn = Runner(env, build_system(STANDALONE, policy, []))

    def place(qpos9, qvel9):
        t = rn.u.get_state_dict(); art = t["articulations"][next(iter(t["articulations"]))]
        art[..., -18:-9] = torch.as_tensor(qpos9, dtype=art.dtype, device=art.device)
        art[..., -9:] = torch.as_tensor(qvel9, dtype=art.dtype, device=art.device)
        rn.u.set_state_dict(t)

    def coherent_prev(j, cur_frame):
        """Demo frame j-1 (same episode) with SCENE dims spliced from the current
        scene, so the injected (prev, current) pair is internally consistent."""
        p = j - 1 if (j - 1 >= 0 and EP[j - 1] == EP[j]) else j
        f = S[p].copy()
        f[GOAL] = cur_frame[GOAL]; f[OBJ] = cur_frame[OBJ]
        # SIGN CONVENTION, verified against the dataset (residual exactly 0.0):
        #   dim 36:39 tcp_to_obj = obj - tcp      (NOT tcp - obj)
        #   dim 39:42 obj_to_goal = goal - obj    (NOT obj - goal)
        # These are the dims encoding "which way must I move"; negating them
        # points the policy at the mirror image of the target.
        f[T2O] = f[OBJ][:3] - f[PC_TCP_POS]
        f[O2G] = f[GOAL] - f[OBJ][:3]
        return f

    def roll(pe, j, arm):
        rn.start(pe.policy_sampling_seed); raw, _ = env.reset(seed=pe.env_seed)
        cur = rn.state42(raw).copy()
        qpos, qvel = S[j][PC_QPOS], S[j][9:18]
        place(qpos, np.zeros(9) if arm == "a_zero_vel" else qvel)
        raw = rn.u.get_obs()
        frames = [coherent_prev(j, cur)] if arm == "c_perfect" else []
        rn.start(pe.policy_sampling_seed + 11, frames=frames)
        once = False
        for _ in range(MAX_STEPS):
            raw, _r, _te, _tr, info = rn.act_env(raw); once = once or _success(info)
        return once

    scenes = list(panel_episodes(make_panel("diagnostic")))[: args.n]
    ARMS = ["a_zero_vel", "b_matched_vel", "c_perfect"]
    rows = []
    for i, pe in enumerate(scenes):
        raw, _ = env.reset(seed=pe.env_seed)
        cube = rn.state42(raw)[PC_OBJ_POS].copy()
        for h in HEIGHTS:
            j = retrieve(cube, h)
            for arm in ARMS:
                wins = sum(roll(pe, j, arm) for _ in range(args.k))
                rows.append({"env_seed": pe.env_seed, "height": h, "arm": arm,
                             "actual_z": float(TZ[j]), "recovery": wins / args.k})
        if (i + 1) % 15 == 0:
            print(f"[perfect] scene {i + 1}/{len(scenes)}", flush=True)
    env.close()

    curves = {a: [{"height": h,
                   "recovery": float(np.mean([r["recovery"] for r in rows
                                              if r["height"] == h and r["arm"] == a]))}
                  for h in HEIGHTS] for a in ARMS}
    json.dump({"curves": curves, "rows": rows, "k": args.k, "n_scenes": len(scenes)},
              open(args.out, "w"), indent=1)
    print(f"\nRECOVERY vs RESET HEIGHT by input completeness "
          f"({len(scenes)} scenes x K={args.k}):")
    print(f"  {'height':>8}" + "".join(f"{a:>16}" for a in ARMS))
    for hi, h in enumerate(HEIGHTS):
        print(f"  {h:>8.2f}" + "".join(f"{curves[a][hi]['recovery']:>15.0%} " for a in ARMS))
    for a in ARMS:
        v = [c["recovery"] for c in curves[a]]
        mono = all(v[i] >= v[i + 1] - 1e-9 for i in range(len(v) - 1))
        print(f"  {a:<16} range {min(v):.0%}-{max(v):.0%}  monotonic-decreasing: {mono}")
    print(f"[perfect] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

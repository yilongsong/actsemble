#!/usr/bin/env python
"""HOW HIGH should pick-and-place recovery reset to? Simplest form.

Recovery = place the robot at an in-distribution config for the current cube.
There is a whole range of such configs along the demo approach (different
fingertip heights). Which height gives the policy the best success rate? Just
measure it: over a set of scenes, for each height h, teleport the robot to the
nearest-cube demo config at height h and roll the policy. Success-rate vs height.

    python scripts/recovery_height_ablation.py --n 100 --k 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from recovery_oracle_pickcube import MAX_STEPS, STANDALONE, Runner, _success, teleport_robot  # noqa: E402

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.panels import make_panel, panel_episodes  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import (  # noqa: E402
    PC_OBJ_POS,
    PC_QPOS,
    PC_TCP_POS,
    pickcube_approach_len,
)
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402

# fingertip z above table (m). REFERENCE: table=0.00, cube top=0.04,
# robot HOME=0.168, demo pre-grasp reaches 0.316 -> must cover the genuinely
# retracted range, not stop at 0.15 (which is only the median approach height).
HEIGHTS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.19]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--n", type=int, default=100, help="scenes from the diagnostic panel")
    ap.add_argument("--k", type=int, default=6, help="policy rollouts per (scene,height)")
    ap.add_argument("--out", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_height_ablation.json"))
    ap.add_argument("--velocity", default="zero", choices=["zero","demo"],
                    help="joint velocity at the reset: zeroed (teleport default) or the demo's actual velocity at that frame")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # demo pre-grasp frame bank: qpos, cube xyz, tcp z
    reader = DatasetReader(args.dataset)
    Q, QV, CUBE, TZ = [], [], [], []
    for eid in reader.episode_ids:
        s = reader.episode(eid).state
        s = s[: pickcube_approach_len(s)]
        Q.append(s[:, PC_QPOS]); QV.append(s[:, 9:18])
        CUBE.append(s[:, PC_OBJ_POS]); TZ.append(s[:, PC_TCP_POS][:, 2])
    Q = np.concatenate(Q); QV = np.concatenate(QV); CUBE = np.concatenate(CUBE); TZ = np.concatenate(TZ)

    def retrieve(cube_xyz, h, tol=0.02):
        m = np.abs(TZ - h) < tol
        if not m.any():
            m = np.abs(TZ - h) < 0.04
        idx = np.where(m)[0]
        j = idx[int(np.argmin(np.linalg.norm(CUBE[idx] - cube_xyz, axis=1)))]
        return Q[j].copy(), QV[j].copy(), float(np.linalg.norm(CUBE[j] - cube_xyz)), float(TZ[j])

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode=None, max_episode_steps=MAX_STEPS)
    rn = Runner(env, build_system(STANDALONE, policy, []))
    scenes = list(panel_episodes(make_panel("diagnostic")))[: args.n]

    import torch

    def place(qpos9, qvel9):
        """Teleport robot to qpos with the GIVEN joint velocity (zero, or the
        demo's velocity at that frame). Cube untouched."""
        t = rn.u.get_state_dict()
        art = t["articulations"][next(iter(t["articulations"]))]
        art[..., -18:-9] = torch.as_tensor(qpos9, dtype=art.dtype, device=art.device)
        art[..., -9:] = torch.as_tensor(qvel9, dtype=art.dtype, device=art.device)
        rn.u.set_state_dict(t)

    def place_and_roll(pe, qpos, qvel, tag):
        rn.start(pe.policy_sampling_seed); raw, _ = env.reset(seed=pe.env_seed)
        place(qpos, qvel)
        raw = rn.u.get_obs(); rn.start(pe.policy_sampling_seed + tag); once = False
        for _ in range(MAX_STEPS):
            raw, _r, _te, _tr, info = rn.act_env(raw); once = once or _success(info)
        return once

    rows = []
    for i, pe in enumerate(scenes):
        rn.start(pe.policy_sampling_seed); raw, _ = env.reset(seed=pe.env_seed)
        cube = rn.state42(raw)[PC_OBJ_POS].copy()
        for hi, h in enumerate(HEIGHTS):
            q, qv, match, actual_z = retrieve(cube, h)
            vel = qv if args.velocity == "demo" else np.zeros(9)
            wins = sum(place_and_roll(pe, q, vel, 9000 + hi * 100 + k) for k in range(args.k))
            rows.append({"env_seed": pe.env_seed, "height": h, "actual_z": actual_z,
                         "cube_match": match, "demo_qvel": float(np.linalg.norm(qv)),
                         "recovery": wins / args.k})
        if (i + 1) % 20 == 0:
            print(f"[height] scene {i + 1}/{len(scenes)}", flush=True)
    env.close()

    curve = []
    for h in HEIGHTS:
        hr = [r for r in rows if r["height"] == h]
        curve.append({"height": h, "n": len(hr),
                      "recovery": float(np.mean([r["recovery"] for r in hr])),
                      "demo_qvel_median": float(np.median([r["demo_qvel"] for r in hr])),
                      "actual_z_median": float(np.median([r["actual_z"] for r in hr])),
                      "cube_match_median": float(np.median([r["cube_match"] for r in hr]))})
    json.dump({"curve": curve, "rows": rows, "k": args.k, "n_scenes": len(scenes), "velocity": args.velocity},
              open(args.out, "w"), indent=1)
    print(f"\nPOLICY SUCCESS vs RESET HEIGHT  ({len(scenes)} scenes x K={args.k}, velocity={args.velocity}):")
    print(f"  {'height':>8}{'actual_z':>9}{'cube_match':>11}{'success':>9}")
    for c in curve:
        print(f"  {c['height']:>8.2f}{c['actual_z_median']:>9.3f}{c['cube_match_median']:>10.3f} {c['recovery']:>8.0%}")
    best = max(curve, key=lambda c: c["recovery"])
    print(f"\n  BEST height {best['height']:.2f}m -> {best['recovery']:.0%}   (grasp-level {curve[0]['recovery']:.0%})")
    print(f"[height] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

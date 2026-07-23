#!/usr/bin/env python
"""WHICH reset state to choose -- measured with PHYSICAL DRIVING, not teleport.

Teleport structurally hides the cost of near-grasp targets: it places the robot
at the instant before a successful grasp with no approach, so no contact risk.
The real tradeoff only exists when you physically drive to the target:

  near-grasp target  -> little task left, BUT the drive must penetrate next to
                        the cube -> contact risk (may knock it).
  retracted target   -> drive stays clear, BUT more approach left to re-execute
                        -> more chance to re-fail, more time spent.

Protocol per failure scene: roll to a mid-failure boundary, snapshot, then for
each candidate reset height h: joint-space DRIVE (pd_joint_pos, jaws open) to the
cube-matched demo config at h, settle, hand back to the policy. Records
end-to-end success, cube disturbance DURING THE DRIVE (the collision cost), and
drive length (the time cost). Optionally injects the demo velocity into the
observation at handback (the hybrid: drive to pose, then inject the rest).

    python scripts/recovery_drive_height.py --n 40 --k 4
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

from recovery_oracle_pickcube import (  # noqa: E402
    MAX_STEPS,
    STANDALONE,
    Runner,
    _success,
    restore_state,
    snap_state,
    teleport_robot,
)

from actsemble.data.reader import DatasetReader  # noqa: E402
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
BOUNDARY = 8          # roll this far into the failure before recovering
OPEN = 1.0            # normalized gripper action = fully open
DRIVE_MAX = 22
SETTLE = 3


def load_failures(path, n):
    """Failure scenes from EITHER a recovery-oracle JSON or a plain evaluation
    JSON (final_test.json / screening step_*.json). The oracle format is tied to
    whichever policy produced it, so accepting an evaluation lets this sweep run
    against any freshly trained policy without first building an oracle."""
    d = json.load(open(path))
    if "reference" in d:
        rows = [(e["env_seed"], e["seed"]) for e in d["reference"] if not e["success"]]
    else:
        eps = d.get("episodes") or []
        rows = [(e["env_seed"], e["policy_sampling_seed"])
                for e in eps if not e.get("success_once", False)]
    if not rows:
        raise SystemExit(f"no failure episodes found in {path}")
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--oracle-json", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_oracle.json"))
    ap.add_argument("--n", type=int, default=40, help="failure scenes")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-teleport", action="store_true",
                    help="skip the non-physical teleport (oracle-ceiling) arm")
    ap.add_argument("--inject-velocity", action="store_true",
                    help="hybrid: after driving+settling, inject the demo velocity into the observation")
    ap.add_argument("--out", default=str(REPO / "outputs/pickcube/recovery_oracle/drive_height.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    fails = load_failures(args.oracle_json, args.n)
    print(f"[drive] {len(fails)} failure scenes from {args.oracle_json}", flush=True)

    reader = DatasetReader(args.dataset)
    Q, QV, CUBE, TZ = [], [], [], []
    for eid in reader.episode_ids:
        s = reader.episode(eid).state
        s = s[: pickcube_approach_len(s)]
        Q.append(s[:, PC_QPOS]); QV.append(s[:, 9:18])
        CUBE.append(s[:, PC_OBJ_POS]); TZ.append(s[:, PC_TCP_POS][:, 2])
    Q = np.concatenate(Q); QV = np.concatenate(QV); CUBE = np.concatenate(CUBE); TZ = np.concatenate(TZ)
    print("[bank] reset-target coverage (approach frames only):")
    for r in pickcube_height_bank_report(TZ, CUBE, HEIGHTS):
        print(f"  z={r['height']:.2f}  {r['n_frames']:5d} frames  worst cube-match "
              f"{100 * r['worst_cube_match']:.1f} cm", flush=True)

    def retrieve(cube, h, tol=0.025):
        m = np.abs(TZ - h) < tol
        if not m.any():
            m = np.abs(TZ - h) < 0.05
        idx = np.where(m)[0]
        j = idx[int(np.argmin(np.linalg.norm(CUBE[idx] - cube, axis=1)))]
        return Q[j].copy(), QV[j].copy()

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode=None, max_episode_steps=MAX_STEPS)
    rn = Runner(env, build_system(STANDALONE, policy, []))
    agent = rn.u.agent

    def drive_to(qtarget, cube0):
        """Physical joint-space drive with jaws open. Returns (steps, max cube
        displacement caused BY THE DRIVE, reached_config)."""
        agent.set_control_mode("pd_joint_pos"); agent.controller.reset()
        act = torch.tensor(np.concatenate([qtarget[:7], [OPEN]])[None],
                           dtype=torch.float32, device=agent.robot.device)
        steps, bump, once = 0, 0.0, False
        while steps < DRIVE_MAX:
            _, _r, _te, _tr, info = env.step(act); steps += 1
            once = once or _success(info)
            s = rn.state42(rn.u.get_obs())
            bump = max(bump, float(np.linalg.norm(s[PC_OBJ_POS] - cube0)))
            if np.linalg.norm(s[:7] - qtarget[:7]) < 0.02:
                break
        for _ in range(SETTLE):
            env.step(act); steps += 1
            s = rn.state42(rn.u.get_obs())
            bump = max(bump, float(np.linalg.norm(s[PC_OBJ_POS] - cube0)))
        err = float(np.linalg.norm(rn.state42(rn.u.get_obs())[:7] - qtarget[:7]))
        agent.set_control_mode("pd_ee_delta_pose"); agent.controller.reset()
        return steps, bump, err, once

    def handback(pe_seed, tag, inject_vel, budget):
        raw = rn.u.get_obs(); rn.start(pe_seed + tag); once = False
        for st in range(budget):
            r2 = raw
            if inject_vel is not None and st < 4:
                r2 = raw.clone(); r2[0, 9:18] = torch.as_tensor(inject_vel, dtype=r2.dtype, device=r2.device)
            obs = rn.obs_adapter.observe(r2); a = rn.system.act(obs)
            a = np.clip(np.asarray(a.value, np.float32).reshape(-1), rn.low, rn.high)
            raw, _r, _te, _tr, info = env.step(torch.as_tensor(a.reshape(1, -1)))
            once = once or _success(info)
        return once

    rows = []
    for i, (es, ps) in enumerate(fails):
        # roll into the failure, snapshot the mid-failure state
        rn.start(ps); raw, _ = env.reset(seed=es)
        for _ in range(BOUNDARY): raw, *_ = rn.act_env(raw)
        snap = snap_state(rn.u)
        cube0 = rn.state42(raw)[PC_OBJ_POS].copy()
        # no-recovery control: let the failure play out
        base = 0
        for k in range(args.k):
            restore_state(env, rn.u, snap); rn.start(ps + 50 + k)
            base += handback(ps, 50 + k, None, MAX_STEPS - BOUNDARY)
        rows.append({"env_seed": es, "height": None, "arm": "no_recovery",
                     "success": base / args.k, "drive_bump": 0.0, "drive_steps": 0, "cfg_err": 0.0})
        for hi, h in enumerate(HEIGHTS):
            q, qv = retrieve(cube0, h)
            # ORACLE arm: teleport the robot to the target. Non-physical -- it
            # cannot knock the cube and costs no steps -- so it is the CEILING
            # for this reset height, not an achievable result. Run from the same
            # snapshot as the drive arm, so the gap between them IS the cost of
            # physically getting there.
            if not args.no_teleport:
                twins = 0
                for k in range(args.k):
                    restore_state(env, rn.u, snap)
                    teleport_robot(rn.u, q)
                    twins += handback(ps, 700 + hi * 50 + k, None, MAX_STEPS - BOUNDARY)
                rows.append({"env_seed": es, "height": h, "arm": "teleport",
                             "success": twins / args.k, "drive_bump": 0.0,
                             "drive_steps": 0, "cfg_err": 0.0})
            wins, bumps, stepss, errs = 0, [], [], []
            for k in range(args.k):
                restore_state(env, rn.u, snap)
                st, bump, err, once = drive_to(q, cube0)
                budget = max(MAX_STEPS - BOUNDARY - st, 0)
                ok = once or (handback(ps, 900 + hi * 50 + k, qv if args.inject_velocity else None, budget)
                              if budget > 0 else False)
                wins += ok; bumps.append(bump); stepss.append(st); errs.append(err)
            rows.append({"env_seed": es, "height": h, "arm": "drive",
                         "success": wins / args.k, "drive_bump": float(np.mean(bumps)),
                         "drive_steps": float(np.mean(stepss)), "cfg_err": float(np.mean(errs))})
        if (i + 1) % 10 == 0:
            print(f"[drive] {i + 1}/{len(fails)}", flush=True)
    env.close()

    base_rows = [r for r in rows if r["arm"] == "no_recovery"]
    curve = [{"height": h,
              "success": float(np.mean([r["success"] for r in rows
                                        if r["height"] == h and r["arm"] == "drive"])),
              "teleport": (float(np.mean([r["success"] for r in rows
                                          if r["height"] == h and r["arm"] == "teleport"]))
                           if any(r["arm"] == "teleport" for r in rows) else None),
              "drive_bump": float(np.mean([r["drive_bump"] for r in rows if r["height"] == h])),
              "drive_steps": float(np.mean([r["drive_steps"] for r in rows if r["height"] == h])),
              "cfg_err": float(np.mean([r["cfg_err"] for r in rows if r["height"] == h]))}
             for h in HEIGHTS]
    base = float(np.mean([r["success"] for r in base_rows]))
    json.dump({"curve": curve, "no_recovery": base, "rows": rows, "k": args.k,
               "inject_velocity": args.inject_velocity, "n_failures": len(fails)},
              open(args.out, "w"), indent=1)
    print(f"\nPHYSICAL-DRIVE recovery vs reset height  ({len(fails)} failures x K={args.k}"
          f"{', velocity injected at handback' if args.inject_velocity else ''}):")
    print(f"  no recovery (let failure play out): {base:.0%}")
    print(f"  {'height':>8}{'ORACLE':>9}{'DRIVEN':>9}{'gap':>7}"
          f"{'cube bump (drive)':>19}{'drive steps':>13}{'cfg err':>9}")
    for c in curve:
        tp = c.get("teleport")
        tps = f"{tp:>9.0%}" if tp is not None else f"{'--':>9}"
        gap = f"{tp - c['success']:>+7.0%}" if tp is not None else f"{'--':>7}"
        print(f"  {c['height']:>8.2f}{tps}{c['success']:>9.0%}{gap}"
              f"{c['drive_bump']:>19.3f}{c['drive_steps']:>13.1f}{c['cfg_err']:>9.3f}")
    print("  ORACLE = teleport (non-physical ceiling); DRIVEN = physically reachable;")
    print("  gap    = what it costs to actually get there")
    best = max(curve, key=lambda c: c["success"])
    print(f"\n  BEST reset height {best['height']:.2f}m -> {best['success']:.0%} "
          f"(bump {best['drive_bump']:.3f}, {best['drive_steps']:.0f} drive steps)")
    print(f"[drive] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

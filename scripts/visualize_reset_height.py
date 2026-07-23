#!/usr/bin/env python
"""What each reset height LOOKS like when you physically drive to it.

The height sweep gives a number per height; this gives the footage behind it.
One row per arm on the same failure, fired at the same boundary:

    NO-RECOVERY  |  drive to h1  |  drive to h2  |  ...

Every recovery row switches to pd_joint_pos with the jaws FULLY OPEN (normalized
action +1.0 -- passing 0.04 gives semi-closed jaws at 0.016 m and rams the cube),
PD-drives to the cube-matched demo config at that height, switches back, and hands
to the policy. The overlay carries the two costs the success number alone hides:

    bump   cube displacement accumulated DURING the drive (the collision cost;
           near-grasp targets have to penetrate next to the cube to arrive)
    steps  control steps the drive consumed out of the episode budget

so a row that succeeds cheaply and a row that succeeds after shoving the cube
15 cm are visibly different events.

    python scripts/visualize_reset_height.py --seeds 84886883,1611412778
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
)
from visualize_recovery_pickcube import annotate, filmstrip, grab  # noqa: E402

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import (  # noqa: E402
    PC_OBJ_POS,
    PC_QPOS,
    PC_TCP_POS,
    PickCubeSupportIndex,
    pickcube_approach_len,
)
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402

BOUNDARY = 8            # match recovery_drive_height.py
OPEN = 1.0              # normalized pd_joint_pos gripper action = fully open
DRIVE_MAX = 22
SETTLE = 3
# at-grasp / mid / high / top-of-approach (approach data ends at z=0.209;
# above ~0.19 the bank thins out -- see pickcube_height_bank_report)
DEFAULT_HEIGHTS = "0.02,0.08,0.14,0.19"
BORDER = [(120, 210, 255), (235, 160, 60), (120, 220, 140), (200, 120, 235)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--oracle-json", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_oracle.json"))
    ap.add_argument("--seeds", default="", help="env seeds; default = first failures in the oracle")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--heights", default=DEFAULT_HEIGHTS)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/pickcube/recovery_oracle/reset_height_viz"))
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    heights = [float(h) for h in args.heights.split(",")]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    oracle = json.load(open(args.oracle_json))
    seed_of = {e["env_seed"]: e["seed"] for e in oracle["reference"]}
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [e["env_seed"] for e in oracle["reference"] if not e["success"]][: args.n_seeds]

    # demo pre-grasp bank
    reader = DatasetReader(args.dataset)
    Q, CUBE, TZ, states, ep_idx = [], [], [], [], []
    for j, eid in enumerate(reader.episode_ids):
        s = reader.episode(eid).state
        states.append(s); ep_idx.append(np.full(len(s), j))
        s = s[: pickcube_approach_len(s)]
        Q.append(s[:, PC_QPOS]); CUBE.append(s[:, PC_OBJ_POS]); TZ.append(s[:, PC_TCP_POS][:, 2])
    Q = np.concatenate(Q); CUBE = np.concatenate(CUBE); TZ = np.concatenate(TZ)
    index = PickCubeSupportIndex(np.concatenate(states), np.concatenate(ep_idx))
    kc, ke = index.calibrate(percentile=99.0)

    def retrieve(cube, h, tol=0.025):
        m = np.abs(TZ - h) < tol
        if not m.any():
            m = np.abs(TZ - h) < 0.05
        idx = np.where(m)[0]
        return Q[idx[int(np.argmin(np.linalg.norm(CUBE[idx] - cube, axis=1)))]].copy()

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    rn = Runner(env, build_system(STANDALONE, policy, []))
    agent = rn.u.agent

    from actsemble.evaluation.video import save_video

    def run_to_boundary(seed, pol_seed):
        rn.start(pol_seed); raw, _ = env.reset(seed=seed)
        frames = []
        for step in range(BOUNDARY):
            s = rn.state42(raw); c, e = index.stats(s)
            frames.append(annotate(grab(env), step=step, phase="NOMINAL", c=c, e=e,
                                   kc=kc, ke=ke, s=s))
            raw, *_ = rn.act_env(raw)
        return raw, frames

    def play_out(raw, pol_seed, tag, pre, phase, start_step, extra_fn=None, border=None):
        """Hand back to the policy for whatever budget remains, rendering.
        start_step is the control step the arm has already reached (BOUNDARY for
        the control row, BOUNDARY+drive_steps for a recovery row) -- the drive is
        spent out of the SAME episode budget, so a long drive leaves less."""
        frames, once = list(pre), False
        rn.start(pol_seed + tag)
        for step in range(start_step, MAX_STEPS):
            s = rn.state42(raw); c, e = index.stats(s)
            frames.append(annotate(grab(env), step=step, phase=phase, c=c, e=e, kc=kc, ke=ke,
                                   s=s, extra=extra_fn() if extra_fn else "", border=border))
            raw, _r, _te, _tr, info = rn.act_env(raw)
            once = once or _success(info)
        return frames, once

    summary = []
    for seed in seeds:
        pol_seed = seed_of.get(seed, 0)
        raw, pre = run_to_boundary(seed, pol_seed)
        snap = snap_state(rn.u)
        cube0 = rn.state42(raw)[PC_OBJ_POS].copy()
        print(f"\n[rh] seed {seed} boundary t={BOUNDARY}", flush=True)

        rows, res = [], []
        # control row: no recovery, let the failure play out
        f_nom, ok_nom = play_out(raw, pol_seed, 40, pre, "NOMINAL (no recovery)", BOUNDARY)
        rows.append(("NO RECOVERY", f_nom)); res.append(("none", ok_nom, 0.0, 0))
        print(f"[rh]   no-recovery {'S' if ok_nom else 'F'}", flush=True)

        for hi, h in enumerate(heights):
            restore_state(env, rn.u, snap)
            raw = rn.u.get_obs()
            q = retrieve(cube0, h)
            agent.set_control_mode("pd_joint_pos"); agent.controller.reset()
            act = torch.tensor(np.concatenate([q[:7], [OPEN]])[None], dtype=torch.float32,
                               device=agent.robot.device)
            frames, bump, once, step = list(pre), 0.0, False, BOUNDARY

            def render_drive(d, tail_phase=False):
                nonlocal bump
                s = rn.state42(raw); c, e = index.stats(s)
                cfg = float(np.linalg.norm(s[:7] - q[:7]))
                bump = max(bump, float(np.linalg.norm(s[PC_OBJ_POS] - cube0)))
                frames.append(annotate(grab(env), step=step, phase=f"DRIVE to z={h:.2f}",
                                       c=c, e=e, kc=kc, ke=ke, s=s, border=BORDER[hi % len(BORDER)],
                                       extra=f"cfg->tgt {cfg:.3f}  bump {bump*100:5.2f}cm  "
                                             f"{'settle' if tail_phase else 'drive'} step {d + 1}"))
                return cfg

            for d in range(DRIVE_MAX):
                cfg = render_drive(d)
                raw, _r, _te, _tr, info = env.step(act)
                once = once or _success(info); step += 1
                if cfg < 0.02:
                    break
            for d in range(SETTLE):
                render_drive(d, tail_phase=True)
                raw, _r, _te, _tr, info = env.step(act)
                once = once or _success(info); step += 1
            drive_steps = step - BOUNDARY
            agent.set_control_mode("pd_ee_delta_pose"); agent.controller.reset()
            tail, ok = play_out(raw, pol_seed, 900 + hi * 50, frames,
                                f"POST-drive z={h:.2f}", step,
                                extra_fn=lambda h=h, b=bump, d=drive_steps:
                                    f"reset z={h:.2f}  bump {b*100:5.2f}cm  drive {d} steps",
                                border=BORDER[hi % len(BORDER)])
            ok = ok or once
            rows.append((f"DRIVE z={h:.2f}", tail)); res.append((h, ok, bump, drive_steps))
            print(f"[rh]   z={h:.2f} {'S' if ok else 'F'}  bump {bump*100:5.2f}cm  "
                  f"drive {drive_steps} steps", flush=True)

        n = max(len(f) for _, f in rows)
        pad = lambda L: L + [L[-1]] * (n - len(L))
        save_video([np.concatenate(cols, axis=1) for cols in zip(*[pad(f) for _, f in rows])],
                   out / f"heights_{seed}.mp4", fps=args.fps)
        filmstrip(rows, seed, out / f"filmstrip_heights_{seed}.png")
        summary.append({"env_seed": seed,
                        "arms": [{"height": h, "success": bool(ok), "bump_cm": 100 * b,
                                  "drive_steps": d} for h, ok, b, d in res]})
        print(f"[rh]   wrote heights_{seed}.mp4 + filmstrip", flush=True)
    env.close()
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"\n[rh] artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

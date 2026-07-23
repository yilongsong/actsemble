#!/usr/bin/env python
"""4-arm recovery comparison video: NOMINAL | TELEPORT | CARTESIAN-return |
JOINT-return, fired at the oracle's tp-winning boundary (teleport recovers by
construction). The JOINT arm switches the agent to pd_joint_pos for the takeover
window, PD-drives to the demo qpos, switches back to pd_ee_delta_pose, and hands
to the policy. Overlays c/e vs thresholds, grasp, cube->goal, and per-step
tip->target AND arm-config->target so the Cartesian plateau vs joint convergence
is visible. Filmstrip (4 rows) + config-trace comparison plot per seed.

    python scripts/visualize_joint_return.py --seeds 1041404040,2080359323,1611412778,84886883
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
from visualize_recovery_pickcube import (  # noqa: E402
    annotate,
    arm_nominal,
    arm_return,
    arm_teleport,
    cube_goal,
    filmstrip,
    grab,
    tp_winner,
)

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import PC_TCP_POS, PickCubeSupportIndex  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402


def arm_joint_return(rn, env, seed, pol_seed, index, kc, ke, fire_t, cand):
    """Physical joint-space return: switch to pd_joint_pos, PD-drive to the
    demo qpos, switch back, hand to policy."""
    agent = rn.u.agent
    rn.start(pol_seed)
    raw, _ = env.reset(seed=seed)
    frames, recs, once = [], [], False
    for step in range(fire_t):
        s = rn.state42(raw)
        c, e = index.stats(s)
        recs.append((step, c, e, "NOMINAL"))
        frames.append(annotate(grab(env), step=step, phase="NOMINAL", c=c, e=e, kc=kc, ke=ke, s=s))
        raw, _r, _te, _tr, info = rn.act_env(raw)
        once = once or _success(info)
    # ---- takeover in joint control ----
    agent.set_control_mode("pd_joint_pos")
    agent.controller.reset()
    cand_arm = np.asarray(cand[0])[0:7]
    # pd_joint_pos gripper action is NORMALIZED [-1,1]: +1.0 = fully open (0.04m
    # fingers), 0 = semi-closed. Pre-grasp candidate -> jaws fully OPEN so the
    # approach clears the cube instead of colliding with it.
    grip = 1.0 if not cand[2] else -1.0
    jact = torch.tensor(np.concatenate([cand_arm, [grip]])[None], dtype=torch.float32,
                        device=agent.robot.device)
    tip_trace, cfg_trace = [], []
    step = fire_t
    while step < MAX_STEPS:
        s = rn.state42(raw)
        cfg = float(np.linalg.norm(s[0:7] - cand_arm))
        tip = float(np.linalg.norm(s[PC_TCP_POS] - cand[1]))
        cfg_trace.append((step, cfg)); tip_trace.append((step, tip))
        c, e = index.stats(s)
        recs.append((step, c, e, "JOINT"))
        frames.append(annotate(grab(env), step=step, phase="JOINT-RETURN (pd_joint_pos)",
                               c=c, e=e, kc=kc, ke=ke, s=s, border=(60, 185, 90),
                               extra=f"CONFIG(arm7)->tgt {cfg:.3f}  tip->tgt {tip:.3f}"))
        raw, _r, _te, _tr, info = env.step(jact)
        once = once or _success(info)
        step += 1
        if cfg < 0.03:
            break
    s_hb = rn.state42(raw)
    cfg_hb = float(np.linalg.norm(s_hb[0:7] - cand_arm))
    agent.set_control_mode("pd_ee_delta_pose")
    agent.controller.reset()
    rn.start(pol_seed + 5555)
    hb = f"config REACHED {cfg_hb:.3f}" if cfg_hb < 0.15 else f"config {cfg_hb:.3f}"
    for step in range(step, MAX_STEPS):
        s = rn.state42(raw)
        c, e = index.stats(s)
        recs.append((step, c, e, "POST"))
        frames.append(annotate(grab(env), step=step, phase="POST-joint (policy)", c=c, e=e,
                               kc=kc, ke=ke, s=s, extra=hb if step == recs[-1][0] else ""))
        raw, _r, _te, _tr, info = rn.act_env(raw)
        once = once or _success(info)
    return frames, recs, once, tip_trace, cfg_trace, cfg_hb


def config_plot(seed, cart, joint, fire_t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ct, chb, cout = cart
    jt, jhb, jout = joint
    if ct:
        ax.plot([x[0] for x in ct], [x[1] for x in ct], "-s", ms=4, color="#c0392b",
                label=f"CARTESIAN return config->tgt (handback {chb:.3f}, {'S' if cout else 'FAIL'})")
    if jt:
        ax.plot([x[0] for x in jt], [x[1] for x in jt], "-o", ms=4, color="#2a9d3a",
                label=f"JOINT return config->tgt (handback {jhb:.3f}, {'S' if jout else 'FAIL'})")
    ax.axhline(0.15, color="#888", ls="--", lw=1, label="config-reached 0.15")
    ax.axvline(fire_t, color="#111", lw=1.2)
    ax.set_xlabel("control step"); ax.set_ylabel("arm-config error to target (rad)")
    ax.set_title(f"seed {seed}: Cartesian return PLATEAUS vs joint return REACHES config")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=135); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--oracle-json", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_oracle.json"))
    ap.add_argument("--seeds", default="1041404040,2080359323,1611412778,84886883")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/pickcube/recovery_oracle/joint_return_viz"))
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    oracle = json.load(open(args.oracle_json))
    seed_of = {e["env_seed"]: e["seed"] for e in oracle["reference"]}
    fail_of = {r["env_seed"]: r for r in oracle["failures"]}

    reader = DatasetReader(args.dataset)
    states, ep_idx = [], []
    for j, eid in enumerate(reader.episode_ids):
        ep = reader.episode(eid)
        states.append(ep.state); ep_idx.append(np.full(len(ep.state), j))
    index = PickCubeSupportIndex(np.concatenate(states), np.concatenate(ep_idx))
    kc, ke = index.calibrate(percentile=99.0)

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    rn = Runner(env, build_system(STANDALONE, policy, []))

    from actsemble.evaluation.video import save_video
    summary = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        pol_seed = seed_of.get(seed, 0)
        fire_t, cand, vex, vtp = tp_winner(fail_of[seed])
        mode = fail_of[seed]["mode"]
        print(f"\n[jr] seed {seed} {mode} boundary t={fire_t} (v_ex={vex:.1f} v_tp={vtp:.1f})", flush=True)
        fn, rn_nom, on = arm_nominal(rn, env, seed, pol_seed, index, kc, ke)
        ft, rn_tp, ot = arm_teleport(rn, env, seed, pol_seed, index, kc, ke, fire_t, cand)
        fc, rn_c, oc, tipc, tiphc, cfgc, cfghc = arm_return(rn, env, seed, pol_seed, index, kc, ke, fire_t, cand)
        fj, rn_j, oj, tipj, cfgj, cfghj = arm_joint_return(rn, env, seed, pol_seed, index, kc, ke, fire_t, cand)
        print(f"[jr]   nominal={'S' if on else 'F'} teleport={'S' if ot else 'F'} "
              f"CARTESIAN={'S' if oc else 'F'}(cfg {cfghc:.2f}) JOINT={'S' if oj else 'F'}(cfg {cfghj:.2f})",
              flush=True)
        summary.append({"seed": seed, "mode": mode, "nominal": on, "teleport": ot,
                        "cartesian": oc, "cartesian_cfg": cfghc, "joint": oj, "joint_cfg": cfghj})
        n = max(len(fn), len(ft), len(fc), len(fj))
        pad = lambda L: L + [L[-1]] * (n - len(L))
        save_video([np.concatenate([a, b, c, d], axis=1)
                    for a, b, c, d in zip(pad(fn), pad(ft), pad(fc), pad(fj))],
                   out / f"combined4_{seed}.mp4", fps=args.fps)
        filmstrip([("NOMINAL", fn), ("TELEPORT", ft), ("CARTESIAN-return", fc), ("JOINT-return", fj)],
                  seed, out / f"filmstrip4_{seed}.png")
        config_plot(seed, (cfgc, cfghc, oc), (cfgj, cfghj, oj), fire_t, out / f"config_{seed}.png")
        print(f"[jr]   wrote combined4/filmstrip4/config mp4+png", flush=True)
    env.close()
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    nS = lambda k: sum(r[k] for r in summary)
    print(f"\n[jr] {len(summary)} seeds: teleport {nS('teleport')} | CARTESIAN {nS('cartesian')} | "
          f"JOINT {nS('joint')} recovered")
    print(f"[jr] artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

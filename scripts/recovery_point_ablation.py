#!/usr/bin/env python
"""WHERE along the approach should recovery reset to? Dense ablation, CLEAN.

Isolates reset-point from episode-match quality: for each RECOVERABLE failure
(oracle tp_final >= thresh), take the oracle's tp-winning candidate config
(known to recover under teleport), locate it in the demo data (episode, frame),
and sweep the reset target along THAT SAME demo's approach — earlier/higher
frames (offset>0) and slightly later/lower (offset<0) around the winning point.
Every target is a COMPLETE in-distribution demo qpos. TELEPORT to each, roll K.
Because the episode is held fixed at a known-good one, the only thing varying is
HOW FAR BACK along a good approach we reset to — the user's question, cleanly.

    python scripts/recovery_point_ablation.py --k 4
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
from visualize_recovery_pickcube import tp_winner  # noqa: E402

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import PC_QPOS, PC_TCP_POS, pickcube_approach_len  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402

OFFSETS = [-4, -2, 0, 2, 4, 6, 8, 10, 12]  # frames earlier(+)/later(-) than the winning point


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--oracle-json", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_oracle.json"))
    ap.add_argument("--tp-thresh", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_point_ablation.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    oracle = json.load(open(args.oracle_json))
    seed_of = {e["env_seed"]: e["seed"] for e in oracle["reference"]}
    fails = [r for r in oracle["failures"] if r["tp_final"] >= args.tp_thresh]

    reader = DatasetReader(args.dataset)
    ep_states, flat_q, flat_ep, flat_fr = [], [], [], []
    for j, eid in enumerate(reader.episode_ids):
        ep = reader.episode(eid)
        ep_states.append(ep.state)
        flat_q.append(ep.state[:, PC_QPOS]); flat_ep.append(np.full(len(ep.state), j))
        flat_fr.append(np.arange(len(ep.state)))
    flat_q = np.concatenate(flat_q); flat_ep = np.concatenate(flat_ep); flat_fr = np.concatenate(flat_fr)

    def locate(qpos):
        i = int(np.argmin(np.linalg.norm(flat_q - qpos[None, :], axis=1)))
        return int(flat_ep[i]), int(flat_fr[i])

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode=None, max_episode_steps=MAX_STEPS)
    rn = Runner(env, build_system(STANDALONE, policy, []))

    def teleport_roll(es, ps, fire_t, qpos, off):
        rn.start(ps); raw, _ = env.reset(seed=es)
        for _ in range(fire_t): raw, *_ = rn.act_env(raw)
        teleport_robot(rn.u, qpos)
        raw = rn.u.get_obs(); rn.start(ps + 9000 + (off + 4) * 50); once = False
        for _ in range(MAX_STEPS):
            raw, _r, _te, _tr, info = rn.act_env(raw); once = once or _success(info)
        return once

    rows = []
    for i, r in enumerate(fails):
        es = r["env_seed"]; ps = seed_of[es]
        fire_t, cand, _, _ = tp_winner(r)
        ej, fw = locate(np.asarray(cand[0]))
        st = ep_states[ej]
        grasp_f = pickcube_approach_len(st)
        for off in OFFSETS:
            fidx = fw - off  # off>0 => earlier frame (higher approach)
            if fidx < 0 or fidx >= grasp_f:   # keep it pre-grasp, in-episode
                continue
            q = st[fidx, PC_QPOS].copy(); tcp_z = float(st[fidx, PC_TCP_POS][2])
            wins = sum(teleport_roll(es, ps, fire_t, q, off) for _ in range(args.k))
            rows.append({"env_seed": es, "mode": r["mode"], "offset": off,
                         "tcp_z": tcp_z, "frames_before_grasp": grasp_f - fidx,
                         "recovery": wins / args.k})
        if (i + 1) % 8 == 0:
            print(f"[ablation] {i + 1}/{len(fails)}", flush=True)
    env.close()

    byoff = {}
    for r in rows:
        byoff.setdefault(r["offset"], []).append(r)
    curve = [{"offset": o, "n": len(v),
              "recovery": float(np.mean([x["recovery"] for x in v])),
              "tcp_z_median": float(np.median([x["tcp_z"] for x in v])),
              "fbg_median": float(np.median([x["frames_before_grasp"] for x in v]))}
             for o, v in sorted(byoff.items())]
    # also bin by absolute frames-before-grasp (task-general axis)
    fb = np.array([r["frames_before_grasp"] for r in rows]); rr = np.array([r["recovery"] for r in rows])
    fbcurve = [{"fbg_lo": lo, "fbg_hi": hi, "n": int(((fb >= lo) & (fb < hi)).sum()),
                "recovery": float(rr[(fb >= lo) & (fb < hi)].mean()) if ((fb >= lo) & (fb < hi)).any() else None}
               for lo, hi in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 20)]]
    json.dump({"curve_by_offset": curve, "curve_by_frames_before_grasp": fbcurve,
               "rows": rows, "k": args.k, "n_recoverable_failures": len(fails)}, open(args.out, "w"), indent=1)

    print(f"\nRECOVERY vs RESET POINT on {len(fails)} recoverable failures (teleport, K={args.k}):")
    print(f"  {'off':>4}{'fbg':>5}{'tcp_z':>8}{'recovery':>10}  (off>0 = earlier/higher on the approach)")
    for c in curve:
        print(f"  {c['offset']:>4}{c['fbg_median']:>5.0f}{c['tcp_z_median']:>8.3f}{c['recovery']:>9.0%}")
    print("  by absolute frames-before-grasp:")
    for c in fbcurve:
        if c["recovery"] is not None:
            print(f"    fbg[{c['fbg_lo']:>2},{c['fbg_hi']:>2}) n={c['n']:<3} recovery {c['recovery']:.0%}")
    print(f"[ablation] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

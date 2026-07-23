#!/usr/bin/env python
"""Target-selection experiment (isolated from the return controller): for each
failure, retrieve TWO recovery poses from the dataset by the same cube-position
kNN, differing only in demo phase --

  NEAR-GRASP : the most committed nearby pre-grasp state (lowest fingertip z)
  RETRACTED  : the most retracted nearby pre-grasp state (highest fingertip z)

-- TELEPORT to each (non-physical; isolates target quality, no transport), roll
the frozen policy K times, and compare recovery rate + cube disturbance. Answers
the design question 'should the recovery pose be retracted or near-grasp?' at the
CEILING, before we build any physical return. Uses recovery_oracle.json's failure
list for the (env_seed, policy_seed) reconstruction.

    python scripts/recovery_target_comparison.py --k 5
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
    teleport_robot,
)

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import (  # noqa: E402
    PC_GRASPED,
    PC_OBJ_POS,
    PC_QPOS,
    PC_TCP_POS,
    PickCubeSupportIndex,
)
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402


def retrieve_pair(index, s, n_scan=12):
    """From the nearby-cube PRE-GRASP demo states (distinct episodes), return the
    lowest-z (near-grasp) and highest-z (retracted) as (qpos, tcp, z)."""
    d = index._d_env(s[PC_OBJ_POS])
    picks, seen = [], set()
    for i in np.argsort(d):
        ep = index.episode_index[i]
        if ep in seen or index.grasped[i]:
            continue
        seen.add(ep)
        picks.append((index.qpos[i].copy(), index.tcp_pos[i].copy(),
                      float(index.tcp_pos[i][2])))
        if len(picks) == n_scan:
            break
    if len(picks) < 2:
        return None, None
    picks.sort(key=lambda x: x[2])
    return picks[0], picks[-1]  # near-grasp (low z), retracted (high z)


def teleport_value(rn, env, seed, pol_seed, qpos, K):
    wins, bumps = 0, []
    for k in range(K):
        rn.start(pol_seed)
        raw, _ = env.reset(seed=seed)
        teleport_robot(rn.u, qpos)
        raw = rn.u.get_obs()
        rn.start(pol_seed + 5000 + k)
        cube0 = rn.state42(raw)[PC_OBJ_POS].copy()
        maxbump, once = 0.0, False
        for _ in range(MAX_STEPS):
            raw, _r, _te, _tr, info = rn.act_env(raw)
            maxbump = max(maxbump, float(np.linalg.norm(rn.state42(raw)[PC_OBJ_POS] - cube0)))
            once = once or _success(info)
        wins += once
        bumps.append(maxbump)
    return wins / K, float(np.mean(bumps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--oracle-json", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_oracle.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "outputs/pickcube/recovery_oracle/target_comparison.json"))
    args = ap.parse_args()

    oracle = json.load(open(args.oracle_json))
    fails = [(e["env_seed"], e["seed"], e["mode"]) for e in oracle["reference"] if not e["success"]]

    reader = DatasetReader(args.dataset)
    states, ep_idx = [], []
    for j, eid in enumerate(reader.episode_ids):
        ep = reader.episode(eid)
        states.append(ep.state)
        ep_idx.append(np.full(len(ep.state), j))
    index = PickCubeSupportIndex(np.concatenate(states), np.concatenate(ep_idx))

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode=None, max_episode_steps=MAX_STEPS)
    rn = Runner(env, build_system(STANDALONE, policy, []))

    rows = []
    for i, (es, ps, mode) in enumerate(fails):
        # reconstruct a mid-failure state: roll nominal to a fixed early boundary
        rn.start(ps)
        raw, _ = env.reset(seed=es)
        for _ in range(8):  # one replan boundary in
            raw, *_ = rn.act_env(raw)
        s = rn.state42(raw)
        near, retr = retrieve_pair(index, s)
        if near is None:
            continue
        vn, bn = teleport_value(rn, env, es, ps, near[0], args.k)
        vr, br = teleport_value(rn, env, es, ps, retr[0], args.k)
        rows.append({"env_seed": es, "mode": mode, "near_z": near[2], "retr_z": retr[2],
                     "v_near": vn, "v_retr": vr, "bump_near": bn, "bump_retr": br})
        if (i + 1) % 10 == 0:
            print(f"[target] {i + 1}/{len(fails)}", flush=True)
    env.close()

    def agg(key):
        return float(np.mean([r[key] for r in rows]))
    summary = {
        "n": len(rows), "K": args.k,
        "near_grasp_z_median": float(np.median([r["near_z"] for r in rows])),
        "retracted_z_median": float(np.median([r["retr_z"] for r in rows])),
        "teleport_recovery_near": agg("v_near"),
        "teleport_recovery_retracted": agg("v_retr"),
        "cube_bump_near": agg("bump_near"),
        "cube_bump_retracted": agg("bump_retr"),
    }
    # per mode
    modes = {}
    for r in rows:
        modes.setdefault(r["mode"], []).append(r)
    per_mode = {m: {"n": len(v), "v_near": float(np.mean([x["v_near"] for x in v])),
                    "v_retr": float(np.mean([x["v_retr"] for x in v]))}
                for m, v in modes.items()}
    json.dump({"summary": summary, "per_mode": per_mode, "rows": rows}, open(args.out, "w"), indent=1)
    print("\n" + "=" * 62)
    print(f"n={summary['n']} failures, K={args.k} rollouts each, TELEPORT (target quality isolated)")
    print(f"  near-grasp target (fingertip z~{summary['near_grasp_z_median']:.3f}):  "
          f"recovery {summary['teleport_recovery_near']:.1%}  cube-bump {summary['cube_bump_near']:.3f}")
    print(f"  retracted  target (fingertip z~{summary['retracted_z_median']:.3f}):  "
          f"recovery {summary['teleport_recovery_retracted']:.1%}  cube-bump {summary['cube_bump_retracted']:.3f}")
    print("  per mode (recovery near -> retracted):")
    for m, v in sorted(per_mode.items(), key=lambda x: -x[1]["n"]):
        print(f"    {m:<24} n={v['n']:<3} {v['v_near']:.0%} -> {v['v_retr']:.0%}")
    print(f"[target] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

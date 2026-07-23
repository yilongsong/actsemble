#!/usr/bin/env python
"""Failure-detection + recovery ceiling ORACLE, PickCube-v1.

PickCube instantiation of scripts/recovery_oracle.py (docs/recovery_scheme_
and_oracle.md Part II), same four-stage design (reference+snapshots -> sweep
-> controls -> deployable prototype), targeted at the GROUNDED failure
taxonomy's dominant class (`grasp_slip_at_pick`, 57% of failures at the
n_demos=100 operating point — see outputs/failure_taxonomy/pickcube_n100/):
gripper closes, loses the cube within ~a step, cube stays near the pick
zone. Registered prediction: this class should be highly teleport- AND
executed-recoverable (scene-intact retryable), an order above PushT's
+2.6pt ceiling — this run measures whether that holds.

Differences from the PushT driver (recovery_oracle.py), all load-bearing:
  * PickCubeSupportIndex / PickCubeReturnController (recovery.py) — cube
    xyz support metric (no yaw term), full 9-dim qpos (arm+gripper) as the
    conditional-surprisal feature, candidates carry `is_grasped`.
  * teleport_robot writes 9-dim qpos (not 7) and zeros the full 18-dim qvel.
  * action dim 7 (6 pose delta + 1 gripper), not 6.
  * MAX_STEPS = 50 (PickCube's registered/task-native horizon; also what
    the n-demos sweep and failure taxonomy used) — episodes are short, so
    despite K=16*M branching this run is CHEAPER per-episode than PushT's.
  * per-failure-mode breakdown of the ceiling (the grounded taxonomy modes),
    not just the aggregate — the point of this run is "does recovery help
    grasp_slip specifically", not just an aggregate number.

Diagnostic tier: single policy seed, dev panel. Never used for training or
checkpoint selection. Teleport results labeled non-physical.

    python scripts/recovery_oracle_pickcube.py --smoke   # 3-episode dry run
    python scripts/recovery_oracle_pickcube.py           # full run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from failure_taxonomy import PickCubeAdapter  # noqa: E402

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel, panel_episodes  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import PC_GOAL_POS, PC_OBJ_POS, PC_QPOS, PickCubeReturnController, PickCubeSupportIndex  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.sim.rollout import ObservationAdapter  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

STANDALONE = {"policy": {"num_candidates": 1}, "components": [],
              "selection": {"type": "candidate_zero"}, "execution": {}}
# Episode horizon. 50 was PickCube-v1's registered value and matched the RL
# bundle, whose demos finish in 19-29 steps. The DEFAULT POOL IS NOW THE
# TELEOP-LIKE ONE, whose demos run 38-110 steps, so a policy trained on it
# cannot finish inside 50 and every episode would score 0 -- the same trap that
# silently wasted a full training pipeline on 2026-07-21. Override with
# --max-steps; 160 for teleop-trained policies.
MAX_STEPS = 50  # PickCube-v1 registered/task-native horizon (RL-bundle era)


# ---------- sim-state + rollout plumbing (mirrors recovery_oracle.py) -------
def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def snap_state(u):
    d = u.get_state_dict()
    return {g: {k: v.clone() for k, v in grp.items()} for g, grp in d.items()}


def restore_state(env, u, snap):
    u.set_state_dict({g: {k: v.clone() for k, v in grp.items()} for g, grp in snap.items()})


def teleport_robot(u, qpos9):
    """Write 9-dim qpos (arm+gripper), zero the full 18-dim qvel. Cube/goal
    untouched (only the robot articulation slice is written) — the same
    scene-preserving teleport PushT's version does, generalized to the
    PickCube robot's 9+9 (vs 7+7) layout."""
    t = u.get_state_dict()
    art = t["articulations"][next(iter(t["articulations"]))]
    art[..., -18:-9] = torch.as_tensor(qpos9, dtype=art.dtype, device=art.device)
    art[..., -9:] = 0.0
    u.set_state_dict(t)


def _success(info) -> bool:
    s = info.get("success")
    return s is not None and bool(_np(s).reshape(-1)[0])


class Runner:
    def __init__(self, env, system):
        self.env, self.u, self.system = env, env.unwrapped, system
        low = np.asarray(self.u.single_action_space.low, dtype=np.float32)
        high = np.asarray(self.u.single_action_space.high, dtype=np.float32)
        self.low, self.high = low, high
        self.obs_adapter = ObservationAdapter(action_dim=low.shape[0])

    def state42(self, raw) -> np.ndarray:
        return _np(raw).reshape(-1)

    def act_env(self, raw):
        observation = self.obs_adapter.observe(raw)
        action = self.system.act(observation)
        a = np.clip(np.asarray(action.value, np.float32).reshape(-1), self.low, self.high)
        return self.env.step(torch.as_tensor(a.reshape(1, -1)))

    def start(self, seed_root: int, frames=()):
        self.system.candidate_root_seed = int(seed_root)
        self.system.reset(episode_seed=int(seed_root))
        for f in frames:
            self.system._history.append(np.array(f, copy=True))

    def run(self, raw, remaining: int) -> bool:
        for _ in range(int(remaining)):
            raw, _r, _te, _tr, info = self.act_env(raw)
            if _success(info):
                return True
        return False

    def step_raw(self, a7: np.ndarray):
        a = np.clip(np.asarray(a7, np.float32).reshape(-1), self.low, self.high)
        return self.env.step(torch.as_tensor(a.reshape(1, -1)))


# ---------- value estimators -------------------------------------------------
def v_nom(rn: Runner, snapb, frames, remaining, K, seed0) -> float:
    wins = 0
    for k in range(K):
        restore_state(rn.env, rn.u, snapb)
        rn.start(seed0 + k, frames)
        wins += rn.run(rn.u.get_obs(), remaining)
    return wins / K


def v_rec_tp(rn: Runner, snapb, qpos_cand, remaining, K, seed0) -> float:
    wins = 0
    for k in range(K):
        restore_state(rn.env, rn.u, snapb)
        teleport_robot(rn.u, qpos_cand)
        rn.start(seed0 + k)
        wins += rn.run(rn.u.get_obs(), remaining)
    return wins / K


def v_rec_ex(rn: Runner, snapb, tip_cand, grasped_cand, remaining, K, seed0, tret_max=20):
    wins, trets, bumps = 0, [], []
    for k in range(K):
        restore_state(rn.env, rn.u, snapb)
        raw = rn.u.get_obs()
        s = rn.state42(raw)
        obj0 = s[PC_OBJ_POS].copy()
        ctl = PickCubeReturnController(tip_cand, grasped_cand, max_steps=tret_max)
        used, once = 0, False
        while used < remaining and not ctl.done(rn.state42(raw)[19:22]):
            raw, _r, _te, _tr, info = rn.step_raw(ctl.action(rn.state42(raw)[19:22]))
            used += 1
            once = once or _success(info)
        s = rn.state42(raw)
        bumps.append(float(np.linalg.norm(s[PC_OBJ_POS] - obj0)))
        trets.append(used)
        rn.start(seed0 + k)
        wins += once or rn.run(raw, remaining - used)
    return wins / K, float(np.mean(trets)), float(np.mean(bumps))


def _snap_state42(rn: Runner, b) -> np.ndarray:
    restore_state(rn.env, rn.u, b["snap"])
    return rn.state42(rn.u.get_obs())


# ---------- grounded failure-mode classifier (mirrors failure_taxonomy.py) --
def classify_failure(rec_states) -> str:
    """rec_states: list of 42-dim states across the episode. Same grounded
    rule as failure_taxonomy.py's PickCube reclassify (fit to the measured
    distributions — see outputs/failure_taxonomy/pickcube_n100/)."""
    grasped = np.array([s[18] > 0.5 for s in rec_states])
    tcp = np.stack([s[19:22] for s in rec_states])
    obj = np.stack([s[PC_OBJ_POS] for s in rec_states])
    goal = rec_states[-1][PC_GOAL_POS]
    ever_grasped = bool(grasped.any())
    min_tcp_obj = float(np.linalg.norm(tcp - obj, axis=1).min())
    obj_xy_disp = float(np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1).max())
    lift = obj[:, 2] - obj[0, 2]
    final_gdist = float(np.linalg.norm(obj[-1] - goal))
    if not ever_grasped:
        if min_tcp_obj > 0.05:
            return "never_reached"
        if obj_xy_disp > 0.05:
            return "knocked_away"
        return "grasp_failure_no_close"
    grasp_frac = float(grasped.mean())
    max_lift = float(lift.max())
    if grasp_frac < 0.10 and max_lift < 0.02:
        return "grasp_slip_at_pick"
    if max_lift >= 0.02 and final_gdist > 0.05:
        return "lifted_then_dropped"
    return "delivered_imprecise"


# ---------- main --------------------------------------------------------------
def main() -> int:
    global MAX_STEPS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=None,
                    help="demo pool for the support index; default = the policy's own subset")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/pickcube/recovery_oracle"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first SKIP panel episodes (300 available). Lets a second "
                         "instance run a DISJOINT slice on another GPU for pooling.")
    ap.add_argument("--k-nom", type=int, default=4)
    ap.add_argument("--k-sel", type=int, default=3)
    ap.add_argument("--k-eval", type=int, default=10)
    ap.add_argument("--m-poses", type=int, default=3)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS,
                    help="episode horizon; MUST match the demonstrator the policy "
                         "was trained on (50 = RL bundle, 160 = teleop-like pool)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    MAX_STEPS = int(args.max_steps)
    print(f"[oracle] episode horizon = {MAX_STEPS} steps", flush=True)
    if args.smoke:
        args.n, args.k_nom, args.k_sel, args.k_eval, args.m_poses = 3, 1, 1, 2, 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    policy = load_policy(args.policy_checkpoint, device=args.device)
    dataset = args.dataset or REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"
    reader = DatasetReader(dataset)
    states, ep_idx = [], []
    for j, eid in enumerate(reader.episode_ids):
        ep = reader.episode(eid)
        states.append(ep.state)
        ep_idx.append(np.full(len(ep.state), j))
    index = PickCubeSupportIndex(np.concatenate(states), np.concatenate(ep_idx))
    kappa_c, kappa_e = index.calibrate(percentile=args.percentile)
    print(f"[recovery-pc] thresholds: kappa_c={kappa_c:.4f} kappa_e={kappa_e:.4f} "
          f"(L={index.L:.3f})", flush=True)

    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode=None, max_episode_steps=MAX_STEPS)
    system = build_system(STANDALONE, policy, [])
    rn = Runner(env, system)
    H_a = system.action_horizon
    H_o = policy.meta.obs_horizon
    # --skip takes a DISJOINT slice of the 300-episode diagnostic panel so a
    # second instance can run on another GPU and have its failures POOLED with
    # the first (same policy, independent episodes). Sharding by failure INDEX
    # would not be sound here: physx_cuda is nondeterministic, so two processes
    # do not necessarily derive the same failure list from the same episodes.
    eps = list(panel_episodes(make_panel("diagnostic")))[args.skip: args.skip + args.n]

    # ---- stage 1: reference run + snapshots --------------------------------
    print(f"[recovery-pc] REFERENCE: {len(eps)} episodes, boundary grid H_a={H_a}", flush=True)
    episodes = []
    for pe in eps:
        rn.start(pe.policy_sampling_seed)
        raw, _ = env.reset(seed=pe.env_seed)
        bounds, once, step, rec_states = [], False, 0, []
        while step < MAX_STEPS:
            s42 = rn.state42(raw)
            rec_states.append(s42)
            if step % H_a == 0:
                c, e = index.stats(s42)
                bounds.append({
                    "t": step, "c": c, "e": e,
                    "snap": snap_state(rn.u),
                    "frames": [np.array(f, copy=True)
                               for f in list(rn.system._history)][-(max(H_o - 1, 0)) or 0:]
                    if H_o > 1 else [],
                })
            raw, _r, _te, _tr, info = rn.act_env(raw)
            once = once or _success(info)
            step += 1
        mode = "success" if once else classify_failure(rec_states)
        episodes.append({"env_seed": pe.env_seed, "seed": pe.policy_sampling_seed,
                         "success": once, "mode": mode, "bounds": bounds})
    n_succ = sum(e["success"] for e in episodes)
    print(f"[recovery-pc] nominal J = {n_succ}/{len(eps)} = {n_succ / len(eps):.1%}", flush=True)
    from collections import Counter
    mode_counts = Counter(e["mode"] for e in episodes if not e["success"])
    print(f"[recovery-pc] failure modes: {dict(mode_counts.most_common())}", flush=True)

    # ---- stage 2: sweep on failures ---------------------------------------
    fails = [e for e in episodes if not e["success"]]
    print(f"[recovery-pc] SWEEP over {len(fails)} failures "
          f"(K_nom={args.k_nom} K_sel={args.k_sel} K_eval={args.k_eval} M={args.m_poses})",
          flush=True)
    rows = []
    for i, e in enumerate(fails):
        cur = {"env_seed": e["env_seed"], "mode": e["mode"], "grid": []}
        for b in e["bounds"]:
            remaining = MAX_STEPS - b["t"]
            seed0 = 100000 + e["env_seed"] * 1000 + b["t"]
            cands = index.candidates(_snap_state42(rn, b), m=args.m_poses)
            vn = v_nom(rn, b["snap"], b["frames"], remaining, args.k_nom, seed0)
            tp = [v_rec_tp(rn, b["snap"], q, remaining, args.k_sel, seed0 + 71 * (j + 1))
                  for j, (q, _t, _g) in enumerate(cands)]
            ex = [v_rec_ex(rn, b["snap"], t3, g, remaining, args.k_sel, seed0 + 91 * (j + 1))
                  for j, (_q, t3, g) in enumerate(cands)]
            cur["grid"].append({
                "t": b["t"], "c": b["c"], "e": b["e"], "v_nom": vn,
                "v_tp": [float(x) for x in tp],
                "v_ex": [float(x[0]) for x in ex],
                "t_ret": [float(x[1]) for x in ex],
                "bump": [float(x[2]) for x in ex],
                "cand_qpos": [q.tolist() for q, _, _ in cands],
                "cand_tip": [t.tolist() for _, t, _ in cands],
                "cand_grasped": [g for _, _, g in cands],
            })
        g = cur["grid"]
        if not g:
            cur["tp_final"] = cur["ex_final"] = cur["reroll_final"] = 0.0
            rows.append(cur)
            continue
        bi = int(np.argmax([max(r["v_tp"]) for r in g]))
        bj = int(np.argmax(g[bi]["v_tp"]))
        cur["tp_final"] = v_rec_tp(rn, e["bounds"][bi]["snap"],
                                   np.array(g[bi]["cand_qpos"][bj]),
                                   MAX_STEPS - g[bi]["t"], args.k_eval,
                                   900000 + e["env_seed"])
        bi = int(np.argmax([max(r["v_ex"]) for r in g]))
        bj = int(np.argmax(g[bi]["v_ex"]))
        cur["ex_final"] = v_rec_ex(rn, e["bounds"][bi]["snap"],
                                   np.array(g[bi]["cand_tip"][bj]),
                                   g[bi]["cand_grasped"][bj],
                                   MAX_STEPS - g[bi]["t"], args.k_eval,
                                   910000 + e["env_seed"])[0]
        bi = int(np.argmax([r["v_nom"] for r in g]))
        cur["reroll_final"] = v_nom(rn, e["bounds"][bi]["snap"], e["bounds"][bi]["frames"],
                                    MAX_STEPS - g[bi]["t"], args.k_eval,
                                    920000 + e["env_seed"])
        rows.append(cur)
        print(f"[sweep] {i + 1}/{len(fails)} seed={e['env_seed']} mode={e['mode']}: "
              f"tp={cur['tp_final']:.2f} ex={cur['ex_final']:.2f} "
              f"reroll={cur['reroll_final']:.2f}", flush=True)

    # ---- stage 3: false-alarm arm -----------------------------------------
    fa = []
    for e in [x for x in episodes if x["success"]][:10]:
        if len(e["bounds"]) < 3:
            continue
        for b in [e["bounds"][len(e["bounds"]) // 3], e["bounds"][2 * len(e["bounds"]) // 3]]:
            remaining = MAX_STEPS - b["t"]
            seed0 = 800000 + e["env_seed"] * 100 + b["t"]
            cands = index.candidates(_snap_state42(rn, b), m=1)
            vn = v_nom(rn, b["snap"], b["frames"], remaining, 3, seed0)
            vr = v_rec_tp(rn, b["snap"], cands[0][0], remaining, 3, seed0 + 7)
            fa.append({"env_seed": e["env_seed"], "t": b["t"], "delta": float(vr - vn)})
    print(f"[recovery-pc] false-alarm arm: mean Delta on successes = "
          f"{np.mean([x['delta'] for x in fa]) if fa else float('nan'):+.3f}", flush=True)
    env.close()

    # ---- summary -----------------------------------------------------------
    n = len(episodes)
    J_nom = n_succ / n
    tp_sum = (n_succ + sum(r["tp_final"] for r in rows)) / n
    ex_sum = (n_succ + sum(r["ex_final"] for r in rows)) / n
    rr_sum = (n_succ + sum(r["reroll_final"] for r in rows)) / n

    per_mode = {}
    for m, seeds in mode_counts.items():
        rs = [r for r in rows if r["mode"] == m]
        per_mode[m] = {
            "n": len(rs),
            "tp_mean": float(np.mean([r["tp_final"] for r in rs])) if rs else None,
            "ex_mean": float(np.mean([r["ex_final"] for r in rs])) if rs else None,
            "reroll_mean": float(np.mean([r["reroll_final"] for r in rs])) if rs else None,
        }

    summary = {
        "policy": args.policy_checkpoint, "panel": "diagnostic", "n": n,
        "H_a": H_a, "kappa_c": kappa_c, "kappa_e": kappa_e,
        "J_nom": J_nom, "J_nom_ci": list(wilson_interval(n_succ, n)),
        "J_reroll": rr_sum, "J_ceil_tp": tp_sum, "J_ceil_ex": ex_sum,
        "headroom_tp_vs_reroll": tp_sum - rr_sum,
        "headroom_ex_vs_reroll": ex_sum - rr_sum,
        "false_alarm_mean_delta": float(np.mean([x["delta"] for x in fa])) if fa else None,
        "failure_mode_counts": dict(mode_counts.most_common()),
        "per_mode_ceiling": per_mode,
        "runtime_min": (time.time() - t_start) / 60.0,
    }
    save_json({"summary": summary, "failures": rows, "false_alarm": fa,
               "reference": [{k: v for k, v in e.items() if k != "bounds"} |
                             {"stats": [{kk: b[kk] for kk in ("t", "c", "e")}
                                        for b in e["bounds"]]}
                             for e in episodes]},
              out / "recovery_oracle.json")
    print("=" * 70)
    print(f"J_nom       {J_nom:.1%}   (baseline, this panel)")
    print(f"J_reroll    {rr_sum:.1%}   (oracle-timed do-nothing — luck control)")
    print(f"J_ceil(ex)  {ex_sum:.1%}   headroom vs reroll {ex_sum - rr_sum:+.1%}")
    print(f"J_ceil(tp)  {tp_sum:.1%}   headroom vs reroll {tp_sum - rr_sum:+.1%}  [non-physical]")
    print("\nper-mode ceiling (tp_final mean | ex_final mean | reroll mean):")
    for m, d in sorted(per_mode.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {m:<24} n={d['n']:>3}  tp={d['tp_mean']:.2f}  ex={d['ex_mean']:.2f}  "
              f"reroll={d['reroll_mean']:.2f}")
    print(f"\n[recovery-pc] wrote {out / 'recovery_oracle.json'}  "
          f"({summary['runtime_min']:.0f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

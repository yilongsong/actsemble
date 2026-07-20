#!/usr/bin/env python
"""Failure-detection + recovery: ceiling ORACLE and deployable v0 PROTOTYPE.

Implements docs/recovery_scheme_and_oracle.md Part II on one nominal policy:

  stage 1  REFERENCE  nominal rollouts on the panel; snapshot (sim state,
                      obs-history) at every replan boundary; detector stats
                      (c, e) recorded everywhere (calibration deliverable).
  stage 2  SWEEP      failures only: MC values V_nom(t) (warm-started, exact
                      history) and V_rec(t, pose) under BOTH transports —
                      R_tp teleport (non-physical upper rung) and R_ex
                      executed lift-translate-descend return. Two-phase
                      estimation (search K_sel, winners re-evaluated with
                      fresh seeds at K_eval) kills the winner's curse.
  stage 3  CONTROLS   re-roll control J_reroll (oracle-timed do-nothing with
                      fresh noise) + false-alarm arm on successes (Delta<=0).
  stage 4  PROTOTYPE  the deployable v0 switched system, closed loop, paired:
                      fire on c>kappa_c & e<=kappa_e at replan boundaries
                      (99th-pct demo thresholds), executed return, resume.

Diagnostic tier: single policy seed, dev panel. Never used for training or
checkpoint selection. Teleport results labeled non-physical.

    python scripts/recovery_oracle.py --smoke          # 3-episode dry run
    python scripts/recovery_oracle.py                  # full (overnight)
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

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel, panel_episodes  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import ReturnController, SupportIndex  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.sim.rollout import ObservationAdapter  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

STANDALONE = {"policy": {"num_candidates": 1}, "components": [],
              "selection": {"type": "candidate_zero"}, "execution": {}}
MAX_STEPS = 100


# ---------- sim-state + rollout plumbing -----------------------------------
def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def snap_state(u):
    d = u.get_state_dict()
    return {g: {k: v.clone() for k, v in grp.items()} for g, grp in d.items()}


def restore_state(env, u, snap):
    # No env.reset needed per restore: set_state_dict fully overwrites the
    # relevant sim state and the controller is stateless (use_target=False) —
    # validated by the glued drive-target capture (14k teleports, one reset).
    u.set_state_dict({g: {k: v.clone() for k, v in grp.items()} for g, grp in snap.items()})


def teleport_robot(u, qpos7):
    t = u.get_state_dict()
    art = t["articulations"][next(iter(t["articulations"]))]
    art[..., -14:-7] = torch.as_tensor(qpos7, dtype=art.dtype, device=art.device)
    art[..., -7:] = 0.0
    u.set_state_dict(t)


def _success(info) -> bool:
    s = info.get("success")
    return s is not None and bool(_np(s).reshape(-1)[0])


class Runner:
    """Obs/action plumbing around one env + one system (mirrors sim/rollout)."""

    def __init__(self, env, system):
        self.env, self.u, self.system = env, env.unwrapped, system
        low = np.asarray(self.u.single_action_space.low, dtype=np.float32)
        high = np.asarray(self.u.single_action_space.high, dtype=np.float32)
        self.low, self.high = low, high
        self.obs_adapter = ObservationAdapter(action_dim=low.shape[0])

    def state31(self, raw) -> np.ndarray:
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
        """Roll the system for <= remaining steps; True on success-once."""
        for _ in range(int(remaining)):
            raw, _r, _te, _tr, info = self.act_env(raw)
            if _success(info):
                return True
        return False

    def step_raw(self, a6: np.ndarray):
        a = np.clip(np.asarray(a6, np.float32).reshape(-1), self.low, self.high)
        return self.env.step(torch.as_tensor(a.reshape(1, -1)))


# ---------- value estimators ------------------------------------------------
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
        rn.start(seed0 + k)  # fresh internal state (recovery invalidates history)
        wins += rn.run(rn.u.get_obs(), remaining)
    return wins / K


def v_rec_ex(rn: Runner, snapb, tip_cand, remaining, K, seed0, tret_max=16):
    wins, trets, bumps = 0, [], []
    for k in range(K):
        restore_state(rn.env, rn.u, snapb)
        raw = rn.u.get_obs()
        s = rn.state31(raw)
        block0 = s[24:27].copy()
        ctl = ReturnController(tip_cand, max_steps=tret_max)
        used, once = 0, False
        while used < remaining and not ctl.done(rn.state31(raw)[14:17]):
            raw, _r, _te, _tr, info = rn.step_raw(ctl.action(rn.state31(raw)[14:17]))
            used += 1
            once = once or _success(info)
        s = rn.state31(raw)
        bumps.append(float(np.linalg.norm(s[24:27] - block0)))
        trets.append(used)
        rn.start(seed0 + k)
        wins += once or rn.run(raw, remaining - used)
    return wins / K, float(np.mean(trets)), float(np.mean(bumps))


# ---------- main ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/active_min/overnight/flow/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "data/active_min/subset_0400.h5"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs/active_min/recovery_oracle"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k-nom", type=int, default=4)
    ap.add_argument("--k-sel", type=int, default=3)
    ap.add_argument("--k-eval", type=int, default=10)
    ap.add_argument("--m-poses", type=int, default=3)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.k_nom, args.k_sel, args.k_eval, args.m_poses = 3, 1, 1, 2, 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # support index + thresholds from the TRAINING demos (train-nothing v0)
    reader = DatasetReader(args.dataset)
    states, ep_idx = [], []
    for j, eid in enumerate(reader.episode_ids):
        ep = reader.episode(eid)
        states.append(ep.state)
        ep_idx.append(np.full(len(ep.state), j))
    index = SupportIndex(np.concatenate(states), np.concatenate(ep_idx))
    kappa_c, kappa_e = index.calibrate(percentile=args.percentile)
    print(f"[recovery] thresholds: kappa_c={kappa_c:.4f} kappa_e={kappa_e:.4f} "
          f"(L={index.L:.3f}, lam={index.lam:.3f})", flush=True)

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode=None)
    system = build_system(STANDALONE, policy, [])
    rn = Runner(env, system)
    H_a = system.action_horizon
    H_o = policy.meta.obs_horizon
    eps = list(panel_episodes(make_panel("diagnostic")))[: args.n]

    # ---- stage 1: reference run + snapshots --------------------------------
    print(f"[recovery] REFERENCE: {len(eps)} episodes, boundary grid H_a={H_a}", flush=True)
    episodes = []
    for pe in eps:
        rn.start(pe.policy_sampling_seed)
        raw, _ = env.reset(seed=pe.env_seed)
        bounds, once, step = [], False, 0
        while step < MAX_STEPS:
            if step % H_a == 0:
                s31 = rn.state31(raw)
                c, e = index.stats(s31)
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
        episodes.append({"env_seed": pe.env_seed, "seed": pe.policy_sampling_seed,
                         "success": once, "bounds": bounds})
    n_succ = sum(e["success"] for e in episodes)
    print(f"[recovery] nominal J = {n_succ}/{len(eps)} = {n_succ / len(eps):.1%}", flush=True)

    # ---- stage 2: sweep on failures ---------------------------------------
    fails = [e for e in episodes if not e["success"]]
    print(f"[recovery] SWEEP over {len(fails)} failures "
          f"(K_nom={args.k_nom} K_sel={args.k_sel} K_eval={args.k_eval} M={args.m_poses})",
          flush=True)
    rows = []
    for i, e in enumerate(fails):
        cur = {"env_seed": e["env_seed"], "grid": []}
        for b in e["bounds"]:
            remaining = MAX_STEPS - b["t"]
            seed0 = 100000 + e["env_seed"] * 1000 + b["t"]
            cands = index.candidates(_snap_state31(rn, b), m=args.m_poses)
            vn = v_nom(rn, b["snap"], b["frames"], remaining, args.k_nom, seed0)
            tp = [v_rec_tp(rn, b["snap"], q, remaining, args.k_sel, seed0 + 71 * (j + 1))
                  for j, (q, _t) in enumerate(cands)]
            ex = [v_rec_ex(rn, b["snap"], t3, remaining, args.k_sel, seed0 + 91 * (j + 1))
                  for j, (_q, t3) in enumerate(cands)]
            cur["grid"].append({
                "t": b["t"], "c": b["c"], "e": b["e"], "v_nom": vn,
                "v_tp": [float(x) for x in tp],
                "v_ex": [float(x[0]) for x in ex],
                "t_ret": [float(x[1]) for x in ex],
                "bump": [float(x[2]) for x in ex],
                "cand_qpos": [q.tolist() for q, _ in cands],
                "cand_tip": [t.tolist() for _, t in cands],
            })
        # phase 2: re-evaluate winners with fresh seeds
        g = cur["grid"]
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
                                   MAX_STEPS - g[bi]["t"], args.k_eval,
                                   910000 + e["env_seed"])[0]
        bi = int(np.argmax([r["v_nom"] for r in g]))
        cur["reroll_final"] = v_nom(rn, e["bounds"][bi]["snap"], e["bounds"][bi]["frames"],
                                    MAX_STEPS - g[bi]["t"], args.k_eval,
                                    920000 + e["env_seed"])
        rows.append(cur)
        print(f"[sweep] {i + 1}/{len(fails)} seed={e['env_seed']}: "
              f"tp={cur['tp_final']:.2f} ex={cur['ex_final']:.2f} "
              f"reroll={cur['reroll_final']:.2f}", flush=True)

    # ---- stage 3: false-alarm arm -----------------------------------------
    fa = []
    for e in [x for x in episodes if x["success"]][:10]:
        for b in [e["bounds"][len(e["bounds"]) // 3], e["bounds"][2 * len(e["bounds"]) // 3]]:
            remaining = MAX_STEPS - b["t"]
            seed0 = 800000 + e["env_seed"] * 100 + b["t"]
            cands = index.candidates(_snap_state31(rn, b), m=1)
            vn = v_nom(rn, b["snap"], b["frames"], remaining, 3, seed0)
            vr = v_rec_tp(rn, b["snap"], cands[0][0], remaining, 3, seed0 + 7)
            fa.append({"env_seed": e["env_seed"], "t": b["t"],
                       "delta": float(vr - vn)})
    print(f"[recovery] false-alarm arm: mean Delta on successes = "
          f"{np.mean([x['delta'] for x in fa]) if fa else float('nan'):+.3f}", flush=True)

    # ---- stage 4: deployable prototype (paired) ---------------------------
    print("[recovery] PROTOTYPE (executed return, N_max=1)", flush=True)
    proto = []
    for pe in eps:
        rn.start(pe.policy_sampling_seed)
        raw, _ = env.reset(seed=pe.env_seed)
        once, fired, fire_t, step = False, False, None, 0
        while step < MAX_STEPS:
            if (not fired) and step % H_a == 0:
                s31 = rn.state31(raw)
                c, ev = index.stats(s31)
                if c > kappa_c and ev <= kappa_e:
                    fired, fire_t = True, step
                    cand = min(index.candidates(s31, m=args.m_poses),
                               key=lambda qt: float(np.linalg.norm(qt[0] - s31[:7])))
                    ctl = ReturnController(cand[1])
                    while step < MAX_STEPS and not ctl.done(rn.state31(raw)[14:17]):
                        raw, _r, _te, _tr, info = rn.step_raw(
                            ctl.action(rn.state31(raw)[14:17]))
                        once = once or _success(info)
                        step += 1
                    rn.start(pe.policy_sampling_seed + 5555)
                    continue
            raw, _r, _te, _tr, info = rn.act_env(raw)
            once = once or _success(info)
            step += 1
        proto.append({"env_seed": pe.env_seed, "success": once,
                      "fired": fired, "fire_t": fire_t})
    env.close()

    # ---- summary -----------------------------------------------------------
    n = len(episodes)
    J_nom = n_succ / n
    tp_sum = (n_succ + sum(r["tp_final"] for r in rows)) / n
    ex_sum = (n_succ + sum(r["ex_final"] for r in rows)) / n
    rr_sum = (n_succ + sum(r["reroll_final"] for r in rows)) / n
    J_proto = sum(p["success"] for p in proto) / n
    fired = sum(p["fired"] for p in proto)
    paired = {"saved": sum(1 for p, e in zip(proto, episodes) if p["success"] and not e["success"]),
              "broken": sum(1 for p, e in zip(proto, episodes) if not p["success"] and e["success"])}
    summary = {
        "policy": args.policy_checkpoint, "panel": "diagnostic", "n": n,
        "H_a": H_a, "kappa_c": kappa_c, "kappa_e": kappa_e,
        "J_nom": J_nom, "J_nom_ci": list(wilson_interval(n_succ, n)),
        "J_reroll": rr_sum, "J_ceil_tp": tp_sum, "J_ceil_ex": ex_sum,
        "headroom_tp_vs_reroll": tp_sum - rr_sum,
        "headroom_ex_vs_reroll": ex_sum - rr_sum,
        "J_prototype": J_proto, "prototype_fired": fired,
        "prototype_paired": paired,
        "false_alarm_mean_delta": float(np.mean([x["delta"] for x in fa])) if fa else None,
        "runtime_min": (time.time() - t_start) / 60.0,
    }
    save_json({"summary": summary, "failures": rows, "false_alarm": fa,
               "prototype": proto,
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
    print(f"J_proto     {J_proto:.1%}   fired {fired}/{n}  saved/broken "
          f"{paired['saved']}/{paired['broken']}")
    print(f"[recovery] wrote {out / 'recovery_oracle.json'}  "
          f"({summary['runtime_min']:.0f} min)")
    return 0


def _snap_state31(rn: Runner, b) -> np.ndarray:
    """31-dim state at a saved boundary (restore-free: rebuilt from snapshot
    is unnecessary — we recorded c/e already; for candidate retrieval we
    restore)."""
    restore_state(rn.env, rn.u, b["snap"])
    return rn.state31(rn.u.get_obs())


if __name__ == "__main__":
    sys.exit(main())

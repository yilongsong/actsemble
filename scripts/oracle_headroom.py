#!/usr/bin/env python
"""Rollout-oracle selection headroom.

Measures the CEILING of per-step selection among the policy's K=16 candidates,
using the simulator as the value function (sparse reward is a non-problem with a
perfect model): at each replan, branch the sim on every candidate, continue each
branch to the episode end under the base policy (candidate-zero), and keep the
candidate whose branch actually SUCCEEDS (tie-break by final goal coverage).
Commit its H_a actions, advance, repeat. This is a rollout policy (one-step
policy improvement) — a strong, well-defined LOWER bound on the true (tree-
search) ceiling that uses the real success signal, not a proxy.

Reports the triplet candidate_zero (floor) / verifier (current) / Oracle@K
(ceiling) on a shared panel, so:  headroom = Oracle − candidate_zero  and
captured = (verifier − candidate_zero) / headroom.

  run --start S --count N --device cuda   # oracle on panel episodes [S, S+N)
  analyze                                 # merge shards + pair vs cz/verifier

NOT the frozen protocol — a diagnostic upper bound that uses privileged sim
access. Reuses the hold-trimmed policy/verifier under outputs/active_min.
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

from actsemble.policies.diffusion.policy import DiffusionPolicy  # noqa: E402
from actsemble.seed import derive_seed  # noqa: E402
from actsemble.sim.action_adapter import ActionAdapter  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.sim.observation_adapter import ObservationAdapter  # noqa: E402
from actsemble.types import RobotAction  # noqa: E402
from actsemble.evaluation.panels import Panel, make_panel, panel_episodes  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402

POLICY = REPO / "outputs/active_min/policy_seed_0/selected_policy.pt"
VERIF = REPO / "outputs/active_min/verifier_seed_0/selected_verifier.pt"
OUT = REPO / "outputs/active_min/oracle"
PANEL = make_panel("diagnostic")  # registered diagnostic bench (root 20000, 300 eps)
K = 16
MAX_STEPS = 100
CZ_JSON = REPO / "outputs/active_min/compare/candidate_zero.json"
VE_JSON = REPO / "outputs/active_min/compare/verifier_argmax.json"


def _clone(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


def _to_bool(v):
    if torch.is_tensor(v):
        return bool(v.reshape(-1)[0].item())
    return bool(np.asarray(v).reshape(-1)[0])


def _obs_window(hist, h_o):
    frames = list(hist)[-h_o:]
    while len(frames) < h_o:
        frames.insert(0, frames[0])
    return np.stack(frames, axis=0)


class OracleRollout:
    """Two-env rollout oracle.

    ``env`` (clean) holds the committed trajectory and is ONLY ever reset +
    stepped forward — never ``set_state_dict``-restored — so its rollout is
    bitwise-faithful (like the base policy) and the measured success matches a
    clean deployment replay. ``benv`` (scratch) is the ONLY env that gets
    branched (restored + rolled out) for candidate scoring; its GPU-nondeter-
    minism drift therefore affects candidate SELECTION only, never the measured
    outcome. This fixes the earlier bug where branching drift (~1e-2) leaked
    into the committed rollout and optimistically flipped near-boundary episodes.
    """

    def __init__(self, policy: DiffusionPolicy, env, benv, device, n_candidates=K, n_continuations=1):
        self.p = policy
        self.env = env                      # clean committed env (forward-only)
        self.benv = benv                    # scratch branch env (restored freely)
        self.u = env.unwrapped
        self.bu = benv.unwrapped
        self.K = int(n_candidates)          # candidates per replan (proposal axis)
        self.M = int(n_continuations)       # Monte-Carlo continuations per candidate (variance axis)
        m = policy.meta
        self.h_o, self.h_a, self.h_p = m.obs_horizon, m.action_horizon, m.prediction_horizon
        low = np.asarray(self.u.single_action_space.low, dtype=np.float32)
        high = np.asarray(self.u.single_action_space.high, dtype=np.float32)
        self.obs = ObservationAdapter(action_dim=low.shape[0])
        self.act = ActionAdapter(low, high)
        self.ckpt = policy.checkpoint_hash

    def _state(self, raw_obs):
        return np.asarray(self.obs.observe(raw_obs).state, dtype=np.float32).reshape(-1)

    def _step(self, e, action_vec):
        env_action = self.act.adapt(RobotAction(value=np.asarray(action_vec, dtype=np.float32)))
        raw, _, term, trunc, info = e.step(env_action)
        succ = _to_bool(info["success"]) if "success" in info else False
        return self._state(raw), succ, _to_bool(term) or _to_bool(trunc)

    def _cov(self, e):
        return float(e.unwrapped.pseudo_render_intersection().item())

    def _set_benv(self, state):
        """Restore the scratch env AND reset its episode clock — set_state_dict
        does NOT reset elapsed_steps, so without this every branch after the
        env's 100-step budget is exhausted truncates after one step (value
        estimates become garbage). Manual `st >= MAX_STEPS` bounds the rollout."""
        self.bu.set_state_dict(_clone(state))
        esc = getattr(self.bu, "elapsed_steps", None)
        if torch.is_tensor(esc):
            self.bu._elapsed_steps = torch.zeros_like(esc)  # backing field (property has no setter)

    def _img(self):
        f = self.env.render()
        if torch.is_tensor(f):
            f = f.detach().cpu().numpy()
        f = np.asarray(f)
        if f.ndim == 4:
            f = f[0]
        return f.astype(np.uint8)

    def base_episode(self, env_seed, policy_sampling_seed, render=False):
        """Candidate-zero base policy (single sample), with trajectory capture."""
        raw, _ = self.env.reset(seed=int(env_seed))
        s = self._state(raw)
        hist = [s]
        obj_xy, tcp_xy, cov = [s[24:26].copy()], [s[14:16].copy()], [self._cov(self.env)]
        self.frames = [self._img()] if render else []
        step = 0
        success_once = False
        ridx = 0
        while step < MAX_STEPS and not success_once:
            seed = derive_seed(int(policy_sampling_seed), "candidates", self.ckpt, ridx)
            chunk = self._sample(hist, 1, seed)[0]
            done = False
            for a in chunk[: self.h_a]:
                s, sc, d = self._step(self.env, a)
                hist.append(s)
                obj_xy.append(s[24:26].copy()); tcp_xy.append(s[14:16].copy()); cov.append(self._cov(self.env))
                if render:
                    self.frames.append(self._img())
                step += 1
                if sc:
                    success_once = True
                if success_once or d:
                    done = True
                    break
            ridx += 1
            if done and not success_once:
                break
        return self._capture_row(env_seed, policy_sampling_seed, success_once, step,
                                 obj_xy, tcp_xy, cov, hist[-1])

    def _capture_row(self, env_seed, pss, success, step, obj_xy, tcp_xy, cov, last_state):
        return {"env_seed": int(env_seed), "policy_sampling_seed": int(pss),
                "success_once": bool(success), "num_steps": int(step),
                "obj_xy": np.asarray(obj_xy, np.float32).round(4).tolist(),
                "tcp_xy": np.asarray(tcp_xy, np.float32).round(4).tolist(),
                "coverage": np.asarray(cov, np.float32).round(4).tolist(),
                "goal_xy": np.asarray(last_state[21:23], np.float32).round(4).tolist(),
                "final_obj_quat": np.asarray(last_state[27:31], np.float32).round(5).tolist()}

    def _sample(self, hist, num, seed):
        win = _obs_window(hist, self.h_o)
        gen = self.p.new_generator(seed)
        c = self.p.sample_action_chunks(win, num_samples=num, generator=gen)
        return c.detach().cpu().numpy().astype(np.float32)  # [num, h_p, 6]

    def _continue(self, e, hist, step, cont_seed):
        """Base policy (single-sample) to episode end IN env ``e`` (the scratch
        branch env); returns (success, coverage, step)."""
        h = list(hist)
        succ = False
        ridx = 0
        while step < MAX_STEPS:
            chunk = self._sample(h, 1, cont_seed + 7919 * ridx)[0]  # [h_p,6]
            done = False
            for a in chunk[: self.h_a]:
                s, sc, d = self._step(e, a)
                h.append(s)
                step += 1
                if sc:
                    succ = True
                if succ or step >= MAX_STEPS or d:
                    done = True
                    break
            ridx += 1
            if succ or done and step >= MAX_STEPS:
                break
            if done and not succ:  # env terminated without success
                break
        return succ, self._cov(e), step

    def episode(self, env_seed, policy_sampling_seed, capture=False, render=False):
        raw, _ = self.env.reset(seed=int(env_seed))     # clean committed env
        self.benv.reset(seed=int(env_seed))             # scratch env (config; overwritten by set_state)
        s0 = self._state(raw)
        hist = [s0]
        obj_xy, tcp_xy, cov = [s0[24:26].copy()], [s0[14:16].copy()], [self._cov(self.env)]
        self.frames = [self._img()] if render else []
        exec_a = []
        step = 0
        success_once = False
        replan_index = 0
        replans = []
        while step < MAX_STEPS and not success_once:
            # clean committed state to branch FROM (a read; does not perturb env)
            snap = _clone(self.u.get_state_dict())
            hist_snap = list(hist)
            seed = derive_seed(int(policy_sampling_seed), "candidates", self.ckpt, replan_index)
            cands = self._sample(hist, self.K, seed)  # [K, h_p, 6]
            best_k, best_key = 0, (-1.0, -2.0)
            per_k_succ = []
            for k in range(self.K):
                if not np.isfinite(cands[k]).all():
                    per_k_succ.append(False)
                    continue
                # (1) execute this candidate's chunk in the SCRATCH env
                self._set_benv(snap)
                h = list(hist_snap)
                st = step
                sc = False
                done = False
                for a in cands[k][: self.h_a]:
                    s, s_ok, d = self._step(self.benv, a)
                    h.append(s)
                    st += 1
                    if s_ok:
                        sc = True
                    if sc or st >= MAX_STEPS or d:
                        done = True
                        break
                if sc or done:  # chunk itself ended the episode
                    v_succ, v_cov = (1.0 if sc else 0.0), self._cov(self.benv)
                else:
                    # (2) Monte-Carlo: M base-policy continuations to the end.
                    # Common random seeds across candidates (k-independent) both
                    # reduce variance and isolate the candidate's own effect.
                    post = _clone(self.bu.get_state_dict())
                    succs, covs = [], []
                    for mi in range(self.M):
                        self._set_benv(post)
                        s_m, c_m, _ = self._continue(self.benv, list(h), st, cont_seed=seed + 131 + 997 * mi)
                        succs.append(1.0 if s_m else 0.0); covs.append(c_m)
                    v_succ, v_cov = float(np.mean(succs)), float(np.mean(covs))
                per_k_succ.append(v_succ >= 0.5)
                key = (v_succ, v_cov)
                if key > best_key:
                    best_key, best_k = key, k
            # commit winner's H_a actions on the CLEAN env (forward-only, no restore:
            # it is already at the committed state, untouched by the scratch branching)
            obj_before = hist_snap[-1][24:26].copy()
            for a in cands[best_k][: self.h_a]:
                exec_a.append(np.asarray(a, dtype=np.float32).copy())
                s, sc, d = self._step(self.env, a)
                hist.append(s)
                obj_xy.append(s[24:26].copy()); tcp_xy.append(s[14:16].copy()); cov.append(self._cov(self.env))
                if render:
                    self.frames.append(self._img())
                step += 1
                if sc:
                    success_once = True
                if success_once or d:
                    break
            moved_T = float(np.linalg.norm(hist[-1][24:26] - obj_before))
            replans.append({"replan_index": replan_index, "selected_index": int(best_k),
                            "n_branch_success": int(sum(per_k_succ)),
                            "best_branch_success": bool(best_key[0]),
                            "changed_from_zero": bool(best_k != 0),
                            "obj_moved": moved_T})
            replan_index += 1
        row = {"env_seed": int(env_seed), "policy_sampling_seed": int(policy_sampling_seed),
               "success_once": bool(success_once), "num_steps": int(step), "replans": replans}
        if capture:
            row.update(self._capture_row(env_seed, policy_sampling_seed, success_once, step,
                                         obj_xy, tcp_xy, cov, hist[-1]))
            row["exec_actions"] = np.asarray(exec_a, dtype=np.float32).tolist()
        return row


def _tag(k, m):
    return "" if (k == K and m == 1) else f"K{k}_M{m}_"


def _make_envs(policy, render=False):
    """(clean committed env, scratch branch env). Clean env optionally renders."""
    def mk(rm):
        return make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                        sim_backend=policy.meta.simulation_backend, obs_mode="state", render_mode=rm)
    return mk("rgb_array" if render else None), mk(None)


def cmd_run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    eps = panel_episodes(PANEL)
    start, count = int(args.start), int(args.count)
    k, m = int(args.k), int(args.m)
    sel = eps[start:start + count]
    dest = OUT / f"shard_{_tag(k, m)}{start:04d}_{start + count:04d}.json"
    if dest.exists() and not args.force:
        print(f"[oracle] {dest.name} exists; skipping"); return 0
    policy = DiffusionPolicy.from_checkpoint(str(POLICY), device=args.device or "cuda", use_ema=True)
    env, benv = _make_envs(policy)
    roll = OracleRollout(policy, env, benv, args.device or "cuda", n_candidates=k, n_continuations=m)
    rows = []
    t0 = time.time()
    for i, ep in enumerate(sel):
        r = roll.episode(ep.env_seed, ep.policy_sampling_seed)
        rows.append(r)
        if (i + 1) % 5 == 0 or i == 0:
            rate = np.mean([x["success_once"] for x in rows])
            print(f"[oracle K{k}M{m}] {start+i+1}/{start+len(sel)} run={rate:.1%} "
                  f"({(time.time()-t0)/(i+1):.1f}s/ep)", flush=True)
    env.close(); benv.close()
    save_json({"panel": PANEL.to_dict(), "start": start, "count": len(sel),
               "num_candidates": k, "n_continuations": m, "episodes": rows}, dest)
    print(f"[oracle K{k}M{m}] wrote {dest.name}: {np.mean([r['success_once'] for r in rows]):.1%}")
    return 0


def cmd_capture(args):
    """Re-run with trajectory capture for failure-mode visualization.
    --mode base = candidate_zero single-sample; --mode oracle = rollout oracle."""
    cap_dir = OUT / "capture"; cap_dir.mkdir(parents=True, exist_ok=True)
    eps = panel_episodes(PANEL)
    start, count, mode = int(args.start), int(args.count), args.mode
    sel = eps[start:start + count]
    dest = cap_dir / f"{mode}_{start:04d}_{start + count:04d}.json"
    if dest.exists() and not args.force:
        print(f"[capture] {dest.name} exists; skipping"); return 0
    policy = DiffusionPolicy.from_checkpoint(str(POLICY), device=args.device or "cuda", use_ema=True)
    env, benv = _make_envs(policy)
    roll = OracleRollout(policy, env, benv, args.device or "cuda")
    rows = []
    t0 = time.time()
    for i, ep in enumerate(sel):
        if mode == "base":
            r = roll.base_episode(ep.env_seed, ep.policy_sampling_seed)
        else:
            r = roll.episode(ep.env_seed, ep.policy_sampling_seed, capture=True)
        rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"[capture:{mode}] {start+i+1}/{start+len(sel)} ({(time.time()-t0)/(i+1):.1f}s/ep)", flush=True)
    env.close(); benv.close()
    save_json({"mode": mode, "panel": PANEL.to_dict(), "episodes": rows}, dest)
    print(f"[capture:{mode}] wrote {dest.name} ({len(rows)} eps, {np.mean([r['success_once'] for r in rows]):.1%})")
    return 0


def _categorize(ep):
    if ep["success_once"]:
        return "success"
    cov = np.asarray(ep["coverage"]); obj = np.asarray(ep["obj_xy"]); goal = np.asarray(ep["goal_xy"])
    disp = np.linalg.norm(obj - obj[0], axis=1).max(); pos = np.linalg.norm(obj[-1] - goal)
    if disp < 0.02:
        return "never_engaged"
    if cov.max() >= 0.75:
        return "near_miss"
    if pos <= 0.05:
        return "misaligned"
    return "mispositioned"


def _filmstrip(path, es, label, fb, rb, fo, ro):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 6
    fig, ax = plt.subplots(2, ncol, figsize=(2.2 * ncol, 4.8))
    fig.suptitle(f"seed {es}  ({label})   base={'SUCCESS' if rb['success_once'] else 'FAIL'} "
                 f"(maxcov {np.asarray(rb['coverage']).max():.2f})   →   "
                 f"oracle={'SUCCESS' if ro['success_once'] else 'FAIL'} "
                 f"(maxcov {np.asarray(ro['coverage']).max():.2f})", fontsize=11, fontweight="bold")
    for row, (frames, tag) in enumerate([(fb, "base"), (fo, "oracle")]):
        idx = np.linspace(0, len(frames) - 1, ncol).astype(int)
        for c, fi in enumerate(idx):
            ax[row, c].imshow(frames[fi]); ax[row, c].axis("off")
            ax[row, c].set_title(f"{tag} t={fi}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path, dpi=110); plt.close(fig)


def cmd_video(args):
    from actsemble.evaluation.video import save_video
    vdir = OUT / "videos"; vdir.mkdir(parents=True, exist_ok=True)
    policy = DiffusionPolicy.from_checkpoint(str(POLICY), device=args.device or "cuda", use_ema=True)
    env, benv = _make_envs(policy, render=True)
    roll = OracleRollout(policy, env, benv, args.device or "cuda")
    pss = {e.env_seed: e.policy_sampling_seed for e in panel_episodes(PANEL)}

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        labels = {s: "custom" for s in seeds}
    else:  # auto-pick one of each illustrative case from the capture data
        cap = OUT / "capture"
        base = {e["env_seed"]: e for e in load_json(cap / "base_0000_0300.json")["episodes"]}
        orac = {}
        for f in sorted(cap.glob("oracle_*.json")):
            for e in load_json(f)["episodes"]:
                orac[e["env_seed"]] = e
        common = sorted(set(base) & set(orac))

        def pick(pred):
            return next((es for es in common if pred(es)), None)
        rn = pick(lambda es: _categorize(base[es]) == "near_miss" and orac[es]["success_once"])
        rm = pick(lambda es: _categorize(base[es]) == "mispositioned" and orac[es]["success_once"])
        of = pick(lambda es: _categorize(orac[es]) == "mispositioned" and not orac[es]["success_once"])
        labels = {rn: "recovered_near_miss", rm: "recovered_mispositioned", of: "oracle_fail_proposal_limited"}
        seeds = [s for s in (rn, rm, of) if s is not None]

    for es in seeds:
        rb = roll.base_episode(es, pss[es], render=True); fb = list(roll.frames)
        save_video(fb, vdir / f"seed{es}_{labels[es]}_base.mp4")
        ro = roll.episode(es, pss[es], capture=True, render=True); fo = list(roll.frames)
        save_video(fo, vdir / f"seed{es}_{labels[es]}_oracle.mp4")
        _filmstrip(vdir / f"seed{es}_{labels[es]}.png", es, labels[es], fb, rb, fo, ro)
        print(f"[video] seed {es} {labels[es]}: base={'S' if rb['success_once'] else 'F'} "
              f"oracle={'S' if ro['success_once'] else 'F'}  ({len(fb)}|{len(fo)} frames)", flush=True)
    env.close(); benv.close()
    print(f"[video] wrote mp4s + filmstrips to {vdir}")
    return 0


def _mcnemar(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    b10 = int(((a == 1) & (b == 0)).sum()); b01 = int(((a == 0) & (b == 1)).sum())
    disc = b10 + b01
    z = (b10 - b01) / np.sqrt(disc) if disc else 0.0
    return {"a_wins": b10, "b_wins": b01, "z": float(z), "sig": bool(abs(z) > 1.96)}


def _load_oracle(k, m):
    tag = _tag(k, m)
    if tag:
        shards = sorted(OUT.glob(f"shard_{tag}*.json"))
    else:  # baseline K16 M1: files WITHOUT a K*_M* tag (variants start with shard_K)
        shards = [p for p in sorted(OUT.glob("shard_*.json")) if not p.name.startswith("shard_K")]
    rows = {}
    for s in shards:
        for r in load_json(s)["episodes"]:
            rows[r["env_seed"]] = r
    return rows


def cmd_analyze(args):
    k, m = int(args.k), int(args.m)
    rows = _load_oracle(k, m)
    env_seeds = sorted(rows)
    oracle = {es: rows[es]["success_once"] for es in env_seeds}
    base_rows = _load_oracle(K, 1) if _tag(k, m) else rows  # baseline oracle for the shift

    cz = load_json(CZ_JSON); ve = load_json(VE_JSON)
    eps = panel_episodes(PANEL)
    es_order = [e.env_seed for e in eps]
    cz_by = dict(zip(es_order, cz["successes"]))
    ve_by = dict(zip(es_order, ve["successes"]))

    paired = [es for es in env_seeds if es in cz_by and es in ve_by]
    O = np.array([oracle[es] for es in paired], int)
    C = np.array([cz_by[es] for es in paired], int)
    V = np.array([ve_by[es] for es in paired], int)
    B = np.array([base_rows[es]["success_once"] for es in paired if es in base_rows], int)
    n = len(paired)
    rO, rC, rV = O.mean(), C.mean(), V.mean()
    rB = B.mean() if len(B) == n else float("nan")
    headroom = rO - rC
    captured = (rV - rC) / headroom if headroom > 1e-9 else float("nan")

    # contact-phase diagnostic: where does the oracle's selection change from
    # candidate 0, and is it at contact chunks (obj moved) vs free-space?
    ch_contact = ch_free = tot_contact = tot_free = 0
    for es in paired:
        for rp in rows[es]["replans"]:
            contact = rp["obj_moved"] > 1e-3
            tot_contact += contact; tot_free += (not contact)
            if rp["changed_from_zero"]:
                ch_contact += contact; ch_free += (not contact)

    contrasts = {
        "oracle_vs_cz": _mcnemar(O, C),
        "oracle_vs_verifier": _mcnemar(O, V),
        "verifier_vs_cz": _mcnemar(V, C),
    }
    if _tag(k, m) and len(B) == n:  # variant vs baseline oracle: the ceiling shift
        contrasts["oracle_vs_baseline_oracle"] = _mcnemar(O, B)
    report = {
        "note": "Rollout-oracle headroom (diagnostic, privileged sim access).",
        "config": {"num_candidates": k, "n_continuations": m,
                   "continuation": "candidate_zero base policy"},
        "panel": PANEL.to_dict(), "n_paired": n,
        "success": {"candidate_zero": float(rC), "verifier": float(rV), "oracle": float(rO),
                    "baseline_oracle_K16_M1": float(rB)},
        "headroom_oracle_minus_cz": float(headroom),
        "ceiling_shift_vs_baseline_oracle": float(rO - rB) if rB == rB else None,
        "verifier_gain_over_cz": float(rV - rC),
        "fraction_of_headroom_captured_by_verifier": float(captured),
        "contrasts": contrasts,
        "selection_change": {
            "changed_at_contact": ch_contact, "changed_at_freespace": ch_free,
            "total_contact_replans": tot_contact, "total_freespace_replans": tot_free,
        },
    }
    save_json(report, OUT / f"headroom_{_tag(k, m) or 'K16_M1_'}.json".replace("__", "_"))
    print(f"\n=== Rollout-oracle headroom  K={k} M={m}  (n={n} paired episodes) ===")
    print(f"  candidate_zero : {rC:.1%}")
    print(f"  verifier       : {rV:.1%}")
    print(f"  Oracle@{k} (M={m}) : {rO:.1%}")
    if _tag(k, m) and rB == rB:
        print(f"  baseline Oracle@16 (M=1) on same eps : {rB:.1%}   ceiling shift {rO - rB:+.1%}")
    print(f"  headroom (oracle - cz)         : {headroom:+.1%}")
    print(f"  verifier gain  (ver - cz)      : {rV - rC:+.1%}")
    print(f"  fraction of headroom captured  : {captured:.0%}")
    for name, c in contrasts.items():
        print(f"  {name:26s} z={c['z']:+.2f} ({'sig' if c['sig'] else 'ns'}; +{c['a_wins']}/-{c['b_wins']})")
    sc = report["selection_change"]
    print(f"  oracle changed pick: contact {sc['changed_at_contact']}/{sc['total_contact_replans']}, "
          f"free-space {sc['changed_at_freespace']}/{sc['total_freespace_replans']}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["run", "analyze", "capture", "video"])
    p.add_argument("--start", default=0)
    p.add_argument("--count", default=75)
    p.add_argument("--k", default=K, help="candidates per replan (proposal axis)")
    p.add_argument("--m", default=1, help="Monte-Carlo continuations per candidate (variance axis)")
    p.add_argument("--mode", default="oracle", choices=["base", "oracle"])
    p.add_argument("--seeds", default=None, help="video: comma-separated env_seeds (else auto-pick)")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    return {"run": cmd_run, "analyze": cmd_analyze, "capture": cmd_capture,
            "video": cmd_video}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())

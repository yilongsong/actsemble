#!/usr/bin/env python
"""Debug visualization for the recovery v0 prototype (E/F track).

Replays chosen diagnostic-panel episodes two ways with full instrumentation:

  * NOMINAL   — the baseline policy, no intervention: watch the drift.
  * PROTOTYPE — the exact stage-4 switched system from recovery_oracle.py
                (same seeds, same fire rule c>kappa_c & e<=kappa_e at replan
                boundaries, same nearest-candidate ReturnController takeover,
                same history-reset handback, N_max=1).

Per control step it records the support statistics (c, e) — the oracle only
logged replan boundaries — plus the detector decision points, the takeover
span (who has agency), tip-to-target distance during the return, block bump,
and two diagnostics for the transport gap:

  * c at HANDBACK: did the executed return actually restore support in the
    statistic that fired? (ReturnController controls the 3-DOF tip; c is
    measured on 7-DOF qpos — a 4-dim null space can stay off-manifold.)
  * c IF TELEPORTED: counterfactual kNN arithmetic (candidate qpos swapped
    into the fire-step state), no sim.

Outputs per episode (out-dir): nominal_<seed>.mp4, prototype_<seed>.mp4,
combined_<seed>.mp4 (side by side), timeline_<seed>.png (c/e traces, both
arms, thresholds, fire/handback, oracle per-boundary values from the sweep
JSON), and trace_<seed>.json (all numbers).

    python scripts/visualize_recovery.py --env-seeds 1229399060 1120782160

Diagnostic tier; physx_cuda replays are nondeterministic (~96% outcome
agreement), so realized fire steps may shift vs the recorded run — the
report prints recorded vs realized.
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

from recovery_oracle import MAX_STEPS, STANDALONE, Runner, _np, _success  # noqa: E402

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.video import save_video  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import ReturnController, SupportIndex  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402


# ---------- frame annotation -------------------------------------------------
def _font(size: int):
    from matplotlib import font_manager
    from PIL import ImageFont

    path = font_manager.findfont("DejaVu Sans Mono")
    return ImageFont.truetype(path, size)


HEADER = 150


def _grab_frame(env) -> np.ndarray:
    f = _np(env.render())
    f = f.reshape(f.shape[-3], f.shape[-2], f.shape[-1])
    if f.dtype != np.uint8:
        f = np.clip(f, 0, 255).astype(np.uint8)
    return f


def annotate(
    frame: np.ndarray,
    *,
    step: int,
    phase: str,
    c: float,
    e: float,
    kappa_c: float,
    kappa_e: float,
    banner: str | None = None,
    border: tuple[int, int, int] | None = None,
    extra: str = "",
) -> np.ndarray:
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    canvas = Image.new("RGB", (w, h + HEADER), (12, 12, 12))
    canvas.paste(Image.fromarray(frame), (0, HEADER))
    d = ImageDraw.Draw(canvas)
    f16, f20 = _font(16), _font(20)
    phase_color = {
        "NOMINAL": (220, 220, 220),
        "RECOVERY": (255, 160, 40),
        "POST-HANDBACK": (90, 200, 255),
    }.get(phase.split(" ")[0], (220, 220, 220))
    d.text((10, 8), f"step {step:>3}   {phase}", font=f20, fill=phase_color)
    c_col = (235, 80, 80) if c > kappa_c else (110, 220, 110)
    e_col = (235, 80, 80) if e > kappa_e else (110, 220, 110)
    d.text((10, 42), f"c = {c:6.3f}  (kappa_c {kappa_c:.3f})  robot-given-env", font=f16, fill=c_col)
    d.text((10, 66), f"e = {e:6.3f}  (kappa_e {kappa_e:.3f})  env familiarity", font=f16, fill=e_col)
    fire_ok = "FIRE-ELIGIBLE (c>k_c & e<=k_e)" if (c > kappa_c and e <= kappa_e) else ""
    d.text((10, 90), extra or fire_ok, font=f16, fill=(255, 200, 60))
    if banner:
        d.text((10, 118), banner, font=f20, fill=(255, 80, 80))
    out = np.asarray(canvas)
    if border is not None:
        out = out.copy()
        out[:6, :] = border
        out[-6:, :] = border
        out[:, :6] = border
        out[:, -6:] = border
    return out


# ---------- instrumented rollouts ---------------------------------------------
def rollout_nominal(rn, index, env, env_seed, pol_seed, kc, ke):
    rn.start(pol_seed)
    raw, _ = env.reset(seed=env_seed)
    rec = {"steps": [], "success": False, "success_t": None}
    frames = []
    for step in range(MAX_STEPS):
        s = rn.state31(raw)
        c, e = index.stats(s)
        rec["steps"].append(
            {"t": step, "c": c, "e": e, "phase": "NOMINAL",
             "boundary": step % rn.system.action_horizon == 0}
        )
        frames.append(
            annotate(_grab_frame(env), step=step, phase="NOMINAL", c=c, e=e,
                     kappa_c=kc, kappa_e=ke)
        )
        raw, _r, _te, _tr, info = rn.act_env(raw)
        if _success(info) and not rec["success"]:
            rec["success"], rec["success_t"] = True, step
    color = (60, 200, 60) if rec["success"] else (220, 50, 50)
    tail = annotate(
        _grab_frame(env), step=MAX_STEPS, phase="NOMINAL",
        c=rec["steps"][-1]["c"], e=rec["steps"][-1]["e"], kappa_c=kc, kappa_e=ke,
        banner=f"OUTCOME: {'SUCCESS' if rec['success'] else 'FAILURE'}", border=color,
    )
    frames.extend([tail] * 15)
    return rec, frames


def rollout_prototype(rn, index, env, env_seed, pol_seed, kc, ke, m_poses=3):
    """Verbatim stage-4 loop from recovery_oracle.py, instrumented."""
    rn.start(pol_seed)
    raw, _ = env.reset(seed=env_seed)
    rec = {
        "steps": [], "success": False, "success_t": None,
        "fired": False, "fire": None, "handback": None,
    }
    frames = []
    once, fired, step = False, False, 0
    H_a = rn.system.action_horizon
    while step < MAX_STEPS:
        s31 = rn.state31(raw)
        c, ev = index.stats(s31)
        if (not fired) and step % H_a == 0:
            if c > kc and ev <= ke:
                fired = True
                cands = index.candidates(s31, m=m_poses)
                dists = [float(np.linalg.norm(q - s31[:7])) for q, _t in cands]
                pick = int(np.argmin(dists))
                cand = cands[pick]
                # counterfactual: c if we could TELEPORT to the candidate qpos
                s_tp = s31.copy()
                s_tp[:7] = cand[0]
                c_tp, _ = index.stats(s_tp)
                rec["fired"] = True
                rec["fire"] = {
                    "t": step, "c": c, "e": ev,
                    "cand_qpos_dists": dists, "picked": pick,
                    "target_tip": cand[1].tolist(),
                    "c_if_teleported": c_tp,
                    "block_pos": s31[24:27].tolist(),
                }
                ctl = ReturnController(cand[1])
                block0 = s31[24:27].copy()
                # ---- takeover: ReturnController has agency ----
                while step < MAX_STEPS and not ctl.done(rn.state31(raw)[14:17]):
                    s = rn.state31(raw)
                    c2, e2 = index.stats(s)
                    tip_d = float(np.linalg.norm(s[14:17] - cand[1]))
                    rec["steps"].append(
                        {"t": step, "c": c2, "e": e2, "phase": "RECOVERY",
                         "tip_dist": tip_d, "boundary": False}
                    )
                    frames.append(
                        annotate(_grab_frame(env), step=step,
                                 phase="RECOVERY  (agency: ReturnController)",
                                 c=c2, e=e2, kappa_c=kc, kappa_e=ke,
                                 banner="FIRED" if step == rec["fire"]["t"] else None,
                                 border=(255, 140, 0),
                                 extra=f"tip->target {tip_d:.3f}  "
                                       f"(lift-translate-descend, tol 0.01)")
                    )
                    raw, _r, _te, _tr, info = rn.step_raw(
                        ctl.action(rn.state31(raw)[14:17]))
                    once = once or _success(info)
                    step += 1
                s = rn.state31(raw)
                c_hb, e_hb = index.stats(s)
                rec["handback"] = {
                    "t": step, "c": c_hb, "e": e_hb,
                    "tip_err": float(np.linalg.norm(s[14:17] - cand[1])),
                    "qpos_dist_to_cand": float(np.linalg.norm(s[:7] - cand[0])),
                    "block_bump": float(np.linalg.norm(s[24:27] - block0)),
                    "steps_used": step - rec["fire"]["t"],
                }
                rn.start(pol_seed + 5555)  # exact prototype handback (history reset)
                continue
        phase = "POST-HANDBACK" if fired else "NOMINAL"
        rec["steps"].append(
            {"t": step, "c": c, "e": ev, "phase": phase,
             "boundary": (not fired) and step % H_a == 0}
        )
        frames.append(
            annotate(_grab_frame(env), step=step,
                     phase=phase + ("  (agency: policy, fresh history)"
                                    if fired else ""),
                     c=c, e=ev, kappa_c=kc, kappa_e=ke)
        )
        raw, _r, _te, _tr, info = rn.act_env(raw)
        if _success(info) and not once:
            once = True
            rec["success_t"] = step
        step += 1
    rec["success"] = once
    color = (60, 200, 60) if once else (220, 50, 50)
    s = rn.state31(raw)
    c, ev = index.stats(s)
    tail = annotate(
        _grab_frame(env), step=MAX_STEPS,
        phase="POST-HANDBACK" if fired else "NOMINAL", c=c, e=ev,
        kappa_c=kc, kappa_e=ke,
        banner=f"OUTCOME: {'SUCCESS' if once else 'FAILURE'}", border=color,
    )
    frames.extend([tail] * 15)
    return rec, frames


# ---------- timeline figure ----------------------------------------------------
def timeline(env_seed, nom, pro, sweep_row, kc, ke, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )
    ax_c, ax_e, ax_v = axes
    t_n = [s["t"] for s in nom["steps"]]
    t_p = [s["t"] for s in pro["steps"]]

    for ax, key, kappa, label in ((ax_c, "c", kc, "c  (robot-given-env surprisal)"),
                                  (ax_e, "e", ke, "e  (env familiarity distance)")):
        ax.plot(t_n, [s[key] for s in nom["steps"]], color="#777777", lw=1.8,
                label="NOMINAL (baseline)")
        ax.plot(t_p, [s[key] for s in pro["steps"]], color="#d95f02", lw=1.8,
                label="PROTOTYPE (switched)")
        ax.axhline(kappa, color="#c02020", ls="--", lw=1.2,
                   label=f"threshold {kappa:.3f} (99th pct demos)")
        bx = [s["t"] for s in nom["steps"] if s["boundary"]]
        ax.plot(bx, [nom["steps"][t][key] for t in bx], "o", ms=4, color="#333333",
                label="decision point (replan boundary)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    if pro["fired"]:
        f, hb = pro["fire"], pro["handback"]
        for ax in (ax_c, ax_e, ax_v):
            ax.axvspan(f["t"], hb["t"], color="#ffae42", alpha=0.25)
            ax.axvline(f["t"], color="#d95f02", lw=1.5)
            ax.axvline(hb["t"], color="#1f78b4", lw=1.5)
        ax_c.annotate(f"FIRE t={f['t']}\nc={f['c']:.2f}", (f["t"], f["c"]),
                      xytext=(f["t"] + 1, f["c"] + 0.12), fontsize=9, color="#d95f02")
        ax_c.annotate(
            f"HANDBACK t={hb['t']}\nc={hb['c']:.2f} (tp-counterfactual {f['c_if_teleported']:.2f})",
            (hb["t"], hb["c"]), xytext=(hb["t"] + 1, hb["c"] + 0.25),
            fontsize=9, color="#1f78b4")

    if sweep_row is not None:
        g = sweep_row["grid"]
        ts = [r["t"] for r in g]
        ax_v.step(ts, [r["v_nom"] for r in g], where="post", color="#777777",
                  lw=1.8, label="V_nom(t)  (K=4)")
        ax_v.step(ts, [max(r["v_tp"]) for r in g], where="post", color="#33a02c",
                  lw=1.8, label="best V_teleport(t)  [non-physical]")
        ax_v.step(ts, [max(r["v_ex"]) for r in g], where="post", color="#d95f02",
                  lw=1.8, label="best V_executed(t)")
        ax_v.set_ylim(-0.05, 1.05)
        ax_v.set_ylabel("oracle value at boundary")
    else:
        ax_v.text(0.5, 0.5, "no sweep row (nominal success in recorded run)",
                  ha="center", va="center", transform=ax_v.transAxes)
    ax_v.set_xlabel("control step")
    ax_v.grid(alpha=0.25)

    outcome = (f"nominal: {'SUCCESS' if nom['success'] else 'FAILURE'}   |   "
               f"prototype: {'SUCCESS' if pro['success'] else 'FAILURE'}"
               + (f"   |   fired at t={pro['fire']['t']}" if pro["fired"]
                  else "   |   never fired"))
    fig.suptitle(f"Recovery debug — env_seed {env_seed}\n{outcome}", fontsize=12)
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- main ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/active_min/overnight/flow/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "data/active_min/subset_0400.h5"))
    ap.add_argument("--oracle-json",
                    default=str(REPO / "outputs/active_min/recovery_oracle/flow/recovery_oracle.json"))
    ap.add_argument("--out-dir",
                    default=str(REPO / "outputs/active_min/recovery_oracle/flow/debug_viz"))
    ap.add_argument("--env-seeds", type=int, nargs="+",
                    default=[1229399060, 1120782160])
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--m-poses", type=int, default=3)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oracle = json.load(open(args.oracle_json))
    ref = {e["env_seed"]: e for e in oracle["reference"]}
    sweep = {r["env_seed"]: r for r in oracle["failures"]}
    proto_rec = {p["env_seed"]: p for p in oracle["prototype"]}

    reader = DatasetReader(args.dataset)
    states, ep_idx = [], []
    for j, eid in enumerate(reader.episode_ids):
        ep = reader.episode(eid)
        states.append(ep.state)
        ep_idx.append(np.full(len(ep.state), j))
    index = SupportIndex(np.concatenate(states), np.concatenate(ep_idx))
    kc, ke = index.calibrate(percentile=args.percentile)
    print(f"[viz] thresholds kappa_c={kc:.4f} kappa_e={ke:.4f}")

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode="rgb_array")
    system = build_system(STANDALONE, policy, [])
    rn = Runner(env, system)

    report = {}
    for env_seed in args.env_seeds:
        if env_seed not in ref:
            print(f"[viz] {env_seed}: not in oracle reference — skipping")
            continue
        pol_seed = ref[env_seed]["seed"]
        rec_p = proto_rec.get(env_seed, {})
        print(f"\n[viz] === env_seed {env_seed} (recorded: nominal "
              f"{'S' if ref[env_seed]['success'] else 'F'}, fired={rec_p.get('fired')}"
              f" at t={rec_p.get('fire_t')}) ===")

        nom, f_nom = rollout_nominal(rn, index, env, env_seed, pol_seed, kc, ke)
        print(f"[viz] nominal replay: {'SUCCESS' if nom['success'] else 'FAILURE'}"
              f" (recorded {'S' if ref[env_seed]['success'] else 'F'})")
        pro, f_pro = rollout_prototype(rn, index, env, env_seed, pol_seed, kc, ke,
                                       m_poses=args.m_poses)
        if pro["fired"]:
            f, hb = pro["fire"], pro["handback"]
            print(f"[viz] prototype: FIRED t={f['t']} (recorded {rec_p.get('fire_t')}) "
                  f"c={f['c']:.2f} e={f['e']:.3f}")
            print(f"[viz]   candidates qpos-dist {['%.2f' % x for x in f['cand_qpos_dists']]}"
                  f" -> picked #{f['picked']}")
            print(f"[viz]   RETURN used {hb['steps_used']} steps | tip_err "
                  f"{hb['tip_err']:.4f} | block bump {hb['block_bump']:.4f}")
            print(f"[viz]   c: fire {f['c']:.2f} -> handback {hb['c']:.2f} "
                  f"(teleport counterfactual {f['c_if_teleported']:.2f}) | "
                  f"qpos-dist to cand at handback {hb['qpos_dist_to_cand']:.3f}")
        else:
            print("[viz] prototype: never fired in this replay")
        print(f"[viz] prototype outcome: {'SUCCESS' if pro['success'] else 'FAILURE'}")

        save_video(f_nom, out / f"nominal_{env_seed}.mp4", fps=args.fps)
        save_video(f_pro, out / f"prototype_{env_seed}.mp4", fps=args.fps)
        n = max(len(f_nom), len(f_pro))
        f_nom += [f_nom[-1]] * (n - len(f_nom))
        f_pro += [f_pro[-1]] * (n - len(f_pro))
        save_video([np.concatenate([a, b], axis=1) for a, b in zip(f_nom, f_pro)],
                   out / f"combined_{env_seed}.mp4", fps=args.fps)
        timeline(env_seed, nom, pro, sweep.get(env_seed), kc, ke,
                 out / f"timeline_{env_seed}.png")
        trace = {"env_seed": env_seed, "policy_seed": pol_seed,
                 "kappa_c": kc, "kappa_e": ke,
                 "recorded": {"nominal_success": ref[env_seed]["success"],
                              "prototype": rec_p},
                 "nominal": nom, "prototype": pro}
        with open(out / f"trace_{env_seed}.json", "w") as fh:
            json.dump(trace, fh, indent=1)
        report[env_seed] = trace
        print(f"[viz] wrote nominal/prototype/combined mp4 + timeline png + trace json")

    env.close()
    print(f"\n[viz] all artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

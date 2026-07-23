#!/usr/bin/env python
"""PickCube recovery visualization: at the detector's decision boundary, show
BOTH recovery transports side by side against the do-nothing nominal —

  NOMINAL   : baseline policy, no intervention (the failure).
  TELEPORT  : instantaneously overwrite the robot's 9-dof qpos with a matched
              demo config (cube untouched). NON-PHYSICAL upper bound; the jump
              frame is flagged. Then hand back to the policy.
  RETURN    : the scripted PickCubeReturnController (lift->translate->descend->
              grip) drives to the same demo tip target, physically. Then hand
              back.

Fire rule = the deployable prototype's: first replan boundary with
c > kappa_c AND e <= kappa_e (support statistics). Candidate = nearest demo
config by full qpos among the m nearest-by-cube-pose demos. Overlays every
step: phase/agency, c & e vs thresholds, is_grasped, cube->goal, and (RETURN)
tip->target + return phase. Timeline plots all three c/e traces, the fire
boundary, the takeover span, the return's tip-error trace, and the teleport
instant.

    python scripts/visualize_recovery_pickcube.py --seeds 84886883,145124051
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
from actsemble.evaluation.video import save_video  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import (  # noqa: E402
    PC_GOAL_POS,
    PC_GRASPED,
    PC_OBJ_POS,
    PC_TCP_POS,
    PickCubeReturnController,
    PickCubeSupportIndex,
)
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402


def _font(sz):
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), sz)


HEADER = 128


def grab(env):
    f = env.render()
    if torch.is_tensor(f):
        f = f.detach().cpu().numpy()
    f = np.asarray(f)
    return f.reshape(f.shape[-3], f.shape[-2], f.shape[-1]).astype(np.uint8)


def cube_goal(s):
    return float(np.linalg.norm(s[PC_OBJ_POS] - s[PC_GOAL_POS]))


def annotate(frame, *, step, phase, c, e, kc, ke, s, banner=None, border=None, extra=""):
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    cv = Image.new("RGB", (w, h + HEADER), (12, 12, 12))
    cv.paste(Image.fromarray(frame), (0, HEADER))
    d = ImageDraw.Draw(cv)
    f14, f18 = _font(14), _font(18)
    pc = {"NOMINAL": (210, 210, 210), "TELEPORT": (120, 210, 255),
          "RETURN": (255, 165, 45), "POST": (150, 220, 150)}.get(phase.split()[0], (210, 210, 210))
    d.text((8, 6), f"step {step:>2}  {phase}", font=f18, fill=pc)
    cc = (235, 90, 90) if c > kc else (120, 220, 120)
    ec = (235, 90, 90) if e > ke else (120, 220, 120)
    d.text((8, 34), f"c={c:5.3f} (kc {kc:.3f}) robot-given-cube", font=f14, fill=cc)
    d.text((8, 54), f"e={e:5.3f} (ke {ke:.3f}) cube familiarity", font=f14, fill=ec)
    d.text((8, 74), f"grasped {int(s[PC_GRASPED][0]>0.5)}   cube->goal {cube_goal(s):.3f} (win<=0.025)",
           font=f14, fill=(210, 210, 210))
    d.text((8, 94), extra, font=f14, fill=(255, 205, 70))
    if banner:
        d.text((8, 112) if HEADER > 112 else (8, 96), banner, font=f18, fill=(255, 210, 60))
    out = np.asarray(cv)
    if border is not None:
        out = out.copy()
        out[:5], out[-5:], out[:, :5], out[:, -5:] = border, border, border, border
    return out


def tp_winner(row):
    """Oracle's TELEPORT-winning (boundary_t, candidate): the boundary+candidate
    where teleport recovery scores highest. Firing BOTH arms here isolates
    TRANSPORT — teleport recovers by construction, so if the scripted return
    fails at the same target it is the physical transport, not the target, that
    is at fault. Returns (t, cand, v_ex_of_that_cand, v_tp_of_that_cand)."""
    g = row["grid"]
    bi = int(np.argmax([max(r["v_tp"]) for r in g]))
    b = g[bi]
    bj = int(np.argmax(b["v_tp"]))
    cand = (np.array(b["cand_qpos"][bj]), np.array(b["cand_tip"][bj]), bool(b["cand_grasped"][bj]))
    return b["t"], cand, b["v_ex"][bj], b["v_tp"][bj]


def arm_nominal(rn, env, seed, pol_seed, index, kc, ke):
    rn.start(pol_seed)
    raw, _ = env.reset(seed=seed)
    frames, recs, once = [], [], False
    for step in range(MAX_STEPS):
        s = rn.state42(raw)
        c, e = index.stats(s)
        recs.append((step, c, e, "NOMINAL"))
        frames.append(annotate(grab(env), step=step, phase="NOMINAL", c=c, e=e, kc=kc, ke=ke, s=s))
        raw, _r, _te, _tr, info = rn.act_env(raw)
        once = once or _success(info)
    return frames, recs, once


def arm_teleport(rn, env, seed, pol_seed, index, kc, ke, fire_t, cand):
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
    # the instantaneous, non-physical jump
    s_before = rn.state42(raw)
    c, e = index.stats(s_before)
    frames.append(annotate(grab(env), step=fire_t, phase="TELEPORT (non-physical)", c=c, e=e,
                           kc=kc, ke=ke, s=s_before, banner="INSTANT JUMP to demo qpos ->",
                           border=(90, 180, 255)))
    teleport_robot(rn.u, cand[0])
    raw = rn.u.get_obs()
    rn.start(pol_seed + 5555)  # fresh policy history after the jump
    s_after = rn.state42(raw)
    c, e = index.stats(s_after)
    frames.append(annotate(grab(env), step=fire_t, phase="TELEPORT done", c=c, e=e, kc=kc, ke=ke,
                           s=s_after, banner="<- landed (cube untouched)", border=(90, 180, 255),
                           extra=f"c {index.stats(s_before)[0]:.2f} -> {c:.2f}"))
    for step in range(fire_t, MAX_STEPS):
        s = rn.state42(raw)
        c, e = index.stats(s)
        recs.append((step, c, e, "POST"))
        frames.append(annotate(grab(env), step=step, phase="POST-teleport (policy)", c=c, e=e,
                               kc=kc, ke=ke, s=s))
        raw, _r, _te, _tr, info = rn.act_env(raw)
        once = once or _success(info)
    return frames, recs, once


def arm_return(rn, env, seed, pol_seed, index, kc, ke, fire_t, cand):
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
    ctl = PickCubeReturnController(cand[1], cand[2], max_steps=20)
    cand_arm = np.asarray(cand[0])[0:7]
    tip_trace, cfg_trace = [], []
    step = fire_t
    while step < MAX_STEPS and not ctl.done(rn.state42(raw)[PC_TCP_POS]):
        s = rn.state42(raw)
        tip = float(np.linalg.norm(s[PC_TCP_POS] - cand[1]))
        cfg = float(np.linalg.norm(s[0:7] - cand_arm))  # ARM joint-config error
        tip_trace.append((step, tip))
        cfg_trace.append((step, cfg))
        c, e = index.stats(s)
        recs.append((step, c, e, "RETURN"))
        frames.append(annotate(grab(env), step=step, phase="RETURN (agency: controller)", c=c, e=e,
                               kc=kc, ke=ke, s=s, border=(255, 150, 30),
                               extra=f"tip->tgt {tip:.3f}  CONFIG(arm7)->tgt {cfg:.3f}  grip={int(cand[2])}"))
        raw, _r, _te, _tr, info = rn.step_raw(ctl.action(rn.state42(raw)[PC_TCP_POS]))
        once = once or _success(info)
        step += 1
    s_hb = rn.state42(raw)
    tip_hb = float(np.linalg.norm(s_hb[PC_TCP_POS] - cand[1]))
    cfg_hb = float(np.linalg.norm(s_hb[0:7] - cand_arm))
    tip_ok = "tip CONVERGED" if tip_hb < 0.015 else f"tip PLATEAU {tip_hb:.3f}"
    cfg_ok = "config REACHED" if cfg_hb < 0.15 else f"config OFF {cfg_hb:.3f}"
    hb_banner = f"handback: {tip_ok} / {cfg_ok}"
    rn.start(pol_seed + 5555)
    for step in range(step, MAX_STEPS):
        s = rn.state42(raw)
        c, e = index.stats(s)
        recs.append((step, c, e, "POST"))
        frames.append(annotate(grab(env), step=step, phase="POST-return (policy)", c=c, e=e, kc=kc,
                               ke=ke, s=s, extra=hb_banner if step == recs[-1][0] else ""))
        raw, _r, _te, _tr, info = rn.act_env(raw)
        once = once or _success(info)
    return frames, recs, once, tip_trace, tip_hb, cfg_trace, cfg_hb


def timeline(seed, arms, fire_t, kc, ke, tip_trace, tip_hb, cfg_trace, cfg_hb, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axc, axe, axt) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2, 3]})
    col = {"nominal": "#7a7a7a", "teleport": "#1f9fd8", "return": "#d9721f"}
    for name, recs, _out in arms:
        t = [r[0] for r in recs]
        axc.plot(t, [r[1] for r in recs], color=col[name], lw=1.8, label=name)
        axe.plot(t, [r[2] for r in recs], color=col[name], lw=1.8, label=name)
    axc.axhline(kc, color="#c02020", ls="--", lw=1.1, label=f"kappa_c {kc:.3f}")
    axe.axhline(ke, color="#c02020", ls="--", lw=1.1, label=f"kappa_e {ke:.3f}")
    span_end = fire_t + (tip_trace[-1][0] - fire_t if tip_trace else 1)
    for ax in (axc, axe, axt):
        ax.axvline(fire_t, color="#111", lw=1.4)
        ax.axvspan(fire_t, span_end, color="#ffce9e", alpha=0.25)
    axc.annotate(f"FIRE t={fire_t}", (fire_t, axc.get_ylim()[1] * 0.9), fontsize=9, color="#111")
    # bottom panel: BOTH the tip error and the ARM CONFIG error during the return
    if tip_trace:
        axt.plot([x[0] for x in tip_trace], [x[1] for x in tip_trace], "-o", ms=3,
                 color="#2a7fb8", label=f"tip->target (handback {tip_hb:.3f})")
        axt.axhline(0.015, color="#2a7fb8", ls=":", lw=1, label="tip tol 0.015")
    if cfg_trace:
        axt.plot([x[0] for x in cfg_trace], [x[1] for x in cfg_trace], "-s", ms=3,
                 color="#c0392b", label=f"ARM CONFIG->target (handback {cfg_hb:.3f})")
        axt.axhline(0.15, color="#c0392b", ls=":", lw=1, label="config reached < 0.15")
    axc.set_ylabel("c (robot-given-cube)")
    axe.set_ylabel("e (cube familiarity)")
    axt.set_ylabel("return error (tip m / config rad)")
    axt.set_xlabel("control step")
    outs = "   ".join(f"{n}:{'SUCCESS' if o else 'FAIL'}" for n, _r, o in arms)
    fig.suptitle(f"PickCube recovery — seed {seed}   [{outs}]\n"
                 f"return handback: tip {tip_hb:.3f} / arm-config {cfg_hb:.3f} rad", fontsize=11)
    for ax in (axc, axe, axt):
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def filmstrip(frames_by_arm, seed, path, ncol=6):
    from PIL import Image

    import numpy as np

    rows = []
    for name, frames in frames_by_arm:
        core = frames
        idx = np.linspace(0, len(core) - 1, ncol).astype(int)
        strip = np.concatenate([core[i] for i in idx], axis=1)
        rows.append(strip)
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, W - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def summary_scatter(summary, kc, path):
    """The diagnostic punchline: return outcome vs the arm-config error the
    scripted controller left at handback. If success tracks config-error (not
    tip-error), the fix is joint-space return."""
    if not summary:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for r in summary:
        x, y = r["cfg_hb"], r["tip_hb"]
        ok = r["return"]
        ax.scatter(x, y, s=130, marker="o" if ok else "X",
                   c="#2a9d3a" if ok else "#c0392b", edgecolors="#222", linewidths=1, zorder=3)
        ax.annotate(str(r["seed"])[:6], (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.axvline(0.15, color="#c0392b", ls="--", lw=1.2, label="config-reached threshold 0.15")
    ax.axhline(0.015, color="#2a7fb8", ls="--", lw=1.2, label="tip-converged threshold 0.015")
    ax.set_xlabel("ARM CONFIG error at handback  (rad, arm7 dist to demo target)")
    ax.set_ylabel("tip error at handback (m)")
    ax.set_title("PickCube scripted return: outcome vs handback errors\n"
                 "green O = return recovered, red X = failed  (many at tip~0 but config large = null-space)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"))
    ap.add_argument("--dataset", default=str(REPO / "data/pickcube/subset_0100.h5"))
    ap.add_argument("--oracle-json", default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_oracle.json"))
    ap.add_argument("--seeds", default="84886883,145124051")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/pickcube/recovery_oracle/debug_viz"))
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--m-poses", type=int, default=3)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oracle = json.load(open(args.oracle_json))
    seed_of = {e["env_seed"]: e["seed"] for e in oracle["reference"]}
    ex_of = {r["env_seed"]: (r["tp_final"], r["ex_final"], r["mode"]) for r in oracle["failures"]}

    fail_of = {r["env_seed"]: r for r in oracle["failures"]}
    reader = DatasetReader(args.dataset)
    states, ep_idx = [], []
    for j, eid in enumerate(reader.episode_ids):
        ep = reader.episode(eid)
        states.append(ep.state)
        ep_idx.append(np.full(len(ep.state), j))
    index = PickCubeSupportIndex(np.concatenate(states), np.concatenate(ep_idx))
    kc, ke = index.calibrate(percentile=args.percentile)
    print(f"[viz-pc] kappa_c={kc:.4f} kappa_e={ke:.4f}", flush=True)

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode="rgb_array", max_episode_steps=MAX_STEPS)
    system = build_system(STANDALONE, policy, [])
    rn = Runner(env, system)
    H_a = system.action_horizon
    summary = []

    for seed in [int(s) for s in args.seeds.split(",")]:
        pol_seed = seed_of.get(seed, 0)
        tp, ex, mode = ex_of.get(seed, (None, None, "?"))
        # oracle's ex-winning boundary + candidate: same target the ceiling was
        # measured on, so teleport-vs-return isolates transport (§ diagnosis).
        fire_t, cand, vex, vtp = tp_winner(fail_of[seed])
        print(f"\n[viz-pc] seed {seed} mode={mode} oracle tp={tp} ex={ex} "
              f"| tp-winning boundary t={fire_t} (this cand v_ex={vex:.1f} v_tp={vtp:.1f}) "
              f"cand grasped={cand[2]} tip={np.round(cand[1],3)}", flush=True)

        fn, rn_nom, on = arm_nominal(rn, env, seed, pol_seed, index, kc, ke)
        ft, rn_tp, ot = arm_teleport(rn, env, seed, pol_seed, index, kc, ke, fire_t, cand)
        fr, rn_re, ore, tip_trace, tip_hb, cfg_trace, cfg_hb = arm_return(
            rn, env, seed, pol_seed, index, kc, ke, fire_t, cand)
        print(f"[viz-pc]   outcomes  nominal={'S' if on else 'F'}  teleport={'S' if ot else 'F'}  "
              f"return={'S' if ore else 'F'} (tip {tip_hb:.3f} / arm-config {cfg_hb:.3f})", flush=True)
        summary.append({"seed": seed, "mode": mode, "oracle_tp": tp, "oracle_ex": ex,
                        "fire_t": fire_t, "tip_hb": tip_hb, "cfg_hb": cfg_hb,
                        "teleport": bool(ot), "return": bool(ore), "nominal": bool(on)})

        save_video(fn, out / f"nominal_{seed}.mp4", fps=args.fps)
        save_video(ft, out / f"teleport_{seed}.mp4", fps=args.fps)
        save_video(fr, out / f"return_{seed}.mp4", fps=args.fps)
        n = max(len(fn), len(ft), len(fr))
        pad = lambda L: L + [L[-1]] * (n - len(L))
        save_video([np.concatenate([a, b, c], axis=1) for a, b, c in zip(pad(fn), pad(ft), pad(fr))],
                   out / f"combined_{seed}.mp4", fps=args.fps)
        timeline(seed, [("nominal", rn_nom, on), ("teleport", rn_tp, ot), ("return", rn_re, ore)],
                 fire_t, kc, ke, tip_trace, tip_hb, cfg_trace, cfg_hb, out / f"timeline_{seed}.png")
        filmstrip([("NOMINAL", fn), ("TELEPORT", ft), ("RETURN", fr)], seed, out / f"filmstrip_{seed}.png")
        print(f"[viz-pc]   wrote nominal/teleport/return/combined mp4 + timeline + filmstrip", flush=True)
    env.close()
    summary_scatter(summary, kc, out / "SUMMARY_config_vs_outcome.png")
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"\n[viz-pc] all artifacts in {out}  (+ SUMMARY_config_vs_outcome.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

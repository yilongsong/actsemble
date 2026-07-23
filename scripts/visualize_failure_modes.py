#!/usr/bin/env python
"""Film the n=25 policy's FAILURE MODES, with the state variables overlaid.

Motivation. The recovery oracle's reference pass on the teleop-trained n=25
policy reported a mode histogram dominated by `lifted_then_dropped` (58% of
failures), where the RL-bundle-era prediction had been `grasp_slip_at_pick`.
Before reading anything into that, note what the oracle's own rule actually
says (recovery_oracle_pickcube.classify_failure):

    if max_lift >= 0.02 and final_gdist > 0.05:  return "lifted_then_dropped"

That fires on ANY episode that raised the cube 2 cm and finished more than
5 cm from the goal. Two opposite behaviours share the label:

  DROPPED  the cube was lifted and is back near the table at the end
           (final cube z close to its starting z, not grasped)
  CARRYING the cube is still in the gripper, off the table, just not at the
           goal when the horizon expired -- a STALL, not a drop

This script separates them by measurement and films examples of each, so the
mode histogram is read from behaviour rather than from a label's name.

Outputs, per selected episode:
  mode_<mode>_<sub>_seed<seed>.mp4   annotated rollout
  mode_<mode>_<sub>_seed<seed>.png   filmstrip
  timeline_<mode>_<sub>_seed<seed>.png   cube z / goal z / grasp / distances
plus summary.json and modes_summary.png over all episodes rolled.

    python scripts/visualize_failure_modes.py --n 60 --max-steps 160
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
    STANDALONE,
    Runner,
    _success,
    classify_failure,
)

from actsemble.evaluation.video import save_video  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import PC_GOAL_POS, PC_GRASPED, PC_OBJ_POS, PC_TCP_POS  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402

HEADER = 116


def _font(sz):
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), sz)


def grab(env):
    f = env.render()
    if torch.is_tensor(f):
        f = f.detach().cpu().numpy()
    f = np.asarray(f)
    return f.reshape(f.shape[-3], f.shape[-2], f.shape[-1]).astype(np.uint8)


def annotate(frame, *, step, s, z0, banner=""):
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    cv = Image.new("RGB", (w, h + HEADER), (12, 12, 12))
    cv.paste(Image.fromarray(frame), (0, HEADER))
    d = ImageDraw.Draw(cv)
    f13, f17 = _font(13), _font(17)
    obj, goal, tcp = s[PC_OBJ_POS], s[PC_GOAL_POS], s[PC_TCP_POS]
    grasped = int(s[PC_GRASPED][0] > 0.5)
    gd = float(np.linalg.norm(obj - goal))
    lift = float(obj[2] - z0)
    d.text((8, 5), f"step {step:>3}", font=f17, fill=(225, 225, 225))
    d.text((8, 28), f"cube z {obj[2]:.3f}  (lift {lift:+.3f})   goal z {goal[2]:.3f}",
           font=f13, fill=(150, 210, 255))
    d.text((8, 46), f"cube->goal {gd:.3f}  (win <= 0.025)", font=f13,
           fill=(120, 220, 120) if gd <= 0.025 else (235, 150, 90))
    d.text((8, 64), f"grasped {grasped}   tcp->cube {float(np.linalg.norm(tcp - obj)):.3f}",
           font=f13, fill=(120, 220, 120) if grasped else (235, 90, 90))
    d.text((8, 84), banner, font=f13, fill=(255, 205, 70))
    return np.asarray(cv)


def sub_label(rec) -> str:
    """Split `lifted_then_dropped` into the two behaviours it conflates."""
    s = rec["states"]
    obj = np.stack([x[PC_OBJ_POS] for x in s])
    grasped_end = bool(s[-1][PC_GRASPED][0] > 0.5)
    z0, z_end, z_max = float(obj[0, 2]), float(obj[-1, 2]), float(obj[:, 2].max())
    back_down = (z_end - z0) < 0.02
    if grasped_end and not back_down:
        return "carrying"      # still holding it, off the table: a STALL
    if back_down:
        return "dropped"       # cube is back near the table
    return "held_low"          # off table but released


def timeline(rec, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = rec["states"]
    t = np.arange(len(s))
    obj = np.stack([x[PC_OBJ_POS] for x in s])
    tcp = np.stack([x[PC_TCP_POS] for x in s])
    goal = np.stack([x[PC_GOAL_POS] for x in s])
    grasped = np.array([float(x[PC_GRASPED][0] > 0.5) for x in s])
    gd = np.linalg.norm(obj - goal, axis=1)
    td = np.linalg.norm(tcp - obj, axis=1)

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2, 3]})
    a1.plot(t, obj[:, 2], color="#1f9fd8", lw=2, label="cube z")
    a1.axhline(goal[0, 2], color="#2a9d3a", ls="--", lw=1.4, label=f"goal z {goal[0,2]:.3f}")
    a1.axhline(obj[0, 2], color="#888", ls=":", lw=1.2, label=f"table z {obj[0,2]:.3f}")
    a1.set_ylabel("height (m)")
    a2.fill_between(t, 0, grasped, step="pre", color="#2a9d3a", alpha=0.45)
    a2.set_ylabel("grasped")
    a2.set_ylim(-0.05, 1.05)
    a3.plot(t, gd, color="#c0392b", lw=2, label="cube -> goal")
    a3.plot(t, td, color="#7a5cc0", lw=1.6, label="tcp -> cube")
    a3.axhline(0.025, color="#2a9d3a", ls="--", lw=1.3, label="success 0.025")
    a3.axhline(0.05, color="#c0392b", ls=":", lw=1.2, label="mode rule 0.05")
    a3.set_ylabel("distance (m)")
    a3.set_xlabel("control step")
    fig.suptitle(f"{rec['mode']} / {rec['sub']} — seed {rec['seed']}   "
                 f"(horizon {len(s)} steps, success={rec['success']})", fontsize=12)
    for ax in (a1, a2, a3):
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def filmstrip(frames, path, ncol=8):
    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, ncol).astype(int)
    Image.fromarray(np.concatenate([frames[i] for i in idx], axis=1)).save(path)


def modes_summary(records, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fails = [r for r in records if not r["success"]]
    keys = {}
    for r in fails:
        k = r["mode"] if r["mode"] != "lifted_then_dropped" else f"lifted:{r['sub']}"
        keys[k] = keys.get(k, 0) + 1
    if not keys:
        return
    ks = sorted(keys, key=lambda k: -keys[k])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    cols = ["#c0392b" if k.startswith("lifted:dropped") else
            "#e08a1e" if k.startswith("lifted:carrying") else "#7a7a7a" for k in ks]
    ax.barh(ks[::-1], [keys[k] for k in ks][::-1], color=cols[::-1])
    ax.set_xlabel(f"failures (of {len(records)} episodes, {len(fails)} failed)")
    ax.set_title("PickCube n=25 failure modes — `lifted_then_dropped` split by behaviour")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/pickcube/teleop_v1/sweep_n25/diffusion/selected_policy.pt"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs/pickcube/teleop_v1/failure_modes_n25"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--per-mode", type=int, default=2, help="episodes to film per (mode,sub)")
    ap.add_argument("--dump-all", action="store_true",
                    help="save EVERY episode as {index}_{success|failed}.mp4 for manual review")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(Path(args.policy_checkpoint), device=args.device)
    system = build_system(STANDALONE, policy, [])
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode="rgb_array", max_episode_steps=args.max_steps)
    rn = Runner(env, system)
    print(f"[viz] horizon {args.max_steps} steps, {args.n} episodes", flush=True)

    records = []
    for i in range(args.n):
        seed = 900000 + i
        rn.start(seed)
        raw, _ = env.reset(seed=seed)
        states, frames, once = [], [], False
        z0 = float(rn.state42(raw)[PC_OBJ_POS][2])
        for step in range(args.max_steps):
            s = rn.state42(raw)
            states.append(s.copy())
            frames.append(annotate(grab(env), step=step, s=s, z0=z0))
            raw, _r, _te, _tr, info = rn.act_env(raw)
            once = once or _success(info)
        mode = "success" if once else classify_failure(states)
        rec = {"seed": seed, "success": bool(once), "mode": mode, "states": states}
        rec["sub"] = sub_label(rec) if mode == "lifted_then_dropped" else "-"
        rec["frames"] = frames
        if args.dump_all:
            # Write EVERY episode as {index}_{success|failed}.mp4 for manual review,
            # then drop the frames: 160 frames x 512x628x3 is ~154 MB per episode, so
            # retaining them all would need ~18 GB at n=120.
            save_video(frames, out / f"{i}_{'success' if once else 'failed'}.mp4", fps=args.fps)
            rec["frames"] = None
        records.append(rec)
        print(f"[viz] {i+1}/{args.n} seed={seed} {'SUCCESS' if once else mode}"
              f"{'/' + rec['sub'] if rec['sub'] != '-' else ''}", flush=True)

    # ---- film up to --per-mode examples of each (mode, sub) --------------
    seen: dict[tuple, int] = {}
    for r in records:
        if r["success"] or r["frames"] is None:
            continue
        key = (r["mode"], r["sub"])
        if seen.get(key, 0) >= args.per_mode:
            continue
        seen[key] = seen.get(key, 0) + 1
        tag = f"{r['mode']}_{r['sub']}_seed{r['seed']}" if r["sub"] != "-" else f"{r['mode']}_seed{r['seed']}"
        save_video(r["frames"], out / f"mode_{tag}.mp4", fps=args.fps)
        filmstrip(r["frames"], out / f"mode_{tag}.png")
        timeline(r, out / f"timeline_{tag}.png")
        print(f"[viz] filmed {tag}", flush=True)

    modes_summary(records, out / "modes_summary.png")
    summary = {
        "n": args.n, "max_steps": args.max_steps,
        "checkpoint": str(args.policy_checkpoint),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "episodes": [{k: v for k, v in r.items() if k not in ("states", "frames")}
                     for r in records],
    }
    hist: dict[str, int] = {}
    for r in records:
        if r["success"]:
            continue
        k = r["mode"] if r["sub"] == "-" else f"{r['mode']}:{r['sub']}"
        hist[k] = hist.get(k, 0) + 1
    summary["mode_histogram"] = hist
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[viz] success {summary['success_rate']*100:.1f}%  modes {hist}", flush=True)
    print(f"[viz] wrote -> {out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

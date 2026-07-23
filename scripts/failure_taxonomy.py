#!/usr/bin/env python
"""Behavioral failure-mode census with video exemplars (E/F track).

Pass 1 (stats): roll the standalone policy over the diagnostic panel with
per-step instrumentation, extract per-episode features, classify failures
into task-specific BEHAVIORAL modes (rule-based, thresholds printed and
stored so reclassification needs no re-roll), report the histogram.
Pass 2 (video): re-run 2-3 exemplars per mode with rendering + live overlay
(mode, step, task metrics). physx_cuda replays are nondeterministic, so an
exemplar whose class flips on replay is swapped for the next candidate.

    python scripts/failure_taxonomy.py --task pusht \
        --policy-checkpoint outputs/active_min/overnight/flow/selected_policy.pt
    python scripts/failure_taxonomy.py --task pickcube --max-steps 50 \
        --policy-checkpoint outputs/pickcube/ndemos_v1/ndemos_50/seed_0/policy/selected_policy.pt
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

from actsemble.evaluation.panels import make_panel, panel_episodes  # noqa: E402
from actsemble.evaluation.video import save_video  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import quat_yaw, wrap_angle  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.sim.rollout import ObservationAdapter  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402

STANDALONE = {"policy": {"num_candidates": 1}, "components": [],
              "selection": {"type": "candidate_zero"}, "execution": {}}
GOAL_YAW_PUSHT = (5.0 / 3.0) * np.pi


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ---------------- task adapters ----------------------------------------------
class PushTAdapter:
    name = "pusht"
    goal_xy = np.array([-0.156, -0.1])

    def __init__(self, env):
        self.env = env

    def step_record(self, s):
        return {
            "block_xy": s[24:26].copy(),
            "yaw": float(quat_yaw(s[27:31])),
            "tcp": s[14:16].copy(),
            "cov": float(self.env.unwrapped.pseudo_render_intersection().item()),
        }

    def features(self, rec):
        cov = np.array([r["cov"] for r in rec])
        bxy = np.stack([r["block_xy"] for r in rec])
        tcp = np.stack([r["tcp"] for r in rec])
        pos_err = float(np.linalg.norm(bxy[-1] - self.goal_xy))
        rot_err = float(abs(wrap_angle(rec[-1]["yaw"] - GOAL_YAW_PUSHT)))
        last = slice(max(len(rec) - 30, 0), None)
        return {
            "max_cov": float(cov.max()), "final_cov": float(cov[-1]),
            "t_max_cov": int(cov.argmax()),
            "block_disp_max": float(np.linalg.norm(bxy - bxy[0], axis=1).max()),
            "pos_err_final": pos_err, "rot_err_final": rot_err,
            "tcp_path_last30": float(np.linalg.norm(np.diff(tcp[last], axis=0), axis=1).sum()),
            "cov_gain_last30": float(cov[-1] - cov[last][0]),
        }

    def classify(self, f):
        if f["block_disp_max"] < 0.02:
            return "never_engaged"
        if f["max_cov"] >= 0.85 and f["final_cov"] < f["max_cov"] - 0.30:
            return "achieved_then_lost"
        if f["max_cov"] >= 0.70:
            if f["pos_err_final"] <= 0.05 and f["rot_err_final"] > 0.26:
                return "near_miss_rotation"
            return "near_miss_position"
        if f["tcp_path_last30"] < 0.05:
            return "stall_midtask"
        if f["tcp_path_last30"] > 0.60 and f["cov_gain_last30"] < 0.02:
            return "thrash_no_progress"
        return "coarse_mispositioned"

    def overlay_lines(self, r, f):
        return [
            f"coverage {r['cov']:.2f}  (max {f['max_cov']:.2f}, success needs 0.90)",
            f"pos err {np.linalg.norm(r['block_xy'] - self.goal_xy):.3f}  "
            f"rot err {abs(wrap_angle(r['yaw'] - GOAL_YAW_PUSHT)):.2f} rad",
        ]


class PickCubeAdapter:
    name = "pickcube"

    def __init__(self, env):
        self.env = env
        from actsemble.sim.demonstration_source import probe_state_layout
        lay_env = make_env(task_id="PickCube-v1",
                           control_mode=env.unwrapped.control_mode,
                           sim_backend="physx_cuda", obs_mode="state_dict",
                           render_mode=None)
        layout = probe_state_layout(lay_env)
        lay_env.close()
        self.sl = {}
        for k, v in layout.items():
            kk = k.lower()
            sl = slice(int(v[0]), int(v[1]))
            if "is_grasped" in kk:
                self.sl["grasp"] = sl
            elif "tcp_pose" in kk:
                self.sl["tcp"] = sl
            elif "goal_pos" in kk and "to" not in kk:
                self.sl["goal"] = sl
            elif ("obj_pose" in kk or "cube" in kk) and "to" not in kk:
                self.sl["obj"] = sl
            elif "qvel" in kk:
                self.sl["qvel"] = sl
        missing = {"grasp", "tcp", "goal", "obj"} - set(self.sl)
        if missing:
            raise RuntimeError(f"layout keys missing {missing}; probed: {layout}")

    def step_record(self, s):
        return {
            "grasped": bool(s[self.sl["grasp"]][0] > 0.5),
            "cube": s[self.sl["obj"]][:3].copy(),
            "goal": s[self.sl["goal"]][:3].copy(),
            "tcp": s[self.sl["tcp"]][:3].copy(),
            "qvel_n": float(np.linalg.norm(s[self.sl["qvel"]])) if "qvel" in self.sl else 0.0,
        }

    def features(self, rec):
        cube = np.stack([r["cube"] for r in rec])
        tcp = np.stack([r["tcp"] for r in rec])
        goal = rec[-1]["goal"]
        grasped = np.array([r["grasped"] for r in rec])
        gdist = np.linalg.norm(cube - goal, axis=1)
        lift = cube[:, 2] - cube[0, 2]
        drop_t = None
        if grasped.any():
            g0 = int(np.argmax(grasped))
            rel = np.where(grasped[g0:][:-1] & ~grasped[g0 + 1:])[0]
            if len(rel):
                drop_t = int(g0 + rel[0] + 1)
        return {
            "ever_grasped": bool(grasped.any()),
            "grasp_frac": float(grasped.mean()),
            "min_tcp_cube": float(np.linalg.norm(tcp - cube, axis=1).min()),
            "max_lift": float(lift.max()),
            "cube_xy_disp": float(np.linalg.norm(cube[:, :2] - cube[0, :2], axis=1).max()),
            "final_gdist": float(gdist[-1]), "min_gdist": float(gdist.min()),
            "gdist_at_drop": float(gdist[drop_t]) if drop_t is not None else None,
            "dropped": drop_t is not None,
            "final_qvel": rec[-1]["qvel_n"],
        }

    def classify(self, f):
        if not f["ever_grasped"]:
            if f["min_tcp_cube"] > 0.05:
                return "never_reached"
            if f["cube_xy_disp"] > 0.05:
                return "knocked_away"
            return "grasp_failure"
        if f["dropped"] and (f["gdist_at_drop"] or 1) > 0.05 and f["final_gdist"] > 0.05:
            return "dropped_in_transit"
        if f["final_gdist"] <= 0.025:
            return "at_goal_no_settle"
        if f["final_gdist"] <= 0.10:
            return "misplaced_near_goal"
        return "carried_astray"

    def overlay_lines(self, r, f):
        return [
            f"grasped {int(r['grasped'])}   cube->goal {np.linalg.norm(r['cube'] - r['goal']):.3f}"
            f"  (success <= 0.025 + static)",
            f"tcp->cube {np.linalg.norm(r['tcp'] - r['cube']):.3f}   lift z {r['cube'][2]:.3f}",
        ]


ADAPTERS = {"pusht": PushTAdapter, "pickcube": PickCubeAdapter}


# ---------------- rollout machinery ------------------------------------------
def roll(env, system, adapter, pe, max_steps, render=False, font=None):
    oa = ObservationAdapter(action_dim=env.unwrapped.single_action_space.shape[-1])
    low = np.asarray(env.unwrapped.single_action_space.low, np.float32)
    high = np.asarray(env.unwrapped.single_action_space.high, np.float32)
    system.candidate_root_seed = pe.policy_sampling_seed
    system.reset(episode_seed=pe.env_seed)
    raw, _ = env.reset(seed=pe.env_seed)
    rec, frames, succ = [], [], False
    for step in range(max_steps):
        s = _np(raw).reshape(-1)
        rec.append(adapter.step_record(s))
        if render:
            frames.append((step, _grab(env)))
        act = system.act(oa.observe(raw))
        a = np.clip(np.asarray(act.value, np.float32).reshape(-1), low, high)
        raw, _r, term, trunc, info = env.step(torch.as_tensor(a.reshape(1, -1)))
        sf = info.get("success")
        if sf is not None and bool(_np(sf).reshape(-1)[0]):
            succ = True
        if bool(_np(term).reshape(-1)[0]) or bool(_np(trunc).reshape(-1)[0]):
            break
    return rec, succ, frames


def _grab(env):
    f = _np(env.render())
    return f.reshape(f.shape[-3], f.shape[-2], f.shape[-1]).astype(np.uint8)


def _annotate(frames, rec, feats, adapter, mode, seed):
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), 16)
    out = []
    for step, f in frames:
        h, w = f.shape[:2]
        canvas = Image.new("RGB", (w, h + 92), (12, 12, 12))
        canvas.paste(Image.fromarray(f), (0, 92))
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), f"FAILURE MODE: {mode}   seed {seed}   step {step}",
               font=font, fill=(255, 120, 80))
        for i, line in enumerate(adapter.overlay_lines(rec[min(step, len(rec) - 1)], feats)):
            d.text((10, 30 + 22 * i), line, font=font, fill=(210, 210, 210))
        arr = np.asarray(canvas).copy()
        arr[:4], arr[-4:], arr[:, :4], arr[:, -4:] = (200, 60, 60), (200, 60, 60), (200, 60, 60), (200, 60, 60)
        out.append(arr)
    out.extend([out[-1]] * 10)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(ADAPTERS), required=True)
    ap.add_argument("--policy-checkpoint", required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--exemplars", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir or (REPO / f"outputs/failure_taxonomy/{args.task}"))
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.policy_checkpoint, device=args.device, use_ema=True)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode="rgb_array")
    adapter = ADAPTERS[args.task](env)
    system = build_system(STANDALONE, policy, [])
    eps = list(panel_episodes(make_panel("diagnostic")))[: args.count]

    # ---- pass 1: stats --------------------------------------------------------
    rows = []
    for i, pe in enumerate(eps):
        rec, succ, _ = roll(env, system, adapter, pe, args.max_steps)
        f = adapter.features(rec)
        mode = "success" if succ else adapter.classify(f)
        rows.append({"env_seed": pe.env_seed, "success": succ, "mode": mode, **f})
        if (i + 1) % 50 == 0:
            print(f"[taxonomy:{args.task}] {i + 1}/{len(eps)}", flush=True)
    modes = {}
    for r in rows:
        modes.setdefault(r["mode"], []).append(r["env_seed"])
    n_fail = sum(1 for r in rows if not r["success"])
    print(f"\n=== {args.task} failure taxonomy — {policy.meta.task_id}, "
          f"n={len(rows)}, success {1 - n_fail / len(rows):.1%} ===")
    for m, seeds in sorted(modes.items(), key=lambda kv: -len(kv[1])):
        if m == "success":
            continue
        print(f"  {m:<22} {len(seeds):>4}  ({len(seeds) / max(n_fail, 1):.0%} of failures)")

    # ---- pass 2: exemplar videos ---------------------------------------------
    made = {}
    for m, seeds in modes.items():
        if m == "success":
            continue
        made[m] = []
        for seed in seeds:
            if len(made[m]) >= args.exemplars:
                break
            pe = next(e for e in eps if e.env_seed == seed)
            rec, succ, frames = roll(env, system, adapter, pe, args.max_steps, render=True)
            f = adapter.features(rec)
            mode2 = "success" if succ else adapter.classify(f)
            if mode2 != m:   # nondeterministic replay changed the class; skip
                continue
            vid = _annotate(frames, rec, f, adapter, m, seed)
            save_video(vid, out / f"{m}_seed{seed}.mp4", fps=10)
            made[m].append(seed)
        print(f"[taxonomy:{args.task}] {m}: videos for seeds {made[m]}", flush=True)
    env.close()

    with open(out / "taxonomy.json", "w") as fh:
        json.dump({"task": args.task, "policy": args.policy_checkpoint,
                   "checkpoint_hash": policy.checkpoint_hash,
                   "n": len(rows), "episodes": rows,
                   "mode_counts": {m: len(s) for m, s in modes.items()},
                   "exemplar_videos": made}, fh, indent=1)
    print(f"[taxonomy:{args.task}] wrote {out}/taxonomy.json + videos")
    return 0


if __name__ == "__main__":
    sys.exit(main())

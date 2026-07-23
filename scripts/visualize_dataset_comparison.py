#!/usr/bin/env python
"""Play recorded demonstrations back as video: bang-bang RL vs teleop-like.

Renders episodes by WRITING THE RECORDED STATE into the simulator frame by
frame (robot qpos/qvel, cube pose, goal marker) rather than re-executing the
actions. Replay is therefore exact -- no physics divergence, no need for the
original env seed -- so what you watch is the data itself, not an approximation
of it.

Each frame is captioned with the quantities that distinguish the two
demonstrators: hand height, cube height, per-step hand speed, and which phase
the frame belongs to (APPROACH before the cube leaves the table, CARRY after,
FROZEN once the arm stops moving).

    python scripts/visualize_dataset_comparison.py --n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.video import save_video  # noqa: E402
from actsemble.recovery import PC_OBJ_POS, PC_TCP_POS, pickcube_approach_len  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402

QPOS = slice(0, 9); QVEL = slice(9, 18)
GOAL = slice(26, 29); OBJ_POSE = slice(29, 36)
HEADER = 92


def _font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def write_state(u, s):
    """Put the simulator exactly into the recorded state s (42-dim)."""
    d = u.get_state_dict()
    art = d["articulations"]["panda"]
    art[..., -18:-9] = torch.as_tensor(s[QPOS], dtype=art.dtype, device=art.device)
    art[..., -9:] = torch.as_tensor(s[QVEL], dtype=art.dtype, device=art.device)
    cube = d["actors"]["cube"]
    cube[..., :7] = torch.as_tensor(s[OBJ_POSE], dtype=cube.dtype, device=cube.device)
    cube[..., 7:] = 0.0
    gs = d["actors"]["goal_site"]
    gs[..., :3] = torch.as_tensor(s[GOAL], dtype=gs.dtype, device=gs.device)
    u.set_state_dict(d)


def grab(env):
    f = env.render()
    if torch.is_tensor(f):
        f = f.detach().cpu().numpy()
    f = np.asarray(f)
    return f.reshape(f.shape[-3], f.shape[-2], f.shape[-1]).astype(np.uint8)


def caption(frame, *, title, tcolor, t, n, hz, cz, spd, phase):
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    cv = Image.new("RGB", (w, h + HEADER), (12, 12, 12))
    cv.paste(Image.fromarray(frame), (0, HEADER))
    d = ImageDraw.Draw(cv)
    f13, f17 = _font(13), _font(17)
    pc = {"APPROACH": (120, 220, 140), "CARRY": (120, 200, 255), "FROZEN": (235, 120, 120)}[phase]
    d.text((8, 5), title, font=f17, fill=tcolor)
    d.text((8, 27), f"frame {t:>3}/{n:<3}   {phase}", font=f13, fill=pc)
    d.text((8, 45), f"hand z {hz:5.3f}   cube z {cz:5.3f}", font=f13, fill=(215, 215, 215))
    d.text((8, 63), f"hand speed {100 * spd:5.2f} cm/frame", font=f13,
           fill=(255, 205, 70) if spd > 0.03 else (215, 215, 215))
    return np.asarray(cv)


def episode_frames(env, u, s, title, tcolor, around_grasp=None):
    """Frames for one recorded episode. around_grasp=N renders only the window
    [grasp-N, grasp+N] -- the moment the jaws close, where placement quality is
    actually visible; the full episode makes the grasp a couple of frames."""
    n = len(s)
    ap = pickcube_approach_len(s)
    if around_grasp:
        lo, hi = max(ap - around_grasp, 0), min(ap + around_grasp, n)
        s = s[lo:hi]
        n = len(s)
        ap = ap - lo
    mv = np.concatenate([[0.0], np.linalg.norm(np.diff(s[:, PC_TCP_POS], axis=0), axis=1)])
    act = np.where(mv > 0.001)[0]
    frozen_from = int(act[-1]) + 1 if len(act) else 0
    out = []
    for t in range(n):
        write_state(u, s[t])
        phase = "APPROACH" if t < ap else ("FROZEN" if t >= frozen_from else "CARRY")
        out.append(caption(grab(env), title=title, tcolor=tcolor, t=t, n=n - 1,
                           hz=float(s[t, PC_TCP_POS][2]), cz=float(s[t, PC_OBJ_POS][2]),
                           spd=float(mv[t]), phase=phase))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=str(REPO / "outputs/pickcube/ndemos_v1/ndemos_100/subset_100.h5"))
    ap.add_argument("--new", default=str(REPO / "outputs/pickcube/teleop_v1/teleop_100.h5"))
    ap.add_argument("--n", type=int, default=3, help="episodes from each dataset")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/pickcube/teleop_v1/comparison"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--around-grasp", type=int, default=None,
                    help="render only +/- N frames around the moment the cube leaves the table")
    ap.add_argument("--old-label", default="RL BUNDLE (bang-bang)")
    ap.add_argument("--new-label", default="TELEOP-LIKE (slow + wander)")
    ap.add_argument("--camera", choices=("wide", "close"), default="wide")
    ap.add_argument("--single", action="store_true",
                    help="render only --new, full width (no side-by-side)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.camera == "close":
        # Replay writes sim state directly, so the camera is free: pull it in
        # close on the grasp region, where wrist alignment and jaw placement are
        # actually judgeable by eye. The default wide view makes the gripper a
        # few pixels across.
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils import sapien_utils
        env = gym.make(
            "PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose",
            sim_backend="physx_cuda", num_envs=1, render_mode="rgb_array",
            human_render_camera_configs=dict(
                pose=sapien_utils.look_at(eye=[0.30, 0.30, 0.24], target=[-0.04, 0.0, 0.03]),
                width=512, height=512, fov=0.75),
        )
    else:
        env = make_env(task_id="PickCube-v1", control_mode="pd_ee_delta_pose",
                       sim_backend="physx_cuda", obs_mode="state",
                       render_mode="rgb_array", max_episode_steps=200)
    env.reset(seed=0)
    u = env.unwrapped

    old = DatasetReader(args.old); new = DatasetReader(args.new)
    for i in range(args.n):
        so = old.episode(old.episode_ids[i]).state
        sn = new.episode(new.episode_ids[i]).state
        fn = episode_frames(env, u, sn, args.new_label, (120, 220, 140), args.around_grasp)
        if args.single:
            save_video(fn, out / f"episode_{i}.mp4", fps=args.fps)
            print(f"[cmp] ep {i}: {len(fn)} frames -> episode_{i}.mp4", flush=True)
            strips = [fn]
        else:
            fo = episode_frames(env, u, so, args.old_label, (255, 165, 45), args.around_grasp)
            n = max(len(fo), len(fn))
            pad = lambda L: L + [L[-1]] * (n - len(L))
            save_video([np.concatenate([a, b], axis=1) for a, b in zip(pad(fo), pad(fn))],
                       out / f"compare_{i}.mp4", fps=args.fps)
            print(f"[cmp] ep {i}: RL {len(fo)} frames, teleop {len(fn)} frames "
                  f"-> compare_{i}.mp4", flush=True)
            strips = [fo, fn]

        # filmstrip: 8 evenly spaced frames from each, one row per demonstrator
        from PIL import Image
        rows = []
        for frames in strips:
            idx = np.linspace(0, len(frames) - 1, 8).astype(int)
            rows.append(np.concatenate([frames[j] for j in idx], axis=1))
        W = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 6), (0, W - r.shape[1]), (0, 0))) for r in rows]
        Image.fromarray(np.concatenate(rows, axis=0)).save(out / f"filmstrip_{i}.png")
    env.close()
    print(f"[cmp] artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

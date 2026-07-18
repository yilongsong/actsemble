#!/usr/bin/env python
"""Build the STANDARDIZED Actsemble rollout-comparison dashboard.

Two stages:

  record   render a selector system's rollouts to the standard capture schema
           (per-step T/EE trajectory + goal coverage + optional mp4 + auto
           failure classification). Works for ANY AutonomySystem via the factory
           (candidate_zero / consensus / verifier). GPU. Oracle rollouts are
           produced by scripts/oracle_headroom.py (capture/video).

  build    ingest available captures + paired eval + headroom into one manifest
           and emit a self-contained dashboard/<name>.html (viewer template with
           DATA injected). Videos copied to dashboard/videos/. No GPU.

The manifest schema (the stable project contract) is documented in
dashboard/README.md; the viewer is dashboard/_viewer.html. Any experiment that
emits this manifest renders in the same interface — this is the standard for the
whole project's comparisons and iterations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.utils.serialization import load_json  # noqa: E402

DASH = REPO / "dashboard"
GOAL_YAW = (5 / 3) * np.pi

# ---- canonical failure taxonomy (shared by every producer + the viewer) ----
FAILURE_TAXONOMY = {
    "success": {"label": "Success", "color": "#2a9d8f"},
    "near_miss": {"label": "Near miss", "color": "#e76f51"},
    "misaligned": {"label": "Misaligned", "color": "#e9c46a"},
    "mispositioned": {"label": "Mispositioned", "color": "#457b9d"},
    "never_engaged": {"label": "Never engaged", "color": "#8d99ae"},
    "unknown": {"label": "Unknown", "color": "#c7ccd4"},
}
MODE_ORDER = [
    "success",
    "near_miss",
    "misaligned",
    "mispositioned",
    "never_engaged",
    "unknown",
]
WORLD = {"x0": -0.46, "x1": 0.24, "y0": -0.36, "y1": 0.14}  # PushT table extent


def classify(success, obj_xy, coverage, goal_xy):
    """Canonical outcome classification from a rollout's T-block trajectory."""
    if success:
        return "success"
    if obj_xy is None or len(obj_xy) < 2:
        return "unknown"
    obj = np.asarray(obj_xy, float)
    cov = np.asarray(coverage, float)
    disp = np.linalg.norm(obj - obj[0], axis=1).max()
    pos = np.linalg.norm(obj[-1] - np.asarray(goal_xy, float))
    if disp < 0.02:
        return "never_engaged"
    if cov.max() >= 0.75:
        return "near_miss"
    if pos <= 0.05:
        return "misaligned"
    return "mispositioned"


def _ds(arr, n=28):
    """Downsample a sequence to ~n points for compact embedding."""
    arr = list(arr)
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return [arr[i] for i in idx]


def episode_from_traj(
    env_seed, success, obj_xy, coverage, goal_xy, num_steps, video=None
):
    obj = np.asarray(obj_xy, float)
    cov = np.asarray(coverage, float)
    ftype = classify(success, obj_xy, coverage, goal_xy)
    return {
        "env_seed": int(env_seed),
        "success": bool(success),
        "failure_type": ftype,
        "final_cov": round(float(cov[-1]), 3),
        "max_cov": round(float(cov.max()), 3),
        "steps": int(num_steps),
        "pos_error": round(
            float(np.linalg.norm(obj[-1] - np.asarray(goal_xy, float))), 4
        ),
        "obj_disp": round(float(np.linalg.norm(obj - obj[0], axis=1).max()), 4),
        "traj": {
            "obj": [[round(x, 4), round(y, 4)] for x, y in _ds(obj.tolist())],
            "cov": [round(float(c), 3) for c in _ds(cov.tolist())],
        },
        "video": video,
    }


def load_rich_capture(path, video_for=None):
    """Load a capture json (base/oracle/recorded) -> list of episode records."""
    rows = load_json(path)["episodes"]
    eps = []
    for r in rows:
        vid = video_for(r["env_seed"]) if video_for else None
        eps.append(
            episode_from_traj(
                r["env_seed"],
                r["success_once"],
                r["obj_xy"],
                r["coverage"],
                r["goal_xy"],
                r.get("num_steps", len(r["coverage"])),
                vid,
            )
        )
    return eps


def load_success_only(path):
    """Load a paired-eval json (evaluate_system) -> {env_seed: success}."""
    d = load_json(path)
    seeds = d.get("environment_seeds") or [e["env_seed"] for e in d["episodes"]]
    return {int(s): bool(ok) for s, ok in zip(seeds, d["successes"])}


def system_block(
    name, label, kind, color, episodes=None, success_map=None, goal_xy=None
):
    """Build one system entry (rich episodes OR success-only)."""
    if episodes is None and success_map is not None:
        episodes = [
            {
                "env_seed": int(s),
                "success": bool(ok),
                "failure_type": "unknown",
                "final_cov": None,
                "max_cov": None,
                "steps": None,
                "traj": None,
                "video": None,
            }
            for s, ok in sorted(success_map.items())
        ]
    succ = [e["success"] for e in episodes]
    n, sc = len(succ), int(np.sum(succ))
    fc = {}
    for e in episodes:
        k = "success" if e["success"] else e["failure_type"]
        fc[k] = fc.get(k, 0) + 1
    steps = [e["steps"] for e in episodes if e.get("steps") is not None]
    fcov = [e["final_cov"] for e in episodes if e.get("final_cov") is not None]
    return {
        "name": name,
        "label": label,
        "kind": kind,
        "color": color,
        "metrics": {
            "success_rate": sc / n if n else None,
            "success_count": sc,
            "n": n,
            "wilson_ci": list(wilson_interval(sc, n)) if n else None,
            "mean_steps": round(float(np.mean(steps)), 1) if steps else None,
            "mean_final_cov": round(float(np.mean(fcov)), 3) if fcov else None,
        },
        "failure_counts": fc,
        "episodes": episodes,
    }


def mcnemar(succ_map_a, succ_map_b):
    seeds = sorted(set(succ_map_a) & set(succ_map_b))
    a = np.array([succ_map_a[s] for s in seeds], int)
    b = np.array([succ_map_b[s] for s in seeds], int)
    a1 = int(((a == 1) & (b == 0)).sum())
    b1 = int(((a == 0) & (b == 1)).sum())
    disc = a1 + b1
    z = (a1 - b1) / np.sqrt(disc) if disc else 0.0
    return {
        "delta": float(a.mean() - b.mean()) if len(seeds) else 0.0,
        "mcnemar_z": float(z),
        "sig": bool(abs(z) > 1.96),
        "a_wins": a1,
        "b_wins": b1,
        "n": len(seeds),
    }


def succ_map(system):
    return {e["env_seed"]: e["success"] for e in system["episodes"]}


# --------------------------------------------------------------- build ----
def video_index(video_dir, suffix):
    """seed -> relative video path for files matching *_{suffix}.mp4."""
    out = {}
    vdir = Path(video_dir)
    if not vdir.exists():
        return out
    for f in vdir.glob(f"*_{suffix}.mp4"):
        # filenames: seed<ES>_<label>_<suffix>.mp4
        try:
            es = int(f.name.split("_")[0].replace("seed", ""))
        except ValueError:
            continue
        out[es] = f
    return out


def cmd_build(args):
    DASH.mkdir(exist_ok=True)
    vids_out = DASH / "videos"
    vids_out.mkdir(exist_ok=True)
    cap = REPO / "outputs/active_min/oracle/capture"
    cmp = REPO / "outputs/active_min/compare"
    vdir = REPO / "outputs/active_min/oracle/videos"

    # video maps (copy referenced videos into dashboard/videos/)
    copied = {}

    def relvid(src: Path):
        if src is None:
            return None
        dst = vids_out / src.name
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
        copied[src.name] = True
        return f"videos/{src.name}"

    base_vids = video_index(vdir, "base")
    orac_vids = video_index(vdir, "oracle")

    systems = []
    # candidate_zero — rich (base capture, trajectories)
    base_path = cap / "base_0000_0300.json"
    if base_path.exists():
        eps = load_rich_capture(
            base_path, video_for=lambda es: relvid(base_vids.get(es))
        )
        systems.append(
            system_block(
                "candidate_zero",
                "Candidate Zero (base)",
                "base",
                "#6c757d",
                episodes=eps,
            )
        )
    # oracle — rich (fixed capture, trajectories) if present
    orac_shards = sorted(cap.glob("oracle_*.json"))
    if orac_shards:
        rows = {}
        for s in orac_shards:
            for r in load_json(s)["episodes"]:
                rows[r["env_seed"]] = r
        eps = [
            episode_from_traj(
                r["env_seed"],
                r["success_once"],
                r["obj_xy"],
                r["coverage"],
                r["goal_xy"],
                r.get("num_steps", len(r["coverage"])),
                relvid(orac_vids.get(r["env_seed"])),
            )
            for r in rows.values()
        ]
        systems.append(
            system_block(
                "oracle", "Oracle@16 (ceiling)", "oracle", "#2a9d8f", episodes=eps
            )
        )
    # verifier / medoid — success-only from paired eval (rich via `record` later)
    for nm, label, color, fn in [
        (
            "verifier_argmax",
            "Verifier (learned)",
            "#6a4c93",
            cmp / "verifier_argmax.json",
        ),
        (
            "full_chunk_medoid",
            "Medoid (consensus)",
            "#457b9d",
            cmp / "full_chunk_medoid.json",
        ),
    ]:
        if fn.exists():
            systems.append(
                system_block(
                    nm, label, "selector", color, success_map=load_success_only(fn)
                )
            )

    # recorded rich captures (from `record`) override/extend if present
    for rc in (
        sorted((DASH / "captures").glob("*.json"))
        if (DASH / "captures").exists()
        else []
    ):
        meta = load_json(rc).get("system", {})
        nm = meta.get("name")
        nmvids = video_index(vdir, nm)
        eps = load_rich_capture(rc, video_for=lambda es, _v=nmvids: relvid(_v.get(es)))
        blk = system_block(
            nm,
            meta.get("label", nm),
            meta.get("kind", "selector"),
            meta.get("color", "#888"),
            episodes=eps,
        )
        systems = [s for s in systems if s["name"] != nm] + [blk]

    smaps = {s["name"]: succ_map(s) for s in systems}
    names = [s["name"] for s in systems]

    # contrasts (paired McNemar on shared seeds) — canonical ordering
    contrasts = []

    def add(a, b, note=""):
        if a in smaps and b in smaps:
            c = mcnemar(smaps[a], smaps[b])
            c.update({"a": a, "b": b, "note": note})
            contrasts.append(c)

    add("verifier_argmax", "candidate_zero", "learned selector vs base")
    add("full_chunk_medoid", "candidate_zero", "consensus vs base")
    add("oracle", "candidate_zero", "ceiling vs base = selection headroom")
    add("oracle", "verifier_argmax", "ceiling vs learned = unrealized headroom")

    highlights = []
    if {"oracle", "candidate_zero"} <= set(names):
        seeds = sorted(set(smaps["oracle"]) & set(smaps["candidate_zero"]))
        o = np.mean([smaps["oracle"][s] for s in seeds])
        c = np.mean([smaps["candidate_zero"][s] for s in seeds])
        v = None
        if "verifier_argmax" in smaps:
            vseeds = sorted(set(smaps["verifier_argmax"]) & set(seeds))
            v = (
                np.mean([smaps["verifier_argmax"][s] for s in vseeds])
                if vseeds
                else None
            )
        highlights.append(
            {
                "type": "headroom",
                "oracle": float(o),
                "cz": float(c),
                "verifier": float(v) if v is not None else None,
                "captured": float((v - c) / (o - c))
                if (v is not None and o > c)
                else None,
                "k": 16,
                "note": f"paired on {len(seeds)} episodes; oracle is a diagnostic "
                "upper bound (privileged sim access, drift-fixed two-env).",
            }
        )

    goal_xy = [-0.156, -0.10]
    for s in systems:
        for e in s["episodes"]:
            if e.get("traj"):
                goal_xy = (
                    load_json(base_path)["episodes"][0]["goal_xy"]
                    if base_path.exists()
                    else goal_xy
                )
                break
        else:
            continue
        break

    manifest = {
        "schema_version": "1.0",
        "experiment": args.name,
        "title": args.title
        or "Selection headroom — candidate_zero → verifier → Oracle@16 "
        "(PushT-v1, hold-trimmed data)",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "panel": make_panel("diagnostic").to_dict(),
        "goal_xy": goal_xy,
        "goal_tolerance": 0.05,
        "world": WORLD,
        "failure_taxonomy": FAILURE_TAXONOMY,
        "mode_order": MODE_ORDER,
        "systems": systems,
        "contrasts": contrasts,
        "highlights": highlights,
    }

    # optional recovery cross-tab (base -> oracle)
    if {"candidate_zero", "oracle"} <= set(names):
        bmap = {e["env_seed"]: e for e in sysget(systems, "candidate_zero")["episodes"]}
        omap = {e["env_seed"]: e for e in sysget(systems, "oracle")["episodes"]}
        rows = []
        for mode in ["near_miss", "misaligned", "mispositioned", "never_engaged"]:
            ids = [
                es
                for es, e in bmap.items()
                if es in omap and not e["success"] and e["failure_type"] == mode
            ]
            rec = sum(omap[es]["success"] for es in ids)
            if ids:
                rows.append({"mode": mode, "count": len(ids), "recovered": rec})
        if rows:
            manifest["recovery"] = {
                "title": "Recovery by base failure mode (base fails → oracle succeeds)",
                "rows": rows,
            }

    out_html = DASH / f"{args.name}.html"
    tpl = (DASH / "_viewer.html").read_text()
    out_html.write_text(
        tpl.replace("__ACTSEMBLE_DATA__", json.dumps(manifest, separators=(",", ":")))
    )
    (DASH / f"{args.name}.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"[dashboard] {out_html}  ({len(systems)} systems, "
        f"{sum(len(s['episodes']) for s in systems)} episode records, {len(copied)} videos)"
    )
    for s in systems:
        m = s["metrics"]
        print(
            f"    {s['label']:26s} {(m['success_rate'] or 0):.1%}  n={m['n']}  "
            f"modes={ {k: v for k, v in s['failure_counts'].items() if k != 'success'} }"
        )
    return 0


def sysget(systems, name):
    return next(s for s in systems if s["name"] == name)


# -------------------------------------------------------------- record ----
def cmd_record(args):
    """Render a selector system's rollouts to the standard capture schema."""
    from actsemble.policies.diffusion.policy import DiffusionPolicy
    from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility
    from actsemble.systems.factory import build_system
    from actsemble.sim.env_factory import make_env
    from actsemble.sim.observation_adapter import ObservationAdapter
    from actsemble.sim.action_adapter import ActionAdapter
    from actsemble.evaluation.video import save_video
    from actsemble.evaluation.panels import Panel, panel_episodes
    import torch

    cfg = json.loads(
        args.system
    )  # {"name","label","kind","color","selection":{...},"num_candidates":16,"components":[...]}
    policy = DiffusionPolicy.from_checkpoint(
        args.policy, device=args.device or "cuda", use_ema=True
    )
    comps = [
        ActionChunkCompatibility.from_checkpoint(p, device=args.device or "cuda")
        for p in cfg.get("components", [])
    ]
    system = build_system(
        {
            "policy": {"num_candidates": cfg.get("num_candidates", 16)},
            "selection": cfg["selection"],
        },
        policy,
        comps,
    )
    vseeds = {int(s) for s in args.video_seeds.split(",")} if args.video_seeds else None
    want_video = vseeds is not None or int(args.videos) > 0
    env = make_env(
        task_id=policy.meta.task_id,
        control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend,
        obs_mode="state",
        render_mode="rgb_array" if want_video else None,
    )
    u = env.unwrapped
    obs_ad = ObservationAdapter(action_dim=policy.meta.action_dim)
    act_ad = ActionAdapter(
        np.asarray(u.single_action_space.low, np.float32),
        np.asarray(u.single_action_space.high, np.float32),
    )
    diag = make_panel("diagnostic")
    panel = Panel(diag.name, diag.env_seed, int(args.count))
    vdir = REPO / "outputs/active_min/oracle/videos"
    vdir.mkdir(parents=True, exist_ok=True)
    (DASH / "captures").mkdir(parents=True, exist_ok=True)

    def img():
        f = env.render()
        f = f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f)
        return (f[0] if f.ndim == 4 else f).astype(np.uint8)

    rows = []
    t0 = time.time()
    for i, ep in enumerate(panel_episodes(panel)):
        system.candidate_root_seed = ep.policy_sampling_seed
        system.reset(episode_seed=ep.env_seed)
        raw, _ = env.reset(seed=int(ep.env_seed))
        obs_ad = ObservationAdapter(action_dim=policy.meta.action_dim)
        do_render = want_video and (
            ep.env_seed in vseeds if vseeds is not None else i < int(args.videos)
        )
        s = np.asarray(obs_ad.observe(raw).state, np.float32).reshape(-1)
        obj = [s[24:26].copy()]
        cov = [float(u.pseudo_render_intersection().item())]
        frames = [img()] if do_render else []
        succ_once = False
        step = 0
        while step < 100 and not succ_once:
            action = system.act(obs_ad.observe(raw))
            raw, _, term, trunc, info = env.step(act_ad.adapt(action))
            obs_ad.after_step(act_ad.adapt(action))
            s = np.asarray(obs_ad.observe(raw).state, np.float32).reshape(-1)
            obj.append(s[24:26].copy())
            cov.append(float(u.pseudo_render_intersection().item()))
            if do_render:
                frames.append(img())
            succ_once = succ_once or (
                bool(info["success"].reshape(-1)[0]) if "success" in info else False
            )
            step += 1
            if bool(np.asarray(term).reshape(-1)[0]) or bool(
                np.asarray(trunc).reshape(-1)[0]
            ):
                break
        goal = s[21:23]
        rows.append(
            {
                "env_seed": int(ep.env_seed),
                "policy_sampling_seed": int(ep.policy_sampling_seed),
                "success_once": bool(succ_once),
                "num_steps": int(step),
                "obj_xy": np.asarray(obj, np.float32).round(4).tolist(),
                "coverage": np.asarray(cov, np.float32).round(4).tolist(),
                "goal_xy": np.asarray(goal, np.float32).round(4).tolist(),
            }
        )
        if do_render:
            save_video(frames, vdir / f"seed{ep.env_seed}_{cfg['name']}.mp4")
        if (i + 1) % 20 == 0:
            print(
                f"[record:{cfg['name']}] {i + 1}/{panel.num_episodes} ({(time.time() - t0) / (i + 1):.1f}s/ep)",
                flush=True,
            )
    env.close()
    out = DASH / "captures" / f"{cfg['name']}.json"
    out.write_text(
        json.dumps(
            {
                "system": {
                    k: cfg[k] for k in ("name", "label", "kind", "color") if k in cfg
                },
                "episodes": rows,
            }
        )
    )
    print(
        f"[record:{cfg['name']}] wrote {out} ({len(rows)} eps, "
        f"{np.mean([r['success_once'] for r in rows]):.1%})"
    )
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", default="selection_headroom")
    b.add_argument("--title", default=None)
    r = sub.add_parser("record")
    r.add_argument("--system", required=True)
    r.add_argument("--policy", required=True)
    r.add_argument("--count", default=300)
    r.add_argument("--videos", default=0, type=int)
    r.add_argument(
        "--video-seeds",
        default=None,
        help="comma-separated env_seeds to render video for",
    )
    r.add_argument("--device", default=None)
    args = p.parse_args()
    return {"build": cmd_build, "record": cmd_record}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Prove: higher replanning frequency -> better closed-loop performance.

Sweeps the receding-horizon execution horizon ``H_a`` (steps executed per replan)
of the plain standalone policy — NO temporal ensembling, no averaging, one knob.
Replanning frequency is the *true* control-time rate of issuing a fresh plan:

    replan_freq (Hz) = control_freq (Hz) / H_a

not the inference-latency rate (1/time_to_replan) and not H_a itself. The ceiling
is ``control_freq`` (replan every control step, H_a=1); going higher needs a finer
control timestep. Reports success vs replan_freq (monotone if the hypothesis
holds) with McNemar vs the slowest (most open-loop) horizon.

Usage:
    python scripts/sweep_replan_frequency.py --horizons 1,2,4,8,16 --count 300
    python scripts/sweep_replan_frequency.py --report-only
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402
from compare_temporal import mcnemar  # noqa: E402


def run_sweep(policy_ckpt, horizons, count, panel_name, out_dir: Path, device) -> float:
    panel = make_panel(panel_name)
    eval_cfg = {
        "name": panel_name, "regime": "nominal", "max_steps": 100,
        "panel": {"name": panel.name, "env_seed": panel.env_seed, "num_episodes": count},
        "perturbations": [],
    }
    policy = load_policy(policy_ckpt, device=device, use_ema=True)
    env = make_env(
        task_id=policy.meta.task_id, control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend, obs_mode="state", render_mode="rgb_array",
    )
    control_freq = float(getattr(env.unwrapped, "control_freq", 20))
    del policy
    out_dir.mkdir(parents=True, exist_ok=True)
    for ha in horizons:
        cfg = {
            "name": "standalone_diffusion",
            "policy": {"type": "state_diffusion", "frozen": True, "num_candidates": 1},
            "components": [], "selection": {"type": "candidate_zero"},
            "execution": {"action_horizon": int(ha)},
        }
        print(f"[replan-hz] standalone H_a={ha} ({control_freq/ha:.2f} Hz) on {panel_name} "
              f"({count} ep) ...", flush=True)
        evaluate_system(
            system_cfg=cfg, eval_cfg=eval_cfg, policy_checkpoint=policy_ckpt,
            num_episodes=count, output_path=out_dir / f"eval_standalone_ha{int(ha):02d}.json",
            device=device, env=env, force=True,
        )
    env.close()
    save_json({"control_freq_hz": control_freq}, out_dir / "replan_meta.json")
    return control_freq


def build_and_report(out_dir: Path) -> dict | None:
    files = sorted(glob.glob(str(out_dir / "eval_standalone_ha*.json")))
    if not files:
        print("[replan-hz] no results found in", out_dir)
        return None
    meta_p = out_dir / "replan_meta.json"
    control_freq = float(load_json(meta_p)["control_freq_hz"]) if meta_p.exists() else 20.0
    results = [load_json(f) for f in files]
    # H_a from the executed system config; slowest (largest H_a) is the reference
    def h_a(r):
        return int(r["config"]["system"]["execution"]["action_horizon"])
    results.sort(key=h_a)
    n = results[0]["num_episodes"]
    ref = max(results, key=h_a)  # most open-loop = lowest replan freq
    points = []
    for r in results:
        ci = wilson_interval(r["success_count"], n)
        mc = mcnemar(r["successes"], ref["successes"])
        points.append({
            "h_a": h_a(r), "replan_hz": control_freq / h_a(r),
            "success_rate": r["success_rate"], "wilson_ci": list(ci),
            "vs_slowest": {"diff": r["success_rate"] - ref["success_rate"],
                           "mcnemar": mc},
            "mean_replans": float(np.mean([len(e["replans"]) for e in r["episodes"]])),
        })
    successes = [p["success_rate"] for p in sorted(points, key=lambda q: q["replan_hz"])]
    monotone = all(b >= a - 1e-9 for a, b in zip(successes, successes[1:]))
    report = {"panel": results[0]["panel"]["name"], "num_episodes": n,
              "control_freq_hz": control_freq, "monotone_in_freq": monotone, "points": points}
    print(_format(report))
    save_json(report, out_dir / "replan_frequency.json")
    _plot(report, out_dir / "replan_frequency.png")
    print(f"[replan-hz] wrote {out_dir/'replan_frequency.json'} and replan_frequency.png")
    return report


def _format(rep: dict) -> str:
    L = ["=" * 74,
         f"Replanning frequency sweep — {rep['panel']} panel, n={rep['num_episodes']}, "
         f"control={rep['control_freq_hz']:.0f} Hz",
         "=" * 74,
         f"{'H_a':>4}{'replan Hz':>11}{'success':>10}{'wilson 95% CI':>20}{'vs slowest':>15}"]
    L.append("-" * 74)
    for p in sorted(rep["points"], key=lambda q: q["h_a"], reverse=True):
        ci = p["wilson_ci"]
        d = p["vs_slowest"]
        sig = "*" if d["mcnemar"]["p_exact"] < 0.05 else ""
        ci_s = f"[{ci[0]:.1%},{ci[1]:.1%}]"
        diff_s = f"{d['diff']:+.1%}{sig}"
        L.append(f"{p['h_a']:>4}{p['replan_hz']:>10.2f}{p['success_rate']:>10.1%}"
                 f"{ci_s:>20}{diff_s:>15}")
    L.append("-" * 74)
    L.append(f"monotone increasing in replan frequency: {rep['monotone_in_freq']}")
    return "\n".join(L)


def _plot(rep: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[replan-hz] skipping plot ({e})")
        return
    pts = sorted(rep["points"], key=lambda q: q["replan_hz"])
    xs = [p["replan_hz"] for p in pts]
    ys = [p["success_rate"] for p in pts]
    lo = [p["success_rate"] - p["wilson_ci"][0] for p in pts]
    hi = [p["wilson_ci"][1] - p["success_rate"] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color="#2a6f97", capsize=3, lw=2)
    for p in pts:
        ax.annotate(f"H_a={p['h_a']}", (p["replan_hz"], p["success_rate"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")
    ax.axvline(rep["control_freq_hz"], color="#c1121f", ls="--", lw=1.2,
               label=f"control freq ceiling {rep['control_freq_hz']:.0f} Hz")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("replanning frequency  (Hz = control_freq / H_a)")
    ax.set_ylabel("success rate")
    ax.set_title(f"More replanning → better — {rep['panel']} panel (n={rep['num_episodes']})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/active_min/policy_seed_0/selected_policy.pt"))
    ap.add_argument("--horizons", default="1,2,4,8,16")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--panel", default="diagnostic")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/active_min/replan_freq"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not args.report_only:
        horizons = [int(x) for x in args.horizons.split(",")]
        run_sweep(args.policy_checkpoint, horizons, args.count, args.panel, out_dir, args.device)
    build_and_report(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Sweep the ACT temporal-ensemble decay `m` — the one ACT hyperparameter —
against the receding-horizon baseline, on the diagnostic panel.

Weight of a prediction of age `a` is `exp(-m*a)` (a=0 freshest, renormalized):
  * m = 0        -> uniform average over all overlapping predictions (max ensembling)
  * m -> infinity -> only the freshest prediction  == `temporal_latest` (no ensembling)
So the sweep traces how much the AVERAGING helps on top of querying-every-step.
`standalone_diffusion` (baseline) and `temporal_latest` (the m->inf reference) are
read from the same out-dir if already evaluated there; only `temporal_mean@m` is run.

Usage:
    python scripts/sweep_temporal_decay.py --decays 0.0,0.05,0.1,0.2,0.5 --count 300
    python scripts/sweep_temporal_decay.py --report-only            # just re-tabulate + plot
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from actsemble.config import load_config  # noqa: E402
from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402
from compare_temporal import mcnemar  # noqa: E402  (shared paired-outcome test)

BASELINE_JSON = "eval_standalone_diffusion.json"
LATEST_JSON = "eval_temporal_latest.json"


def run_sweep(policy_ckpt, decays, count, panel_name, out_dir: Path, device):
    panel = make_panel(panel_name)
    eval_cfg = {
        "name": panel_name,
        "regime": "nominal",
        "max_steps": 100,
        "panel": {
            "name": panel.name,
            "env_seed": panel.env_seed,
            "num_episodes": count,
        },
        "perturbations": [],
    }
    policy = load_policy(policy_ckpt, device=device, use_ema=True)
    env = make_env(
        task_id=policy.meta.task_id,
        control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend,
        obs_mode="state",
        render_mode="rgb_array",
    )
    del policy
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(str(REPO / "configs/systems/temporal_mean.yaml"))
    for m in decays:
        cfg = {**base_cfg, "selection": {**base_cfg["selection"], "decay": float(m)}}
        print(
            f"[sweep] temporal_mean decay={m} on {panel_name} ({count} ep) ...",
            flush=True,
        )
        evaluate_system(
            system_cfg=cfg,
            eval_cfg=eval_cfg,
            policy_checkpoint=policy_ckpt,
            num_episodes=count,
            output_path=out_dir / f"eval_temporal_mean_m{m:.2f}.json",
            device=device,
            env=env,
            force=True,
        )
    env.close()


def _decay_of(result: dict) -> float:
    return float(result["config"]["system"]["selection"].get("decay", 0.1))


def build_and_report(out_dir: Path) -> dict | None:
    base_p, latest_p = out_dir / BASELINE_JSON, out_dir / LATEST_JSON
    base = load_json(base_p) if base_p.exists() else None
    latest = load_json(latest_p) if latest_p.exists() else None
    means = [
        load_json(p)
        for p in sorted(glob.glob(str(out_dir / "eval_temporal_mean*.json")))
    ]
    means.sort(key=_decay_of)
    if not means:
        print("[sweep] no temporal_mean results found in", out_dir)
        return None
    n = means[0]["num_episodes"]

    def row(r, label):
        ci = wilson_interval(r["success_count"], n)
        d = {"label": label, "success_rate": r["success_rate"], "wilson_ci": list(ci)}
        if base is not None:
            d["mcnemar_vs_baseline"] = mcnemar(r["successes"], base["successes"])
            d["diff_vs_baseline"] = r["success_rate"] - base["success_rate"]
        return d

    sweep = [
        {**row(r, f"mean@m={_decay_of(r):g}"), "decay": _decay_of(r)} for r in means
    ]
    report = {
        "panel": means[0]["panel"]["name"],
        "num_episodes": n,
        "baseline": row(base, "standalone_diffusion") if base is not None else None,
        "latest": row(latest, "temporal_latest") if latest is not None else None,
        "sweep": sweep,
    }
    print(_format(report))
    save_json(report, out_dir / "decay_sweep.json")
    _plot(report, out_dir / "decay_sweep.png")
    print(f"[sweep] wrote {out_dir / 'decay_sweep.json'} and decay_sweep.png")
    return report


def _format(rep: dict) -> str:
    L = [
        "=" * 72,
        f"ACT temporal-ensemble decay sweep — {rep['panel']} panel, n={rep['num_episodes']}",
        "=" * 72,
    ]
    if rep["baseline"]:
        b = rep["baseline"]
        ci = b["wilson_ci"]
        L.append(
            f"baseline (standalone) {b['success_rate']:>7.1%}  [{ci[0]:.1%},{ci[1]:.1%}]"
        )
    if rep["latest"]:
        lt = rep["latest"]
        ci = lt["wilson_ci"]
        L.append(
            f"temporal_latest (m=inf){lt['success_rate']:>6.1%}  [{ci[0]:.1%},{ci[1]:.1%}]"
            + (
                f"  vs base {lt['diff_vs_baseline']:+.1%} p={lt['mcnemar_vs_baseline']['p_exact']:.3f}"
                if "diff_vs_baseline" in lt
                else ""
            )
        )
    L.append("-" * 72)
    L.append(f"{'decay m':>9}{'success':>10}{'wilson 95% CI':>20}{'vs baseline':>14}")
    for s in rep["sweep"]:
        ci = s["wilson_ci"]
        extra = ""
        if "diff_vs_baseline" in s:
            sig = "*" if s["mcnemar_vs_baseline"]["p_exact"] < 0.05 else ""
            extra = f"{s['diff_vs_baseline']:+.1%} p={s['mcnemar_vs_baseline']['p_exact']:.3f}{sig}"
        L.append(
            f"{s['decay']:>9g}{s['success_rate']:>9.1%}{f'[{ci[0]:.1%},{ci[1]:.1%}]':>20}{extra:>16}"
        )
    return "\n".join(L)


def _plot(rep: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print(f"[sweep] skipping plot ({e})")
        return
    ms = [s["decay"] for s in rep["sweep"]]
    ys = [s["success_rate"] for s in rep["sweep"]]
    lo = [s["success_rate"] - s["wilson_ci"][0] for s in rep["sweep"]]
    hi = [s["wilson_ci"][1] - s["success_rate"] for s in rep["sweep"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(
        ms,
        ys,
        yerr=[lo, hi],
        marker="o",
        color="#2a6f97",
        capsize=3,
        lw=2,
        label="temporal_mean (ACT) @ m",
    )
    if rep["baseline"]:
        b = rep["baseline"]
        ax.axhspan(b["wilson_ci"][0], b["wilson_ci"][1], color="#999999", alpha=0.18)
        ax.axhline(
            b["success_rate"],
            color="#555555",
            ls="-",
            lw=1.2,
            label=f"baseline (H_a=8) {b['success_rate']:.0%}",
        )
    if rep["latest"]:
        ax.axhline(
            rep["latest"]["success_rate"],
            color="#c1121f",
            ls="--",
            lw=1.4,
            label=f"temporal_latest (m=inf) {rep['latest']['success_rate']:.0%}",
        )
    ax.set_xlabel("decay m   (0 = uniform average  →  large = only freshest)")
    ax.set_ylabel("success rate")
    ax.set_title(
        f"ACT temporal-ensemble decay sweep — {rep['panel']} panel (n={rep['num_episodes']})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy-checkpoint",
        default=str(REPO / "outputs/active_min/policy_seed_0/selected_policy.pt"),
    )
    ap.add_argument("--decays", default="0.0,0.05,0.1,0.2,0.5")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--panel", default="diagnostic")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/active_min/temporal"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not args.report_only:
        decays = [float(x) for x in args.decays.split(",")]
        run_sweep(
            args.policy_checkpoint, decays, args.count, args.panel, out_dir, args.device
        )
    build_and_report(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

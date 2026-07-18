#!/usr/bin/env python
"""A4 temporal ensembling vs the receding-horizon baseline (diagnostic panel).

Runs, on the same frozen policy and the same `diagnostic` panel:
  * standalone_diffusion  — receding-horizon baseline (K=1, replan every H_a)
  * temporal_latest       — replan every step, execute-first (compute-matched control)
  * temporal_mean         — classic ACT temporal ensembling (weighted average)
  * temporal_projection   — weighted mean projected onto a real prediction
  * temporal_medoid       — weighted medoid of the overlapping predictions

Two comparisons, with the right fairness control for each (docs/system_architecture.md §2.5):
  1. PRIMARY — each temporal variant vs the standalone baseline. Different
     propose cadence (temporal replans every step), so this is BUDGET-MATCHED,
     not candidate-identity-matched: paired on the env seed (same initial state),
     reported with McNemar + paired bootstrap, alongside the compute cost.
  2. AGGREGATION KNOB — the ensembling variants (mean/projection/medoid) vs
     temporal_latest. All replan every step (identical compute + same cached
     plans at a shared observation), so this isolates the aggregation rule from
     the dense-replanning that temporal_latest already has. In CLOSED loop the
     variants diverge via the emission itself (different action -> different
     trajectory -> different later candidates), which is expected; the exact
     same-plan property is the OPEN-loop one proven in
     tests/systems/test_temporal_ensemble.py.

Usage:
    # development / selection (300-ep diagnostic panel, all variants):
    python scripts/compare_temporal.py --panel diagnostic --count 300
    # reported result (500-ep final_test panel, frozen winner + controls):
    python scripts/compare_temporal.py --panel final_test --count 500 \
        --systems standalone_diffusion,temporal_latest,temporal_medoid \
        --out-dir outputs/active_min/temporal_final
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.config import load_config  # noqa: E402
from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.metrics import (  # noqa: E402
    paired_bootstrap_diff,
    paired_outcome_counts,
    wilson_interval,
)
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

BASELINE = "standalone_diffusion"
CONTROL = (
    "temporal_latest"  # replan-every-step, no ensembling (compute-matched control)
)
ENSEMBLE = ["temporal_mean", "temporal_projection", "temporal_medoid"]
TEMPORAL = [CONTROL, *ENSEMBLE]


def mcnemar(a_succ: list[bool], b_succ: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (a vs baseline b).
    b_wins = a succeeds where b fails; c_wins = b succeeds where a fails."""
    a = np.asarray(a_succ, dtype=bool)
    b = np.asarray(b_succ, dtype=bool)
    n_ab = int(np.sum(a & ~b))  # a wins
    n_ba = int(np.sum(~a & b))  # baseline wins
    n = n_ab + n_ba
    if n == 0:
        return {"a_wins": 0, "b_wins": 0, "z": 0.0, "p_exact": 1.0}
    k = min(n_ab, n_ba)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    p_exact = min(1.0, 2.0 * tail)
    z = (abs(n_ab - n_ba) - 1) / math.sqrt(n)  # continuity-corrected normal approx
    return {"a_wins": n_ab, "b_wins": n_ba, "z": float(z), "p_exact": float(p_exact)}


def run_all(
    policy_ckpt: str,
    count: int,
    decay: float,
    out_dir: Path,
    device: str,
    panel_name: str,
    labels: list[str],
) -> dict:
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
    # one env, shared across systems (episodes reset deterministically by seed)
    policy = load_policy(policy_ckpt, device=device, use_ema=True)
    env = make_env(
        task_id=policy.meta.task_id,
        control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend,
        obs_mode="state",
        render_mode="rgb_array",
    )
    del policy  # evaluate_system reloads it per system

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for label in labels:
        cfg = load_config(str(REPO / f"configs/systems/{label}.yaml"))
        if label in TEMPORAL:
            cfg.setdefault("selection", {})["decay"] = decay
        print(
            f"[compare_temporal] evaluating {label} on {panel_name} ({count} ep) ...",
            flush=True,
        )
        results[label] = evaluate_system(
            system_cfg=cfg,
            eval_cfg=eval_cfg,
            policy_checkpoint=policy_ckpt,
            num_episodes=count,
            output_path=out_dir / f"eval_{label}.json",
            device=device,
            env=env,
            force=True,
        )
    env.close()
    return results


def build_report(results: dict[str, dict], count: int, panel_name: str) -> dict:
    rows = {}
    for label, r in results.items():
        ci = wilson_interval(r["success_count"], count)
        rows[label] = {
            "success_rate": r["success_rate"],
            "success_count": r["success_count"],
            "wilson_ci": list(ci),
            "mean_policy_latency_ms": 1000 * r["latency"]["mean_policy_s"],
            "mean_decision_latency_ms": 1000 * r["latency"]["mean_decision_s"],
            "mean_replans": float(np.mean([len(e["replans"]) for e in r["episodes"]])),
        }

    def paired(a_label: str, b_label: str) -> dict:
        a, b = results[a_label], results[b_label]
        return {
            "abs_diff": a["success_rate"] - b["success_rate"],
            "mcnemar": mcnemar(a["successes"], b["successes"]),
            "bootstrap": paired_bootstrap_diff(a["successes"], b["successes"], seed=0),
            "counts": paired_outcome_counts(a["successes"], b["successes"]),
            "policy_calls_ratio": rows[a_label]["mean_replans"]
            / max(1e-9, rows[b_label]["mean_replans"]),
        }

    # PRIMARY: each temporal variant present vs the receding-horizon baseline (budget-matched)
    vs_baseline = (
        {label: paired(label, BASELINE) for label in TEMPORAL if label in results}
        if BASELINE in results
        else {}
    )
    # AGGREGATION KNOB: ensembling variants vs replan-every-step-only (compute-matched)
    vs_control = (
        {label: paired(label, CONTROL) for label in ENSEMBLE if label in results}
        if CONTROL in results
        else {}
    )
    return {
        "panel": panel_name,
        "num_episodes": count,
        "baseline": BASELINE,
        "control": CONTROL,
        "systems": rows,
        "temporal_vs_baseline": vs_baseline,
        "ensemble_vs_control": vs_control,
    }


def format_report(rep: dict) -> str:
    L = [
        "=" * 84,
        f"A4 temporal ensembling — {rep['panel']} panel, n={rep['num_episodes']} paired episodes",
        "=" * 84,
        f"{'system':<24}{'success':>9}{'wilson 95% CI':>20}{'policy ms':>11}{'replans':>9}",
    ]
    L.append("-" * 73)
    for label, s in rep["systems"].items():
        ci = s["wilson_ci"]
        L.append(
            f"{label:<24}{s['success_rate']:>8.1%}{f'[{ci[0]:.1%},{ci[1]:.1%}]':>20}"
            f"{s['mean_policy_latency_ms']:>10.1f}{s['mean_replans']:>9.1f}"
        )

    def block(pairs: dict) -> None:
        for label, p in pairs.items():
            mc, b, c = p["mcnemar"], p["bootstrap"], p["counts"]
            sig = "*" if mc["p_exact"] < 0.05 else " "
            L.append(
                f"  {label:<22} diff {p['abs_diff']:+.1%}{sig}  "
                f"win/loss/tie {c['a_wins']}/{c['b_wins']}/{c['both_succeed'] + c['both_fail']}  "
                f"McNemar z={mc['z']:.2f} p={mc['p_exact']:.3f}  "
                f"boot95[{b['ci_low']:+.1%},{b['ci_high']:+.1%}]  "
                f"{p['policy_calls_ratio']:.1f}x calls"
            )

    if rep["temporal_vs_baseline"]:
        L.append("")
        L.append(
            f"PRIMARY — temporal vs {rep['baseline']} (budget-matched, env-paired; * = McNemar p<0.05):"
        )
        block(rep["temporal_vs_baseline"])
    if rep["ensemble_vs_control"]:
        L.append("")
        L.append(
            f"AGGREGATION KNOB — ensembling vs {rep['control']} (compute-matched, env-paired):"
        )
        block(rep["ensemble_vs_control"])
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy-checkpoint",
        default=str(REPO / "outputs/active_min/policy_seed_0/selected_policy.pt"),
    )
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--decay", type=float, default=0.1)
    ap.add_argument(
        "--panel",
        default="diagnostic",
        help="panel name (panels.py): diagnostic (development) or final_test (claim)",
    )
    ap.add_argument(
        "--systems",
        default=None,
        help="comma-separated system configs to run (default: baseline + all temporal)",
    )
    ap.add_argument("--out-dir", default=str(REPO / "outputs/active_min/temporal"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    labels = args.systems.split(",") if args.systems else [BASELINE, *TEMPORAL]
    results = run_all(
        args.policy_checkpoint,
        args.count,
        args.decay,
        Path(args.out_dir),
        args.device,
        args.panel,
        labels,
    )
    report = build_report(results, args.count, args.panel)
    print(format_report(report))
    out = Path(args.out_dir) / "comparison.json"
    save_json(report, out)
    print(f"\n[compare_temporal] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Selector baselines v1 — formal comparison driver.

Evaluates each selection rule over the same frozen Phase 0A policies and the
same K=16 candidate tensors on a NEW untouched final-test panel, then compares
them offline (paired). Reads Phase 0A frozen artifacts read-only; never reruns
or modifies Phase 0A.

  run     --system NAME --seed K    # evaluate one (system, seed); idempotent
  analyze                           # offline paired comparison + report

Systems: candidate_zero | full_chunk_medoid | early_weighted_medoid |
         coordinate_median_projection | largest_cluster_medoid |
         verifier_argmax | verifier_ensemble
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.panels import DEFAULT_PANELS, Panel  # noqa: E402
from actsemble.utils.repo import current_git_commit  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402

OUT = REPO / "outputs" / "selector_baselines_v1" / "final_v1"
SEEDS = [0, 1, 2, 3, 4]
K = 16
PANEL = Panel(name="selector_final_test", env_seed=12000, num_episodes=1200)
POLICY = "outputs/phase0a_v1/final_policy_runs/policy_seed_{k}/selected_policy.pt"
VERIF = "outputs/phase0a_v1/verifier_runs/verifier_seed_{k}/selected_verifier.pt"

SELECT_CFG = {
    "candidate_zero": {"type": "candidate_zero"},
    "full_chunk_medoid": {"type": "full_chunk_medoid"},
    "early_weighted_medoid": {"type": "early_weighted_medoid", "early_weight_decay": 0.25},
    "coordinate_median_projection": {"type": "coordinate_median_projection"},
    "largest_cluster_medoid": {"type": "largest_cluster_medoid"},
    "verifier_argmax": {"type": "highest_component_score"},
    "verifier_ensemble": {"type": "mean_component_score"},
}
SYSTEMS = list(SELECT_CFG)
CONSENSUS = ["full_chunk_medoid", "early_weighted_medoid",
             "coordinate_median_projection", "largest_cluster_medoid"]


def _components(system: str, seed: int) -> list[str]:
    if system == "verifier_argmax":
        return [str(REPO / VERIF.format(k=seed))]
    if system == "verifier_ensemble":
        return [str(REPO / VERIF.format(k=k)) for k in SEEDS]
    return []


def _assert_panel_disjoint():
    roots = {v["env_seed"] for v in DEFAULT_PANELS.values()} | {9100}  # +dev panel
    assert PANEL.env_seed not in roots, f"panel {PANEL.env_seed} collides"


def cmd_run(args):
    _assert_panel_disjoint()
    system, seed = args.system, int(args.seed)
    out = OUT / system
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"seed_{seed}.json"
    if dest.exists() and not args.force:
        print(f"[run] {system} seed{seed} already done; skipping")
        return 0
    eval_cfg = {"regime": "nominal", "panel": PANEL.to_dict(), "max_steps": 100,
                "perturbations": [], "video": {"max_success": 0, "max_failure": 0}}
    evaluate_system(
        system_cfg={"policy": {"num_candidates": 1}, "selection": SELECT_CFG[system]},
        eval_cfg=eval_cfg, policy_checkpoint=str(REPO / POLICY.format(k=seed)),
        component_checkpoints=_components(system, seed), output_path=dest,
        device=args.device or "cuda", env=None, num_candidates_override=K, force=args.force)
    r = load_json(dest)
    print(f"[run] {system} seed{seed}: {r['success_count']}/{r['num_episodes']} = {r['success_rate']:.1%}")
    return 0


def _t_crit(df):  # 95% two-sided
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(df, 1.96)


def _paired(a_succ, b_succ):
    """Seed-level + pooled per-episode paired stats for a vs b (per-seed lists)."""
    deltas = np.array([a.mean() - b.mean() for a, b in zip(a_succ, b_succ)])
    n = len(deltas); mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    ci = [mean - _t_crit(n - 1) * se, mean + _t_crit(n - 1) * se]
    # pooled McNemar over all paired episodes
    A = np.concatenate(a_succ); B = np.concatenate(b_succ)
    b01 = int(((A == 0) & (B == 1)).sum())   # b wins
    b10 = int(((A == 1) & (B == 0)).sum())   # a wins
    disc = b01 + b10
    z = (b10 - b01) / np.sqrt(disc) if disc else 0.0
    return {"per_seed_delta": [float(x) for x in deltas], "mean_delta": mean,
            "sd": sd, "ci95": ci,
            "sign_counts": {"a_gt_b": int((deltas > 0).sum()),
                            "lt": int((deltas < 0).sum()), "eq": int((deltas == 0).sum())},
            "pooled_episodes": int(len(A)), "a_wins": b10, "b_wins": b01,
            "mcnemar_z": float(z),
            "mcnemar_significant_0.05": bool(abs(z) > 1.96)}


def cmd_analyze(args):
    succ, rate = {}, {}
    missing = []
    for s in SYSTEMS:
        per_seed = []
        for k in SEEDS:
            p = OUT / s / f"seed_{k}.json"
            if not p.exists():
                missing.append(f"{s}/seed_{k}"); continue
            per_seed.append(np.array(load_json(p)["successes"], dtype=int))
        succ[s] = per_seed
        rate[s] = [float(x.mean()) for x in per_seed]
    if missing:
        print(f"[analyze] WARNING missing {len(missing)} runs: {missing[:6]}...")

    # per-system mean success across seeds
    summary = {s: {"per_seed_success": rate[s],
                   "mean_success": float(np.mean(rate[s])) if rate[s] else None,
                   "n_seeds": len(rate[s])} for s in SYSTEMS}

    # best consensus by mean success (diagnostic pick)
    best_consensus = max(CONSENSUS, key=lambda s: np.mean(rate[s]) if rate[s] else -1)

    contrasts = {}
    def add(a, b):
        if succ.get(a) and succ.get(b) and len(succ[a]) == len(succ[b]) and len(succ[a]) > 0:
            contrasts[f"{a}__vs__{b}"] = _paired(succ[a], succ[b])
    for a in ["verifier_argmax", "verifier_ensemble", best_consensus, *CONSENSUS]:
        add(a, "candidate_zero")
    add("verifier_argmax", best_consensus)
    add("verifier_ensemble", best_consensus)
    add("verifier_ensemble", "verifier_argmax")

    report = {
        "experiment": "selector_baselines_v1/final_v1",
        "panel": PANEL.to_dict(), "num_candidates": K, "seeds": SEEDS,
        "git_commit": current_git_commit(),
        "identity_note": "Init (replan 0) bitwise-identical; unbiased ~1e-2 cross-episode "
                         "GPU drift from replan 1 (does not bias mean paired deltas).",
        "best_consensus_by_mean": best_consensus,
        "per_system": summary, "contrasts": contrasts, "missing_runs": missing,
    }
    (OUT.parent / "final_v1").mkdir(parents=True, exist_ok=True)
    save_json(report, OUT / "results.json")
    _write_md(OUT / "comparison.md", report)
    print(f"[analyze] wrote {OUT/'results.json'} and comparison.md")
    for s in SYSTEMS:
        if rate[s]:
            print(f"  {s:30s} mean={np.mean(rate[s]):.1%}  seeds={rate[s]}")
    return 0


def _write_md(path, rep):
    L = ["# Selector baselines v1 — formal comparison", "",
         f"Panel `{rep['panel']}` (disjoint from all Phase 0A + dev panels), K={rep['num_candidates']}, "
         f"seeds {rep['seeds']}. {rep['identity_note']}", "",
         "## Mean success by selector (across seeds)", "",
         "| selector | mean success | per-seed |", "|---|---|---|"]
    for s, v in rep["per_system"].items():
        if v["mean_success"] is not None:
            L.append(f"| {s} | {v['mean_success']:.1%} | {[round(x,3) for x in v['per_seed_success']]} |")
    L += ["", f"Best consensus by mean: **{rep['best_consensus_by_mean']}**", "",
          "## Paired contrasts (a vs b)", "",
          "| contrast | mean Δ | 95% CI (seed-level) | signs +/-/= | McNemar z | sig? |",
          "|---|---|---|---|---|---|"]
    for name, c in rep["contrasts"].items():
        sc = c["sign_counts"]
        L.append(f"| {name.replace('__vs__',' vs ')} | {c['mean_delta']:+.2%} | "
                 f"[{c['ci95'][0]:+.2%}, {c['ci95'][1]:+.2%}] | "
                 f"{sc['a_gt_b']}/{sc['lt']}/{sc['eq']} | {c['mcnemar_z']:+.2f} | "
                 f"{'YES' if c['mcnemar_significant_0.05'] else 'no'} |")
    L += ["", "Seed-level (5 seeds) is the conservative replication unit; the pooled "
          "per-episode McNemar z is the higher-power view. Key question: does "
          "`verifier_argmax` / `verifier_ensemble` beat the best consensus rule?"]
    path.write_text("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_argument("stage", choices=["run", "analyze", "make-jobs"])
    p.add_argument("--system", default=None)
    p.add_argument("--seed", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.stage == "make-jobs":
        import json
        jobs = [{"script": "scripts/selector_final_v1.py",
                 "args": ["run", "--system", s, "--seed", str(k)],
                 "log": f"outputs/selector_baselines_v1/final_v1/logs/{s}_s{k}.log"}
                for s, k in itertools.product(SYSTEMS, SEEDS)]
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        Path("outputs/selector_baselines_v1/final_v1/jobs.json").write_text(json.dumps(jobs, indent=2))
        print(f"wrote {len(jobs)} jobs")
        return 0
    return {"run": cmd_run, "analyze": cmd_analyze}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())

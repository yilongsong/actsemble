#!/usr/bin/env python
"""Overnight rigorous run for the reference-faithful state policies.

Per policy family (diffusion / act / flow), end to end:

    1. train      fixed-budget, sim-free, with interval snapshots (checkpoint_every)
    2. screen     (re-)screen every snapshot on the screening panel (50 ep)     §5
    3. confirm    confirmation panel (200 ep) + lexicographic selection          §6
    4. final      500-episode standalone final-test on the SELECTED checkpoint  §13
    5. record     the SELECTED best epoch (+ screening curve, final success/CI)

Selection is by closed-loop SUCCESS (screening -> confirmation), never by
validation loss. Screening/confirmation/final panels are disjoint (panels.py).
The execution offset is derived from each policy's window alignment, so
diffusion-policy-aligned chunks execute the action for time t (not a past one).

Development-tier by default (single policy seed). For a reported claim, run >= 5
policy seeds and pair (docs/deferred_work.md).

Usage:
    python scripts/overnight_canonical.py --family all --device cuda
    python scripts/overnight_canonical.py --family diffusion --device cuda   # one GPU
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.config import load_config  # noqa: E402
from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.protocol.confirmation import confirm_and_select  # noqa: E402
from actsemble.protocol.screening import screen_all_snapshots, screening_history_path  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.training.train_act_policy import train_act_policy  # noqa: E402
from actsemble.training.train_diffusion_policy import train_diffusion_policy  # noqa: E402
from actsemble.training.train_flow_policy import train_flow_policy  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402

FAMILIES = {
    "diffusion": ("configs/policies/state_diffusion_canonical.yaml", train_diffusion_policy),
    "act": ("configs/policies/state_act_canonical.yaml", train_act_policy),
    "flow": ("configs/policies/state_flow_canonical.yaml", train_flow_policy),
}

# Standalone candidate-zero; build_system fills in the execution offset from meta.
STANDALONE = {"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"},
              "execution": {}}


def run_family(family: str, dataset: str, out_root: Path, device: str,
               final_n: int, max_steps: int, train_max_steps: int | None = None) -> dict:
    config_path, trainer = FAMILIES[family]
    cfg = load_config(str(REPO / config_path))
    run_dir = out_root / family
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[overnight:{family}] ===== TRAIN =====", flush=True)
    summary = trainer(policy_cfg=cfg, dataset_path=dataset, output_dir=run_dir,
                      device=device, max_steps=train_max_steps)
    spe = int(summary["steps_per_epoch"])
    print(f"[overnight:{family}] trained {summary['steps']} steps "
          f"({spe} steps/epoch); {len(list((run_dir / 'checkpoints').glob('step_*.pt')))} snapshots",
          flush=True)

    meta = DatasetReader(dataset).metadata
    env = make_env(task_id=meta.task_id, control_mode=meta.controller,
                   sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)
    try:
        print(f"[overnight:{family}] ===== SCREEN (50 ep) =====", flush=True)
        screen_all_snapshots(run_dir, make_panel("screening"), env=env, device=device,
                             max_steps=max_steps)
        print(f"[overnight:{family}] ===== CONFIRM + SELECT (200 ep) =====", flush=True)
        report = confirm_and_select(run_dir, make_panel("confirmation"), env=env,
                                    device=device, max_steps=max_steps, force=True)
        selected_step = int(report["selected_step"])

        print(f"[overnight:{family}] ===== FINAL TEST ({final_n} ep) =====", flush=True)
        fpanel = make_panel("final_test")
        final = evaluate_system(
            system_cfg=STANDALONE,
            eval_cfg={"name": "final_test", "regime": "nominal", "max_steps": max_steps,
                      "panel": {"name": fpanel.name, "env_seed": fpanel.env_seed,
                                "num_episodes": final_n},
                      "perturbations": []},
            policy_checkpoint=report["selected_policy_path"],
            num_episodes=final_n, output_path=run_dir / "final_test.json",
            device=device, env=env, force=True,
        )
    finally:
        env.close()

    ci = wilson_interval(final["success_count"], final_n)
    screening = load_json(screening_history_path(run_dir))
    record = {
        "family": family,
        "config": config_path,
        "selected_step": selected_step,
        "steps_per_epoch": spe,
        "selected_epoch": round(selected_step / spe, 2),
        "total_steps": int(summary["steps"]),
        "max_epochs": (cfg.get("training", {}) or {}).get("max_epochs"),
        "best_val_loss": summary.get("best_val_loss"),
        "final_success_rate": final["success_rate"],
        "final_success_count": final["success_count"],
        "final_wilson_ci": list(ci),
        "final_n": final_n,
        "policy_ms_per_query": 1000.0 * final["latency"]["mean_policy_s"],
        "selected_policy_path": report["selected_policy_path"],
        "screening_curve": [
            {"step": h["step"], "epoch": round(h["step"] / spe, 2),
             "success_rate": h["success_rate"]}
            for h in screening
        ],
    }
    save_json(record, out_root / f"overnight_{family}.json")
    print(f"[overnight:{family}] SELECTED step {selected_step} "
          f"(epoch {record['selected_epoch']}) | final {final['success_rate']:.1%} "
          f"CI [{ci[0]:.1%},{ci[1]:.1%}] | {record['policy_ms_per_query']:.1f} ms/query",
          flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="all", choices=[*FAMILIES, "all"])
    ap.add_argument("--dataset", default=str(REPO / "data/active_min/subset_0400.h5"))
    ap.add_argument("--out-root", default=str(REPO / "outputs/active_min/overnight"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--final-n", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=100, help="per-episode step cap")
    ap.add_argument("--train-max-steps", type=int, default=None,
                    help="override the config training budget (for quick end-to-end tests)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    families = list(FAMILIES) if args.family == "all" else [args.family]
    results, failures = {}, {}
    for fam in families:
        try:
            results[fam] = run_family(fam, args.dataset, out_root, args.device,
                                      args.final_n, args.max_steps, args.train_max_steps)
        except Exception as exc:  # keep going so one failure doesn't waste the night
            failures[fam] = repr(exc)
            print(f"[overnight:{fam}] FAILED: {exc}\n{traceback.format_exc()}", flush=True)
        # Rebuild the summary from the per-family files so that separate
        # --family invocations sharing this out-root MERGE instead of clobbering.
        merged = {}
        for p in sorted(out_root.glob("overnight_*.json")):
            if p.stem != "overnight_summary":
                merged[p.stem.replace("overnight_", "")] = load_json(p)
        save_json({"results": merged, "failures": failures},
                  out_root / "overnight_summary.json")

    print("\n" + "=" * 78)
    print(f"{'family':<12}{'sel.epoch':>10}{'sel.step':>10}{'final':>9}"
          f"{'wilson 95%':>18}{'ms/query':>10}")
    print("-" * 78)
    for fam, r in results.items():
        ci = r["final_wilson_ci"]
        print(f"{fam:<12}{r['selected_epoch']:>10}{r['selected_step']:>10}"
              f"{r['final_success_rate']:>8.1%}{f'[{ci[0]:.1%},{ci[1]:.1%}]':>18}"
              f"{r['policy_ms_per_query']:>9.1f}")
    for fam, err in failures.items():
        print(f"{fam:<12}  FAILED: {err}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

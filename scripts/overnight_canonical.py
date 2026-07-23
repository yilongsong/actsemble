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
from actsemble.protocol.screening import (  # noqa: E402
    make_screening_callback,
    screen_all_snapshots,
    screening_history_path,
)
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.training.factory import policy_trainer  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402

FAMILIES = {
    "diffusion": "configs/policies/state_diffusion_canonical.yaml",
    "act": "configs/policies/state_act_canonical.yaml",
    "flow": "configs/policies/state_flow_canonical.yaml",
}

# Standalone candidate-zero; build_system fills in the execution offset from meta.
STANDALONE = {
    "policy": {"num_candidates": 1},
    "selection": {"type": "candidate_zero"},
    "execution": {},
}


def run_family(
    family: str,
    dataset: str,
    out_root: Path,
    device: str,
    final_n: int,
    max_steps: int,
    train_max_steps: int | None = None,
    config_override: str | None = None,
    screen_videos: bool = True,
    checkpoint_every: int | None = None,
    train_epochs: int | None = None,
    snapshots: int | None = None,
) -> dict:
    config_path = config_override or FAMILIES[family]
    cfg = load_config(str(REPO / config_path))
    if train_epochs is not None:
        # Budget in EPOCHS, so the number of gradient steps scales with dataset
        # size automatically. A fixed STEP budget silently gives small datasets
        # many more passes than large ones (n=25 ran 2100 epochs at the same
        # 10500 steps that gave n=100 only ~525), which is exactly the confound
        # that made the v1-vs-v3 comparison uninterpretable.
        cfg.setdefault("training", {})["max_epochs"] = int(train_epochs)
        cfg["training"]["max_steps"] = None
    if snapshots is not None:
        # Same number of snapshots per cell => same curve resolution AND same
        # screening cost regardless of dataset size. steps_per_epoch mirrors
        # train_diffusion_policy: floor(0.9 * windows / batch_size), drop_last.
        _n = sum(len(e.state) for e in DatasetReader(dataset).episodes)
        _bs = int(cfg.get("training", {}).get("batch_size", 256))
        _vf = float(cfg.get("training", {}).get("val_fraction", 0.1))
        _spe = max(1, int(_n * (1.0 - _vf)) // _bs)
        _total = int(train_epochs) * _spe if train_epochs is not None else 10000
        checkpoint_every = max(1, _total // int(snapshots))
        print(f"[overnight] {_n} transitions, {_spe} steps/epoch, "
              f"{_total} total steps, checkpoint_every={checkpoint_every} "
              f"(~{snapshots} snapshots)", flush=True)
    if checkpoint_every is not None:
        cfg.setdefault("training", {})["checkpoint_every"] = int(checkpoint_every)
    trainer = policy_trainer(cfg)
    run_dir = out_root / family
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = DatasetReader(dataset).metadata
    env = make_env(
        task_id=meta.task_id,
        control_mode=meta.controller,
        sim_backend=meta.simulation_backend,
        obs_mode="state",
        # rgb_array so screening can record a few rollouts; frames are only
        # grabbed for the handful of episodes actually being saved.
        render_mode="rgb_array" if screen_videos else None,
        # MUST be passed: the task's registered horizon truncates the episode
        # regardless of the max_steps handed to the evaluator, which is only a
        # ceiling on the rollout loop. Without this, --max-steps 160 on
        # PickCube-v1 still ends at its registered 50 and every episode scores
        # 0. (No-op where the two already agree: PushT-v1 registers 100, which
        # is this script's default.)
        max_episode_steps=max_steps,
    )
    try:
        print(f"\n[overnight:{family}] ===== TRAIN (screening every snapshot) =====", flush=True)
        summary = trainer(
            policy_cfg=cfg,
            dataset_path=dataset,
            output_dir=run_dir,
            device=device,
            max_steps=train_max_steps,
            # Screen INSIDE training, at every snapshot: the closed-loop curve
            # (and its videos) appear as the run proceeds, so a broken setup
            # shows up at the first snapshot instead of after the whole budget.
            # Runs in the trainer's RNG guard, so it cannot perturb training.
            on_checkpoint=make_screening_callback(
                run_dir,
                make_panel("screening"),
                env=env,
                device=device,
                max_steps=max_steps,
                save_videos=screen_videos,
            ),
        )
        spe = int(summary["steps_per_epoch"])
        print(
            f"[overnight:{family}] trained {summary['steps']} steps "
            f"({spe} steps/epoch); {len(list((run_dir / 'checkpoints').glob('step_*.pt')))} snapshots",
            flush=True,
        )

        # Catch-up pass: a no-op when the hook already screened every snapshot,
        # and the resume path if the run was interrupted mid-training.
        print(f"[overnight:{family}] ===== SCREEN (catch-up) =====", flush=True)
        screen_all_snapshots(
            run_dir,
            make_panel("screening"),
            env=env,
            device=device,
            max_steps=max_steps,
            save_videos=screen_videos,
            skip_screened=True,
        )
        print(f"[overnight:{family}] ===== CONFIRM + SELECT (200 ep) =====", flush=True)
        report = confirm_and_select(
            run_dir,
            make_panel("confirmation"),
            env=env,
            device=device,
            max_steps=max_steps,
            force=True,
        )
        selected_step = int(report["selected_step"])

        print(f"[overnight:{family}] ===== FINAL TEST ({final_n} ep) =====", flush=True)
        fpanel = make_panel("final_test")
        final = evaluate_system(
            system_cfg=STANDALONE,
            eval_cfg={
                "name": "final_test",
                "regime": "nominal",
                "max_steps": max_steps,
                "panel": {
                    "name": fpanel.name,
                    "env_seed": fpanel.env_seed,
                    "num_episodes": final_n,
                },
                "perturbations": [],
            },
            policy_checkpoint=report["selected_policy_path"],
            num_episodes=final_n,
            output_path=run_dir / "final_test.json",
            device=device,
            env=env,
            force=True,
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
            {
                "step": h["step"],
                "epoch": round(h["step"] / spe, 2),
                "success_rate": h["success_rate"],
            }
            for h in screening
        ],
    }
    save_json(record, out_root / f"overnight_{family}.json")
    print(
        f"[overnight:{family}] SELECTED step {selected_step} "
        f"(epoch {record['selected_epoch']}) | final {final['success_rate']:.1%} "
        f"CI [{ci[0]:.1%},{ci[1]:.1%}] | {record['policy_ms_per_query']:.1f} ms/query",
        flush=True,
    )
    return record


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--family", default="all", choices=[*FAMILIES, "all"])
    ap.add_argument("--dataset", default=str(REPO / "data/active_min/subset_0400.h5"))
    ap.add_argument("--out-root", default=str(REPO / "outputs/active_min/overnight"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--final-n", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=100, help="per-episode step cap")
    ap.add_argument(
        "--train-max-steps",
        type=int,
        default=None,
        help="override the config training budget (for quick end-to-end tests)",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="override snapshot interval IN STEPS (config default 1000). Small "
             "datasets get few steps per epoch, so a fixed step interval is very "
             "coarse in epochs for them (n=10 at ~2 steps/epoch => one snapshot "
             "per ~500 epochs). Lower this for small-n cells to resolve the peak.",
    )
    ap.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="training budget IN EPOCHS (steps then scale with dataset size). "
             "Preferred over --train-max-steps for cross-dataset comparisons.",
    )
    ap.add_argument(
        "--snapshots",
        type=int,
        default=None,
        help="target NUMBER of snapshots across the run; checkpoint_every is "
             "derived from it so every cell gets the same curve resolution "
             "(and the same screening cost) regardless of dataset size.",
    )
    ap.add_argument(
        "--no-screen-videos",
        action="store_true",
        help="skip recording screening rollouts (videos are on by default: a few "
             "successes and failures per snapshot, so a flat curve can be SEEN)",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="override the family's config path (ablation variants; single --family only)",
    )
    args = ap.parse_args()
    if args.config is not None and args.family == "all":
        ap.error("--config requires a single --family")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    families = list(FAMILIES) if args.family == "all" else [args.family]
    results, failures = {}, {}
    for fam in families:
        try:
            results[fam] = run_family(
                fam,
                args.dataset,
                out_root,
                args.device,
                args.final_n,
                args.max_steps,
                args.train_max_steps,
                args.config,
                not args.no_screen_videos,
                args.checkpoint_every,
                args.train_epochs,
                args.snapshots,
            )
        except Exception as exc:  # keep going so one failure doesn't waste the night
            failures[fam] = repr(exc)
            print(
                f"[overnight:{fam}] FAILED: {exc}\n{traceback.format_exc()}", flush=True
            )
        # Rebuild the summary from the per-family files so that separate
        # --family invocations sharing this out-root MERGE instead of clobbering.
        merged = {}
        for p in sorted(out_root.glob("overnight_*.json")):
            if p.stem != "overnight_summary":
                merged[p.stem.replace("overnight_", "")] = load_json(p)
        save_json(
            {"results": merged, "failures": failures},
            out_root / "overnight_summary.json",
        )

    print("\n" + "=" * 78)
    print(
        f"{'family':<12}{'sel.epoch':>10}{'sel.step':>10}{'final':>9}"
        f"{'wilson 95%':>18}{'ms/query':>10}"
    )
    print("-" * 78)
    for fam, r in results.items():
        ci = r["final_wilson_ci"]
        print(
            f"{fam:<12}{r['selected_epoch']:>10}{r['selected_step']:>10}"
            f"{r['final_success_rate']:>8.1%}{f'[{ci[0]:.1%},{ci[1]:.1%}]':>18}"
            f"{r['policy_ms_per_query']:>9.1f}"
        )
    for fam, err in failures.items():
        print(f"{fam:<12}  FAILED: {err}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

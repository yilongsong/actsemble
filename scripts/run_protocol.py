#!/usr/bin/env python
"""Checkpoint-selection protocol orchestrator (docs/checkpoint_selection_protocol.md).

Stages (per experiment version, per policy-training seed):

    init            freeze the experiment spec into the experiment directory
    train-policy    fixed-budget training + interval snapshots + RNG-isolated
                    screening on the screening panel                     (§3-§5)
    screen-policy   (re-)screen saved snapshots out-of-process            (§5)
    confirm-policy  confirmation panel + lexicographic selection          (§6)
    train-verifier  fixed-budget verifier training + offline history      (§8)
    select-verifier offline-only lexicographic verifier selection         (§9)
    freeze          write the system freeze manifest                      (§10)
    integration     §12 implementation checklist on the integration panel
    final-test      gated paired evaluation on the final panel            (§13)
    aggregate       across-seed report                                    (§14)
    all             run every stage for one seed

Example:
    python scripts/run_protocol.py init --spec configs/protocol/smoke.yaml \
        --experiment-dir outputs/experiments/smoke_v1
    python scripts/run_protocol.py all --experiment-dir outputs/experiments/smoke_v1 \
        --policy-seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.evaluation.panels import load_panels
from actsemble.protocol.experiment import (
    init_experiment,
    load_experiment_spec,
    resolved_policy_config,
    resolved_verifier_config,
    seed_dir,
    verifier_seed_for,
)


def _device(args):
    if args.device:
        return args.device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_env_from_dataset(dataset_path, render_mode="rgb_array"):
    from actsemble.data.reader import DatasetReader
    from actsemble.sim.env_factory import make_env

    meta = DatasetReader(dataset_path).metadata
    return make_env(
        task_id=meta.task_id,
        control_mode=meta.controller,
        sim_backend=meta.simulation_backend,
        obs_mode="state",
        render_mode=render_mode,
    )


def stage_train_policy(spec, sdir, args):
    from actsemble.training.factory import policy_trainer, resolve_policy_family

    run_dir = sdir / "policy"
    policy_cfg = resolved_policy_config(spec, args.policy_seed)
    family = resolve_policy_family(policy_cfg)
    summary = policy_trainer(policy_cfg)(
        policy_cfg=policy_cfg,
        dataset_path=spec["dataset"],
        output_dir=run_dir,
        device=_device(args),
        resume=args.resume,
    )
    snapshots = list((run_dir / "checkpoints").glob("step_*.pt"))
    print(
        f"[protocol] {family} policy training done: {summary['steps']} steps, "
        f"{len(snapshots)} snapshots saved"
    )


def stage_screen_policy(spec, sdir, args):
    from actsemble.protocol.screening import screen_all_snapshots

    panels = load_panels(spec.get("panels"))
    env = _make_env_from_dataset(spec["dataset"], render_mode=None)
    try:
        screen_all_snapshots(
            sdir / "policy",
            panels["screening"],
            env=env,
            device=_device(args),
            max_steps=int(spec.get("episode_max_steps", 100)),
        )
    finally:
        env.close()


def stage_confirm_policy(spec, sdir, args):
    from actsemble.protocol.confirmation import confirm_and_select

    panels = load_panels(spec.get("panels"))
    rule = spec.get("confirmation_rule", {}) or {}
    env = _make_env_from_dataset(spec["dataset"], render_mode=None)
    try:
        confirm_and_select(
            sdir / "policy",
            panels["confirmation"],
            env=env,
            device=_device(args),
            max_steps=int(spec.get("episode_max_steps", 100)),
            top_k=int(rule.get("top_k", 5)),
            within_of_best=float(rule.get("within_of_best", 0.10)),
            force=args.force,
        )
    finally:
        env.close()


def stage_train_verifier(spec, sdir, args):
    from actsemble.training.train_component import train_component

    verifier_seed = verifier_seed_for(spec, args.policy_seed)
    summary = train_component(
        component_cfg=resolved_verifier_config(spec, verifier_seed),
        dataset_path=spec["dataset"],
        output_dir=sdir / "verifier",
        device=_device(args),
        resume=args.resume,
    )
    print(
        f"[protocol] verifier training done: {summary['steps']} steps, "
        f"{len(summary['snapshots'])} snapshots in offline history"
    )


def stage_select_verifier(spec, sdir, args):
    from actsemble.protocol.verifier_selection import select_verifier

    rule = spec.get("verifier_selection", {}) or {}
    select_verifier(
        sdir / "verifier",
        primary=rule.get("primary", "pairwise_ranking_accuracy"),
        secondary=rule.get("secondary", "balanced_accuracy"),
        force=args.force,
    )


def stage_freeze(spec, sdir, args):
    from actsemble.protocol.freeze import build_freeze_manifest, write_freeze

    manifest = build_freeze_manifest(
        policy_path=sdir / "policy" / "selected_policy.pt",
        verifier_path=sdir / "verifier" / "selected_verifier.pt",
        dataset_path=spec["dataset"],
        panels=load_panels(spec.get("panels")),
        num_candidates=int(spec.get("paired_num_candidates", 16)),
        device="cpu",
    )
    path = write_freeze(sdir, manifest, force=args.force)
    print(f"[protocol] systems frozen: {path}")
    print(f"[protocol]   policy   {manifest['policy']['checkpoint_hash'][:16]}")
    print(f"[protocol]   verifier {manifest['verifier']['checkpoint_hash'][:16]}")


def stage_integration(spec, sdir, args):
    from actsemble.protocol.freeze import load_freeze
    from actsemble.protocol.integration import run_integration

    panels = load_panels(spec.get("panels"))
    env = _make_env_from_dataset(spec["dataset"])
    try:
        report = run_integration(
            seed_dir=sdir,
            freeze=load_freeze(sdir),
            panel=panels["integration"],
            env=env,
            device=_device(args),
            max_steps=int(spec.get("episode_max_steps", 100)),
            force=args.force,
        )
    finally:
        env.close()
    if not report["passed"]:
        sys.exit(1)


def stage_final_test(spec, sdir, args):
    from actsemble.protocol.final_test import run_final_test

    regimes = {r["name"]: r for r in spec.get("final_regimes", [{"name": "nominal"}])}
    if args.regime not in regimes:
        raise SystemExit(
            f"Regime {args.regime!r} not declared in the frozen spec "
            f"(have {sorted(regimes)}); changing regimes after freezing "
            f"requires a new experiment version."
        )
    run_final_test(
        seed_dir=sdir,
        device=_device(args),
        regime=regimes[args.regime],
        force=args.force,
    )


def stage_aggregate(spec, experiment_dir, args):
    from actsemble.protocol.seed_report import aggregate_experiment, format_aggregate

    report = aggregate_experiment(experiment_dir, regime=args.regime)
    print(format_aggregate(report))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stage",
        choices=[
            "init",
            "train-policy",
            "screen-policy",
            "confirm-policy",
            "train-verifier",
            "select-verifier",
            "freeze",
            "integration",
            "final-test",
            "aggregate",
            "all",
        ],
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--spec", default=None, help="spec YAML (init only)")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--regime", default="nominal")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly allow overwriting completed artifacts (§17)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if args.stage == "init":
        if not args.spec:
            raise SystemExit("init requires --spec")
        init_experiment(args.spec, experiment_dir)
        print(f"[protocol] experiment initialized at {experiment_dir}")
        return 0

    spec = load_experiment_spec(experiment_dir)
    if args.policy_seed not in [int(s) for s in spec.get("policy_seeds", [0])]:
        raise SystemExit(
            f"policy seed {args.policy_seed} is not in the frozen spec's policy_seeds "
            f"{spec.get('policy_seeds')} — extend the spec in a NEW experiment version."
        )
    sdir = seed_dir(experiment_dir, args.policy_seed)

    if args.stage == "aggregate":
        stage_aggregate(spec, experiment_dir, args)
        return 0

    stages = {
        "train-policy": stage_train_policy,
        "screen-policy": stage_screen_policy,
        "confirm-policy": stage_confirm_policy,
        "train-verifier": stage_train_verifier,
        "select-verifier": stage_select_verifier,
        "freeze": stage_freeze,
        "integration": stage_integration,
        "final-test": stage_final_test,
    }
    if args.stage == "all":
        for name in [
            "train-policy",
            "screen-policy",
            "confirm-policy",
            "train-verifier",
            "select-verifier",
            "freeze",
            "integration",
            "final-test",
        ]:
            print(f"\n[protocol] ===== stage: {name} (seed {args.policy_seed}) =====")
            stages[name](spec, sdir, args)
        return 0
    stages[args.stage](spec, sdir, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

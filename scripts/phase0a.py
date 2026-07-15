#!/usr/bin/env python
"""Phase 0A driver: executes the frozen experiment spec end-to-end.

Thin orchestration over the existing protocol library (no method logic
here). Stages:

    init              freeze spec copy, verify panel disjointness
    master-report     Step 2: audit the master dataset
    make-subsets      Step 3: nested subsets + manifests
    policy-pipeline   Steps 4/7: train + screen + confirm + dev-eval (one N, one seed)
    verifier-pipeline Step 8: train + offline-select (one seed)
    learning-curve    Step 4/5 aggregation (results.json/csv, summary.md)
    select-primary    Step 5: apply the frozen n_demos rule
    finalize-policies Step 6/7: reuse decision + final_policy_runs layout
    pair-setup        Steps 9-11: freeze + integration for one seed pair
    pair-final        Step 12: gated final test for one seed pair
    identity-reports  Steps 10/13: candidate + action identity reports
    aggregate         Steps 13-14: paired_results.json/csv + comparison_report.md
    manifest          reproducibility_manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.config import load_config, merge_config
from actsemble.utils.serialization import load_json, save_json

# Env overrides exist ONLY for mechanical dry-runs of this driver in a
# scratch area; the real experiment always uses the defaults.
import os

OUT = Path(os.environ.get("ACTSEMBLE_PHASE0A_OUT", REPO / "outputs" / "phase0a_v1"))
SPEC_SOURCE = Path(os.environ.get(
    "ACTSEMBLE_PHASE0A_SPEC", REPO / "experiments" / "phase0a_v1" / "experiment_spec.yaml"
))
SPEC_FROZEN = OUT / "experiment_spec.yaml"
SUBSET_DIR = Path(os.environ.get("ACTSEMBLE_PHASE0A_DATA", REPO / "data" / "phase0a_v1"))


def spec() -> dict:
    return load_config(SPEC_FROZEN)


def subset_path(n: int) -> Path:
    return SUBSET_DIR / f"subset_{n:04d}.h5"


def panels_of(s: dict):
    from actsemble.evaluation.panels import load_panels

    return load_panels(s["panels"])


def _env_for(dataset: Path, render: str | None = None):
    from actsemble.data.reader import DatasetReader
    from actsemble.sim.env_factory import make_env

    meta = DatasetReader(dataset).metadata
    return make_env(
        task_id=meta.task_id, control_mode=meta.controller,
        sim_backend=meta.simulation_backend, obs_mode="state", render_mode=render,
    )


# ---------------------------------------------------------------- init ----
def cmd_init(args):
    OUT.mkdir(parents=True, exist_ok=True)
    if SPEC_FROZEN.exists():
        if load_config(SPEC_FROZEN) != load_config(SPEC_SOURCE):
            raise SystemExit("Frozen spec differs from source — new experiment version required")
        print("[init] spec already frozen (identical)")
    else:
        shutil.copy(SPEC_SOURCE, SPEC_FROZEN)
        print(f"[init] spec frozen -> {SPEC_FROZEN}")
    s = spec()
    from actsemble.evaluation.panels import assert_panels_disjoint
    from actsemble.protocol.freeze import demonstration_seeds

    assert_panels_disjoint(
        panels_of(s),
        extra_seed_sets={"demonstrations": demonstration_seeds(REPO / s["dataset"]["master_path"])},
    )
    print("[init] panels pairwise disjoint and disjoint from demonstration seeds")


# ------------------------------------------------------- master report ----
def cmd_master_report(args):
    import numpy as np

    from actsemble.data.reader import DatasetReader
    from actsemble.data.schema import FORBIDDEN_EPISODE_KEYS
    from actsemble.data.validation import validate_dataset, validate_success_only_provenance

    s = spec()
    path = REPO / s["dataset"]["master_path"]
    reader = DatasetReader(path)
    summary = validate_dataset(reader)
    prov = validate_success_only_provenance(path)
    lengths = [len(ep) for ep in reader.episodes]
    assert reader.dataset_hash == s["dataset"]["master_dataset_hash"], "master hash changed!"
    report = {
        "dataset_path": str(path),
        "dataset_hash": reader.dataset_hash,
        "matches_frozen_spec_hash": True,
        "num_successful_episodes": summary["num_episodes"],
        "num_transitions": summary["num_transitions"],
        "episode_length": {"min": int(np.min(lengths)), "max": int(np.max(lengths)),
                           "mean": float(np.mean(lengths)), "median": float(np.median(lengths))},
        "state_dimension": summary["state_dim"],
        "state_layout": reader.metadata.state_layout,
        "action_dimension": summary["action_dim"],
        "action_range": [summary["action_min"], summary["action_max"]],
        "controller": summary["controller"],
        "simulator": f"{reader.metadata.simulator} {reader.metadata.simulator_version}",
        "simulation_backend": summary["simulation_backend"],
        "state_projection_provenance": {
            "conversion_method": prov.get("conversion_method"),
            "source": prov.get("source", {}).get("h5_path"),
            "rejected_count": prov.get("rejected_count"),
            "conversion_failure_count": prov.get("conversion_failure_count"),
        },
        "action_clipping_provenance": prov.get("action_clipping"),
        "success_only_confirmed": True,
        "no_failed_trajectories": prov.get("rejected_count", 0) == 0
        and all(e.get("source_success") for e in prov["exported_episodes"]),
        "no_reward_or_success_model_inputs": True,  # schema-forbidden; validated
        "forbidden_fields_checked": list(FORBIDDEN_EPISODE_KEYS),
        "validator": "actsemble.data.validation.validate_dataset (full re-run, hash recomputed)",
    }
    save_json(report, OUT / "master_dataset_report.json")
    print(f"[master-report] OK: {summary['num_episodes']} episodes, "
          f"{summary['num_transitions']} transitions, hash verified")


# ------------------------------------------------------------- subsets ----
def cmd_make_subsets(args):
    from actsemble.data.reader import DatasetReader
    from actsemble.data.windows import split_episodes

    s = spec()
    counts = [int(n) for n in (args.counts or s["dataset"]["candidate_demo_counts"])]
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest_dir = OUT / "subset_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for n in counts:
        out = subset_path(n)
        if not out.exists():
            print(f"[subsets] preparing N={n}")
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "prepare_dataset.py"),
                 "--config", str(REPO / s["dataset"]["data_config"]),
                 "--output", str(out), "--subset-size", str(n)],
                check=True,
            )
        reader = DatasetReader(out)
        prov = load_json(Path(str(out) + ".provenance.json"))
        split = split_episodes(
            reader.episode_ids,
            val_fraction=float(s["dataset"]["val_fraction"]),
            seed=int(s["dataset"]["split_seed"]),
        )
        manifest = {
            "n_demos": n,
            "dataset_path": str(out),
            "master_dataset_hash": s["dataset"]["master_dataset_hash"],
            "source_bundle_sha256": reader.metadata.extra.get("source_bundle_sha256"),
            "subset_hash": reader.metadata.extra.get("subset_hash"),
            "dataset_hash": reader.dataset_hash,
            "subset_seed": reader.metadata.extra.get("subset_seed"),
            "episode_ids": reader.episode_ids,
            "source_episode_ids": [e["source_episode_id"] for e in prov["exported_episodes"]],
            "source_episode_seeds": [e["source_episode_seed"] for e in prov["exported_episodes"]],
            "num_episodes": len(reader.episodes),
            "num_transitions": reader.num_transitions,
            "train_episode_ids": split.train_episode_ids,
            "val_episode_ids": split.val_episode_ids,
            "split_hash": split.hash,
            "split_seed": int(s["dataset"]["split_seed"]),
            "val_fraction": float(s["dataset"]["val_fraction"]),
        }
        save_json(manifest, manifest_dir / f"subset_{n:04d}.json")
        print(f"[subsets] N={n}: {reader.num_transitions} transitions, "
              f"subset_hash {manifest['subset_hash'][:12]}, split {split.hash[:12]}")
    # nesting check: smaller ⊂ larger (by source episode ids)
    sets = {}
    for n in counts:
        m = load_json(manifest_dir / f"subset_{n:04d}.json")
        sets[n] = set(m["source_episode_ids"])
    ordered = sorted(sets)
    for a, b in zip(ordered, ordered[1:]):
        assert sets[a] <= sets[b], f"nesting violated: {a} not subset of {b}"
    print(f"[subsets] nesting verified across {ordered}")


# ----------------------------------------------------- policy pipeline ----
def cmd_policy_pipeline(args):
    from actsemble.protocol.confirmation import confirm_and_select
    from actsemble.protocol.screening import evaluate_checkpoint_on_panel, make_screening_callback
    from actsemble.training.train_diffusion_policy import train_diffusion_policy

    s = spec()
    n, seed = int(args.n), int(args.seed)
    out_dir = Path(args.out) if args.out else (
        OUT / "learning_curve" / f"ndemos_{n:04d}" / f"policy_seed_{seed}"
    )
    if (out_dir / "pipeline_summary.json").exists() and not args.force:
        print(f"[policy-pipeline] {out_dir} already complete; skipping")
        return
    dataset = subset_path(n)
    cfg = merge_config(load_config(REPO / s["policy_config"]), s.get("policy_config_overrides", {}))
    cfg = merge_config(cfg, {"training": {"seed": seed,
                                          "split_seed": int(s["dataset"]["split_seed"]),
                                          "val_fraction": float(s["dataset"]["val_fraction"])}})
    panels = panels_of(s)
    max_steps = int(s["episode_max_steps"])
    device = args.device or "cuda"
    env = _env_for(dataset)
    try:
        t0 = time.time()
        train_summary = train_diffusion_policy(
            policy_cfg=cfg, dataset_path=dataset, output_dir=out_dir, device=device,
            on_checkpoint=make_screening_callback(
                out_dir, panels["screening"], env=env, device=device, max_steps=max_steps),
        )
        t_train = time.time() - t0
        rule = s["confirmation_rule"]
        t0 = time.time()
        confirm = confirm_and_select(
            out_dir, panels["confirmation"], env=env, device=device, max_steps=max_steps,
            top_k=int(rule["top_k"]), within_of_best=float(rule["within_of_best"]),
            force=args.force,
        )
        t_confirm = time.time() - t0
        t0 = time.time()
        dev = evaluate_checkpoint_on_panel(
            out_dir / "selected_policy.pt", panels["dataset_size_development"],
            env=env, device=device, max_steps=max_steps,
        )
        t_dev = time.time() - t0
    finally:
        env.close()

    lengths = [r["num_steps"] for r in dev["episodes"]]
    screening_best = max(c["screening"]["success_rate"] for c in confirm["candidates"])
    winner = next(c for c in confirm["candidates"] if c["step"] == confirm["selected_step"])
    save_json(
        {
            "n_demos": n,
            "policy_seed": seed,
            "dataset_path": str(dataset),
            "dataset_hash": train_summary["dataset_hash"],
            "split_hash": train_summary["split_hash"],
            "normalization_hash": train_summary["normalization_hash"],
            "selected_step": confirm["selected_step"],
            "selected_policy_hash": confirm["selected_policy_hash"],
            "screening_best_rate": screening_best,
            "winner_screening_rate": winner["screening"]["success_rate"],
            "confirmation_rate": winner["confirmation"]["success_rate"],
            "confirmation_ci": winner["confirmation"]["wilson_ci"],
            "num_confirmation_candidates": len(confirm["candidates"]),
            "dev_success_rate": dev["success_rate"],
            "dev_success_count": dev["success_count"],
            "dev_wilson_ci": dev["wilson_ci"],
            "dev_num_episodes": dev["num_episodes"],
            "dev_mean_episode_length": sum(lengths) / len(lengths),
            "dev_timeouts": sum(r["timed_out"] for r in dev["episodes"]),
            "final_train_loss": train_summary["final_train_loss"],
            "best_val_diffusion_loss_diagnostic_only": train_summary["best_val_loss"],
            "train_wall_s": t_train, "confirm_wall_s": t_confirm, "dev_wall_s": t_dev,
        },
        out_dir / "pipeline_summary.json",
    )
    save_json(dev, out_dir / "dev_eval.json")
    print(f"[policy-pipeline] N={n} seed={seed}: selected step {confirm['selected_step']}, "
          f"confirmation {winner['confirmation']['success_rate']:.1%}, "
          f"dev {dev['success_rate']:.1%}")


# --------------------------------------------------- verifier pipeline ----
def cmd_verifier_pipeline(args):
    from actsemble.protocol.verifier_selection import select_verifier
    from actsemble.training.train_component import train_component

    s = spec()
    seed = int(args.seed)
    n = int(args.n)
    out_dir = OUT / "verifier_runs" / f"verifier_seed_{seed}"
    if (out_dir / "selected_verifier.pt").exists() and not args.force:
        print(f"[verifier-pipeline] {out_dir} already complete; skipping")
        return
    cfg = merge_config(load_config(REPO / s["verifier_config"]), s.get("verifier_config_overrides", {}))
    cfg = merge_config(cfg, {"training": {"seed": seed,
                                          "split_seed": int(s["dataset"]["split_seed"]),
                                          "val_fraction": float(s["dataset"]["val_fraction"])}})
    t0 = time.time()
    summary = train_component(
        component_cfg=cfg, dataset_path=subset_path(n), output_dir=out_dir,
        device=args.device or "cuda",
    )
    sel = select_verifier(
        out_dir,
        primary=s["verifier_selection"]["primary"],
        secondary=s["verifier_selection"]["secondary"],
        force=args.force,
    )
    save_json(
        {"verifier_seed": seed, "n_demos": n, "dataset_hash": summary["dataset_hash"],
         "split_hash": summary["split_hash"], "normalization_hash": summary["normalization_hash"],
         "selected_step": sel["selected_step"], "selected_verifier_hash": sel["selected_verifier_hash"],
         "winner_metrics": sel["winner_metrics"], "train_wall_s": time.time() - t0,
         "offline_eval": summary["offline_eval"]},
        out_dir / "pipeline_summary.json",
    )
    print(f"[verifier-pipeline] seed={seed}: selected step {sel['selected_step']}, "
          f"ranking acc {sel['winner_metrics']['pairwise_ranking_accuracy']:.4f}")


# ------------------------------------------------ learning curve aggregation
def cmd_learning_curve(args):
    import numpy as np

    from actsemble.protocol.seed_report import t_critical_95

    s = spec()
    lc_dir = OUT / "learning_curve"
    rows, per_n = [], {}
    for ndir in sorted(lc_dir.glob("ndemos_*")):
        n = int(ndir.name.split("_")[1])
        seeds = []
        for sdir in sorted(ndir.glob("policy_seed_*")):
            p = sdir / "pipeline_summary.json"
            if p.exists():
                seeds.append(load_json(p))
        if not seeds:
            continue
        rates = [x["dev_success_rate"] for x in seeds]
        mean, sd = float(np.mean(rates)), float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
        half = (t_critical_95(len(rates) - 1) * sd / np.sqrt(len(rates))) if len(rates) > 1 else None
        per_n[n] = {
            "n_demos": n,
            "seeds": [
                {"policy_seed": x["policy_seed"], "selected_step": x["selected_step"],
                 "screening_best": x["screening_best_rate"],
                 "confirmation": x["confirmation_rate"],
                 "dev_success": x["dev_success_rate"], "dev_ci": x["dev_wilson_ci"],
                 "mean_episode_length": x["dev_mean_episode_length"],
                 "val_diffusion_loss_diagnostic": x["best_val_diffusion_loss_diagnostic_only"],
                 "train_wall_s": x["train_wall_s"]}
                for x in seeds
            ],
            "mean_success": mean,
            "std_success": sd,
            "min_success": float(np.min(rates)),
            "max_success": float(np.max(rates)),
            "ci95_across_seeds": [mean - half, mean + half] if half is not None else None,
            "in_target_band": s["target_success_range"][0] <= mean <= s["target_success_range"][1],
        }
        for x in seeds:
            rows.append((n, x["policy_seed"], x["selected_step"], x["screening_best_rate"],
                         x["confirmation_rate"], x["dev_success_rate"],
                         x["best_val_diffusion_loss_diagnostic_only"], x["train_wall_s"]))
    results = {"target_success_range": s["target_success_range"], "per_n": per_n}
    save_json(results, lc_dir / "results.json")
    with open(lc_dir / "results.csv", "w") as f:
        f.write("n_demos,policy_seed,selected_step,screening_best,confirmation,"
                "dev_success,val_diffusion_loss,train_wall_s\n")
        for r in sorted(rows):
            f.write(",".join(str(v) for v in r) + "\n")
    lines = ["# Phase 0A learning curve (dataset_size_development panel, 200 episodes)\n",
             "| N | mean | sd | min | max | in 25-50% band | per-seed (step@success) |",
             "|---|------|----|----|-----|----------------|--------------------------|"]
    for n in sorted(per_n):
        e = per_n[n]
        per_seed = ", ".join(f"s{x['policy_seed']}:{x['selected_step']}@{x['dev_success']:.0%}"
                             for x in e["seeds"])
        lines.append(f"| {n} | {e['mean_success']:.1%} | {e['std_success']:.1%} | "
                     f"{e['min_success']:.0%} | {e['max_success']:.0%} | "
                     f"{'YES' if e['in_target_band'] else 'no'} | {per_seed} |")
    (lc_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_select_primary(args):
    s = spec()
    results = load_json(OUT / "learning_curve" / "results.json")
    lo, hi = s["target_success_range"]
    in_band = [int(n) for n, e in results["per_n"].items() if e["in_target_band"]]
    selected = max(in_band) if in_band else None
    record = {
        "rule": s["n_demos_primary_rule"],
        "target_success_range": [lo, hi],
        "full_learning_curve": results["per_n"],
        "counts_in_band": sorted(in_band),
        "n_demos_primary": selected,
        "selected_subset_hash": (
            load_json(OUT / "subset_manifests" / f"subset_{selected:04d}.json")["subset_hash"]
            if selected else None
        ),
        "intermediate_counts_added": args.intermediates or [],
        "reason": (
            f"largest N with mean dev success in [{lo:.0%},{hi:.0%}]: "
            + ", ".join(f"N={n}: {results['per_n'][str(n)]['mean_success']:.1%}"
                        for n in sorted((int(k) for k in results["per_n"]), key=int))
            if selected else "no candidate count in band — intermediate counts required"
        ),
    }
    save_json(record, OUT / "primary_dataset_selection.json")
    if selected is None:
        print("[select-primary] NO count in target band — add intermediates")
        sys.exit(1)
    print(f"[select-primary] n_demos_primary = {selected}")


# --------------------------------------------------------- stage B glue ----
def cmd_finalize_policies(args):
    """Reuse calibration runs at n_primary where valid; lay out final_policy_runs."""
    s = spec()
    n = load_json(OUT / "primary_dataset_selection.json")["n_demos_primary"]
    final_dir = OUT / "final_policy_runs"
    final_dir.mkdir(parents=True, exist_ok=True)
    decisions = []
    for seed in [int(x) for x in s["final_policy_seeds"]]:
        dst = final_dir / f"policy_seed_{seed}"
        calib = OUT / "learning_curve" / f"ndemos_{n:04d}" / f"policy_seed_{seed}"
        if dst.exists():
            decisions.append({"policy_seed": seed, "action": "already_present"})
            continue
        if calib.exists() and (calib / "selected_policy.pt").exists():
            # Calibration used the identical frozen spec, code commit, subset,
            # split, budget, panels, and protocol -> reuse (Step 6).
            dst.symlink_to(calib.resolve(), target_is_directory=True)
            decisions.append({"policy_seed": seed, "action": "reused_calibration_run",
                              "source": str(calib)})
        else:
            decisions.append({"policy_seed": seed, "action": "requires_training",
                              "expected_dir": str(calib)})
    save_json({"n_demos_primary": n, "decisions": decisions},
              final_dir / "reuse_decision.json")
    for d in decisions:
        print(f"[finalize] seed {d['policy_seed']}: {d['action']}")


def cmd_pair_setup(args):
    from actsemble.protocol.freeze import build_freeze_manifest, load_freeze, write_freeze
    from actsemble.protocol.integration import run_integration

    s = spec()
    seed = int(args.seed)
    n = load_json(OUT / "primary_dataset_selection.json")["n_demos_primary"]
    pair_dir = OUT / "pairs" / f"pair_{seed}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    if not (pair_dir / "freeze.json").exists() or args.force:
        manifest = build_freeze_manifest(
            policy_path=OUT / "final_policy_runs" / f"policy_seed_{seed}" / "selected_policy.pt",
            verifier_path=OUT / "verifier_runs" / f"verifier_seed_{seed}" / "selected_verifier.pt",
            dataset_path=subset_path(n),
            panels=panels_of(s),
            num_candidates=int(s["num_candidates"]),
            device="cpu",
        )
        write_freeze(pair_dir, manifest, force=args.force)
    env = _env_for(subset_path(n), render="rgb_array")
    try:
        report = run_integration(
            seed_dir=pair_dir, freeze=load_freeze(pair_dir),
            panel=panels_of(s)["integration"], env=env,
            device=args.device or "cuda", max_steps=int(s["episode_max_steps"]),
            force=args.force,
        )
    finally:
        env.close()
    if not report["passed"]:
        sys.exit(1)


def cmd_pair_final(args):
    from actsemble.protocol.final_test import run_final_test

    s = spec()
    seed = int(args.seed)
    pair_dir = OUT / "pairs" / f"pair_{seed}"
    out = run_final_test(
        seed_dir=pair_dir, device=args.device or "cuda",
        regime=s["final_regime"], force=args.force,
    )
    # Mirror into the required layout.
    mirror = {"standalone": "standalone", "control": "candidate_zero_control",
              "actsemble": "actsemble"}
    src_dir = Path(out["output_dir"])
    for src_name, dst_name in mirror.items():
        dst = OUT / "final_test" / f"policy_seed_{seed}" / dst_name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_dir / f"eval_{src_name}.json", dst / "result.json")
    shutil.copy(src_dir / "comparison.json",
                OUT / "final_test" / f"policy_seed_{seed}" / "comparison.json")
    if (src_dir / "videos").exists():
        shutil.copytree(src_dir / "videos",
                        OUT / "final_test" / f"policy_seed_{seed}" / "videos",
                        dirs_exist_ok=True)


# -------------------------------------------------------------- manifest ----
def cmd_manifest(args):
    import platform

    import torch

    import gymnasium
    import mani_skill
    import sapien

    from actsemble.evaluation.panels import panel_episodes
    from actsemble.utils.repo import current_git_commit

    s = spec()
    n_primary = None
    sel_path = OUT / "primary_dataset_selection.json"
    if sel_path.exists():
        n_primary = load_json(sel_path)["n_demos_primary"]
    panels = panels_of(s)
    seed_banks = {
        name: {
            "root": p.env_seed,
            "num_episodes": p.num_episodes,
            "environment_seeds": [e.env_seed for e in panel_episodes(p)],
            "policy_sampling_seeds": [e.policy_sampling_seed for e in panel_episodes(p)],
        }
        for name, p in panels.items()
    }
    policy_hashes, verifier_hashes = {}, {}
    for seed in [int(x) for x in s["final_policy_seeds"]]:
        pj = OUT / "pairs" / f"pair_{seed}" / "freeze.json"
        if pj.exists():
            fz = load_json(pj)
            policy_hashes[f"policy_seed_{seed}"] = fz["policy"]["checkpoint_hash"]
            verifier_hashes[f"verifier_seed_{seed}"] = fz["verifier"]["checkpoint_hash"]
    uncommitted = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                 capture_output=True, text=True).stdout.strip()
    manifest = {
        "experiment_version": s["experiment_version"],
        "git_commit": current_git_commit(REPO),
        "uncommitted_changes": uncommitted.splitlines() if uncommitted else [],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mani_skill_version": mani_skill.__version__,
        "sapien_version": sapien.__version__,
        "gymnasium_version": gymnasium.__version__,
        "task": s["task"],
        "master_dataset_hash": s["dataset"]["master_dataset_hash"],
        "n_demos_primary": n_primary,
        "selected_subset_hash": (
            load_json(sel_path)["selected_subset_hash"] if sel_path.exists() else None
        ),
        "split_seed": s["dataset"]["split_seed"],
        "subset_seed": s["dataset"]["subset_seed"],
        "policy_training_seeds": s["final_policy_seeds"],
        "calibration_policy_seeds": s["calibration_policy_seeds"],
        "verifier_training_seeds": s["verifier_seeds"],
        "negative_generation_seed": 1234,
        "policy_checkpoint_hashes": policy_hashes,
        "verifier_checkpoint_hashes": verifier_hashes,
        "num_candidates": s["num_candidates"],
        "inference": s["inference"],
        "systems": s["systems"],
        "seed_banks": seed_banks,
        "exact_commands": [
            "python scripts/phase0a.py init",
            "python scripts/phase0a.py master-report",
            "python scripts/phase0a.py make-subsets",
            "python scripts/phase0a_runner.py outputs/phase0a_v1/learning_curve_jobs.json 4 2",
            "python scripts/phase0a.py learning-curve",
            "python scripts/phase0a.py select-primary",
            "python scripts/phase0a.py finalize-policies",
            "python scripts/phase0a_runner.py outputs/phase0a_v1/stageb_jobs.json 4 2",
            "python scripts/phase0a.py pair-setup --seed {0..4}",
            "python scripts/phase0a.py pair-final --seed {0..4}",
            "python scripts/phase0a.py identity-reports",
            "python scripts/phase0a.py aggregate",
            "python scripts/phase0a.py manifest",
        ],
    }
    save_json(manifest, OUT / "reproducibility_manifest.json")
    print(f"[manifest] written; git {manifest['git_commit'][:12]}, "
          f"{len(manifest['uncommitted_changes'])} uncommitted paths")


# ----------------------------------------------------- identity reports ----
def _pair_results(seed: int, where: str) -> dict:
    base = OUT / "pairs" / f"pair_{seed}" / where
    return {name: load_json(base / f"eval_{name}.json")
            for name in ("standalone", "control", "actsemble")}


def cmd_identity_reports(args):
    s = spec()
    seeds = [int(x) for x in s["final_policy_seeds"]]
    cand_report, act_report = {"pairs": {}}, {"pairs": {}}
    for seed in seeds:
        for where, tag in (("integration", "integration"), ("final_test/nominal", "final")):
            try:
                r = _pair_results(seed, where)
            except FileNotFoundError:
                continue
            sa, co, ac = r["standalone"], r["control"], r["actsemble"]
            n = len(sa["episodes"])
            # candidate identity
            full_sc = sum(
                ea["candidate_hashes"] == eb["candidate_hashes"]
                for ea, eb in zip(sa["episodes"], co["episodes"])
            )
            prefix_ok = 0
            for eb, ec in zip(co["episodes"], ac["episodes"]):
                hb, hc = eb["candidate_hashes"], ec["candidate_hashes"]
                sb, sc_ = eb["selected_indices"], ec["selected_indices"]
                shared = min(len(hb), len(hc))
                div = next((k for k in range(shared) if sb[k] != sc_[k]), shared - 1)
                m = min(shared, div + 1)
                prefix_ok += hb[:m] == hc[:m]
            cand_report["pairs"].setdefault(f"pair_{seed}", {})[tag] = {
                "episodes": n,
                "standalone_control_full_hash_identity": f"{full_sc}/{n}",
                "control_actsemble_prefix_identity": f"{prefix_ok}/{n}",
                "all_identical_where_required": full_sc == n and prefix_ok == n,
            }
            # action identity (standalone vs control)
            same_actions = sum(
                ea["action_digest"] == eb["action_digest"]
                for ea, eb in zip(sa["episodes"], co["episodes"])
            )
            same_outcome = sum(
                ea["success_once"] == eb["success_once"]
                for ea, eb in zip(sa["episodes"], co["episodes"])
            )
            same_length = sum(
                ea["num_steps"] == eb["num_steps"]
                for ea, eb in zip(sa["episodes"], co["episodes"])
            )
            total_replans = sum(len(e["candidate_hashes"]) for e in sa["episodes"])
            same_replans = sum(
                sum(x == y for x, y in zip(ea["candidate_hashes"], eb["candidate_hashes"]))
                for ea, eb in zip(sa["episodes"], co["episodes"])
            )
            act_report["pairs"].setdefault(f"pair_{seed}", {})[tag] = {
                "episodes": n,
                "identical_executed_action_sequences": f"{same_actions}/{n}",
                "identical_episode_lengths": f"{same_length}/{n}",
                "identical_success_outcomes": f"{same_outcome}/{n}",
                "replans_with_identical_candidate_tensors": f"{same_replans}/{total_replans}",
                "note": ("standalone and control both execute candidate zero of the "
                         "same K-tensor; digests cover the full executed action sequence"),
            }
    for rep in (cand_report, act_report):
        rep["all_pairs_pass"] = all(
            v["all_identical_where_required"] if "all_identical_where_required" in v
            else v["identical_executed_action_sequences"].split("/")[0]
            == v["identical_executed_action_sequences"].split("/")[1]
            for pair in rep["pairs"].values() for v in pair.values()
        )
    (OUT / "integration").mkdir(parents=True, exist_ok=True)
    save_json(cand_report, OUT / "integration" / "candidate_identity_report.json")
    save_json(act_report, OUT / "integration" / "action_identity_report.json")
    print(f"[identity] candidate identity pass: {cand_report['all_pairs_pass']}")
    print(f"[identity] action identity pass: {act_report['all_pairs_pass']}")


# ------------------------------------------------------------ aggregate ----
def cmd_aggregate(args):
    import numpy as np

    from actsemble.evaluation.metrics import paired_bootstrap_diff, paired_outcome_counts
    from actsemble.protocol.seed_report import t_critical_95

    s = spec()
    seeds = [int(x) for x in s["final_policy_seeds"]]
    per_pair, csv_rows = [], []
    for seed in seeds:
        r = _pair_results(seed, "final_test/nominal")
        sa, co, ac = r["standalone"], r["control"], r["actsemble"]
        d_stand = ac["success_rate"] - sa["success_rate"]
        d_ctrl = ac["success_rate"] - co["success_rate"]
        counts = paired_outcome_counts(ac["successes"], sa["successes"])
        boot = paired_bootstrap_diff(ac["successes"], sa["successes"], seed=0)
        per_pair.append({
            "pair": seed,
            "standalone_success": sa["success_rate"],
            "control_success": co["success_rate"],
            "actsemble_success": ac["success_rate"],
            "delta_vs_standalone": d_stand,
            "delta_vs_control": d_ctrl,
            "paired_win": counts["a_wins"], "paired_loss": counts["b_wins"],
            "both_succeed": counts["both_succeed"], "both_fail": counts["both_fail"],
            "bootstrap_ci_vs_standalone": [boot["ci_low"], boot["ci_high"]],
            "candidate_change_rate": ac["selection_change_rate"],
            "fallback_rate": ac["fallback_rate"],
            "policy_latency_ms": ac["latency"]["mean_policy_s"] * 1000,
            "verifier_latency_ms": ac["latency"]["mean_component_s"] * 1000,
            "standalone_ci": sa["confidence_interval"],
            "actsemble_ci": ac["confidence_interval"],
        })
        csv_rows.append([seed, sa["success_rate"], co["success_rate"], ac["success_rate"],
                         d_stand, d_ctrl, counts["a_wins"], counts["b_wins"],
                         counts["both_succeed"], counts["both_fail"],
                         ac["selection_change_rate"], ac["fallback_rate"]])

    def across(key):
        vals = np.array([p[key] for p in per_pair])
        m, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        half = t_critical_95(len(vals) - 1) * sd / np.sqrt(len(vals)) if len(vals) > 1 else None
        return {"per_pair": vals.tolist(), "mean": m, "std": sd, "median": float(np.median(vals)),
                "ci95_across_pairs": [m - half, m + half] if half else None,
                "positive": int((vals > 0).sum()), "zero": int((vals == 0).sum()),
                "negative": int((vals < 0).sum())}

    result = {
        "primary_replication_unit": "policy/verifier training seed pair",
        "num_pairs": len(per_pair),
        "episodes_per_pair": per_pair and _pair_results(seeds[0], "final_test/nominal")["standalone"]["num_episodes"],
        "per_pair": per_pair,
        "across_pairs": {
            "standalone_success": across("standalone_success"),
            "actsemble_success": across("actsemble_success"),
            "delta_vs_standalone": across("delta_vs_standalone"),
            "delta_vs_control": across("delta_vs_control"),
            "candidate_change_rate": across("candidate_change_rate"),
        },
        "note": ("Seed pairs are the replication unit; the pooled episodes are NOT "
                 "independent training replications. Episode-level bootstrap CIs are "
                 "secondary, conditional on the trained checkpoints."),
    }
    ana = OUT / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    save_json(result, ana / "paired_results.json")
    with open(ana / "paired_results.csv", "w") as f:
        f.write("pair,standalone,control,actsemble,delta_vs_standalone,delta_vs_control,"
                "win,loss,both_succeed,both_fail,candidate_change_rate,fallback_rate\n")
        for row in csv_rows:
            f.write(",".join(str(v) for v in row) + "\n")
    d = result["across_pairs"]["delta_vs_standalone"]
    print(f"[aggregate] Actsemble - standalone across {len(per_pair)} pairs: "
          f"mean {d['mean']:+.2%} (sd {d['std']:.2%}), CI {d['ci95_across_pairs']}, "
          f"+/0/-: {d['positive']}/{d['zero']}/{d['negative']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--counts", type=int, nargs="*", default=None)
    parser.add_argument("--intermediates", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stages = {
        "init": cmd_init,
        "master-report": cmd_master_report,
        "make-subsets": cmd_make_subsets,
        "policy-pipeline": cmd_policy_pipeline,
        "verifier-pipeline": cmd_verifier_pipeline,
        "learning-curve": cmd_learning_curve,
        "select-primary": cmd_select_primary,
        "finalize-policies": cmd_finalize_policies,
        "pair-setup": cmd_pair_setup,
        "pair-final": cmd_pair_final,
        "identity-reports": cmd_identity_reports,
        "aggregate": cmd_aggregate,
        "manifest": cmd_manifest,
    }
    stages[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

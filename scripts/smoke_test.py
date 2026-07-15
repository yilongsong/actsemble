#!/usr/bin/env python
"""Actsemble end-to-end smoke test.

Verifies correctness of the complete pipeline, NOT task success:
dependencies -> env -> demos -> dataset -> windows -> policy training ->
sampling -> component training -> three systems -> rollouts -> metrics ->
videos -> hash/identity/determinism checks. Exits nonzero on any failure.

    python scripts/smoke_test.py [--workdir outputs/smoke] [--keep]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok)))
    mark = "PASS" if ok else "FAIL"
    print(f"[smoke] {mark:4s} | {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"[smoke] FAILED at: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=None, help="working directory (default: temp)")
    parser.add_argument("--keep", action="store_true", help="keep workdir on success")
    parser.add_argument("--episodes", type=int, default=8, help="source demos to convert")
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--eval-episodes", type=int, default=2)
    args = parser.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="actsemble_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] workdir: {workdir}")

    # 1. Dependencies -------------------------------------------------------
    try:
        import gymnasium
        import h5py
        import mani_skill
        import torch
        import yaml  # noqa: F401

        check("dependencies import", True,
              f"mani_skill {mani_skill.__version__}, torch {torch.__version__}")
    except Exception as exc:
        print(traceback.format_exc())
        check("dependencies import", False, str(exc))

    from actsemble.config import load_config

    repo = Path(__file__).resolve().parents[1]
    data_cfg = load_config(repo / "configs" / "data" / "smoke.yaml")
    policy_cfg = load_config(repo / "configs" / "policies" / "state_diffusion.yaml")
    component_cfg = load_config(repo / "configs" / "components" / "action_chunk_compatibility.yaml")
    # shrink for speed; correctness only
    policy_cfg["model"]["channels"] = [32, 64, 128]
    policy_cfg["training"].update({"batch_size": 64, "eval_every": args.train_steps})
    component_cfg["training"].update({"batch_size": 64, "eval_every": args.train_steps})

    # 2. Environment instantiation ------------------------------------------
    from actsemble.sim.env_factory import EnvironmentMismatchError, env_contract, make_env, verify_env_matches

    env = make_env(
        task_id=data_cfg["task_id"],
        control_mode=data_cfg["control_mode"],
        sim_backend=data_cfg["sim_backend"],
        obs_mode="state",
        render_mode="rgb_array",
    )
    contract = env_contract(env)
    check("ManiSkill task instantiation", True,
          f"{contract['task_id']} / {contract['controller']} / {contract['simulation_backend']}")
    try:
        verify_env_matches(env, {"controller": "definitely_wrong"}, what="smoke")
        check("environment mismatch detection", False, "mismatch not detected")
    except EnvironmentMismatchError:
        check("environment mismatch detection", True)

    # 3. Demos -> dataset ----------------------------------------------------
    import json

    import numpy as np

    sys.path.insert(0, str(repo / "scripts"))
    from prepare_dataset import default_bundle_path

    from actsemble.data.reader import DatasetReader
    from actsemble.data.schema import DatasetMetadata
    from actsemble.data.validation import validate_dataset, validate_success_only_provenance
    from actsemble.data.writer import write_dataset, write_private_provenance
    from actsemble.sim.demonstration_source import (
        ManiSkillTrajectorySource,
        convert_demonstrations,
        probe_state_layout,
    )

    bundle = default_bundle_path(
        data_cfg["task_id"], data_cfg["control_mode"], data_cfg["sim_backend"]
    )
    source = ManiSkillTrajectorySource(bundle, max_episodes=args.episodes)
    episodes, provenance = convert_demonstrations(source, env)
    check("demonstration conversion", len(episodes) >= 2,
          f"{len(episodes)} successful episodes")

    layout_env = make_env(
        task_id=data_cfg["task_id"], control_mode=data_cfg["control_mode"],
        sim_backend=data_cfg["sim_backend"], obs_mode="state_dict", render_mode=None,
    )
    layout = probe_state_layout(layout_env)
    layout_env.close()

    dataset_path = workdir / "smoke.h5"
    metadata = DatasetMetadata(
        simulator="ManiSkill3", simulator_version=contract["simulator_version"],
        task_id=data_cfg["task_id"], robot=contract["robot"], observation_mode="state",
        state_dimension=episodes[0].state.shape[1], state_layout=json.dumps(layout),
        controller=contract["controller"], action_dimension=episodes[0].action.shape[1],
        action_definition=json.dumps({
            "semantics": "normalized EE delta pose", "frame": "root aligned", "units": "[-1,1]",
            "bounds": [contract["action_low"].tolist(), contract["action_high"].tolist()],
            "scaling": "controller", "clipping_rules": "clipped at export",
        }),
        control_frequency=contract["control_frequency"],
        simulation_backend=contract["simulation_backend"],
        source_dataset=str(bundle), generation_or_replay_seed=0,
    )
    dataset_hash = write_dataset(dataset_path, episodes, metadata)
    write_private_provenance(dataset_path, provenance)

    # 4. Validation ----------------------------------------------------------
    reader = DatasetReader(dataset_path)
    summary = validate_dataset(reader)
    validate_success_only_provenance(dataset_path)
    check("dataset validation", True,
          f"{summary['num_episodes']} eps, {summary['num_transitions']} transitions")
    check("dataset hash recomputation", reader.dataset_hash == dataset_hash)

    # 5. Windows --------------------------------------------------------------
    from actsemble.data.normalization import Normalizer, compute_stats
    from actsemble.data.torch_dataset import DiffusionWindowDataset

    ds = DiffusionWindowDataset(
        reader.episodes, Normalizer(compute_stats(reader.episodes)),
        obs_horizon=2, prediction_horizon=16,
    )
    item = ds[0]
    check("window construction",
          item["obs_history"].shape == (2, reader.state_dim)
          and item["action_chunk"].shape == (16, reader.action_dim))

    # 6. Policy training -------------------------------------------------------
    from actsemble.training.train_diffusion_policy import train_diffusion_policy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_out = train_diffusion_policy(
        policy_cfg=policy_cfg, dataset_path=dataset_path,
        output_dir=workdir / "policy", max_steps=args.train_steps, device=device,
    )
    check("policy training", policy_out["steps"] == args.train_steps,
          f"final loss {policy_out['final_train_loss']:.4f}")

    # 7. Sampling ----------------------------------------------------------------
    from actsemble.policies.diffusion.policy import DiffusionPolicy

    policy = DiffusionPolicy.from_checkpoint(
        policy_out["checkpoints"]["best_ema"], device=device, use_ema=True
    )
    obs_hist = reader.episodes[0].state[:2]
    chunks = policy.sample_action_chunks(obs_hist, num_samples=4,
                                         generator=policy.new_generator(0))
    low = torch.tensor(policy.meta.action_low, device=chunks.device)
    high = torch.tensor(policy.meta.action_high, device=chunks.device)
    check("action-chunk sampling",
          chunks.shape == (4, 16, reader.action_dim)
          and torch.isfinite(chunks).all()
          and (chunks >= low - 1e-6).all() and (chunks <= high + 1e-6).all())

    chunks2 = policy.sample_action_chunks(obs_hist, num_samples=4,
                                          generator=policy.new_generator(0))
    check("deterministic candidate sampling", torch.equal(chunks, chunks2))

    # 8. Component training -------------------------------------------------------
    from actsemble.training.train_component import train_component

    comp_out = train_component(
        component_cfg=component_cfg, dataset_path=dataset_path,
        output_dir=workdir / "component", max_steps=args.train_steps, device=device,
    )
    check("component training", comp_out["steps"] == args.train_steps,
          f"ranking acc {comp_out['offline_eval']['pairwise_ranking_accuracy']:.3f}")

    # 9. Three systems ---------------------------------------------------------------
    from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility
    from actsemble.systems.factory import build_system

    component = ActionChunkCompatibility.from_checkpoint(
        comp_out["checkpoints"]["best"], device=device
    )
    system_cfgs = {
        "standalone": load_config(repo / "configs" / "systems" / "standalone_diffusion.yaml"),
        "control": load_config(repo / "configs" / "systems" / "multisample_control.yaml"),
        "actsemble": load_config(repo / "configs" / "systems" / "compatibility_reranking.yaml"),
    }
    systems = {
        "standalone": build_system(system_cfgs["standalone"], policy, []),
        "control": build_system(system_cfgs["control"], policy, []),
        "actsemble": build_system(system_cfgs["actsemble"], policy, [component]),
    }
    check("three runtime systems constructed", len(systems) == 3)
    check("policy-checkpoint identity across systems",
          systems["standalone"].policy.checkpoint_hash
          == systems["control"].policy.checkpoint_hash
          == systems["actsemble"].policy.checkpoint_hash)
    check("dataset hash identity policy/component",
          policy.dataset_hash == component.dataset_hash == dataset_hash)

    # 10./11. Rollouts + metrics + videos ------------------------------------------
    from actsemble.evaluation.evaluator import evaluate_system

    eval_cfg = {"name": "smoke", "regime": "nominal", "seed": 5,
                "num_episodes": args.eval_episodes, "max_steps": 40,
                "perturbations": [], "video": {"max_success": 1, "max_failure": 1}}
    results = {}
    for name, sys_cfg in system_cfgs.items():
        # env=None: each system gets its own freshly warmed env so simulator
        # histories are identical (candidate-identity precondition, §11).
        results[name] = evaluate_system(
            system_cfg=sys_cfg, eval_cfg=eval_cfg,
            policy_checkpoint=policy_out["checkpoints"]["best_ema"],
            component_checkpoints=(
                [comp_out["checkpoints"]["best"]] if name == "actsemble" else []
            ),
            output_path=workdir / f"eval_{name}.json",
            video_dir=workdir / "videos",
            device=device, env=None,
        )
        r = results[name]
        check(f"rollout: {r['system_name']}",
              r["exception_rate"] == 0.0,
              f"{r['success_count']}/{r['num_episodes']} success (rate not judged)")
    videos = list((workdir / "videos").glob("*.mp4"))
    check("rollout videos saved", len(videos) >= 1, f"{len(videos)} files")

    # 12.-14. Identity checks over saved results -------------------------------------
    check("results share dataset hash",
          all(r["dataset_hash"] == dataset_hash for r in results.values()))
    check("results share policy checkpoint hash",
          len({r["policy_checkpoint_hash"] for r in results.values()}) == 1)
    check("paired seeds identical across systems",
          all(r["environment_seeds"] == results["standalone"]["environment_seeds"]
              for r in results.values()))

    from actsemble.evaluation.reports import compare_systems

    report = compare_systems(
        [results["standalone"], results["control"], results["actsemble"]]
    )
    check("comparison report generated", "pairwise_vs_baseline" in report,
          "; ".join(report["warnings"]) or "no warnings")

    env.close()
    print(f"\n[smoke] ALL {len(CHECKS)} CHECKS PASSED")
    if not args.keep and args.workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)
        print("[smoke] cleaned up workdir")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)

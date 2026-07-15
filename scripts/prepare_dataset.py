#!/usr/bin/env python
"""Prepare a frozen Actsemble dataset from official ManiSkill demonstrations.

Locates the demonstration bundle, converts it via state-projection replay
in an environment matching the recorded controller and backend, filters to
successful episodes, writes the dataset + private provenance sidecar, and
validates the result.

Usage:
    python scripts/prepare_dataset.py --config configs/data/smoke.yaml \
        --output data/push_t_smoke.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.config import load_config
from actsemble.data.reader import DatasetReader
from actsemble.data.schema import DatasetMetadata
from actsemble.data.validation import validate_dataset, validate_success_only_provenance
from actsemble.data.writer import write_dataset, write_private_provenance
from actsemble.sim.demonstration_source import (
    ManiSkillTrajectorySource,
    convert_demonstrations,
    probe_state_layout,
)
from actsemble.sim.env_factory import env_contract, make_env


def default_bundle_path(task_id: str, control_mode: str, backend: str) -> Path:
    root = Path.home() / ".maniskill" / "demos" / task_id
    matches = sorted(root.glob(f"**/trajectory.none.{control_mode}.{backend}.h5"))
    if not matches:
        raise FileNotFoundError(
            f"No demonstration bundle for {task_id} / {control_mode} / {backend} under {root}. "
            f"Run: python -m mani_skill.utils.download_demo {task_id!r}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="data config YAML")
    parser.add_argument("--output", required=True, help="output dataset .h5 path")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="deterministic nested subset size (protocol §7); "
                             "overrides the config value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_id = cfg["task_id"]
    control_mode = cfg["control_mode"]
    sim_backend = cfg["sim_backend"]
    max_episodes = cfg.get("max_episodes")
    subset_size = args.subset_size if args.subset_size is not None else cfg.get("subset_size")
    subset_seed = int(cfg.get("subset_seed", 0))

    traj_path = cfg.get("trajectory_path") or default_bundle_path(
        task_id, control_mode, sim_backend
    )
    print(f"[prepare] source bundle: {traj_path}")
    source = ManiSkillTrajectorySource(
        traj_path, max_episodes=max_episodes,
        subset_size=subset_size, subset_seed=subset_seed,
    )
    if subset_size is not None:
        print(f"[prepare] nested subset: size={subset_size} seed={subset_seed}")

    print(f"[prepare] creating env: {task_id} / {control_mode} / {sim_backend}")
    env = make_env(
        task_id=task_id,
        control_mode=control_mode,
        sim_backend=sim_backend,
        obs_mode="state",
        render_mode=None,
    )
    contract = env_contract(env)

    print("[prepare] probing state layout (obs_mode=state_dict)...")
    layout_env = make_env(
        task_id=task_id,
        control_mode=control_mode,
        sim_backend=sim_backend,
        obs_mode="state_dict",
        render_mode=None,
    )
    state_layout = probe_state_layout(layout_env)
    layout_env.close()
    print(f"[prepare] state layout: {state_layout}")

    print(f"[prepare] converting up to {max_episodes or 'all'} episodes via state projection...")
    episodes, provenance = convert_demonstrations(source, env)
    if not episodes:
        print("[prepare] ERROR: no successful episodes exported", file=sys.stderr)
        return 1
    state_dim = episodes[0].state.shape[1]
    action_dim = episodes[0].action.shape[1]

    metadata = DatasetMetadata(
        simulator="ManiSkill3",
        simulator_version=contract["simulator_version"],
        task_id=task_id,
        robot=contract["robot"],
        observation_mode="state",
        state_dimension=state_dim,
        state_layout=json.dumps(state_layout),
        controller=contract["controller"],
        action_dimension=action_dim,
        action_definition=json.dumps(
            {
                "semantics": cfg.get(
                    "action_semantics",
                    "normalized end-effector delta pose: xyz translation then "
                    "axis-angle rotation, scaled by the controller",
                ),
                "frame": cfg.get("action_frame", "root_translation:root_aligned_body_rotation"),
                "units": "normalized [-1, 1] per dimension",
                "bounds": [contract["action_low"].tolist(), contract["action_high"].tolist()],
                "scaling": "controller clip_and_scale from [-1,1] to physical deltas",
                "clipping_rules": "recorded RL actions clipped to bounds at export "
                "(controller clips identically at execution)",
            }
        ),
        control_frequency=contract["control_frequency"],
        simulation_backend=contract["simulation_backend"],
        source_dataset=str(traj_path),
        generation_or_replay_seed=int(cfg.get("seed", 0)),
    )
    metadata.extra["source_num_envs_at_record_time"] = int(
        source.env_info.get("env_kwargs", {}).get("num_envs", 1)
    )
    metadata.extra["conversion_method"] = "state_projection_replay"
    # Audit trail (protocol §7, §17): master source identity + nested subset.
    from actsemble.utils.hashing import hash_file, hash_json

    manifest = source.subset_manifest()
    metadata.extra["source_bundle_sha256"] = hash_file(traj_path)
    metadata.extra["subset_size"] = -1 if subset_size is None else int(subset_size)
    metadata.extra["subset_seed"] = subset_seed
    metadata.extra["subset_hash"] = hash_json(manifest)
    provenance["subset_manifest"] = manifest

    dataset_hash = write_dataset(args.output, episodes, metadata)
    sidecar = write_private_provenance(args.output, provenance)
    env.close()

    print(f"[prepare] wrote {len(episodes)} episodes -> {args.output}")
    print(f"[prepare] dataset_hash: {dataset_hash}")
    print(f"[prepare] provenance sidecar: {sidecar}")
    print(f"[prepare] rejected: {provenance['rejected_count']}, "
          f"conversion failures: {provenance['conversion_failure_count']}")

    print("[prepare] validating...")
    reader = DatasetReader(args.output)
    summary = validate_dataset(reader)
    validate_success_only_provenance(args.output)
    print(f"[prepare] validation OK: {summary['num_episodes']} episodes, "
          f"{summary['num_transitions']} transitions, "
          f"state_dim={summary['state_dim']}, action_dim={summary['action_dim']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

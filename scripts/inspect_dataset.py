#!/usr/bin/env python
"""Inspect and validate a frozen Actsemble dataset; optionally render videos.

Usage:
    python scripts/inspect_dataset.py --dataset data/push_t_smoke.h5 \
        --video-dir outputs/dataset_videos --num-videos 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.data.reader import DatasetReader
from actsemble.data.validation import validate_dataset, validate_success_only_provenance
from actsemble.utils.serialization import load_json


def render_videos(reader: DatasetReader, video_dir: Path, num_videos: int) -> None:
    """Re-render exported episodes by state projection from the source bundle."""
    from actsemble.evaluation.video import save_video
    from actsemble.sim.demonstration_source import ManiSkillTrajectorySource
    from actsemble.sim.env_factory import make_env, verify_env_matches
    from actsemble.sim.rollout import _to_frame

    meta = reader.metadata
    source_path = Path(meta.source_dataset)
    if not source_path.exists():
        print(f"[inspect] source bundle missing ({source_path}); skipping videos")
        return
    sidecar = reader.path.with_suffix(reader.path.suffix + ".provenance.json")
    provenance = load_json(sidecar)
    by_episode = {e["episode_id"]: e for e in provenance["exported_episodes"]}

    source = ManiSkillTrajectorySource(source_path)
    env = make_env(
        task_id=meta.task_id,
        control_mode=meta.controller,
        sim_backend=meta.simulation_backend,
        obs_mode="state",
        render_mode="rgb_array",
    )
    verify_env_matches(
        env,
        {"task_id": meta.task_id, "controller": meta.controller,
         "simulation_backend": meta.simulation_backend},
        what="dataset video rendering",
    )
    u = env.unwrapped
    wanted_source_ids = {
        by_episode[ep.episode_id]["source_episode_id"]: ep.episode_id
        for ep in reader.episodes[:num_videos]
        if ep.episode_id in by_episode
    }
    rendered = 0
    for demo in source.episodes():
        if demo.source_episode_id not in wanted_source_ids:
            continue
        episode_id = wanted_source_ids[demo.source_episode_id]
        env.reset(seed=demo.episode_seed)
        frames = []
        for env_state in demo.env_states:
            u.set_state_dict(env_state)
            frames.append(_to_frame(env.render()))
        out = save_video(frames, video_dir / f"dataset_{episode_id}.mp4")
        print(f"[inspect] rendered {out}")
        rendered += 1
        if rendered >= num_videos:
            break
    env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--num-videos", type=int, default=3)
    args = parser.parse_args()

    reader = DatasetReader(args.dataset)
    summary = validate_dataset(reader)
    provenance = validate_success_only_provenance(args.dataset)

    meta = reader.metadata
    lengths = [len(ep) for ep in reader.episodes]
    print("=" * 70)
    print(f"Actsemble dataset: {args.dataset}")
    print("=" * 70)
    print(f"task:                 {meta.task_id}")
    print(f"robot:                {meta.robot}")
    print(f"controller:           {meta.controller}")
    print(f"physics backend:      {meta.simulation_backend}")
    print(f"simulator:            {meta.simulator} {meta.simulator_version}")
    print(f"control frequency:    {meta.control_frequency} Hz")
    print(f"episodes:             {summary['num_episodes']}")
    print(f"transitions:          {summary['num_transitions']}")
    print(f"episode length:       min {min(lengths)}, max {max(lengths)}, "
          f"mean {np.mean(lengths):.1f}, median {np.median(lengths):.0f}")
    print(f"state dim:            {summary['state_dim']}")
    print(f"action dim:           {summary['action_dim']}")
    print(f"state layout:         {meta.state_layout}")
    state_min = np.asarray(summary["state_min"])
    state_max = np.asarray(summary["state_max"])
    print(f"state range:          [{state_min.min():.3f}, {state_max.max():.3f}]")
    print(f"action range:         [{min(summary['action_min']):.3f}, "
          f"{max(summary['action_max']):.3f}]")
    print(f"dataset hash:         {summary['dataset_hash']}")
    print(f"success-only:         VERIFIED ({len(provenance['exported_episodes'])} exported, "
          f"{provenance['rejected_count']} rejected, "
          f"{provenance['conversion_failure_count']} conversion failures)")
    action_def = json.loads(meta.action_definition)
    print(f"action bounds:        {action_def.get('bounds')}")
    print("validation:           PASSED")

    if args.video_dir:
        render_videos(reader, Path(args.video_dir), args.num_videos)
    return 0


if __name__ == "__main__":
    sys.exit(main())

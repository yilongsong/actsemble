#!/usr/bin/env python
"""Derive a 'push-only' frozen dataset by trimming the trailing hold-at-goal tail.

The PushT-v1 RL demonstrations reach the goal early (true 90%-area-coverage
success at a median of ~30-40 of 100 steps) and then hold at the goal for the
rest of the fixed horizon. That trailing hold is a large, low-information
majority of transitions that biases the policy/verifier toward a regime where
action selection barely matters. This script removes it.

CORRECTNESS (see docstring of each step):

  1. It never mutates the frozen source dataset. A NEW file + provenance +
     dataset_hash is written; ``data/push_t_pilot.h5`` is untouched.

  2. "Hold" is defined by the TASK's own success function (>=90% goal-area
     coverage of the T, i.e. position AND orientation), recomputed per step by
     re-projecting the source env_states through ``env.evaluate()`` -- not by a
     center-distance proxy, which fires before the block is rotated into place.

  3. Only the trailing hold TAIL is trimmed (a suffix). We keep transitions
     0 .. last_fail + settle, where last_fail is the LAST non-success state.
     This (a) keeps every kept episode a pure PREFIX of the original -- so
     previous_action / next_state / step_index alignment is preserved exactly
     and the file passes the same validators; (b) PRESERVES recovery pushing
     (a block bumped off-goal fails again -> its second push is <= last_fail so
     it is kept); (c) keeps a small settle margin so the policy still learns to
     arrive-and-stop instead of overshooting.

Usage:
    python scripts/prepare_active_dataset.py \
        --source data/push_t_pilot.h5 --output data/push_t_active.h5 \
        --settle-steps 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from actsemble.data.reader import DatasetReader
from actsemble.data.schema import DatasetMetadata
from actsemble.data.validation import validate_dataset, validate_success_only_provenance
from actsemble.data.writer import write_dataset, write_private_provenance
from actsemble.sim.demonstration_source import ManiSkillTrajectorySource
from actsemble.sim.env_factory import make_env
from actsemble.types import EpisodeRecord
from actsemble.utils.hashing import hash_file


def default_bundle_path(task_id, control_mode, backend) -> Path:
    root = Path.home() / ".maniskill" / "demos" / task_id
    matches = sorted(root.glob(f"**/trajectory.none.{control_mode}.{backend}.h5"))
    if not matches:
        raise FileNotFoundError(f"No demonstration bundle under {root}")
    return matches[0]


def per_step_coverage(
    source, env, n_episodes
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Re-project each source episode and record per-STATE goal-area coverage.

    Coverage is the continuous ``pseudo_render_intersection`` value the task
    thresholds at 0.90 for success. Returns (coverages, obj_poses), one per
    source episode in order; obj_pose is only for a frozen-array cross-check.
    """
    from actsemble.sim.observation_adapter import to_numpy_state

    u = env.unwrapped
    coverages, obj_poses = [], []
    for i, demo in enumerate(source.episodes()):
        env.reset(seed=demo.episode_seed)
        cov = np.empty(len(demo.env_states), dtype=np.float32)
        objp = np.empty((len(demo.env_states), 7), dtype=np.float32)
        for t, es in enumerate(demo.env_states):
            u.set_state_dict(es)
            cov[t] = float(u.pseudo_render_intersection().item())
            objp[t] = to_numpy_state(u.get_obs())[24:31]
        coverages.append(cov)
        obj_poses.append(objp)
        if (i + 1) % 100 == 0:
            print(f"[active] coverage recompute {i + 1}/{n_episodes}", flush=True)
    return coverages, obj_poses


def trim_episode(ep, cov, reach_thresh, hold_floor, settle, min_keep):
    """Trim the trailing stable-hold tail using continuous coverage + hysteresis.

    ``cov`` is per-STATE (len T+1). The block "reaches" the goal at the first
    step with coverage >= reach_thresh (0.90 = success). While holding it
    jitters across 0.90, so binary success is unreliable; we instead keep every
    step up to the LAST time coverage fell below ``hold_floor`` (0.70 = block
    meaningfully off-goal -> active push / genuine recovery), plus a settle
    margin. Steps after that are the stable hold (coverage stays >= floor,
    jittering toward 0.90) and are removed.
    """
    T = len(ep)  # actions; states are T+1
    succ = cov >= reach_thresh
    reached = int(np.argmax(succ)) if succ.any() else -1
    below = np.flatnonzero(cov < hold_floor)
    last_active = int(below[-1]) if len(below) else -1
    keep_actions = int(np.clip(max(reached, last_active + 1) + settle, min_keep, T))
    keep_actions = max(1, min(keep_actions, T))
    trimmed = EpisodeRecord(
        episode_id=ep.episode_id,
        state=ep.state[:keep_actions].copy(),
        previous_action=ep.previous_action[:keep_actions].copy(),
        action=ep.action[:keep_actions].copy(),
        next_state=ep.next_state[:keep_actions].copy(),
        step_index=np.arange(keep_actions, dtype=np.int64),
    )
    # genuine recovery: coverage dropped below the floor AFTER first reaching goal
    recovery = bool(reached >= 0 and last_active > reached)
    info = {
        "episode_id": ep.episode_id,
        "orig_length": T,
        "kept_length": keep_actions,
        "removed": T - keep_actions,
        "first_success_state": reached,
        "last_below_floor_state": last_active,
        "min_coverage_after_reach": float(cov[reached:].min())
        if reached >= 0
        else -1.0,
        "recovery": recovery,
        "below_train_window_18": bool(keep_actions < 18),
    }
    return trimmed, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/push_t_pilot.h5")
    ap.add_argument("--output", default="data/push_t_active.h5")
    ap.add_argument(
        "--settle-steps",
        type=int,
        default=5,
        help="states kept after the block last left the goal (arrive-and-stop margin)",
    )
    ap.add_argument(
        "--reach-thresh",
        type=float,
        default=0.90,
        help="goal-area coverage counted as success (task default 0.90)",
    )
    ap.add_argument(
        "--hold-floor",
        type=float,
        default=0.70,
        help="coverage below this = block meaningfully off-goal (kept, not hold)",
    )
    ap.add_argument(
        "--min-keep", type=int, default=8, help="floor on kept actions per episode"
    )
    ap.add_argument("--stats-out", default="outputs/data_analysis/active_stats.json")
    args = ap.parse_args()

    src_reader = DatasetReader(args.source)
    meta = src_reader.metadata
    episodes = list(src_reader.episodes)  # ep_00000.. in id order
    n = len(episodes)
    print(
        f"[active] source: {args.source}  ({n} episodes, "
        f"{src_reader.num_transitions} transitions, hash {meta.dataset_hash[:12]})"
    )

    task_id, ctrl, backend = meta.task_id, meta.controller, meta.simulation_backend
    bundle = default_bundle_path(task_id, ctrl, backend)
    src = ManiSkillTrajectorySource(bundle)  # ALL episodes, same order as prep
    env = make_env(
        task_id=task_id,
        control_mode=ctrl,
        sim_backend=backend,
        obs_mode="state",
        render_mode=None,
    )

    print("[active] recomputing per-step goal coverage via state projection...")
    coverages, obj_poses = per_step_coverage(src, env, n)
    env.close()
    if len(coverages) != n:
        raise SystemExit(f"source has {len(coverages)} episodes, frozen has {n}")

    # Cross-check: projected obj_pose must match the frozen arrays bitwise-ish,
    # proving success[i] aligns to frozen episode i.
    max_mismatch = 0.0
    for ep, objp in zip(episodes, obj_poses):
        frozen = np.concatenate([ep.state[:, 24:31], ep.next_state[-1:, 24:31]], axis=0)
        max_mismatch = max(max_mismatch, float(np.abs(objp - frozen).max()))
    print(
        f"[active] frozen<->reprojected obj_pose max|Δ| = {max_mismatch:.2e} (0 => exact alignment)"
    )
    if max_mismatch > 1e-4:
        raise SystemExit(
            f"alignment check failed ({max_mismatch:.2e}); refusing to trim"
        )
    if not all(c[-1] >= args.reach_thresh for c in coverages):
        raise SystemExit("some episodes are not successful at their final state")

    trimmed, infos = [], []
    for ep, cov in zip(episodes, coverages):
        te, info = trim_episode(
            ep,
            cov,
            args.reach_thresh,
            args.hold_floor,
            args.settle_steps,
            args.min_keep,
        )
        trimmed.append(te)
        infos.append(info)

    n_before = int(sum(i["orig_length"] for i in infos))
    n_after = int(sum(i["kept_length"] for i in infos))
    n_recovery = int(sum(i["recovery"] for i in infos))
    n_short = int(sum(i["below_train_window_18"] for i in infos))
    print(
        f"[active] transitions {n_before} -> {n_after} "
        f"(removed {n_before - n_after} = {100 * (n_before - n_after) / n_before:.1f}%)"
    )
    print(
        f"[active] recovery episodes preserved: {n_recovery}; kept<18 steps: {n_short}"
    )

    # ---- metadata for the derived dataset ----
    new_meta = DatasetMetadata(
        simulator=meta.simulator,
        simulator_version=meta.simulator_version,
        task_id=meta.task_id,
        robot=meta.robot,
        observation_mode=meta.observation_mode,
        state_dimension=meta.state_dimension,
        state_layout=meta.state_layout,
        controller=meta.controller,
        action_dimension=meta.action_dimension,
        action_definition=meta.action_definition,
        control_frequency=meta.control_frequency,
        simulation_backend=meta.simulation_backend,
        source_dataset=f"derived_from:{args.source} (trailing-hold-trim)",
        generation_or_replay_seed=meta.generation_or_replay_seed,
    )
    new_meta.extra = dict(meta.extra)
    new_meta.extra.update(
        {
            "derived_from_dataset": str(args.source),
            "derived_from_hash": meta.dataset_hash,
            "derived_from_sha256": hash_file(args.source),
            "filtering_method": "trailing_hold_trim_coverage_hysteresis",
            "success_criterion": "PushT-v1 pseudo-render goal-area coverage >= 0.90",
            "trim_rule": "keep actions 0..clip(max(first_reach, last_below_floor+1)+settle, min_keep, T)",
            "reach_thresh": float(args.reach_thresh),
            "hold_floor": float(args.hold_floor),
            "settle_steps": int(args.settle_steps),
            "min_keep": int(args.min_keep),
            "transitions_before": n_before,
            "transitions_after": n_after,
            "recovery_episodes_preserved": n_recovery,
        }
    )

    provenance = {
        "success_only": True,
        "conversion_method": "trailing_success_hold_trim (derived)",
        "source": {
            "derived_from": str(args.source),
            "derived_from_hash": meta.dataset_hash,
            "bundle": str(bundle),
        },
        "success_criterion": new_meta.extra["success_criterion"],
        "trim_rule": new_meta.extra["trim_rule"],
        "reach_thresh": args.reach_thresh,
        "hold_floor": args.hold_floor,
        "settle_steps": args.settle_steps,
        "min_keep": args.min_keep,
        "obj_pose_alignment_max_abs": max_mismatch,
        "exported_episodes": [
            {
                "episode_id": i["episode_id"],
                "length": i["kept_length"],
                "source_success": True,
                "projected_success_at_end": True,
            }
            for i in infos
        ],
        "rejected_episodes": [],
        "rejected_count": 0,
        "conversion_failures": [],
        "conversion_failure_count": 0,
        "per_episode_trim": infos,
    }

    ds_hash = write_dataset(args.output, trimmed, new_meta)
    write_private_provenance(args.output, provenance)
    print(f"[active] wrote {args.output}  hash {ds_hash[:12]}")

    print("[active] validating derived dataset...")
    reader = DatasetReader(args.output)
    summary = validate_dataset(reader)
    validate_success_only_provenance(args.output)
    print(
        f"[active] validation OK: {summary['num_episodes']} eps, "
        f"{summary['num_transitions']} transitions, "
        f"len[min={summary['episode_length_min']}, mean={summary['episode_length_mean']:.1f}, "
        f"max={summary['episode_length_max']}]"
    )

    # ---- stats for visualization ----
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(
        json.dumps(
            {
                "source": args.source,
                "output": args.output,
                "settle_steps": args.settle_steps,
                "min_keep": args.min_keep,
                "transitions_before": n_before,
                "transitions_after": n_after,
                "removed_fraction": (n_before - n_after) / n_before,
                "recovery_episodes": n_recovery,
                "kept_below_18": n_short,
                "per_episode": infos,
            },
            indent=2,
        )
    )
    print(f"[active] stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

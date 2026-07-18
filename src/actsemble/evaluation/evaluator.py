"""Paired closed-loop evaluation of one autonomy system.

Fairness verification before any rollout:
* the live env must match the policy checkpoint's task / controller /
  simulation backend (loud EnvironmentMismatchError otherwise);
* components must share the policy's dataset/split/normalization hashes
  (checked at system construction).

Episode seeds come from a fixed panel (see evaluation/panels.py): every
system evaluated on the same panel receives identical environment seeds,
policy-sampling seeds, and perturbation seeds. Per protocol §11, each
episode's ``candidate_root_seed`` is its policy-sampling seed, and every
replan records a candidate-tensor hash so paired comparisons can verify
candidate identity after the fact.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

import mani_skill

from ..components.action_chunk_compatibility import ActionChunkCompatibility
from ..policies.interface import sampler_provenance
from ..policies.loader import load_policy
from ..sim.env_factory import env_contract, make_env, verify_env_matches
from ..sim.perturbations.base import build_perturbations
from ..sim.rollout import run_episode
from ..systems.factory import build_system
from ..types import StateObservation
from ..utils.provenance import runtime_provenance
from ..utils.repo import current_git_commit, git_provenance
from ..utils.hashing import hash_json
from ..utils.serialization import save_json
from .metrics import wilson_interval
from .panels import Panel, PanelEpisode, panel_episodes
from .video import save_video

PRIMARY_METRIC = "success_once"  # ManiSkill PushT-v1 terminates on success;
# success_at_end is recorded alongside for completeness.
RESULT_SCHEMA = "actsemble_evaluation_v2"


def panel_from_eval_cfg(eval_cfg: dict, num_episodes: int | None = None) -> Panel:
    """Panel from an eval config: explicit ``panel:`` block, or the legacy
    (seed, num_episodes) pair treated as a panel root."""
    if "panel" in eval_cfg and eval_cfg["panel"]:
        p = eval_cfg["panel"]
        return Panel(
            name=str(p.get("name", "custom")),
            env_seed=int(p["env_seed"]),
            num_episodes=int(
                num_episodes if num_episodes is not None else p["num_episodes"]
            ),
        )
    n = int(
        num_episodes if num_episodes is not None else eval_cfg.get("num_episodes", 20)
    )
    return Panel(name="legacy", env_seed=int(eval_cfg.get("seed", 0)), num_episodes=n)


def run_panel_episode(
    env,
    system,
    ep: PanelEpisode,
    *,
    max_steps: int,
    pert_specs: list[dict],
    capture_video: bool = False,
):
    """One panel episode: seeds wired per protocol, then a plain rollout."""
    perturbations = build_perturbations(
        [{**spec, "seed": ep.perturbation_seed} for spec in pert_specs]
    )
    system.candidate_root_seed = ep.policy_sampling_seed
    return run_episode(
        env,
        system,
        episode_seed=ep.env_seed,
        max_steps=max_steps,
        perturbations=perturbations,
        capture_video=capture_video,
    )


def episode_row(ep: PanelEpisode, result) -> dict:
    d = result.diagnostics
    candidate_hashes = d.get("candidate_hashes", [])
    replans_compact = [
        {
            "replan_index": r.get("replan_index"),
            "selected_index": r.get("selected_index"),
            "component_scores": r.get("component_scores"),
            "candidate_mean_abs": r.get("candidate_mean_abs"),
            "candidate_smoothness": r.get("candidate_smoothness"),
            "fallback": r.get("fallback", False),
            "policy_latency_s": r.get("policy_latency_s"),
            "component_latency_s": r.get("component_latency_s"),
            # optional selector diagnostics (present for consensus selectors;
            # None for the standalone / control / verifier systems)
            "selector_type": r.get("selector_type"),
            "selector_latency_s": r.get("selector_latency_s"),
            "mean_pairwise_distance": r.get("mean_pairwise_distance"),
            "changed_from_candidate_zero": r.get("changed_from_candidate_zero"),
        }
        for r in d.get("replans", [])
    ]
    return {
        "action_digest": d.get("action_digest"),
        "replans": replans_compact,
        "episode_index": ep.episode_index,
        "env_seed": ep.env_seed,
        "policy_sampling_seed": ep.policy_sampling_seed,
        "perturbation_seed": ep.perturbation_seed,
        "success_once": result.success_once,
        "success_at_end": result.success_at_end,
        "num_steps": result.num_steps,
        "timed_out": result.timed_out,
        "exception": result.exception,
        "fallback_count": d.get("fallback_count", 0),
        "num_replans": d.get("num_replans", 0),
        "action_clip_rate": d.get("action_clip_rate", 0.0),
        "actions_clipped": d.get("actions_clipped", 0),
        "actions_total": d.get("actions_total", result.num_steps),
        "mean_policy_latency_s": d.get("mean_policy_latency_s", 0.0),
        "mean_component_latency_s": d.get("mean_component_latency_s", 0.0),
        "mean_decision_latency_s": d.get("mean_decision_latency_s", 0.0),
        "decision_latencies_s": d.get("decision_latencies_s", []),
        "selected_indices": d.get("selected_indices", []),
        "selection_change_rate": d.get("selection_change_rate", 0.0),
        "selection_change_count": d.get("selection_change_count", 0),
        "candidate_hashes": candidate_hashes,
        "candidate_digest": hashlib.sha256(
            "".join(candidate_hashes).encode()
        ).hexdigest()
        if candidate_hashes
        else None,
    }


def _warmup_system(system, policy) -> None:
    """Run one unmeasured decision to initialize policy/component CUDA kernels."""
    system.reset(episode_seed=-1)
    system.candidate_root_seed = 0
    system.act(
        StateObservation(
            state=np.zeros(policy.meta.state_dim, dtype=np.float32),
            previous_action=np.zeros(policy.meta.action_dim, dtype=np.float32),
            step_index=0,
        )
    )
    system.reset(episode_seed=-1)


def evaluate_system(
    *,
    system_cfg: dict,
    eval_cfg: dict,
    policy_checkpoint: str | Path,
    component_checkpoints: list[str | Path] | None = None,
    num_episodes: int | None = None,
    output_path: str | Path | None = None,
    video_dir: str | Path | None = None,
    device: str = "cuda",
    use_ema: bool = True,
    env=None,
    num_candidates_override: int | None = None,
    force: bool = False,
) -> dict:
    if output_path is not None and Path(output_path).exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Completed evaluations must not be "
            f"silently overwritten (protocol §17): pass force=True / --force, "
            f"or write into a new experiment-version directory."
        )
    component_checkpoints = component_checkpoints or []
    policy = load_policy(policy_checkpoint, device=device, use_ema=use_ema)
    components = [
        ActionChunkCompatibility.from_checkpoint(p, device=device)
        for p in component_checkpoints
    ]
    if num_candidates_override is not None:
        system_cfg = {
            **system_cfg,
            "policy": {
                **system_cfg.get("policy", {}),
                "num_candidates": int(num_candidates_override),
            },
        }
    system = build_system(system_cfg, policy, components)

    panel = panel_from_eval_cfg(eval_cfg, num_episodes)
    if panel.num_episodes < 1:
        raise ValueError(
            f"Evaluation panel must contain at least one episode, got {panel.num_episodes}"
        )
    max_steps = int(eval_cfg.get("max_steps", 100))
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    regime = str(eval_cfg.get("regime", eval_cfg.get("name", "nominal")))
    pert_specs = eval_cfg.get("perturbations", []) or []

    owns_env = env is None
    if env is None:
        env = make_env(
            task_id=policy.meta.task_id,
            control_mode=policy.meta.controller,
            sim_backend=policy.meta.simulation_backend,
            obs_mode="state",
            render_mode="rgb_array",
        )
    expected_environment = {
        "task_id": policy.meta.task_id,
        "controller": policy.meta.controller,
        "simulation_backend": policy.meta.simulation_backend,
        "action_dimension": policy.meta.action_dim,
        "action_low": np.asarray(policy.meta.action_low, dtype=np.float32),
        "action_high": np.asarray(policy.meta.action_high, dtype=np.float32),
    }
    recorded_environment = policy.meta.extra.get("environment", {})
    for key in ("simulator_version", "robot", "control_frequency"):
        if key in recorded_environment:
            expected_environment[key] = recorded_environment[key]
    verify_env_matches(
        env,
        expected_environment,
        what=f"evaluation of {system.name}",
    )
    live_environment = env_contract(env)

    video_cfg = eval_cfg.get("video", {}) or {}
    want_videos = video_dir is not None
    max_success_videos = int(video_cfg.get("max_success", 2))
    max_failure_videos = int(video_cfg.get("max_failure", 2))
    saved_videos: dict[str, list[str]] = {"success": [], "failure": []}

    _warmup_system(system, policy)
    seeds = panel_episodes(panel)
    episodes_out = []
    t_start = time.time()
    for ep in seeds:
        capture = want_videos and (
            len(saved_videos["success"]) < max_success_videos
            or len(saved_videos["failure"]) < max_failure_videos
        )
        result, frames = run_panel_episode(
            env,
            system,
            ep,
            max_steps=max_steps,
            pert_specs=pert_specs,
            capture_video=capture,
        )
        success = getattr(result, PRIMARY_METRIC)
        if capture:
            category = "success" if success else "failure"
            limit = max_success_videos if success else max_failure_videos
            if len(saved_videos[category]) < limit:
                assert video_dir is not None
                path = save_video(
                    frames,
                    Path(video_dir)
                    / f"{system.name}_{regime}_{category}_seed{ep.env_seed}.mp4",
                )
                if path is not None:
                    saved_videos[category].append(str(path))
        episodes_out.append(episode_row(ep, result))

    successes = [e[PRIMARY_METRIC] for e in episodes_out]
    n_success = int(np.sum(successes))
    n_episodes = panel.num_episodes
    ci = wilson_interval(n_success, n_episodes)
    policy_latencies = [
        r["policy_latency_s"]
        for e in episodes_out
        for r in e["replans"]
        if r.get("policy_latency_s") is not None
    ]
    component_latencies = [
        r["component_latency_s"]
        for e in episodes_out
        for r in e["replans"]
        if r.get("component_latency_s") is not None
    ]
    decision_latencies = [x for e in episodes_out for x in e["decision_latencies_s"]]

    def latency_summary(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "mean_s": 0.0, "p95_s": 0.0, "p99_s": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean_s": float(arr.mean()),
            "p95_s": float(np.quantile(arr, 0.95)),
            "p99_s": float(np.quantile(arr, 0.99)),
        }

    result_json = {
        "project": "Actsemble",
        "result_schema": RESULT_SCHEMA,
        "phase": "state_based_mechanism_validation",
        "system_name": system.name,
        "task": policy.meta.task_id,
        "policy_checkpoint": str(policy_checkpoint),
        "policy_checkpoint_hash": policy.checkpoint_hash,
        "policy_weights_kind": policy.weights_kind,
        "component_checkpoints": [str(p) for p in component_checkpoints],
        "component_checkpoint_hashes": [c.checkpoint_hash for c in components],
        "dataset_hash": policy.dataset_hash,
        "split_hash": policy.meta.split_hash,
        "subset_hash": policy.meta.extra.get("subset_hash"),
        "normalization_hash": hash_json(policy.meta.normalization),
        "policy_config_hash": hash_json(policy.config),
        "simulation_backend": policy.meta.simulation_backend,
        "simulator_version": mani_skill.__version__,
        "controller": policy.meta.controller,
        "environment_contract": live_environment,
        "git_commit": current_git_commit(),
        "source": git_provenance(),
        "runtime": {**runtime_provenance(), "maniskill": mani_skill.__version__},
        "regime": regime,
        "panel": panel.to_dict(),
        "primary_metric": PRIMARY_METRIC,
        "num_candidates": system.num_candidates,
        "sampler": sampler_provenance(policy),
        "selection_type": (system_cfg.get("selection", {}) or {}).get(
            "type", "candidate_zero"
        ),
        "execution": {
            "action_horizon": system.action_horizon,
            "action_offset": system.execution_offset,
        },
        "fallback_rule": "candidate_zero_if_finite_else_first_valid; no_finite_candidate_raises",
        "num_episodes": n_episodes,
        "max_steps": max_steps,
        "environment_seeds": [e.env_seed for e in seeds],
        "perturbation_seeds": [e.perturbation_seed for e in seeds],
        "policy_sampling_seeds": [e.policy_sampling_seed for e in seeds],
        # legacy alias for policy_sampling_seeds (pre-protocol result files)
        "candidate_seeds": [e.policy_sampling_seed for e in seeds],
        "candidate_digests": [e["candidate_digest"] for e in episodes_out],
        "successes": [bool(s) for s in successes],
        "successes_at_end": [bool(e["success_at_end"]) for e in episodes_out],
        "success_count": n_success,
        "success_rate": n_success / n_episodes,
        "success_at_end_rate": float(
            np.mean([e["success_at_end"] for e in episodes_out])
        ),
        "confidence_interval": list(ci),
        "timeout_rate": float(np.mean([e["timed_out"] for e in episodes_out])),
        "exception_rate": float(
            np.mean([e["exception"] is not None for e in episodes_out])
        ),
        "average_episode_length": float(
            np.mean([e["num_steps"] for e in episodes_out])
        ),
        "fallback_episode_rate": float(
            np.mean([e["fallback_count"] > 0 for e in episodes_out])
        ),
        "fallback_replan_rate": sum(e["fallback_count"] for e in episodes_out)
        / max(1, sum(e["num_replans"] for e in episodes_out)),
        "action_clip_rate": sum(e["actions_clipped"] for e in episodes_out)
        / max(1, sum(e["actions_total"] for e in episodes_out)),
        "selection_change_rate": sum(e["selection_change_count"] for e in episodes_out)
        / max(1, sum(e["num_replans"] for e in episodes_out)),
        "latency": {
            "policy": latency_summary(policy_latencies),
            "component": latency_summary(component_latencies),
            "decision": latency_summary(decision_latencies),
        },
        "wall_time_s": time.time() - t_start,
        "videos": saved_videos,
        "episodes": episodes_out,
        "config": {"system": system_cfg, "evaluation": eval_cfg},
    }
    # Backward-compatible name, now unambiguously defined per replan.
    result_json["fallback_rate"] = result_json["fallback_replan_rate"]
    result_json["latency"].update(
        mean_policy_s=result_json["latency"]["policy"]["mean_s"],
        mean_component_s=result_json["latency"]["component"]["mean_s"],
        mean_decision_s=result_json["latency"]["decision"]["mean_s"],
    )
    if output_path is not None:
        save_json(result_json, output_path)
    if owns_env:
        env.close()
    return result_json

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
from ..policies.loader import load_policy
from ..sim.env_factory import make_env, verify_env_matches
from ..sim.perturbations.base import build_perturbations
from ..sim.rollout import run_episode
from ..systems.factory import build_system
from ..utils.repo import current_git_commit
from ..utils.serialization import save_json
from .metrics import wilson_interval
from .panels import Panel, PanelEpisode, panel_episodes
from .video import save_video

PRIMARY_METRIC = "success_once"  # ManiSkill PushT-v1 terminates on success;
# success_at_end is recorded alongside for completeness.


def panel_from_eval_cfg(eval_cfg: dict, num_episodes: int | None = None) -> Panel:
    """Panel from an eval config: explicit ``panel:`` block, or the legacy
    (seed, num_episodes) pair treated as a panel root."""
    if "panel" in eval_cfg and eval_cfg["panel"]:
        p = eval_cfg["panel"]
        return Panel(
            name=str(p.get("name", "custom")),
            env_seed=int(p["env_seed"]),
            num_episodes=int(num_episodes if num_episodes is not None else p["num_episodes"]),
        )
    n = int(num_episodes if num_episodes is not None else eval_cfg.get("num_episodes", 20))
    return Panel(name="legacy", env_seed=int(eval_cfg.get("seed", 0)), num_episodes=n)


def _sampler_provenance(policy) -> dict:
    """Exact inference-sampler settings, recorded into the result so a run is
    reproducible from the eval output — not just from source code. For diffusion
    this is the sampler, the spacing rule, and the EXACT timestep subsequence."""
    sched = getattr(policy, "scheduler", None)
    if sched is not None and hasattr(policy, "num_inference_steps"):
        n = int(policy.num_inference_steps)
        return {
            "family": "diffusion",
            "sampler": getattr(policy, "sampler", "ddim"),
            "num_train_steps": int(sched.num_train_steps),
            "num_inference_steps": n,
            "timestep_spacing": getattr(sched, "timestep_spacing", "leading"),
            "timesteps": sched.inference_timesteps(n).tolist(),
            "temperature": float(getattr(policy, "temperature", 1.0)),
        }
    if hasattr(policy, "num_inference_steps") and hasattr(policy, "time_scale"):
        return {"family": "flow", "sampler": "euler",
                "num_steps": int(policy.num_inference_steps),
                "time_scale": float(policy.time_scale)}
    return {"family": "deterministic"}  # ACT: latent pinned to 0 at inference


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
        "action_clip_rate": d.get("action_clip_rate", 0.0),
        "mean_policy_latency_s": d.get("mean_policy_latency_s", 0.0),
        "mean_component_latency_s": d.get("mean_component_latency_s", 0.0),
        "mean_decision_latency_s": d.get("mean_decision_latency_s", 0.0),
        "selected_indices": d.get("selected_indices", []),
        "selection_change_rate": d.get("selection_change_rate", 0.0),
        "candidate_hashes": candidate_hashes,
        "candidate_digest": hashlib.sha256(
            "".join(candidate_hashes).encode()
        ).hexdigest()[:16]
        if candidate_hashes
        else None,
    }


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
            "policy": {**system_cfg.get("policy", {}), "num_candidates": int(num_candidates_override)},
        }
    system = build_system(system_cfg, policy, components)

    panel = panel_from_eval_cfg(eval_cfg, num_episodes)
    max_steps = int(eval_cfg.get("max_steps", 100))
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
    verify_env_matches(
        env,
        {
            "task_id": policy.meta.task_id,
            "controller": policy.meta.controller,
            "simulation_backend": policy.meta.simulation_backend,
            "action_dimension": policy.meta.action_dim,
            "action_low": np.asarray(policy.meta.action_low, dtype=np.float32),
            "action_high": np.asarray(policy.meta.action_high, dtype=np.float32),
        },
        what=f"evaluation of {system.name}",
    )

    video_cfg = eval_cfg.get("video", {}) or {}
    want_videos = video_dir is not None
    max_success_videos = int(video_cfg.get("max_success", 2))
    max_failure_videos = int(video_cfg.get("max_failure", 2))
    saved_videos: dict[str, list[str]] = {"success": [], "failure": []}

    seeds = panel_episodes(panel)
    episodes_out = []
    t_start = time.time()
    for ep in seeds:
        capture = want_videos and (
            len(saved_videos["success"]) < max_success_videos
            or len(saved_videos["failure"]) < max_failure_videos
        )
        result, frames = run_panel_episode(
            env, system, ep, max_steps=max_steps, pert_specs=pert_specs, capture_video=capture
        )
        success = getattr(result, PRIMARY_METRIC)
        if capture:
            category = "success" if success else "failure"
            limit = max_success_videos if success else max_failure_videos
            if len(saved_videos[category]) < limit:
                path = save_video(
                    frames,
                    Path(video_dir) / f"{system.name}_{regime}_{category}_seed{ep.env_seed}.mp4",
                )
                if path is not None:
                    saved_videos[category].append(str(path))
        episodes_out.append(episode_row(ep, result))

    successes = [e[PRIMARY_METRIC] for e in episodes_out]
    n_success = int(np.sum(successes))
    n_episodes = panel.num_episodes
    ci = wilson_interval(n_success, n_episodes)
    result_json = {
        "project": "Actsemble",
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
        "simulation_backend": policy.meta.simulation_backend,
        "simulator_version": mani_skill.__version__,
        "controller": policy.meta.controller,
        "git_commit": current_git_commit(),
        "regime": regime,
        "panel": panel.to_dict(),
        "primary_metric": PRIMARY_METRIC,
        "num_candidates": system.num_candidates,
        "sampler": _sampler_provenance(policy),
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
        "success_at_end_rate": float(np.mean([e["success_at_end"] for e in episodes_out])),
        "confidence_interval": list(ci),
        "timeout_rate": float(np.mean([e["timed_out"] for e in episodes_out])),
        "exception_rate": float(np.mean([e["exception"] is not None for e in episodes_out])),
        "average_episode_length": float(np.mean([e["num_steps"] for e in episodes_out])),
        "fallback_rate": float(
            np.mean([e["fallback_count"] > 0 for e in episodes_out])
        ),
        "action_clip_rate": float(np.mean([e["action_clip_rate"] for e in episodes_out])),
        "selection_change_rate": float(
            np.mean([e["selection_change_rate"] for e in episodes_out])
        ),
        "latency": {
            "mean_policy_s": float(np.mean([e["mean_policy_latency_s"] for e in episodes_out])),
            "mean_component_s": float(
                np.mean([e["mean_component_latency_s"] for e in episodes_out])
            ),
            "mean_decision_s": float(
                np.mean([e["mean_decision_latency_s"] for e in episodes_out])
            ),
        },
        "wall_time_s": time.time() - t_start,
        "videos": saved_videos,
        "episodes": episodes_out,
        "config": {"system": system_cfg, "evaluation": eval_cfg},
    }
    if output_path is not None:
        save_json(result_json, output_path)
    if owns_env:
        env.close()
    return result_json

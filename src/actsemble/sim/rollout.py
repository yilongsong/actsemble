"""Closed-loop episode execution.

Runs one AutonomySystem in one environment episode with optional
perturbations and frame capture. Simulator exceptions are recorded in the
RolloutResult (never silently ignored); component failures are already
handled inside the systems and never crash an episode.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from ..types import RolloutResult
from .action_adapter import ActionAdapter
from .observation_adapter import ObservationAdapter


def _to_bool(value) -> bool:
    """Scalar bool from python/np/torch (possibly CUDA, possibly [1]-shaped)."""
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _to_frame(render_out) -> np.ndarray:
    frame = render_out
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8)


def run_episode(
    env,
    system,
    *,
    episode_seed: int,
    max_steps: int,
    perturbations: list | None = None,
    capture_video: bool = False,
) -> tuple[RolloutResult, list[np.ndarray]]:
    perturbations = perturbations or []
    u = env.unwrapped
    contract_low = np.asarray(u.single_action_space.low, dtype=np.float32)
    contract_high = np.asarray(u.single_action_space.high, dtype=np.float32)
    obs_adapter = ObservationAdapter(action_dim=contract_low.shape[0])
    act_adapter = ActionAdapter(contract_low, contract_high)

    system.reset(episode_seed=episode_seed)
    for p in perturbations:
        p.reset(episode_seed=episode_seed)

    frames: list[np.ndarray] = []
    success_once = False
    success_at_end = False
    step = 0
    exception: str | None = None
    step_latencies: list[float] = []
    # Digest of the exact executed action sequence (post-clip, pre-env):
    # two episodes with equal digests executed bitwise-identical commands.
    action_hasher = hashlib.sha256()

    try:
        raw_obs, _ = env.reset(seed=episode_seed)
        if capture_video:
            frames.append(_to_frame(env.render()))
        while step < max_steps:
            observation = obs_adapter.observe(raw_obs)
            for p in perturbations:
                observation = p.modify_observation(observation, step)

            t0 = time.perf_counter()
            action = system.act(observation)
            step_latencies.append(time.perf_counter() - t0)

            for p in perturbations:
                action = p.modify_action(action, step)
            env_action = act_adapter.adapt(action)
            action_hasher.update(
                np.ascontiguousarray(env_action, dtype=np.float32).tobytes()
            )

            for p in perturbations:
                p.before_step(env, step)
            raw_obs, _reward, terminated, truncated, info = env.step(env_action)
            for p in perturbations:
                p.after_step(env, step)

            obs_adapter.after_step(env_action)
            success = _to_bool(info["success"]) if "success" in info else False
            success_once = success_once or success
            success_at_end = success
            if capture_video:
                frames.append(_to_frame(env.render()))
            step += 1
            if _to_bool(terminated) or _to_bool(truncated):
                break
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"

    timed_out = step >= max_steps and not success_once and exception is None
    diagnostics = {
        **system.diagnostics(),
        "action_clip_rate": act_adapter.clip_rate,
        "actions_clipped": act_adapter.clipped_actions,
        "actions_total": act_adapter.total_actions,
        "mean_decision_latency_s": float(np.mean(step_latencies))
        if step_latencies
        else 0.0,
        "decision_latencies_s": step_latencies,
        "perturbations": [p.name for p in perturbations],
        "action_digest": action_hasher.hexdigest(),
    }
    result = RolloutResult(
        episode_seed=episode_seed,
        num_steps=step,
        success_once=success_once,
        success_at_end=success_at_end,
        timed_out=timed_out,
        exception=exception,
        diagnostics=diagnostics,
    )
    return result, frames

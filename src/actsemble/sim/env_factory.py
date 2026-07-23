"""ManiSkill environment construction with loud consistency checks.

The SAME controller and physics backend must be used for demonstration
conversion, training-data interpretation, and evaluation. Any mismatch
between a dataset/checkpoint contract and a live environment raises
EnvironmentMismatchError.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
import mani_skill
import mani_skill.envs  # noqa: F401  (registers ManiSkill environments)


class EnvironmentMismatchError(Exception):
    pass


def make_env(
    *,
    task_id: str,
    control_mode: str,
    sim_backend: str,
    obs_mode: str = "state",
    render_mode: str | None = "rgb_array",
    max_episode_steps: int | None = None,
    warmup_episodes: int = 2,
):
    kwargs: dict[str, Any] = dict(
        obs_mode=obs_mode,
        control_mode=control_mode,
        sim_backend=sim_backend,
        num_envs=1,
        render_mode=render_mode,
    )
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(task_id, **kwargs)
    if max_episode_steps is not None:
        got = effective_horizon(env)
        if got is not None and got != max_episode_steps:
            raise EnvironmentMismatchError(
                f"{task_id}: asked for max_episode_steps={max_episode_steps} but the "
                f"env truncates at {got}. Episodes would be cut short and score 0."
            )
    _warmup(env, warmup_episodes)
    return env


def effective_horizon(env) -> int | None:
    """The step count at which the env will actually set ``truncated``.

    Do NOT read ``env.spec.max_episode_steps`` for this: ManiSkill registers the
    horizon on its own TimeLimitWrapper, and ``spec.max_episode_steps`` reads
    None even when the env truncates at 50 -- which looks like "no limit" and is
    how a 160-step evaluation silently became a 50-step one, scoring every
    episode 0. Walk the wrapper chain instead.

    Returns None if no limiting wrapper is found (genuinely unlimited).
    """
    seen, depth = None, 0
    node = env
    while node is not None and depth < 16:
        value = getattr(node, "_max_episode_steps", None)
        if value is not None:
            seen = int(value) if seen is None else min(seen, int(value))
        node = getattr(node, "env", None)
        depth += 1
    return seen


# Fixed private seeds for simulator warmup; never part of any panel.
_WARMUP_SEEDS = (987654321, 987654322)


def _warmup(env, episodes: int) -> None:
    """Run throwaway contact-exercising episodes after env creation.

    GPU PhysX lazily grows internal buffers on first contact-heavy use;
    the very first episode of a fresh env can differ from a bitwise-exact
    replay at ~1e-7 once differently-seeded episodes interleave. Warming
    the simulator before any recorded episode makes every subsequent
    same-seed episode bitwise reproducible regardless of position, which
    the paired-evaluation and candidate-identity checks rely on.
    """
    rng = np.random.default_rng(0)
    space = env.unwrapped.single_action_space
    for i in range(episodes):
        env.reset(seed=_WARMUP_SEEDS[i % len(_WARMUP_SEEDS)])
        for _ in range(20):
            action = rng.uniform(space.low, space.high).astype(np.float32)
            env.step(action)


def env_contract(env) -> dict:
    """The environment facts that must match dataset/checkpoint metadata."""
    u = env.unwrapped
    single = u.single_action_space
    return {
        "simulator": "ManiSkill3",
        "simulator_version": mani_skill.__version__,
        "task_id": env.spec.id if env.spec else "",
        "robot": str(u.agent.robot.name),
        "controller": str(u.control_mode),
        "simulation_backend": str(u.backend.sim_backend),
        "control_frequency": float(u.control_freq),
        "action_dimension": int(np.prod(single.shape)),
        "action_low": np.asarray(single.low, dtype=np.float32),
        "action_high": np.asarray(single.high, dtype=np.float32),
    }


def verify_env_matches(env, expected: dict, *, what: str) -> None:
    """Fail loudly when live env facts differ from recorded metadata.

    ``expected`` maps env_contract keys to required values; only provided
    keys are checked.
    """
    actual = env_contract(env)
    problems = []
    for key, want in expected.items():
        have = actual.get(key)
        if isinstance(want, np.ndarray) or isinstance(have, np.ndarray):
            ok = np.allclose(
                np.asarray(have, dtype=np.float64), np.asarray(want, dtype=np.float64)
            )
        else:
            ok = have == want
        if not ok:
            problems.append(f"{key}: expected {want!r}, live env has {have!r}")
    if problems:
        raise EnvironmentMismatchError(
            f"{what}: live environment is incompatible with recorded metadata:\n  - "
            + "\n  - ".join(problems)
        )

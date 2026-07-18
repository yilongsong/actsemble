"""Stage 1 — policy checkpoint screening (§5).

Cheap periodic evaluation of interval checkpoints on the fixed screening
panel: EMA weights, ONE action-chunk sample per replanning step, the
checkpoint's own frozen sampler settings, identical seed bank for every
checkpoint. Screening only generates confirmation candidates; it never
selects the final checkpoint, never stops training, and — because it runs
either out-of-process or inside the trainer's RNG guard — never alters
the training trajectory (§3).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..evaluation.evaluator import episode_row, run_panel_episode
from ..evaluation.metrics import wilson_interval
from ..evaluation.panels import Panel, panel_episodes
from ..policies.loader import load_policy
from ..systems.factory import build_system
from ..utils.hashing import hash_file
from ..utils.serialization import load_json, save_json

# Standalone candidate-zero config; build_system derives the execution offset
# from the policy's window alignment (H_o-1 for diffusion-policy chunks).
_STANDALONE_CFG = {
    "policy": {"num_candidates": 1},
    "selection": {"type": "candidate_zero"},
    "execution": {},
}


def evaluate_checkpoint_on_panel(
    checkpoint_path: str | Path,
    panel: Panel,
    *,
    env,
    device: str = "cuda",
    max_steps: int = 100,
    pert_specs: list[dict] | None = None,
) -> dict:
    """Standalone (K=1, candidate zero) evaluation of one checkpoint on a
    fixed panel. Shared by screening and confirmation so both stages use
    exactly the same rollout semantics. Policy-architecture-agnostic (diffusion /
    ACT / flow via ``load_policy``); the execution offset is applied for
    diffusion-policy-aligned checkpoints via ``build_system``."""
    policy = load_policy(checkpoint_path, device=device, use_ema=True)
    system = build_system(_STANDALONE_CFG, policy, [])
    rows = []
    for ep in panel_episodes(panel):
        result, _ = run_panel_episode(
            env, system, ep, max_steps=max_steps, pert_specs=pert_specs or []
        )
        rows.append(episode_row(ep, result))
    successes = int(np.sum([r["success_once"] for r in rows]))
    n = panel.num_episodes
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": hash_file(checkpoint_path),
        "panel": panel.to_dict(),
        "num_episodes": n,
        "success_count": successes,
        "success_rate": successes / n,
        "wilson_ci": list(wilson_interval(successes, n)),
        "environment_seeds": [r["env_seed"] for r in rows],
        "policy_sampling_seeds": [r["policy_sampling_seed"] for r in rows],
        "episodes": rows,
        "exceptions": int(np.sum([r["exception"] is not None for r in rows])),
    }


def checkpoint_step(checkpoint_path: str | Path) -> int:
    m = re.search(r"step_(\d+)\.pt$", str(checkpoint_path))
    if not m:
        raise ValueError(f"Cannot parse training step from {checkpoint_path}")
    return int(m.group(1))


def screening_history_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "screening" / "screening_history.json"


def screen_checkpoint(
    checkpoint_path: str | Path,
    run_dir: str | Path,
    panel: Panel,
    *,
    env,
    device: str = "cuda",
    max_steps: int = 100,
) -> dict:
    """Screen one interval checkpoint and append it to the run's history."""
    step = checkpoint_step(checkpoint_path)
    record = evaluate_checkpoint_on_panel(
        checkpoint_path, panel, env=env, device=device, max_steps=max_steps
    )
    record["step"] = step
    screening_dir = Path(run_dir) / "screening"
    save_json(record, screening_dir / f"step_{step:06d}.json")

    hist_path = screening_history_path(run_dir)
    history = load_json(hist_path) if hist_path.exists() else []
    history = [h for h in history if h["step"] != step]  # idempotent re-screen
    history.append(
        {
            "step": step,
            "checkpoint_path": record["checkpoint_path"],
            "checkpoint_hash": record["checkpoint_hash"],
            "success_count": record["success_count"],
            "success_rate": record["success_rate"],
            "wilson_ci": record["wilson_ci"],
            "exceptions": record["exceptions"],
        }
    )
    history.sort(key=lambda h: h["step"])
    save_json(history, hist_path)
    print(
        f"[screen] step {step}: {record['success_count']}/{record['num_episodes']} "
        f"= {record['success_rate']:.1%} CI [{record['wilson_ci'][0]:.1%}, "
        f"{record['wilson_ci'][1]:.1%}]"
    )
    return record


def make_screening_callback(run_dir: str | Path, panel: Panel, *, env, device: str,
                            max_steps: int = 100):
    """The trainer's on_checkpoint hook (invoked inside the RNG guard)."""

    def on_checkpoint(*, step: int, checkpoint_path: Path) -> None:
        screen_checkpoint(
            checkpoint_path, run_dir, panel, env=env, device=device, max_steps=max_steps
        )

    return on_checkpoint


def screen_all_snapshots(run_dir: str | Path, panel: Panel, *, env, device: str,
                         max_steps: int = 100) -> list[dict]:
    """(Re-)screen every saved interval snapshot of a training run."""
    snapshots = sorted((Path(run_dir) / "checkpoints").glob("step_*.pt"))
    if not snapshots:
        raise FileNotFoundError(f"No interval snapshots under {run_dir}/checkpoints")
    return [
        screen_checkpoint(p, run_dir, panel, env=env, device=device, max_steps=max_steps)
        for p in snapshots
    ]

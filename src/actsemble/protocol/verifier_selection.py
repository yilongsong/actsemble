"""Offline verifier checkpoint selection (§9).

Entirely offline: only held-out, episode-disjoint demonstration-derived
validation metrics recorded during training (offline_history.json). The
verifier receives NO simulator-derived model-selection signal.

Lexicographic rule:
    1. best primary offline metric (default: validation pairwise ranking
       accuracy; ``validation_loss`` is also supported, lower = better);
    2. best secondary offline metric (default: balanced accuracy);
    3. earliest checkpoint under a complete tie.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..utils.hashing import hash_file
from ..utils.repo import current_git_commit
from ..utils.serialization import load_json, save_json

# Metrics where LOWER is better; everything else is higher-is-better.
_LOWER_IS_BETTER = {"validation_loss"}


def _oriented(value: float, metric: str) -> float:
    return value if metric in _LOWER_IS_BETTER else -value


def select_verifier_record(
    offline_history: list[dict],
    *,
    primary: str = "pairwise_ranking_accuracy",
    secondary: str = "balanced_accuracy",
) -> dict:
    if not offline_history:
        raise ValueError("Empty offline history; train the verifier with checkpoint_every set")
    for h in offline_history:
        if primary not in h["metrics"]:
            raise KeyError(f"Primary metric {primary!r} missing at step {h['step']}")
    return sorted(
        offline_history,
        key=lambda h: (
            _oriented(h["metrics"][primary], primary),
            _oriented(h["metrics"].get(secondary, 0.0), secondary),
            h["step"],
        ),
    )[0]


def select_verifier(
    run_dir: str | Path,
    *,
    primary: str = "pairwise_ranking_accuracy",
    secondary: str = "balanced_accuracy",
    force: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    selected_path = run_dir / "selected_verifier.pt"
    if selected_path.exists() and not force:
        raise FileExistsError(
            f"{selected_path} already exists (protocol §17: pass force or use a "
            f"new experiment version)"
        )
    history = load_json(run_dir / "offline_history.json")
    winner = select_verifier_record(history, primary=primary, secondary=secondary)
    print(f"[select-verifier] SELECTED step {winner['step']} "
          f"({primary}={winner['metrics'][primary]:.4f}, evaluated on "
          f"{winner['evaluated_on']})")

    ckpt = torch.load(winner["checkpoint_path"], map_location="cpu", weights_only=False)
    train_cfg_path = run_dir / "train_config.json"
    train_config = load_json(train_cfg_path) if train_cfg_path.exists() else {}
    ckpt["meta"]["selection"] = {
        "protocol": "actsemble_verifier_selection_v1",
        "selected_step": winner["step"],
        "original_checkpoint_path": winner["checkpoint_path"],
        "original_checkpoint_hash": hash_file(winner["checkpoint_path"]),
        "selection_rule": (
            f"lexicographic({primary} {'asc' if primary in _LOWER_IS_BETTER else 'desc'}, "
            f"{secondary} {'asc' if secondary in _LOWER_IS_BETTER else 'desc'}, step asc); "
            f"offline validation only — no simulator signal"
        ),
        "offline_validation_history": history,
        "training_seed": train_config.get("training_seed"),
        "named_generator_seeds": train_config.get("named_generator_seeds"),
        "git_commit": current_git_commit(),
    }
    torch.save(ckpt, selected_path)

    report = {
        "selected_step": winner["step"],
        "selected_verifier_path": str(selected_path),
        "selected_verifier_hash": hash_file(selected_path),
        "primary_metric": primary,
        "secondary_metric": secondary,
        "winner_metrics": winner["metrics"],
        "history_length": len(history),
    }
    save_json(report, run_dir / "verifier_selection.json")
    return report

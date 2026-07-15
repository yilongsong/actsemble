"""Stage 2 — policy confirmation and final selection (§6).

Candidate set = (top-5 screening success rates) ∪ (every checkpoint within
0.10 absolute of the screening best); all candidates re-evaluated on the
confirmation panel; selection is lexicographic:

    1. highest confirmation success rate;
    2. highest screening success rate;
    3. earliest training step.

The selected checkpoint is re-saved as ``selected_policy.pt`` with the
complete screening + confirmation history and audit fields (§17) embedded
in its metadata. Confirmation never sees final-test results, verifier
scores, or Actsemble system performance.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..evaluation.panels import Panel
from ..utils.hashing import hash_file
from ..utils.repo import current_git_commit
from ..utils.serialization import load_json, save_json


def build_candidate_set(
    screening_history: list[dict], *, top_k: int = 5, within_of_best: float = 0.10
) -> list[dict]:
    """§6: top-k by screening success ∪ all within ``within_of_best`` of the
    best. Ties in screening rate favor the earlier checkpoint."""
    if not screening_history:
        raise ValueError("Empty screening history; run screening first")
    ranked = sorted(screening_history, key=lambda h: (-h["success_rate"], h["step"]))
    best_rate = ranked[0]["success_rate"]
    selected_steps = {h["step"] for h in ranked[:top_k]}
    selected_steps |= {
        h["step"] for h in ranked if h["success_rate"] >= best_rate - within_of_best
    }
    return sorted(
        (h for h in screening_history if h["step"] in selected_steps),
        key=lambda h: h["step"],
    )


def select_policy(candidates_with_confirmation: list[dict]) -> dict:
    """Lexicographic §6 rule over records that carry both screening and
    confirmation success rates."""
    return sorted(
        candidates_with_confirmation,
        key=lambda c: (
            -c["confirmation"]["success_rate"],
            -c["screening"]["success_rate"],
            c["step"],
        ),
    )[0]


def confirm_and_select(
    run_dir: str | Path,
    confirmation_panel: Panel,
    *,
    env,
    device: str = "cuda",
    max_steps: int = 100,
    top_k: int = 5,
    within_of_best: float = 0.10,
    force: bool = False,
) -> dict:
    from .screening import evaluate_checkpoint_on_panel  # lazy: pulls sim deps

    run_dir = Path(run_dir)
    selected_path = run_dir / "selected_policy.pt"
    report_path = run_dir / "confirmation.json"
    if selected_path.exists() and not force:
        raise FileExistsError(
            f"{selected_path} already exists (protocol §17: pass force or use a "
            f"new experiment version)"
        )

    history = load_json(run_dir / "screening" / "screening_history.json")
    candidates = build_candidate_set(history, top_k=top_k, within_of_best=within_of_best)
    print(f"[confirm] {len(candidates)} candidate checkpoint(s): "
          f"steps {[c['step'] for c in candidates]}")

    confirmed = []
    for cand in candidates:
        record = evaluate_checkpoint_on_panel(
            cand["checkpoint_path"], confirmation_panel, env=env, device=device,
            max_steps=max_steps,
        )
        confirmed.append({"step": cand["step"], "screening": cand, "confirmation": record})
        print(f"[confirm] step {cand['step']}: confirmation "
              f"{record['success_count']}/{record['num_episodes']} = "
              f"{record['success_rate']:.1%} (screening {cand['success_rate']:.1%})")

    winner = select_policy(confirmed)
    print(f"[confirm] SELECTED step {winner['step']} "
          f"(confirmation {winner['confirmation']['success_rate']:.1%})")

    # Re-save the winning checkpoint as the canonical frozen policy with the
    # full §17 audit trail embedded in its metadata.
    ckpt = torch.load(winner["confirmation"]["checkpoint_path"], map_location="cpu",
                      weights_only=False)
    train_cfg_path = run_dir / "train_config.json"
    train_config = load_json(train_cfg_path) if train_cfg_path.exists() else {}
    ckpt["meta"].setdefault("extra", {})["selection"] = {
        "protocol": "actsemble_checkpoint_selection_v1",
        "selected_step": winner["step"],
        "original_checkpoint_path": winner["confirmation"]["checkpoint_path"],
        "original_checkpoint_hash": winner["confirmation"]["checkpoint_hash"],
        "selection_rule": "lexicographic(confirmation_rate desc, screening_rate desc, step asc)",
        "candidate_rule": {"top_k": top_k, "within_of_best": within_of_best},
        "screening_panel": history and candidates[0].get("panel"),
        "screening_history": history,
        "confirmation_panel": confirmation_panel.to_dict(),
        "confirmation_history": [
            {"step": c["step"],
             "success_rate": c["confirmation"]["success_rate"],
             "success_count": c["confirmation"]["success_count"],
             "wilson_ci": c["confirmation"]["wilson_ci"],
             "checkpoint_hash": c["confirmation"]["checkpoint_hash"]}
            for c in confirmed
        ],
        "training_seed": train_config.get("training_seed"),
        "training_budget": (train_config.get("policy_config", {}) or {}).get("training", {}),
        "named_generator_seeds": train_config.get("named_generator_seeds"),
        "git_commit": current_git_commit(),
    }
    torch.save(ckpt, selected_path)

    report = {
        "selected_step": winner["step"],
        "selected_policy_path": str(selected_path),
        "selected_policy_hash": hash_file(selected_path),
        "original_checkpoint_hash": winner["confirmation"]["checkpoint_hash"],
        "candidates": confirmed,
        "confirmation_panel": confirmation_panel.to_dict(),
    }
    save_json(report, report_path)
    return report

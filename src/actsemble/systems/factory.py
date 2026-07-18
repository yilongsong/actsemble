"""Build autonomy systems from YAML system configs + loaded checkpoints.

Enforces the fairness safeguards before assembly:
* the policy is used frozen (inference only);
* components must match the policy's dataset/split/normalization hashes
  and horizons (``require_same_dataset_hash``, default true).
"""

from __future__ import annotations

from .candidate_reranking import CandidateRerankingActsemble
from .consensus_selection import (
    SELECTOR_TYPES,
    ConsensusSelectionSystem,
    build_selector,
)
from .interface import ReplanningSystemBase, check_same_data
from .multisample_control import MultiSampleControlSystem
from .standalone import StandaloneDiffusionSystem
from .temporal_ensemble import TemporalEnsembleSystem
from .verifier_ensemble import MeanScoreRerankingActsemble

SYSTEM_TYPES = (
    "candidate_zero",
    "uniform_random",
    "first_candidate",
    "highest_component_score",
    "mean_component_score",
    "temporal_ensemble",
    *SELECTOR_TYPES,
)


def build_system(
    system_cfg: dict,
    policy,
    components: list,
    *,
    candidate_root_seed: int = 0,
) -> ReplanningSystemBase:
    """Build a system and apply the execution-window offset (``execution.action_offset``,
    e.g. H_o-1 for the Diffusion-Policy window alignment; default 0)."""
    system = _build_system_impl(
        system_cfg, policy, components, candidate_root_seed=candidate_root_seed
    )
    # Default the execution offset from the policy's training window alignment so
    # any system using a Diffusion-Policy-aligned checkpoint executes the action
    # for time t (chunk index H_o-1), not a past-aligned action, even when the
    # system config does not mention it. An explicit execution.action_offset wins.
    align = (getattr(policy.meta, "extra", None) or {}).get(
        "window_alignment", "future_only"
    )
    default_offset = (policy.meta.obs_horizon - 1) if align == "diffusion_policy" else 0
    offset = int(system_cfg.get("execution", {}).get("action_offset", default_offset))
    if offset:
        system.set_execution_offset(offset)
    return system


def _build_system_impl(
    system_cfg: dict,
    policy,
    components: list,
    *,
    candidate_root_seed: int = 0,
) -> ReplanningSystemBase:
    policy_cfg = system_cfg.get("policy", {})
    selection = system_cfg.get("selection", {})
    sel_type = selection.get("type", "candidate_zero")
    num_candidates = int(policy_cfg.get("num_candidates", 1))
    action_horizon = system_cfg.get("execution", {}).get("action_horizon")
    if not policy_cfg.get("frozen", True):
        raise ValueError(
            "Actsemble systems require frozen policies (policy.frozen must be true)"
        )

    require_same = bool(selection.get("require_same_dataset_hash", True))
    check_same_data(policy, components, require_same_dataset_hash=require_same)

    if sel_type == "candidate_zero":
        if components:
            raise ValueError("standalone system takes no components")
        # num_candidates > 1 is paired-comparison mode (protocol §11): the
        # standalone system samples the shared K-candidate tensor and
        # executes candidate zero, so its action is bitwise-identical to
        # candidate zero of the control and Actsemble systems.
        return StandaloneDiffusionSystem(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
    if sel_type == "temporal_ensemble":
        # A4 temporal-execution variant: replans at a cadence into a control-time
        # cache, emits an aggregate of the overlapping predictions. Selection over
        # the shared candidate tensor (K defaults to 1 = candidate zero); no
        # components. Aggregation/decay/window are the emission knobs.
        if components:
            raise ValueError("temporal_ensemble takes no components")
        execution = system_cfg.get("execution", {})
        return TemporalEnsembleSystem(
            policy,
            num_candidates=num_candidates,
            aggregation=str(selection.get("aggregation", "mean")),
            decay=float(selection.get("decay", 0.1)),
            recency=str(selection.get("recency", "recent")),
            window=selection.get("window"),
            replan_interval=int(execution.get("replan_interval", 1)),
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
    if sel_type in ("uniform_random", "first_candidate"):
        if components:
            raise ValueError("multi-sample control takes no components")
        return MultiSampleControlSystem(
            policy,
            num_candidates=num_candidates,
            selection_rule=sel_type,
            selection_seed=int(selection.get("selection_seed", 7)),
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
    if sel_type in SELECTOR_TYPES:
        # non-learned consensus selectors: no components, selection over the
        # shared candidate tensor only.
        if components:
            raise ValueError(f"consensus selector {sel_type!r} takes no components")
        return ConsensusSelectionSystem(
            policy,
            build_selector(sel_type, selection),
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
            early_weight_decay=float(selection.get("early_weight_decay", 0.25)),
            diagnostic_mode=bool(selection.get("diagnostic_mode", False)),
        )
    if sel_type == "mean_component_score":
        if len(components) < 1:
            raise ValueError("mean_component_score needs at least one component")
        return MeanScoreRerankingActsemble(
            policy,
            components,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
    if sel_type == "highest_component_score":
        if len(components) != 1:
            raise ValueError(
                f"highest_component_score needs exactly one component, got {len(components)}"
            )
        return CandidateRerankingActsemble(
            policy,
            components[0],
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
    raise ValueError(
        f"Unknown selection.type: {sel_type!r}; expected one of {SYSTEM_TYPES}"
    )

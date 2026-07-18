"""System 1: standalone diffusion policy (the scientific baseline).

Selects candidate zero — the base pipeline's default Select when no Score
stage is present, so this system needs no seam override. With
num_candidates=1 (default) that is one reproducibly sampled chunk per
replan. In paired-comparison mode (protocol §11) num_candidates is the
shared K, so the executed action is bitwise-identical to candidate zero of
the control/Actsemble systems.
"""

from __future__ import annotations

from .interface import ReplanningSystemBase


class StandaloneDiffusionSystem(ReplanningSystemBase):
    name = "standalone_diffusion"

    def __init__(
        self,
        policy,
        *,
        num_candidates: int = 1,
        action_horizon=None,
        candidate_root_seed: int = 0,
    ):
        super().__init__(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )

    # Selection = the base-pipeline default (candidate zero); no override.

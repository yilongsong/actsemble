"""System 2: multi-sample control (compute-matched sampling control).

Samples the same K candidates as the Actsemble system from the same frozen
policy, then selects WITHOUT any learned component: uniformly at random
(seeded) or always the first candidate.

This is NOT claimed to be a strong selection method. It exists to separate
"benefit from sampling more policy outputs" from "benefit from the learned
component".
"""

from __future__ import annotations

import numpy as np
import torch

from ..seed import derive_seed
from .interface import ReplanningSystemBase


class MultiSampleControlSystem(ReplanningSystemBase):
    name = "multisample_control"

    def __init__(
        self,
        policy,
        *,
        num_candidates: int,
        selection_rule: str = "uniform_random",
        selection_seed: int = 7,
        action_horizon=None,
        candidate_root_seed: int = 0,
    ):
        super().__init__(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
        if selection_rule not in ("uniform_random", "first_candidate"):
            raise ValueError(f"Unknown selection rule: {selection_rule}")
        self.selection_rule = selection_rule
        self.selection_seed = int(selection_seed)

    def _select(self, candidates: torch.Tensor, scores, valid: torch.Tensor,
                ctx, record: dict) -> int:
        if self.selection_rule == "first_candidate":
            return 0
        rng = np.random.default_rng(
            derive_seed(self.selection_seed, "select", self.episode_seed, ctx.replan_index)
        )
        return int(rng.integers(self.num_candidates))

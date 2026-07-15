"""System 1: standalone diffusion policy (the scientific baseline).

Selects candidate zero. With num_candidates=1 (default) that is one
reproducibly sampled chunk per replan. In paired-comparison mode
(protocol §11) num_candidates is the shared K, so the executed action is
bitwise-identical to candidate zero of the control/Actsemble systems.
"""

from __future__ import annotations

import numpy as np
import torch

from .interface import ReplanningSystemBase


class StandaloneDiffusionSystem(ReplanningSystemBase):
    name = "standalone_diffusion"

    def __init__(
        self, policy, *, num_candidates: int = 1, action_horizon=None, candidate_root_seed: int = 0
    ):
        super().__init__(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )

    def _select(self, candidates: torch.Tensor, valid: torch.Tensor, history: np.ndarray, record: dict) -> int:
        return 0

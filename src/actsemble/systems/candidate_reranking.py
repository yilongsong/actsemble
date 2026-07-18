"""System 3: Actsemble candidate reranking.

Exact same frozen policy checkpoint and candidate sets as the multi-sample
control; the ONLY difference is selection: a separately trained same-data
compatibility component scores every candidate and the highest-scoring
valid chunk is executed.

Fallback contract:
* all component scores invalid  -> request candidate 0 (recorded);
* component raises              -> request candidate 0 (recorded);
* if candidate 0 is non-finite  -> shared layer selects first finite candidate;
* no finite policy candidate    -> raise instead of executing invalid actions;
* an episode is never terminated because the component failed.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..components.interface import ActionChunkCompatibilityScorer
from .interface import ReplanningSystemBase, synchronize_device


class CandidateRerankingActsemble(ReplanningSystemBase):
    name = "candidate_reranking_actsemble"

    def __init__(
        self,
        policy,
        component: ActionChunkCompatibilityScorer,
        *,
        num_candidates: int,
        action_horizon=None,
        candidate_root_seed: int = 0,
    ):
        super().__init__(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
        self.component = component

    def components(self) -> list:
        return [self.component]

    def _select(
        self, candidates: torch.Tensor, scores, valid: torch.Tensor, ctx, record: dict
    ) -> int:
        synchronize_device(getattr(self.component, "device", torch.device("cpu")))
        t0 = time.perf_counter()
        try:
            scores = self.component.score(
                torch.from_numpy(np.ascontiguousarray(ctx.observation_history)),
                candidates,
            )
            synchronize_device(scores.device)
            record["component_latency_s"] = time.perf_counter() - t0
            if scores.shape != (self.num_candidates,):
                raise RuntimeError(f"Component returned shape {tuple(scores.shape)}")
            scores = scores.detach().cpu()
            record["component_scores"] = scores.tolist()
            mask = torch.isfinite(scores) & valid.cpu()
            if not mask.any():
                record["fallback"] = True
                record["fallback_reason"] = "no_valid_scores"
                return 0
            masked = scores.clone()
            masked[~mask] = -float("inf")
            return int(torch.argmax(masked).item())
        except Exception as exc:  # never kill the episode because scoring failed
            record["component_latency_s"] = time.perf_counter() - t0
            record["fallback"] = True
            record["fallback_reason"] = (
                f"component_exception: {type(exc).__name__}: {exc}"
            )
            return 0

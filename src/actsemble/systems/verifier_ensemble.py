"""Verifier ensemble: mean-score reranking over several same-data verifiers.

A non-drastic "push" on the learned selector — it trains nothing new. It reuses
the existing frozen verifiers (each an ActionChunkCompatibility trained on the
same fixed dataset with a different seed), averages their scores over the shared
candidate set, and executes the highest-mean-scoring candidate. Tests whether
the learned selection signal strengthens when the same-data verifiers are
ensembled, versus a single verifier (Actsemble) or none (candidate zero).

Selection is the ONLY behavioural difference from the other paired systems; the
fallback contract matches Actsemble (any scoring failure / no finite scores ->
candidate zero, recorded)."""

from __future__ import annotations

import time

import numpy as np
import torch

from .interface import ReplanningSystemBase


class MeanScoreRerankingActsemble(ReplanningSystemBase):
    name = "verifier_ensemble_mean"

    def __init__(self, policy, components, *, num_candidates: int,
                 action_horizon=None, candidate_root_seed: int = 0):
        super().__init__(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
        if not components:
            raise ValueError("verifier ensemble needs at least one component")
        self._components = list(components)

    def components(self) -> list:
        return self._components

    def _select(self, candidates: torch.Tensor, scores, valid: torch.Tensor,
                ctx, record: dict) -> int:
        t0 = time.perf_counter()
        try:
            hist_t = torch.from_numpy(np.ascontiguousarray(ctx.observation_history))
            per_verifier = []
            for c in self._components:
                s = c.score(hist_t, candidates)
                if s.shape != (self.num_candidates,):
                    raise RuntimeError(f"component returned shape {tuple(s.shape)}")
                per_verifier.append(s.detach().cpu())
            stacked = torch.stack(per_verifier)  # [n_verifiers, K]
            scores = stacked.mean(dim=0)
            record["component_latency_s"] = time.perf_counter() - t0
            record["component_scores"] = scores.tolist()
            record["num_verifiers"] = len(self._components)
            mask = torch.isfinite(scores) & valid.cpu()
            if not mask.any():
                record["fallback"] = True
                record["fallback_reason"] = "no_valid_scores"
                return 0
            masked = scores.clone()
            masked[~mask] = -float("inf")
            return int(torch.argmax(masked).item())
        except Exception as exc:  # never end the episode because scoring failed
            record["component_latency_s"] = time.perf_counter() - t0
            record["fallback"] = True
            record["fallback_reason"] = f"component_exception: {type(exc).__name__}: {exc}"
            return 0

"""Autonomy-system interface and shared replanning machinery.

An AutonomySystem wraps a FROZEN ActionChunkPolicy (plus optional
same-data components) behind a per-step ``act`` API. All three Phase 0
systems share the same receding-horizon loop and candidate sampler; they
differ ONLY in how a candidate chunk is selected.

Candidate-set identity (protocol §11): the candidate generator for a
replan is seeded from exactly (episode policy-sampling seed, policy
checkpoint hash, replanning index) — ``candidate_root_seed`` must be set
to the episode's policy-sampling seed before ``reset``. Systems with the
same frozen checkpoint, same K, and same seeds therefore sample
bitwise-identical candidate tensors — the selection rule is the only
difference. Every replan records a SHA-256 digest of its candidate tensor
so comparisons can verify identity after the fact (mismatch invalidates a
paired comparison). Verified by tests/test_system_candidate_identity.py.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..policies.interface import ActionChunkPolicy
from ..seed import derive_seed
from ..types import RobotAction, StateObservation
from .context import DecisionContext


def candidate_tensor_hash(candidates: torch.Tensor) -> str:
    """Stable digest of a candidate tensor (float32 bytes, C order)."""
    arr = np.ascontiguousarray(candidates.detach().cpu().numpy().astype(np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


@runtime_checkable
class AutonomySystem(Protocol):
    name: str

    def reset(self, *, episode_seed: int) -> None: ...

    def act(self, observation: StateObservation) -> RobotAction: ...

    def diagnostics(self) -> dict: ...


class ReplanningSystemBase:
    """Receding-horizon execution: sample K candidates, select, execute H_a."""

    name = "replanning_base"

    def __init__(
        self,
        policy: ActionChunkPolicy,
        *,
        num_candidates: int,
        action_horizon: int | None = None,
        candidate_root_seed: int = 0,
    ):
        self.policy = policy
        self.num_candidates = int(num_candidates)
        meta = policy.meta  # DiffusionPolicy exposes PolicyMeta
        self.obs_horizon = meta.obs_horizon
        self.prediction_horizon = meta.prediction_horizon
        self.action_horizon = int(action_horizon or meta.action_horizon)
        if not 1 <= self.action_horizon <= self.prediction_horizon:
            raise ValueError(
                f"action_horizon {self.action_horizon} must be in [1, {self.prediction_horizon}]"
            )
        self.include_previous_action = meta.include_previous_action
        self.candidate_root_seed = int(candidate_root_seed)
        self._action_dim = int(meta.action_dim)
        self._reset_state(episode_seed=0)

    # -- AutonomySystem ------------------------------------------------------
    def reset(self, *, episode_seed: int) -> None:
        self.policy.reset()
        for comp in self.components():
            comp.reset()
        self._reset_state(episode_seed=episode_seed)

    def act(self, observation: StateObservation) -> RobotAction:
        frame = self._frame(observation)
        self._history.append(frame)
        if not self._queue:
            self._replan()
        return RobotAction(value=self._queue.popleft())

    def diagnostics(self) -> dict:
        replans = self._replan_records
        n = max(1, len(replans))
        selection_changes = sum(r["selected_index"] != 0 for r in replans)
        return {
            "system": self.name,
            "num_candidates": self.num_candidates,
            "action_horizon": self.action_horizon,
            "num_replans": len(replans),
            "fallback_count": sum(r.get("fallback", False) for r in replans),
            "fallback_rate": sum(r.get("fallback", False) for r in replans) / n,
            "mean_policy_latency_s": float(
                np.mean([r["policy_latency_s"] for r in replans]) if replans else 0.0
            ),
            "mean_component_latency_s": float(
                np.mean([r.get("component_latency_s", 0.0) for r in replans]) if replans else 0.0
            ),
            "selected_indices": [r["selected_index"] for r in replans],
            "selection_change_count": int(selection_changes),
            "selection_change_rate": selection_changes / n,
            "candidate_hashes": [r.get("candidate_hash", "") for r in replans],
            "replans": replans,
        }

    # -- shared machinery ------------------------------------------------------
    def components(self) -> list:
        return []

    def _reset_state(self, *, episode_seed: int) -> None:
        self.episode_seed = int(episode_seed)
        self._history: deque[np.ndarray] = deque(maxlen=self.obs_horizon)
        self._queue: deque[np.ndarray] = deque()
        self._executed: list[np.ndarray] = []  # committed actions (past coherence — A5)
        self._replan_index = 0
        self._replan_records: list[dict] = []

    def _frame(self, observation: StateObservation) -> np.ndarray:
        state = np.asarray(observation.state, dtype=np.float32).reshape(-1)
        if self.include_previous_action:
            prev = np.asarray(observation.previous_action, dtype=np.float32).reshape(-1)
            return np.concatenate([state, prev])
        return state

    def _observation_history(self) -> np.ndarray:
        frames = list(self._history)
        while len(frames) < self.obs_horizon:  # left-pad by repeating oldest
            frames.insert(0, frames[0])
        return np.stack(frames, axis=0)

    def _candidate_generator(self) -> torch.Generator:
        # Protocol §11: seed from exactly (episode policy-sampling seed,
        # policy checkpoint hash, replanning index). candidate_root_seed is
        # the per-episode policy-sampling seed.
        seed = derive_seed(
            self.candidate_root_seed,
            "candidates",
            self.policy.checkpoint_hash,
            self._replan_index,
        )
        return self.policy.new_generator(seed)

    def _context(self) -> DecisionContext:
        """The shared per-decision context every stage receives
        (systems/context.py; docs/system_architecture.md §2.2)."""
        executed = (
            np.stack(self._executed, axis=0)
            if self._executed
            else np.zeros((0, self._action_dim), dtype=np.float32)
        )
        return DecisionContext(
            observation_history=self._observation_history(),
            executed_actions=executed,
            replan_index=self._replan_index,
            policy=self.policy,
            components=self.components(),
        )

    def _replan(self) -> None:
        # Base-pipeline stages (docs/system_architecture.md §2): Propose ->
        # Predict -> Score -> Select -> Schedule. Subclasses override any seam;
        # the defaults reproduce candidate-zero. §11 candidate identity lives in
        # Propose, so any Score/Select swap keeps the K-tensor bitwise-identical.
        ctx = self._context()
        record: dict = {"replan_index": self._replan_index, "episode_seed": self.episode_seed}
        candidates = self._propose(ctx, record)
        valid = torch.isfinite(candidates).all(dim=(1, 2))
        record["num_valid_candidates"] = int(valid.sum())
        preds = self._predict(candidates, ctx)
        scores = self._score(candidates, preds, valid, ctx, record)
        selected = int(self._select(candidates, scores, valid, ctx, record))
        record["selected_index"] = selected
        self._replan_records.append(record)
        h_a = int(self._schedule(candidates, selected, ctx))
        chunk = candidates[selected].detach().cpu().numpy().astype(np.float32)
        for a in chunk[:h_a]:
            self._queue.append(a.copy())
            self._executed.append(a.copy())
        self._replan_index += 1

    # -- stages: override any seam; defaults reproduce candidate-zero ----------
    def _propose(self, ctx: DecisionContext, record: dict) -> torch.Tensor:
        """Propose: sample the shared K-candidate tensor; record its §11 hash and
        compact per-candidate diagnostics (mean |a|, mean |Δa|)."""
        t0 = time.perf_counter()
        candidates = self.policy.sample_action_chunks(
            ctx.observation_history,
            num_samples=self.num_candidates,
            generator=self._candidate_generator(),
        )
        record["policy_latency_s"] = time.perf_counter() - t0
        if candidates.shape != (
            self.num_candidates,
            self.prediction_horizon,
            candidates.shape[-1],
        ):
            raise RuntimeError(f"Bad candidate shape {tuple(candidates.shape)}")
        with torch.no_grad():
            record["candidate_mean_abs"] = candidates.abs().mean(dim=(1, 2)).cpu().tolist()
            record["candidate_smoothness"] = (
                (candidates[:, 1:] - candidates[:, :-1]).abs().mean(dim=(1, 2)).cpu().tolist()
            )
        record["candidate_hash"] = candidate_tensor_hash(candidates)
        return candidates

    def _predict(self, candidates: torch.Tensor, ctx: DecisionContext):
        """Predict (optional, Track C): forecast candidate consequences. Default none."""
        return None

    def _score(self, candidates: torch.Tensor, preds, valid: torch.Tensor,
               ctx: DecisionContext, record: dict):
        """Score (optional): per-candidate scalar ``[K]`` (higher = better).
        Return None to let Select fall through to candidate-zero."""
        return None

    def _select(self, candidates: torch.Tensor, scores, valid: torch.Tensor,
                ctx: DecisionContext, record: dict) -> int:
        """Select: choose the executed candidate index. Default = fallback-aware
        argmax over ``scores`` (candidate-zero when scores are absent or all
        invalid). A Score-then-argmax system need only implement ``_score``."""
        if scores is None:
            return 0
        s = scores.detach().cpu()
        mask = torch.isfinite(s) & valid.detach().cpu()
        if not mask.any():
            record["fallback"] = True
            record["fallback_reason"] = "no_valid_scores"
            return 0
        masked = s.clone()
        masked[~mask] = float("-inf")
        return int(torch.argmax(masked).item())

    def _schedule(self, candidates: torch.Tensor, selected: int, ctx: DecisionContext) -> int:
        """Schedule: how many actions of the selected chunk to execute before
        replanning. Default = the fixed action horizon."""
        return self.action_horizon


def check_same_data(policy, components: list, *, require_same_dataset_hash: bool = True) -> None:
    """Fairness safeguard: reject assembling models from different data."""
    if not require_same_dataset_hash:
        return
    pm = policy.meta
    for comp in components:
        cm = comp.meta
        problems = []
        if comp.dataset_hash != policy.dataset_hash:
            problems.append(
                f"dataset_hash: policy {policy.dataset_hash[:12]} != component {comp.dataset_hash[:12]}"
            )
        if cm.get("split_hash") != pm.split_hash:
            problems.append("split_hash differs")
        if cm.get("normalization") != pm.normalization:
            problems.append("normalization statistics differ")
        for key in ("obs_horizon", "prediction_horizon", "include_previous_action",
                    "state_dim", "action_dim"):
            if cm.get(key) != getattr(pm, key):
                problems.append(f"{key}: policy {getattr(pm, key)} != component {cm.get(key)}")
        if problems:
            raise ValueError(
                "Component/policy same-data contract violated:\n  - " + "\n  - ".join(problems)
            )

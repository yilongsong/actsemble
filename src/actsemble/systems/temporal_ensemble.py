"""A4 — temporal ensembling (a Tier-1 temporal-execution variant).

Ordinary receding-horizon execution samples a chunk, executes ``H_a`` of it
open-loop, discards, and replans. Temporal ensembling instead predicts a chunk
every ``replan_interval`` control steps and keeps the recent chunks in a
control-time-indexed cache: at each step, several past chunks predict the
current action, and the emitted action is a temporally-weighted aggregate of
those overlapping predictions (Zhao et al., ACT). This is the generalized-queue
case of the base pipeline (docs/system_architecture.md §2.3) — one policy, one
stage-chain, action out of this chain, so it is still **Tier 1**. It reuses the
base Propose/Predict/Score/Select seams for each cached plan; with the default
``K=1`` each plan is candidate zero, so every aggregation variant caches
bitwise-identical plans and differs ONLY in the emission rule (a clean one-knob
comparison, exactly like a Score/Select swap).

Aggregation modes (the ``emission`` knob):
* ``latest``     — the freshest prediction only; no ensembling. This is the
                   compute-matched control (replan-every-step, execute-first)
                   that isolates dense replanning from the aggregation itself.
* ``mean``       — a recency-weighted average (freshest prediction gets the most
                   weight). This is the ACT-style temporal-ensemble *mechanism*,
                   but note the weighting convention differs from the original ACT
                   code, which is near-uniform and OLDEST-weighted (``exp(-m*i)``,
                   i=0 the oldest, small m). Freshest-weighting is the charitable
                   variant (it trusts the most recent observation most); averaging
                   continuous actions is unsafe across distinct action modes.
* ``projection`` — the weighted mean projected onto the nearest real prediction.
                   Multimodal-safe: it always emits an action the policy actually
                   produced, never a blend of two modes.
* ``medoid``     — the weighted medoid of the overlapping predictions. Also
                   multimodal-safe (no averaging), robust to an outlier chunk.

Weights follow an EWMA in the prediction age ``a`` (0 = freshest): ``w(a) =
exp(-decay * a)``, renormalized. ``decay = 0`` is a uniform average;
``decay -> inf`` collapses ``mean`` toward ``latest``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..types import RobotAction
from .interface import ReplanningSystemBase

AGGREGATIONS = ("latest", "mean", "projection", "medoid")
RECENCY = (
    "recent",
    "oldest",
)  # weight direction: freshest-highest vs oldest-highest (ACT)


def aggregate_predictions(
    preds: np.ndarray,
    ages: np.ndarray,
    *,
    decay: float,
    mode: str,
    recency: str = "recent",
) -> np.ndarray:
    """Combine the overlapping predictions for one control step into one action.

    ``preds`` is ``[n, A]`` (n >= 1 predictions for the current control-time,
    freshest first) and ``ages`` is ``[n]`` (0 = predicted this step). Returns
    the ``[A]`` action to execute (float32). Distances are in raw action space;
    ties break to the freshest prediction (lowest row index, which is freshest).

    ``recency`` sets the weight direction: ``"recent"`` (freshest prediction
    highest, ``exp(-decay*age)`` — the reactive/charitable variant) or
    ``"oldest"`` (oldest prediction highest, ``exp(+decay*age)`` — the ORIGINAL
    ACT convention `w_i=exp(-m*i)`, i=0 oldest; a low-pass/smoothing filter).
    """
    preds = np.asarray(preds, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    if preds.ndim != 2 or preds.shape[0] < 1:
        raise ValueError(f"preds must be [n>=1, A], got {preds.shape}")
    if ages.shape != (preds.shape[0],):
        raise ValueError(f"ages shape {ages.shape} != expected {(preds.shape[0],)}")
    if not np.isfinite(preds).all() or not np.isfinite(ages).all():
        raise ValueError("preds and ages must be finite")
    if decay < 0:
        raise ValueError(f"decay must be nonnegative, got {decay}")
    if recency not in RECENCY:
        raise ValueError(f"recency must be one of {RECENCY}, got {recency!r}")
    if mode == "latest":
        return preds[int(np.argmin(ages))].astype(np.float32)
    sign = -1.0 if recency == "recent" else 1.0  # oldest (ACT) flips the decay sign
    logits = sign * float(decay) * ages
    logits -= np.max(logits)
    w = np.exp(logits)
    total = w.sum()
    w = w / total if total > 0 else np.full_like(w, 1.0 / len(w))
    mean = (w[:, None] * preds).sum(axis=0)  # [A]
    if mode == "mean":
        return mean.astype(np.float32)
    if mode == "projection":
        dist = np.linalg.norm(preds - mean[None, :], axis=1)  # [n]
        return preds[int(np.argmin(dist))].astype(np.float32)
    if mode == "medoid":
        # weighted medoid: the prediction minimizing sum_j w_j ||P_i - P_j||
        pair = np.linalg.norm(preds[:, None, :] - preds[None, :, :], axis=2)  # [n, n]
        cost = (pair * w[None, :]).sum(axis=1)  # [n]
        return preds[int(np.argmin(cost))].astype(np.float32)
    raise ValueError(
        f"unknown temporal aggregation mode {mode!r}; expected {AGGREGATIONS}"
    )


class TemporalEnsembleSystem(ReplanningSystemBase):
    """Temporal-execution variant of the base pipeline (A4). Overrides the
    execution/emission model (a control-time cache + an aggregation rule) while
    reusing the base decision stages via ``_decide``."""

    AGGREGATIONS = AGGREGATIONS

    def __init__(
        self,
        policy,
        *,
        num_candidates: int = 1,
        aggregation: str = "mean",
        decay: float = 0.1,
        recency: str = "recent",
        window: int | None = None,
        replan_interval: int = 1,
        action_horizon=None,
        candidate_root_seed: int = 0,
    ):
        super().__init__(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
            )
        if recency not in RECENCY:
            raise ValueError(f"recency must be one of {RECENCY}, got {recency!r}")
        self.aggregation = aggregation
        self.recency = recency
        self.name = f"temporal_{aggregation}" + (
            "" if recency == "recent" else f"_{recency}"
        )
        self.decay = float(decay)
        if self.decay < 0:
            raise ValueError("decay (EWMA rate) must be >= 0")
        # a plan predicts H_p steps ahead, so the ensemble window can't exceed it
        self.window = self.prediction_horizon if window is None else int(window)
        self.window = max(1, min(self.window, self.prediction_horizon))
        self.replan_interval = int(replan_interval)
        if self.replan_interval < 1:
            raise ValueError("replan_interval must be >= 1")

    # -- execution model: control-time cache + aggregate emission ---------------
    def _reset_state(self, *, episode_seed: int) -> None:
        super()._reset_state(episode_seed=episode_seed)
        self._control_step = 0
        # recent plans, oldest first: each (origin_control_step, chunk[H_p, A])
        self._plan_cache: deque[tuple[int, np.ndarray]] = deque()
        self._ensemble_sizes: list[int] = []

    def act(self, observation) -> RobotAction:
        self._history.append(self._frame(observation))
        t = self._control_step
        if t % self.replan_interval == 0 or not self._covers(t):
            self._cache_plan()
        action = self._emit(t)
        self._executed.append(action.copy())
        self._control_step += 1
        return RobotAction(value=action)

    def _covers(self, t: int) -> bool:
        off = self.execution_offset
        return any(
            0 <= t - origin < self.window
            and off + (t - origin) < self.prediction_horizon
            for origin, _ in self._plan_cache
        )

    def _cache_plan(self) -> None:
        ctx = self._context()
        record: dict = {
            "replan_index": self._replan_index,
            "episode_seed": self.episode_seed,
            "control_step": self._control_step,
        }
        candidates, selected, _h_a = self._decide(
            ctx, record
        )  # cadence, not H_a, drives replans
        chunk = candidates[selected].detach().cpu().numpy().astype(np.float32)
        self._plan_cache.append((self._control_step, chunk))
        oldest_covering = self._control_step - self.window + 1
        while self._plan_cache and self._plan_cache[0][0] < oldest_covering:
            self._plan_cache.popleft()
        self._replan_index += 1

    def _emit(self, t: int) -> np.ndarray:
        off = self.execution_offset
        preds, ages = [], []
        for origin, chunk in reversed(self._plan_cache):  # freshest first
            age = t - origin
            idx = (
                off + age
            )  # DP-aligned chunks put the action for the origin step at index off
            if 0 <= age < self.window and idx < self.prediction_horizon:
                preds.append(chunk[idx])
                ages.append(age)
        self._ensemble_sizes.append(len(preds))
        return aggregate_predictions(
            np.stack(preds),
            np.asarray(ages),
            decay=self.decay,
            mode=self.aggregation,
            recency=self.recency,
        )

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        sizes = self._ensemble_sizes
        d.update(
            {
                "aggregation": self.aggregation,
                "recency": self.recency,
                "decay": self.decay,
                "window": self.window,
                "replan_interval": self.replan_interval,
                "num_control_steps": self._control_step,
                "mean_ensemble_size": float(np.mean(sizes)) if sizes else 0.0,
                "max_ensemble_size": int(np.max(sizes)) if sizes else 0,
            }
        )
        return d

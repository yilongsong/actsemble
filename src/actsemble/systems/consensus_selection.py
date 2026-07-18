"""Non-learned consensus selectors over K frozen-policy action-chunk samples.

These systems reuse :class:`ReplanningSystemBase` unchanged — candidate
sampling, observation-history management, execution-horizon queueing,
deterministic candidate seeds, validity checking, candidate hashing, reset,
and latency accounting all stay identical to the standalone / control /
Actsemble systems. The ONLY thing implemented here is *selection over the
candidate tensor*: given the same ``[K, prediction_horizon, action_dim]``
tensor that every paired system receives, pick the index of one real
candidate (never an average) by a deterministic self-consistency rule.

Scientific role: controls between "execute candidate zero" and the learned
action-chunk verifier — do the gains from sampling K chunks come from generic
consensus, without any learned component?

All distances are squared-Euclidean over the **normalized** action chunk
(policy/checkpoint action normalization), so the rules stay scale-valid for
future tasks with differently-scaled action dimensions. Selection math runs on
CPU in float64 for exact determinism; the raw candidate tensor is never
mutated and the returned action is the original raw candidate.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .interface import ReplanningSystemBase

SELECTOR_TYPES = (
    "full_chunk_medoid",
    "early_weighted_medoid",
    "coordinate_median_projection",
    "largest_cluster_medoid",
)


# --------------------------------------------------------------------------- #
# Distance machinery
# --------------------------------------------------------------------------- #
def timestep_weights(prediction_horizon: int, decay: float) -> np.ndarray:
    """w_tau = exp(-decay * tau), normalized so sum_tau w_tau = 1."""
    tau = np.arange(int(prediction_horizon), dtype=np.float64)
    w = np.exp(-float(decay) * tau)
    return w / w.sum()


class ChunkDistance:
    """Squared-Euclidean distance over normalized, flattened action chunks,
    with optional per-timestep weighting.

    d(A_i, A_j) = sum_tau w_tau * sum_d (A_i[tau, d] - A_j[tau, d])^2

    Operates on flattened arrays ``[K, prediction_horizon * action_dim]`` so a
    single weighted reduction covers both the full-chunk (uniform weights) and
    early-weighted variants.
    """

    def __init__(
        self,
        prediction_horizon: int,
        action_dim: int,
        per_timestep_weights: np.ndarray | None = None,
    ):
        self.prediction_horizon = int(prediction_horizon)
        self.action_dim = int(action_dim)
        if per_timestep_weights is None:
            self.feature_weights = None
        else:
            w = np.asarray(per_timestep_weights, dtype=np.float64)
            # repeat each timestep weight across that timestep's action dims
            self.feature_weights = np.repeat(w, self.action_dim)

    def matrix(self, flat: np.ndarray) -> np.ndarray:
        """Full [K, K] weighted squared-distance matrix (symmetric, zero diag)."""
        diff = flat[:, None, :] - flat[None, :, :]
        sq = diff * diff
        if self.feature_weights is not None:
            sq = sq * self.feature_weights
        return sq.sum(axis=-1)

    def to_point(self, flat: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Weighted squared distance [K] from each row of ``flat`` to ``point``."""
        diff = flat - point[None, :]
        sq = diff * diff
        if self.feature_weights is not None:
            sq = sq * self.feature_weights
        return sq.sum(axis=-1)


# --------------------------------------------------------------------------- #
# Selectors — each returns a GLOBAL candidate index and writes diagnostics
# --------------------------------------------------------------------------- #
class ConsensusSelector:
    """Base: selects over normalized flattened candidates ``flat`` [K, D],
    restricted to ``valid_idx`` (global indices of valid candidates, len >= 2).
    Writes selector-specific diagnostics into ``record`` and returns the chosen
    global index. Implementations must be deterministic and must break ties by
    lowest candidate index."""

    name = "consensus"

    def select(
        self,
        flat: np.ndarray,
        valid_idx: np.ndarray,
        dist: ChunkDistance,
        record: dict,
        *,
        diag_mode: bool = False,
    ) -> int:
        raise NotImplementedError


def _medoid_from_matrix(D: np.ndarray, members: np.ndarray) -> tuple[int, np.ndarray]:
    """Medoid over ``members`` (global indices) given full distance matrix D:
    the member minimizing the summed distance to the other members. Returns
    (global_index, per_member_score). Lowest-index tie-break (argmin is stable
    and ``members`` is ascending)."""
    sub = D[np.ix_(members, members)]
    scores = sub.sum(axis=1)
    winner = members[int(np.argmin(scores))]
    return int(winner), scores


class FullChunkMedoidSelector(ConsensusSelector):
    name = "full_chunk_medoid"

    def select(self, flat, valid_idx, dist, record, *, diag_mode=False) -> int:
        D = dist.matrix(flat)
        winner, scores = _medoid_from_matrix(D, valid_idx)
        record["medoid_scores"] = {int(i): float(s) for i, s in zip(valid_idx, scores)}
        record["selected_medoid_score"] = float(scores[np.argmin(scores)])
        if diag_mode:
            record["pairwise_distance_matrix"] = D[
                np.ix_(valid_idx, valid_idx)
            ].tolist()
        return winner


class EarlyWeightedMedoidSelector(ConsensusSelector):
    name = "early_weighted_medoid"

    def __init__(self, decay: float):
        self.decay = float(decay)

    def select(self, flat, valid_idx, dist, record, *, diag_mode=False) -> int:
        # ``dist`` already carries the early-weighting for this selector.
        D = dist.matrix(flat)
        winner, scores = _medoid_from_matrix(D, valid_idx)
        record["medoid_scores"] = {int(i): float(s) for i, s in zip(valid_idx, scores)}
        record["selected_medoid_score"] = float(scores[np.argmin(scores)])
        record["early_weight_decay"] = self.decay
        record["timestep_weights"] = dist.feature_weights.reshape(
            dist.prediction_horizon, dist.action_dim
        )[:, 0].tolist()
        if diag_mode:
            record["pairwise_distance_matrix"] = D[
                np.ix_(valid_idx, valid_idx)
            ].tolist()
        return winner


class CoordinateMedianProjectionSelector(ConsensusSelector):
    name = "coordinate_median_projection"

    def select(self, flat, valid_idx, dist, record, *, diag_mode=False) -> int:
        median = np.median(flat[valid_idx], axis=0)  # coordinate-wise, over valid
        d_to_median = dist.to_point(flat, median)
        # argmin over valid candidates (ascending order -> lowest-index tie-break)
        winner = int(valid_idx[int(np.argmin(d_to_median[valid_idx]))])
        record["distance_to_median"] = {
            int(i): float(d_to_median[i]) for i in valid_idx
        }
        record["selected_distance_to_median"] = float(d_to_median[winner])
        return winner


class LargestClusterMedoidSelector(ConsensusSelector):
    name = "largest_cluster_medoid"

    def __init__(self, max_iterations: int = 20):
        self.max_iterations = int(max_iterations)

    def select(self, flat, valid_idx, dist, record, *, diag_mode=False) -> int:
        D = dist.matrix(flat)
        m = valid_idx
        # all-identical guard: every valid pairwise distance ~0 -> candidate 0
        if np.max(D[np.ix_(m, m)]) == 0.0:
            record["cluster_note"] = "all_candidates_identical"
            record["cluster_assignment"] = {int(i): 0 for i in m}
            record["cluster_sizes"] = [int(len(m))]
            record["selected_cluster"] = 0
            record["selected_cluster_medoid"] = 0 if 0 in m else int(m[0])
            record["within_cluster_mean_distances"] = [0.0]
            record["clustering_iterations"] = 0
            return 0 if 0 in m else int(m[0])

        # deterministic two-cluster k-medoids init:
        # medoid 1 = candidate 0 (or lowest valid index if 0 invalid),
        # medoid 2 = valid candidate farthest from medoid 1.
        med1 = 0 if 0 in m else int(m[0])
        med2 = int(m[int(np.argmax(D[med1, m]))])
        assign = None
        iters = 0
        for iters in range(1, self.max_iterations + 1):
            # assignment: nearest of {med1, med2}; ties -> med1 (lower is med1
            # by construction only when equal distance, deterministic)
            d1, d2 = D[m, med1], D[m, med2]
            new_assign = np.where(d1 <= d2, 0, 1)
            if assign is not None and np.array_equal(new_assign, assign):
                break
            assign = new_assign
            c0, c1 = m[assign == 0], m[assign == 1]
            # empty-cluster guard: keep old medoid if a cluster vanished
            med1 = _medoid_from_matrix(D, c0)[0] if len(c0) else med1
            med2 = _medoid_from_matrix(D, c1)[0] if len(c1) else med2
        if assign is None:  # pragma: no cover - loop always runs once
            assign = np.zeros(len(m), dtype=int)

        clusters = {0: m[assign == 0], 1: m[assign == 1]}
        sizes = {c: len(idx) for c, idx in clusters.items()}
        within_mean = {}
        for c, idx in clusters.items():
            if len(idx) <= 1:
                within_mean[c] = 0.0
            else:
                sub = D[np.ix_(idx, idx)]
                within_mean[c] = float(sub.sum() / (len(idx) * (len(idx) - 1)))
        # pick cluster: largest size; tie -> lower mean within-cluster distance;
        # tie -> cluster containing the lowest candidate index.
        nonempty = [c for c in (0, 1) if sizes[c] > 0]
        min_idx = {c: int(min(clusters[c])) if sizes[c] else np.inf for c in (0, 1)}
        sel = min(nonempty, key=lambda c: (-sizes[c], within_mean[c], min_idx[c]))
        medoid = _medoid_from_matrix(D, clusters[sel])[0]

        record["cluster_assignment"] = {int(i): int(a) for i, a in zip(m, assign)}
        record["cluster_sizes"] = [int(sizes[0]), int(sizes[1])]
        record["selected_cluster"] = int(sel)
        record["selected_cluster_medoid"] = int(medoid)
        record["within_cluster_mean_distances"] = [within_mean[0], within_mean[1]]
        record["clustering_iterations"] = int(iters)
        if diag_mode:
            record["pairwise_distance_matrix"] = D[np.ix_(m, m)].tolist()
        return medoid


def build_selector(sel_type: str, selection_cfg: dict) -> ConsensusSelector:
    if sel_type == "full_chunk_medoid":
        return FullChunkMedoidSelector()
    if sel_type == "early_weighted_medoid":
        return EarlyWeightedMedoidSelector(
            decay=float(selection_cfg.get("early_weight_decay", 0.25))
        )
    if sel_type == "coordinate_median_projection":
        return CoordinateMedianProjectionSelector()
    if sel_type == "largest_cluster_medoid":
        clu = selection_cfg.get("clustering", {}) or {}
        return LargestClusterMedoidSelector(
            max_iterations=int(clu.get("max_iterations", 20))
        )
    raise ValueError(
        f"Unknown consensus selector type: {sel_type!r}; "
        f"expected one of {SELECTOR_TYPES}"
    )


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
class ConsensusSelectionSystem(ReplanningSystemBase):
    """Receding-horizon system whose selection is a non-learned consensus rule.

    Uses NO learned components. ``diagnostic_mode`` additionally logs the full
    per-replan pairwise distance matrix (off by default for large evals)."""

    def __init__(
        self,
        policy,
        selector: ConsensusSelector,
        *,
        num_candidates: int,
        action_horizon=None,
        candidate_root_seed: int = 0,
        early_weight_decay: float = 0.25,
        diagnostic_mode: bool = False,
    ):
        super().__init__(
            policy,
            num_candidates=num_candidates,
            action_horizon=action_horizon,
            candidate_root_seed=candidate_root_seed,
        )
        self.selector = selector
        self.name = f"consensus_{selector.name}"
        self.diagnostic_mode = bool(diagnostic_mode)
        self.normalizer = policy.normalizer
        # early-weighted medoid uses per-timestep weights in its distance
        if isinstance(selector, EarlyWeightedMedoidSelector):
            w = timestep_weights(self.prediction_horizon, selector.decay)
        else:
            w = None
        self._dist = ChunkDistance(self.prediction_horizon, policy.meta.action_dim, w)

    def _select(
        self, candidates: torch.Tensor, scores, valid: torch.Tensor, ctx, record: dict
    ) -> int:
        t0 = time.perf_counter()
        record["selector_type"] = self.selector.name
        valid_np = valid.detach().cpu().numpy().astype(bool)
        valid_idx = np.nonzero(valid_np)[0]
        n_valid = int(valid_idx.size)

        if n_valid == 0:
            record["fallback"] = True
            record["fallback_reason"] = "no_valid_candidates"
            record["changed_from_candidate_zero"] = False
            record["selector_latency_s"] = time.perf_counter() - t0
            return 0
        if n_valid == 1:
            idx = int(valid_idx[0])
            if idx != 0:  # candidate 0 was invalid; the one valid candidate wins
                record["fallback"] = True
                record["fallback_reason"] = "single_valid_candidate"
            record["changed_from_candidate_zero"] = idx != 0
            record["selector_latency_s"] = time.perf_counter() - t0
            return idx

        # normalize on CPU in float64 for exact, GPU-independent determinism
        norm = self.normalizer.normalize_action(
            candidates.detach().cpu().to(torch.float64)
        ).numpy()
        flat = np.ascontiguousarray(norm.reshape(norm.shape[0], -1))
        # zero invalid rows so distance math never touches NaN/inf; selection
        # only ever indexes valid candidates, so this cannot change any result.
        flat[~valid_np] = 0.0

        idx = self.selector.select(
            flat, valid_idx, self._dist, record, diag_mode=self.diagnostic_mode
        )

        # common consensus diagnostics (over valid candidates only)
        full = ChunkDistance(self.prediction_horizon, self._dist.action_dim)
        Dv = full.matrix(flat[valid_idx])
        offdiag = Dv[~np.eye(len(valid_idx), dtype=bool)]
        record["mean_pairwise_distance"] = (
            float(offdiag.mean()) if offdiag.size else 0.0
        )
        record["max_pairwise_distance"] = float(offdiag.max()) if offdiag.size else 0.0
        if valid_np[0]:
            pos0 = int(np.where(valid_idx == 0)[0][0])
            pos_sel = int(np.where(valid_idx == idx)[0][0])
            record["distance_to_candidate_zero"] = float(Dv[pos_sel, pos0])
        else:
            record["distance_to_candidate_zero"] = None
        record["changed_from_candidate_zero"] = idx != 0
        record["selector_latency_s"] = time.perf_counter() - t0
        return idx

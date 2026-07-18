"""Unit tests for the non-learned consensus selectors (handcrafted candidates,
no policy training / GPU needed). Exercises the full ConsensusSelectionSystem
._select path: normalization, invalid handling, tie-breaking, determinism.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from actsemble.systems.consensus_selection import (
    ConsensusSelectionSystem,
    build_selector,
)
from actsemble.systems.context import DecisionContext

SELECTORS = [
    "full_chunk_medoid",
    "early_weighted_medoid",
    "coordinate_median_projection",
    "largest_cluster_medoid",
]


class _Meta:
    def __init__(self, action_dim, prediction_horizon):
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        self.obs_horizon = 2
        self.action_horizon = prediction_horizon
        self.include_previous_action = False


class _IdentityNorm:
    def normalize_action(self, x):
        return x


class _ScaleNorm:
    """Divide each action dim by its scale (a diagonal normalizer)."""

    def __init__(self, scale):
        self.scale = scale

    def normalize_action(self, x):
        s = torch.as_tensor(self.scale, dtype=x.dtype, device=x.device)
        return x / s


class _FakePolicy:
    def __init__(self, action_dim, prediction_horizon, normalizer=None):
        self.meta = _Meta(action_dim, prediction_horizon)
        self.normalizer = normalizer or _IdentityNorm()


def make_system(sel_type, action_dim, prediction_horizon, *, normalizer=None,
                decay=0.25, num_candidates=16):
    policy = _FakePolicy(action_dim, prediction_horizon, normalizer)
    selector = build_selector(sel_type, {"early_weight_decay": decay})
    return ConsensusSelectionSystem(
        policy, selector, num_candidates=num_candidates, early_weight_decay=decay
    )


def run(system, candidates):
    cand = torch.as_tensor(candidates, dtype=torch.float32)
    valid = torch.isfinite(cand).all(dim=(1, 2))
    record = {}
    ctx = DecisionContext(
        observation_history=np.zeros((2, 1), dtype=np.float32),
        executed_actions=np.zeros((0, cand.shape[-1]), dtype=np.float32),
        replan_index=0, policy=system.policy,
    )
    idx = system._select(cand, None, valid, ctx, record)
    return idx, record, cand


def chunks(vals):
    """vals: list of [H_p, A] arrays -> [K, H_p, A]."""
    return np.asarray(vals, dtype=np.float32)


# 1. K == 1 -----------------------------------------------------------------
@pytest.mark.parametrize("sel", SELECTORS)
def test_k_equals_one(sel):
    system = make_system(sel, 6, 16, num_candidates=1)
    cands = chunks([np.full((16, 6), 0.3)])
    idx, rec, _ = run(system, cands)
    assert idx == 0


# 2. identical candidates ---------------------------------------------------
@pytest.mark.parametrize("sel", SELECTORS)
def test_identical_candidates(sel):
    system = make_system(sel, 6, 16)
    cands = chunks([np.full((16, 6), 0.42)] * 5)
    idx, rec, _ = run(system, cands)
    assert idx == 0


# 3. obvious medoid (central group, not outlier) ----------------------------
def test_obvious_medoid_full_chunk():
    system = make_system("full_chunk_medoid", 1, 1)
    cands = chunks([[[0.0]], [[0.1]], [[0.2]], [[5.0]]])
    idx, rec, _ = run(system, cands)
    # scores: 0->25.05, 1->24.03, 2->23.09, 3->72.05  => medoid index 2
    assert idx == 2
    assert idx != 3  # never the far outlier


# 4. multimodal: largest-cluster medoid picks from the bigger group ---------
def test_largest_cluster_prefers_bigger_group():
    system = make_system("largest_cluster_medoid", 1, 1)
    # 3 candidates near 0.0, 2 near 10.0 -> larger cluster is {0,1,2}
    cands = chunks([[[0.0]], [[0.05]], [[-0.05]], [[10.0]], [[10.1]]])
    idx, rec, _ = run(system, cands)
    assert idx in (0, 1, 2)
    assert rec["cluster_sizes"] in ([3, 2], [2, 3])
    assert max(rec["cluster_sizes"]) == 3


# 5. cluster-size tie -> lowest-candidate-index cluster ---------------------
def test_cluster_size_tie_breaks_to_lowest_index():
    system = make_system("largest_cluster_medoid", 1, 1)
    # two equal-size, equal-spread clusters -> tie -> cluster with candidate 0
    cands = chunks([[[0.0]], [[0.1]], [[10.0]], [[10.1]]])
    idx, rec, _ = run(system, cands)
    assert idx in (0, 1)  # the cluster containing candidate 0


# 6. coordinate median projection (median not itself a candidate) -----------
def test_coordinate_median_projection():
    system = make_system("coordinate_median_projection", 2, 1)
    cands = chunks([[[0.0, 0.0]], [[2.0, 1.0]], [[1.0, 2.0]]])
    idx, rec, _ = run(system, cands)
    # coordinate median = (1,1) (not a candidate); dist: c0=2, c1=1, c2=1
    # -> tie c1/c2 -> lowest index 1
    assert idx == 1


# 7. early weighting changes the medoid in the expected direction ----------
def test_early_weighting_changes_selection():
    # H_p=2, A=1. early = [1, 0, 2] -> early medoid = c0
    #            late  = [0, 3, 3] -> late/uniform medoid = c1
    cands = chunks([[[1.0], [0.0]], [[0.0], [3.0]], [[2.0], [3.0]]])
    low = make_system("early_weighted_medoid", 1, 2, decay=0.01)
    high = make_system("early_weighted_medoid", 1, 2, decay=5.0)
    idx_low, _, _ = run(low, cands)
    idx_high, rec_high, _ = run(high, cands)
    assert idx_low == 1       # late actions dominate under ~uniform weighting
    assert idx_high == 0      # early actions dominate under strong decay
    assert idx_low != idx_high
    assert abs(sum(rec_high["timestep_weights"]) - 1.0) < 1e-9


# 8. invalid candidates -----------------------------------------------------
@pytest.mark.parametrize("sel", SELECTORS)
def test_one_nan_candidate_excluded(sel):
    system = make_system(sel, 1, 1)
    cands = chunks([[[0.0]], [[0.1]], [[0.2]], [[5.0]]])
    cands[3, 0, 0] = np.nan  # outlier becomes invalid
    idx, rec, _ = run(system, cands)
    assert idx in (0, 1, 2)  # never the invalid one


@pytest.mark.parametrize("sel", SELECTORS)
def test_one_inf_candidate_excluded(sel):
    system = make_system(sel, 1, 1)
    cands = chunks([[[0.0]], [[0.1]], [[0.2]], [[5.0]]])
    cands[3, 0, 0] = np.inf
    idx, rec, _ = run(system, cands)
    assert idx != 3


@pytest.mark.parametrize("sel", SELECTORS)
def test_all_invalid_falls_back_to_candidate_zero(sel):
    system = make_system(sel, 1, 1)
    cands = chunks([[[np.nan]], [[np.inf]], [[np.nan]]])
    idx, rec, _ = run(system, cands)
    assert idx == 0
    assert rec["fallback"] is True
    assert rec["fallback_reason"] == "no_valid_candidates"


@pytest.mark.parametrize("sel", SELECTORS)
def test_only_one_valid_candidate(sel):
    system = make_system(sel, 1, 1)
    cands = chunks([[[np.nan]], [[0.7]], [[np.inf]]])
    idx, rec, _ = run(system, cands)
    assert idx == 1
    assert rec["fallback"] is True
    assert rec["fallback_reason"] == "single_valid_candidate"


# 9. tie-breaking -> lowest valid index -------------------------------------
@pytest.mark.parametrize("sel", SELECTORS)
def test_exact_tie_returns_lowest_index(sel):
    system = make_system(sel, 2, 1)
    # symmetric pair around origin: exact tie -> index 0
    cands = chunks([[[1.0, 0.0]], [[-1.0, 0.0]]])
    idx, rec, _ = run(system, cands)
    assert idx == 0


# 10. determinism -----------------------------------------------------------
@pytest.mark.parametrize("sel", SELECTORS)
def test_determinism(sel):
    system = make_system(sel, 6, 16)
    rng = np.random.default_rng(0)
    cands = chunks(rng.standard_normal((16, 16, 6)))
    idx1, rec1, _ = run(system, cands)
    idx2, rec2, _ = run(system, cands)
    assert idx1 == idx2
    # exclude wall-clock latency, which is inherently non-deterministic
    rec1.pop("selector_latency_s", None)
    rec2.pop("selector_latency_s", None)
    assert rec1 == rec2  # bitwise-identical diagnostics


# 11. no mutation of the input candidate tensor -----------------------------
@pytest.mark.parametrize("sel", SELECTORS)
def test_no_mutation(sel):
    system = make_system(sel, 6, 16)
    rng = np.random.default_rng(1)
    cands = chunks(rng.standard_normal((16, 16, 6)))
    before = copy.deepcopy(cands)
    _, _, cand_tensor = run(system, cands)
    assert torch.equal(cand_tensor, torch.as_tensor(before))


# 12. normalization: selection is in normalized space; raw candidate intact -
def test_selection_uses_normalized_space():
    # raw medoid = c2; but scaling dim1 down by 1000 makes dim0 dominate -> c0
    cands = chunks([[[0.0, 0.0]], [[0.0, 5.0]], [[0.5, 2.5]]])
    ident = make_system("full_chunk_medoid", 2, 1, normalizer=_IdentityNorm())
    scaled = make_system("full_chunk_medoid", 2, 1, normalizer=_ScaleNorm([1.0, 1000.0]))
    idx_ident, _, cand_i = run(ident, cands)
    idx_scaled, _, cand_s = run(scaled, cands)
    assert idx_ident == 2   # raw-space medoid
    assert idx_scaled == 0  # normalized-space medoid differs
    # raw candidates returned unchanged in both cases
    assert torch.equal(cand_i, torch.as_tensor(cands))
    assert torch.equal(cand_s, torch.as_tensor(cands))


# 13. complexity sanity: K=64, H_p=16, A=6 completes quickly ---------------
@pytest.mark.parametrize("sel", SELECTORS)
def test_complexity_sanity_k64(sel):
    import time

    system = make_system(sel, 6, 16, num_candidates=64)
    rng = np.random.default_rng(2)
    cands = chunks(rng.standard_normal((64, 16, 6)))
    t0 = time.perf_counter()
    idx, rec, _ = run(system, cands)
    dt = time.perf_counter() - t0
    assert 0 <= idx < 64
    assert dt < 1.0  # generous ceiling; selection is milliseconds

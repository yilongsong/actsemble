"""Unit tests for the mean-score verifier ensemble (no training)."""

from __future__ import annotations

import numpy as np
import torch

from actsemble.systems.verifier_ensemble import MeanScoreRerankingActsemble


class _Meta:
    def __init__(self, action_dim=6, prediction_horizon=16):
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        self.obs_horizon = 2
        self.action_horizon = prediction_horizon
        self.include_previous_action = False


class _Policy:
    def __init__(self):
        self.meta = _Meta()


class _FakeVerifier:
    def __init__(self, scores, raise_it=False):
        self._scores = torch.as_tensor(scores, dtype=torch.float32)
        self._raise = raise_it

    def score(self, hist, candidates):
        if self._raise:
            raise RuntimeError("boom")
        return self._scores

    def reset(self):
        pass


def _run(components, K=4):
    system = MeanScoreRerankingActsemble(_Policy(), components, num_candidates=K)
    cand = torch.zeros(K, 16, 6)
    valid = torch.ones(K, dtype=torch.bool)
    rec = {}
    idx = system._select(cand, valid, np.zeros((2, 31), dtype=np.float32), rec)
    return idx, rec


def test_mean_of_two_verifiers_argmax():
    # v1=[0,3,0,0], v2=[0,1,2,0] -> mean=[0,2,1,0] -> argmax index 1
    idx, rec = _run([_FakeVerifier([0, 3, 0, 0]), _FakeVerifier([0, 1, 2, 0])])
    assert idx == 1
    assert rec["component_scores"] == [0.0, 2.0, 1.0, 0.0]
    assert rec["num_verifiers"] == 2


def test_tie_breaks_to_lowest_index():
    # mean = [0.5,0.5,0,0] -> tie between 0 and 1 -> lowest index 0
    idx, _ = _run([_FakeVerifier([0, 1, 0, 0]), _FakeVerifier([1, 0, 0, 0])])
    assert idx == 0


def test_component_exception_falls_back_to_candidate_zero():
    idx, rec = _run([_FakeVerifier([0, 0, 0, 0], raise_it=True)])
    assert idx == 0
    assert rec["fallback"] is True
    assert "component_exception" in rec["fallback_reason"]


def test_all_nonfinite_scores_fall_back():
    inf = float("inf")
    idx, rec = _run([_FakeVerifier([inf, inf, inf, inf])])
    assert idx == 0
    assert rec["fallback"] is True


def test_single_verifier_equals_argmax():
    idx, rec = _run([_FakeVerifier([1, 5, 2, 0])])
    assert idx == 1
    assert rec["num_verifiers"] == 1


def test_determinism():
    comps = [_FakeVerifier([0, 3, 1, 0]), _FakeVerifier([1, 2, 0, 0])]
    idx1, rec1 = _run(comps)
    idx2, rec2 = _run(comps)
    assert idx1 == idx2
    rec1.pop("component_latency_s", None)
    rec2.pop("component_latency_s", None)
    assert rec1 == rec2

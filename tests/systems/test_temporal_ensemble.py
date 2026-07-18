"""System subsystem: A4 temporal ensembling (no policy training/GPU needed).

Two layers: (1) the pure aggregation math (``aggregate_predictions``), and
(2) the system's control-time-cache execution model driven by a deterministic
fake policy — including the variant candidate-identity property (all aggregation
variants cache bitwise-identical plans at K=1, differing only in emission).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from actsemble.systems.factory import build_system
from actsemble.systems.temporal_ensemble import (
    AGGREGATIONS,
    TemporalEnsembleSystem,
    aggregate_predictions,
)
from actsemble.types import StateObservation


# ---- (1) aggregation math -------------------------------------------------
def test_latest_returns_freshest():
    preds = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])  # freshest first
    ages = np.array([0, 1, 2])
    out = aggregate_predictions(preds, ages, decay=0.5, mode="latest")
    assert np.allclose(out, [0.0, 0.0])  # age-0 row


def test_mean_zero_decay_is_uniform_average():
    preds = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    ages = np.array([0, 1, 2])
    out = aggregate_predictions(preds, ages, decay=0.0, mode="mean")
    assert np.allclose(out, [2.0, 0.0])


def test_mean_high_decay_approaches_latest():
    preds = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    ages = np.array([0, 1, 2])
    out = aggregate_predictions(preds, ages, decay=20.0, mode="mean")
    assert np.allclose(out, [0.0, 0.0], atol=1e-6)


def test_projection_snaps_to_nearest_real_prediction():
    preds = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    ages = np.array([0, 1, 2])
    # uniform mean = (5/3, 0) ~ 1.67 -> nearest real prediction is [1, 0]
    out = aggregate_predictions(preds, ages, decay=0.0, mode="projection")
    assert np.allclose(out, [1.0, 0.0])
    assert any(np.allclose(out, p) for p in preds)  # a real prediction


def test_recency_oldest_vs_recent_flips_weighting():
    preds = np.array(
        [[0.0], [10.0]]
    )  # row 0 = freshest (age 0), row 1 = oldest (age 1)
    ages = np.array([0, 1])
    recent = aggregate_predictions(
        preds, ages, decay=1.0, mode="mean", recency="recent"
    )
    oldest = aggregate_predictions(
        preds, ages, decay=1.0, mode="mean", recency="oldest"
    )
    # recent trusts the freshest (0) most -> pulled toward 0; oldest (ACT) trusts old (10) most
    assert recent[0] < 5.0 < oldest[0]


def test_oldest_weighting_is_stable_at_extreme_decay():
    preds = np.array([[0.0], [10.0]], dtype=np.float32)
    out = aggregate_predictions(
        preds, np.array([0, 100]), decay=100.0, mode="mean", recency="oldest"
    )
    assert np.isfinite(out).all()
    assert np.allclose(out, [10.0])


def test_medoid_picks_central_not_outlier():
    preds = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])
    ages = np.array([0, 1, 2])
    out = aggregate_predictions(preds, ages, decay=0.0, mode="medoid")
    assert np.allclose(out, [0.1, 0.0])  # central point, never the outlier


@pytest.mark.parametrize("mode", AGGREGATIONS)
def test_single_prediction_returns_it(mode):
    preds = np.array([[0.7, -0.3, 0.2]])
    out = aggregate_predictions(preds, np.array([0]), decay=0.1, mode=mode)
    assert np.allclose(out, [0.7, -0.3, 0.2])
    assert out.dtype == np.float32


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown temporal aggregation"):
        aggregate_predictions(np.zeros((1, 2)), np.array([0]), decay=0.1, mode="bogus")


# ---- deterministic fake policy for system-level tests ---------------------
class _Meta:
    obs_horizon = 2
    prediction_horizon = 16
    action_horizon = 8
    action_dim = 3
    state_dim = 3
    include_previous_action = False


class _FakePolicy:
    """Chunks depend only on the candidate generator seed -> reproducible."""

    def __init__(self):
        self.meta = _Meta()
        self.checkpoint_hash = "fakehash1234"

    def reset(self):
        pass

    def new_generator(self, seed):
        return np.random.default_rng(int(seed) % (2**32))

    def sample_action_chunks(self, obs_hist, *, num_samples, generator):
        arr = generator.standard_normal(
            (num_samples, self.meta.prediction_horizon, self.meta.action_dim)
        )
        return torch.as_tensor(arr, dtype=torch.float32)


def _obs_stream(n, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield StateObservation(
            state=rng.uniform(-1, 1, _Meta.state_dim).astype(np.float32),
            previous_action=np.zeros(_Meta.action_dim, np.float32),
            step_index=i,
        )


def _make(aggregation="mean", **kw):
    return TemporalEnsembleSystem(_FakePolicy(), aggregation=aggregation, **kw)


def _run(system, n, obs_seed=0, episode_seed=7):
    system.candidate_root_seed = 999
    system.reset(episode_seed=episode_seed)
    actions = []
    for obs in _obs_stream(n, seed=obs_seed):
        actions.append(system.act(obs).value.copy())
    return np.stack(actions)


# ---- (2) system execution model -------------------------------------------
@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_runs_and_bookkeeps(aggregation):
    system = _make(aggregation)
    n = 20
    actions = _run(system, n)
    d = system.diagnostics()
    assert actions.shape == (n, _Meta.action_dim)
    assert np.isfinite(actions).all()
    assert d["num_control_steps"] == n
    assert d["num_replans"] == n  # replan_interval=1 -> one plan per step
    assert len(system._executed) == n
    assert d["aggregation"] == aggregation
    # ensemble grows and caps at the window (== H_p = 16)
    assert d["max_ensemble_size"] == min(n, _Meta.prediction_horizon)
    assert d["mean_ensemble_size"] > 1.0


def test_name_encodes_aggregation():
    assert _make("mean").name == "temporal_mean"
    assert _make("projection").name == "temporal_projection"
    assert _make("latest").name == "temporal_latest"


def test_step_zero_matches_candidate_zero_across_variants():
    # first step has a single covering prediction -> every variant emits it
    a_mean = _run(_make("mean"), 1)
    a_latest = _run(_make("latest"), 1)
    a_medoid = _run(_make("medoid"), 1)
    assert np.allclose(a_mean[0], a_latest[0])
    assert np.allclose(a_mean[0], a_medoid[0])


def test_variants_cache_identical_plans_but_diverge_in_emission():
    """The §11-style property for the temporal family: at K=1 all aggregation
    variants replan on the same schedule with the same seeds, so their cached
    plans are bitwise-identical (equal candidate_hashes); only the emitted
    action differs."""
    mean = _make("mean")
    medoid = _make("medoid")
    a_mean = _run(mean, 20)
    a_medoid = _run(medoid, 20)
    hm = mean.diagnostics()["candidate_hashes"]
    hd = medoid.diagnostics()["candidate_hashes"]
    assert hm == hd and len(hm) == 20 and all(h for h in hm)
    assert np.allclose(a_mean[0], a_medoid[0])  # agree on step 0
    assert not np.allclose(a_mean, a_medoid)  # diverge once ensembled


def test_latest_equals_replan_every_step_first_action():
    """`latest` must emit the freshest chunk's first action every step."""
    system = _make("latest")
    actions = _run(system, 10)
    # rebuild the freshest chunk[0] at each replan from the same seeds
    from actsemble.seed import derive_seed

    pol = system.policy
    for t in range(10):
        seed = derive_seed(999, "candidates", pol.checkpoint_hash, t)
        chunk0 = pol.sample_action_chunks(
            None, num_samples=1, generator=pol.new_generator(seed)
        )[0, 0].numpy()
        assert np.allclose(actions[t], chunk0, atol=1e-6)


def test_determinism():
    a1 = _run(_make("medoid"), 15)
    a2 = _run(_make("medoid"), 15)
    assert np.array_equal(a1, a2)


def test_reset_clears_cache_and_control_step():
    system = _make("mean")
    _run(system, 5)
    assert system._control_step == 5 and len(system._plan_cache) > 0
    system.reset(episode_seed=2)
    assert system._control_step == 0
    assert len(system._plan_cache) == 0
    assert len(system._executed) == 0
    assert system.diagnostics()["num_replans"] == 0


def test_window_clamped_to_prediction_horizon():
    assert _make("mean", window=999).window == _Meta.prediction_horizon
    assert _make("mean", window=4).window == 4
    assert _make("mean", window=0).window == 1


def test_invalid_params_raise():
    with pytest.raises(ValueError, match="aggregation must be"):
        _make("nope")
    with pytest.raises(ValueError, match="decay"):
        _make("mean", decay=-1.0)
    with pytest.raises(ValueError, match="replan_interval"):
        _make("mean", replan_interval=0)


def test_replan_interval_sparser_cadence_still_covers():
    # replan every 4 steps: fewer plans, but every step is still covered/aggregated
    system = _make("mean", replan_interval=4)
    actions = _run(system, 12)
    d = system.diagnostics()
    assert np.isfinite(actions).all()
    assert d["num_replans"] == 3  # steps 0, 4, 8
    assert d["mean_ensemble_size"] >= 1.0


# ---- factory --------------------------------------------------------------
@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_factory_builds_each_variant(aggregation):
    system = build_system(
        {
            "policy": {"num_candidates": 1},
            "selection": {
                "type": "temporal_ensemble",
                "aggregation": aggregation,
                "decay": 0.2,
            },
            "execution": {"replan_interval": 1},
        },
        _FakePolicy(),
        [],
    )
    assert isinstance(system, TemporalEnsembleSystem)
    assert system.name == f"temporal_{aggregation}"
    assert system.decay == 0.2


def test_factory_rejects_components():
    # require_same_dataset_hash: False skips the same-data check so we reach the
    # temporal branch's own "takes no components" guard.
    with pytest.raises(ValueError, match="no components"):
        build_system(
            {
                "policy": {"num_candidates": 1},
                "selection": {
                    "type": "temporal_ensemble",
                    "aggregation": "mean",
                    "require_same_dataset_hash": False,
                },
            },
            _FakePolicy(),
            ["some_component"],
        )

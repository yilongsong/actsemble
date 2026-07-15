"""Paired seeds, metrics, comparability verification, exception recording."""

import numpy as np
import pytest

from actsemble.evaluation.metrics import (
    paired_bootstrap_diff,
    paired_outcome_counts,
    wilson_interval,
)
from actsemble.evaluation.paired_seeds import paired_seeds
from actsemble.evaluation.reports import compare_systems, verify_comparable


def test_paired_seeds_deterministic_and_distinct():
    a = paired_seeds(root_seed=100, num_episodes=10)
    b = paired_seeds(root_seed=100, num_episodes=10)
    assert a == b
    c = paired_seeds(root_seed=101, num_episodes=10)
    assert a != c
    env_seeds = [s.env_seed for s in a]
    assert len(set(env_seeds)) == 10
    # env / perturbation / candidate streams are independent
    assert a[0].env_seed != a[0].perturbation_seed != a[0].candidate_seed


def test_wilson_interval_properties():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi < 0.35
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo > 0.65
    lo, hi = wilson_interval(25, 50)
    assert lo < 0.5 < hi
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_paired_bootstrap_detects_direction():
    a = [True] * 40 + [False] * 10
    b = [True] * 25 + [False] * 25
    out = paired_bootstrap_diff(a, b, num_resamples=2000, seed=0)
    assert out["mean_difference"] == pytest.approx(0.3)
    assert out["ci_low"] > 0.0
    same = paired_bootstrap_diff(a, a, num_resamples=500, seed=0)
    assert same["mean_difference"] == 0.0


def test_paired_outcome_counts():
    a = [True, True, False, False]
    b = [True, False, True, False]
    c = paired_outcome_counts(a, b)
    assert c == {"a_wins": 1, "b_wins": 1, "both_succeed": 1, "both_fail": 1}


def _result(name, successes, **overrides):
    base = {
        "system_name": name,
        "task": "FakeTask-v1",
        "dataset_hash": "d" * 8,
        "policy_checkpoint_hash": "p" * 8,
        "policy_weights_kind": "ema",
        "controller": "fake_delta",
        "simulation_backend": "fake_cpu",
        "regime": "nominal",
        "primary_metric": "success_once",
        "max_steps": 50,
        "num_episodes": len(successes),
        "environment_seeds": list(range(len(successes))),
        "perturbation_seeds": list(range(100, 100 + len(successes))),
        "successes": successes,
        "success_count": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "latency": {"mean_decision_s": 0.01},
        "config": {"system": {"policy": {"num_candidates": 1, "type": "state_diffusion"}}},
    }
    base.update(overrides)
    return base


def test_verify_comparable_catches_mismatches():
    a = _result("standalone", [True, False, True])
    b = _result("actsemble", [True, True, True])
    assert verify_comparable([a, b]) == []
    for field, value in [
        ("dataset_hash", "other"),
        ("policy_checkpoint_hash", "other"),
        ("regime", "perturbed"),
        ("environment_seeds", [9, 9, 9]),
        ("controller", "other"),
        ("simulation_backend", "other"),
    ]:
        bad = _result("actsemble", [True, True, True], **{field: value})
        assert verify_comparable([a, bad]), f"{field} mismatch not caught"
    with pytest.raises(ValueError, match="not comparable"):
        compare_systems([a, _result("x", [True, True, True], dataset_hash="zzz")])


def test_compare_systems_report_content():
    a = _result("standalone", [True, False, False, True] * 15)
    b = _result("actsemble", [True, True, False, True] * 15)
    report = compare_systems([a, b])
    assert report["systems"]["standalone"]["success_rate"] == pytest.approx(0.5)
    pair = report["pairwise_vs_baseline"]["actsemble"]
    assert pair["absolute_difference"] == pytest.approx(0.25)
    assert pair["win_loss_tie"]["wins"] == 15
    assert pair["win_loss_tie"]["losses"] == 0
    assert not report["warnings"]  # n=60 >= 50
    small = compare_systems([_result("a", [True] * 3), _result("b", [False] * 3)])
    assert small["warnings"]


def test_rollout_records_simulator_exceptions():
    """run_episode must record env exceptions, not swallow or crash."""
    from actsemble.sim.rollout import run_episode

    class FakeSpace:
        low = np.full(3, -1.0, np.float32)
        high = np.full(3, 1.0, np.float32)

    class ExplodingEnv:
        class unwrapped:
            single_action_space = FakeSpace()

        def reset(self, seed=None):
            return np.zeros((1, 8), np.float32), {}

        def step(self, action):
            raise RuntimeError("physx exploded")

        def render(self):
            return np.zeros((1, 4, 4, 3), np.uint8)

    class DoNothingSystem:
        name = "noop"

        def reset(self, *, episode_seed):
            pass

        def act(self, obs):
            from actsemble.types import RobotAction

            return RobotAction(value=np.zeros(3, np.float32))

        def diagnostics(self):
            return {}

    result, _ = run_episode(
        ExplodingEnv(), DoNothingSystem(), episode_seed=0, max_steps=5
    )
    assert result.exception is not None and "physx exploded" in result.exception
    assert result.success_once is False
    assert result.num_steps == 0

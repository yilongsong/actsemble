"""Data subsystem: normalization contracts shared by policy and components."""

from __future__ import annotations

import numpy as np

from actsemble.data.normalization import (
    MINMAX,
    STANDARDIZE,
    NormalizationStats,
    Normalizer,
    compute_stats,
)


def test_standardize_roundtrip_and_zero_mean():
    stats = NormalizationStats(
        method=STANDARDIZE,
        state_mean=np.array([1.0, 2.0, 3.0], np.float32),
        state_std=np.array([2.0, 0.5, 1.0], np.float32),
        action_mean=np.array([0.0, -1.0], np.float32),
        action_std=np.array([1.0, 2.0], np.float32),
    )
    normalizer = Normalizer(stats)
    state = np.array([[1.0, 2.0, 3.0], [3.0, 2.5, 4.0]], np.float32)
    normalized_state = normalizer.normalize_state(state)
    assert np.allclose(normalized_state[0], [0, 0, 0], atol=1e-6)
    assert np.allclose(normalizer.unnormalize_state(normalized_state), state, atol=1e-5)
    action = np.array([[0.0, -1.0], [2.0, 3.0]], np.float32)
    assert np.allclose(
        normalizer.unnormalize_action(normalizer.normalize_action(action)),
        action,
        atol=1e-5,
    )


def test_standardize_dict_roundtrip():
    stats = NormalizationStats(
        method=STANDARDIZE,
        state_mean=np.zeros(2, np.float32),
        state_std=np.ones(2, np.float32),
        action_mean=np.zeros(2, np.float32),
        action_std=np.ones(2, np.float32),
    )
    serialized = stats.to_dict()
    assert serialized["method"] == STANDARDIZE and "state_mean" in serialized
    restored = NormalizationStats.from_dict(serialized)
    assert restored.method == STANDARDIZE
    assert np.allclose(restored.action_std, stats.action_std)


def test_minmax_is_still_the_default_and_backward_compatible():
    stats = NormalizationStats(
        state_min=-np.ones(2, np.float32),
        state_max=np.ones(2, np.float32),
        action_min=-np.ones(2, np.float32),
        action_max=np.ones(2, np.float32),
    )
    assert stats.method == MINMAX
    serialized = stats.to_dict()
    assert serialized["method"] == MINMAX and "state_min" in serialized
    NormalizationStats.from_dict(serialized)


def test_standardize_clips_near_constant_dims():
    scale, _ = Normalizer._standardize(
        np.array([0.0, 5.0], np.float32), np.array([0.001, 2.0], np.float32)
    )
    assert np.isclose(scale[0], 100.0)
    assert np.isclose(scale[1], 0.5)


def test_compute_stats_observation_only_toggle():
    class Episode:
        state = np.array([[0.0], [1.0], [2.0]], np.float32)
        next_state = np.array([[10.0], [11.0], [12.0]], np.float32)
        action = np.zeros((3, 1), np.float32)

    with_next = compute_stats([Episode()], method=STANDARDIZE, include_next_state=True)
    without_next = compute_stats(
        [Episode()], method=STANDARDIZE, include_next_state=False
    )
    assert np.allclose(without_next.state_mean, [1.0])
    assert not np.allclose(with_next.state_mean, without_next.state_mean)

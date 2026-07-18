"""Component subsystem: negative generation, bounds, behavior, and guards."""

import numpy as np
import pytest
import torch

from actsemble.components.action_chunk_compatibility import (
    ALL_NEGATIVE_TYPES,
    ActionChunkCompatibility,
    NegativeConfig,
    NegativeGenerator,
    window_negative_rng,
)
from actsemble.data.windows import extract_window


def _generator(episodes, **overrides):
    cfg = NegativeConfig.from_dict(
        {
            "position_dims": [0, 1],
            "rotation_dims": [2],
            "temporal_shift_min": 2,
            "temporal_shift_max": 6,
            "incompatible_min_distance": 16,
            **overrides,
        }
    )
    return NegativeGenerator(
        cfg,
        action_low=np.full(3, -1.0, np.float32),
        action_high=np.full(3, 1.0, np.float32),
        prediction_horizon=8,
        obs_horizon=2,
    )


def test_all_enabled_types_produce_valid_negatives(episodes):
    gen = _generator(episodes)
    w = extract_window(episodes[0], 10, obs_horizon=2, prediction_horizon=8)
    for ntype in gen.cfg.enabled_types:
        rng = np.random.default_rng(0)
        neg, name = gen.generate(
            episodes, 0, 10, w.action_chunk, rng, negative_type=ntype
        )
        assert name == ntype
        assert neg.shape == w.action_chunk.shape
        assert np.isfinite(neg).all()
        assert (neg >= -1.0 - 1e-6).all() and (neg <= 1.0 + 1e-6).all()
        assert not np.allclose(neg, w.action_chunk), (
            f"{ntype} produced the positive chunk"
        )


def test_negatives_deterministic(episodes):
    gen = _generator(episodes)
    w = extract_window(episodes[1], 5, obs_horizon=2, prediction_horizon=8)
    rng_a = window_negative_rng(42, "ep_00001", 5, 0)
    rng_b = window_negative_rng(42, "ep_00001", 5, 0)
    rng_c = window_negative_rng(42, "ep_00001", 5, 1)
    neg_a, type_a = gen.generate(episodes, 1, 5, w.action_chunk, rng_a)
    neg_b, type_b = gen.generate(episodes, 1, 5, w.action_chunk, rng_b)
    neg_c, _ = gen.generate(episodes, 1, 5, w.action_chunk, rng_c)
    np.testing.assert_array_equal(neg_a, neg_b)
    assert type_a == type_b
    assert not np.array_equal(neg_a, neg_c)


def test_other_trajectory_uses_different_episode(episodes):
    gen = _generator(episodes, enabled_types=["other_trajectory"])
    w = extract_window(episodes[0], 3, obs_horizon=2, prediction_horizon=8)
    rng = np.random.default_rng(1)
    neg, _ = gen.generate(episodes, 0, 3, w.action_chunk, rng)
    all_own = [
        extract_window(episodes[0], t, obs_horizon=2, prediction_horizon=8).action_chunk
        for t in range(len(episodes[0]))
    ]
    assert not any(np.allclose(neg, own) for own in all_own)


def test_temporal_shift_is_a_real_chunk_elsewhere(episodes):
    gen = _generator(episodes, enabled_types=["temporal_shift"])
    w = extract_window(episodes[2], 20, obs_horizon=2, prediction_horizon=8)
    rng = np.random.default_rng(3)
    neg, _ = gen.generate(episodes, 2, 20, w.action_chunk, rng)
    candidates = [
        extract_window(
            episodes[2], 20 + d, obs_horizon=2, prediction_horizon=8
        ).action_chunk
        for d in list(range(-6, -1)) + list(range(2, 7))
        if 0 <= 20 + d < len(episodes[2])
    ]
    assert any(np.allclose(neg, c) for c in candidates)


def test_negative_chunks_use_policy_window_alignment(episodes):
    gen = NegativeGenerator(
        NegativeConfig.from_dict({"enabled_types": ["temporal_shift"]}),
        action_low=np.full(3, -1.0, np.float32),
        action_high=np.full(3, 1.0, np.float32),
        prediction_horizon=8,
        obs_horizon=2,
        alignment="diffusion_policy",
    )
    t = 10
    chunk = gen._chunk_at(episodes, 0, t)
    expected = extract_window(
        episodes[0],
        t,
        obs_horizon=2,
        prediction_horizon=8,
        alignment="diffusion_policy",
    ).action_chunk
    np.testing.assert_array_equal(chunk, expected)


def test_gripper_mismatch_requires_gripper_dims():
    with pytest.raises(ValueError, match="gripper"):
        NegativeConfig.from_dict(
            {"enabled_types": ["gripper_mismatch"], "gripper_dims": []}
        )


def test_unknown_negative_type_rejected():
    with pytest.raises(ValueError, match="Unknown negative types"):
        NegativeConfig.from_dict({"enabled_types": ["not_a_type"]})


def test_component_score_shape_and_dataset_hash(trained_checkpoints):
    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    obs = torch.rand(2, int(comp.meta["state_dim"]))
    chunks = torch.rand(
        5, int(comp.meta["prediction_horizon"]), int(comp.meta["action_dim"])
    )
    scores = comp.score(obs, chunks)
    assert scores.shape == (5,)
    assert torch.isfinite(scores).all()
    assert comp.dataset_hash == trained_checkpoints["policy_summary"]["dataset_hash"]


def test_all_negative_types_registry_has_implementations(episodes):
    gen = _generator(
        episodes, enabled_types=list(ALL_NEGATIVE_TYPES[:-1])
    )  # sans gripper
    for ntype in ALL_NEGATIVE_TYPES:
        assert hasattr(gen, f"_neg_{ntype}"), f"missing implementation for {ntype}"

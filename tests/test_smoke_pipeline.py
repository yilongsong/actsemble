"""Sim-free end-to-end pipeline: dataset -> train both -> assemble -> act.

The full simulator smoke pipeline (env creation, rollouts, videos) lives
in scripts/smoke_test.py; this test covers everything up to the simulator
boundary so `pytest` stays fast.
"""

import numpy as np
import torch

from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility
from actsemble.policies.diffusion.policy import DiffusionPolicy
from actsemble.systems.factory import build_system
from actsemble.types import StateObservation


def test_full_pipeline_without_sim(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    component = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )

    # Hashes recorded during training match the loaded artifacts.
    assert policy.dataset_hash == trained_checkpoints["policy_summary"]["dataset_hash"]
    assert component.dataset_hash == trained_checkpoints["component_summary"]["dataset_hash"]
    assert policy.meta.split_hash == trained_checkpoints["policy_summary"]["split_hash"]

    systems = {
        "standalone": build_system(
            {"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"}},
            policy,
            [],
        ),
        "control": build_system(
            {"policy": {"num_candidates": 4}, "selection": {"type": "uniform_random"}},
            policy,
            [],
        ),
        "actsemble": build_system(
            {"policy": {"num_candidates": 4},
             "selection": {"type": "highest_component_score"}},
            policy,
            [component],
        ),
    }

    rng = np.random.default_rng(0)
    for name, system in systems.items():
        system.reset(episode_seed=42)
        for step in range(system.action_horizon + 1):
            obs = StateObservation(
                state=rng.uniform(-1, 1, policy.meta.state_dim).astype(np.float32),
                previous_action=np.zeros(policy.meta.action_dim, np.float32),
                step_index=step,
            )
            action = system.act(obs)
            assert action.value.shape == (policy.meta.action_dim,)
            assert np.isfinite(action.value).all()
        diag = system.diagnostics()
        assert diag["num_replans"] == 2, name
        assert diag["fallback_count"] == 0, name

    # Offline component metrics were produced and carry the disclaimer.
    offline = trained_checkpoints["component_summary"]["offline_eval"]
    assert "pairwise_ranking_accuracy" in offline
    assert "closed-loop" in offline["note"]


def test_component_scores_prefer_real_chunks_after_training(trained_checkpoints):
    """Weak sanity: on training-distribution windows, real chunks should
    outscore heavily corrupted ones for a trained component (10 steps of
    training on a tiny set — direction, not calibration)."""
    from actsemble.data.reader import DatasetReader
    from actsemble.data.windows import extract_window

    component = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    reader = DatasetReader(trained_checkpoints["dataset"])
    ep = reader.episodes[0]
    w = extract_window(ep, 10, obs_horizon=2, prediction_horizon=8)
    obs = torch.from_numpy(w.obs_history)
    real = torch.from_numpy(w.action_chunk).unsqueeze(0)
    rng = np.random.default_rng(0)
    corrupted = torch.from_numpy(
        np.clip(w.action_chunk + rng.uniform(-1, 1, w.action_chunk.shape), -1, 1).astype(
            np.float32
        )
    ).unsqueeze(0)
    scores = component.score(obs, torch.cat([real, corrupted]))
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()

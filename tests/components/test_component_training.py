"""Component subsystem: training-data alignment and validation metrics."""

from __future__ import annotations

import numpy as np
import torch

from actsemble.components.action_chunk_compatibility import (
    NegativeConfig,
    NegativeGenerator,
)
from actsemble.data.normalization import STANDARDIZE, NormalizationStats, Normalizer
from actsemble.training.train_component import (
    CompatibilityDataset,
    evaluate_compatibility,
)


class LastValueScore(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, -1]


def test_verifier_pairwise_metric_preserves_positive_negative_groups():
    batch = {
        "obs_history": torch.zeros(2, 1, 1),
        "positive_chunk": torch.tensor([[[5.0]], [[10.0]]]),
        "negative_chunks": torch.tensor([[[[4.0]], [[6.0]]], [[[9.0]], [[11.0]]]]),
        "negative_types": torch.zeros(2, 2, dtype=torch.long),
    }
    metrics = evaluate_compatibility(LastValueScore(), [batch], torch.device("cpu"))
    assert metrics["pairwise_ranking_accuracy"] == 0.5


def test_positive_chunk_respects_policy_alignment():
    class Episode:
        def __init__(self):
            self.state = np.arange(30, dtype=np.float32).reshape(10, 3)
            self.next_state = self.state
            self.previous_action = np.zeros((10, 2), np.float32)
            self.action = np.arange(20, dtype=np.float32).reshape(10, 2)
            self.episode_id = "e"

        def __len__(self):
            return len(self.state)

    episode = Episode()
    normalizer = Normalizer(
        NormalizationStats(
            method=STANDARDIZE,
            state_mean=np.zeros(3, np.float32),
            state_std=np.ones(3, np.float32),
            action_mean=np.zeros(2, np.float32),
            action_std=np.ones(2, np.float32),
        )
    )
    negatives = NegativeGenerator(
        NegativeConfig.from_dict({}),
        action_low=np.array([-99.0, -99.0]),
        action_high=np.array([99.0, 99.0]),
        prediction_horizon=8,
        obs_horizon=2,
    )
    dataset = CompatibilityDataset(
        [episode],
        normalizer,
        negatives,
        obs_horizon=2,
        prediction_horizon=8,
        include_previous_action=False,
        negatives_per_positive=1,
        negative_seed=0,
        alignment="diffusion_policy",
        action_horizon=4,
    )
    positive = dataset[3]["positive_chunk"].numpy()
    assert np.array_equal(positive[1], episode.action[3])
    assert np.array_equal(positive[0], episode.action[2])

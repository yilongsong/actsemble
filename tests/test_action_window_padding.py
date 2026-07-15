"""Observation-history and action-chunk window padding with explicit masks."""

import numpy as np

from actsemble.data.windows import enumerate_window_indices, extract_window


def test_interior_window_no_padding(episodes):
    ep = episodes[0]
    w = extract_window(ep, 10, obs_horizon=3, prediction_horizon=8)
    assert w.obs_history.shape == (3, ep.state.shape[1])
    assert w.action_chunk.shape == (8, ep.action.shape[1])
    assert w.obs_mask.all() and w.action_mask.all()
    np.testing.assert_array_equal(w.obs_history, ep.state[8:11])
    np.testing.assert_array_equal(w.action_chunk, ep.action[10:18])
    np.testing.assert_array_equal(w.prev_action_history, ep.previous_action[8:11])


def test_start_padding_replicates_first_state(episodes):
    ep = episodes[0]
    w = extract_window(ep, 0, obs_horizon=3, prediction_horizon=4)
    np.testing.assert_array_equal(w.obs_history[0], ep.state[0])
    np.testing.assert_array_equal(w.obs_history[1], ep.state[0])
    np.testing.assert_array_equal(w.obs_history[2], ep.state[0])
    assert list(w.obs_mask) == [False, False, True]


def test_end_padding_replicates_last_action(episodes):
    ep = episodes[0]
    T = len(ep)
    w = extract_window(ep, T - 2, obs_horizon=2, prediction_horizon=6)
    np.testing.assert_array_equal(w.action_chunk[0], ep.action[T - 2])
    np.testing.assert_array_equal(w.action_chunk[1], ep.action[T - 1])
    for i in range(2, 6):
        np.testing.assert_array_equal(w.action_chunk[i], ep.action[T - 1])
    assert list(w.action_mask) == [True, True, False, False, False, False]


def test_every_timestep_has_a_window(episodes):
    indices = enumerate_window_indices(episodes)
    assert len(indices) == sum(len(ep) for ep in episodes)
    ei, t = indices[0]
    assert (ei, t) == (0, 0)


def test_normalized_dataset_shapes(episodes):
    from actsemble.data.normalization import Normalizer, compute_stats
    from actsemble.data.torch_dataset import DiffusionWindowDataset

    ds = DiffusionWindowDataset(
        episodes,
        Normalizer(compute_stats(episodes)),
        obs_horizon=2,
        prediction_horizon=8,
    )
    item = ds[0]
    assert item["obs_history"].shape == (2, episodes[0].state.shape[1])
    assert item["action_chunk"].shape == (8, episodes[0].action.shape[1])
    assert item["action_chunk"].abs().max() <= 1.0 + 1e-6
    assert item["action_mask"].shape == (8,)


def test_include_previous_action_concatenates(episodes):
    from actsemble.data.normalization import Normalizer, compute_stats
    from actsemble.data.torch_dataset import DiffusionWindowDataset

    state_dim = episodes[0].state.shape[1]
    action_dim = episodes[0].action.shape[1]
    ds = DiffusionWindowDataset(
        episodes,
        Normalizer(compute_stats(episodes)),
        obs_horizon=2,
        prediction_horizon=8,
        include_previous_action=True,
    )
    assert ds[0]["obs_history"].shape == (2, state_dim + action_dim)

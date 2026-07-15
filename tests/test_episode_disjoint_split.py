"""Episode-disjoint train/validation splits."""

import pytest

from actsemble.data.windows import split_episodes


def test_split_is_disjoint_and_complete():
    ids = [f"ep_{i:03d}" for i in range(20)]
    split = split_episodes(ids, val_fraction=0.25, seed=0)
    assert set(split.train_episode_ids) | set(split.val_episode_ids) == set(ids)
    assert set(split.train_episode_ids) & set(split.val_episode_ids) == set()
    assert len(split.val_episode_ids) == 5


def test_split_deterministic():
    ids = [f"ep_{i:03d}" for i in range(20)]
    a = split_episodes(ids, val_fraction=0.3, seed=7)
    b = split_episodes(ids, val_fraction=0.3, seed=7)
    c = split_episodes(ids, val_fraction=0.3, seed=8)
    assert a.train_episode_ids == b.train_episode_ids
    assert a.hash == b.hash
    assert a.hash != c.hash


def test_split_order_independent():
    ids = [f"ep_{i:03d}" for i in range(20)]
    a = split_episodes(ids, val_fraction=0.3, seed=7)
    b = split_episodes(list(reversed(ids)), val_fraction=0.3, seed=7)
    assert a.train_episode_ids == b.train_episode_ids


def test_single_episode_goes_to_train():
    split = split_episodes(["only"], val_fraction=0.5, seed=0)
    assert split.train_episode_ids == ["only"]
    assert split.val_episode_ids == []


def test_transitions_never_split(episodes):
    """All windows of an episode stay on one side: split operates on ids only."""
    split = split_episodes([ep.episode_id for ep in episodes], val_fraction=0.34, seed=0)
    for ep in episodes:
        in_train = ep.episode_id in split.train_episode_ids
        in_val = ep.episode_id in split.val_episode_ids
        assert in_train != in_val


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        split_episodes(["a", "a", "b"], val_fraction=0.3, seed=0)

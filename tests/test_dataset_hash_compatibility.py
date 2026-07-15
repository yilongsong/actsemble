"""Dataset hash stability and sensitivity."""

from pathlib import Path

from actsemble.data.reader import DatasetReader
from actsemble.data.writer import compute_dataset_hash, write_dataset

from conftest import make_episodes, make_metadata


def test_hash_stable_across_rewrites(tmp_path):
    eps = make_episodes(4, T=30)
    h1 = write_dataset(tmp_path / "a.h5", eps, make_metadata())
    h2 = write_dataset(tmp_path / "b.h5", eps, make_metadata())
    assert h1 == h2
    assert DatasetReader(tmp_path / "a.h5").dataset_hash == h1


def test_hash_reader_recompute_matches(dataset_path):
    reader = DatasetReader(dataset_path)
    assert compute_dataset_hash(reader.episodes, reader.metadata) == reader.dataset_hash


def test_hash_changes_when_data_changes(tmp_path):
    eps = make_episodes(4, T=30)
    h1 = write_dataset(tmp_path / "a.h5", eps, make_metadata())
    eps[0].action[0, 0] += 1e-4
    h2 = write_dataset(tmp_path / "b.h5", eps, make_metadata())
    assert h1 != h2


def test_hash_changes_when_metadata_changes(tmp_path):
    eps = make_episodes(4, T=30)
    h1 = write_dataset(tmp_path / "a.h5", eps, make_metadata())
    meta = make_metadata()
    meta.controller = "different_controller"
    h2 = write_dataset(tmp_path / "b.h5", eps, meta)
    assert h1 != h2


def test_hash_independent_of_episode_insertion_order(tmp_path):
    eps = make_episodes(4, T=30)
    h1 = write_dataset(tmp_path / "a.h5", eps, make_metadata())
    h2 = write_dataset(tmp_path / "b.h5", list(reversed(eps)), make_metadata())
    assert h1 == h2

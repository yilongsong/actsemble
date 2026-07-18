"""Data subsystem: schema roundtrip, metadata, alignment, and leakage guards."""

import h5py
import numpy as np
import pytest

from actsemble.data.reader import DatasetReader
from actsemble.data.schema import FORBIDDEN_EPISODE_KEYS, REQUIRED_METADATA_KEYS
from actsemble.data.validation import DatasetValidationError, validate_dataset
from actsemble.data.writer import write_dataset

from conftest import make_metadata


def test_roundtrip_preserves_arrays(dataset_path, episodes):
    reader = DatasetReader(dataset_path)
    assert reader.episode_ids == [ep.episode_id for ep in episodes]
    for original, loaded in zip(episodes, reader.episodes):
        np.testing.assert_array_equal(original.state, loaded.state)
        np.testing.assert_array_equal(original.action, loaded.action)
        np.testing.assert_array_equal(original.next_state, loaded.next_state)
        np.testing.assert_array_equal(original.previous_action, loaded.previous_action)
        np.testing.assert_array_equal(original.step_index, loaded.step_index)


def test_metadata_complete(dataset_path):
    reader = DatasetReader(dataset_path)
    attrs = reader.metadata.to_attrs()
    for key in REQUIRED_METADATA_KEYS:
        assert key in attrs, f"missing metadata key {key}"
    assert reader.metadata.dataset_hash


def test_validation_passes(dataset_path):
    summary = validate_dataset(DatasetReader(dataset_path))
    assert summary["num_episodes"] == 6
    assert summary["state_dim"] == 8
    assert summary["action_dim"] == 3


def test_validation_catches_broken_alignment(tmp_path, episodes):
    episodes[2].next_state[5] += 1.0  # break state[t+1] == next_state[t]
    path = tmp_path / "broken.h5"
    write_dataset(path, episodes, make_metadata())
    with pytest.raises(DatasetValidationError, match="alignment"):
        validate_dataset(DatasetReader(path))


def test_validation_catches_nan(tmp_path, episodes):
    episodes[0].action[3, 1] = np.nan
    path = tmp_path / "nan.h5"
    write_dataset(path, episodes, make_metadata())
    with pytest.raises(DatasetValidationError, match="non-finite"):
        validate_dataset(DatasetReader(path))


def test_validation_catches_out_of_bounds_action(tmp_path, episodes):
    episodes[0].action[0, 0] = 2.5  # bounds are [-1, 1]
    # keep previous_action consistent so the bounds error is what fires
    episodes[0].previous_action[1, 0] = 2.5
    path = tmp_path / "oob.h5"
    write_dataset(path, episodes, make_metadata())
    with pytest.raises(DatasetValidationError, match="bounds"):
        validate_dataset(DatasetReader(path))


def test_no_forbidden_fields_in_episodes(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        for episode_id in f["episodes"]:
            keys = set(f["episodes"][episode_id].keys())
            forbidden = keys & set(FORBIDDEN_EPISODE_KEYS)
            assert not forbidden, f"leakage fields present: {forbidden}"


def test_empty_dataset_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_dataset(tmp_path / "empty.h5", [], make_metadata())


def test_validation_catches_smuggled_reward_array(tmp_path, episodes):
    """Even arrays the reader ignores must fail validation (leakage guard)."""
    path = tmp_path / "smuggled.h5"
    write_dataset(path, episodes, make_metadata())
    with h5py.File(path, "r+") as f:
        g = f["episodes"]["ep_00000"]
        g.create_dataset("reward", data=np.ones(40, np.float32))
    import pytest as _pytest

    with _pytest.raises(DatasetValidationError, match="forbidden"):
        validate_dataset(DatasetReader(path), check_hash=False)


def test_validation_catches_metadata_missing_from_file(tmp_path, episodes):
    path = tmp_path / "missing_metadata.h5"
    write_dataset(path, episodes, make_metadata())
    with h5py.File(path, "r+") as f:
        del f["metadata"].attrs["robot"]
    with pytest.raises(DatasetValidationError, match="metadata missing key robot"):
        validate_dataset(DatasetReader(path), check_hash=False)

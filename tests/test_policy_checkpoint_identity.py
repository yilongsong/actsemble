"""Checkpoint identity + same-dataset enforcement across policy/component."""

import shutil

import pytest

from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility
from actsemble.policies.diffusion.policy import DiffusionPolicy
from actsemble.systems.interface import check_same_data
from actsemble.training.checkpointing import checkpoint_hash, verify_same_dataset


def test_checkpoint_hash_matches_file_identity(trained_checkpoints, tmp_path):
    src = trained_checkpoints["policy_best"]
    copy = tmp_path / "copy.pt"
    shutil.copy(src, copy)
    a = DiffusionPolicy.from_checkpoint(src, device="cpu")
    b = DiffusionPolicy.from_checkpoint(copy, device="cpu")
    assert a.checkpoint_hash == b.checkpoint_hash == checkpoint_hash(src)


def test_different_checkpoints_have_different_hashes(trained_checkpoints):
    best = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    final = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_final"], device="cpu")
    assert best.checkpoint_hash != final.checkpoint_hash


def test_policy_and_component_share_dataset_identity(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    check_same_data(policy, [comp])  # must not raise
    verify_same_dataset(policy.meta.to_dict(), comp.meta, what="test")


def test_mismatched_dataset_hash_rejected(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    comp.meta = dict(comp.meta)
    comp.meta["dataset_hash"] = "0" * 64
    with pytest.raises(ValueError, match="dataset_hash"):
        check_same_data(policy, [comp])


def test_mismatched_horizons_rejected(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    comp.meta = dict(comp.meta)
    comp.meta["prediction_horizon"] = 99
    with pytest.raises(ValueError, match="prediction_horizon"):
        check_same_data(policy, [comp])


def test_wrong_checkpoint_kind_rejected(trained_checkpoints):
    with pytest.raises(ValueError, match="not an Actsemble diffusion-policy"):
        DiffusionPolicy.from_checkpoint(trained_checkpoints["component_best"], device="cpu")
    with pytest.raises(ValueError, match="not an Actsemble compatibility"):
        ActionChunkCompatibility.from_checkpoint(
            trained_checkpoints["policy_best"], device="cpu"
        )

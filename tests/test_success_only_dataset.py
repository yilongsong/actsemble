"""Success-only provenance: the auditable record of demonstration filtering."""

import pytest

from actsemble.data.validation import (
    DatasetValidationError,
    validate_success_only_provenance,
)
from actsemble.data.writer import write_private_provenance


def test_valid_provenance_passes(dataset_path):
    prov = validate_success_only_provenance(dataset_path)
    assert prov["success_only"] is True
    assert len(prov["exported_episodes"]) == 6


def test_missing_sidecar_fails(tmp_path, episodes):
    from actsemble.data.writer import write_dataset

    from conftest import make_metadata

    path = tmp_path / "nosidecar.h5"
    write_dataset(path, episodes, make_metadata())
    with pytest.raises(DatasetValidationError, match="provenance"):
        validate_success_only_provenance(path)


def test_non_successful_source_episode_fails(dataset_path):
    write_private_provenance(
        dataset_path,
        {
            "success_only": True,
            "exported_episodes": [
                {"episode_id": "ep_00000", "source_success": True},
                {"episode_id": "ep_00001", "source_success": False},
            ],
        },
    )
    with pytest.raises(DatasetValidationError, match="source_success"):
        validate_success_only_provenance(dataset_path)


def test_success_only_flag_required(dataset_path):
    write_private_provenance(
        dataset_path,
        {"success_only": False, "exported_episodes": [{"episode_id": "x", "source_success": True}]},
    )
    with pytest.raises(DatasetValidationError, match="success_only"):
        validate_success_only_provenance(dataset_path)

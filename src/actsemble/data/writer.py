"""Writing frozen Actsemble datasets (HDF5) and private provenance sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..types import EpisodeRecord
from ..utils.hashing import hash_arrays, hash_json
from ..utils.serialization import save_json
from .schema import EPISODE_ARRAY_KEYS, DatasetMetadata


def compute_dataset_hash(
    episodes: list[EpisodeRecord], metadata: DatasetMetadata
) -> str:
    """Content hash over all episode arrays (sorted by id) + key metadata.

    Excludes ``dataset_hash`` itself. Stable across processes and file
    rewrites as long as content is identical.
    """
    named: list[tuple[str, np.ndarray]] = []
    for ep in sorted(episodes, key=lambda e: e.episode_id):
        for key in EPISODE_ARRAY_KEYS:
            named.append((f"{ep.episode_id}/{key}", getattr(ep, key)))
    meta_attrs = metadata.to_attrs()
    meta_attrs.pop("dataset_hash", None)
    named.append(
        ("__metadata__", np.frombuffer(hash_json(meta_attrs).encode(), dtype=np.uint8))
    )
    return hash_arrays(named)


def write_dataset(
    path: str | Path,
    episodes: list[EpisodeRecord],
    metadata: DatasetMetadata,
) -> str:
    """Write episodes + metadata to ``path``; returns the dataset hash.

    The hash is computed from content and stored in metadata, so readers
    can recompute and verify it.
    """
    if not episodes:
        raise ValueError("Refusing to write an empty dataset")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata.dataset_hash = compute_dataset_hash(episodes, metadata)

    with h5py.File(path, "w") as f:
        meta_group = f.create_group("metadata")
        for key, value in metadata.to_attrs().items():
            meta_group.attrs[key] = value
        eps_group = f.create_group("episodes")
        for ep in episodes:
            g = eps_group.create_group(ep.episode_id)
            for key in EPISODE_ARRAY_KEYS:
                arr = getattr(ep, key)
                g.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
    return metadata.dataset_hash


def write_private_provenance(
    dataset_path: str | Path, provenance: dict[str, Any]
) -> Path:
    """Save preparation provenance (success filtering counts, source episode
    ids, replay failures, ...) NEXT TO the dataset, not inside it.

    Success status is provenance, never a model input.
    """
    dataset_path = Path(dataset_path)
    out = dataset_path.with_suffix(dataset_path.suffix + ".provenance.json")
    save_json(provenance, out)
    return out

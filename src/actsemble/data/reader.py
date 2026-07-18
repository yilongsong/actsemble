"""Reading frozen Actsemble datasets."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..types import EpisodeRecord
from .schema import EPISODE_ARRAY_KEYS, DatasetMetadata


class DatasetReader:
    """In-memory reader for a frozen Actsemble dataset.

    Phase 0 state datasets are tiny (MBs), so we load everything eagerly;
    this keeps window sampling trivial and removes h5py from the training
    hot path.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        with h5py.File(self.path, "r") as f:
            if "metadata" not in f or "episodes" not in f:
                raise ValueError(
                    f"{self.path} is not an Actsemble dataset (missing metadata/episodes groups)"
                )
            self.metadata = DatasetMetadata.from_attrs(dict(f["metadata"].attrs))
            self.episodes: list[EpisodeRecord] = []
            for episode_id in sorted(f["episodes"].keys()):
                g = f["episodes"][episode_id]
                arrays = {key: np.asarray(g[key]) for key in EPISODE_ARRAY_KEYS}
                self.episodes.append(EpisodeRecord(episode_id=episode_id, **arrays))

    @property
    def dataset_hash(self) -> str:
        return self.metadata.dataset_hash

    @property
    def episode_ids(self) -> list[str]:
        return [ep.episode_id for ep in self.episodes]

    @property
    def num_transitions(self) -> int:
        return int(sum(len(ep) for ep in self.episodes))

    @property
    def state_dim(self) -> int:
        return int(self.metadata.state_dimension)

    @property
    def action_dim(self) -> int:
        return int(self.metadata.action_dimension)

    def episode(self, episode_id: str) -> EpisodeRecord:
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                return ep
        raise KeyError(f"Episode not found: {episode_id}")

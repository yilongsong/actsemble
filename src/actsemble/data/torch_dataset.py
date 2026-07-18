"""PyTorch datasets over frozen Actsemble data.

These are the ONLY data entry points for training. They read dataset
files; they never touch the simulator (enforced by
tests/training/test_training_has_no_sim_dependency.py).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..types import EpisodeRecord
from .normalization import Normalizer
from .windows import Window, enumerate_window_indices, extract_window


def window_to_sample(
    w: Window, normalizer: Normalizer, ei: int, t: int, include_previous_action: bool
) -> dict[str, torch.Tensor]:
    """(raw window) -> normalized tensors, shared by every window dataset."""
    obs = np.asarray(normalizer.normalize_state(w.obs_history), dtype=np.float32)
    if include_previous_action:
        prev = np.asarray(
            normalizer.normalize_action(w.prev_action_history), dtype=np.float32
        )
        obs = np.concatenate([obs, prev], axis=1)
    return {
        "obs_history": torch.from_numpy(obs),
        "action_chunk": torch.from_numpy(
            np.asarray(normalizer.normalize_action(w.action_chunk), dtype=np.float32)
        ),
        "obs_mask": torch.from_numpy(w.obs_mask.astype(np.bool_)),
        "action_mask": torch.from_numpy(w.action_mask.astype(np.bool_)),
        "episode_index": torch.tensor(ei, dtype=torch.long),
        "t": torch.tensor(t, dtype=torch.long),
    }


class DiffusionWindowDataset(Dataset):
    """(normalized obs history, normalized action chunk, masks) windows."""

    def __init__(
        self,
        episodes: list[EpisodeRecord],
        normalizer: Normalizer,
        *,
        obs_horizon: int,
        prediction_horizon: int,
        include_previous_action: bool = False,
        alignment: str = "future_only",
        action_horizon: int | None = None,
    ):
        self.episodes = episodes
        self.normalizer = normalizer
        self.obs_horizon = int(obs_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.include_previous_action = bool(include_previous_action)
        self.alignment = str(alignment)
        self.action_horizon = None if action_horizon is None else int(action_horizon)
        self.indices = enumerate_window_indices(
            episodes,
            alignment=self.alignment,
            obs_horizon=self.obs_horizon,
            prediction_horizon=self.prediction_horizon,
            action_horizon=self.action_horizon,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ei, t = self.indices[i]
        w = extract_window(
            self.episodes[ei],
            t,
            obs_horizon=self.obs_horizon,
            prediction_horizon=self.prediction_horizon,
            alignment=self.alignment,
        )
        return window_to_sample(w, self.normalizer, ei, t, self.include_previous_action)


class ACTEpisodeDataset(Dataset):
    """Episode-indexed dataset for canonical ACT: ``__len__ == num_episodes`` and
    each ``__getitem__`` draws a RANDOM start timestep from that episode. Reference
    ACT weights EPISODES (not transitions) equally per epoch and augments with a
    random start; the random draw uses a dedicated generator seeded in ``__init__``,
    so with ``num_workers=0`` the whole training stream stays reproducible from the
    training seed. Use ``DiffusionWindowDataset`` (deterministic, all transitions)
    for validation."""

    def __init__(
        self,
        episodes: list[EpisodeRecord],
        normalizer: Normalizer,
        *,
        obs_horizon: int,
        prediction_horizon: int,
        include_previous_action: bool = False,
        alignment: str = "future_only",
        start_seed: int = 0,
        fixed: bool = False,
    ):
        self.episodes = episodes
        self.normalizer = normalizer
        self.obs_horizon = int(obs_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.include_previous_action = bool(include_previous_action)
        self.alignment = str(alignment)
        self._rng = np.random.default_rng(int(start_seed))
        # ``fixed`` (validation): draw ONE deterministic start per episode up front,
        # so the episode-weighted val set is stable across checkpoints.
        self.fixed = bool(fixed)
        self._fixed_t = (
            [int(self._rng.integers(0, len(ep))) for ep in episodes]
            if self.fixed
            else None
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, ei: int) -> dict[str, torch.Tensor]:
        ep = self.episodes[ei]
        if self.fixed:
            assert self._fixed_t is not None
            t = self._fixed_t[ei]
        else:
            t = int(self._rng.integers(0, len(ep)))
        w = extract_window(
            ep,
            t,
            obs_horizon=self.obs_horizon,
            prediction_horizon=self.prediction_horizon,
            alignment=self.alignment,
        )
        return window_to_sample(w, self.normalizer, ei, t, self.include_previous_action)

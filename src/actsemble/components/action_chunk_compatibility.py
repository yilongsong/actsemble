"""Action-Chunk Compatibility Component.

Estimates C_phi(s_{t-H_o+1:t}, a_{t:t+H_p-1}): how compatible a proposed
action chunk is with successful behavior in the frozen dataset.

Positives are real demonstration windows. Negatives are deterministic
transformations of chunks from the SAME dataset — no simulator, no failed
rollouts, no reward, no success labels, no external data.

High offline compatibility accuracy does not imply improved closed-loop
task success. Closed-loop evaluation is the decisive test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.normalization import NormalizationStats, Normalizer
from ..data.windows import extract_window
from ..seed import derive_seed
from ..types import EpisodeRecord
from ..utils.hashing import hash_file

# All negative types implemented for Phase 0. Gripper mismatch is listed
# for completeness but must stay disabled for tasks without a gripper
# (PushT's panda_stick has none).
ALL_NEGATIVE_TYPES = (
    "additive_noise",
    "action_scaling",
    "translation_offset",
    "rotation_perturbation",
    "temporal_shift",
    "other_trajectory",
    "incompatible_same_trajectory",
    "partial_reversal",
    "dimension_shuffle",
    "discontinuity_injection",
    "gripper_mismatch",
)


@dataclass
class NegativeConfig:
    enabled_types: list[str] = field(
        default_factory=lambda: [
            "additive_noise",
            "action_scaling",
            "translation_offset",
            "rotation_perturbation",
            "temporal_shift",
            "other_trajectory",
            "incompatible_same_trajectory",
            "partial_reversal",
            "dimension_shuffle",
            "discontinuity_injection",
        ]
    )
    additive_noise_scale: float = 0.4
    scaling_range_low: tuple[float, float] = (0.1, 0.5)
    scaling_range_high: tuple[float, float] = (1.8, 3.0)
    translation_offset_scale: float = 0.5
    rotation_offset_scale: float = 0.7
    position_dims: list[int] = field(default_factory=lambda: [0, 1, 2])
    rotation_dims: list[int] = field(default_factory=lambda: [3, 4, 5])
    gripper_dims: list[int] = field(default_factory=list)
    temporal_shift_min: int = 4
    temporal_shift_max: int = 12
    incompatible_min_distance: int = 32
    reversal_min_fraction: float = 0.5
    discontinuity_scale: float = 0.8

    @classmethod
    def from_dict(cls, d: dict) -> "NegativeConfig":
        kwargs = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        cfg = cls(**kwargs)
        unknown = set(cfg.enabled_types) - set(ALL_NEGATIVE_TYPES)
        if unknown:
            raise ValueError(f"Unknown negative types: {sorted(unknown)}")
        if "gripper_mismatch" in cfg.enabled_types and not cfg.gripper_dims:
            raise ValueError(
                "gripper_mismatch negatives enabled but gripper_dims is empty "
                "(disable this type for tasks without a gripper, e.g. PushT)"
            )
        return cfg

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class NegativeGenerator:
    """Deterministic negative-chunk generation from the frozen dataset.

    ``generate(episodes, ei, t, chunk, rng)`` returns (negative_chunk,
    type_name). All randomness comes from the caller-provided
    ``np.random.Generator``; identical rng state => identical negative.
    Transforms operate on raw actions and are clipped to action bounds.
    """

    def __init__(
        self,
        cfg: NegativeConfig,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        prediction_horizon: int,
        obs_horizon: int,
    ):
        self.cfg = cfg
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.h_p = int(prediction_horizon)
        self.h_o = int(obs_horizon)

    def _clip(self, chunk: np.ndarray) -> np.ndarray:
        return np.clip(chunk, self.action_low, self.action_high).astype(np.float32)

    def _chunk_at(self, episodes: list[EpisodeRecord], ei: int, t: int) -> np.ndarray:
        w = extract_window(
            episodes[ei], t, obs_horizon=self.h_o, prediction_horizon=self.h_p
        )
        return w.action_chunk

    def generate(
        self,
        episodes: list[EpisodeRecord],
        ei: int,
        t: int,
        chunk: np.ndarray,
        rng: np.random.Generator,
        *,
        negative_type: str | None = None,
    ) -> tuple[np.ndarray, str]:
        ntype = negative_type or self.cfg.enabled_types[
            int(rng.integers(len(self.cfg.enabled_types)))
        ]
        neg = getattr(self, f"_neg_{ntype}")(episodes, ei, t, chunk.copy(), rng)
        return self._clip(neg), ntype

    # -- transform implementations ------------------------------------------
    def _neg_additive_noise(self, episodes, ei, t, chunk, rng):
        noise = rng.uniform(-1.0, 1.0, size=chunk.shape).astype(np.float32)
        return chunk + self.cfg.additive_noise_scale * noise

    def _neg_action_scaling(self, episodes, ei, t, chunk, rng):
        lo, hi = (
            self.cfg.scaling_range_low
            if rng.random() < 0.5
            else self.cfg.scaling_range_high
        )
        return chunk * float(rng.uniform(lo, hi))

    def _neg_translation_offset(self, episodes, ei, t, chunk, rng):
        dims = self.cfg.position_dims
        offset = rng.uniform(-1.0, 1.0, size=len(dims)).astype(np.float32)
        norm = np.linalg.norm(offset) + 1e-8
        offset = offset / norm * self.cfg.translation_offset_scale
        chunk[:, dims] = chunk[:, dims] + offset
        return chunk

    def _neg_rotation_perturbation(self, episodes, ei, t, chunk, rng):
        dims = self.cfg.rotation_dims
        offset = rng.uniform(
            -self.cfg.rotation_offset_scale, self.cfg.rotation_offset_scale, size=len(dims)
        ).astype(np.float32)
        chunk[:, dims] = chunk[:, dims] + offset
        return chunk

    def _neg_temporal_shift(self, episodes, ei, t, chunk, rng):
        T = len(episodes[ei])
        lo, hi = self.cfg.temporal_shift_min, self.cfg.temporal_shift_max
        offsets = [
            d for d in list(range(-hi, -lo + 1)) + list(range(lo, hi + 1)) if 0 <= t + d < T
        ]
        if not offsets:
            return self._neg_additive_noise(episodes, ei, t, chunk, rng)
        d = offsets[int(rng.integers(len(offsets)))]
        return self._chunk_at(episodes, ei, t + d)

    def _neg_other_trajectory(self, episodes, ei, t, chunk, rng):
        if len(episodes) < 2:
            return self._neg_additive_noise(episodes, ei, t, chunk, rng)
        others = [i for i in range(len(episodes)) if i != ei]
        oi = others[int(rng.integers(len(others)))]
        ot = int(rng.integers(len(episodes[oi])))
        return self._chunk_at(episodes, oi, ot)

    def _neg_incompatible_same_trajectory(self, episodes, ei, t, chunk, rng):
        T = len(episodes[ei])
        candidates = [
            u for u in range(T) if abs(u - t) >= self.cfg.incompatible_min_distance
        ]
        if not candidates:
            return self._neg_other_trajectory(episodes, ei, t, chunk, rng)
        u = candidates[int(rng.integers(len(candidates)))]
        return self._chunk_at(episodes, ei, u)

    def _neg_partial_reversal(self, episodes, ei, t, chunk, rng):
        h = chunk.shape[0]
        seg = max(2, int(np.ceil(self.cfg.reversal_min_fraction * h)))
        start = int(rng.integers(0, h - seg + 1))
        chunk[start : start + seg] = chunk[start : start + seg][::-1]
        return chunk

    def _neg_dimension_shuffle(self, episodes, ei, t, chunk, rng):
        # Permute physically comparable dims consistently across the chunk
        # (e.g. swap x/y translation) — a "wrong direction" negative.
        dims = np.asarray(self.cfg.position_dims, dtype=int)
        perm = _non_identity_permutation(len(dims), rng)
        chunk[:, dims] = chunk[:, dims[perm]]
        return chunk

    def _neg_discontinuity_injection(self, episodes, ei, t, chunk, rng):
        h = chunk.shape[0]
        pos = int(rng.integers(1, h))
        jump = rng.uniform(-1.0, 1.0, size=chunk.shape[1]).astype(np.float32)
        jump = jump / (np.abs(jump).max() + 1e-8) * self.cfg.discontinuity_scale
        chunk[pos:] = chunk[pos:] + jump
        return chunk

    def _neg_gripper_mismatch(self, episodes, ei, t, chunk, rng):
        dims = self.cfg.gripper_dims
        if not dims:
            raise ValueError("gripper_mismatch requires gripper_dims")
        chunk[:, dims] = -chunk[:, dims]
        return chunk


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("dimension_shuffle needs at least 2 dims")
    while True:
        perm = rng.permutation(n)
        if not np.array_equal(perm, np.arange(n)):
            return perm


def window_negative_rng(seed: int, episode_id: str, t: int, replica: int) -> np.random.Generator:
    """The canonical deterministic RNG for a window's negative sample."""
    return np.random.default_rng(derive_seed(seed, "negative", episode_id, t, replica))


class CompatibilityMLP(nn.Module):
    """MLP over [flattened normalized obs history ++ flattened normalized chunk]."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (512, 512, 256), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ActionChunkCompatibility:
    """Runtime component (implements ActionChunkCompatibilityScorer)."""

    def __init__(
        self,
        model: CompatibilityMLP,
        normalizer: Normalizer,
        meta: dict,
        *,
        device: torch.device,
        checkpoint_hash: str = "",
        checkpoint_path: str = "",
    ):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.meta = meta
        self.device = device
        self._checkpoint_hash = checkpoint_hash
        self.checkpoint_path = checkpoint_path

    @property
    def dataset_hash(self) -> str:
        return self.meta["dataset_hash"]

    @property
    def checkpoint_hash(self) -> str:
        return self._checkpoint_hash

    def reset(self) -> None:
        pass  # stateless between episodes

    @torch.no_grad()
    def score(
        self, observation_history: torch.Tensor, candidate_chunks: torch.Tensor
    ) -> torch.Tensor:
        """Raw obs [H_o, feat] and raw chunks [K, H_p, A] -> [K] scores."""
        obs = observation_history.to(self.device).float()
        chunks = candidate_chunks.to(self.device).float()
        if obs.dim() != 2:
            raise ValueError(f"observation_history must be 2D, got {tuple(obs.shape)}")
        if chunks.dim() != 3:
            raise ValueError(f"candidate_chunks must be 3D, got {tuple(chunks.shape)}")
        sd = int(self.meta["state_dim"])
        state_norm = self.normalizer.normalize_state(obs[:, :sd])
        obs_norm = (
            torch.cat([state_norm, self.normalizer.normalize_action(obs[:, sd:])], dim=1)
            if self.meta.get("include_previous_action", False)
            else state_norm
        )
        chunks_norm = self.normalizer.normalize_action(chunks)
        k = chunks.shape[0]
        x = torch.cat(
            [obs_norm.reshape(1, -1).expand(k, -1), chunks_norm.reshape(k, -1)], dim=1
        )
        return self.model(x)

    # -- persistence ---------------------------------------------------------
    @staticmethod
    def save_checkpoint(path: str | Path, *, config: dict, meta: dict, model_state: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "actsemble_compatibility_component",
                "config": config,
                "meta": meta,
                "model_state": model_state,
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: torch.device | str = "cpu"
    ) -> "ActionChunkCompatibility":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("kind") != "actsemble_compatibility_component":
            raise ValueError(f"{path} is not an Actsemble compatibility checkpoint")
        meta = ckpt["meta"]
        model_cfg = ckpt["config"].get("model", {})
        feat = int(meta["state_dim"]) + (
            int(meta["action_dim"]) if meta.get("include_previous_action") else 0
        )
        input_dim = int(meta["obs_horizon"]) * feat + int(meta["prediction_horizon"]) * int(
            meta["action_dim"]
        )
        model = CompatibilityMLP(
            input_dim,
            hidden=tuple(model_cfg.get("hidden", [512, 512, 256])),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        model.load_state_dict(ckpt["model_state"])
        normalizer = Normalizer(NormalizationStats.from_dict(meta["normalization"]))
        return cls(
            model,
            normalizer,
            meta,
            device=torch.device(device),
            checkpoint_hash=hash_file(path),
            checkpoint_path=str(path),
        )

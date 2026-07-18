"""Per-dimension min-max normalization to [-1, 1].

Statistics are computed once from the frozen dataset (all successful
demonstrations) and stored — with a content hash — inside every policy and
component checkpoint. The policy and component MUST share identical stats;
system construction verifies the hashes match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..utils.hashing import hash_json


MINMAX = "minmax_to_unit_range"  # Diffusion Policy / Flow Matching convention
STANDARDIZE = "standardize"  # ACT convention (zero-mean / unit-std)


@dataclass
class NormalizationStats:
    """Per-dimension affine normalization, either min-max to [-1, 1] (the
    Diffusion-Policy convention) or standardization to zero-mean/unit-std (the
    ACT convention). Both reduce to an affine map; ``Normalizer`` computes the
    scale/offset from whichever set of stats ``method`` selects.

    All fields default to None so the min-max keyword constructor stays
    backward-compatible with existing checkpoints/callers."""

    method: str = MINMAX
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None

    def to_dict(self) -> dict:
        d: dict = {"method": self.method}
        if self.method == MINMAX:
            assert self.state_min is not None
            assert self.state_max is not None
            assert self.action_min is not None
            assert self.action_max is not None
            d.update(
                state_min=self.state_min.tolist(),
                state_max=self.state_max.tolist(),
                action_min=self.action_min.tolist(),
                action_max=self.action_max.tolist(),
            )
        elif self.method == STANDARDIZE:
            assert self.state_mean is not None
            assert self.state_std is not None
            assert self.action_mean is not None
            assert self.action_std is not None
            d.update(
                state_mean=self.state_mean.tolist(),
                state_std=self.state_std.tolist(),
                action_mean=self.action_mean.tolist(),
                action_std=self.action_std.tolist(),
            )
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationStats":
        method = d.get("method", MINMAX)
        f = lambda k: np.asarray(d[k], dtype=np.float32)  # noqa: E731
        if method == MINMAX:
            return cls(
                method=method,
                state_min=f("state_min"),
                state_max=f("state_max"),
                action_min=f("action_min"),
                action_max=f("action_max"),
            )
        if method == STANDARDIZE:
            return cls(
                method=method,
                state_mean=f("state_mean"),
                state_std=f("state_std"),
                action_mean=f("action_mean"),
                action_std=f("action_std"),
            )
        raise ValueError(f"Unknown normalization method: {method}")

    @property
    def hash(self) -> str:
        return hash_json(self.to_dict())


def compute_stats(
    episodes, method: str = MINMAX, *, include_next_state: bool = True
) -> NormalizationStats:
    """Normalization statistics over all transitions of all given episodes.

    ``include_next_state`` (default True) also folds the terminal (next) states
    in, so min-max ranges cover evaluation-time observations near episode ends.
    Reference ACT computes state statistics from the recorded observation
    sequence only (each interior state counted once), so the ACT trainer passes
    ``include_next_state=False``."""
    states = np.concatenate([ep.state for ep in episodes], axis=0)
    if include_next_state:
        next_states = np.concatenate([ep.next_state for ep in episodes], axis=0)
        all_states = np.concatenate([states, next_states], axis=0)
    else:
        all_states = states
    actions = np.concatenate([ep.action for ep in episodes], axis=0)
    if method == MINMAX:
        return NormalizationStats(
            method=MINMAX,
            state_min=all_states.min(axis=0).astype(np.float32),
            state_max=all_states.max(axis=0).astype(np.float32),
            action_min=actions.min(axis=0).astype(np.float32),
            action_max=actions.max(axis=0).astype(np.float32),
        )
    if method == STANDARDIZE:
        return NormalizationStats(
            method=STANDARDIZE,
            state_mean=all_states.mean(axis=0).astype(np.float32),
            state_std=all_states.std(axis=0).astype(np.float32),
            action_mean=actions.mean(axis=0).astype(np.float32),
            action_std=actions.std(axis=0).astype(np.float32),
        )
    raise ValueError(f"Unknown normalization method: {method}")


class Normalizer:
    """Maps raw values to [-1, 1] and back. Constant dimensions map to 0."""

    def __init__(self, stats: NormalizationStats):
        self.stats = stats
        if stats.method == MINMAX:
            assert stats.state_min is not None
            assert stats.state_max is not None
            assert stats.action_min is not None
            assert stats.action_max is not None
            self._state_scale, self._state_offset = self._affine(
                stats.state_min, stats.state_max
            )
            self._action_scale, self._action_offset = self._affine(
                stats.action_min, stats.action_max
            )
        elif stats.method == STANDARDIZE:
            assert stats.state_mean is not None
            assert stats.state_std is not None
            assert stats.action_mean is not None
            assert stats.action_std is not None
            self._state_scale, self._state_offset = self._standardize(
                stats.state_mean, stats.state_std
            )
            self._action_scale, self._action_offset = self._standardize(
                stats.action_mean, stats.action_std
            )
        else:
            raise ValueError(f"Unknown normalization method: {stats.method}")

    @staticmethod
    def _affine(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Min-max to [-1, 1]. Constant dimensions map to 0."""
        span = hi - lo
        degenerate = span < 1e-8
        safe_span = np.where(degenerate, 1.0, span)
        # For a constant dimension use x -> x - lo. Dataset values still map
        # to zero, while the inverse maps zero back to the actual constant.
        # Keeping a nonzero scale also makes the affine transform invertible.
        scale = np.where(degenerate, 1.0, 2.0 / safe_span).astype(np.float32)
        offset = np.where(degenerate, -lo, -(hi + lo) / safe_span).astype(np.float32)
        return scale, offset

    STD_CLIP_MIN = 1e-2  # reference ACT clips std to >= 0.01 (detr/…/utils.py)

    @staticmethod
    def _standardize(
        mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Zero-mean / unit-std. Std is clipped to >= ``STD_CLIP_MIN`` (reference
        ACT) so near-constant dimensions are not amplified without bound; a truly
        constant dimension then normalizes to 0 (since ``x == mean``)."""
        safe_std = np.maximum(std, Normalizer.STD_CLIP_MIN).astype(np.float32)
        scale = (1.0 / safe_std).astype(np.float32)
        offset = (-mean / safe_std).astype(np.float32)
        return scale, offset

    # -- numpy / torch polymorphic helpers ---------------------------------
    @staticmethod
    def _apply(x, scale, offset):
        if isinstance(x, torch.Tensor):
            s = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
            o = torch.as_tensor(offset, dtype=x.dtype, device=x.device)
            return x * s + o
        return x * scale + offset

    @staticmethod
    def _unapply(x, scale, offset):
        if isinstance(x, torch.Tensor):
            s = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
            o = torch.as_tensor(offset, dtype=x.dtype, device=x.device)
            safe = torch.where(s == 0, torch.ones_like(s), s)
            out = (x - o) / safe
            return torch.where(s == 0, torch.zeros_like(out), out)
        safe_numpy = np.where(scale == 0, 1.0, scale)
        out = (x - offset) / safe_numpy
        return np.where(scale == 0, 0.0, out)

    def normalize_state(self, x):
        return self._apply(x, self._state_scale, self._state_offset)

    def unnormalize_state(self, x):
        return self._unapply(x, self._state_scale, self._state_offset)

    def normalize_action(self, x):
        return self._apply(x, self._action_scale, self._action_offset)

    def unnormalize_action(self, x):
        return self._unapply(x, self._action_scale, self._action_offset)

    @property
    def hash(self) -> str:
        return self.stats.hash

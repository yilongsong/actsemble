"""Shared checkpoint identity helpers.

Checkpoint files are ordinary torch.save archives; their identity is the
SHA-256 of the file bytes (``actsemble.utils.hashing.hash_file``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..utils.hashing import hash_file


def checkpoint_hash(path: str | Path) -> str:
    return hash_file(path)


def load_checkpoint_meta(path: str | Path) -> dict:
    """Read only kind/config/meta from a checkpoint (no model construction)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "kind": ckpt.get("kind"),
        "config": ckpt.get("config", {}),
        "meta": ckpt.get("meta", {}),
    }


def verify_same_dataset(meta_a: dict, meta_b: dict, *, what: str) -> None:
    """Fail loudly unless two checkpoints share dataset + normalization."""
    for key in ("dataset_hash", "split_hash"):
        if meta_a.get(key) != meta_b.get(key):
            raise ValueError(
                f"{what}: {key} mismatch\n  a: {meta_a.get(key)}\n  b: {meta_b.get(key)}"
            )
    norm_a, norm_b = meta_a.get("normalization"), meta_b.get("normalization")
    if norm_a != norm_b:
        raise ValueError(f"{what}: normalization statistics differ")

"""Deterministic seeding utilities.

Every stochastic stage (dataset preparation, training, candidate sampling,
perturbations, evaluation) derives its randomness from an explicit integer
seed so paired comparisons are exactly reproducible.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def derive_seed(root_seed: int, *tags: object) -> int:
    """Derive a stable sub-seed from a root seed and a tag path.

    Uses SHA-256 so derived streams are independent of Python hash
    randomization and stable across processes and platforms.
    """
    payload = ":".join([str(root_seed), *[str(t) for t in tags]]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A torch.Generator seeded with ``seed`` on ``device``."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen

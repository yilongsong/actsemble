"""Device selection."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a device string; ``None``/"auto" prefers CUDA when available."""
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

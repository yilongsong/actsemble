"""Stable content hashing for datasets, splits, normalization stats, and checkpoints.

These hashes are the backbone of the fairness safeguards: every policy and
component checkpoint records the dataset/split/normalization hashes it was
trained on, and system construction rejects mismatches by default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _update_with_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    # Hash dtype, shape, and C-contiguous bytes so the digest is
    # independent of in-memory layout.
    a = np.ascontiguousarray(arr)
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())


def hash_arrays(named_arrays: list[tuple[str, np.ndarray]]) -> str:
    """SHA-256 over (name, dtype, shape, bytes) in the given order."""
    h = hashlib.sha256()
    for name, arr in named_arrays:
        h.update(name.encode())
        _update_with_array(h, arr)
    return h.hexdigest()


def hash_json(obj: Any) -> str:
    """SHA-256 of a JSON-serializable object with sorted keys."""
    payload = json.dumps(obj, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def hash_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()

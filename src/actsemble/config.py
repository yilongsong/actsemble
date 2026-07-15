"""Minimal YAML configuration loading.

Configs are plain nested dicts loaded from YAML. We deliberately avoid a
heavy config framework: every entry point takes ``--config`` paths plus a
few explicit CLI overrides, and the fully resolved config is saved next to
every checkpoint and evaluation result for reproducibility.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(cfg)}")
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Fetch ``a.b.c`` from a nested dict, raising a clear error if absent."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing required config key: {dotted_key!r}")
        node = node[part]
    return node

"""Repository provenance for audit trails."""

from __future__ import annotations

import subprocess
from pathlib import Path


def current_git_commit(cwd: str | Path | None = None) -> str | None:
    """The current git commit hash, or None when not in a git repository.

    Protocol audit trails record this; run experiments from a committed
    tree so results are attributable to exact code.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None

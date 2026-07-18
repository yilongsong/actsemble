"""Rollout video export (mp4 via imageio-ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_video(
    frames: list[np.ndarray], path: str | Path, *, fps: int = 20
) -> Path | None:
    """Write frames to mp4; returns the path, or None when nothing to save."""
    if not frames:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))  # type: ignore[attr-defined]
    return path

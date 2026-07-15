"""JSONL + TensorBoard training logs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..utils.serialization import to_jsonable


class TrainingLogger:
    """Writes every scalar dict to metrics.jsonl and TensorBoard."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.output_dir / "metrics.jsonl", "a")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        except Exception:
            self._tb = None

    def log(self, step: int, scalars: dict, *, prefix: str = "") -> None:
        record = {"step": int(step), "time": time.time()}
        for key, value in scalars.items():
            record[f"{prefix}{key}"] = value
        self._jsonl.write(json.dumps(to_jsonable(record)) + "\n")
        self._jsonl.flush()
        if self._tb is not None:
            for key, value in scalars.items():
                if isinstance(value, (int, float)):
                    self._tb.add_scalar(f"{prefix}{key}", value, step)

    def close(self) -> None:
        self._jsonl.close()
        if self._tb is not None:
            self._tb.close()

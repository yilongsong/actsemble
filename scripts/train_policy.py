#!/usr/bin/env python
"""Train the state-conditioned diffusion policy on a frozen dataset.

Usage:
    python scripts/train_policy.py --config configs/policies/state_diffusion.yaml \
        --dataset data/push_t_smoke.h5 --max-steps 200 --output-dir outputs/policy_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.config import load_config
from actsemble.training.train_diffusion_policy import train_diffusion_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = train_diffusion_policy(
        policy_cfg=load_config(args.config),
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=device,
        resume=args.resume,
    )
    print(f"[train_policy] steps: {summary['steps']}")
    print(f"[train_policy] final train loss: {summary['final_train_loss']:.5f}")
    if summary["best_val_loss"] is not None:
        print(f"[train_policy] best val loss: {summary['best_val_loss']:.5f}")
    for name, path in summary["checkpoints"].items():
        print(f"[train_policy] checkpoint {name}: {path}")
    print(f"[train_policy] dataset_hash: {summary['dataset_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

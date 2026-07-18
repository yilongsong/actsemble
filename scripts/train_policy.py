#!/usr/bin/env python
"""Train a state-conditioned action-chunk policy on a frozen dataset.

Dispatches on ``model.type`` in the config: ``conditional_unet_1d`` -> diffusion
policy, ``act`` -> ACT (CVAE). Both produce interchangeable ActionChunkPolicy
checkpoints.

Usage:
    python scripts/train_policy.py --config configs/policies/state_diffusion.yaml \
        --dataset data/push_t_smoke.h5 --max-steps 200 --output-dir outputs/policy_smoke
    python scripts/train_policy.py --config configs/policies/state_act.yaml \
        --dataset data/active_min/subset_0400.h5 --output-dir outputs/active_min/act_seed_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.config import load_config
from actsemble.training.factory import policy_trainer, resolve_policy_family


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

    cfg = load_config(args.config)
    resolve_policy_family(cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = policy_trainer(cfg)(
        policy_cfg=cfg,
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

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
from actsemble.training.train_act_policy import train_act_policy
from actsemble.training.train_diffusion_policy import train_diffusion_policy
from actsemble.training.train_flow_policy import train_flow_policy

TRAINERS = {"diffusion": train_diffusion_policy, "act": train_act_policy, "flow": train_flow_policy}


def resolve_family(cfg: dict) -> str:
    """Training family: explicit top-level ``type`` if present, else inferred from
    ``model.type`` (conditional_unet_1d -> diffusion, act -> act) for older configs.
    Flow and diffusion share the U-Net, so flow configs must set ``type: flow``."""
    if cfg.get("type"):
        return str(cfg["type"])
    return {"conditional_unet_1d": "diffusion", "act": "act"}.get(
        cfg.get("model", {}).get("type", "conditional_unet_1d"), "diffusion"
    )


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
    family = resolve_family(cfg)
    if family not in TRAINERS:
        raise ValueError(f"Unknown training family {family!r}; expected one of {list(TRAINERS)}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summary = TRAINERS[family](
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

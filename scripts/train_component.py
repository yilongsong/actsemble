#!/usr/bin/env python
"""Train the action-chunk compatibility component on the same frozen dataset.

Usage:
    python scripts/train_component.py \
        --config configs/components/action_chunk_compatibility.yaml \
        --dataset data/push_t_smoke.h5 --max-steps 200 --output-dir outputs/component_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.config import load_config
from actsemble.training.train_component import train_component


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
    summary = train_component(
        component_cfg=load_config(args.config),
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=device,
        resume=args.resume,
    )
    print(f"[train_component] steps: {summary['steps']}")
    print(f"[train_component] final train loss: {summary['final_train_loss']:.5f}")
    ev = summary["offline_eval"]
    print(f"[train_component] offline ({ev['evaluated_on']}): "
          f"pos_acc={ev['positive_accuracy']:.3f} "
          f"neg_acc={ev['negative_accuracy']:.3f} "
          f"ranking_acc={ev['pairwise_ranking_accuracy']:.3f}")
    print("[train_component] NOTE: high offline compatibility accuracy does not imply "
          "improved closed-loop task success; closed-loop evaluation is the decisive test.")
    for name, path in summary["checkpoints"].items():
        print(f"[train_component] checkpoint {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

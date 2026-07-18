#!/usr/bin/env python
"""Closed-loop paired evaluation of one autonomy system.

Usage:
    python scripts/evaluate.py \
        --system-config configs/systems/standalone_diffusion.yaml \
        --policy-checkpoint outputs/policy_smoke/best_ema.pt \
        --eval-config configs/evaluation/nominal.yaml \
        --num-episodes 10 --output outputs/eval_standalone.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.config import load_config
from actsemble.evaluation.evaluator import evaluate_system


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-config", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument(
        "--component-checkpoint",
        action="append",
        default=[],
        help="repeatable; order matters",
    )
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="evaluate raw (non-EMA) policy weights",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly overwrite an existing result file",
    )
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_system(
        system_cfg=load_config(args.system_config),
        eval_cfg=load_config(args.eval_config),
        policy_checkpoint=args.policy_checkpoint,
        component_checkpoints=args.component_checkpoint,
        num_episodes=args.num_episodes,
        output_path=args.output,
        video_dir=args.video_dir,
        device=device,
        use_ema=not args.raw_weights,
        force=args.force,
    )
    ci = result["confidence_interval"]
    print(f"[evaluate] system: {result['system_name']}  regime: {result['regime']}")
    print(
        f"[evaluate] success ({result['primary_metric']}): "
        f"{result['success_count']}/{result['num_episodes']} = "
        f"{result['success_rate']:.1%}  Wilson 95% CI [{ci[0]:.1%}, {ci[1]:.1%}]"
    )
    print(f"[evaluate] success_at_end rate: {result['success_at_end_rate']:.1%}")
    print(
        f"[evaluate] timeouts: {result['timeout_rate']:.1%}  "
        f"exceptions: {result['exception_rate']:.1%}  "
        f"fallback replans: {result['fallback_replan_rate']:.1%}  "
        f"episodes with fallback: {result['fallback_episode_rate']:.1%}"
    )
    print(
        f"[evaluate] mean decision latency: "
        f"{result['latency']['mean_decision_s'] * 1000:.1f} ms "
        f"(policy {result['latency']['mean_policy_s'] * 1000:.1f} ms, "
        f"component {result['latency']['mean_component_s'] * 1000:.1f} ms)"
    )
    print(
        f"[evaluate] p95/p99 decision latency: "
        f"{result['latency']['decision']['p95_s'] * 1000:.1f}/"
        f"{result['latency']['decision']['p99_s'] * 1000:.1f} ms"
    )
    print(f"[evaluate] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

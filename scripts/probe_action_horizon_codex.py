#!/usr/bin/env python
"""Production-code action-horizon sweep for one frozen policy.

Development-only diagnostic. Uses the ordinary candidate-zero system at
H_a in {1,2,4,8,16}; no runtime implementation is modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--panel-root", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    policy = load_policy(args.checkpoint, device=args.device, use_ema=True)
    env = make_env(
        task_id=policy.meta.task_id,
        control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend,
        obs_mode="state",
        render_mode=None,
    )
    del policy
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_cfg = {
        "name": "codex_action_horizon_probe",
        "regime": "nominal",
        "max_steps": 100,
        "panel": {
            "name": "codex_action_horizon_probe",
            "env_seed": args.panel_root,
            "num_episodes": args.count,
        },
        "perturbations": [],
    }
    summary = {}
    for horizon in (1, 2, 4, 8, 16):
        print(f"[horizon-probe] H_a={horizon}", flush=True)
        result = evaluate_system(
            system_cfg={
                "name": f"standalone_h{horizon}",
                "policy": {"num_candidates": 1},
                "selection": {"type": "candidate_zero"},
                "execution": {"action_horizon": horizon},
            },
            eval_cfg=eval_cfg,
            policy_checkpoint=args.checkpoint,
            num_episodes=args.count,
            output_path=out / f"h{horizon}.json",
            device=args.device,
            env=env,
            force=True,
        )
        summary[f"h{horizon}"] = {
            "success_count": result["success_count"],
            "num_episodes": result["num_episodes"],
            "success_rate": result["success_rate"],
            "confidence_interval": result["confidence_interval"],
            "successes": result["successes"],
        }
    env.close()
    save_json(
        {
            "kind": "development_only_action_horizon_sweep",
            "checkpoint": args.checkpoint,
            "panel": {"root": args.panel_root, "count": args.count},
            "results": summary,
        },
        out / "summary.json",
    )
    for name, row in summary.items():
        print(
            f"  {name}: {row['success_count']}/{row['num_episodes']} "
            f"= {row['success_rate']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Sampler ablation on a SINGLE frozen policy checkpoint — no retraining, no
re-selection. Evaluates one checkpoint under several inference-sampler settings
on a fixed panel and reports success + Wilson CI + latency per setting:

  * Diffusion: DDIM-16 (benchmark) vs DDPM-100 (reference)
        python scripts/sampler_comparison.py --checkpoint <dp.pt> \
            --samplers ddim:16,ddpm:100 --n 200
  * Flow: Euler step-count convergence (5/10/20)
        python scripts/sampler_comparison.py --checkpoint <flow.pt> \
            --flow-steps 5,10,20 --n 200

The EXACT sampler settings (incl. the DDIM timestep sequence) are recorded per
row, so the comparison is reproducible from the output JSON. Development-tier
(single checkpoint, single seed) — a sampler sanity/convergence check, not a claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.evaluation.evaluator import (  # noqa: E402
    episode_row,
    run_panel_episode,
    sampler_provenance,
)
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel, panel_episodes  # noqa: E402
from actsemble.policies.diffusion.policy import DiffusionPolicy  # noqa: E402
from actsemble.policies.flow.policy import FlowMatchingPolicy  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

STANDALONE = {"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"},
              "execution": {}}


def _eval_on_panel(policy, panel, env, max_steps: int) -> tuple[int, int, float]:
    system = build_system(STANDALONE, policy, [])
    rows = [
        episode_row(ep, run_panel_episode(env, system, ep, max_steps=max_steps, pert_specs=[])[0])
        for ep in panel_episodes(panel)
    ]
    n = panel.num_episodes
    succ = int(sum(r["success_once"] for r in rows))
    ms = 1000.0 * float(np.mean([r["mean_policy_latency_s"] for r in rows]))
    return succ, n, ms


def _load_variant(kind: str, ckpt: str, spec: str, device: str):
    """Build the policy under one sampler setting; returns (policy, label)."""
    if kind == "actsemble_diffusion_policy":
        sampler, steps = spec.split(":")
        pol = DiffusionPolicy.from_checkpoint(
            ckpt, device=device,
            sampler_overrides={"inference_sampler": sampler, "inference_steps": int(steps)},
        )
        return pol, f"{sampler}-{steps}"
    if kind == "actsemble_flow_policy":
        pol = FlowMatchingPolicy.from_checkpoint(
            ckpt, device=device, sampler_overrides={"inference_steps": int(spec)}
        )
        return pol, f"euler-{spec}"
    raise ValueError(f"sampler_comparison supports diffusion/flow, not {kind!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samplers", default="ddim:16,ddpm:100",
                    help="diffusion: comma list of sampler:steps")
    ap.add_argument("--flow-steps", default="5,10,20", help="flow: comma list of Euler step counts")
    ap.add_argument("--n", type=int, default=200, help="panel episodes (confirmation-sized by default)")
    ap.add_argument("--panel", default="confirmation")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kind = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("kind")
    specs = (args.flow_steps if kind == "actsemble_flow_policy" else args.samplers).split(",")
    base = make_panel(args.panel)
    panel = make_panel(args.panel, {"env_seed": base.env_seed, "num_episodes": args.n})

    ref = _load_variant(kind, args.checkpoint, specs[0], "cpu")[0].meta
    env = make_env(task_id=ref.task_id, control_mode=ref.controller,
                   sim_backend=ref.simulation_backend, obs_mode="state", render_mode=None)
    rows = []
    try:
        for spec in specs:
            policy, label = _load_variant(kind, args.checkpoint, spec, args.device)
            succ, n, ms = _eval_on_panel(policy, panel, env, args.max_steps)
            ci = wilson_interval(succ, n)
            rows.append({"setting": label, "success_count": succ, "num_episodes": n,
                         "success_rate": succ / n, "wilson_ci": list(ci), "policy_ms": ms,
                         "sampler": sampler_provenance(policy)})
            print(f"[sampler] {label:>10}: {succ}/{n} = {succ / n:.1%} "
                  f"CI [{ci[0]:.1%},{ci[1]:.1%}]  {ms:.1f} ms/query", flush=True)
    finally:
        env.close()

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "sampler_comparison.json"
    save_json({"checkpoint": args.checkpoint, "kind": kind, "panel": panel.to_dict(),
               "results": rows}, out)
    print(f"[sampler] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Faithful replication of ACT's temporal-ensembling claim, on our ACT policy.

ACT's actual claim: temporal ensembling (query every step + weighted average, the
ORIGINAL oldest-weighted near-uniform `w_i = exp(-m*i)`, i=0 oldest, m=0.01) beats
executing the full chunk open-loop and replanning per chunk. We run EXACTLY that
comparison, and also insert the decomposition point ACT never tested (query every
step, NO average):

  * act_openloop  — standalone, H_a = H_p = 16 (full chunk open-loop) = ACT "no TE"
  * act_te        — temporal, mean, recency=oldest, decay=0.01     = ACT temporal ensemble
  * act_latest    — temporal, latest (query every step, no average) = our decomposition point

Reports ACT's claim (act_te vs act_openloop) AND our nuance (act_latest vs both).
Diagnostic panel; ACT here is a lightweight variant (see docs/deferred_work.md).

Usage:
    python scripts/replicate_act.py --count 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402
from compare_temporal import mcnemar  # noqa: E402

SYSTEMS = {
    "act_openloop": {"policy": {"num_candidates": 1}, "components": [],
                     "selection": {"type": "candidate_zero"},
                     "execution": {"action_horizon": 16}},
    "act_latest": {"policy": {"num_candidates": 1}, "components": [],
                   "selection": {"type": "temporal_ensemble", "aggregation": "latest"},
                   "execution": {"replan_interval": 1}},
    "act_te": {"policy": {"num_candidates": 1}, "components": [],
               "selection": {"type": "temporal_ensemble", "aggregation": "mean",
                             "recency": "oldest", "decay": 0.01},
               "execution": {"replan_interval": 1}},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-checkpoint",
                    default=str(REPO / "outputs/active_min/act_masked_seed_0/best_ema.pt"))
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/active_min/act_replication"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    panel = make_panel("diagnostic")
    eval_cfg = {"name": "diagnostic", "regime": "nominal", "max_steps": 100,
                "panel": {"name": panel.name, "env_seed": panel.env_seed, "num_episodes": args.count},
                "perturbations": []}
    policy = load_policy(args.policy_checkpoint, device=args.device, use_ema=True)
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state",
                   render_mode="rgb_array")
    del policy
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for label, cfg in SYSTEMS.items():
        print(f"[replicate-act] {label} ({args.count} ep) ...", flush=True)
        res[label] = evaluate_system(
            system_cfg={"name": label, **cfg}, eval_cfg=eval_cfg,
            policy_checkpoint=args.policy_checkpoint, num_episodes=args.count,
            output_path=out / f"eval_{label}.json", device=args.device, env=env, force=True)
    env.close()

    n = args.count

    def row(label):
        r = res[label]
        ci = wilson_interval(r["success_count"], n)
        return f"  {label:<14}{r['success_rate']:>7.1%}  [{ci[0]:.1%}, {ci[1]:.1%}]"

    def pair(a, b):
        mc = mcnemar(res[a]["successes"], res[b]["successes"])
        d = res[a]["success_rate"] - res[b]["success_rate"]
        sig = "*" if mc["p_exact"] < 0.05 else ""
        return (f"{a} vs {b}:  {d:+.1%}{sig}   win/loss {mc['a_wins']}/{mc['b_wins']}   "
                f"McNemar p={mc['p_exact']:.3f}")

    L = ["=" * 68, f"ACT temporal-ensembling replication — diagnostic panel, n={n}", "=" * 68]
    L += [row(k) for k in SYSTEMS]
    L += ["", "ACT'S PAPER CLAIM  (temporal ensembling > full open-loop chunk):",
          "  " + pair("act_te", "act_openloop"),
          "", "OUR DECOMPOSITION (dense replanning WITHOUT averaging):",
          "  " + pair("act_latest", "act_te"),
          "  " + pair("act_latest", "act_openloop")]
    print("\n".join(L))
    save_json({"n": n, "systems": {k: {"success_rate": res[k]["success_rate"],
                                       "success_count": res[k]["success_count"]} for k in SYSTEMS}},
              out / "replication.json")
    print(f"\n[replicate-act] wrote {out / 'replication.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

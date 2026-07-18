#!/usr/bin/env python
"""Side-by-side standalone eval of the authoritative state policies on the
diagnostic panel: Diffusion, ACT, and Flow Matching. Each system's execution
offset is set from the policy's window alignment (H_o-1 for the DP alignment,
0 for future-only), read from the checkpoint meta. Reports success + the
per-query inference latency (the compute axis behind the latency contract).

Development-tier (diagnostic panel, single seed) — validates the reproductions,
NOT a claim. Usage: python scripts/eval_canonical.py --count 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

POLICIES = {
    "diffusion": REPO / "outputs/active_min/dp_canonical_seed_0/best_ema.pt",
    "act": REPO / "outputs/active_min/act_canonical_seed_0/best_ema.pt",
    "flow": REPO / "outputs/active_min/flow_canonical_seed_0/best_ema.pt",
}


def exec_offset(meta) -> int:
    return (
        meta.obs_horizon - 1
        if meta.extra.get("window_alignment") == "diffusion_policy"
        else 0
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument(
        "--out-dir", default=str(REPO / "outputs/active_min/canonical_eval")
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    panel = make_panel("diagnostic")
    eval_cfg = {
        "name": "diagnostic",
        "regime": "nominal",
        "max_steps": 100,
        "panel": {
            "name": panel.name,
            "env_seed": panel.env_seed,
            "num_episodes": args.count,
        },
        "perturbations": [],
    }
    ref = load_policy(str(POLICIES["diffusion"]), device="cpu")  # for the shared env
    env = make_env(
        task_id=ref.meta.task_id,
        control_mode=ref.meta.controller,
        sim_backend=ref.meta.simulation_backend,
        obs_mode="state",
        render_mode="rgb_array",
    )
    del ref
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for label, ckpt in POLICIES.items():
        meta = load_policy(str(ckpt), device="cpu").meta
        offset = exec_offset(meta)
        cfg = {
            "name": f"standalone_{label}",
            "policy": {"num_candidates": 1},
            "components": [],
            "selection": {"type": "candidate_zero"},
            "execution": {"action_offset": offset},
        }
        print(
            f"[eval-canonical] {label}  (offset {offset}, norm {meta.normalization['method']}, "
            f"obs_horizon {meta.obs_horizon}) ...",
            flush=True,
        )
        r = evaluate_system(
            system_cfg=cfg,
            eval_cfg=eval_cfg,
            policy_checkpoint=str(ckpt),
            num_episodes=args.count,
            output_path=out / f"eval_{label}.json",
            device=args.device,
            env=env,
            force=True,
        )
        ci = wilson_interval(r["success_count"], args.count)
        rows[label] = {
            "success_rate": r["success_rate"],
            "wilson_ci": list(ci),
            "policy_ms": 1000 * r["latency"]["mean_policy_s"],
            "offset": offset,
            "norm": meta.normalization["method"],
        }
    env.close()

    L = [
        "=" * 70,
        f"Authoritative state policies — diagnostic panel, n={args.count}",
        "=" * 70,
        f"{'policy':<12}{'success':>10}{'wilson 95% CI':>20}{'infer ms/query':>16}",
    ]
    L.append("-" * 58)
    for label, s in rows.items():
        ci = s["wilson_ci"]
        L.append(
            f"{label:<12}{s['success_rate']:>9.1%}{f'[{ci[0]:.1%},{ci[1]:.1%}]':>20}"
            f"{s['policy_ms']:>15.1f}"
        )
    print("\n".join(L))
    save_json({"n": args.count, "policies": rows}, out / "canonical_eval.json")
    print(f"\n[eval-canonical] wrote {out / 'canonical_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

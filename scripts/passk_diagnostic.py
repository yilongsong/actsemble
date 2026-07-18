#!/usr/bin/env python
"""pass@k via INDEPENDENT clean rollouts — branch-free upper bound on what
sampling+selection could buy. For each panel episode, run k independent clean
forward rollouts (different policy-sampling seeds) and count success if any
succeeds. No branching / no scratch env => cannot have the oracle's two-env
selection artifact. If pass@16 ~ Oracle@16 (~43%), the ceiling is real
(proposal-limited); if pass@16 >> that, the oracle is under-selecting.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location(
    "oh", str(REPO / "scripts/oracle_headroom.py")
)
oh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oh)

from actsemble.policies.diffusion.policy import DiffusionPolicy  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.evaluation.panels import panel_episodes  # noqa: E402
from actsemble.seed import derive_seed  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

N, KMAX = 48, 16
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
policy = DiffusionPolicy.from_checkpoint(str(oh.POLICY), device=dev, use_ema=True)
env = make_env(
    task_id=policy.meta.task_id,
    control_mode=policy.meta.controller,
    sim_backend=policy.meta.simulation_backend,
    obs_mode="state",
    render_mode=None,
)
roll = oh.OracleRollout(policy, env, env, dev)  # base_episode only uses the clean env

rows = []
for i, ep in enumerate(panel_episodes(oh.PANEL)[:N]):
    per = []
    for j in range(KMAX):
        pss = derive_seed(int(ep.policy_sampling_seed), "passk", j)
        per.append(bool(roll.base_episode(ep.env_seed, pss)["success_once"]))
    rows.append({"env_seed": int(ep.env_seed), "per_sample": per})
    if (i + 1) % 5 == 0:
        p16 = np.mean([any(r["per_sample"]) for r in rows])
        print(f"[passk] {i + 1}/{N}  pass@16 so far {p16:.1%}", flush=True)
env.close()

M = np.array([r["per_sample"] for r in rows], bool)  # [N, KMAX]


def passk(k):
    return float(np.mean(M[:, :k].any(axis=1)))


curve = {k: passk(k) for k in [1, 2, 4, 8, 16]}
report = {
    "n_episodes": len(rows),
    "kmax": KMAX,
    "pass_at_1": curve[1],
    "pass_at_16": curve[16],
    "pass_at_k_curve": curve,
    "mean_successes_per_16": float(M.sum(axis=1).mean()),
    "episodes_with_zero_success": int((M.sum(axis=1) == 0).sum()),
    "episodes_with_all16_success": int((M.sum(axis=1) == 16).sum()),
    "rows": rows,
}
out = REPO / "outputs/active_min/oracle/passk.json"
save_json(report, out)
print("\n=== pass@k (independent clean rollouts, no branching) ===")
for k, v in curve.items():
    print(f"  pass@{k:<2d} = {v:.1%}")
print(f"  mean successes per 16 samples: {report['mean_successes_per_16']:.2f}/16")
print(
    f"  episodes never solved in 16 tries: {report['episodes_with_zero_success']}/{len(rows)}"
)
print(
    f"  episodes solved by ALL 16 samples: {report['episodes_with_all16_success']}/{len(rows)}"
)
print("\ncompare: candidate_zero 38.7% · verifier 42.7% · Oracle@16 43.3% (fixed)")

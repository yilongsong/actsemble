#!/usr/bin/env python
"""Measure counterfactual fresh-plan revisions along committed H8 rollouts.

The system executes an ordinary eight-action committed chunk. At every
intermediate control step it additionally queries the deterministic ACT policy
and records what fresh slot 0 would have commanded at that same state. The
shadow query never affects execution. This isolates plan-revision magnitude on
the *same state distribution* and avoids comparing unrelated H8/latest paths.

Development-only diagnostic; no production code or weights are modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import Panel, panel_episodes  # noqa: E402
from actsemble.evaluation.evaluator import run_panel_episode  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.interface import ReplanningSystemBase  # noqa: E402
from actsemble.types import RobotAction  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402


class ShadowReplanH8System(ReplanningSystemBase):
    name = "shadow_replan_h8"

    def __init__(self, policy):
        super().__init__(policy, num_candidates=1, action_horizon=8)

    def _reset_state(self, *, episode_seed: int) -> None:
        super()._reset_state(episode_seed=episode_seed)
        self._phase = 0
        self._committed_chunk: np.ndarray | None = None
        self._rows: list[dict] = []

    def act(self, observation) -> RobotAction:
        self._history.append(self._frame(observation))
        ctx = self._context()
        record = {
            "replan_index": self._replan_index,
            "episode_seed": self.episode_seed,
            "control_step": len(self._rows),
            "shadow": self._phase != 0,
        }
        candidates, selected, _ = self._decide(ctx, record)
        fresh = candidates[selected].detach().cpu().numpy().astype(np.float32)
        if self._phase == 0 or self._committed_chunk is None:
            self._committed_chunk = fresh.copy()
        committed = self._committed_chunk[self._phase].copy()
        fresh0 = fresh[0].copy()

        if committed.shape == (7,):
            anchor = np.asarray(observation.state[:7], dtype=np.float32)
        else:
            anchor = np.zeros_like(committed)
        cr = committed - anchor
        fr = fresh0 - anchor
        cn = float(np.linalg.norm(cr))
        fn = float(np.linalg.norm(fr))
        cosine = float(np.dot(cr, fr) / (cn * fn + 1e-12))
        self._rows.append(
            {
                "phase": self._phase,
                "revision": float(np.linalg.norm(fresh0 - committed)),
                "committed_magnitude": cn,
                "fresh0_magnitude": fn,
                "direction_cosine": cosine,
            }
        )
        self._executed.append(committed.copy())
        self._phase = (self._phase + 1) % self.action_horizon
        self._replan_index += 1
        return RobotAction(value=committed)

    def probe_rows(self) -> list[dict]:
        return self._rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--panel-root", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    policy = load_policy(args.checkpoint, device=args.device, use_ema=True)
    env = make_env(
        task_id=policy.meta.task_id,
        control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend,
        obs_mode="state",
        render_mode=None,
    )
    system = ShadowReplanH8System(policy)
    seeds = panel_episodes(Panel("codex_shadow_h8", args.panel_root, args.count))
    episodes = []
    for i, ep in enumerate(seeds):
        if i % 10 == 0:
            print(f"[shadow-h8] {i}/{args.count}", flush=True)
        result, _ = run_panel_episode(
            env, system, ep, max_steps=100, pert_specs=[], capture_video=False
        )
        episodes.append(
            {
                "env_seed": ep.env_seed,
                "success": bool(result.success_once),
                "steps": int(result.num_steps),
                "rows": system.probe_rows(),
            }
        )
    env.close()

    rows = [r for ep in episodes for r in ep["rows"]]
    nonzero = [r for r in rows if r["phase"] != 0]
    revision = np.asarray([r["revision"] for r in nonzero])
    committed = np.asarray([r["committed_magnitude"] for r in nonzero])
    fresh = np.asarray([r["fresh0_magnitude"] for r in nonzero])
    cosine = np.asarray([r["direction_cosine"] for r in nonzero])
    successes = [ep["success"] for ep in episodes]
    count = int(sum(successes))
    summary = {
        "success_count": count,
        "num_episodes": args.count,
        "success_rate": count / args.count,
        "wilson_95": list(wilson_interval(count, args.count)),
        "nonboundary_steps": len(nonzero),
        "revision_mean": float(revision.mean()),
        "revision_median": float(np.median(revision)),
        "revision_over_committed_median": float(np.median(revision / (committed + 1e-12))),
        "committed_magnitude_mean": float(committed.mean()),
        "committed_magnitude_median": float(np.median(committed)),
        "fresh0_magnitude_mean": float(fresh.mean()),
        "fresh0_magnitude_median": float(np.median(fresh)),
        "direction_cosine_mean": float(cosine.mean()),
        "direction_cosine_median": float(np.median(cosine)),
        "opposite_direction_fraction": float(np.mean(cosine < 0)),
        "fresh_stay_committed_move_fraction": float(
            np.mean((fresh < 0.1) & (committed >= 0.1))
        ),
    }
    save_json(
        {
            "kind": "development_only_shadow_replan_h8_probe",
            "checkpoint": args.checkpoint,
            "panel": {"root": args.panel_root, "count": args.count},
            "summary": summary,
            "episodes": episodes,
        },
        args.output,
    )
    print(f"[shadow-h8] wrote {args.output}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

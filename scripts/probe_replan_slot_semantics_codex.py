#!/usr/bin/env python
"""Diagnostic-only probe for absolute-target dense replanning.

This script does not participate in training or the frozen evaluation protocol.
It compares, on one paired diagnostic seed bank:

* ordinary H8 execution (one plan committed for eight steps),
* the production temporal-latest system (fresh plan, slot 0),
* a second implementation of fresh-plan/slot-0 as an equivalence check,
* fresh plans while executing a fixed future slot (1, 2, 4, or 7), and
* fresh plans while cycling the executed slot through 0..7.

The fixed/cyclic variants separate three candidate mechanisms:

1. a slot-0-specific identity/stay attractor;
2. loss of the increasing lead encoded by later chunk slots; and
3. incoherence from changing the whole plan every control step.

Nothing here modifies policy weights, controller state, or repository runtime
code. Results are development-tier diagnostics only.
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
from actsemble.systems.factory import build_system  # noqa: E402
from actsemble.systems.interface import ReplanningSystemBase  # noqa: E402
from actsemble.types import RobotAction  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402


class FreshSlotSystem(ReplanningSystemBase):
    """Replan every step, then emit a selected slot of the fresh chunk."""

    def __init__(self, policy, *, fixed_slot: int | None, cycle: int | None = None):
        super().__init__(policy, num_candidates=1, action_horizon=1)
        if (fixed_slot is None) == (cycle is None):
            raise ValueError("choose exactly one of fixed_slot or cycle")
        self.fixed_slot = fixed_slot
        self.cycle = cycle
        self.name = f"fresh_slot_{fixed_slot}" if fixed_slot is not None else f"fresh_cycle_{cycle}"

    def _reset_state(self, *, episode_seed: int) -> None:
        super()._reset_state(episode_seed=episode_seed)
        self._control_step = 0
        self._slot_trace: list[int] = []
        self._lead_trace: list[float] = []
        self._revision_trace: list[float] = []
        self._chunk_lead_trace: list[list[float]] = []
        self._previous_chunk: np.ndarray | None = None

    def act(self, observation) -> RobotAction:
        self._history.append(self._frame(observation))
        ctx = self._context()
        record = {
            "replan_index": self._replan_index,
            "episode_seed": self.episode_seed,
            "control_step": self._control_step,
        }
        candidates, selected, _ = self._decide(ctx, record)
        chunk = candidates[selected].detach().cpu().numpy().astype(np.float32)
        slot = self.fixed_slot if self.fixed_slot is not None else self._control_step % int(self.cycle)
        assert slot is not None
        action = chunk[int(slot)].copy()

        # Absolute joint targets are measured by their controller stretch
        # ||target-q||. Delta-EE commands already represent a displacement, so
        # their command norm is the corresponding magnitude diagnostic.
        if action.shape == (7,):
            anchor = np.asarray(observation.state[:7], dtype=np.float32)
        else:
            anchor = np.zeros_like(action)
        self._slot_trace.append(int(slot))
        self._lead_trace.append(float(np.linalg.norm(action - anchor)))
        self._chunk_lead_trace.append(
            np.linalg.norm(chunk - anchor[None], axis=1).tolist()
        )
        if self._previous_chunk is not None:
            # Both terms target the current control time: the previous plan's
            # slot 1 and the freshly replanned slot 0.
            self._revision_trace.append(float(np.linalg.norm(chunk[0] - self._previous_chunk[1])))
        self._previous_chunk = chunk

        self._executed.append(action.copy())
        self._replan_index += 1
        self._control_step += 1
        return RobotAction(value=action)

    def probe_diagnostics(self) -> dict:
        return {
            "slots": self._slot_trace,
            "leads": self._lead_trace,
            "revisions": self._revision_trace,
            "chunk_leads": self._chunk_lead_trace,
        }


def summarize(rows: list[dict]) -> dict:
    successes = np.asarray([r["success"] for r in rows], dtype=bool)
    leads = np.concatenate([np.asarray(r["leads"], dtype=np.float64) for r in rows])
    revisions = np.concatenate(
        [np.asarray(r["revisions"], dtype=np.float64) for r in rows if r["revisions"]]
    ) if any(r["revisions"] for r in rows) else np.zeros(0, dtype=np.float64)
    n = len(rows)
    count = int(successes.sum())
    return {
        "success_count": count,
        "num_episodes": n,
        "success_rate": count / n,
        "wilson_95": list(wilson_interval(count, n)),
        "lead_mean": float(leads.mean()),
        "lead_median": float(np.median(leads)),
        "lead_p10": float(np.quantile(leads, 0.1)),
        "fraction_lead_lt_0p1": float(np.mean(leads < 0.1)),
        "same_time_plan_revision_mean": float(revisions.mean()) if len(revisions) else None,
        "same_time_plan_revision_median": float(np.median(revisions)) if len(revisions) else None,
        "successes": successes.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "outputs/active_min/rerun_jointpos/act/selected_policy.pt"),
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--panel-root", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default=str(REPO / "outputs/active_min/rerun_jointpos/autopsy/codex_slot_probe.json"),
    )
    args = parser.parse_args()

    policy = load_policy(args.checkpoint, device=args.device, use_ema=True)
    env = make_env(
        task_id=policy.meta.task_id,
        control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend,
        obs_mode="state",
        render_mode=None,
    )
    panel = Panel("codex_slot_probe", args.panel_root, args.count)
    seeds = panel_episodes(panel)

    systems = {
        "h8": build_system(
            {
                "policy": {"num_candidates": 1},
                "selection": {"type": "candidate_zero"},
                "execution": {"action_horizon": 8},
            },
            policy,
            [],
        ),
        "production_latest": build_system(
            {
                "policy": {"num_candidates": 1},
                "selection": {"type": "temporal_ensemble", "aggregation": "latest"},
                "execution": {"replan_interval": 1},
            },
            policy,
            [],
        ),
        "fresh_slot_0": FreshSlotSystem(policy, fixed_slot=0),
        "fresh_slot_1": FreshSlotSystem(policy, fixed_slot=1),
        "fresh_slot_2": FreshSlotSystem(policy, fixed_slot=2),
        "fresh_slot_4": FreshSlotSystem(policy, fixed_slot=4),
        "fresh_slot_7": FreshSlotSystem(policy, fixed_slot=7),
        "fresh_cycle_8": FreshSlotSystem(policy, fixed_slot=None, cycle=8),
    }

    all_rows: dict[str, list[dict]] = {}
    for name, system in systems.items():
        print(f"[slot-probe] {name}: {args.count} paired episodes", flush=True)
        rows = []
        for ep in seeds:
            result, _ = run_panel_episode(
                env,
                system,
                ep,
                max_steps=100,
                pert_specs=[],
                capture_video=False,
            )
            extra = system.probe_diagnostics() if isinstance(system, FreshSlotSystem) else {}
            # For the production systems, reconstruct raw lead from their private
            # executed-action list only; q is unavailable after the rollout, so
            # leave lead diagnostics empty rather than inventing a proxy.
            rows.append(
                {
                    "env_seed": ep.env_seed,
                    "success": bool(result.success_once),
                    "steps": int(result.num_steps),
                    "action_digest": result.diagnostics.get("action_digest"),
                    "leads": extra.get("leads", []),
                    "revisions": extra.get("revisions", []),
                    "slots": extra.get("slots", []),
                    "chunk_leads": extra.get("chunk_leads", []),
                }
            )
        all_rows[name] = rows

    summaries = {}
    for name, rows in all_rows.items():
        if any(r["leads"] for r in rows):
            summaries[name] = summarize(rows)
        else:
            count = sum(r["success"] for r in rows)
            summaries[name] = {
                "success_count": count,
                "num_episodes": len(rows),
                "success_rate": count / len(rows),
                "wilson_95": list(wilson_interval(count, len(rows))),
                "successes": [r["success"] for r in rows],
            }

    # The independent slot-0 implementation must be command-identical to the
    # production temporal-latest implementation on this deterministic policy.
    prod = all_rows["production_latest"]
    independent = all_rows["fresh_slot_0"]
    digest_match = [a["action_digest"] == b["action_digest"] for a, b in zip(prod, independent)]

    output = {
        "kind": "development_only_replan_slot_semantics_probe",
        "checkpoint": args.checkpoint,
        "panel": {"root": args.panel_root, "count": args.count},
        "production_latest_vs_independent_slot0_digest_match_rate": float(np.mean(digest_match)),
        "summaries": summaries,
        "episodes": all_rows,
    }
    save_json(output, args.output)
    print(f"[slot-probe] wrote {args.output}")
    for name, row in summaries.items():
        print(
            f"  {name:>18}: {row['success_count']:>2}/{row['num_episodes']} "
            f"= {row['success_rate']:.1%}"
            + (f"  median lead {row['lead_median']:.3f}" if "lead_median" in row else "")
        )
    print(
        "  production/latest action-digest equivalence: "
        f"{sum(digest_match)}/{len(digest_match)}"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

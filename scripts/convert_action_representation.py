#!/usr/bin/env python
"""Convert a recorded dataset to a different ACTION REPRESENTATION.

v1 target: ``pd_joint_pos`` (absolute target joint positions — ACT's native
action space).

Method — GLUED DRIVE-TARGET CAPTURE. Two simpler methods fail:
  * naive ``a_t = qpos_{t+1}``: a PD controller *chases* targets (never
    reaches them in one step) -> ~0.1 rad systematic lag, 0% replay success;
  * free open-loop capture replay: correct targets (0.0004 rad tracking) but
    GPU-PhysX nondeterminism kills contact-rich open-loop replays -> only
    138/400 episodes re-earn success (would confound the representation A/B
    with dataset size).
The fix exploits that the source controller is stateless in its target
computation (``use_target: False``): ``_target_qpos = IK(qpos_t (+) scaled
delta_t)`` is a pointwise function of the recorded state and action, and does
not depend on block state. So per step: TELEPORT the sim to the recorded
state[t] (qpos+qvel exact, block pose exact, block velocity unrecorded ->
zeroed — irrelevant to the capture), step once with the recorded delta action,
read ``controller.controllers['arm']._target_qpos``, discard the stepped
state. The written dataset keeps the RECORDED states bit-for-bit; only the
action arrays are new. All 400 episodes retained; no drift, no re-earning.

``--validate N`` reports (a) PRIMARY: one-step teleport tracking — from
recorded state[t], apply the captured target under ``pd_joint_pos``, compare
achieved qpos to recorded next_state[t] (no compounding); (b) SECONDARY:
free open-loop replay success (descriptive only — fragile to sim noise for
ANY correct action sequence, as measured on the source controller too).

    python scripts/convert_action_representation.py --validate 25
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.data.writer import write_dataset  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.types import EpisodeRecord  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

TARGET_CONTROLLER = "pd_joint_pos"
QPOS = slice(0, 7)
OBJ = slice(24, 31)  # extra.obj_pose (pos 3 + quat 4)


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


QVEL = slice(7, 14)


def _teleport(u, state: np.ndarray) -> float:
    """Set the sim to a recorded 31-dim state (block velocity zeroed — it is
    unrecorded); returns max obs error over the recorded dims."""
    template = u.get_state_dict()
    art = template["articulations"][next(iter(template["articulations"]))]
    art[..., -14:-7] = torch.as_tensor(state[QPOS], dtype=art.dtype, device=art.device)
    art[..., -7:] = torch.as_tensor(state[QVEL], dtype=art.dtype, device=art.device)
    actors = template["actors"]
    tee = actors["Tee" if "Tee" in actors else next(k for k in actors if "tee" in k.lower())]
    tee[..., :7] = torch.as_tensor(state[OBJ], dtype=tee.dtype, device=tee.device)
    tee[..., 7:] = 0.0
    u.set_state_dict(template)
    obs = _to_np(u.get_obs()).reshape(-1)
    return float(np.max(np.abs(obs - state)))


def probe_bounds(meta) -> tuple[np.ndarray, np.ndarray]:
    env = make_env(task_id=meta.task_id, control_mode=TARGET_CONTROLLER,
                   sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)
    try:
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    finally:
        env.close()
    if low.shape[0] != 7:
        raise SystemExit(f"expected 7-dim {TARGET_CONTROLLER} space, got {low.shape}")
    return low, high


def convert(source: str, out: str) -> dict:
    reader = DatasetReader(source)
    meta = reader.metadata
    low, high = probe_bounds(meta)

    env = make_env(task_id=meta.task_id, control_mode=meta.controller,
                   sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)
    u = env.unwrapped
    episodes = []
    teleport_errs = []
    try:
        env.reset(seed=0)
        arm = u.agent.controller.controllers["arm"]
        for eid in reader.episode_ids:
            ep = reader.episode(eid)
            T = len(ep.action)
            targets = np.empty((T, 7), dtype=np.float32)
            for t in range(T):
                # glue to the recorded state, one dynamic step, capture target
                teleport_errs.append(_teleport(u, ep.state[t]))
                env.step(torch.as_tensor(ep.action[t][None], dtype=torch.float32))
                targets[t] = _to_np(arm._target_qpos).reshape(-1)
            actions = np.clip(targets, low, high)
            # schema contract: previous_action[0] == 0 (delta-space no-op
            # convention, enforced by the dataset validator). The field is
            # unused by all current policies (include_previous_action: False);
            # a policy that consumes it in absolute space must special-case t=0.
            prev = np.concatenate([np.zeros((1, 7), np.float32), actions[:-1]])
            episodes.append(EpisodeRecord(
                episode_id=eid, state=ep.state, previous_action=prev,
                action=actions, next_state=ep.next_state,
                step_index=ep.step_index,
            ))
    finally:
        env.close()

    new_meta = dataclasses.replace(
        meta,
        controller=TARGET_CONTROLLER,
        action_dimension=7,
        action_definition=json.dumps({
            "semantics": "absolute target joint positions (PD drive setpoints)",
            "frame": "joint space",
            "units": "radians, unnormalized",
            "bounds": [low.tolist(), high.tolist()],
            "scaling": "none (controller consumes raw joint targets)",
            "clipping_rules": "captured drive targets clipped to joint limits at export",
        }),
        source_dataset=f"derived_from:{source} (action-representation conversion)",
        dataset_hash="",
    )
    new_meta.extra = {
        **(meta.extra or {}),
        "conversion_method": "drive_target_capture_glued",
        "derived_from_dataset": str(source),
        "derived_from_hash": meta.dataset_hash,
        "target_controller": TARGET_CONTROLLER,
        "previous_action_t0": "zeros (schema contract; field unused by current policies)",
    }
    h = write_dataset(out, episodes, new_meta)
    stats = {
        "episodes": len(episodes),
        "max_teleport_obs_err": float(np.max(teleport_errs)),
    }
    print(f"[convert] wrote {out}  hash={h[:12]}  episodes={stats['episodes']}"
          f"  teleport_obs_err<={stats['max_teleport_obs_err']:.2e}"
          f"  (states = recorded originals; actions = captured drive targets)")
    return stats


def validate(out: str, n: int) -> dict:
    reader = DatasetReader(out)
    meta = reader.metadata
    env = make_env(task_id=meta.task_id, control_mode=TARGET_CONTROLLER,
                   sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)
    u = env.unwrapped
    rows = []
    try:
        env.reset(seed=0)
        for eid in reader.episode_ids[:n]:
            ep = reader.episode(eid)
            # PRIMARY: one-step teleport tracking (no compounding)
            step_errs = []
            for t in range(len(ep.action)):
                _teleport(u, ep.state[t])
                o, *_ = env.step(torch.as_tensor(ep.action[t][None], dtype=torch.float32))
                o = _to_np(o).reshape(-1)
                step_errs.append(float(np.max(np.abs(o[QPOS] - ep.next_state[t][QPOS]))))
            # SECONDARY: free open-loop replay (descriptive; sim-noise fragile)
            _teleport(u, ep.state[0])
            succ = False
            for t in range(len(ep.action)):
                _o, _r, _te, _tr, info = env.step(
                    torch.as_tensor(ep.action[t][None], dtype=torch.float32)
                )
                s = info.get("success")
                if s is not None and bool(_to_np(s).reshape(-1)[0]):
                    succ = True
            rows.append({"episode_id": eid, "open_loop_success": succ,
                         "one_step_qpos_err_mean": float(np.mean(step_errs)),
                         "one_step_qpos_err_max": float(np.max(step_errs))})
    finally:
        env.close()

    report = {
        "n": len(rows),
        "one_step_qpos_err_mean": float(np.mean([r["one_step_qpos_err_mean"] for r in rows])),
        "one_step_qpos_err_max": float(np.max([r["one_step_qpos_err_max"] for r in rows])),
        "open_loop_success_rate": float(np.mean([r["open_loop_success"] for r in rows])),
        "episodes": rows,
    }
    save_json(report, Path(out).with_suffix(".validation.json"))
    print(f"[validate] {TARGET_CONTROLLER}: one-step qpos err mean "
          f"{report['one_step_qpos_err_mean']:.5f} max {report['one_step_qpos_err_max']:.5f}"
          f" | open-loop replay success {report['open_loop_success_rate']:.1%} (descriptive)"
          f" on n={report['n']}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(REPO / "data/active_min/subset_0400.h5"))
    ap.add_argument("--out", default=str(REPO / "data/active_min/subset_0400_jointpos.h5"))
    ap.add_argument("--validate", type=int, default=25, help="episodes to replay-validate (0=skip)")
    args = ap.parse_args()

    convert(args.source, args.out)
    if args.validate > 0:
        validate(args.out, args.validate)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Generate a PickCube dataset with a SLOW, NOISY, teleop-like approach.

Why: the official RL bundle is bang-bang. Its recorded actions saturate at
dz=-1.0 for three steps (full-speed dive), then +1.0 (full-speed lift), so the
whole approach occupies ~6 of 50 frames and the arm never rises above its own
start height. Two-thirds of each episode is then a frozen hold at the goal.
That leaves ~8 approach poses per episode -- the entire vocabulary a recovery
policy can reset to -- and every one of them is a pose the demonstrator only
ever passed through at 2-5 cm/frame.

This demonstrator is deliberately unlike that:

  slow      a per-step translation cap (--descent-cap, in normalized action
            units) replaces the saturated dive, so the approach spans many
            frames at a controlled speed.
  noisy     temporally CORRELATED noise (Ornstein-Uhlenbeck, --noise-alpha)
            rather than white jitter, so the path wanders and corrects the way
            a human teleoperator's does instead of tracking a straight line.
            White noise would average out over a chunk and change nothing.
  no dead   the hold is trimmed to --hold frames once success registers,
    tail    instead of freezing for ~33 frames to fill a fixed horizon.

Phases: HOVER above the cube -> DESCEND slowly with wander -> CLOSE -> LIFT ->
CARRY to goal -> brief HOLD. Only successful episodes are written, matching the
existing dataset contract (s_t, a_t, s_{t+1}).

    python scripts/generate_pickcube_teleop.py --n 100 --out outputs/pickcube/teleop_v1/teleop_100.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.data.schema import DatasetMetadata  # noqa: E402
from actsemble.data.writer import write_dataset, write_private_provenance  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.types import EpisodeRecord  # noqa: E402

TCP = slice(19, 22)
TCP_QUAT = slice(22, 26)
GOAL = slice(26, 29)
OBJ = slice(29, 32)
OBJ_QUAT = slice(32, 36)
GRIP_Q = slice(7, 9)


def yaw_of(q) -> float:
    """Rotation about world z, from a (w,x,y,z) quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def face_error(tcp_quat, obj_quat) -> float:
    """Signed wrist rotation needed to line the jaws up with the cube's nearest
    face, in (-45deg, +45deg]. The cube is 4-fold symmetric about z, so any of
    its four faces will do and the hand never turns more than 45 degrees."""
    d = yaw_of(obj_quat) - yaw_of(tcp_quat)
    return (d + np.pi / 4) % (np.pi / 2) - np.pi / 4

TASK, CTRL, BACKEND = "PickCube-v1", "pd_ee_delta_pose", "physx_cuda"


def state42(raw) -> np.ndarray:
    a = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
    return a.reshape(-1)


class Wander:
    """Ornstein-Uhlenbeck noise: correlated across time, so the trajectory
    wanders smoothly instead of jittering. alpha is the retention factor
    (0 = white noise, ->1 = very smooth drift)."""

    def __init__(self, rng, sigma, alpha, dim=3):
        self.rng, self.sigma, self.alpha = rng, sigma, alpha
        self.x = np.zeros(dim)

    def __call__(self):
        self.x = self.alpha * self.x + np.sqrt(1 - self.alpha**2) * self.sigma * self.rng.standard_normal(self.x.shape)
        return self.x.copy()


def run_episode(env, seed, args, rng):
    """Scripted teleop-like pick. Returns (states, actions, success) or None."""
    raw, _ = env.reset(seed=seed)
    s = state42(raw)
    wander = Wander(rng, args.noise_sigma, args.noise_alpha)
    # a per-episode bias: a human doesn't approach dead-centre every time
    bias = rng.normal(0, args.approach_bias, 3) * np.array([1, 1, 0.3])

    states, actions, next_states = [], [], []
    phase, closed_for, held = "HOVER", 0, 0
    success = False

    for t in range(args.max_steps):
        s = state42(raw)
        tcp, obj, goal = s[TCP], s[OBJ], s[GOAL]
        grip_open = float(s[GRIP_Q].mean()) > 0.03

        if phase == "HOVER":
            target = obj + np.array([0.0, 0.0, args.hover_height]) + bias
            if np.linalg.norm(tcp - target) < args.hover_tol:
                phase = "DESCEND"
        elif phase == "DESCEND":
            # Only give up height once laterally lined up. Without this the
            # wander keeps pushing the hand sideways right up to contact and
            # the jaws close wherever it happened to be -- 38% of grasps landed
            # >12mm off a 20mm-half-width cube. A teleoperator lines up over
            # the object and then comes down.
            lat = float(np.linalg.norm(tcp[:2] - obj[:2]))
            target = obj.copy()
            # SOFT alignment gate. The previous rule was hard -- hold height
            # until lat <= align_tol -- and it put 33.2% of all demo frames in
            # a hover where the height command is exactly zero and the lateral
            # command is dominated by zero-mean OU noise. BC fits the
            # conditional mean, so it learns ~zero action at those states, and
            # hovering is absorbing: the state never changes, so the policy
            # emits ~zero again and the hand parks above an unmoved cube for
            # the whole horizon. Instead, keep a STANDOFF HEIGHT proportional
            # to the lateral error: the hand always descends as it aligns, so
            # every state carries a non-zero, deterministic vertical signal
            # and no state is absorbing.
            if args.hard_gate:
                # EXACT v1 behaviour: hold the current height until aligned.
                if lat > args.align_tol:
                    target[2] = tcp[2]
            else:
                standoff = max(0.0, lat - args.align_tol) * args.descend_couple
                target[2] = obj[2] + standoff
            if tcp[2] - obj[2] < args.grasp_gap and lat <= args.align_tol:
                phase = "CLOSE"
        elif phase == "CLOSE":
            target = tcp.copy()
            closed_for += 1
            if closed_for >= args.close_steps:
                phase = "LIFT"
        elif phase == "LIFT":
            target = np.array([obj[0], obj[1], goal[2]])
            if obj[2] > args.lift_clear:
                phase = "CARRY"
        else:  # CARRY / HOLD
            target = goal.copy()

        err = target - tcp
        # proportional term, capped -> a controlled speed instead of bang-bang
        cmd = args.kp * err
        if phase in ("HOVER", "DESCEND"):
            # Taper the wander as the hand closes on the cube: a teleoperator
            # wanders while traversing and steadies up for the grasp. Without
            # the taper the noise is still at full amplitude at contact.
            gap = max(float(tcp[2] - obj[2]), 0.0)
            # Taper on BOTH the height gap and the lateral error. Tapering on
            # height alone left the wander at near-full amplitude throughout
            # the old hover (where the gap is held constant by construction),
            # which both lengthened the hover and filled it with pure noise.
            lat_now = float(np.linalg.norm(tcp[:2] - obj[:2]))
            taper = min(1.0, gap / args.noise_taper) * min(1.0, lat_now / args.noise_lat_taper)
            cmd = cmd + wander() * taper
            cap = args.descent_cap
        else:
            cap = args.carry_cap
        n = np.linalg.norm(cmd)
        if n > cap:
            cmd = cmd * (cap / n)
        # Wrist: turn to meet the cube's nearest face while travelling, then
        # hold. Leaving these dims at zero makes the hand rigid for the whole
        # episode -- it grasps whatever yaw it started at, which is 15 deg off
        # the cube on average and looks visibly wrong.
        if phase in ("HOVER", "DESCEND"):
            # NOTE the minus: this controller's rotation dims are axis-angle in
            # a root-ALIGNED BODY frame, whose z runs opposite to world z for
            # the top-down grasp pose, so a positive drz turns the wrist the
            # wrong way. Verified empirically -- with +kyaw the wrist converges
            # to 44.3 deg misalignment (the worst case for a 4-fold symmetric
            # cube, i.e. the anti-aligned fixed point); with -kyaw, 2.1 deg.
            yaw_err = face_error(s[TCP_QUAT], s[OBJ_QUAT])
            drz = np.clip(-args.kyaw * yaw_err + args.yaw_noise * rng.standard_normal(),
                          -args.yaw_cap, args.yaw_cap)
        else:
            drz = 0.0
        grip = 1.0 if (phase in ("HOVER", "DESCEND") and grip_open) else -1.0
        act = np.concatenate([cmd, [0.0, 0.0, drz], [grip]]).astype(np.float32)
        act = np.clip(act, -1.0, 1.0)

        states.append(s.copy())
        actions.append(act.copy())
        raw, _r, _te, _tr, info = env.step(torch.as_tensor(act.reshape(1, -1)))
        next_states.append(state42(raw).copy())

        ok = info.get("success")
        ok = bool(ok.any()) if torch.is_tensor(ok) else bool(ok)
        if ok:
            success = True
            held += 1
            if held >= args.hold:
                break
        elif success:
            held = 0  # lost it again; keep going

    if not success:
        return None
    S = np.asarray(states, np.float32)
    A = np.asarray(actions, np.float32)
    NS = np.asarray(next_states, np.float32)
    P = np.concatenate([np.zeros((1, A.shape[1]), np.float32), A[:-1]], 0)
    return S, A, NS, P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="successful episodes to collect")
    ap.add_argument("--out", default=str(REPO / "outputs/pickcube/teleop_v1/teleop_100.h5"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=90)
    # --- the two things being changed vs the RL bundle ---
    ap.add_argument("--descent-cap", type=float, default=0.16,
                    help="max normalized translation per step during approach (RL bundle saturates at 1.0)")
    ap.add_argument("--noise-sigma", type=float, default=0.05, help="OU noise scale on the approach")
    ap.add_argument("--noise-alpha", type=float, default=0.85, help="OU retention: higher = smoother wander")
    ap.add_argument("--approach-bias", type=float, default=0.012, help="per-episode offset (m), human off-centre-ness")
    # --- shape of the script ---
    ap.add_argument("--kp", type=float, default=6.0)
    ap.add_argument("--hover-height", type=float, default=0.12)
    ap.add_argument("--hover-tol", type=float, default=0.035)
    ap.add_argument("--grasp-gap", type=float, default=0.005)
    ap.add_argument("--align-tol", type=float, default=0.006,
                    help="lateral error (m) that must be met before descending the last bit / closing")
    ap.add_argument("--noise-taper", type=float, default=0.06,
                    help="height above the cube (m) over which the wander fades to zero")
    ap.add_argument("--descend-couple", type=float, default=2.0,
                    help="standoff height held per metre of lateral error during DESCEND: the "
                         "hand descends continuously as it aligns. NOTE 0 does NOT restore the "
                         "old gate -- it descends straight onto the cube regardless of "
                         "alignment. Use --hard-gate for the v1 behaviour.")
    ap.add_argument("--hard-gate", action="store_true",
                    help="restore the exact v1 DESCEND rule (hold current height while lateral "
                         "error > align_tol). Kept so the hold-length experiment can change ONE "
                         "variable against the v1 baseline (47.0%%); the soft gate on its own "
                         "measured 44.2%% [39.9,48.6], i.e. no improvement.")
    ap.add_argument("--noise-lat-taper", type=float, default=0.03,
                    help="lateral error (m) below which the wander fades out, so the hand "
                         "steadies up for the fine alignment instead of being pushed around")
    ap.add_argument("--kyaw", type=float, default=1.2,
                    help="proportional gain turning the wrist onto the cube's nearest face")
    ap.add_argument("--yaw-cap", type=float, default=0.35, help="max normalized wrist-rotation per step")
    ap.add_argument("--yaw-noise", type=float, default=0.03, help="jitter on the wrist command")
    ap.add_argument("--close-steps", type=int, default=4)
    ap.add_argument("--lift-clear", type=float, default=0.06)
    ap.add_argument("--carry-cap", type=float, default=0.35)
    ap.add_argument("--hold", type=int, default=4, help="frames to hold after success (RL bundle holds ~33)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    env = make_env(task_id=TASK, control_mode=CTRL, sim_backend=BACKEND,
                   obs_mode="state", render_mode=None, max_episode_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)
    episodes, tried, lens = [], 0, []
    seed = args.seed * 100000
    while len(episodes) < args.n and tried < args.n * 6:
        tried += 1
        out = run_episode(env, seed, args, rng)
        seed += 1
        if out is None:
            continue
        S, A, NS, P = out
        episodes.append(EpisodeRecord(
            episode_id=f"ep_{len(episodes):05d}", state=S, previous_action=P,
            action=A, next_state=NS, step_index=np.arange(len(S), dtype=np.int64)))
        lens.append(len(S))
        if len(episodes) % 20 == 0:
            print(f"[teleop] {len(episodes)}/{args.n} episodes ({tried} attempts)", flush=True)
    env.close()
    if not episodes:
        print("[teleop] ERROR: no successful episodes", file=sys.stderr)
        return 1

    import mani_skill

    meta = DatasetMetadata(
        # MUST be the real version: evaluate_system compares a checkpoint's
        # recorded metadata against the live env and refuses to run on any
        # mismatch. An empty string is not "unknown", it is a value that never
        # matches -- it passed screening/confirmation (which skip the check)
        # and only blew up at the 500-episode final test.
        simulator="ManiSkill3", simulator_version=mani_skill.__version__, task_id=TASK,
        robot="panda", observation_mode="state",
        state_dimension=int(episodes[0].state.shape[1]),
        state_layout=json.dumps({}), controller=CTRL,
        action_dimension=int(episodes[0].action.shape[1]),
        action_definition=json.dumps({
            "semantics": "normalized end-effector delta pose: xyz translation then axis-angle rotation",
            "units": "normalized [-1, 1] per dimension",
            "bounds": [[-1.0] * 7, [1.0] * 7],
        }),
        control_frequency=20.0, simulation_backend=BACKEND,
        source_dataset="scripted_teleop_like (scripts/generate_pickcube_teleop.py)",
        generation_or_replay_seed=int(args.seed),
    )
    meta.extra["conversion_method"] = "scripted_generation"
    meta.extra["demonstrator"] = "scripted teleop-like: capped speed + OU-correlated wander"
    for k in ("descent_cap", "noise_sigma", "noise_alpha", "approach_bias", "kp",
              "hover_height", "hold", "max_steps"):
        meta.extra[f"gen_{k}"] = getattr(args, k)
    h = write_dataset(args.out, episodes, meta)
    # Sidecar contract (validation.validate_success_only_provenance): the .h5
    # never stores success flags, so the auditable record that every exported
    # episode succeeded lives here. Attempts that never registered success are
    # dropped in run_episode and counted as rejected.
    write_private_provenance(args.out, {
        "success_only": True,
        "attempts": tried,
        "kept": len(episodes),
        "rejected_count": tried - len(episodes),
        "exported_episodes": [
            {"episode_id": e.episode_id, "source_success": True, "length": len(e)}
            for e in episodes
        ],
        "args": vars(args),
    })
    from actsemble.data.reader import DatasetReader
    from actsemble.data.validation import validate_dataset, validate_success_only_provenance
    stats = validate_dataset(DatasetReader(args.out))
    validate_success_only_provenance(args.out)
    print(f"\n[teleop] validated: {stats['num_episodes']} episodes, "
          f"{stats['num_transitions']} transitions, hash OK")
    print(f"[teleop] wrote {len(episodes)} episodes -> {args.out}")
    print(f"[teleop] success rate {len(episodes)}/{tried} = {len(episodes)/tried:.0%}")
    print(f"[teleop] episode length: min {min(lens)} median {int(np.median(lens))} max {max(lens)}")
    print(f"[teleop] dataset hash {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

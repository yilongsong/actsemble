#!/usr/bin/env python
"""Why does PickCube executed recovery capture ~0 of the +17.0pt teleport
headroom (recovery_oracle_pickcube.py, 2026-07-20: J_ceil_tp 88.4% but
J_ceil_ex 71.7% ~= J_reroll 71.4%)? Re-run the EX-winning (boundary,
candidate) for a sample of failures with full instrumentation to
discriminate three explanations:

  (a) NULL-SPACE MISMATCH (the PushT debug-viz finding, generalized): the
      controller converges in 3-DOF TIP space but the redundant 7-DOF arm
      lands in a different config -> tip_err small yet qpos_arm7 dist large.
  (b) GRASP MECHANICS (PickCube-specific, no PushT analog): tip AND arm
      converge but the scripted close-and-hold does not re-establish
      contact -> is_grasped at handback != candidate's grasp state.
  (c) BUDGET: the lift-translate-descend-hold simply doesn't finish in the
      remaining horizon -> hit_budget.

The oracle JSON stores no raw sim snapshots, so each target boundary state
is reconstructed by re-running the SAME reference rollout (same env_seed +
policy_sampling_seed) to the boundary step. physx_cuda run-to-run noise
means the replayed state only ~matches the original; that is fine here --
we are measuring whether the controller reaches the candidate target it is
GIVEN, not reproducing an exact episode.

    python scripts/diagnose_pickcube_transport.py --n 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from recovery_oracle_pickcube import MAX_STEPS, STANDALONE, Runner, _success  # noqa: E402

from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.recovery import (  # noqa: E402
    PC_GRASPED,
    PC_GRIP_QPOS,
    PC_OBJ_POS,
    PC_QPOS,
    PC_TCP_POS,
    PickCubeReturnController,
)
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402

ARM7 = slice(0, 7)


def ex_winner(row):
    """(boundary_index, candidate_index, boundary_row) that produced ex_final
    -- argmax over best per-boundary v_ex, matching the oracle's own logic."""
    g = row["grid"]
    bi = int(np.argmax([max(r["v_ex"]) for r in g]))
    bj = int(np.argmax(g[bi]["v_ex"]))
    return bi, bj, g[bi]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--oracle-json",
        default=str(REPO / "outputs/pickcube/recovery_oracle/recovery_oracle.json"),
    )
    ap.add_argument(
        "--policy-checkpoint",
        default=str(
            REPO
            / "outputs/pickcube/ndemos_v1/ndemos_100/seed_0/policy/selected_policy.pt"
        ),
    )
    ap.add_argument("--n", type=int, default=25, help="sample size, biggest tp-ex gap first")
    ap.add_argument(
        "--modes", default="grasp_slip_at_pick,grasp_failure_no_close"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out",
        default=str(
            REPO / "outputs/pickcube/recovery_oracle/transport_diagnosis.json"
        ),
    )
    args = ap.parse_args()

    oracle = json.load(open(args.oracle_json))
    modes = set(args.modes.split(","))
    seed_by_env = {e["env_seed"]: e["seed"] for e in oracle["reference"]}
    # only failures with a positive tp-ex gap are informative (tp recovered,
    # ex did not): those are exactly the episodes where transport lost value.
    fails = [
        r
        for r in oracle["failures"]
        if r["mode"] in modes and r["grid"] and (r["tp_final"] - r["ex_final"]) > 0.05
    ]
    fails.sort(key=lambda r: -(r["tp_final"] - r["ex_final"]))
    sample = fails[: args.n]
    print(
        f"[diag] {len(sample)} failures (tp-ex gap > 0.05, modes={sorted(modes)}), "
        f"largest-gap first",
        flush=True,
    )

    policy = load_policy(args.policy_checkpoint, device=args.device)
    env = make_env(
        task_id=policy.meta.task_id,
        control_mode=policy.meta.controller,
        sim_backend=policy.meta.simulation_backend,
        obs_mode="state",
        render_mode=None,
        max_episode_steps=MAX_STEPS,
    )
    system = build_system(STANDALONE, policy, [])
    rn = Runner(env, system)

    rows = []
    for i, row in enumerate(sample):
        _, bj, b = ex_winner(row)
        target_t = b["t"]
        cand_qpos = np.array(b["cand_qpos"][bj])
        cand_tip = np.array(b["cand_tip"][bj])
        cand_grasped = bool(b["cand_grasped"][bj])

        # reconstruct the boundary state: re-run the reference rollout to t
        seed = seed_by_env[row["env_seed"]]
        rn.start(seed)
        raw, _ = env.reset(seed=row["env_seed"])
        for _ in range(target_t):
            raw, _r, _te, _tr, info = rn.act_env(raw)

        # drive the scripted return to the candidate, instrumented
        ctl = PickCubeReturnController(cand_tip, cand_grasped, max_steps=20)
        obj0 = rn.state42(raw)[PC_OBJ_POS].copy()
        budget = MAX_STEPS - target_t
        used, once = 0, False
        while not ctl.done(rn.state42(raw)[PC_TCP_POS]) and used < budget:
            raw, _r, _te, _tr, info = rn.step_raw(ctl.action(rn.state42(raw)[PC_TCP_POS]))
            used += 1
            once = once or _success(info)
        s = rn.state42(raw)

        tip_err = float(np.linalg.norm(s[PC_TCP_POS] - cand_tip))
        qpos_arm7 = float(np.linalg.norm(s[ARM7] - cand_qpos[ARM7]))
        qpos_fingers = float(np.linalg.norm(s[PC_GRIP_QPOS] - cand_qpos[PC_GRIP_QPOS]))
        grasped_hb = bool(s[PC_GRASPED][0] > 0.5)
        tip_conv = tip_err < 0.015
        rec = {
            "env_seed": row["env_seed"],
            "mode": row["mode"],
            "target_t": target_t,
            "tp_final": row["tp_final"],
            "ex_final": row["ex_final"],
            "target_grasped": cand_grasped,
            "steps_used": used,
            "hit_budget": used >= budget or ctl.steps >= ctl.max_steps,
            "tip_err_handback": tip_err,
            "tip_converged": tip_conv,
            "qpos_arm7_dist": qpos_arm7,
            "qpos_fingers_dist": qpos_fingers,
            "grasped_at_handback": grasped_hb,
            "grasp_matches_target": grasped_hb == cand_grasped,
            "cube_bump": float(np.linalg.norm(s[PC_OBJ_POS] - obj0)),
            "succeeded_during_return": once,
        }
        rows.append(rec)
        print(
            f"[diag] {i + 1}/{len(sample)} seed={row['env_seed']} {row['mode']} "
            f"tp={row['tp_final']:.2f} ex={row['ex_final']:.2f} | "
            f"tip={tip_err:.4f}{'*' if tip_conv else ' '} arm7={qpos_arm7:.3f} "
            f"fing={qpos_fingers:.3f} grasp_ok={rec['grasp_matches_target']} "
            f"budget={'HIT' if rec['hit_budget'] else 'ok'} bump={rec['cube_bump']:.4f}",
            flush=True,
        )
    env.close()

    n = len(rows)
    if n:
        tip_conv = sum(r["tip_converged"] for r in rows)
        arm_off = sum(r["tip_converged"] and r["qpos_arm7_dist"] > 0.3 for r in rows)
        grasp_bad = sum(r["tip_converged"] and not r["grasp_matches_target"] for r in rows)
        budget = sum(r["hit_budget"] for r in rows)
        bumped = sum(r["cube_bump"] > 0.02 for r in rows)
        summary = {
            "n": n,
            "tip_converged_frac": tip_conv / n,
            "of_tip_conv_arm_nullspace_off_frac": arm_off / n,
            "of_tip_conv_grasp_mismatch_frac": grasp_bad / n,
            "hit_budget_frac": budget / n,
            "cube_disturbed_by_return_frac": bumped / n,
            "qpos_arm7_dist_median": float(np.median([r["qpos_arm7_dist"] for r in rows])),
            "qpos_fingers_dist_median": float(
                np.median([r["qpos_fingers_dist"] for r in rows])
            ),
        }
        print("\n" + "=" * 68)
        print(f"n={n} (tp recovered but ex did not; modes={sorted(modes)})")
        print(f"(a) null-space:   tip converged but arm(7dof)>30cm off : {arm_off}/{n} ({arm_off/n:.0%})")
        print(f"(b) grasp mech:   tip converged but grasp mismatched   : {grasp_bad}/{n} ({grasp_bad/n:.0%})")
        print(f"(c) budget:       return didn't finish in horizon      : {budget}/{n} ({budget/n:.0%})")
        print(f"    tip converged at all                              : {tip_conv}/{n} ({tip_conv/n:.0%})")
        print(f"    return disturbed the cube (>2cm)                  : {bumped}/{n} ({bumped/n:.0%})")
        print(f"    median arm7 dist {summary['qpos_arm7_dist_median']:.3f} | median finger dist {summary['qpos_fingers_dist_median']:.3f}")
    else:
        summary = {"n": 0}
        print("no qualifying failures")

    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\n[diag] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

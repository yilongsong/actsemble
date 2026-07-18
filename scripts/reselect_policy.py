#!/usr/bin/env python
"""Re-SELECT a policy from EXISTING interval snapshots under the CURRENT sampler
— no retraining. Use after an inference-relevant change invalidates a prior
selection (e.g. the DDIM timestep spacing switched linspace -> leading): the
trained snapshots are still valid, but their screening/confirmation are not.

Symlinks the snapshots into a fresh output dir (leaving the source run intact),
then re-screens (50) -> re-confirms (200) -> 500-ep final under current code, and
records the selected step/epoch + the exact sampler used.

    python scripts/reselect_policy.py \
        --source-dir outputs/active_min/overnight/diffusion \
        --out-dir    outputs/active_min/rerun/diffusion
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.metrics import wilson_interval  # noqa: E402
from actsemble.evaluation.panels import make_panel  # noqa: E402
from actsemble.protocol.confirmation import confirm_and_select  # noqa: E402
from actsemble.protocol.screening import screen_all_snapshots, screening_history_path  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402

STANDALONE = {"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"},
              "execution": {}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True, help="existing run dir with checkpoints/")
    ap.add_argument("--out-dir", required=True, help="fresh output dir for the re-selection")
    ap.add_argument("--dataset", default=str(REPO / "data/active_min/subset_0400.h5"))
    ap.add_argument("--final-n", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="allow a non-empty --out-dir")
    args = ap.parse_args()

    source, out = Path(args.source_dir), Path(args.out_dir)
    snaps = sorted((source / "checkpoints").glob("step_*.pt"))
    if not snaps:
        raise SystemExit(f"No interval snapshots under {source}/checkpoints")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"--out-dir {out} is not empty (pass --force to reuse)")

    # Symlink the (large) snapshots + copy the small train_config.json for provenance.
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    for s in snaps:
        link = out / "checkpoints" / s.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(s.resolve(), link)
    if (source / "train_config.json").exists():
        shutil.copy(source / "train_config.json", out / "train_config.json")

    meta = DatasetReader(args.dataset).metadata
    env = make_env(task_id=meta.task_id, control_mode=meta.controller,
                   sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)
    try:
        print(f"[reselect] screening {len(snaps)} snapshot(s) under the current sampler ...")
        screen_all_snapshots(out, make_panel("screening"), env=env, device=args.device,
                             max_steps=args.max_steps)
        report = confirm_and_select(out, make_panel("confirmation"), env=env,
                                    device=args.device, max_steps=args.max_steps, force=True)
        fp = make_panel("final_test")
        final = evaluate_system(
            system_cfg=STANDALONE,
            eval_cfg={"name": "final_test", "regime": "nominal", "max_steps": args.max_steps,
                      "panel": {"name": fp.name, "env_seed": fp.env_seed,
                                "num_episodes": args.final_n},
                      "perturbations": []},
            policy_checkpoint=report["selected_policy_path"], num_episodes=args.final_n,
            output_path=out / "final_test.json", device=args.device, env=env, force=True,
        )
    finally:
        env.close()

    tc = load_json(out / "train_config.json") if (out / "train_config.json").exists() else {}
    spe = tc.get("steps_per_epoch")
    sel = int(report["selected_step"])
    ci = wilson_interval(final["success_count"], args.final_n)
    record = {
        "source_dir": str(source),
        "selected_step": sel,
        "steps_per_epoch": spe,
        "selected_epoch": round(sel / spe, 2) if spe else None,
        "final_success_rate": final["success_rate"],
        "final_success_count": final["success_count"],
        "final_wilson_ci": list(ci),
        "final_n": args.final_n,
        "policy_ms_per_query": 1000.0 * final["latency"]["mean_policy_s"],
        "sampler": final.get("sampler"),
        "selected_policy_path": report["selected_policy_path"],
        "screening_curve": [
            {"step": h["step"], "epoch": round(h["step"] / spe, 2) if spe else None,
             "success_rate": h["success_rate"]}
            for h in load_json(screening_history_path(out))
        ],
    }
    save_json(record, out / "reselection.json")
    print(f"[reselect] SELECTED step {sel}"
          + (f" (epoch {record['selected_epoch']})" if spe else "")
          + f" | final {final['success_rate']:.1%} CI [{ci[0]:.1%},{ci[1]:.1%}]"
          f" | {record['policy_ms_per_query']:.1f} ms/query")
    print(f"[reselect] sampler: {final.get('sampler')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

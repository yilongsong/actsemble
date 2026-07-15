#!/usr/bin/env python
"""Dataset-size (n_demos) selection driver (protocol §7).

For every candidate demonstration count: build the deterministic nested
subset, run the COMPLETE per-seed policy pipeline (fixed-budget training,
screening, confirmation) for every policy seed, evaluate each SELECTED
policy on the dataset-size-development panel, and average across seeds.
Selects the largest n_demos whose mean selected-policy success lies in the
predeclared target range (default 25%-50%). The final-test panel is never
touched here.

    python scripts/select_dataset_size.py --spec configs/protocol/default.yaml \
        --data-config configs/data/pilot.yaml \
        --workdir outputs/experiments/dataset_size_v1

Warning: this runs candidate_sizes x policy_seeds full training pipelines.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.config import load_config
from actsemble.evaluation.panels import load_panels
from actsemble.protocol.experiment import resolved_policy_config
from actsemble.utils.serialization import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="protocol spec YAML")
    parser.add_argument("--data-config", required=True, help="data config for subset export")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    spec = load_config(args.spec)
    sizes = spec["dataset_size_selection"]["candidate_sizes"]
    lo, hi = spec["dataset_size_selection"]["target_success_range"]
    subset_seed = int(spec["dataset_size_selection"].get("subset_seed", 0))
    seeds = [int(s) for s in spec["policy_seeds"]]
    panels = load_panels(spec.get("panels"))
    dev_panel = panels["dataset_size_development"]
    workdir = Path(args.workdir)
    repo = Path(__file__).resolve().parents[1]

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for size in sizes:
        size_dir = workdir / f"ndemos_{size}"
        dataset = size_dir / f"subset_{size}.h5"
        if not dataset.exists():
            print(f"[ndemos] preparing nested subset size={size} seed={subset_seed}")
            subprocess.run(
                [sys.executable, str(repo / "scripts" / "prepare_dataset.py"),
                 "--config", args.data_config, "--output", str(dataset),
                 "--subset-size", str(size)],
                check=True,
            )
        per_seed_success = []
        for policy_seed in seeds:
            run_dir = size_dir / f"seed_{policy_seed}" / "policy"
            selected = run_dir / "selected_policy.pt"
            if not selected.exists():
                _train_and_select(spec, dataset, run_dir, policy_seed, device)
            rate = _evaluate_selected(selected, dev_panel, dataset, device,
                                      int(spec.get("episode_max_steps", 100)))
            print(f"[ndemos] size={size} seed={policy_seed}: dev success {rate:.1%}")
            per_seed_success.append(rate)
        mean = sum(per_seed_success) / len(per_seed_success)
        results.append({"n_demos": size, "per_seed_success": per_seed_success,
                        "mean_success": mean, "in_target": lo <= mean <= hi})
        print(f"[ndemos] size={size}: mean selected-policy success {mean:.1%} "
              f"(target [{lo:.0%}, {hi:.0%}])")

    in_range = [r for r in results if r["in_target"]]
    selected = max(in_range, key=lambda r: r["n_demos"]) if in_range else None
    report = {
        "candidate_results": results,
        "target_range": [lo, hi],
        "selected_n_demos": selected["n_demos"] if selected else None,
        "note": ("If no candidate lies in the target range, add intermediate "
                 "nested sizes and rerun (§7). Never use the final-test panel here."),
    }
    save_json(report, workdir / "dataset_size_selection.json")
    if selected:
        print(f"[ndemos] SELECTED n_demos = {selected['n_demos']} "
              f"(mean success {selected['mean_success']:.1%})")
        return 0
    print("[ndemos] NO candidate size in target range — add intermediate sizes (§7)")
    return 1


def _train_and_select(spec, dataset, run_dir, policy_seed, device):
    from actsemble.protocol.confirmation import confirm_and_select
    from actsemble.protocol.screening import make_screening_callback
    from actsemble.training.train_diffusion_policy import train_diffusion_policy

    from actsemble.data.reader import DatasetReader
    from actsemble.sim.env_factory import make_env

    panels = load_panels(spec.get("panels"))
    policy_cfg = resolved_policy_config(spec, policy_seed)
    meta = DatasetReader(dataset).metadata
    env = make_env(task_id=meta.task_id, control_mode=meta.controller,
                   sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)
    try:
        max_steps = int(spec.get("episode_max_steps", 100))
        train_diffusion_policy(
            policy_cfg=policy_cfg, dataset_path=dataset, output_dir=run_dir,
            device=device,
            on_checkpoint=make_screening_callback(
                run_dir, panels["screening"], env=env, device=device, max_steps=max_steps
            ),
        )
        rule = spec.get("confirmation_rule", {}) or {}
        confirm_and_select(
            run_dir, panels["confirmation"], env=env, device=device, max_steps=max_steps,
            top_k=int(rule.get("top_k", 5)),
            within_of_best=float(rule.get("within_of_best", 0.10)),
        )
    finally:
        env.close()


def _evaluate_selected(selected_policy, panel, dataset, device, max_steps) -> float:
    from actsemble.data.reader import DatasetReader
    from actsemble.protocol.screening import evaluate_checkpoint_on_panel
    from actsemble.sim.env_factory import make_env

    meta = DatasetReader(dataset).metadata
    env = make_env(task_id=meta.task_id, control_mode=meta.controller,
                   sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)
    try:
        record = evaluate_checkpoint_on_panel(
            selected_policy, panel, env=env, device=device, max_steps=max_steps
        )
    finally:
        env.close()
    return record["success_rate"]


if __name__ == "__main__":
    sys.exit(main())

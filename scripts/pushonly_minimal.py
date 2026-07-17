#!/usr/bin/env python
"""Minimalist push-only replication — rough check that the full-data findings
hold once the ~63% holding tail is removed.

NOT the frozen protocol: ONE policy seed, ONE verifier seed, small panels, no
8-candidate confirmation (checkpoint picked by the 50-ep screening panel). The
goal is a rough image, not a decisive number. All artifacts live under
outputs/pushonly_min / data/pushonly_min; Phase 0A is never touched.

Stages (each idempotent; `all` runs them in order with a sanity gate):
  subset          nested n=400 subset of data/push_t_pushonly.h5 (same 400
                  underlying episodes as Phase 0A n=400, just hold-trimmed)
  train-policy    train one policy (seed 0), screen, pick best checkpoint
  sanity          200-ep dev eval of the picked policy -> base success (GATE)
  train-verifier  train one verifier (seed 0), offline-select
  compare         candidate_zero vs medoid vs verifier on a small panel; paired
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.config import load_config, merge_config  # noqa: E402
from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.data.schema import DatasetMetadata  # noqa: E402
from actsemble.data.validation import validate_dataset  # noqa: E402
from actsemble.data.writer import write_dataset, write_private_provenance  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.utils.serialization import load_json, save_json  # noqa: E402

SRC = REPO / "data" / "push_t_pushonly.h5"
N, SEED = 400, 0
OUT = REPO / "outputs" / "pushonly_min"
DATA = REPO / "data" / "pushonly_min"
SUBSET = DATA / f"subset_{N:04d}.h5"
POLICY_DIR = OUT / f"policy_seed_{SEED}"
VERIF_DIR = OUT / f"verifier_seed_{SEED}"
SEL_POLICY = POLICY_DIR / "selected_policy.pt"
SEL_VERIF = VERIF_DIR / "selected_verifier.pt"
SANITY_FLOOR = 0.20  # dev success below this => halt (push-only pipeline broken)
COMPARE_PANEL = {"name": "pushonly_min_compare", "env_seed": 20000, "num_episodes": 300}
MAX_STEPS = 30000  # same recipe as Phase 0A; screening picks the best checkpoint

SPEC = load_config(REPO / "outputs" / "phase0a_v1" / "experiment_spec.yaml")


def _env():
    meta = DatasetReader(SUBSET).metadata
    return make_env(task_id=meta.task_id, control_mode=meta.controller,
                    sim_backend=meta.simulation_backend, obs_mode="state", render_mode=None)


# --------------------------------------------------------------- subset ----
def cmd_subset(args):
    if SUBSET.exists() and not args.force:
        print(f"[subset] {SUBSET} exists; skipping"); return 0
    reader = DatasetReader(SRC)
    by_id = {e.episode_id: e for e in reader.episodes}
    total = len(by_id)
    # same nested permutation as Phase 0A (subset_seed=0): first-N indices.
    perm = np.random.default_rng(int(SPEC["dataset"]["subset_seed"])).permutation(total)
    chosen_idx = sorted(int(i) for i in perm[:N])
    ids = [f"ep_{i:05d}" for i in chosen_idx]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} ids missing from {SRC} (e.g. {missing[:3]})")
    episodes = [by_id[i] for i in ids]
    meta = reader.metadata
    new_meta = DatasetMetadata(**{k: getattr(meta, k) for k in [
        "simulator", "simulator_version", "task_id", "robot", "observation_mode",
        "state_dimension", "state_layout", "controller", "action_dimension",
        "action_definition", "control_frequency", "simulation_backend",
        "source_dataset", "generation_or_replay_seed"]})
    new_meta.extra = dict(meta.extra)
    new_meta.extra.update({"subset_size": N, "subset_of": str(SRC),
                           "subset_of_hash": meta.dataset_hash,
                           "subset_selected_ids": ids})
    h = write_dataset(SUBSET, episodes, new_meta)
    write_private_provenance(SUBSET, {
        "success_only": True, "conversion_method": "subset_of_pushonly",
        "source": {"subset_of": str(SRC), "subset_of_hash": meta.dataset_hash},
        "exported_episodes": [{"episode_id": i, "source_success": True,
                               "projected_success_at_end": True} for i in ids],
        "rejected_episodes": [], "rejected_count": 0,
        "conversion_failures": [], "conversion_failure_count": 0})
    s = validate_dataset(DatasetReader(SUBSET))
    print(f"[subset] wrote {SUBSET} : {s['num_episodes']} eps, {s['num_transitions']} transitions, "
          f"len[min={s['episode_length_min']}, mean={s['episode_length_mean']:.1f}], hash {h[:12]}")
    return 0


# --------------------------------------------------------- train policy ----
def cmd_train_policy(args):
    from actsemble.protocol.screening import make_screening_callback, screening_history_path
    from actsemble.training.train_diffusion_policy import train_diffusion_policy
    from actsemble.evaluation.panels import load_panels

    if SEL_POLICY.exists() and not args.force:
        print(f"[train-policy] {SEL_POLICY} exists; skipping"); return 0
    cfg = merge_config(load_config(REPO / SPEC["policy_config"]),
                       SPEC.get("policy_config_overrides", {}))
    cfg = merge_config(cfg, {"training": {"seed": SEED, "max_steps": MAX_STEPS,
                                          "split_seed": int(SPEC["dataset"]["split_seed"]),
                                          "val_fraction": float(SPEC["dataset"]["val_fraction"])}})
    panels = load_panels(SPEC["panels"])
    env = _env()
    try:
        t0 = time.time()
        summary = train_diffusion_policy(
            policy_cfg=cfg, dataset_path=SUBSET, output_dir=POLICY_DIR, device=args.device or "cuda",
            on_checkpoint=make_screening_callback(
                POLICY_DIR, panels["screening"], env=env, device=args.device or "cuda",
                max_steps=int(SPEC["episode_max_steps"])))
    finally:
        env.close()
    hist = load_json(screening_history_path(POLICY_DIR))
    best = max(hist, key=lambda r: (r["success_rate"], -r["step"]))  # max rate, tie->earlier
    shutil.copyfile(best["checkpoint_path"], SEL_POLICY)
    save_json({"selected_step": best["step"], "screening_rate": best["success_rate"],
               "screening_history": [(r["step"], r["success_rate"]) for r in hist],
               "final_train_loss": summary["final_train_loss"],
               "best_val_loss": summary["best_val_loss"], "train_wall_s": time.time() - t0},
              POLICY_DIR / "select_summary.json")
    print(f"[train-policy] best screening step {best['step']} = {best['success_rate']:.1%} "
          f"-> {SEL_POLICY.name}")
    return 0


# --------------------------------------------------------------- sanity ----
def cmd_sanity(args):
    from actsemble.protocol.screening import evaluate_checkpoint_on_panel
    from actsemble.evaluation.panels import load_panels

    panels = load_panels(SPEC["panels"])
    env = _env()
    try:
        dev = evaluate_checkpoint_on_panel(SEL_POLICY, panels["dataset_size_development"],
                                           env=env, device=args.device or "cuda",
                                           max_steps=int(SPEC["episode_max_steps"]))
    finally:
        env.close()
    save_json(dev, POLICY_DIR / "dev_eval.json")
    rate = dev["success_rate"]
    print(f"[sanity] push-only n={N} policy dev success = {rate:.1%} "
          f"({dev['success_count']}/{dev['num_episodes']}) CI "
          f"[{dev['wilson_ci'][0]:.1%},{dev['wilson_ci'][1]:.1%}]")
    if rate < SANITY_FLOOR:
        print(f"[sanity] FAIL: {rate:.1%} < floor {SANITY_FLOOR:.0%} — halting before compare")
        return 2
    print(f"[sanity] OK (>= {SANITY_FLOOR:.0%})")
    return 0


# ------------------------------------------------------- train verifier ----
def cmd_train_verifier(args):
    from actsemble.protocol.verifier_selection import select_verifier
    from actsemble.training.train_component import train_component

    if SEL_VERIF.exists() and not args.force:
        print(f"[train-verifier] {SEL_VERIF} exists; skipping"); return 0
    cfg = merge_config(load_config(REPO / SPEC["verifier_config"]),
                       SPEC.get("verifier_config_overrides", {}))
    cfg = merge_config(cfg, {"training": {"seed": SEED,
                                          "split_seed": int(SPEC["dataset"]["split_seed"]),
                                          "val_fraction": float(SPEC["dataset"]["val_fraction"])}})
    summary = train_component(component_cfg=cfg, dataset_path=SUBSET,
                              output_dir=VERIF_DIR, device=args.device or "cuda")
    sel = select_verifier(VERIF_DIR, primary=SPEC["verifier_selection"]["primary"],
                          secondary=SPEC["verifier_selection"]["secondary"], force=args.force)
    save_json({"selected_step": sel["selected_step"], "winner_metrics": sel["winner_metrics"],
               "offline_eval": summary["offline_eval"]}, VERIF_DIR / "select_summary.json")
    print(f"[train-verifier] step {sel['selected_step']} ranking-acc "
          f"{sel['winner_metrics']['pairwise_ranking_accuracy']:.4f}")
    return 0


# -------------------------------------------------------------- compare ----
SYSTEMS = {
    "candidate_zero": {"type": "candidate_zero"},
    "full_chunk_medoid": {"type": "full_chunk_medoid"},
    "verifier_argmax": {"type": "highest_component_score"},
}


def _mcnemar(a, b):  # a,b per-episode 0/1 arrays; returns z (a>b positive)
    a, b = np.asarray(a), np.asarray(b)
    b10 = int(((a == 1) & (b == 0)).sum()); b01 = int(((a == 0) & (b == 1)).sum())
    disc = b10 + b01
    z = (b10 - b01) / np.sqrt(disc) if disc else 0.0
    return {"a_wins": b10, "b_wins": b01, "z": float(z), "sig_0.05": bool(abs(z) > 1.96)}


def cmd_compare(args):
    from actsemble.evaluation.evaluator import evaluate_system

    cmp_dir = OUT / "compare"; cmp_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = {"regime": "nominal", "panel": COMPARE_PANEL, "max_steps": 100,
                "perturbations": [], "video": {"max_success": 0, "max_failure": 0}}
    succ = {}
    for name, sel in SYSTEMS.items():
        dest = cmp_dir / f"{name}.json"
        if not dest.exists() or args.force:
            comps = [str(SEL_VERIF)] if name == "verifier_argmax" else []
            evaluate_system(system_cfg={"policy": {"num_candidates": 1}, "selection": sel},
                            eval_cfg=eval_cfg, policy_checkpoint=str(SEL_POLICY),
                            component_checkpoints=comps, output_path=dest,
                            device=args.device or "cuda", env=None,
                            num_candidates_override=16, force=args.force)
        r = load_json(dest)
        succ[name] = np.array(r["successes"], dtype=int)
        print(f"[compare] {name:20s} {r['success_count']}/{r['num_episodes']} = {r['success_rate']:.1%}")

    rows = {name: {"success_rate": float(s.mean()), "n": int(s.size)} for name, s in succ.items()}
    contrasts = {
        "verifier_vs_candidate_zero": _mcnemar(succ["verifier_argmax"], succ["candidate_zero"]),
        "medoid_vs_candidate_zero": _mcnemar(succ["full_chunk_medoid"], succ["candidate_zero"]),
        "verifier_vs_medoid": _mcnemar(succ["verifier_argmax"], succ["full_chunk_medoid"]),
    }
    for k, c in contrasts.items():
        a, b = k.split("_vs_")
        c["delta"] = rows[{"verifier": "verifier_argmax", "medoid": "full_chunk_medoid",
                           "candidate_zero": "candidate_zero"}.get(a, a)]["success_rate"] - \
            rows[{"candidate_zero": "candidate_zero", "medoid": "full_chunk_medoid"}.get(b, b)]["success_rate"]
    report = {"note": "MINIMALIST: 1 seed, 300-ep panel, screening-picked checkpoint. Rough image only.",
              "panel": COMPARE_PANEL, "num_candidates": 16, "systems": rows, "contrasts": contrasts,
              "dataset": str(SUBSET)}
    save_json(report, cmp_dir / "results.json")
    print("\n=== push-only minimalist comparison (rough) ===")
    for name, r in rows.items():
        print(f"  {name:20s} {r['success_rate']:.1%}")
    for k, c in contrasts.items():
        print(f"  {k:28s} Δ={c['delta']:+.1%}  McNemar z={c['z']:+.2f} "
              f"({'sig' if c['sig_0.05'] else 'ns'}; +{c['a_wins']}/-{c['b_wins']})")
    return 0


def cmd_all(args):
    for fn in (cmd_subset, cmd_train_policy, cmd_sanity):
        rc = fn(args)
        if rc:  # sanity gate (rc=2) or error halts the chain
            print(f"[all] halted at {fn.__name__} (rc={rc})"); return rc
    for fn in (cmd_train_verifier, cmd_compare):
        rc = fn(args)
        if rc:
            print(f"[all] halted at {fn.__name__} (rc={rc})"); return rc
    print("[all] done")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["subset", "train-policy", "sanity",
                                     "train-verifier", "compare", "all"])
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    return {"subset": cmd_subset, "train-policy": cmd_train_policy, "sanity": cmd_sanity,
            "train-verifier": cmd_train_verifier, "compare": cmd_compare, "all": cmd_all}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Selector-baselines v1: integration checks + development-panel smoke for the
non-learned consensus selectors. NEVER touches the Phase 0A frozen spec,
checkpoints, verifier, or final-test panel — it only READS the frozen seed-4
policy/verifier and evaluates on a fresh development panel disjoint from every
Phase 0A panel. Development results are diagnostic-only (non-claim).

Subcommands: integration | dev-check
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility  # noqa: E402
from actsemble.evaluation.evaluator import evaluate_system  # noqa: E402
from actsemble.evaluation.panels import DEFAULT_PANELS, Panel, panel_episodes  # noqa: E402
from actsemble.evaluation.reports import verify_candidate_identity  # noqa: E402
from actsemble.policies.diffusion.policy import DiffusionPolicy  # noqa: E402
from actsemble.seed import derive_seed  # noqa: E402
from actsemble.sim.env_factory import make_env  # noqa: E402
from actsemble.systems.factory import build_system  # noqa: E402
from actsemble.utils.repo import current_git_commit  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402

OUT = REPO / "outputs" / "selector_baselines_v1"
K = 16
# Frozen Phase 0A seed-4 artifacts, read-only.
POLICY = REPO / "outputs/phase0a_v1/final_policy_runs/policy_seed_4/selected_policy.pt"
VERIFIER = REPO / "outputs/phase0a_v1/verifier_runs/verifier_seed_4/selected_verifier.pt"

# Development panel: root env_seed disjoint from all Phase 0A panels.
DEV_PANEL = Panel(name="selector_development", env_seed=9100, num_episodes=30)

# system_cfg for each selector (num_candidates overridden to K at eval time)
SYSTEMS = {
    "candidate_zero": ({"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"}}, False),
    "full_chunk_medoid": ({"policy": {"num_candidates": 1}, "selection": {"type": "full_chunk_medoid"}}, False),
    "early_weighted_medoid": ({"policy": {"num_candidates": 1}, "selection": {"type": "early_weighted_medoid", "early_weight_decay": 0.25}}, False),
    "coordinate_median_projection": ({"policy": {"num_candidates": 1}, "selection": {"type": "coordinate_median_projection"}}, False),
    "largest_cluster_medoid": ({"policy": {"num_candidates": 1}, "selection": {"type": "largest_cluster_medoid"}}, False),
    "verifier_argmax": ({"policy": {"num_candidates": 1}, "selection": {"type": "highest_component_score"}}, True),
}


def _param_digest(policy) -> str:
    h = hashlib.sha256()
    for name, p in sorted(policy.model.state_dict().items()):
        h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _assert_dev_panel_disjoint():
    for name, spec in DEFAULT_PANELS.items():
        assert spec["env_seed"] != DEV_PANEL.env_seed, f"dev panel collides with {name}"


def _load(device):
    policy = DiffusionPolicy.from_checkpoint(POLICY, device=device, use_ema=True)
    verifier = ActionChunkCompatibility.from_checkpoint(VERIFIER, device=device)
    return policy, verifier


# --------------------------------------------------------------------------- #
def cmd_integration(args):
    device = args.device or "cuda"
    out = OUT / "integration"; out.mkdir(parents=True, exist_ok=True)
    _assert_dev_panel_disjoint()
    policy, verifier = _load(device)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        print(f"[selector-int] {'PASS' if ok else 'FAIL'} | {name}" + (f" — {detail}" if detail else ""))

    # one shared K=16 candidate tensor from a real observation history
    env = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                   sim_backend=policy.meta.simulation_backend, obs_mode="state", render_mode=None)
    from actsemble.sim.observation_adapter import ObservationAdapter
    raw, _ = env.reset(seed=DEV_PANEL.env_seed)
    obs = ObservationAdapter(action_dim=policy.meta.action_dim).observe(raw)
    frame = np.asarray(obs.state, dtype=np.float32).reshape(-1)
    history = np.stack([frame, frame], axis=0)
    gen = policy.new_generator(derive_seed(777, "candidates", policy.checkpoint_hash, 0))
    shared = policy.sample_action_chunks(history, num_samples=K, generator=gen)
    shared_hash = hashlib.sha256(np.ascontiguousarray(
        shared.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:16]

    param_before = _param_digest(policy)
    valid = torch.isfinite(shared).all(dim=(1, 2))
    per_system = {}
    for name, (cfg, needs_v) in SYSTEMS.items():
        comps = [verifier] if needs_v else []
        system = build_system(cfg, policy, comps, candidate_root_seed=777)
        system._replan_index = 0
        rec = {}
        seen = shared.clone()
        seen_hash = hashlib.sha256(np.ascontiguousarray(
            seen.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()[:16]
        t0 = time.perf_counter()
        idx = system._select(seen, valid, history, rec)
        dt = time.perf_counter() - t0
        per_system[name] = {
            "selected_index": int(idx), "seen_tensor_hash": seen_hash,
            "n_components": len(system.components()), "select_latency_s": dt,
            "changed_from_candidate_zero": int(idx) != 0,
        }

    # 3. same candidate hash before selection
    hashes = {v["seen_tensor_hash"] for v in per_system.values()}
    check("identical_candidate_tensor_before_selection",
          hashes == {shared_hash}, f"{len(hashes)} distinct tensor hash(es)")
    # 4. differences only in selected index
    check("differences_only_in_selected_index",
          len({v["selected_index"] for v in per_system.values()}) >= 1
          and all(v["seen_tensor_hash"] == shared_hash for v in per_system.values()),
          "selected: " + ", ".join(f"{k}={v['selected_index']}" for k, v in per_system.items()))
    # 5. policy params byte-identical
    check("policy_params_unchanged", _param_digest(policy) == param_before)
    # 6. no learned component for consensus selectors
    check("no_component_for_consensus",
          all(per_system[n]["n_components"] == 0 for n in SYSTEMS if n != "verifier_argmax")
          and per_system["verifier_argmax"]["n_components"] == 1)
    # 7. reset clears diagnostics
    sys_r = build_system(SYSTEMS["full_chunk_medoid"][0], policy, [], candidate_root_seed=777)
    sys_r._replan_records.append({"dummy": 1})
    sys_r.reset(episode_seed=1)
    check("reset_clears_diagnostics",
          sys_r.diagnostics()["num_replans"] == 0 and len(sys_r._history) == 0)

    # 8. all selectors finish full PushT rollouts (short) + latency separation
    from actsemble.evaluation.evaluator import run_panel_episode
    ep = panel_episodes(DEV_PANEL)[0]
    latency = {}
    finished_ok = True
    for name, (cfg, needs_v) in SYSTEMS.items():
        comps = [verifier] if needs_v else []
        env2 = make_env(task_id=policy.meta.task_id, control_mode=policy.meta.controller,
                        sim_backend=policy.meta.simulation_backend, obs_mode="state", render_mode=None)
        system = build_system({**cfg, "policy": {**cfg["policy"], "num_candidates": K}}, policy, comps,
                              candidate_root_seed=ep.policy_sampling_seed)
        res, _ = run_panel_episode(env2, system, ep, max_steps=100, pert_specs=[])
        env2.close()
        finished_ok &= (res.exception is None)
        d = system.diagnostics(); rp = d["replans"]
        latency[name] = {
            "mean_policy_latency_s": float(np.mean([r["policy_latency_s"] for r in rp])),
            "mean_selector_latency_s": float(np.mean([r.get("selector_latency_s", r.get("component_latency_s", 0.0)) for r in rp])),
            "num_replans": len(rp),
        }
    check("all_selectors_finish_rollouts", finished_ok)
    check("selector_latency_measured_separately",
          all(latency[n]["mean_selector_latency_s"] >= 0 for n in latency),
          "; ".join(f"{n}: sel={latency[n]['mean_selector_latency_s']*1e3:.2f}ms "
                    f"pol={latency[n]['mean_policy_latency_s']*1e3:.1f}ms" for n in latency))
    env.close()

    passed = all(c["passed"] for c in checks)
    save_json({"passed": passed, "checks": checks, "shared_candidate_hash": shared_hash,
               "per_system": per_system, "latency": latency,
               "policy_checkpoint_hash": policy.checkpoint_hash,
               "verifier_checkpoint_hash": verifier.checkpoint_hash,
               "git_commit": current_git_commit()},
              out / "candidate_identity_report.json")
    print(f"[selector-int] {'ALL PASSED' if passed else 'FAILURES PRESENT'}")
    return 0 if passed else 1


# --------------------------------------------------------------------------- #
def cmd_dev_check(args):
    device = args.device or "cuda"
    out = OUT / "development_check"; out.mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    _assert_dev_panel_disjoint()
    policy, verifier = _load(device)
    eval_cfg = {"regime": "selector_dev", "panel": DEV_PANEL.to_dict(), "max_steps": 100,
                "perturbations": [], "video": {"max_success": 1, "max_failure": 1}}
    results = {}
    for name, (cfg, needs_v) in SYSTEMS.items():
        comps = [str(VERIFIER)] if needs_v else []
        results[name] = evaluate_system(
            system_cfg=cfg, eval_cfg=eval_cfg, policy_checkpoint=POLICY,
            component_checkpoints=comps, output_path=out / f"eval_{name}.json",
            video_dir=out / "videos", device=device, env=None,
            num_candidates_override=K, force=args.force)
        r = results[name]
        print(f"[selector-dev] {name}: {r['success_count']}/{r['num_episodes']} "
              f"= {r['success_rate']:.1%} (NON-CLAIM)")

    # candidate identity up to first selection divergence (tolerance-based)
    identity = verify_candidate_identity(list(results.values()))

    def sel_stats(r):
        idxs = [i for e in r["episodes"] for i in e["selected_indices"]]
        changed = [e["selection_change_rate"] for e in r["episodes"]]
        pw = [rp.get("mean_pairwise_distance") for e in r["episodes"] for rp in e["replans"]
              if rp.get("mean_pairwise_distance") is not None]
        sel_lat = [rp.get("selector_latency_s") for e in r["episodes"] for rp in e["replans"]
                   if rp.get("selector_latency_s") is not None]
        return {
            "success_rate_nonclaim": r["success_rate"],
            "selector_change_rate_vs_candidate_zero": float(np.mean(changed)) if changed else 0.0,
            "selected_index_distribution": dict(Counter(idxs)),
            "mean_pairwise_candidate_distance": float(np.mean(pw)) if pw else None,
            "mean_selector_latency_ms": float(np.mean(sel_lat) * 1e3) if sel_lat else None,
            "fallback_rate": r.get("fallback_rate", 0.0),
        }

    stats = {name: sel_stats(r) for name, r in results.items()}
    # episodes where any consensus selector disagrees with candidate zero
    cz = results["candidate_zero"]
    disagree = {}
    for name, r in results.items():
        if name == "candidate_zero":
            continue
        eps = [e["episode_index"] for e, e0 in zip(r["episodes"], cz["episodes"])
               if e["selected_indices"] and e["selected_indices"][0] != 0]
        disagree[name] = eps

    report = {
        "note": "DEVELOPMENT / DIAGNOSTIC ONLY — success rates are NON-CLAIM; "
                "do not select a selector on these numbers.",
        "panel": DEV_PANEL.to_dict(),
        "candidate_identity_until_divergence": identity or "OK (bitwise init + tolerance prefix)",
        "per_selector": stats,
        "episodes_with_selection_change": {k: len(v) for k, v in disagree.items()},
        "git_commit": current_git_commit(),
    }
    save_json(report, out / "results.json")
    _write_dev_markdown(out / "comparison.md", report, stats)
    print(f"[selector-dev] wrote {out/'results.json'}")
    return 0


def _write_dev_markdown(path, report, stats):
    lines = ["# Selector development-panel check (NON-CLAIM diagnostics)", "",
             f"Panel: `{report['panel']}` — disjoint from all Phase 0A panels. "
             "Success rates below are diagnostic only and must not be used to select a method.", "",
             "| selector | success% (nonclaim) | change-rate vs cand0 | mean pairwise dist | selector latency (ms) | fallback |",
             "|---|---|---|---|---|---|"]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['success_rate_nonclaim']:.1%} | "
                     f"{s['selector_change_rate_vs_candidate_zero']:.2f} | "
                     f"{s['mean_pairwise_candidate_distance'] if s['mean_pairwise_candidate_distance'] is None else round(s['mean_pairwise_candidate_distance'],3)} | "
                     f"{s['mean_selector_latency_ms'] if s['mean_selector_latency_ms'] is None else round(s['mean_selector_latency_ms'],3)} | "
                     f"{s['fallback_rate']:.2f} |")
    lines += ["", "## Selected-index distribution", ""]
    for name, s in stats.items():
        lines.append(f"- **{name}**: {s['selected_index_distribution']}")
    lines += ["", "## Candidate identity", "",
              f"{report['candidate_identity_until_divergence']}", ""]
    path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["integration", "dev-check"])
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    return {"integration": cmd_integration, "dev-check": cmd_dev_check}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())

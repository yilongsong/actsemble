#!/usr/bin/env python
"""Phase 0A Step 13-15 analysis: paired categorization, selection diagnostics,
support distance via deterministic replay, category videos, final report.

Subcommands:
    diagnostics      paired categories + selection/score/magnitude statistics
    replay-sample    deterministic replay of sampled final episodes capturing
                     full candidate tensors -> support-distance statistics
    videos           representative videos for the four paired categories
    report           comparison_report.md

Everything here observes frozen systems; nothing retrains or reselects.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.utils.serialization import load_json, save_json

OUT = REPO / "outputs" / "phase0a_v1"
ANA = OUT / "analysis"


def pair_results(seed: int) -> dict:
    base = OUT / "pairs" / f"pair_{seed}" / "final_test" / "nominal"
    return {n: load_json(base / f"eval_{n}.json")
            for n in ("standalone", "control", "actsemble")}


def final_seeds() -> list[int]:
    from actsemble.config import load_config

    return [int(x) for x in load_config(OUT / "experiment_spec.yaml")["final_policy_seeds"]]


# ------------------------------------------------------------ diagnostics ----
def cmd_diagnostics(args):
    ANA.mkdir(parents=True, exist_ok=True)
    per_pair = {}
    all_margins_succ, all_margins_fail = [], []
    for seed in final_seeds():
        r = pair_results(seed)
        sa, ac = r["standalone"], r["actsemble"]
        categories = {"actsemble_only_success": [], "standalone_only_success": [],
                      "both_succeed": [], "both_fail": []}
        for ea, ec in zip(sa["episodes"], ac["episodes"]):
            key = {
                (False, True): "actsemble_only_success",
                (True, False): "standalone_only_success",
                (True, True): "both_succeed",
                (False, False): "both_fail",
            }[(ea["success_once"], ec["success_once"])]
            categories[key].append(ea["env_seed"])

        index_hist = Counter()
        margins, margin_by_outcome = [], {"success": [], "failure": []}
        sel_mag, zero_mag, sel_smooth, zero_smooth = [], [], [], []
        changed, total = 0, 0
        for ec in ac["episodes"]:
            for rp in ec["replans"]:
                idx = rp["selected_index"]
                index_hist[idx] += 1
                total += 1
                changed += idx != 0
                scores = rp.get("component_scores")
                if scores:
                    srt = sorted(scores, reverse=True)
                    margins.append(srt[0] - srt[1])
                    margin_by_outcome["success" if ec["success_once"] else "failure"].append(
                        srt[0] - srt[1]
                    )
                ma, sm = rp["candidate_mean_abs"], rp["candidate_smoothness"]
                sel_mag.append(ma[idx]); zero_mag.append(ma[0])
                sel_smooth.append(sm[idx]); zero_smooth.append(sm[0])
        all_margins_succ += margin_by_outcome["success"]
        all_margins_fail += margin_by_outcome["failure"]

        def stats(x):
            x = np.asarray(x, dtype=np.float64)
            return {"mean": float(x.mean()), "median": float(np.median(x)),
                    "p10": float(np.quantile(x, 0.1)), "p90": float(np.quantile(x, 0.9))}

        per_pair[f"pair_{seed}"] = {
            "paired_categories": {k: len(v) for k, v in categories.items()},
            "category_env_seeds_sample": {k: v[:8] for k, v in categories.items()},
            "candidate_change_rate": changed / max(1, total),
            "selected_index_histogram": dict(sorted(index_hist.items())),
            "verifier_score_margin_top1_top2": stats(margins) if margins else None,
            "margin_success_episodes": stats(margin_by_outcome["success"])
            if margin_by_outcome["success"] else None,
            "margin_failure_episodes": stats(margin_by_outcome["failure"])
            if margin_by_outcome["failure"] else None,
            "selected_chunk_mean_abs": stats(sel_mag),
            "candidate_zero_mean_abs": stats(zero_mag),
            "selected_chunk_smoothness_mean_abs_delta": stats(sel_smooth),
            "candidate_zero_smoothness_mean_abs_delta": stats(zero_smooth),
            "verifier_prefers_lower_magnitude": float(np.mean(sel_mag))
            < float(np.mean(zero_mag)),
            "verifier_prefers_smoother": float(np.mean(sel_smooth))
            < float(np.mean(zero_smooth)),
            "fallback_rate": ac["fallback_rate"],
        }
    report = {
        "per_pair": per_pair,
        "pooled_margin_success": float(np.mean(all_margins_succ)) if all_margins_succ else None,
        "pooled_margin_failure": float(np.mean(all_margins_fail)) if all_margins_fail else None,
        "notes": [
            "candidate diversity + support distance come from replay-sample",
            "all statistics observe the frozen systems; nothing was retrained",
        ],
    }
    save_json(report, ANA / "diagnostic_statistics.json")
    for name, p in per_pair.items():
        c = p["paired_categories"]
        print(f"[diag] {name}: A-only {c['actsemble_only_success']}, "
              f"S-only {c['standalone_only_success']}, both+ {c['both_succeed']}, "
              f"both- {c['both_fail']}, change {p['candidate_change_rate']:.1%}")


# ---------------------------------------------------------- replay sample ----
class _CapturePolicy:
    """Wraps the frozen policy; records candidate tensors per replan."""

    def __init__(self, inner):
        self.inner = inner
        self.tensors = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def sample_action_chunks(self, observation_history, *, num_samples, generator):
        out = self.inner.sample_action_chunks(
            observation_history, num_samples=num_samples, generator=generator
        )
        self.tensors.append(out.detach().cpu().numpy().copy())
        return out


def cmd_replay_sample(args):
    """Replays a few final-panel episodes per pair (deterministic) to get full
    candidate tensors; computes candidate diversity + demo-support distance."""
    import torch

    from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility
    from actsemble.data.reader import DatasetReader
    from actsemble.data.windows import extract_window
    from actsemble.evaluation.evaluator import run_panel_episode
    from actsemble.evaluation.panels import PanelEpisode
    from actsemble.policies.diffusion.policy import DiffusionPolicy
    from actsemble.protocol.freeze import load_freeze
    from actsemble.sim.env_factory import make_env
    from actsemble.systems.candidate_reranking import CandidateRerankingActsemble

    n_eps = int(args.episodes)
    rows = {}
    for seed in final_seeds():
        freeze = load_freeze(OUT / "pairs" / f"pair_{seed}")
        r = pair_results(seed)
        ac = r["actsemble"]
        device = args.device or "cuda"
        policy = _CapturePolicy(DiffusionPolicy.from_checkpoint(
            freeze["policy"]["path"], device=device, use_ema=True))
        verifier = ActionChunkCompatibility.from_checkpoint(
            freeze["verifier"]["path"], device=device)
        system = CandidateRerankingActsemble(
            policy, verifier, num_candidates=freeze["system"]["num_candidates"])
        # demo action-chunk bank for support distance (subsampled)
        reader = DatasetReader(freeze["dataset"]["path"])
        bank = []
        rng = np.random.default_rng(0)
        for ep in reader.episodes:
            for t in rng.choice(len(ep), size=min(20, len(ep)), replace=False):
                w = extract_window(ep, int(t), obs_horizon=2, prediction_horizon=16)
                bank.append(w.action_chunk)
        bank_t = torch.from_numpy(np.stack(bank)).float().flatten(1)  # [B, H*A]

        env = make_env(
            task_id=freeze["environment"]["task_id"],
            control_mode=freeze["environment"]["controller"],
            sim_backend=freeze["environment"]["simulation_backend"],
            obs_mode="state", render_mode=None,
        )
        diversity, support_sel, support_zero, replay_match = [], [], [], []
        try:
            for erow in ac["episodes"][:n_eps]:
                ep = PanelEpisode(
                    episode_index=erow["episode_index"], env_seed=erow["env_seed"],
                    policy_sampling_seed=erow["policy_sampling_seed"],
                    perturbation_seed=erow["perturbation_seed"],
                )
                policy.tensors = []
                result, _ = run_panel_episode(env, system, ep, max_steps=100, pert_specs=[])
                replay_match.append(result.success_once == erow["success_once"])
                sels = system.diagnostics()["selected_indices"]
                for tensor, sel in zip(policy.tensors, sels):
                    flat = torch.from_numpy(tensor).float().flatten(1)  # [K, H*A]
                    # diversity: mean pairwise L2 between candidates
                    d = torch.cdist(flat, flat)
                    k = flat.shape[0]
                    diversity.append(float(d.sum() / (k * (k - 1))))
                    # support distance: min mean-|.| distance to demo bank
                    def support(v):
                        return float((bank_t - v).abs().mean(dim=1).min())
                    support_sel.append(support(flat[sel]))
                    support_zero.append(support(flat[0]))
        finally:
            env.close()
        rows[f"pair_{seed}"] = {
            "episodes_replayed": n_eps,
            "replay_outcome_matches_recorded": f"{sum(replay_match)}/{len(replay_match)}",
            "candidate_diversity_mean_pairwise_l2": float(np.mean(diversity)),
            "support_distance_selected_mean": float(np.mean(support_sel)),
            "support_distance_candidate_zero_mean": float(np.mean(support_zero)),
            "selected_closer_to_demo_support": float(np.mean(support_sel))
            < float(np.mean(support_zero)),
        }
        print(f"[replay] pair_{seed}: outcomes match {rows[f'pair_{seed}']['replay_outcome_matches_recorded']}, "
              f"diversity {rows[f'pair_{seed}']['candidate_diversity_mean_pairwise_l2']:.3f}, "
              f"support sel {np.mean(support_sel):.4f} vs zero {np.mean(support_zero):.4f}")
    save_json(rows, ANA / "replay_sample_statistics.json")


# ---------------------------------------------------------------- videos ----
def cmd_videos(args):
    """Representative videos for the four paired categories (deterministic
    replay of standalone + actsemble on chosen env seeds)."""
    from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility
    from actsemble.evaluation.evaluator import run_panel_episode
    from actsemble.evaluation.panels import PanelEpisode
    from actsemble.evaluation.video import save_video
    from actsemble.policies.diffusion.policy import DiffusionPolicy
    from actsemble.protocol.freeze import load_freeze
    from actsemble.sim.env_factory import make_env
    from actsemble.systems.candidate_reranking import CandidateRerankingActsemble
    from actsemble.systems.standalone import StandaloneDiffusionSystem

    diag = load_json(ANA / "diagnostic_statistics.json")
    per_category = int(args.per_category)
    seed = int(args.seed if args.seed is not None else final_seeds()[0])
    freeze = load_freeze(OUT / "pairs" / f"pair_{seed}")
    device = args.device or "cuda"
    policy = DiffusionPolicy.from_checkpoint(freeze["policy"]["path"], device=device)
    verifier = ActionChunkCompatibility.from_checkpoint(freeze["verifier"]["path"], device=device)
    k = freeze["system"]["num_candidates"]
    systems = {
        "standalone": StandaloneDiffusionSystem(policy, num_candidates=k),
        "actsemble": CandidateRerankingActsemble(policy, verifier, num_candidates=k),
    }
    ac_rows = {e["env_seed"]: e for e in pair_results(seed)["actsemble"]["episodes"]}
    env = make_env(
        task_id=freeze["environment"]["task_id"],
        control_mode=freeze["environment"]["controller"],
        sim_backend=freeze["environment"]["simulation_backend"],
        obs_mode="state", render_mode="rgb_array",
    )
    try:
        cats = diag["per_pair"][f"pair_{seed}"]["category_env_seeds_sample"]
        for cat, env_seeds in cats.items():
            for env_seed in env_seeds[:per_category]:
                row = ac_rows[env_seed]
                ep = PanelEpisode(
                    episode_index=row["episode_index"], env_seed=env_seed,
                    policy_sampling_seed=row["policy_sampling_seed"],
                    perturbation_seed=row["perturbation_seed"],
                )
                for name, system in systems.items():
                    _, frames = run_panel_episode(
                        env, system, ep, max_steps=100, pert_specs=[], capture_video=True
                    )
                    save_video(frames, ANA / "videos" / f"pair_{seed}" / cat
                               / f"{name}_seed{env_seed}.mp4")
        print(f"[videos] saved under {ANA / 'videos' / f'pair_{seed}'}")
    finally:
        env.close()


# ---------------------------------------------------------------- report ----
def cmd_report(args):
    paired = load_json(ANA / "paired_results.json")
    diag = load_json(ANA / "diagnostic_statistics.json")
    replay = (load_json(ANA / "replay_sample_statistics.json")
              if (ANA / "replay_sample_statistics.json").exists() else {})
    cand = load_json(OUT / "integration" / "candidate_identity_report.json")
    act = load_json(OUT / "integration" / "action_identity_report.json")
    sel = load_json(OUT / "primary_dataset_selection.json")

    ap = paired["across_pairs"]
    ds, dc = ap["delta_vs_standalone"], ap["delta_vs_control"]
    lines = [
        "# Phase 0A comparison report",
        "",
        f"Task PushT-v1, nominal only. n_demos_primary = {sel['n_demos_primary']} "
        f"(largest N with mean dev success in [25%, 50%]).",
        f"{paired['num_pairs']} policy/verifier seed pairs × "
        f"{paired['episodes_per_pair']} paired final episodes; K=16 shared candidates.",
        "",
        "## Identity preconditions",
        f"- candidate identity (all pairs, integration + final): "
        f"**{cand['all_pairs_pass']}**",
        f"- standalone vs control executed-action identity: **{act['all_pairs_pass']}**",
        "",
        "## Success rates (final panel, success_once)",
        "",
        "| pair | standalone | control | actsemble | Δ vs standalone | Δ vs control | "
        "win/loss | change rate |",
        "|------|-----------|---------|-----------|------------------|--------------|"
        "----------|-------------|",
    ]
    for p in paired["per_pair"]:
        lines.append(
            f"| {p['pair']} | {p['standalone_success']:.1%} | {p['control_success']:.1%} | "
            f"{p['actsemble_success']:.1%} | {p['delta_vs_standalone']:+.1%} | "
            f"{p['delta_vs_control']:+.1%} | {p['paired_win']}/{p['paired_loss']} | "
            f"{p['candidate_change_rate']:.1%} |"
        )
    lines += [
        "",
        "## Primary result (seed pairs as replication unit)",
        f"- Actsemble − standalone: mean {ds['mean']:+.2%} (sd {ds['std']:.2%}), "
        f"95% CI {_ci(ds)}, seeds +/0/−: "
        f"{ds['positive']}/{ds['zero']}/{ds['negative']}",
        f"- Actsemble − control: mean {dc['mean']:+.2%} (sd {dc['std']:.2%}), "
        f"95% CI {_ci(dc)}, seeds +/0/−: "
        f"{dc['positive']}/{dc['zero']}/{dc['negative']}",
        "",
        "## Diagnostics",
    ]
    for name, p in diag["per_pair"].items():
        c = p["paired_categories"]
        lines.append(
            f"- {name}: A-only {c['actsemble_only_success']} / S-only "
            f"{c['standalone_only_success']} / both+ {c['both_succeed']} / both- "
            f"{c['both_fail']}; change {p['candidate_change_rate']:.1%}; "
            f"selected |a| {p['selected_chunk_mean_abs']['mean']:.3f} vs zero "
            f"{p['candidate_zero_mean_abs']['mean']:.3f}; "
            f"smoothness {p['selected_chunk_smoothness_mean_abs_delta']['mean']:.3f} vs "
            f"{p['candidate_zero_smoothness_mean_abs_delta']['mean']:.3f}"
        )
    if replay:
        lines.append("")
        for name, r in replay.items():
            lines.append(
                f"- {name} (replayed {r['episodes_replayed']} eps, outcomes match "
                f"{r['replay_outcome_matches_recorded']}): candidate diversity "
                f"{r['candidate_diversity_mean_pairwise_l2']:.3f}; support distance "
                f"selected {r['support_distance_selected_mean']:.4f} vs zero "
                f"{r['support_distance_candidate_zero_mean']:.4f}"
            )
    (ANA / "comparison_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _ci(entry):
    ci = entry.get("ci95_across_pairs")
    return f"[{ci[0]:+.2%}, {ci[1]:+.2%}]" if ci else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["diagnostics", "replay-sample", "videos", "report"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    {"diagnostics": cmd_diagnostics, "replay-sample": cmd_replay_sample,
     "videos": cmd_videos, "report": cmd_report}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

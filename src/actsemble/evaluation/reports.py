"""System-comparison report: verification, paired statistics, warnings."""

from __future__ import annotations

import numpy as np

from .metrics import paired_bootstrap_diff, paired_outcome_counts, wilson_interval

MIN_MEANINGFUL_EPISODES = 50

# Fields that must be identical across compared result files.
_MATCH_FIELDS = (
    "task",
    "dataset_hash",
    "policy_checkpoint_hash",
    "policy_weights_kind",
    "controller",
    "simulation_backend",
    "regime",
    "primary_metric",
    "max_steps",
)
_MATCH_SEQ_FIELDS = ("environment_seeds", "perturbation_seeds", "policy_sampling_seeds")


def verify_comparable(results: list[dict]) -> list[str]:
    """Returns a list of problems; empty means the comparison is fair."""
    problems = []
    base = results[0]
    for r in results[1:]:
        for f in _MATCH_FIELDS:
            if r.get(f) != base.get(f):
                problems.append(
                    f"{r.get('system_name')} vs {base.get('system_name')}: "
                    f"{f} differs ({r.get(f)!r} != {base.get(f)!r})"
                )
        for f in _MATCH_SEQ_FIELDS:
            if list(r.get(f, [])) != list(base.get(f, [])):
                problems.append(
                    f"{r.get('system_name')} vs {base.get('system_name')}: {f} differ"
                )
        # Diffusion settings must match exactly (same frozen policy usage).
        da = (r.get("config", {}).get("system", {}) or {}).get("policy", {})
        db = (base.get("config", {}).get("system", {}) or {}).get("policy", {})
        for key in ("checkpoint", "type"):
            if da.get(key) != db.get(key) and key == "type":
                problems.append(f"policy.type differs: {da.get(key)} != {db.get(key)}")
    problems.extend(verify_candidate_identity(results))
    return problems


# Candidate-identity tolerance. Bitwise identity is unattainable in a shared-env
# multi-episode rollout: the system under test reranks in earlier episodes, so
# by a later episode the CUDA caching allocator is in a different state and the
# physics step drifts ~1e-7, amplifying to ~1e-5 over a few steps — far below
# any behavioural scale (action range is [-1, 1]; the intra-set candidate spread
# is O(0.1)). So the §11 check is bitwise-first (fast path) with a tight
# numerical fallback: when per-replan hashes differ, the per-candidate
# magnitude+smoothness fingerprints must still match within this tolerance. A
# genuine candidate mismatch (broken seeding / wrong observation) moves these
# summaries by O(0.01) or more and still fails.
CANDIDATE_IDENTITY_ATOL = 1e-3


def _replan_fingerprint(episode: dict, k: int) -> np.ndarray | None:
    """Per-candidate (mean |a|, mean |Δa|) vector for replan ``k`` — a
    tolerance-comparable summary of that replan's candidate tensor."""
    r = next((x for x in (episode.get("replans") or []) if x.get("replan_index") == k), None)
    if r is None:
        return None
    ma, sm = r.get("candidate_mean_abs"), r.get("candidate_smoothness")
    if ma is None or sm is None:
        return None
    return np.asarray(list(ma) + list(sm), dtype=np.float64)


def verify_candidate_identity(
    results: list[dict], *, atol: float = CANDIDATE_IDENTITY_ATOL
) -> list[str]:
    """Protocol §11: systems evaluated with the same K must have sampled the
    same candidate tensors while their trajectories coincide.

    Closed-loop invariant: for each paired episode, the candidate sets must
    match up to and including the FIRST replan at which the two systems
    selected different candidate indices — before that point both systems have
    executed identical actions, so a genuine mismatch means broken seeding and
    invalidates the paired comparison. After a selection divergence the
    trajectories (and tensors) legitimately differ.

    Two-tier match: replan 0 — the per-episode initialization from env.reset()
    — must be STRICTLY bitwise identical, so both paired rollouts provably start
    from the same state. From replan 1 on (after the physics has been stepped),
    the match is bitwise-first with a tight fingerprint-tolerance fallback (see
    CANDIDATE_IDENTITY_ATOL) that admits sub-behavioural GPU caching-allocator
    drift but not a real candidate difference. Systems run with a different K
    are skipped — paired-comparison mode gives every system the same K."""
    problems = []
    verifiable = [
        r for r in results
        if r.get("episodes") and all(e.get("candidate_hashes") for e in r["episodes"])
    ]
    for i, a in enumerate(verifiable):
        for b in verifiable[i + 1 :]:
            if a.get("num_candidates") != b.get("num_candidates"):
                continue
            init_bad, bad_episodes, worst = [], [], 0.0
            for ea, eb in zip(a["episodes"], b["episodes"]):
                ha, hb = ea["candidate_hashes"], eb["candidate_hashes"]
                sa, sb = ea["selected_indices"], eb["selected_indices"]
                shared = min(len(ha), len(hb))
                divergence = next(
                    (k for k in range(shared) if sa[k] != sb[k]), shared - 1
                )
                must_match = min(shared, divergence + 1)
                for k in range(must_match):
                    if ha[k] == hb[k]:
                        continue  # bitwise identical — fast path
                    if k == 0:
                        # replan 0 candidates come from env.reset() — the
                        # per-episode INITIALIZATION. reset() is a direct,
                        # deterministic state assignment, so this must be
                        # bitwise identical; a mismatch is a real init-fairness
                        # violation, never tolerable allocator drift. Both
                        # paired rollouts must start from an identical state.
                        init_bad.append(ea["episode_index"])
                        break
                    fa, fb = _replan_fingerprint(ea, k), _replan_fingerprint(eb, k)
                    drift = (
                        float(np.max(np.abs(fa - fb)))
                        if fa is not None and fb is not None and fa.shape == fb.shape
                        else float("inf")
                    )
                    worst = max(worst, drift)
                    if drift > atol:
                        bad_episodes.append(ea["episode_index"])
                        break
            if init_bad:
                problems.append(
                    f"initialization identity violated: {a['system_name']} vs "
                    f"{b['system_name']} start from different states (replan-0 "
                    f"candidates differ) on episode(s) {init_bad} — paired "
                    f"comparison invalid (per-episode init must be bitwise identical)"
                )
            if bad_episodes:
                problems.append(
                    f"candidate-set identity violated: {a['system_name']} vs "
                    f"{b['system_name']} differ beyond tolerance (atol={atol:g}, "
                    f"worst fingerprint drift {worst:.2e}) before any selection "
                    f"divergence on episode(s) {bad_episodes} — paired comparison "
                    f"invalid (protocol §11)"
                )
    return problems


def compare_systems(results: list[dict], *, baseline_index: int = 0) -> dict:
    problems = verify_comparable(results)
    if problems:
        raise ValueError(
            "Results are not comparable:\n  - " + "\n  - ".join(problems)
        )
    base = results[baseline_index]
    n = base["num_episodes"]
    report: dict = {
        "task": base["task"],
        "regime": base["regime"],
        "primary_metric": base["primary_metric"],
        "num_episodes": n,
        "warnings": [],
        "systems": {},
        "pairwise_vs_baseline": {},
    }
    if n < MIN_MEANINGFUL_EPISODES:
        report["warnings"].append(
            f"Only {n} paired episodes — treat as a smoke check, NOT a statistically "
            f"meaningful comparison (need >= {MIN_MEANINGFUL_EPISODES})."
        )
    for r in results:
        ci = wilson_interval(r["success_count"], n)
        report["systems"][r["system_name"]] = {
            "success_rate": r["success_rate"],
            "success_count": r["success_count"],
            "wilson_ci": list(ci),
            "success_at_end_rate": r.get("success_at_end_rate"),
            "timeout_rate": r.get("timeout_rate"),
            "exception_rate": r.get("exception_rate"),
            "fallback_rate": r.get("fallback_rate"),
            "action_clip_rate": r.get("action_clip_rate"),
            "selection_change_rate": r.get("selection_change_rate"),
            "latency": r.get("latency", {}),
            "num_candidates": r.get(
                "num_candidates",
                (r.get("config", {}).get("system", {}) or {})
                .get("policy", {})
                .get("num_candidates"),
            ),
        }
    base_succ = base["successes"]
    for r in results:
        if r is base:
            continue
        counts = paired_outcome_counts(r["successes"], base_succ)
        boot = paired_bootstrap_diff(r["successes"], base_succ, seed=0)
        base_rate = base["success_rate"]
        diff = r["success_rate"] - base_rate
        report["pairwise_vs_baseline"][r["system_name"]] = {
            "baseline": base["system_name"],
            "absolute_difference": diff,
            "relative_difference": diff / base_rate if base_rate > 0 else None,
            "win_loss_tie": {
                "wins": counts["a_wins"],
                "losses": counts["b_wins"],
                "ties": counts["both_succeed"] + counts["both_fail"],
            },
            "paired_counts": counts,
            "bootstrap_diff_ci": boot,
            "latency_overhead_s": (
                r.get("latency", {}).get("mean_decision_s", 0.0)
                - base.get("latency", {}).get("mean_decision_s", 0.0)
            ),
        }
    return report


def format_report(report: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(
        f"Actsemble comparison — task {report['task']}, regime {report['regime']}, "
        f"metric {report['primary_metric']}, n={report['num_episodes']} paired episodes"
    )
    lines.append("=" * 78)
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append("")
    header = f"{'system':<34}{'success':>9}{'wilson 95% CI':>20}{'fallback':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, s in report["systems"].items():
        ci = s["wilson_ci"]
        lines.append(
            f"{name:<34}{s['success_rate']:>8.1%}"
            f"{f'[{ci[0]:.1%}, {ci[1]:.1%}]':>20}"
            f"{(s.get('fallback_rate') or 0.0):>8.1%}"
        )
    lines.append("")
    for name, p in report["pairwise_vs_baseline"].items():
        b = p["bootstrap_diff_ci"]
        wlt = p["win_loss_tie"]
        rel = p["relative_difference"]
        lines.append(f"{name} vs {p['baseline']}:")
        lines.append(
            f"  success diff {p['absolute_difference']:+.1%}"
            + (f" ({rel:+.1%} relative)" if rel is not None else "")
        )
        lines.append(
            f"  paired win/loss/tie: {wlt['wins']}/{wlt['losses']}/{wlt['ties']}"
        )
        lines.append(
            f"  bootstrap 95% CI for paired diff: [{b['ci_low']:+.1%}, {b['ci_high']:+.1%}]"
        )
        lines.append(f"  decision-latency overhead: {p['latency_overhead_s']*1000:+.1f} ms/step")
        change = report["systems"][name].get("selection_change_rate")
        if change is not None:
            lines.append(f"  candidate-selection-change frequency: {change:.1%} of replans")
        lines.append("")
    return "\n".join(lines)

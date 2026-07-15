"""Integration evaluation (§12): implementation checking only.

Runs the three frozen systems on the integration panel and executes the
§12 checklist. Integration success rates are recorded but must never be
used for selection, tuning, or claims. Only implementation defects may be
fixed after integration; any method change requires a new experiment
version and a new untouched final-test panel.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..components.action_chunk_compatibility import ActionChunkCompatibility
from ..evaluation.evaluator import evaluate_system, run_panel_episode
from ..evaluation.panels import Panel, panel_episodes
from ..policies.diffusion.policy import DiffusionPolicy
from ..systems.candidate_reranking import CandidateRerankingActsemble
from ..utils.serialization import save_json
from .freeze import verify_result_against_freeze

SYSTEM_CFGS = {
    "standalone": {"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"}},
    "control": {"policy": {"num_candidates": 1}, "selection": {"type": "first_candidate"}},
    "actsemble": {"policy": {"num_candidates": 1},
                  "selection": {"type": "highest_component_score"}},
}


def _param_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class _ExplodingVerifier:
    """Fault injector for the fallback-behavior check."""

    def __init__(self, real):
        self._real = real
        self.meta = real.meta
        self.dataset_hash = real.dataset_hash
        self.checkpoint_hash = real.checkpoint_hash

    def reset(self):
        pass

    def score(self, obs, chunks):
        raise RuntimeError("integration fault injection")


def run_integration(
    *,
    seed_dir: str | Path,
    freeze: dict,
    panel: Panel,
    env,
    device: str = "cuda",
    max_steps: int = 100,
    force: bool = False,
) -> dict:
    out_dir = Path(seed_dir) / "integration"
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"[integration] {'PASS' if passed else 'FAIL'} | {name}"
              + (f" — {detail}" if detail else ""))

    k = int(freeze["system"]["num_candidates"])
    policy_path = freeze["policy"]["path"]
    verifier_path = freeze["verifier"]["path"]
    eval_cfg = {
        "regime": "integration",
        "panel": panel.to_dict(),
        "max_steps": max_steps,
        "perturbations": [],
        "video": {"max_success": 1, "max_failure": 1},
    }

    # -- run the three frozen systems in paired mode -------------------------
    # Each system evaluates in its OWN freshly created (and warmed) env so
    # every system sees a bitwise-identical simulator history; sharing one
    # env would make results depend on evaluation order (see env_factory
    # warmup notes). The caller-provided env is used only for probe checks.
    results = {}
    for name, sys_cfg in SYSTEM_CFGS.items():
        results[name] = evaluate_system(
            system_cfg=sys_cfg,
            eval_cfg=eval_cfg,
            policy_checkpoint=policy_path,
            component_checkpoints=[verifier_path] if name == "actsemble" else [],
            output_path=out_dir / f"eval_{name}.json",
            video_dir=out_dir / "videos",
            device=device,
            env=None,
            num_candidates_override=k,
            force=force,
        )

    # 1. policy checkpoint identity
    hashes = {r["policy_checkpoint_hash"] for r in results.values()}
    check("policy_checkpoint_identity",
          hashes == {freeze["policy"]["checkpoint_hash"]},
          f"{len(hashes)} distinct hash(es)")

    # 2. dataset / subset hash compatibility (also enforced at build time)
    check("dataset_subset_hash_compatibility",
          all(not verify_result_against_freeze(r, freeze)
              or all("hash" not in p for p in verify_result_against_freeze(r, freeze))
              for r in results.values()),
          "results match frozen dataset hashes")

    # 3. verifier checkpoint loading
    verifier = ActionChunkCompatibility.from_checkpoint(verifier_path, device=device)
    check("verifier_checkpoint_loaded",
          verifier.checkpoint_hash == freeze["verifier"]["checkpoint_hash"])

    # 4. observation / action dimensions (verify_env_matches ran inside every
    # evaluate_system call and would have raised)
    check("observation_action_dimensions", True, "verified by env contract checks")

    # 5. history reset between episodes
    policy = DiffusionPolicy.from_checkpoint(policy_path, device=device, use_ema=True)
    probe = CandidateRerankingActsemble(policy, verifier, num_candidates=k)
    ep0 = panel_episodes(panel)[0]
    run_panel_episode(env, probe, ep0, max_steps=min(max_steps, 20), pert_specs=[])
    ran_replans = probe.diagnostics()["num_replans"]
    probe.reset(episode_seed=0)
    check("history_reset_between_episodes",
          ran_replans > 0 and probe.diagnostics()["num_replans"] == 0
          and len(probe._history) == 0 and len(probe._queue) == 0)

    # 6. identical candidate tensors (§11 closed-loop rule: bitwise equal up
    # to the first selection divergence; standalone vs control must match
    # on every replan since both always select candidate zero)
    from ..evaluation.reports import verify_candidate_identity

    identity_problems = verify_candidate_identity(list(results.values()))
    standalone_control_full = all(
        ea["candidate_hashes"] == eb["candidate_hashes"]
        for ea, eb in zip(results["standalone"]["episodes"], results["control"]["episodes"])
    )
    check("identical_candidate_tensors",
          not identity_problems and standalone_control_full,
          "; ".join(identity_problems) or "prefix rule + full standalone/control equality")

    # 7. finite verifier scores on a probe episode
    probe.reset(episode_seed=0)
    result, _ = run_panel_episode(env, probe, ep0, max_steps=min(max_steps, 20), pert_specs=[])
    scores = [s for r in probe.diagnostics()["replans"]
              for s in r.get("component_scores", [])]
    check("finite_verifier_scores",
          len(scores) > 0 and np.isfinite(scores).all()
          and results["actsemble"]["fallback_rate"] == 0.0,
          f"{len(scores)} scores checked")

    # 8. fallback behavior under fault injection
    faulty = CandidateRerankingActsemble(policy, _ExplodingVerifier(verifier), num_candidates=k)
    fresult, _ = run_panel_episode(env, faulty, ep0, max_steps=min(max_steps, 20), pert_specs=[])
    fdiag = faulty.diagnostics()
    check("fallback_behavior",
          fresult.exception is None and fdiag["fallback_count"] == fdiag["num_replans"] > 0
          and all(r["selected_index"] == 0 for r in fdiag["replans"]),
          "episode survived a always-raising verifier via candidate zero")

    # 9. logging outputs
    check("logging_outputs_written",
          all((out_dir / f"eval_{n}.json").exists() for n in SYSTEM_CFGS))

    # 10. video generation
    videos = list((out_dir / "videos").glob("*.mp4"))
    check("video_generation", len(videos) >= 1, f"{len(videos)} file(s)")

    # 11. no simulator information in verifier input: the input layer is
    # exactly sized for (obs history ++ action chunk) — no room for reward,
    # success, or any other simulator-derived feature.
    meta = verifier.meta
    feat = int(meta["state_dim"]) + (
        int(meta["action_dim"]) if meta.get("include_previous_action") else 0
    )
    expected_in = int(meta["obs_horizon"]) * feat + int(meta["prediction_horizon"]) * int(
        meta["action_dim"]
    )
    first_linear = next(m for m in verifier.model.net if isinstance(m, torch.nn.Linear))
    check("no_simulator_information_in_verifier_input",
          first_linear.in_features == expected_in,
          f"input dim {first_linear.in_features} == H_o*feat + H_p*A = {expected_in}")

    # 12. no policy-weight updates during evaluation
    before = _param_digest(policy.model)
    probe2 = CandidateRerankingActsemble(policy, verifier, num_candidates=k)
    run_panel_episode(env, probe2, ep0, max_steps=min(max_steps, 20), pert_specs=[])
    check("no_policy_weight_updates", _param_digest(policy.model) == before)

    passed = all(c["passed"] for c in checks)
    report = {
        "passed": passed,
        "checks": checks,
        "panel": panel.to_dict(),
        "note": ("Integration success rates are for implementation checking only — "
                 "never for selection, tuning, or reported claims (§1.3, §12)."),
        "success_rates_not_for_claims": {
            name: r["success_rate"] for name, r in results.items()
        },
    }
    save_json(report, out_dir / "integration_report.json")
    print(f"[integration] {'ALL CHECKS PASSED' if passed else 'FAILURES PRESENT'}")
    return report

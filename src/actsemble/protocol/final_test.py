"""Final-test evaluation (§13): the only reported numbers.

Gated: requires a freeze manifest and a fully passed integration report.
Runs the three systems in paired-candidate mode on the untouched final
panel, verifies every result against the freeze, verifies candidate-set
identity, and writes the comparison. Rerunning over existing results
requires an explicit force flag (§17).
"""

from __future__ import annotations

from pathlib import Path

from ..evaluation.reports import compare_systems, format_report
from ..utils.serialization import load_json, save_json
from .freeze import final_panel_from_freeze, load_freeze, verify_result_against_freeze
from .integration import SYSTEM_CFGS


def run_final_test(
    *,
    seed_dir: str | Path,
    env=None,  # unused: each system builds its own warmed env (see integration)
    device: str = "cuda",
    regime: dict | None = None,
    force: bool = False,
) -> dict:
    from ..evaluation.evaluator import evaluate_system

    seed_dir = Path(seed_dir)
    freeze = load_freeze(seed_dir)

    integration_report_path = seed_dir / "integration" / "integration_report.json"
    if not integration_report_path.exists():
        raise FileNotFoundError(
            "Integration report missing — run integration before the final test (§12, §13)."
        )
    if not load_json(integration_report_path)["passed"]:
        raise RuntimeError(
            "Integration checks failed — fix implementation defects and re-run "
            "integration before touching the final-test panel (§12)."
        )

    panel = final_panel_from_freeze(freeze)
    k = int(freeze["system"]["num_candidates"])
    regime = regime or {"name": "nominal", "max_steps": 100, "perturbations": []}
    eval_cfg = {
        "regime": regime.get("name", "nominal"),
        "panel": panel.to_dict(),
        "max_steps": int(regime.get("max_steps", 100)),
        "perturbations": regime.get("perturbations", []) or [],
        "video": regime.get("video", {"max_success": 2, "max_failure": 2}),
    }
    out_dir = seed_dir / "final_test" / eval_cfg["regime"]

    results = {}
    for name, sys_cfg in SYSTEM_CFGS.items():
        # Fresh warmed env per system: identical simulator histories are a
        # precondition of the §11 candidate-identity requirement.
        results[name] = evaluate_system(
            system_cfg=sys_cfg,
            eval_cfg=eval_cfg,
            policy_checkpoint=freeze["policy"]["path"],
            component_checkpoints=[freeze["verifier"]["path"]] if name == "actsemble" else [],
            output_path=out_dir / f"eval_{name}.json",
            video_dir=out_dir / "videos",
            device=device,
            env=None,
            num_candidates_override=k,
            force=force,
        )
        problems = verify_result_against_freeze(results[name], freeze)
        if problems:
            raise RuntimeError(
                f"Final-test result for {name} violates the freeze manifest:\n  - "
                + "\n  - ".join(problems)
            )
        r = results[name]
        print(f"[final-test] {r['system_name']}: {r['success_count']}/{r['num_episodes']} "
              f"= {r['success_rate']:.1%}")

    # Candidate identity + full comparability verification (raises on
    # violation — a candidate mismatch invalidates the paired comparison).
    report = compare_systems(
        [results["standalone"], results["control"], results["actsemble"]]
    )
    print(format_report(report))
    save_json(report, out_dir / "comparison.json")
    return {"results": results, "comparison": report, "output_dir": str(out_dir)}

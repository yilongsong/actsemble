"""System freezing (§10) and pre-evaluation verification.

The freeze manifest pins everything §10 lists — policy checkpoint hash,
EMA choice, sampler, inference steps, temperature, horizons, K, verifier
checkpoint hash, normalization, selection and fallback rules — plus the
integration and final-test panels. It is written once, after selection and
before integration/final evaluation; panel disjointness (including the
demonstration-generation seeds) is verified here.
"""

from __future__ import annotations

from pathlib import Path

from ..components.action_chunk_compatibility import ActionChunkCompatibility
from ..data.reader import DatasetReader
from ..evaluation.panels import Panel, assert_panels_disjoint
from ..policies.interface import sampler_provenance
from ..policies.loader import load_policy
from ..systems.interface import check_same_data
from ..utils.hashing import hash_json
from ..utils.provenance import runtime_provenance
from ..utils.repo import current_git_commit, git_provenance
from ..utils.serialization import load_json, save_json
from .experiment import require_absent

FREEZE_FILENAME = "freeze.json"


def demonstration_seeds(dataset_path: str | Path) -> set[int]:
    """Environment seeds used at demonstration generation, from provenance."""
    sidecar = Path(str(dataset_path) + ".provenance.json")
    if not sidecar.exists():
        raise FileNotFoundError(f"Provenance sidecar missing: {sidecar}")
    prov = load_json(sidecar)
    return {
        int(e["source_episode_seed"])
        for e in prov.get("exported_episodes", [])
        if "source_episode_seed" in e
    }


def build_freeze_manifest(
    *,
    policy_path: str | Path,
    verifier_path: str | Path,
    dataset_path: str | Path,
    panels: dict[str, Panel],
    num_candidates: int,
    device: str = "cpu",
) -> dict:
    policy = load_policy(policy_path, device=device, use_ema=True)
    verifier = ActionChunkCompatibility.from_checkpoint(verifier_path, device=device)
    check_same_data(policy, [verifier])  # same-data contract before freezing

    assert_panels_disjoint(
        panels, extra_seed_sets={"demonstrations": demonstration_seeds(dataset_path)}
    )

    reader = DatasetReader(dataset_path)
    return {
        "protocol": "actsemble_system_freeze_v2",
        "git_commit": current_git_commit(),
        "source": git_provenance(),
        "runtime": runtime_provenance(),
        "policy": {
            "path": str(policy_path),
            "checkpoint_hash": policy.checkpoint_hash,
            "weights_kind": policy.weights_kind,
            "family": sampler_provenance(policy)["family"],
            "obs_horizon": policy.meta.obs_horizon,
            "prediction_horizon": policy.meta.prediction_horizon,
            "action_horizon": policy.meta.action_horizon,
            "include_previous_action": policy.meta.include_previous_action,
            "normalization_hash": _norm_hash(policy.meta.normalization),
            "config_hash": hash_json(policy.config),
            "sampler_provenance": sampler_provenance(policy),
            "window_alignment": policy.meta.extra.get(
                "window_alignment", "future_only"
            ),
        },
        "verifier": {
            "path": str(verifier_path),
            "checkpoint_hash": verifier.checkpoint_hash,
            "normalization_hash": _norm_hash(verifier.meta["normalization"]),
        },
        "dataset": {
            "path": str(dataset_path),
            "dataset_hash": policy.dataset_hash,
            "subset_hash": policy.meta.extra.get("subset_hash"),
            "split_hash": policy.meta.split_hash,
        },
        "system": {
            "num_candidates": int(num_candidates),
            "selection_rule": "highest_component_score",
            "fallback_rule": "candidate_zero_if_finite_else_first_valid; no_finite_candidate_raises",
            "paired_candidates": True,
            "allowed_selection_types": [
                "candidate_zero",
                "first_candidate",
                "highest_component_score",
            ],
        },
        "environment": {
            "task_id": policy.meta.task_id,
            "controller": policy.meta.controller,
            "simulation_backend": policy.meta.simulation_backend,
            "simulator": reader.metadata.simulator,
            "simulator_version": reader.metadata.simulator_version,
            "robot": reader.metadata.robot,
            "observation_mode": reader.metadata.observation_mode,
            "control_frequency": reader.metadata.control_frequency,
            "state_dimension": reader.metadata.state_dimension,
            "action_dimension": reader.metadata.action_dimension,
        },
        "panels": {name: p.to_dict() for name, p in panels.items()},
    }


def _norm_hash(normalization: dict) -> str:
    from ..utils.hashing import hash_json

    return hash_json(normalization)


def write_freeze(seed_dir: str | Path, manifest: dict, *, force: bool = False) -> Path:
    path = Path(seed_dir) / FREEZE_FILENAME
    require_absent(path, force=force, what="Freeze manifest")
    save_json(manifest, path)
    return path


def load_freeze(seed_dir: str | Path) -> dict:
    path = Path(seed_dir) / FREEZE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — systems must be frozen (run_protocol.py freeze) "
            f"before integration or final-test evaluation (§10, §13)."
        )
    return load_json(path)


def verify_result_against_freeze(result: dict, manifest: dict) -> list[str]:
    """Field-by-field check of an evaluation result against the freeze."""
    problems = []
    if (
        manifest.get("protocol") == "actsemble_system_freeze_v2"
        and result.get("result_schema") != "actsemble_evaluation_v2"
    ):
        problems.append("result_schema: freeze v2 requires 'actsemble_evaluation_v2'")
    checks = [
        ("policy_checkpoint_hash", manifest["policy"]["checkpoint_hash"]),
        ("policy_weights_kind", manifest["policy"]["weights_kind"]),
        ("dataset_hash", manifest["dataset"]["dataset_hash"]),
        ("controller", manifest["environment"]["controller"]),
        ("simulation_backend", manifest["environment"]["simulation_backend"]),
        ("task", manifest["environment"]["task_id"]),
        ("num_candidates", manifest["system"]["num_candidates"]),
    ]
    optional_checks = [
        ("split_hash", manifest.get("dataset", {}).get("split_hash")),
        ("subset_hash", manifest.get("dataset", {}).get("subset_hash")),
        ("normalization_hash", manifest.get("policy", {}).get("normalization_hash")),
        ("policy_config_hash", manifest.get("policy", {}).get("config_hash")),
    ]
    checks.extend((k, v) for k, v in optional_checks if v is not None)
    for key, expected in checks:
        if result.get(key) != expected:
            problems.append(f"{key}: result {result.get(key)!r} != frozen {expected!r}")
    selection_type = result.get("selection_type")
    allowed = manifest.get("system", {}).get("allowed_selection_types")
    if allowed is not None and selection_type not in allowed:
        problems.append(f"selection_type {selection_type!r} is not frozen/allowed")
    actual_components = result.get("component_checkpoint_hashes", [])
    if selection_type is None and allowed is None:  # legacy v1 manifests/results
        expected_components = (
            [manifest["verifier"]["checkpoint_hash"]] if actual_components else []
        )
    else:
        expected_components = (
            [manifest["verifier"]["checkpoint_hash"]]
            if selection_type == "highest_component_score"
            else []
        )
    if actual_components != expected_components:
        problems.append(
            f"component checkpoint hashes {actual_components!r} != frozen expected "
            f"{expected_components!r}"
        )

    frozen_sampler = manifest.get("policy", {}).get("sampler_provenance")
    if frozen_sampler is not None and result.get("sampler") != frozen_sampler:
        problems.append("sampler provenance differs from frozen manifest")
    frozen_execution = manifest.get("policy", {})
    result_execution = result.get("execution", {})
    expected_offset = None
    if "window_alignment" in frozen_execution:
        expected_offset = (
            int(frozen_execution.get("obs_horizon", 1)) - 1
            if frozen_execution["window_alignment"] == "diffusion_policy"
            else 0
        )
    for key, expected in (
        ("action_horizon", frozen_execution.get("action_horizon")),
        ("action_offset", expected_offset),
    ):
        if expected is not None and result_execution.get(key) != expected:
            problems.append(
                f"execution {key}: result {result_execution.get(key)!r} != frozen {expected!r}"
            )
    expected_fallback = manifest.get("system", {}).get("fallback_rule")
    if (
        expected_fallback is not None
        and result.get("fallback_rule") != expected_fallback
    ):
        problems.append("fallback rule differs from frozen manifest")
    frozen_source = manifest.get("source", {}).get("source_tree_hash")
    if (
        frozen_source is not None
        and result.get("source", {}).get("source_tree_hash") != frozen_source
    ):
        problems.append("source tree hash differs from frozen manifest")
    for key in ("python", "numpy", "torch", "cuda_runtime", "cudnn"):
        expected = manifest.get("runtime", {}).get(key)
        actual = result.get("runtime", {}).get(key)
        if expected is not None and actual != expected:
            problems.append(f"runtime {key}: result {actual!r} != frozen {expected!r}")
    frozen_panels = manifest.get("panels", {})
    frozen_panel = frozen_panels.get(result.get("panel", {}).get("name"))
    if frozen_panels and frozen_panel is None:
        problems.append("evaluation panel is not present in the frozen manifest")
    elif frozen_panel is not None and result.get("panel") != frozen_panel:
        problems.append("evaluation panel differs from frozen manifest")
    frozen_env = manifest.get("environment", {})
    live_env = result.get("environment_contract", {})
    env_key_map = {
        "task_id": "task_id",
        "controller": "controller",
        "simulation_backend": "simulation_backend",
        "simulator_version": "simulator_version",
        "robot": "robot",
        "control_frequency": "control_frequency",
        "action_dimension": "action_dimension",
    }
    if "environment_contract" in result:  # absent in legacy v1 result files
        for frozen_key, live_key in env_key_map.items():
            expected = frozen_env.get(frozen_key)
            if expected is not None and live_env.get(live_key) != expected:
                problems.append(
                    f"environment {live_key}: result {live_env.get(live_key)!r} != frozen {expected!r}"
                )
    return problems


def final_panel_from_freeze(manifest: dict) -> Panel:
    p = manifest["panels"]["final_test"]
    return Panel(
        name="final_test", env_seed=p["env_seed"], num_episodes=p["num_episodes"]
    )

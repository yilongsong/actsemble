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
from ..evaluation.panels import Panel, assert_panels_disjoint
from ..policies.diffusion.policy import DiffusionPolicy
from ..systems.interface import check_same_data
from ..utils.repo import current_git_commit
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
    policy = DiffusionPolicy.from_checkpoint(policy_path, device=device, use_ema=True)
    verifier = ActionChunkCompatibility.from_checkpoint(verifier_path, device=device)
    check_same_data(policy, [verifier])  # same-data contract before freezing

    assert_panels_disjoint(
        panels, extra_seed_sets={"demonstrations": demonstration_seeds(dataset_path)}
    )

    dcfg = policy.config.get("diffusion", {})
    return {
        "protocol": "actsemble_system_freeze_v1",
        "git_commit": current_git_commit(),
        "policy": {
            "path": str(policy_path),
            "checkpoint_hash": policy.checkpoint_hash,
            "weights_kind": "ema",
            "sampler": policy.sampler,
            "inference_steps": policy.num_inference_steps,
            "temperature": policy.temperature,
            "obs_horizon": policy.meta.obs_horizon,
            "prediction_horizon": policy.meta.prediction_horizon,
            "action_horizon": policy.meta.action_horizon,
            "include_previous_action": policy.meta.include_previous_action,
            "normalization_hash": _norm_hash(policy.meta.normalization),
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
            "fallback_rule": "candidate_zero",
            "paired_candidates": True,
        },
        "environment": {
            "task_id": policy.meta.task_id,
            "controller": policy.meta.controller,
            "simulation_backend": policy.meta.simulation_backend,
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
    checks = [
        ("policy_checkpoint_hash", manifest["policy"]["checkpoint_hash"]),
        ("policy_weights_kind", manifest["policy"]["weights_kind"]),
        ("dataset_hash", manifest["dataset"]["dataset_hash"]),
        ("controller", manifest["environment"]["controller"]),
        ("simulation_backend", manifest["environment"]["simulation_backend"]),
        ("task", manifest["environment"]["task_id"]),
        ("num_candidates", manifest["system"]["num_candidates"]),
    ]
    for key, expected in checks:
        if result.get(key) != expected:
            problems.append(f"{key}: result {result.get(key)!r} != frozen {expected!r}")
    if result.get("component_checkpoint_hashes"):
        if result["component_checkpoint_hashes"] != [manifest["verifier"]["checkpoint_hash"]]:
            problems.append("verifier checkpoint hash differs from frozen manifest")
    return problems


def final_panel_from_freeze(manifest: dict) -> Panel:
    p = manifest["panels"]["final_test"]
    return Panel(name="final_test", env_seed=p["env_seed"], num_episodes=p["num_episodes"])

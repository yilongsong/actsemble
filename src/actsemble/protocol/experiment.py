"""Experiment versioning, frozen specs, and overwrite protection (§17, §18).

An experiment directory is created once from a spec file; the spec copy
inside the directory is the frozen source of truth for every later stage.
Completed artifacts are never silently overwritten: a rerun needs an
explicit ``force`` or a new experiment-version directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..config import load_config, merge_config
from ..utils.repo import current_git_commit
from ..utils.serialization import save_json

SPEC_FILENAME = "spec.yaml"


def require_absent(path: str | Path, *, force: bool, what: str) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"{what} already exists at {path}. Completed experiment artifacts "
            f"must not be silently rerun or overwritten (protocol §17): pass "
            f"--force or use a new experiment-version directory."
        )


def init_experiment(spec_path: str | Path, experiment_dir: str | Path) -> dict:
    """Create the experiment dir and freeze the spec inside it.

    Re-running with an identical spec is a no-op; a modified spec against
    an existing experiment dir fails loudly (new version dir required).
    """
    experiment_dir = Path(experiment_dir)
    spec = load_config(spec_path)
    frozen_path = experiment_dir / SPEC_FILENAME
    if frozen_path.exists():
        frozen = load_config(frozen_path)
        if frozen != spec:
            raise ValueError(
                f"Experiment dir {experiment_dir} was initialized with a different "
                f"spec. Changing budgets/panels/rules after inspecting results is "
                f"prohibited (§4, §16) — create a new experiment-version directory."
            )
        return frozen
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(frozen_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    save_json(
        {"git_commit": current_git_commit(), "spec_source": str(spec_path)},
        experiment_dir / "experiment_info.json",
    )
    return spec


def load_experiment_spec(experiment_dir: str | Path) -> dict:
    frozen_path = Path(experiment_dir) / SPEC_FILENAME
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"{frozen_path} not found — initialize the experiment first "
            f"(run_protocol.py init --spec ... --experiment-dir ...)"
        )
    return load_config(frozen_path)


def seed_dir(experiment_dir: str | Path, policy_seed: int) -> Path:
    return Path(experiment_dir) / f"seed_{int(policy_seed)}"


def resolved_policy_config(spec: dict, policy_seed: int) -> dict:
    cfg = load_config(spec["policy_config"])
    cfg = merge_config(cfg, spec.get("policy_config_overrides", {}) or {})
    cfg = merge_config(cfg, {"training": {"seed": int(policy_seed)}})
    return cfg


def resolved_verifier_config(spec: dict, verifier_seed: int) -> dict:
    cfg = load_config(spec["verifier_config"])
    cfg = merge_config(cfg, spec.get("verifier_config_overrides", {}) or {})
    cfg = merge_config(cfg, {"training": {"seed": int(verifier_seed)}})
    return cfg


def verifier_seed_for(spec: dict, policy_seed: int) -> int:
    pairing = spec.get("verifier_seed_pairing", "same_as_policy")
    if pairing == "same_as_policy":
        return int(policy_seed)
    if isinstance(pairing, dict):
        return int(pairing[str(policy_seed)])
    raise ValueError(f"Unknown verifier_seed_pairing: {pairing!r}")

"""Overwrite protection, frozen specs, freeze verification, verifier audit
trail (§10, §16, §17)."""

import pytest
import yaml

from actsemble.protocol.experiment import (
    init_experiment,
    load_experiment_spec,
    require_absent,
    resolved_policy_config,
    verifier_seed_for,
)
from actsemble.protocol.freeze import verify_result_against_freeze
from actsemble.protocol.verifier_selection import select_verifier
from actsemble.utils.serialization import load_json


def test_require_absent_blocks_and_force_allows(tmp_path):
    target = tmp_path / "result.json"
    target.write_text("{}")
    with pytest.raises(FileExistsError, match="§17"):
        require_absent(target, force=False, what="Result")
    require_absent(target, force=True, what="Result")  # no raise


def _write_spec(tmp_path, name="spec.yaml", extra=None):
    spec = {
        "experiment_version": "t1",
        "dataset": "data/x.h5",
        "policy_config": str(tmp_path / "policy.yaml"),
        "verifier_config": str(tmp_path / "verifier.yaml"),
        "policy_seeds": [0, 1],
        **(extra or {}),
    }
    (tmp_path / "policy.yaml").write_text(yaml.safe_dump({"training": {"seed": 99}}))
    (tmp_path / "verifier.yaml").write_text(yaml.safe_dump({"training": {}}))
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec))
    return path


def test_spec_is_frozen_at_init(tmp_path):
    spec_path = _write_spec(tmp_path)
    exp_dir = tmp_path / "exp"
    init_experiment(spec_path, exp_dir)
    assert (exp_dir / "spec.yaml").exists()
    init_experiment(spec_path, exp_dir)  # identical spec: idempotent

    # Changing the spec after init must be rejected (new version required).
    changed = _write_spec(tmp_path, "spec2.yaml", extra={"paired_num_candidates": 64})
    with pytest.raises(ValueError, match="new experiment-version"):
        init_experiment(changed, exp_dir)
    assert load_experiment_spec(exp_dir)["policy_seeds"] == [0, 1]


def test_resolved_config_injects_training_seed(tmp_path):
    spec_path = _write_spec(
        tmp_path, extra={"policy_config_overrides": {"training": {"max_steps": 7}}}
    )
    spec = yaml.safe_load(spec_path.read_text())
    cfg = resolved_policy_config(spec, policy_seed=3)
    assert cfg["training"]["seed"] == 3  # seed comes from the arm, not the base config
    assert cfg["training"]["max_steps"] == 7


def test_verifier_seed_pairing():
    assert verifier_seed_for({"verifier_seed_pairing": "same_as_policy"}, 4) == 4
    assert verifier_seed_for({"verifier_seed_pairing": {"4": 17}}, 4) == 17


def test_verify_result_against_freeze_catches_mismatches():
    manifest = {
        "policy": {"checkpoint_hash": "P", "weights_kind": "ema"},
        "verifier": {"checkpoint_hash": "V"},
        "dataset": {"dataset_hash": "D"},
        "environment": {"task_id": "T-v1", "controller": "c", "simulation_backend": "b"},
        "system": {"num_candidates": 4},
    }
    good = {"policy_checkpoint_hash": "P", "policy_weights_kind": "ema",
            "dataset_hash": "D", "controller": "c", "simulation_backend": "b",
            "task": "T-v1", "num_candidates": 4,
            "component_checkpoint_hashes": ["V"]}
    assert verify_result_against_freeze(good, manifest) == []
    for key, bad_value in [("policy_checkpoint_hash", "X"), ("num_candidates", 16),
                           ("dataset_hash", "X"), ("component_checkpoint_hashes", ["X"])]:
        problems = verify_result_against_freeze({**good, key: bad_value}, manifest)
        assert problems, f"{key} mismatch not caught"


def test_selected_verifier_carries_audit_trail(trained_checkpoints, tmp_path):
    import shutil

    import torch

    # Reuse the session-trained tiny verifier run (it has interval history
    # only if checkpoint_every was set; fabricate the run dir instead).
    run_dir = tmp_path / "verifier_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    snap = run_dir / "checkpoints" / "step_000010.pt"
    shutil.copy(trained_checkpoints["component_best"], snap)
    history = [{"step": 10, "checkpoint_path": str(snap),
                "evaluated_on": "validation_episodes",
                "metrics": {"pairwise_ranking_accuracy": 0.9,
                            "balanced_accuracy": 0.85, "validation_loss": 0.2}}]
    import json

    (run_dir / "offline_history.json").write_text(json.dumps(history))

    report = select_verifier(run_dir)
    ckpt = torch.load(run_dir / "selected_verifier.pt", map_location="cpu",
                      weights_only=False)
    selection = ckpt["meta"]["selection"]
    assert selection["selected_step"] == 10
    assert selection["offline_validation_history"] == history
    assert "no simulator signal" in selection["selection_rule"]
    for key in ("dataset_hash", "split_hash", "normalization", "negatives"):
        assert key in ckpt["meta"], f"verifier meta missing {key}"
    assert report["selected_verifier_hash"]
    saved = load_json(run_dir / "verifier_selection.json")
    assert saved["selected_step"] == 10

    # No-overwrite guard (§17)
    with pytest.raises(FileExistsError, match="§17"):
        select_verifier(run_dir)
    select_verifier(run_dir, force=True)

"""Frozen-dataset validation.

``validate_dataset`` re-checks everything the training and evaluation code
relies on. It is run by ``scripts/inspect_dataset.py``, the smoke test, and
the unit tests. Failures raise ``DatasetValidationError`` with all problems
listed at once.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

from ..utils.serialization import load_json
from .reader import DatasetReader
from .schema import (
    EPISODE_ARRAY_KEYS,
    FORBIDDEN_EPISODE_KEYS,
    PROJECT_NAME,
    REQUIRED_METADATA_KEYS,
    SCHEMA_VERSION,
)
from .writer import compute_dataset_hash


class DatasetValidationError(Exception):
    pass


def validate_dataset(reader: DatasetReader, *, check_hash: bool = True) -> dict:
    """Validate a frozen dataset; returns a summary dict on success."""
    problems: list[str] = []
    meta = reader.metadata

    with h5py.File(reader.path, "r") as f:
        stored_metadata_keys = set(f["metadata"].attrs.keys())
    for key in REQUIRED_METADATA_KEYS:
        if key not in stored_metadata_keys:
            problems.append(f"metadata missing key {key}")
    if meta.project_name != PROJECT_NAME:
        problems.append(f"project_name {meta.project_name!r} != {PROJECT_NAME!r}")
    if meta.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version {meta.schema_version!r} != {SCHEMA_VERSION!r}")
    if meta.state_dimension <= 0:
        problems.append(f"invalid state_dimension {meta.state_dimension}")
    if meta.action_dimension <= 0:
        problems.append(f"invalid action_dimension {meta.action_dimension}")
    if not meta.controller:
        problems.append("controller metadata is empty")
    if not meta.simulation_backend:
        problems.append("simulation_backend metadata is empty")

    if not reader.episodes:
        problems.append("dataset contains no episodes")

    ids = reader.episode_ids
    if len(ids) != len(set(ids)):
        problems.append("episode ids are not unique")

    try:
        action_def = json.loads(meta.action_definition)
    except json.JSONDecodeError:
        action_def = {}
        problems.append("action_definition is not valid JSON")
    bounds = action_def.get("bounds")

    # Leakage guard at the FILE level: episode groups must contain exactly
    # the schema arrays — reward/success/etc. must not exist even as extra
    # arrays the reader would ignore.
    with h5py.File(reader.path, "r") as f:
        for episode_id in f["episodes"]:
            keys = set(f["episodes"][episode_id].keys())
            forbidden = keys & set(FORBIDDEN_EPISODE_KEYS)
            if forbidden:
                problems.append(
                    f"{episode_id}: forbidden field(s) {sorted(forbidden)} present"
                )
            unexpected = keys - set(EPISODE_ARRAY_KEYS)
            if unexpected:
                problems.append(
                    f"{episode_id}: unexpected array(s) {sorted(unexpected)}"
                )

    for ep in reader.episodes:
        T = len(ep)
        if T == 0:
            problems.append(f"{ep.episode_id}: empty episode")
            continue
        # Shape consistency.
        if ep.state.shape != (T, meta.state_dimension):
            problems.append(
                f"{ep.episode_id}: state shape {ep.state.shape} != ({T},{meta.state_dimension})"
            )
        if ep.next_state.shape != (T, meta.state_dimension):
            problems.append(f"{ep.episode_id}: next_state shape {ep.next_state.shape}")
        if ep.action.shape != (T, meta.action_dimension):
            problems.append(
                f"{ep.episode_id}: action shape {ep.action.shape} != ({T},{meta.action_dimension})"
            )
        if ep.previous_action.shape != (T, meta.action_dimension):
            problems.append(
                f"{ep.episode_id}: previous_action shape {ep.previous_action.shape}"
            )
        if ep.step_index.shape != (T,):
            problems.append(f"{ep.episode_id}: step_index shape {ep.step_index.shape}")
        # Finite values.
        for key in ("state", "previous_action", "action", "next_state"):
            arr = getattr(ep, key)
            if not np.isfinite(arr).all():
                problems.append(f"{ep.episode_id}: non-finite values in {key}")
        # Alignment invariants.
        if T >= 2:
            if not np.allclose(ep.state[1:], ep.next_state[:-1], atol=1e-5):
                problems.append(
                    f"{ep.episode_id}: state[t+1] != next_state[t] (alignment broken)"
                )
            if not np.allclose(ep.previous_action[1:], ep.action[:-1], atol=1e-6):
                problems.append(f"{ep.episode_id}: previous_action[t] != action[t-1]")
        if not np.allclose(ep.previous_action[0], 0.0):
            problems.append(f"{ep.episode_id}: previous_action[0] is not zero")
        if not np.array_equal(ep.step_index, np.arange(T)):
            problems.append(f"{ep.episode_id}: step_index is not 0..T-1")
        # Action bounds.
        if bounds is not None:
            low = np.asarray(bounds[0], dtype=np.float32)
            high = np.asarray(bounds[1], dtype=np.float32)
            if (ep.action < low - 1e-5).any() or (ep.action > high + 1e-5).any():
                problems.append(f"{ep.episode_id}: actions outside declared bounds")

    if check_hash and not problems:
        recomputed = compute_dataset_hash(reader.episodes, meta)
        if recomputed != meta.dataset_hash:
            problems.append(
                f"dataset_hash mismatch: stored {meta.dataset_hash[:16]}..., recomputed {recomputed[:16]}..."
            )

    if problems:
        raise DatasetValidationError(
            f"Dataset {reader.path} failed validation with {len(problems)} problem(s):\n  - "
            + "\n  - ".join(problems)
        )

    lengths = [len(ep) for ep in reader.episodes]
    all_states = np.concatenate([ep.state for ep in reader.episodes], axis=0)
    all_actions = np.concatenate([ep.action for ep in reader.episodes], axis=0)
    return {
        "num_episodes": len(reader.episodes),
        "num_transitions": reader.num_transitions,
        "episode_length_min": int(np.min(lengths)),
        "episode_length_max": int(np.max(lengths)),
        "episode_length_mean": float(np.mean(lengths)),
        "state_dim": reader.state_dim,
        "action_dim": reader.action_dim,
        "state_min": all_states.min(axis=0).tolist(),
        "state_max": all_states.max(axis=0).tolist(),
        "action_min": all_actions.min(axis=0).tolist(),
        "action_max": all_actions.max(axis=0).tolist(),
        "dataset_hash": meta.dataset_hash,
        "controller": meta.controller,
        "simulation_backend": meta.simulation_backend,
        "task_id": meta.task_id,
    }


def validate_success_only_provenance(dataset_path) -> dict:
    """Check the private provenance sidecar records success-only filtering.

    The dataset file itself never stores success flags; the sidecar written
    at preparation time is the auditable record that only successful source
    episodes were exported.
    """
    from pathlib import Path

    dataset_path = Path(dataset_path)
    sidecar = dataset_path.with_suffix(dataset_path.suffix + ".provenance.json")
    if not sidecar.exists():
        raise DatasetValidationError(f"Missing provenance sidecar: {sidecar}")
    prov = load_json(sidecar)
    if not prov.get("success_only", False):
        raise DatasetValidationError(
            f"Provenance {sidecar} does not assert success_only=true"
        )
    exported = prov.get("exported_episodes")
    if not exported:
        raise DatasetValidationError("Provenance lists no exported episodes")
    non_successful = [e for e in exported if not e.get("source_success", False)]
    if non_successful:
        raise DatasetValidationError(
            f"{len(non_successful)} exported episode(s) lack source_success=true in provenance"
        )
    return prov

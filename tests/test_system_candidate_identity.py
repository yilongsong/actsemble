"""Candidate-set identity across systems + selection/fallback behavior.

The decisive fairness property: with the same frozen policy, same K, and
same seeds, the multi-sample control and the Actsemble reranker receive
bitwise-identical candidate tensors — only the selection rule differs.
"""

import numpy as np
import pytest
import torch

from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility
from actsemble.policies.diffusion.policy import DiffusionPolicy
from actsemble.systems.candidate_reranking import CandidateRerankingActsemble
from actsemble.systems.factory import build_system
from actsemble.systems.multisample_control import MultiSampleControlSystem
from actsemble.types import StateObservation


class RecordingPolicy:
    """Wraps a DiffusionPolicy, recording every candidate tensor sampled."""

    def __init__(self, inner):
        self.inner = inner
        self.recorded: list[torch.Tensor] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def sample_action_chunks(self, observation_history, *, num_samples, generator):
        out = self.inner.sample_action_chunks(
            observation_history, num_samples=num_samples, generator=generator
        )
        self.recorded.append(out.detach().cpu().clone())
        return out


def _fixed_obs_stream(policy, n, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield StateObservation(
            state=rng.uniform(-1, 1, policy.meta.state_dim).astype(np.float32),
            previous_action=np.zeros(policy.meta.action_dim, np.float32),
            step_index=i,
        )


def _run(system, policy_rec, n_steps, obs_seed=0):
    system.reset(episode_seed=123)
    for obs in _fixed_obs_stream(policy_rec, n_steps, seed=obs_seed):
        system.act(obs)


def test_control_and_actsemble_see_identical_candidates(trained_checkpoints):
    policy_a = RecordingPolicy(
        DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    )
    policy_b = RecordingPolicy(
        DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    )
    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    control = MultiSampleControlSystem(
        policy_a, num_candidates=6, selection_rule="uniform_random", candidate_root_seed=99
    )
    actsemble = CandidateRerankingActsemble(
        policy_b, comp, num_candidates=6, candidate_root_seed=99
    )
    n = control.action_horizon * 2 + 1  # forces 3 replans
    _run(control, policy_a, n)
    _run(actsemble, policy_b, n)
    assert len(policy_a.recorded) == len(policy_b.recorded) == 3
    for ca, cb in zip(policy_a.recorded, policy_b.recorded):
        assert torch.equal(ca, cb)
    # ... and both used the same frozen checkpoint.
    assert policy_a.checkpoint_hash == policy_b.checkpoint_hash


def test_selection_rules_differ_only_in_index(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    actsemble = CandidateRerankingActsemble(policy, comp, num_candidates=6, candidate_root_seed=1)
    actsemble.reset(episode_seed=7)
    obs = next(_fixed_obs_stream(policy, 1))
    actsemble.act(obs)
    rec = actsemble.diagnostics()["replans"][0]
    scores = rec["component_scores"]
    assert len(scores) == 6
    assert rec["selected_index"] == int(np.argmax(scores))


def test_component_exception_falls_back_to_candidate_zero(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")

    class ExplodingComponent:
        dataset_hash = policy.dataset_hash
        checkpoint_hash = "x"
        meta = {}

        def reset(self):
            pass

        def score(self, obs, chunks):
            raise RuntimeError("boom")

    system = CandidateRerankingActsemble(
        policy, ExplodingComponent(), num_candidates=4, candidate_root_seed=0
    )
    system.reset(episode_seed=1)
    action = system.act(next(_fixed_obs_stream(policy, 1)))
    assert np.isfinite(action.value).all()
    diag = system.diagnostics()
    assert diag["fallback_count"] == 1
    assert diag["replans"][0]["selected_index"] == 0
    assert "component_exception" in diag["replans"][0]["fallback_reason"]


def test_nonfinite_scores_fall_back_to_candidate_zero(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")

    class NaNComponent:
        dataset_hash = policy.dataset_hash
        checkpoint_hash = "x"
        meta = {}

        def reset(self):
            pass

        def score(self, obs, chunks):
            return torch.full((chunks.shape[0],), float("nan"))

    system = CandidateRerankingActsemble(
        policy, NaNComponent(), num_candidates=4, candidate_root_seed=0
    )
    system.reset(episode_seed=1)
    system.act(next(_fixed_obs_stream(policy, 1)))
    rec = system.diagnostics()["replans"][0]
    assert rec["fallback"] and rec["fallback_reason"] == "no_valid_scores"
    assert rec["selected_index"] == 0


def test_history_reset_between_episodes(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    system = MultiSampleControlSystem(policy, num_candidates=2, candidate_root_seed=0)
    system.reset(episode_seed=1)
    for obs in _fixed_obs_stream(policy, 3, seed=10):
        system.act(obs)
    assert len(system._history) > 0 and len(system._queue) > 0
    system.reset(episode_seed=2)
    assert len(system._history) == 0 and len(system._queue) == 0
    assert system.diagnostics()["num_replans"] == 0


def test_factory_builds_and_validates(trained_checkpoints):
    policy = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    standalone = build_system(
        {"policy": {"num_candidates": 1}, "selection": {"type": "candidate_zero"}}, policy, []
    )
    assert standalone.name == "standalone_diffusion"
    with pytest.raises(ValueError, match="frozen"):
        build_system(
            {"policy": {"num_candidates": 1, "frozen": False},
             "selection": {"type": "candidate_zero"}},
            policy,
            [],
        )
    with pytest.raises(ValueError, match="exactly one component"):
        build_system(
            {"policy": {"num_candidates": 4},
             "selection": {"type": "highest_component_score"}},
            policy,
            [],
        )
    actsemble = build_system(
        {"policy": {"num_candidates": 4}, "selection": {"type": "highest_component_score"}},
        policy,
        [comp],
    )
    assert actsemble.name == "candidate_reranking_actsemble"


def test_candidate_hashes_identical_across_all_three_paired_systems(trained_checkpoints):
    """Protocol §11: with the same frozen checkpoint and same K, standalone
    (paired mode), control, and Actsemble record identical per-replan
    candidate-tensor hashes; standalone/control select candidate zero."""
    from actsemble.components.action_chunk_compatibility import ActionChunkCompatibility

    def load():
        return DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")

    comp = ActionChunkCompatibility.from_checkpoint(
        trained_checkpoints["component_best"], device="cpu"
    )
    k = 4
    systems = {
        "standalone": build_system(
            {"policy": {"num_candidates": k}, "selection": {"type": "candidate_zero"}},
            load(), [],
        ),
        "control": build_system(
            {"policy": {"num_candidates": k}, "selection": {"type": "first_candidate"}},
            load(), [],
        ),
        "actsemble": build_system(
            {"policy": {"num_candidates": k},
             "selection": {"type": "highest_component_score"}},
            load(), [comp],
        ),
    }
    hashes = {}
    for name, system in systems.items():
        system.candidate_root_seed = 12345  # episode policy-sampling seed
        system.reset(episode_seed=7)
        for obs in _fixed_obs_stream(system.policy, system.action_horizon + 1, seed=3):
            system.act(obs)
        diag = system.diagnostics()
        hashes[name] = diag["candidate_hashes"]
        assert len(hashes[name]) == 2 and all(h for h in hashes[name])
        if name in ("standalone", "control"):
            assert diag["selected_indices"] == [0, 0]
            assert diag["selection_change_rate"] == 0.0
    assert hashes["standalone"] == hashes["control"] == hashes["actsemble"]


def test_candidate_seed_depends_on_checkpoint_hash(trained_checkpoints):
    """§11: the candidate generator folds the policy checkpoint hash, so a
    different frozen checkpoint yields different candidate tensors under
    the same policy-sampling seed."""
    best = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_best"], device="cpu")
    final = DiffusionPolicy.from_checkpoint(trained_checkpoints["policy_final"], device="cpu")
    obs = next(_fixed_obs_stream(best, 1))

    def first_hash(policy):
        system = MultiSampleControlSystem(
            policy, num_candidates=3, selection_rule="first_candidate"
        )
        system.candidate_root_seed = 999
        system.reset(episode_seed=1)
        system.act(obs)
        return system.diagnostics()["candidate_hashes"][0]

    assert first_hash(best) != first_hash(final)

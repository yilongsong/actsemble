"""Training subsystem: resuming to a larger ``max_steps`` must be
identical to a single run to that budget.

The trainer persists all trajectory-driving RNG (diffusion noise / timestep
generators, the global torch RNG, and the dataloader's mid-epoch position)
into ``last.pt``; resuming restores them, so 0->A then A->B reproduces the
same weights as 0->B. Run on CPU, where training is bitwise deterministic
(GPU carries known ULP-level batch-position nondeterminism).
"""

from __future__ import annotations

import copy

import torch

from actsemble.training.train_diffusion_policy import train_diffusion_policy
from actsemble.training.train_diffusion_policy import resolve_total_steps
from actsemble.training.train_component import train_component


def _final_states(run_dir):
    ckpt = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    return ckpt["model_state"], ckpt["ema_state"]


def _assert_states_equal(a, b, what):
    assert set(a) == set(b), f"{what}: different keys"
    for k in a:
        assert torch.equal(a[k], b[k]), f"{what}: tensor '{k}' differs"


def _train(dataset_path, cfg, out, steps, resume=False):
    return train_diffusion_policy(
        policy_cfg=copy.deepcopy(cfg),
        dataset_path=dataset_path,
        output_dir=out,
        max_steps=steps,
        device="cpu",
        resume=resume,
    )


def test_resume_extension_matches_single_shot(tmp_path, dataset_path, tiny_policy_cfg):
    cfg = copy.deepcopy(tiny_policy_cfg)
    cfg["training"]["eval_every"] = 5  # last.pt (with rng_state) written at step A
    A, B = 5, 13

    single = tmp_path / "single"
    _train(dataset_path, cfg, single, B)

    two = tmp_path / "two_stage"
    _train(dataset_path, cfg, two, A)
    _train(dataset_path, cfg, two, B, resume=True)

    ms, es = _final_states(single)
    mt, et = _final_states(two)
    _assert_states_equal(ms, mt, "model_state 0->B vs 0->A->B")
    _assert_states_equal(es, et, "ema_state 0->B vs 0->A->B")


def test_resume_requires_rng_state(tmp_path, dataset_path, tiny_policy_cfg):
    """Resuming a checkpoint that lacks rng_state must fail loudly rather than
    silently produce a non-equivalent continuation (no backward-compat path)."""
    import pytest

    cfg = copy.deepcopy(tiny_policy_cfg)
    cfg["training"]["eval_every"] = 5
    run = tmp_path / "legacy"
    _train(dataset_path, cfg, run, 5)

    # Simulate a pre-extension checkpoint by stripping rng_state.
    ckpt = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    ckpt["train_state"].pop("rng_state", None)
    torch.save(ckpt, run / "last.pt")

    with pytest.raises(ValueError, match="rng_state"):
        _train(dataset_path, cfg, run, 8, resume=True)


def test_resume_extension_is_deterministic(tmp_path, dataset_path, tiny_policy_cfg):
    cfg = copy.deepcopy(tiny_policy_cfg)
    cfg["training"]["eval_every"] = 5
    A, B = 5, 11

    base = tmp_path / "base"
    _train(dataset_path, cfg, base, A)

    # two independent resumes from the same last.pt must agree bitwise
    import shutil

    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    shutil.copytree(base, r1)
    shutil.copytree(base, r2)
    _train(dataset_path, cfg, r1, B, resume=True)
    _train(dataset_path, cfg, r2, B, resume=True)

    m1, e1 = _final_states(r1)
    m2, e2 = _final_states(r2)
    _assert_states_equal(m1, m2, "model_state resume determinism")
    _assert_states_equal(e1, e2, "ema_state resume determinism")


def test_cosine_resume_rejects_changed_planned_budget(
    tmp_path, dataset_path, tiny_policy_cfg
):
    import pytest

    cfg = copy.deepcopy(tiny_policy_cfg)
    cfg["training"].update(lr_scheduler="cosine", lr_warmup_steps=1, eval_every=5)
    run = tmp_path / "cosine"
    _train(dataset_path, cfg, run, 5)
    with pytest.raises(ValueError, match="same total_steps"):
        _train(dataset_path, cfg, run, 10, resume=True)


def test_component_resume_matches_single_shot(
    tmp_path, dataset_path, tiny_component_cfg
):
    cfg = copy.deepcopy(tiny_component_cfg)
    cfg["training"].update(eval_every=4, checkpoint_every=4)

    single = tmp_path / "component_single"
    train_component(
        component_cfg=copy.deepcopy(cfg),
        dataset_path=dataset_path,
        output_dir=single,
        max_steps=9,
        device="cpu",
    )
    staged = tmp_path / "component_staged"
    train_component(
        component_cfg=copy.deepcopy(cfg),
        dataset_path=dataset_path,
        output_dir=staged,
        max_steps=4,
        device="cpu",
    )
    train_component(
        component_cfg=copy.deepcopy(cfg),
        dataset_path=dataset_path,
        output_dir=staged,
        max_steps=9,
        device="cpu",
        resume=True,
    )
    a = torch.load(single / "final.pt", map_location="cpu", weights_only=False)
    b = torch.load(staged / "final.pt", map_location="cpu", weights_only=False)
    _assert_states_equal(a["model_state"], b["model_state"], "component resume")


def test_resolve_total_steps_precedence_and_epoch_conversion():
    assert resolve_total_steps({"max_epochs": 10}, 1000, 100) == 100
    assert resolve_total_steps({"max_epochs": 10}, 2000, 100) == 200
    assert resolve_total_steps({"max_epochs": 10, "max_steps": 500}, 1000, 100) == 500
    assert (
        resolve_total_steps({"max_epochs": 10, "max_steps": 500}, 1000, 100, 42) == 42
    )
    assert resolve_total_steps({}, 1000, 100) == 10000

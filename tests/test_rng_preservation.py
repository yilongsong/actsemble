"""§3: mid-training evaluation must not affect training.

Verifies (a) the RNG-state guard restores every stream exactly, and
(b) end-to-end: training with a randomness-consuming checkpoint callback
(a stand-in for policy screening) at ANY frequency produces bitwise the
same final weights as training with screening disabled.
"""

import copy
import random

import numpy as np
import torch

from actsemble.data.writer import write_dataset
from actsemble.training.train_diffusion_policy import train_diffusion_policy
from actsemble.utils.rng_state import preserve_rng_states

from conftest import TINY_POLICY_CFG, make_episodes, make_metadata


def test_guard_restores_all_streams():
    # Record the draws each stream WOULD produce, then re-seed, consume
    # heavily inside the guard, and check the recorded draws still come out.
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    gen = torch.Generator().manual_seed(4)
    expected = (
        random.random(),
        np.random.rand(),
        torch.randn(5),
        torch.randn(5, generator=gen),
    )
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    gen.manual_seed(4)
    with preserve_rng_states({"g": gen}):
        [random.random() for _ in range(100)]
        np.random.rand(1000)
        torch.randn(1000)
        torch.randn(100, generator=gen)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.randn(5), expected[2])
    assert torch.equal(torch.randn(5, generator=gen), expected[3])


def _train(tmp_path, name, *, checkpoint_every, on_checkpoint, steps=8):
    cfg = copy.deepcopy(TINY_POLICY_CFG)
    cfg["training"]["max_steps"] = steps
    cfg["training"]["eval_every"] = steps
    if checkpoint_every is not None:
        cfg["training"]["checkpoint_every"] = checkpoint_every
    dataset = tmp_path / "ds.h5"
    if not dataset.exists():
        write_dataset(dataset, make_episodes(4, T=30), make_metadata())
    return train_diffusion_policy(
        policy_cfg=cfg,
        dataset_path=dataset,
        output_dir=tmp_path / name,
        device="cpu",
        on_checkpoint=on_checkpoint,
    )


def _noisy_callback(**kwargs):
    """Consumes randomness from every stream, like an in-process screener."""
    [random.random() for _ in range(50)]
    np.random.rand(200)
    torch.randn(500)


def _final_weights(summary):
    ckpt = torch.load(summary["checkpoints"]["final"], map_location="cpu", weights_only=False)
    return ckpt["model_state"], ckpt["ema_state"]


def test_screening_frequency_does_not_change_training(tmp_path):
    base = _train(tmp_path, "no_screening", checkpoint_every=None, on_checkpoint=None)
    every2 = _train(tmp_path, "screen_every_2", checkpoint_every=2, on_checkpoint=_noisy_callback)
    every5 = _train(tmp_path, "screen_every_5", checkpoint_every=5, on_checkpoint=_noisy_callback)

    for other in (every2, every5):
        for a, b in zip(_final_weights(base), _final_weights(other)):
            assert set(a.keys()) == set(b.keys())
            for key in a:
                assert torch.equal(a[key], b[key]), f"weights diverged at {key}"
    # identical loss trajectories too
    assert base["final_train_loss"] == every2["final_train_loss"] == every5["final_train_loss"]


def test_snapshots_saved_at_intervals_and_final_step(tmp_path):
    out = _train(tmp_path, "snaps", checkpoint_every=3, on_checkpoint=None, steps=8)
    steps = sorted(int(p.split("step_")[1][:6]) for p in out["snapshots"])
    assert steps == [3, 6, 8]  # intervals plus the final training step

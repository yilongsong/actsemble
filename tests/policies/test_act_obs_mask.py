"""ACT hard observation mask (ablation/falsifier runs, e.g. qvel-blind).

The mask must hold at BOTH model entry points (CVAE style encoder and decoder),
survive a checkpoint round-trip via the config (so inference is masked whenever
training was), and stay out of the state_dict (so unmasked checkpoints load
into masked configs and vice versa).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from actsemble.data.normalization import NormalizationStats, Normalizer
from actsemble.policies.act.model import ACTModel
from actsemble.policies.act.policy import ACTPolicy, build_act_model
from actsemble.policies.meta import PolicyMeta

SD, AD, HO, HP = 5, 3, 2, 4
MASK = [[1, 3]]  # dims 1 and 2 masked


def _model(mask=None):
    return ACTModel(
        obs_feature_dim=SD,
        action_dim=AD,
        obs_horizon=HO,
        prediction_horizon=HP,
        hidden_dim=32,
        latent_dim=8,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        obs_mask_ranges=mask,
    )


def _perturbed(obs, dims):
    out = obs.clone()
    out[..., dims] += torch.randn_like(out[..., dims]) * 10.0
    return out


def test_masked_dims_ignored_at_both_entry_points():
    torch.manual_seed(0)
    m = _model(mask=MASK).eval()
    obs = torch.randn(4, HO, SD)
    actions = torch.randn(4, HP, AD)
    z = torch.zeros(4, m.latent_dim)
    obs2 = _perturbed(obs, [1, 2])
    # decoder path
    assert torch.equal(m.decode(obs, z), m.decode(obs2, z))
    # style-encoder path
    mu1, lv1 = m.encode_style(obs, actions)
    mu2, lv2 = m.encode_style(obs2, actions)
    assert torch.equal(mu1, mu2) and torch.equal(lv1, lv2)


def test_unmasked_dims_still_matter():
    torch.manual_seed(0)
    m = _model(mask=MASK).eval()
    obs = torch.randn(4, HO, SD)
    z = torch.zeros(4, m.latent_dim)
    assert not torch.allclose(
        m.decode(obs, z), m.decode(_perturbed(obs, [0]), z), atol=1e-4
    )


def test_mask_absent_from_state_dict_both_directions():
    masked, plain = _model(mask=MASK), _model()
    assert "obs_feature_mask" not in masked.state_dict()
    # old (unmasked) weights load into a masked-config model and vice versa
    masked.load_state_dict(plain.state_dict())
    plain.load_state_dict(masked.state_dict())
    assert plain.obs_feature_mask is None
    assert masked.obs_mask_ranges == [(1, 3)]


def test_mask_range_validation():
    for bad in ([[3, 3]], [[-1, 2]], [[4, SD + 1]]):
        with pytest.raises(ValueError, match="obs_mask_ranges"):
            _model(mask=bad)


def _meta():
    stats = NormalizationStats(
        state_min=-np.ones(SD, np.float32),
        state_max=np.ones(SD, np.float32),
        action_min=-np.ones(AD, np.float32),
        action_max=np.ones(AD, np.float32),
    )
    return PolicyMeta(
        dataset_hash="d",
        split_hash="s",
        normalization=stats.to_dict(),
        task_id="PushT-v1",
        controller="c",
        simulation_backend="b",
        state_dim=SD,
        action_dim=AD,
        action_low=[-1.0] * AD,
        action_high=[1.0] * AD,
        obs_horizon=HO,
        prediction_horizon=HP,
        action_horizon=2,
    ), Normalizer(stats)


def test_build_act_model_reads_observation_mask():
    meta, _ = _meta()
    cfg = {"model": {"hidden_dim": 32}, "observation": {"mask_feature_ranges": MASK}}
    assert build_act_model(cfg, meta).obs_mask_ranges == [(1, 3)]
    assert build_act_model({"model": {}}, meta).obs_feature_mask is None


def test_checkpoint_roundtrip_masks_raw_inference(tmp_path):
    """End to end: a checkpoint whose config carries the mask yields a policy
    whose sample_action_chunks is invariant to RAW masked-dim perturbations."""
    torch.manual_seed(0)
    meta, _ = _meta()
    config = {
        "model": {
            "hidden_dim": 32,
            "latent_dim": 8,
            "n_heads": 2,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
        "observation": {"mask_feature_ranges": MASK},
    }
    model = build_act_model(config, meta)
    path = tmp_path / "act_masked.pt"
    ACTPolicy.save_checkpoint(
        path,
        config=config,
        meta=meta,
        model_state=model.state_dict(),
        ema_state=None,
    )
    policy = ACTPolicy.from_checkpoint(path, device="cpu", use_ema=True)
    assert policy.model.obs_mask_ranges == [(1, 3)]
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, (HO, SD)).astype(np.float32)
    obs2 = obs.copy()
    obs2[:, 1:3] = rng.uniform(-1, 1, (HO, 2)).astype(np.float32)
    a = policy.sample_action_chunks(obs, num_samples=1, generator=None)
    b = policy.sample_action_chunks(obs2, num_samples=1, generator=None)
    assert torch.equal(a, b)
    obs3 = obs.copy()
    obs3[:, 0] += 0.5  # unmasked dim must still matter
    c = policy.sample_action_chunks(obs3, num_samples=1, generator=None)
    assert not torch.allclose(a, c, atol=1e-5)

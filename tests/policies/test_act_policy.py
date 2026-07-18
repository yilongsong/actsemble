"""Policy subsystem: ACT tests (CPU, tiny model, no dataset/GPU).

Covers the model shapes + CVAE math, the z=0 deterministic inference contract,
ActionChunkPolicy compliance (so ACT drops into every system), checkpoint
round-trip, a tiny overfit sanity check on the training loss, and that the
temporal-ensemble system runs over an ACT policy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from actsemble.data.normalization import STANDARDIZE, NormalizationStats, Normalizer
from actsemble.data.torch_dataset import ACTEpisodeDataset
from actsemble.policies.act.model import ACTModel
from actsemble.policies.act.policy import ACTPolicy, build_act_model
from actsemble.policies.interface import ActionChunkPolicy
from actsemble.policies.meta import PolicyMeta
from actsemble.systems.temporal_ensemble import TemporalEnsembleSystem
from actsemble.types import StateObservation

SD, AD, HO, HP = 5, 3, 2, 4  # state dim, action dim, obs horizon, pred horizon


def _model(**kw):
    return ACTModel(
        obs_feature_dim=SD,
        action_dim=AD,
        obs_horizon=HO,
        prediction_horizon=HP,
        hidden_dim=kw.get("hidden_dim", 32),
        latent_dim=kw.get("latent_dim", 8),
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dim_feedforward=32,
        dropout=kw.get("dropout", 0.0),
    )


def _meta():
    stats = NormalizationStats(
        state_min=-np.ones(SD, np.float32),
        state_max=np.ones(SD, np.float32),
        action_min=-np.ones(AD, np.float32),
        action_max=np.ones(AD, np.float32),
    )
    meta = PolicyMeta(
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
    )
    return meta, Normalizer(stats)


def _policy(model=None):
    meta, norm = _meta()
    return ACTPolicy(
        model or _model(), norm, meta, config={"model": {}}, device=torch.device("cpu")
    )


# ---- model shapes + CVAE math ---------------------------------------------
def test_forward_and_decode_shapes():
    m = _model().eval()
    obs = torch.randn(4, HO, SD)
    actions = torch.randn(4, HP, AD)
    pred, mu, logvar = m(obs, actions)
    assert pred.shape == (4, HP, AD)
    assert mu.shape == logvar.shape == (4, m.latent_dim)
    out = m.decode(obs, torch.zeros(4, m.latent_dim))
    assert out.shape == (4, HP, AD)


def test_kl_zero_at_standard_normal():
    mu = torch.zeros(3, 8)
    logvar = torch.zeros(3, 8)  # sigma^2 = 1 -> KL to N(0,I) is 0
    assert torch.allclose(
        ACTModel.kl_divergence(mu, logvar), torch.tensor(0.0), atol=1e-6
    )
    # a non-unit posterior has strictly positive KL
    assert ACTModel.kl_divergence(torch.ones(3, 8), torch.zeros(3, 8)).item() > 0


def test_decode_is_deterministic_in_eval():
    m = _model(dropout=0.5).eval()  # eval() must disable dropout
    obs = torch.randn(2, HO, SD)
    z = torch.zeros(2, m.latent_dim)
    assert torch.equal(m.decode(obs, z), m.decode(obs, z))


def test_action_mask_makes_style_encoder_ignore_padding():
    torch.manual_seed(0)
    m = _model().eval()
    obs = torch.randn(4, HO, SD)
    actions = torch.randn(4, HP, AD)
    mask = torch.ones(4, HP, dtype=torch.bool)
    mask[:, 2:] = False  # positions 2.. are replicated padding
    mu1, lv1 = m.encode_style(obs, actions, mask)
    garbage = actions.clone()
    garbage[:, 2:] += 100.0  # corrupt only the padded positions
    mu2, lv2 = m.encode_style(obs, garbage, mask)
    assert torch.allclose(mu1, mu2, atol=1e-5) and torch.allclose(lv1, lv2, atol=1e-5)
    # without the mask, the padded garbage leaks into the posterior
    mu3, _ = m.encode_style(obs, garbage, None)
    assert not torch.allclose(mu1, mu3, atol=1e-3)


# ---- ActionChunkPolicy contract -------------------------------------------
def test_is_action_chunk_policy():
    assert isinstance(_policy(), ActionChunkPolicy)


def test_sample_returns_identical_clipped_candidates():
    policy = _policy()
    obs_hist = np.random.randn(HO, SD).astype(np.float32)
    chunks = policy.sample_action_chunks(obs_hist, num_samples=5, generator=None)
    assert chunks.shape == (5, HP, AD)
    assert torch.isfinite(chunks).all()
    # z=0 deterministic -> every candidate identical
    for k in range(1, 5):
        assert torch.equal(chunks[0], chunks[k])
    # clipped to action bounds [-1, 1]
    assert (chunks >= -1.0 - 1e-6).all() and (chunks <= 1.0 + 1e-6).all()


def test_sample_rejects_wrong_obs_shape():
    policy = _policy()
    with pytest.raises(ValueError, match="observation_history shape"):
        policy.sample_action_chunks(
            np.zeros((HO, SD + 1), np.float32), num_samples=1, generator=None
        )


def test_determinism_across_calls():
    policy = _policy()
    obs_hist = np.random.randn(HO, SD).astype(np.float32)
    a = policy.sample_action_chunks(obs_hist, num_samples=1, generator=None)
    b = policy.sample_action_chunks(obs_hist, num_samples=1, generator=None)
    assert torch.equal(a, b)


# ---- checkpoint round-trip -------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    meta, _ = _meta()
    model = _model()
    path = tmp_path / "act.pt"
    ACTPolicy.save_checkpoint(
        path,
        config={
            "model": {
                "hidden_dim": 32,
                "latent_dim": 8,
                "n_heads": 2,
                "n_encoder_layers": 1,
                "n_decoder_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
            }
        },
        meta=meta,
        model_state=model.state_dict(),
        ema_state=model.state_dict(),
    )
    loaded = ACTPolicy.from_checkpoint(path, device="cpu", use_ema=True)
    assert loaded.checkpoint_hash  # sha256 of the file
    obs_hist = np.random.randn(HO, SD).astype(np.float32)
    # reference policy built from the same weights
    ref = ACTPolicy(
        model,
        Normalizer(NormalizationStats.from_dict(meta.normalization)),
        meta,
        config={},
        device=torch.device("cpu"),
    )
    assert torch.allclose(
        loaded.sample_action_chunks(obs_hist, num_samples=1, generator=None),
        ref.sample_action_chunks(obs_hist, num_samples=1, generator=None),
        atol=1e-6,
    )


def test_from_checkpoint_rejects_wrong_kind(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"kind": "actsemble_diffusion_policy"}, path)
    with pytest.raises(ValueError, match="not an Actsemble ACT-policy checkpoint"):
        ACTPolicy.from_checkpoint(path)


def test_build_act_model_reads_config():
    meta, _ = _meta()
    m = build_act_model(
        {"model": {"hidden_dim": 64, "latent_dim": 16, "n_encoder_layers": 2}}, meta
    )
    assert m.hidden_dim == 64 and m.latent_dim == 16
    assert m.obs_feature_dim == SD and m.action_dim == AD and m.prediction_horizon == HP


def test_canonical_sinusoidal_pos_embedding_is_fixed_and_obs_horizon_1():
    m = ACTModel(
        obs_feature_dim=SD,
        action_dim=AD,
        obs_horizon=1,
        prediction_horizon=HP,
        hidden_dim=32,
        latent_dim=8,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        pos_embedding="sinusoidal",
    )
    params = {n for n, _ in m.named_parameters()}
    buffers = {n for n, _ in m.named_buffers()}
    assert "style_pos" in buffers and "style_pos" not in params  # fixed, not learned
    m.eval()
    obs = torch.randn(2, 1, SD)  # obs_horizon 1 (current observation only)
    pred, mu, logvar = m(obs, torch.randn(2, HP, AD))
    assert pred.shape == (2, HP, AD) and mu.shape == (2, 8)


# ---- training loss wiring (tiny overfit) ----------------------------------
def test_overfit_one_batch_reduces_l1():
    torch.manual_seed(0)
    m = _model().train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    obs = torch.randn(8, HO, SD)
    actions = torch.randn(8, HP, AD)
    gen = torch.Generator().manual_seed(0)
    first = last = None
    for i in range(120):
        pred, mu, logvar = m(obs, actions, generator=gen)
        l1 = torch.nn.functional.l1_loss(pred, actions)
        loss = l1 + 0.01 * ACTModel.kl_divergence(mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = l1.item()
        last = l1.item()
    assert last < 0.5 * first  # the decoder learns to reconstruct the batch


# ---- integration: temporal ensemble over an ACT policy --------------------
def test_temporal_ensemble_runs_over_act():
    policy = _policy()
    system = TemporalEnsembleSystem(policy, aggregation="mean", num_candidates=1)
    system.candidate_root_seed = 3
    system.reset(episode_seed=1)
    rng = np.random.default_rng(0)
    for i in range(6):
        obs = StateObservation(
            state=rng.uniform(-1, 1, SD).astype(np.float32),
            previous_action=np.zeros(AD, np.float32),
            step_index=i,
        )
        action = system.act(obs)
        assert np.isfinite(action.value).all()
    d = system.diagnostics()
    assert d["num_control_steps"] == 6 and d["num_replans"] == 6


# ---- canonical DETR architecture and episode-weighted sampling ------------
def test_act_arch_validates_and_detr_is_deterministic():
    with pytest.raises(ValueError, match="arch"):
        ACTModel(
            obs_feature_dim=6,
            action_dim=2,
            obs_horizon=1,
            prediction_horizon=8,
            arch="bad",
        )
    model = ACTModel(
        obs_feature_dim=6,
        action_dim=2,
        obs_horizon=1,
        prediction_horizon=8,
        hidden_dim=32,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=64,
        pos_embedding="sinusoidal",
        arch="detr",
    ).eval()
    observations, latent = torch.randn(2, 1, 6), torch.zeros(2, model.latent_dim)
    with torch.no_grad():
        first, second = (
            model.decode(observations, latent),
            model.decode(observations, latent),
        )
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (2, 8, 2)


def test_detr_act_has_separate_projection_and_final_norm():
    kwargs = {
        "obs_feature_dim": 6,
        "action_dim": 2,
        "obs_horizon": 1,
        "prediction_horizon": 8,
        "hidden_dim": 32,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "dim_feedforward": 64,
    }
    detr = ACTModel(pos_embedding="sinusoidal", arch="detr", **kwargs)
    assert detr.obs_proj_dec is not detr.obs_proj
    assert hasattr(detr, "decoder_norm")
    torch_builtin = ACTModel(arch="torch_builtin", **kwargs)
    assert not hasattr(torch_builtin, "obs_proj_dec")
    assert not hasattr(torch_builtin, "decoder_norm")


class _Episode:
    def __init__(self, length, action_dim=2, state_dim=3):
        self.state = np.arange(length * state_dim, dtype=np.float32).reshape(
            length, state_dim
        )
        self.next_state = self.state
        self.previous_action = np.zeros((length, action_dim), np.float32)
        self.action = np.arange(length * action_dim, dtype=np.float32).reshape(
            length, action_dim
        )
        self.episode_id = "e"

    def __len__(self):
        return len(self.state)


def _standard_normalizer():
    return Normalizer(
        NormalizationStats(
            method=STANDARDIZE,
            state_mean=np.zeros(3, np.float32),
            state_std=np.ones(3, np.float32),
            action_mean=np.zeros(2, np.float32),
            action_std=np.ones(2, np.float32),
        )
    )


def test_act_episode_dataset_fixed_is_deterministic_across_accesses():
    dataset = ACTEpisodeDataset(
        [_Episode(12), _Episode(20)],
        _standard_normalizer(),
        obs_horizon=1,
        prediction_horizon=8,
        start_seed=3,
        fixed=True,
    )
    assert [dataset[i]["t"].item() for i in range(2)] == [
        dataset[i]["t"].item() for i in range(2)
    ]


def test_act_episode_dataset_len_and_reproducible():
    episodes = [_Episode(12), _Episode(20), _Episode(8)]
    kwargs = {"obs_horizon": 1, "prediction_horizon": 8, "start_seed": 7}
    first = ACTEpisodeDataset(episodes, _standard_normalizer(), **kwargs)
    second = ACTEpisodeDataset(episodes, _standard_normalizer(), **kwargs)
    assert len(first) == 3
    assert [first[i]["t"].item() for i in range(3)] == [
        second[i]["t"].item() for i in range(3)
    ]
    assert first[0]["action_chunk"].shape == (8, 2)

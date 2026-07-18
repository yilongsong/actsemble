"""ACT policy: normalization, z=0 deterministic decoding, checkpoints.

Implements the same ``ActionChunkPolicy`` protocol as ``DiffusionPolicy`` so it
drops into every system (standalone, temporal ensemble, …) unchanged. Inference
follows the canonical ACT scheme — the CVAE style encoder is discarded and the
latent ``z`` is pinned to 0 (the prior mean) — so ``sample_action_chunks`` is a
deterministic function of the observation: all ``num_samples`` candidates are
identical (ACT is single-mode at inference, so K>1 adds no diversity; the K axis
exists only for interface compatibility). Checkpoints mirror the diffusion
policy's format with ``kind = "actsemble_act_policy"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...data.normalization import NormalizationStats, Normalizer
from ...utils.hashing import hash_file
from ..meta import PolicyMeta
from .model import ACTModel


def obs_feature_dim(meta: PolicyMeta) -> int:
    return meta.state_dim + (meta.action_dim if meta.include_previous_action else 0)


def build_act_model(policy_cfg: dict, meta: PolicyMeta) -> ACTModel:
    m = policy_cfg.get("model", {})
    return ACTModel(
        obs_feature_dim=obs_feature_dim(meta),
        action_dim=meta.action_dim,
        obs_horizon=meta.obs_horizon,
        prediction_horizon=meta.prediction_horizon,
        hidden_dim=int(m.get("hidden_dim", 256)),
        latent_dim=int(m.get("latent_dim", 32)),
        n_heads=int(m.get("n_heads", 8)),
        n_encoder_layers=int(m.get("n_encoder_layers", 4)),
        n_decoder_layers=int(m.get("n_decoder_layers", 4)),
        dim_feedforward=int(m.get("dim_feedforward", 512)),
        dropout=float(m.get("dropout", 0.1)),
    )


class ACTPolicy:
    """Frozen inference-time ACT policy (implements ActionChunkPolicy)."""

    def __init__(
        self,
        model: ACTModel,
        normalizer: Normalizer,
        meta: PolicyMeta,
        *,
        config: dict,
        device: torch.device,
        checkpoint_hash: str = "",
        checkpoint_path: str = "",
        weights_kind: str = "ema",
    ):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.meta = meta
        self.config = config
        self.device = device
        self._checkpoint_hash = checkpoint_hash
        self.checkpoint_path = checkpoint_path
        self.weights_kind = weights_kind
        self._action_low = np.asarray(meta.action_low, dtype=np.float32)
        self._action_high = np.asarray(meta.action_high, dtype=np.float32)

    # -- ActionChunkPolicy protocol -----------------------------------------
    @property
    def dataset_hash(self) -> str:
        return self.meta.dataset_hash

    @property
    def checkpoint_hash(self) -> str:
        return self._checkpoint_hash

    def reset(self) -> None:
        pass  # stateless between episodes

    @torch.no_grad()
    def sample_action_chunks(
        self,
        observation_history: np.ndarray,
        *,
        num_samples: int,
        generator: torch.Generator | None = None,  # unused: z=0 deterministic
    ) -> torch.Tensor:
        """[obs_horizon, obs_feature_dim] raw obs -> [K, H_p, A] raw actions."""
        expected_feat = obs_feature_dim(self.meta)
        if observation_history.shape != (self.meta.obs_horizon, expected_feat):
            raise ValueError(
                f"observation_history shape {observation_history.shape} != "
                f"({self.meta.obs_horizon}, {expected_feat})"
            )
        cond = self._normalize_history(observation_history)
        obs = torch.from_numpy(cond).to(self.device).unsqueeze(0)  # [1, H_o, feat]
        z = torch.zeros(1, self.model.latent_dim, device=self.device)  # ACT inference: z=0
        chunk_norm = self.model.decode(obs, z)  # [1, H_p, A] normalized
        chunk = self.normalizer.unnormalize_action(chunk_norm)
        low = torch.as_tensor(self._action_low, device=chunk.device)
        high = torch.as_tensor(self._action_high, device=chunk.device)
        chunk = chunk.clamp(low, high)  # [1, H_p, A]
        return chunk.expand(int(num_samples), -1, -1).contiguous()

    def _normalize_history(self, observation_history: np.ndarray) -> np.ndarray:
        obs = observation_history.astype(np.float32)
        sd = self.meta.state_dim
        state_part = self.normalizer.normalize_state(obs[:, :sd])
        if self.meta.include_previous_action:
            act_part = self.normalizer.normalize_action(obs[:, sd:])
            return np.concatenate([state_part, act_part], axis=1).astype(np.float32)
        return np.asarray(state_part, dtype=np.float32)

    def new_generator(self, seed: int) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        return gen

    # -- persistence ---------------------------------------------------------
    @staticmethod
    def save_checkpoint(
        path: str | Path,
        *,
        config: dict,
        meta: PolicyMeta,
        model_state: dict,
        ema_state: dict | None,
        train_state: dict | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "actsemble_act_policy",
                "config": config,
                "meta": meta.to_dict(),
                "model_state": model_state,
                "ema_state": ema_state,
                "train_state": train_state,
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
        use_ema: bool = True,
    ) -> "ACTPolicy":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("kind") != "actsemble_act_policy":
            raise ValueError(f"{path} is not an Actsemble ACT-policy checkpoint")
        config = ckpt["config"]
        meta = PolicyMeta.from_dict(ckpt["meta"])
        model = build_act_model(config, meta)
        if use_ema:
            if ckpt.get("ema_state") is None:
                raise ValueError(f"{path} has no EMA weights but use_ema=True")
            model.load_state_dict(ckpt["ema_state"])
            weights_kind = "ema"
        else:
            model.load_state_dict(ckpt["model_state"])
            weights_kind = "raw"
        normalizer = Normalizer(NormalizationStats.from_dict(meta.normalization))
        return cls(
            model,
            normalizer,
            meta,
            config=config,
            device=torch.device(device),
            checkpoint_hash=hash_file(path),
            checkpoint_path=str(path),
            weights_kind=weights_kind,
        )

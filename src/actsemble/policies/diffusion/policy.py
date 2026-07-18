"""State-conditioned diffusion policy: normalization, sampling, checkpoints.

The checkpoint stores everything needed to reconstruct the policy plus the
fairness-safeguard metadata (dataset/split/normalization hashes, task,
controller, simulation backend). ``checkpoint_hash`` is the SHA-256 of the
checkpoint file, computed at load time; two systems share a policy iff
their checkpoint hashes match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...data.normalization import NormalizationStats, Normalizer
from ...utils.hashing import hash_file
from ..meta import PolicyMeta  # re-exported here for backward-compatible imports
from .model import ConditionalUnet1d
from .sampling import sample_chunks
from .scheduler import DiffusionScheduler

__all__ = ["PolicyMeta", "DiffusionPolicy", "build_model", "build_scheduler"]


def build_model(policy_cfg: dict, meta: PolicyMeta) -> ConditionalUnet1d:
    obs_feature_dim = meta.state_dim + (
        meta.action_dim if meta.include_previous_action else 0
    )
    model_cfg = policy_cfg.get("model", {})
    return ConditionalUnet1d(
        action_dim=meta.action_dim,
        global_cond_dim=meta.obs_horizon * obs_feature_dim,
        channels=tuple(model_cfg.get("channels", [128, 256, 512])),
        diffusion_embedding_dim=int(model_cfg.get("diffusion_embedding_dim", 128)),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        cond_predict_scale=bool(model_cfg.get("cond_predict_scale", False)),
    )


def build_scheduler(policy_cfg: dict) -> DiffusionScheduler:
    dcfg = policy_cfg.get("diffusion", {})
    pred_type = str(dcfg.get("prediction_type", "epsilon"))
    if pred_type != "epsilon":
        raise NotImplementedError(
            f"prediction_type={pred_type!r} is not implemented; the trainer and sampler "
            "are epsilon-prediction only. Remove the key or set it to 'epsilon'."
        )
    return DiffusionScheduler(
        num_train_steps=int(dcfg.get("training_steps", 100)),
        schedule=str(dcfg.get("beta_schedule", "cosine")),
    )


class DiffusionPolicy:
    """Frozen inference-time diffusion policy (implements ActionChunkPolicy)."""

    def __init__(
        self,
        model: ConditionalUnet1d,
        scheduler: DiffusionScheduler,
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
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.meta = meta
        self.config = config
        self.device = device
        self._checkpoint_hash = checkpoint_hash
        self.checkpoint_path = checkpoint_path
        self.weights_kind = weights_kind
        dcfg = config.get("diffusion", {})
        self.sampler = str(dcfg.get("inference_sampler", "ddim"))
        self.num_inference_steps = int(dcfg.get("inference_steps", 10))
        self.temperature = float(dcfg.get("temperature", 1.0))
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
        generator: torch.Generator,
    ) -> torch.Tensor:
        """[obs_horizon, obs_feature_dim] raw obs -> [K, H_p, A] raw actions."""
        expected_feat = self.meta.state_dim + (
            self.meta.action_dim if self.meta.include_previous_action else 0
        )
        if observation_history.shape != (self.meta.obs_horizon, expected_feat):
            raise ValueError(
                f"observation_history shape {observation_history.shape} != "
                f"({self.meta.obs_horizon}, {expected_feat})"
            )
        cond = self._normalize_history(observation_history)
        chunks_norm = sample_chunks(
            self.model,
            self.scheduler,
            torch.from_numpy(cond.reshape(-1)).to(self.device),
            num_samples=num_samples,
            prediction_horizon=self.meta.prediction_horizon,
            action_dim=self.meta.action_dim,
            sampler=self.sampler,
            num_inference_steps=self.num_inference_steps,
            temperature=self.temperature,
            generator=generator,
            device=self.device,
        )
        chunks = self.normalizer.unnormalize_action(chunks_norm)
        low = torch.as_tensor(self._action_low, device=chunks.device)
        high = torch.as_tensor(self._action_high, device=chunks.device)
        return chunks.clamp(low, high)

    def _normalize_history(self, observation_history: np.ndarray) -> np.ndarray:
        """Normalize a raw [H_o, feat] history (state ++ optional prev action)."""
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
                "kind": "actsemble_diffusion_policy",
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
        sampler_overrides: dict | None = None,
    ) -> "DiffusionPolicy":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("kind") != "actsemble_diffusion_policy":
            raise ValueError(f"{path} is not an Actsemble diffusion-policy checkpoint")
        config = ckpt["config"]
        if sampler_overrides:
            config = {**config, "diffusion": {**config.get("diffusion", {}), **sampler_overrides}}
        meta = PolicyMeta.from_dict(ckpt["meta"])
        model = build_model(config, meta)
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
            build_scheduler(config),
            normalizer,
            meta,
            config=config,
            device=torch.device(device),
            checkpoint_hash=hash_file(path),
            checkpoint_path=str(path),
            weights_kind=weights_kind,
        )

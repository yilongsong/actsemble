"""Flow-matching policy: normalization, Euler-ODE sampling, checkpoints.

Implements the same ``ActionChunkPolicy`` protocol as the diffusion policy and
REUSES its conditional U-Net as the velocity field ``v_theta`` — so diffusion vs
flow-matching is a clean comparison with the architecture held fixed. Training is
conditional flow matching / rectified flow (velocity regression along the
straight path from noise to action); inference integrates the ODE with Euler.
Checkpoints mirror the diffusion format with ``kind = "actsemble_flow_policy"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...data.normalization import NormalizationStats, Normalizer
from ...utils.hashing import hash_file
from ..diffusion.policy import build_model
from ..meta import PolicyMeta
from .sampling import euler_sample


class FlowMatchingPolicy:
    """Frozen inference-time flow-matching policy (implements ActionChunkPolicy)."""

    def __init__(
        self,
        model: torch.nn.Module,
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
        fcfg = config.get("flow", {})
        self.num_inference_steps = int(fcfg.get("inference_steps", 10))
        self.time_scale = float(fcfg.get("time_scale", 1000.0))
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
        pass

    @torch.no_grad()
    def sample_action_chunks(
        self, observation_history: np.ndarray, *, num_samples: int, generator: torch.Generator
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
        chunks_norm = euler_sample(
            self.model,
            torch.from_numpy(cond.reshape(-1)).to(self.device),
            num_samples=num_samples,
            prediction_horizon=self.meta.prediction_horizon,
            action_dim=self.meta.action_dim,
            num_steps=self.num_inference_steps,
            time_scale=self.time_scale,
            generator=generator,
            device=self.device,
        )
        chunks = self.normalizer.unnormalize_action(chunks_norm)
        low = torch.as_tensor(self._action_low, device=chunks.device)
        high = torch.as_tensor(self._action_high, device=chunks.device)
        return chunks.clamp(low, high)

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
        path: str | Path, *, config: dict, meta: PolicyMeta,
        model_state: dict, ema_state: dict | None, train_state: dict | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"kind": "actsemble_flow_policy", "config": config, "meta": meta.to_dict(),
             "model_state": model_state, "ema_state": ema_state, "train_state": train_state},
            path,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: torch.device | str = "cpu", use_ema: bool = True
    ) -> "FlowMatchingPolicy":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("kind") != "actsemble_flow_policy":
            raise ValueError(f"{path} is not an Actsemble flow-policy checkpoint")
        config = ckpt["config"]
        meta = PolicyMeta.from_dict(ckpt["meta"])
        model = build_model(config, meta)
        if use_ema and ckpt.get("ema_state") is not None:
            model.load_state_dict(ckpt["ema_state"])
            weights_kind = "ema"
        else:
            model.load_state_dict(ckpt["model_state"])
            weights_kind = "raw"
        normalizer = Normalizer(NormalizationStats.from_dict(meta.normalization))
        return cls(
            model, normalizer, meta, config=config, device=torch.device(device),
            checkpoint_hash=hash_file(path), checkpoint_path=str(path), weights_kind=weights_kind,
        )

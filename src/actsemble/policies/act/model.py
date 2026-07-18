"""ACT model — action chunking with transformers (Zhao et al., 2023), state-based.

A DETR-style conditional VAE. Two paths share the input embeddings:

* **Style encoder** (CVAE, *training only*): a transformer encoder over
  ``[CLS] ++ obs frames ++ ground-truth action chunk`` produces a latent style
  ``z`` (mean + log-variance from the CLS token). During training ``z`` is
  reparameterized-sampled and KL-regularized toward ``N(0, I)``; the style token
  lets the decoder fit multimodal demonstrations without averaging in the loss.
* **Decoder** (*inference + training*): a transformer encoder over
  ``[z] ++ obs frames`` forms the memory; ``H_p`` learned query embeddings
  cross-attend to it (parallel / non-autoregressive) and a linear head emits the
  whole action chunk ``[H_p, A]`` at once.

At **inference** the style encoder is discarded and ``z`` is pinned to ``0`` (the
prior mean) — so the policy is a deterministic function of the observation. That
determinism is exactly why temporal ensembling (averaging overlapping chunk
predictions) is well behaved for ACT: the only spread across overlapping
predictions is observation-time drift, not sampling noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ACTModel(nn.Module):
    def __init__(
        self,
        *,
        obs_feature_dim: int,
        action_dim: int,
        obs_horizon: int,
        prediction_horizon: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_feature_dim = int(obs_feature_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        # shared input embeddings
        self.obs_proj = nn.Linear(self.obs_feature_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)  # style encoder only

        def enc_stack(n):
            layer = nn.TransformerEncoderLayer(
                hidden_dim, n_heads, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            # no padding mask is ever passed, so disable the nested-tensor path
            return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)

        # -- CVAE style encoder: [CLS, obs(H_o), actions(H_p)] -> (mu, logvar)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.style_pos = nn.Parameter(
            torch.zeros(1, 1 + obs_horizon + prediction_horizon, hidden_dim)
        )
        self.style_encoder = enc_stack(n_encoder_layers)
        self.latent_head = nn.Linear(hidden_dim, 2 * latent_dim)

        # -- decoder path: memory = [z, obs(H_o)]; H_p learned queries cross-attend
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.memory_pos = nn.Parameter(torch.zeros(1, 1 + obs_horizon, hidden_dim))
        self.memory_encoder = enc_stack(n_encoder_layers)
        self.query_embed = nn.Parameter(torch.zeros(1, prediction_horizon, hidden_dim))
        dec_layer = nn.TransformerDecoderLayer(
            hidden_dim, n_heads, dim_feedforward, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        for p in (self.cls_token, self.style_pos, self.memory_pos, self.query_embed):
            nn.init.normal_(p, std=0.02)

    def encode_style(self, obs: torch.Tensor, actions: torch.Tensor,
                     action_mask: torch.Tensor | None = None):
        """[B,H_o,feat], [B,H_p,A] -> mu, logvar  (each [B, latent_dim]).

        ``action_mask`` ([B,H_p] bool, True = real) excludes padded (replicated
        terminal) action tokens from the style encoder's attention, so they do
        not pollute the latent posterior (canonical ACT)."""
        b = obs.shape[0]
        tokens = torch.cat(
            [self.cls_token.expand(b, -1, -1), self.obs_proj(obs), self.action_proj(actions)],
            dim=1,
        ) + self.style_pos
        key_padding_mask = None
        if action_mask is not None:
            prefix = torch.zeros(b, 1 + self.obs_horizon, dtype=torch.bool, device=obs.device)
            key_padding_mask = torch.cat([prefix, ~action_mask.bool()], dim=1)  # True = ignore
        cls = self.style_encoder(tokens, src_key_padding_mask=key_padding_mask)[:, 0]
        mu, logvar = self.latent_head(cls).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """[B,H_o,feat], [B,latent_dim] -> predicted action chunk [B,H_p,A]."""
        b = obs.shape[0]
        memory = torch.cat(
            [self.latent_proj(z).unsqueeze(1), self.obs_proj(obs)], dim=1
        ) + self.memory_pos
        memory = self.memory_encoder(memory)
        h = self.decoder(self.query_embed.expand(b, -1, -1), memory)
        return self.action_head(h)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, *,
                action_mask: torch.Tensor | None = None,
                generator: torch.Generator | None = None):
        """Full CVAE forward (training): returns (pred_chunk, mu, logvar).
        Padded action positions (``action_mask`` False) are excluded from the
        style encoder; the caller masks them out of the reconstruction loss."""
        mu, logvar = self.encode_style(obs, actions, action_mask)
        std = torch.exp(0.5 * logvar)
        eps = torch.empty_like(std).normal_(generator=generator)
        pred = self.decode(obs, mu + eps * std)
        return pred, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mu, sigma^2) || N(0, I)), averaged over the batch."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

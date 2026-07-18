"""ACT model — action chunking with transformers (Zhao et al., 2023), state-based.

A DETR-style conditional VAE. Two paths share the input embeddings:

* **Style encoder** (CVAE, *training only*): a transformer encoder over
  ``[CLS] ++ obs frames ++ ground-truth action chunk`` produces a latent style
  ``z`` (mean + log-variance from the CLS token). During training ``z`` is
  reparameterized-sampled and KL-regularized toward ``N(0, I)``; the style token
  lets the decoder fit multimodal demonstrations without averaging in the loss.
* **Decoder** (*inference + training*): a transformer encoder over
  ``[z] ++ obs frames`` forms the memory; ``H_p`` query embeddings cross-attend
  to it (parallel / non-autoregressive) and a linear head emits the whole action
  chunk ``[H_p, A]`` at once.

At **inference** the style encoder is discarded and ``z`` is pinned to ``0`` (the
prior mean) — so the policy is a deterministic function of the observation.

Two transformer implementations, selected by ``arch``:

* ``detr`` (faithful, canonical ACT): ReLU + **post-norm** layers; positional
  embeddings are injected into the attention **query/key only** (never the
  value); the decoder targets start as **zeros** and the learned query embeddings
  act as query positional embeddings added into the self/cross attention queries
  at every layer (DETR-VAE). This mirrors tonyzhaozh/act's transformer.
* ``torch_builtin`` (default; the lightweight variant kept for the recorded
  findings): stock ``nn.TransformerEncoder``/``Decoder`` with GELU, pre-norm, a
  single positional add on the input, and the learned queries fed as the decoder
  input values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Fixed sinusoidal position table (ACT's ``get_sinusoid_encoding_table``)."""
    pos = torch.arange(n_position).unsqueeze(1).float()
    i = torch.arange(d_hid).unsqueeze(0).float()
    angle = pos / torch.pow(torch.tensor(10000.0), 2 * (i // 2) / d_hid)
    angle[:, 0::2] = torch.sin(angle[:, 0::2])
    angle[:, 1::2] = torch.cos(angle[:, 1::2])
    return angle.unsqueeze(0)  # [1, n_position, d_hid]


def _with_pos(x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return x if pos is None else x + pos


class DETREncoderLayer(nn.Module):
    """Post-norm transformer encoder layer (DETR/ACT). The positional embedding
    is added to the attention query/key only, never to the value; ReLU FFN."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos=None, key_padding_mask=None):
        q = k = _with_pos(src, pos)
        attn = self.self_attn(
            q, k, value=src, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        src = self.norm1(src + self.dropout1(attn))
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src


class DETRDecoderLayer(nn.Module):
    """Post-norm transformer decoder layer (DETR/ACT). Targets start as zeros;
    the learned query positional embedding is injected into the self- and
    cross-attention queries, the memory positional embedding into the
    cross-attention keys (never values); ReLU FFN."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos, memory_pos=None):
        q = k = _with_pos(tgt, query_pos)
        sa = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(sa))
        ca = self.cross_attn(
            query=_with_pos(tgt, query_pos),
            key=_with_pos(memory, memory_pos),
            value=memory,
            need_weights=False,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(ca))
        ff = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff))
        return tgt


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
        pos_embedding: str = "learned",
        arch: str = "torch_builtin",
    ):
        super().__init__()
        if pos_embedding not in ("learned", "sinusoidal"):
            raise ValueError(f"pos_embedding must be learned|sinusoidal, got {pos_embedding!r}")
        if arch not in ("torch_builtin", "detr"):
            raise ValueError(f"arch must be torch_builtin|detr, got {arch!r}")
        self.pos_embedding = pos_embedding
        self.arch = arch
        self.obs_feature_dim = int(obs_feature_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        # shared input embeddings
        self.obs_proj = nn.Linear(self.obs_feature_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)  # style encoder only

        # -- CVAE style encoder: [CLS, obs(H_o), actions(H_p)] -> (mu, logvar)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        n_style = 1 + obs_horizon + prediction_horizon
        if pos_embedding == "sinusoidal":  # canonical ACT: fixed sinusoidal on the CVAE encoder
            self.register_buffer("style_pos", sinusoid_table(n_style, hidden_dim), persistent=False)
        else:
            self.style_pos = nn.Parameter(torch.zeros(1, n_style, hidden_dim))
        self.latent_head = nn.Linear(hidden_dim, 2 * latent_dim)

        # -- decoder path: memory = [z, obs(H_o)]; H_p queries cross-attend
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.memory_pos = nn.Parameter(torch.zeros(1, 1 + obs_horizon, hidden_dim))
        self.query_embed = nn.Parameter(torch.zeros(1, prediction_horizon, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, action_dim)

        if arch == "torch_builtin":
            def enc_stack(n):
                layer = nn.TransformerEncoderLayer(
                    hidden_dim, n_heads, dim_feedforward, dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)

            self.style_encoder = enc_stack(n_encoder_layers)
            self.memory_encoder = enc_stack(n_encoder_layers)
            dec_layer = nn.TransformerDecoderLayer(
                hidden_dim, n_heads, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)
        else:  # detr: faithful post-norm/ReLU layers with per-layer pos injection
            self.style_layers = nn.ModuleList(
                DETREncoderLayer(hidden_dim, n_heads, dim_feedforward, dropout)
                for _ in range(n_encoder_layers)
            )
            self.memory_layers = nn.ModuleList(
                DETREncoderLayer(hidden_dim, n_heads, dim_feedforward, dropout)
                for _ in range(n_encoder_layers)
            )
            self.decoder_layers = nn.ModuleList(
                DETRDecoderLayer(hidden_dim, n_heads, dim_feedforward, dropout)
                for _ in range(n_decoder_layers)
            )
            # Reference ACT uses SEPARATE joint-state projections for the CVAE
            # posterior (obs_proj) and the policy decoder (obs_proj_dec), and a
            # final LayerNorm over the decoder stack before the action head.
            self.obs_proj_dec = nn.Linear(self.obs_feature_dim, hidden_dim)
            self.decoder_norm = nn.LayerNorm(hidden_dim)

        learned_pos = [self.cls_token, self.memory_pos, self.query_embed]
        if pos_embedding == "learned":
            learned_pos.append(self.style_pos)  # sinusoidal style_pos is a fixed buffer
        for p in learned_pos:
            nn.init.normal_(p, std=0.02)

        if arch == "detr":  # DETR _reset_parameters: xavier_uniform on all >1-D weights
            for module in (self.style_layers, self.memory_layers, self.decoder_layers):
                for p in module.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            for lin in (self.obs_proj, self.obs_proj_dec, self.action_proj,
                        self.latent_proj, self.latent_head, self.action_head):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)

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
        )
        key_padding_mask = None
        if action_mask is not None:
            prefix = torch.zeros(b, 1 + self.obs_horizon, dtype=torch.bool, device=obs.device)
            key_padding_mask = torch.cat([prefix, ~action_mask.bool()], dim=1)  # True = ignore
        if self.arch == "torch_builtin":
            cls = self.style_encoder(tokens + self.style_pos, src_key_padding_mask=key_padding_mask)[:, 0]
        else:
            h = tokens
            for layer in self.style_layers:
                h = layer(h, pos=self.style_pos, key_padding_mask=key_padding_mask)
            cls = h[:, 0]
        mu, logvar = self.latent_head(cls).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """[B,H_o,feat], [B,latent_dim] -> predicted action chunk [B,H_p,A]."""
        b = obs.shape[0]
        if self.arch == "torch_builtin":
            memory = torch.cat([self.latent_proj(z).unsqueeze(1), self.obs_proj(obs)], dim=1)
            memory = self.memory_encoder(memory + self.memory_pos)
            h = self.decoder(self.query_embed.expand(b, -1, -1), memory)
            return self.action_head(h)
        # detr: separate decoder obs projection, zero-target decode, final norm
        memory = torch.cat([self.latent_proj(z).unsqueeze(1), self.obs_proj_dec(obs)], dim=1)
        for layer in self.memory_layers:
            memory = layer(memory, pos=self.memory_pos)
        query_pos = self.query_embed.expand(b, -1, -1)
        h = torch.zeros_like(query_pos)  # DETR: zero-initialized decoder targets
        for layer in self.decoder_layers:
            h = layer(h, memory, query_pos=query_pos, memory_pos=self.memory_pos)
        return self.action_head(self.decoder_norm(h))

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

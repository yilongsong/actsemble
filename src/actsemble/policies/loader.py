"""Load any Actsemble policy checkpoint as an ActionChunkPolicy.

Dispatches on the checkpoint's ``kind`` so the systems / evaluation layers stay
policy-architecture-agnostic: a diffusion checkpoint and an ACT checkpoint both
come back as objects implementing the same ``ActionChunkPolicy`` protocol.
"""

from __future__ import annotations

from pathlib import Path

import torch

_LOADERS = {
    "actsemble_diffusion_policy": "actsemble.policies.diffusion.policy:DiffusionPolicy",
    "actsemble_act_policy": "actsemble.policies.act.policy:ACTPolicy",
    "actsemble_flow_policy": "actsemble.policies.flow.policy:FlowMatchingPolicy",
}


def load_policy(checkpoint: str | Path, *, device="cpu", use_ema: bool = True):
    """Return the right policy object for ``checkpoint`` (diffusion or ACT)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = ckpt.get("kind")
    target = _LOADERS.get(kind)
    if target is None:
        raise ValueError(
            f"Unknown policy checkpoint kind {kind!r} in {checkpoint}; "
            f"expected one of {sorted(_LOADERS)}"
        )
    module_name, cls_name = target.split(":")
    import importlib

    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls.from_checkpoint(checkpoint, device=device, use_ema=use_ema)

"""Policy-family dispatch shared by installed and repository CLIs."""

from __future__ import annotations

from .train_act_policy import train_act_policy
from .train_diffusion_policy import train_diffusion_policy
from .train_flow_policy import train_flow_policy

POLICY_TRAINERS = {
    "diffusion": train_diffusion_policy,
    "act": train_act_policy,
    "flow": train_flow_policy,
}


def resolve_policy_family(cfg: dict) -> str:
    if cfg.get("type"):
        family = str(cfg["type"])
    else:
        family = {"conditional_unet_1d": "diffusion", "act": "act"}.get(
            cfg.get("model", {}).get("type", "conditional_unet_1d"), "diffusion"
        )
    if family not in POLICY_TRAINERS:
        raise ValueError(
            f"Unknown policy family {family!r}; expected {sorted(POLICY_TRAINERS)}"
        )
    return family


def policy_trainer(cfg: dict):
    return POLICY_TRAINERS[resolve_policy_family(cfg)]

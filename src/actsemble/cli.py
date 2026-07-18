"""Installed command-line entry points for the core training workflows."""

from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .training.train_component import train_component
from .training.factory import policy_trainer, resolve_policy_family


def train_policy_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an Actsemble action-chunk policy"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    resolve_policy_family(cfg)
    summary = policy_trainer(cfg)(
        policy_cfg=cfg,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        resume=args.resume,
    )
    print(summary)


def train_component_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an Actsemble compatibility component"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = train_component(
        component_cfg=load_config(args.config),
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        resume=args.resume,
    )
    print(summary)


def verify_gpu_main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the benchmark CUDA, ManiSkill, and object-nudge path"
    )
    parser.add_argument("--task", default="PushT-v1")
    parser.add_argument("--controller", default="pd_ee_delta_pose")
    parser.add_argument("--backend", default="physx_cuda")
    args = parser.parse_args()

    from .sim.gpu_verification import verify_gpu_environment

    report = verify_gpu_environment(
        task_id=args.task,
        controller=args.controller,
        simulation_backend=args.backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True))

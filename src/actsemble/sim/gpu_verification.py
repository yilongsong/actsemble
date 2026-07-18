"""Host-side verification of the benchmark CUDA and ManiSkill path."""

from __future__ import annotations

import warnings

import numpy as np
import torch


def verify_gpu_environment(
    *,
    task_id: str = "PushT-v1",
    controller: str = "pd_ee_delta_pose",
    simulation_backend: str = "physx_cuda",
) -> dict:
    """Exercise CUDA discovery, a live ManiSkill reset/step, and object nudge.

    This deliberately fails instead of falling back to CPU. A successful return
    proves that the current process can see a CUDA driver and execute the exact
    simulator backend used by the benchmark.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see CUDA in this process. Run this command directly "
            "on the GPU host; containers also need NVIDIA device/driver passthrough."
        )

    from .env_factory import env_contract, make_env
    from .perturbations.object_nudge import ObjectNudgePerturbation

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA reports that you have .* available devices.*fork_rng",
            category=UserWarning,
        )
        env = make_env(
            task_id=task_id,
            control_mode=controller,
            sim_backend=simulation_backend,
            render_mode=None,
        )
    try:
        env.reset(seed=0)
        state_before = env.unwrapped.get_state_dict()["actors"]["Tee"].clone()
        nudge = ObjectNudgePerturbation(
            step_window=(0, 0), max_translation=0.03, seed=0
        )
        nudge.reset(episode_seed=0)
        nudge.before_step(env, 0)
        state_after = env.unwrapped.get_state_dict()["actors"]["Tee"]
        movement = torch.linalg.norm(state_after[0, :2] - state_before[0, :2]).item()
        if not (0.0 < movement <= 0.03 + 1e-6):
            raise RuntimeError(
                f"Object-nudge verification produced movement {movement}"
            )

        action_space = env.unwrapped.single_action_space
        env.step(np.zeros(action_space.shape, dtype=np.float32))
        torch.cuda.synchronize()
        contract = env_contract(env)
        if contract["simulation_backend"] != simulation_backend:
            raise RuntimeError(
                "ManiSkill created backend "
                f"{contract['simulation_backend']!r}, expected {simulation_backend!r}"
            )
        return {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "environment": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in contract.items()
            },
            "object_nudge_translation": movement,
        }
    finally:
        env.close()

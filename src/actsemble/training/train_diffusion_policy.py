"""Diffusion-policy training loop.

Operates ONLY on frozen dataset files. This module (and everything it
imports) must never import ManiSkill, gymnasium, or any rollout code —
enforced by tests/training/test_training_has_no_sim_dependency.py.

Protocol properties (docs/checkpoint_selection_protocol.md):
* every stochastic stage has its own named generator derived from the
  training seed (§2); the name -> derived-seed map is recorded in
  train_config.json;
* full model snapshots are saved every ``checkpoint_every`` steps and at
  the final step (§4); early stopping does not exist;
* an optional ``on_checkpoint`` callback (used for mid-training policy
  screening) runs inside ``preserve_rng_states``, so screening cannot
  alter the training trajectory (§3). The callback is injected by the
  orchestrator — this module never imports simulator code itself.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.normalization import Normalizer, compute_stats
from ..data.reader import DatasetReader
from ..data.torch_dataset import DiffusionWindowDataset
from ..data.validation import validate_dataset
from ..data.windows import split_episodes
from ..policies.diffusion.policy import (
    DiffusionPolicy,
    PolicyMeta,
    build_model,
    build_scheduler,
)
from ..seed import derive_seed, seed_everything, torch_generator
from ..utils.rng_state import preserve_rng_states
from ..utils.serialization import save_json
from .logging import TrainingLogger

# Callback receives keyword arguments including ``step`` and ``checkpoint_path``.
CheckpointCallback = Callable[..., None]


class EMA:
    """Exponential moving average of model parameters.

    Two warmup schedules: the default ``(1+s)/(10+s)`` ramp (lightweight
    policies), or — when ``power`` is set — the diffusers ``EMAModel`` power
    schedule ``1-(1+s/inv_gamma)^(-power)`` clamped to ``[min_value, max_value]``
    (the authoritative Diffusion-Policy setting: power 0.75, max 0.9999)."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        *,
        power: float | None = None,
        inv_gamma: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.decay = float(decay)
        self.power = None if power is None else float(power)
        self.inv_gamma = float(inv_gamma)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _decay(self) -> float:
        if self.power is None:
            return min(self.decay, (1 + self.step) / (10 + self.step))
        s = (
            self.step - 1
        )  # diffusers: max(0, optimization_step - update_after_step - 1)
        if s <= 0:
            return 0.0
        d = 1.0 - (1.0 + s / self.inv_gamma) ** (-self.power)
        return float(min(max(d, self.min_value), self.max_value))

    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        decay = self._decay()
        with torch.no_grad():
            for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.lerp_(p.detach(), 1.0 - decay)
            for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
                ema_b.copy_(b)


def resolve_total_steps(
    tcfg: dict,
    n_train_windows: int,
    batch_size: int,
    max_steps_override: int | None = None,
) -> int:
    """Training budget, dataset-adaptive. Precedence: ``--max-steps`` override >
    explicit ``training.max_steps`` > ``training.max_epochs`` (converted to steps
    via this dataset's window count) > 10000. Expressing the budget in EPOCHS
    makes it consistent across dataset sizes; the exact number is calibrated to
    exceed the validation plateau and checkpoint selection captures the peak."""
    if max_steps_override is not None:
        return int(max_steps_override)
    if tcfg.get("max_steps") is not None:
        return int(tcfg["max_steps"])
    if tcfg.get("max_epochs") is not None:
        steps_per_epoch = max(
            1, n_train_windows // int(batch_size)
        )  # matches drop_last
        return int(tcfg["max_epochs"]) * steps_per_epoch
    return 10000


def build_lr_scheduler(optimizer, tcfg: dict, total_steps: int):
    """Optional LR schedule. ``constant`` (default) -> None; ``cosine`` -> linear
    warmup for ``lr_warmup_steps`` then cosine decay to 0 (authoritative DP)."""
    kind = str(tcfg.get("lr_scheduler", "constant"))
    if kind == "constant":
        return None
    if kind != "cosine":
        raise ValueError(f"Unknown lr_scheduler: {kind}")
    warmup = int(tcfg.get("lr_warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_policy_meta(
    reader: DatasetReader, policy_cfg: dict, split_hash: str, norm_stats
) -> PolicyMeta:
    action_def = json.loads(reader.metadata.action_definition)
    bounds = action_def.get("bounds")
    if bounds is None:
        raise ValueError("Dataset action_definition has no bounds; cannot build policy")
    obs_cfg = policy_cfg.get("observation", {})
    act_cfg = policy_cfg.get("action", {})
    meta = PolicyMeta(
        dataset_hash=reader.dataset_hash,
        split_hash=split_hash,
        normalization=norm_stats.to_dict(),
        task_id=reader.metadata.task_id,
        controller=reader.metadata.controller,
        simulation_backend=reader.metadata.simulation_backend,
        state_dim=reader.state_dim,
        action_dim=reader.action_dim,
        action_low=list(map(float, bounds[0])),
        action_high=list(map(float, bounds[1])),
        obs_horizon=int(obs_cfg.get("history", 2)),
        prediction_horizon=int(act_cfg.get("prediction_horizon", 16)),
        action_horizon=int(act_cfg.get("execution_horizon", 8)),
        include_previous_action=bool(obs_cfg.get("include_previous_action", False)),
    )
    # Audit trail: carry demonstration-subset provenance from the dataset.
    for key in ("subset_hash", "subset_size", "subset_seed", "source_bundle_sha256"):
        if key in reader.metadata.extra:
            meta.extra[key] = reader.metadata.extra[key]
    meta.extra["environment"] = {
        "simulator": reader.metadata.simulator,
        "simulator_version": reader.metadata.simulator_version,
        "robot": reader.metadata.robot,
        "observation_mode": reader.metadata.observation_mode,
        "control_frequency": reader.metadata.control_frequency,
        "state_dimension": reader.metadata.state_dimension,
        "action_dimension": reader.metadata.action_dimension,
    }
    # Window alignment used at training time (future_only | diffusion_policy);
    # the eval system's execution offset must match (H_o-1 for diffusion_policy).
    meta.extra["window_alignment"] = str(
        policy_cfg.get("action", {}).get("window_alignment", "future_only")
    )
    return meta


def named_training_seeds(seed: int) -> dict[str, int]:
    """Dedicated derived seeds, one per stochastic stage (protocol §2)."""
    return {
        "policy_init": derive_seed(seed, "policy_init"),
        "dataloader_order": derive_seed(seed, "dataloader_order"),
        "diffusion_noise": derive_seed(seed, "diffusion_noise"),
        "diffusion_timesteps": derive_seed(seed, "diffusion_timesteps"),
        "validation": derive_seed(seed, "validation"),
    }


def train_diffusion_policy(
    *,
    policy_cfg: dict,
    dataset_path: str | Path,
    output_dir: str | Path,
    max_steps: int | None = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    resume: bool = False,
    on_checkpoint: CheckpointCallback | None = None,
) -> dict:
    """Train; returns a summary dict with checkpoint paths and final losses."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Training output directory is not empty: {output_dir}. "
            "Use a new directory or pass resume=True."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tcfg = policy_cfg.get("training", {})
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device(device)
    gen_seeds = named_training_seeds(seed)

    reader = DatasetReader(dataset_path)
    validate_dataset(reader)
    split = split_episodes(
        reader.episode_ids,
        val_fraction=float(tcfg.get("val_fraction", 0.1)),
        seed=int(tcfg.get("split_seed", 0)),
    )
    # Normalization statistics come from the full frozen dataset (all
    # successful demonstrations); documented in docs/experiment_contract.md.
    stats = compute_stats(reader.episodes)
    normalizer = Normalizer(stats)
    meta = make_policy_meta(reader, policy_cfg, split.hash, stats)

    ep_by_id = {ep.episode_id: ep for ep in reader.episodes}
    train_eps = [ep_by_id[i] for i in split.train_episode_ids]
    val_eps = [ep_by_id[i] for i in split.val_episode_ids]
    window_alignment = str(
        policy_cfg.get("action", {}).get("window_alignment", "future_only")
    )
    train_ds = DiffusionWindowDataset(
        train_eps,
        normalizer,
        obs_horizon=meta.obs_horizon,
        prediction_horizon=meta.prediction_horizon,
        include_previous_action=meta.include_previous_action,
        alignment=window_alignment,
        action_horizon=meta.action_horizon,
    )
    val_ds = (
        DiffusionWindowDataset(
            val_eps,
            normalizer,
            obs_horizon=meta.obs_horizon,
            prediction_horizon=meta.prediction_horizon,
            include_previous_action=meta.include_previous_action,
            alignment=window_alignment,
            action_horizon=meta.action_horizon,
        )
        if val_eps
        else None
    )

    batch_size = int(tcfg.get("batch_size", 256))
    loader_gen = torch_generator(gen_seeds["dataloader_order"])
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_ds) > batch_size,
        num_workers=0,
        generator=loader_gen,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    # Model initialization draws from the global torch stream; give it a
    # dedicated derived seed so init is independent of anything else.
    torch.manual_seed(gen_seeds["policy_init"])
    model = build_model(policy_cfg, meta).to(device)
    scheduler = build_scheduler(policy_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("learning_rate", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-6)),
        betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
    )
    ema_cfg = tcfg.get("ema", {})
    ema = EMA(
        model,
        decay=float(tcfg.get("ema_decay", 0.999)),
        power=ema_cfg.get("power"),
        inv_gamma=float(ema_cfg.get("inv_gamma", 1.0)),
        min_value=float(ema_cfg.get("min_value", 0.0)),
        max_value=float(ema_cfg.get("max_value", 0.9999)),
    )
    grad_clip = float(tcfg.get("gradient_clip_norm", 1.0))
    mask_padding = bool(tcfg.get("mask_padded_actions", False))
    total_steps = resolve_total_steps(tcfg, len(train_ds), batch_size, max_steps)
    steps_per_epoch = max(1, len(train_ds) // batch_size)  # matches drop_last
    # Optional cosine LR (authoritative DP). Its state and planned total budget
    # are part of the exact-resume contract below.
    lr_scheduler = build_lr_scheduler(optimizer, tcfg, total_steps)
    log_every = int(tcfg.get("log_every", 50))
    eval_every = int(tcfg.get("eval_every", max(1, min(500, total_steps))))
    checkpoint_every = tcfg.get("checkpoint_every")  # None disables snapshots
    checkpoint_dir = output_dir / "checkpoints"

    noise_gen = torch_generator(gen_seeds["diffusion_noise"], device)
    timestep_gen = torch_generator(gen_seeds["diffusion_timesteps"], device)

    start_step = 0
    best_val = float("inf")
    resumed_epoch_state = None  # (loader_gen state at epoch start, batches consumed)
    last_path = output_dir / "last.pt"
    if resume and not last_path.exists():
        raise FileNotFoundError(f"Cannot resume: checkpoint not found: {last_path}")
    if resume:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        train_state = ckpt.get("train_state") or {}
        optimizer.load_state_dict(train_state["optimizer"])
        if lr_scheduler is not None:
            saved_total = train_state.get("schedule_total_steps")
            if saved_total != total_steps:
                raise ValueError(
                    "Exact resume with a scheduled learning rate requires the same "
                    f"total_steps ({saved_total!r} in checkpoint, {total_steps} requested). "
                    "Retrain from scratch when extending a cosine-scheduled run."
                )
            scheduler_state = train_state.get("lr_scheduler")
            if scheduler_state is None:
                raise ValueError(
                    "Resume checkpoint has no learning-rate scheduler state"
                )
            lr_scheduler.load_state_dict(scheduler_state)
        start_step = int(train_state["step"])
        ema.step = int(train_state.get("ema_step", start_step))
        best_val = float(train_state.get("best_val", best_val))
        # Restore every RNG stream that drives the trajectory so that resuming
        # to a larger max_steps is identical to a single run to that budget:
        # the diffusion-noise and timestep generators, the global torch RNG
        # (covers any in-model dropout), and the dataloader's mid-epoch
        # position. Without this a resumed run replays noise and batch order
        # from step 0, so 0->A->B != 0->B.
        rng = train_state.get("rng_state")
        if rng is None:
            raise ValueError(
                f"{last_path} has no saved rng_state; resume requires a checkpoint "
                "written by this trainer. Retrain from scratch at the target budget."
            )
        noise_gen.set_state(rng["noise"])
        timestep_gen.set_state(rng["timesteps"])
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda" and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        resumed_epoch_state = (rng["loader_epoch"], int(rng["batches_into_epoch"]))

    logger = TrainingLogger(output_dir)
    save_json(
        {
            "policy_config": policy_cfg,
            "dataset_path": str(dataset_path),
            "split": split.to_dict(),
            "dataset_hash": reader.dataset_hash,
            "split_hash": split.hash,
            "normalization_hash": stats.hash,
            "device": str(device),
            "training_seed": seed,
            "named_generator_seeds": gen_seeds,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "early_stopping": "disabled",
        },
        output_dir / "train_config.json",
    )

    def diffusion_loss(
        m: torch.nn.Module,
        batch: dict,
        ngen: torch.Generator | None,
        tgen: torch.Generator | None,
    ) -> torch.Tensor:
        x0 = batch["action_chunk"].to(device)
        cond = batch["obs_history"].to(device).flatten(1)
        noise = torch.empty_like(x0).normal_(generator=ngen)
        timesteps = scheduler.sample_timesteps(x0.shape[0], device, generator=tgen)
        x_noisy = scheduler.add_noise(x0, noise, timesteps)
        pred = m(x_noisy, timesteps, cond)
        if mask_padding:
            mask = batch["action_mask"].to(device).unsqueeze(-1).float()
            se = (pred - noise) ** 2 * mask
            return se.sum() / mask.sum().clamp(min=1.0) / x0.shape[-1]
        return F.mse_loss(pred, noise)

    def validate() -> float | None:
        if val_loader is None:
            return None
        ema.shadow.to(device).eval()
        weighted_loss = 0.0
        examples = 0
        vgen = torch_generator(gen_seeds["validation"], device)
        with torch.no_grad():
            for vb in val_loader:
                batch_n = int(vb["action_chunk"].shape[0])
                weighted_loss += (
                    diffusion_loss(ema.shadow, vb, vgen, vgen).item() * batch_n
                )
                examples += batch_n
        return weighted_loss / examples

    def save(path: Path, *, with_train_state: bool) -> None:
        train_state = None
        if with_train_state:
            train_state = {
                "step": step,
                "ema_step": ema.step,
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "lr_scheduler": lr_scheduler.state_dict()
                if lr_scheduler is not None
                else None,
                "schedule_total_steps": total_steps,
                "rng_state": {
                    "noise": noise_gen.get_state(),
                    "timesteps": timestep_gen.get_state(),
                    "torch": torch.get_rng_state(),
                    "torch_cuda": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "loader_epoch": epoch_loader_state,
                    "batches_into_epoch": batches_this_epoch,
                },
            }
        DiffusionPolicy.save_checkpoint(
            path,
            config=policy_cfg,
            meta=meta,
            model_state=model.state_dict(),
            ema_state=ema.shadow.state_dict(),
            train_state=train_state,
        )

    def snapshot_and_screen() -> None:
        """Full snapshot + optional screening callback, RNG-isolated (§3)."""
        path = checkpoint_dir / f"step_{step:06d}.pt"
        save(path, with_train_state=False)
        if on_checkpoint is not None:
            guarded = {
                "dataloader_order": loader_gen,
                "diffusion_noise": noise_gen,
                "diffusion_timesteps": timestep_gen,
            }
            with preserve_rng_states(guarded):
                on_checkpoint(step=step, checkpoint_path=path)
            model.train()

    model.train()
    step = start_step
    # epoch_loader_state is the loader generator's state at the start of the
    # current epoch; batches_this_epoch counts batches drawn since then. The
    # pair pins the dataloader's exact mid-epoch position for resume (§4).
    if resumed_epoch_state is not None:
        loader_gen.set_state(resumed_epoch_state[0])
        epoch_loader_state = loader_gen.get_state()
        data_iter = iter(loader)
        batches_this_epoch = 0
        for _ in range(resumed_epoch_state[1]):
            next(data_iter)
            batches_this_epoch += 1
    else:
        epoch_loader_state = loader_gen.get_state()
        data_iter = iter(loader)
        batches_this_epoch = 0
    final_train_loss = float("nan")
    while step < total_steps:
        try:
            batch = next(data_iter)
            batches_this_epoch += 1
        except StopIteration:
            epoch_loader_state = loader_gen.get_state()
            data_iter = iter(loader)
            batch = next(data_iter)
            batches_this_epoch = 1
        loss = diffusion_loss(model, batch, noise_gen, timestep_gen)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        ema.update(model)
        step += 1
        final_train_loss = loss.item()

        if step % log_every == 0 or step == total_steps:
            logger.log(step, {"train/loss": final_train_loss})
        if step % eval_every == 0 or step == total_steps:
            val_loss = validate()
            model.train()
            score = val_loss if val_loss is not None else final_train_loss
            if val_loss is not None:
                logger.log(step, {"val/loss": val_loss})
            if score <= best_val:
                best_val = score
                save(output_dir / "best_ema.pt", with_train_state=False)
            save(last_path, with_train_state=True)
        # Interval snapshots + screening; also fires at the final step even
        # when total_steps is not a multiple of checkpoint_every (§4, §5).
        if checkpoint_every and (
            step % int(checkpoint_every) == 0 or step == total_steps
        ):
            snapshot_and_screen()

    save(last_path, with_train_state=True)
    save(output_dir / "final.pt", with_train_state=False)
    if not (output_dir / "best_ema.pt").exists():
        save(output_dir / "best_ema.pt", with_train_state=False)
    logger.close()

    snapshots = sorted(checkpoint_dir.glob("step_*.pt")) if checkpoint_every else []
    return {
        "steps": step,
        "steps_per_epoch": steps_per_epoch,
        "final_train_loss": final_train_loss,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "checkpoints": {
            "best_ema": str(output_dir / "best_ema.pt"),
            "final": str(output_dir / "final.pt"),
            "last": str(last_path),
        },
        "snapshots": [str(p) for p in snapshots],
        "dataset_hash": reader.dataset_hash,
        "split_hash": split.hash,
        "normalization_hash": stats.hash,
        "training_seed": seed,
        "named_generator_seeds": gen_seeds,
    }

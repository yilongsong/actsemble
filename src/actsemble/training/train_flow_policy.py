"""Flow-matching policy training loop (conditional flow matching / rectified flow).

One-shot training on a frozen dataset (no simulator import). Reuses the SAME
conditional U-Net + dataset windows + normalization as the diffusion policy, so
diffusion vs flow matching is a controlled comparison. Objective: regress the
straight-path velocity ``v_theta(z_t, t, cond) -> (x1 - z0)`` where
``z_t = (1-t)*z0 + t*x1``, ``z0 ~ N(0, I)``, ``t ~ U[0,1]`` — the standard CFM
loss (Lipman/Liu/Tong; applied to manipulation as in PointFlowMatch / pi0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.normalization import Normalizer, compute_stats
from ..data.reader import DatasetReader
from ..data.torch_dataset import DiffusionWindowDataset
from ..data.windows import split_episodes
from ..policies.diffusion.policy import build_model
from ..policies.flow.policy import FlowMatchingPolicy
from ..seed import derive_seed, seed_everything, torch_generator
from ..utils.serialization import save_json
from .logging import TrainingLogger
from .train_diffusion_policy import EMA, build_lr_scheduler, make_policy_meta, resolve_total_steps


def named_flow_seeds(seed: int) -> dict[str, int]:
    return {
        "flow_init": derive_seed(seed, "flow_init"),
        "dataloader_order": derive_seed(seed, "dataloader_order"),
        "flow_noise": derive_seed(seed, "flow_noise"),
        "flow_time": derive_seed(seed, "flow_time"),
        "validation": derive_seed(seed, "validation"),
    }


def train_flow_policy(
    *, policy_cfg: dict, dataset_path, output_dir,
    max_steps: int | None = None, device="cuda" if torch.cuda.is_available() else "cpu",
    resume: bool = False,
) -> dict:
    if resume:
        raise NotImplementedError("flow policy trains one-shot; retrain at the target budget.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tcfg = policy_cfg.get("training", {})
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device(device)
    gen_seeds = named_flow_seeds(seed)

    reader = DatasetReader(dataset_path)
    split = split_episodes(reader.episode_ids, val_fraction=float(tcfg.get("val_fraction", 0.1)),
                           seed=int(tcfg.get("split_seed", 0)))
    stats = compute_stats(
        reader.episodes,
        method=str(policy_cfg.get("normalization_method", "minmax_to_unit_range")),
    )
    normalizer = Normalizer(stats)
    meta = make_policy_meta(reader, policy_cfg, split.hash, stats)

    ep_by_id = {ep.episode_id: ep for ep in reader.episodes}
    ds_kwargs = dict(
        obs_horizon=meta.obs_horizon, prediction_horizon=meta.prediction_horizon,
        include_previous_action=meta.include_previous_action,
        alignment=str(policy_cfg.get("action", {}).get("window_alignment", "future_only")),
        action_horizon=meta.action_horizon,  # reference SequenceSampler terminal range
    )
    train_ds = DiffusionWindowDataset(
        [ep_by_id[i] for i in split.train_episode_ids], normalizer, **ds_kwargs
    )
    val_eps = [ep_by_id[i] for i in split.val_episode_ids]
    val_ds = DiffusionWindowDataset(val_eps, normalizer, **ds_kwargs) if val_eps else None

    batch_size = int(tcfg.get("batch_size", 256))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        drop_last=len(train_ds) > batch_size, num_workers=0,
                        generator=torch_generator(gen_seeds["dataloader_order"]))
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
                  if val_ds is not None and len(val_ds) > 0 else None)

    torch.manual_seed(gen_seeds["flow_init"])
    model = build_model(policy_cfg, meta).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg.get("learning_rate", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-6)),
        betas=tuple(tcfg.get("adam_betas", (0.9, 0.999))),
    )
    ema_cfg = tcfg.get("ema", {})
    ema = EMA(model, decay=float(tcfg.get("ema_decay", 0.999)), power=ema_cfg.get("power"),
              inv_gamma=float(ema_cfg.get("inv_gamma", 1.0)),
              min_value=float(ema_cfg.get("min_value", 0.0)),
              max_value=float(ema_cfg.get("max_value", 0.9999)))
    time_scale = float(policy_cfg.get("flow", {}).get("time_scale", 1000.0))
    mask_padding = bool(tcfg.get("mask_padded_actions", False))
    grad_clip = float(tcfg.get("gradient_clip_norm", 1.0))
    total_steps = resolve_total_steps(tcfg, len(train_ds), batch_size, max_steps)
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    lr_scheduler = build_lr_scheduler(optimizer, tcfg, total_steps)
    log_every = int(tcfg.get("log_every", 50))
    eval_every = int(tcfg.get("eval_every", max(1, min(1000, total_steps))))
    checkpoint_every = tcfg.get("checkpoint_every")

    noise_gen = torch_generator(gen_seeds["flow_noise"], device)
    time_gen = torch_generator(gen_seeds["flow_time"], device)
    logger = TrainingLogger(output_dir)
    save_json(
        {"policy_config": policy_cfg, "dataset_path": str(dataset_path), "split": split.to_dict(),
         "dataset_hash": reader.dataset_hash, "split_hash": split.hash,
         "normalization_hash": stats.hash, "device": str(device), "training_seed": seed,
         "named_generator_seeds": gen_seeds, "time_scale": time_scale,
         "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
         "early_stopping": "disabled"},
        output_dir / "train_config.json",
    )

    def flow_loss(m, batch, ngen, tgen):
        x1 = batch["action_chunk"].to(device)       # [B, H_p, A] normalized (data)
        cond = batch["obs_history"].to(device).flatten(1)
        z0 = torch.empty_like(x1).normal_(generator=ngen)
        t = torch.rand(x1.shape[0], device=device, generator=tgen)  # [B] ~ U[0,1]
        tt = t[:, None, None]
        z_t = (1.0 - tt) * z0 + tt * x1
        v = m(z_t, t * time_scale, cond)
        target = x1 - z0
        if mask_padding:
            w = batch["action_mask"].to(device).unsqueeze(-1).float()
            return ((v - target) ** 2 * w).sum() / w.sum().clamp(min=1.0) / x1.shape[-1]
        return F.mse_loss(v, target)

    def validate():
        if val_loader is None:
            return None
        ema.shadow.to(device).eval()
        ng = torch_generator(gen_seeds["validation"], device)
        tg = torch_generator(gen_seeds["validation"] + 1, device)
        with torch.no_grad():
            losses = [flow_loss(ema.shadow, vb, ng, tg).item() for vb in val_loader]
        return float(np.mean(losses))

    def save(path, *, with_train_state):
        train_state = ({"step": step, "ema_step": ema.step, "optimizer": optimizer.state_dict(),
                        "best_val": best_val} if with_train_state else None)
        FlowMatchingPolicy.save_checkpoint(
            path, config=policy_cfg, meta=meta, model_state=model.state_dict(),
            ema_state=ema.shadow.state_dict(), train_state=train_state,
        )

    model.train()
    step = 0
    best_val = float("inf")
    last_path = output_dir / "last.pt"
    data_iter = iter(loader)
    final_loss = float("nan")
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        loss = flow_loss(model, batch, noise_gen, time_gen)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        ema.update(model)
        step += 1
        final_loss = loss.item()

        if step % log_every == 0 or step == total_steps:
            logger.log(step, {"train/loss": final_loss})
        if step % eval_every == 0 or step == total_steps:
            val_loss = validate()
            model.train()
            score = val_loss if val_loss is not None else final_loss
            if val_loss is not None:
                logger.log(step, {"val/loss": val_loss})
            if score <= best_val:
                best_val = score
                save(output_dir / "best_ema.pt", with_train_state=False)
            save(last_path, with_train_state=True)
        if checkpoint_every and (step % int(checkpoint_every) == 0 or step == total_steps):
            save(output_dir / "checkpoints" / f"step_{step:06d}.pt", with_train_state=False)

    save(last_path, with_train_state=True)
    save(output_dir / "final.pt", with_train_state=False)
    if not (output_dir / "best_ema.pt").exists():
        save(output_dir / "best_ema.pt", with_train_state=False)
    logger.close()
    return {
        "steps": step, "steps_per_epoch": steps_per_epoch, "final_train_loss": final_loss,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "checkpoints": {"best_ema": str(output_dir / "best_ema.pt"),
                        "final": str(output_dir / "final.pt"), "last": str(last_path)},
        "dataset_hash": reader.dataset_hash, "split_hash": split.hash,
        "normalization_hash": stats.hash, "training_seed": seed, "named_generator_seeds": gen_seeds,
    }

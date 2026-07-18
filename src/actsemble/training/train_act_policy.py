"""ACT policy training loop (state-based CVAE).

Trains one-shot to a fixed budget on a frozen dataset — no simulator import
(enforced by tests/test_training_has_no_sim_dependency.py), same seed discipline
and EMA/checkpoint conventions as the diffusion trainer, and the **same**
``DiffusionWindowDataset`` windows + normalization so ACT and the diffusion
policy see identical training data (a fair architecture comparison). Loss is the
CVAE objective: L1 action reconstruction + ``kl_weight`` * KL(z || N(0, I)).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.normalization import Normalizer, compute_stats
from ..data.reader import DatasetReader
from ..data.torch_dataset import ACTEpisodeDataset, DiffusionWindowDataset
from ..data.windows import split_episodes
from ..policies.act.model import ACTModel
from ..policies.act.policy import ACTPolicy, build_act_model
from ..seed import derive_seed, seed_everything, torch_generator
from ..utils.serialization import save_json
from .logging import TrainingLogger
from .train_diffusion_policy import EMA, make_policy_meta, resolve_total_steps


def named_act_seeds(seed: int) -> dict[str, int]:
    """Dedicated derived seeds, one per stochastic stage (protocol §2)."""
    return {
        "act_init": derive_seed(seed, "act_init"),
        "dataloader_order": derive_seed(seed, "dataloader_order"),
        "act_latent": derive_seed(seed, "act_latent"),
        "act_sampling": derive_seed(seed, "act_sampling"),
        "validation": derive_seed(seed, "validation"),
    }


def train_act_policy(
    *,
    policy_cfg: dict,
    dataset_path,
    output_dir,
    max_steps: int | None = None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    resume: bool = False,
) -> dict:
    from pathlib import Path

    if resume:
        raise NotImplementedError(
            "ACT baseline trains one-shot; retrain from scratch at the target budget."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tcfg = policy_cfg.get("training", {})
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device(device)
    gen_seeds = named_act_seeds(seed)

    reader = DatasetReader(dataset_path)
    split = split_episodes(
        reader.episode_ids,
        val_fraction=float(tcfg.get("val_fraction", 0.1)),
        seed=int(tcfg.get("split_seed", 0)),
    )
    stats = compute_stats(
        reader.episodes,
        method=str(policy_cfg.get("normalization_method", "minmax_to_unit_range")),
    )
    normalizer = Normalizer(stats)
    meta = make_policy_meta(reader, policy_cfg, split.hash, stats)

    ep_by_id = {ep.episode_id: ep for ep in reader.episodes}
    ds_kwargs = dict(
        obs_horizon=meta.obs_horizon,
        prediction_horizon=meta.prediction_horizon,
        include_previous_action=meta.include_previous_action,
    )
    train_eps = [ep_by_id[i] for i in split.train_episode_ids]
    # Canonical ACT weights episodes (not transitions) equally per epoch, with a
    # random start timestep; the lightweight variant samples every transition.
    episode_sampling = bool(tcfg.get("episode_sampling", False))
    if episode_sampling:
        train_ds = ACTEpisodeDataset(
            train_eps, normalizer, start_seed=gen_seeds["act_sampling"], **ds_kwargs
        )
    else:
        train_ds = DiffusionWindowDataset(train_eps, normalizer, **ds_kwargs)
    val_eps = [ep_by_id[i] for i in split.val_episode_ids]
    val_ds = DiffusionWindowDataset(val_eps, normalizer, **ds_kwargs) if val_eps else None

    batch_size = int(tcfg.get("batch_size", 256))
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=len(train_ds) > batch_size, num_workers=0,
        generator=torch_generator(gen_seeds["dataloader_order"]),
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        if val_ds is not None and len(val_ds) > 0 else None
    )

    torch.manual_seed(gen_seeds["act_init"])
    model = build_act_model(policy_cfg, meta).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("learning_rate", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    use_ema = bool(tcfg.get("use_ema", True))  # canonical ACT: False (eval best-val raw weights)
    ema = EMA(model, decay=float(tcfg.get("ema_decay", 0.999))) if use_ema else None
    kl_weight = float(policy_cfg.get("act", {}).get("kl_weight", 10.0))
    mask_padding = bool(tcfg.get("mask_padded_actions", True))  # canonical ACT masks padding
    grad_clip = float(tcfg.get("gradient_clip_norm", 1.0))
    total_steps = resolve_total_steps(tcfg, len(train_ds), batch_size, max_steps)
    steps_per_epoch = max(1, len(train_ds) // batch_size)  # episode-weighted when episode_sampling
    log_every = int(tcfg.get("log_every", 50))
    eval_every = int(tcfg.get("eval_every", max(1, min(1000, total_steps))))
    checkpoint_every = tcfg.get("checkpoint_every")
    checkpoint_dir = output_dir / "checkpoints"

    latent_gen = torch_generator(gen_seeds["act_latent"], device)
    logger = TrainingLogger(output_dir)
    save_json(
        {"policy_config": policy_cfg, "dataset_path": str(dataset_path), "split": split.to_dict(),
         "dataset_hash": reader.dataset_hash, "split_hash": split.hash,
         "normalization_hash": stats.hash, "device": str(device), "training_seed": seed,
         "named_generator_seeds": gen_seeds, "kl_weight": kl_weight,
         "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
         "episode_sampling": episode_sampling, "early_stopping": "disabled"},
        output_dir / "train_config.json",
    )

    def act_loss(m, batch, lgen):
        x0 = batch["action_chunk"].to(device)      # [B, H_p, A] normalized
        obs = batch["obs_history"].to(device)       # [B, H_o, feat] normalized
        amask = batch["action_mask"].to(device) if mask_padding else None  # [B, H_p] real=True
        pred, mu, logvar = m(obs, x0, action_mask=amask, generator=lgen)
        if amask is not None:
            w = amask.unsqueeze(-1).float()  # exclude replicated padding from the L1
            recon = (F.l1_loss(pred, x0, reduction="none") * w).sum() / w.sum().clamp(min=1.0) \
                / x0.shape[-1]
        else:
            recon = F.l1_loss(pred, x0)
        kl = ACTModel.kl_divergence(mu, logvar)
        return recon + kl_weight * kl, recon.detach(), kl.detach()

    def validate():
        if val_loader is None:
            return None
        m = ema.shadow if use_ema else model
        m.to(device).eval()
        vgen = torch_generator(gen_seeds["validation"], device)
        with torch.no_grad():
            losses = [act_loss(m, vb, vgen)[0].item() for vb in val_loader]
        return float(np.mean(losses))

    def save(path, *, with_train_state):
        train_state = (
            {"step": step, "ema_step": ema.step if use_ema else 0,
             "optimizer": optimizer.state_dict(), "best_val": best_val}
            if with_train_state else None
        )
        ACTPolicy.save_checkpoint(
            path, config=policy_cfg, meta=meta, model_state=model.state_dict(),
            ema_state=ema.shadow.state_dict() if use_ema else None, train_state=train_state,
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
        loss, recon, kl = act_loss(model, batch, latent_gen)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if use_ema:
            ema.update(model)
        step += 1
        final_loss = loss.item()

        if step % log_every == 0 or step == total_steps:
            logger.log(step, {"train/loss": final_loss, "train/l1": recon.item(),
                              "train/kl": kl.item()})
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
            save(checkpoint_dir / f"step_{step:06d}.pt", with_train_state=False)

    save(last_path, with_train_state=True)
    save(output_dir / "final.pt", with_train_state=False)
    if not (output_dir / "best_ema.pt").exists():
        save(output_dir / "best_ema.pt", with_train_state=False)
    logger.close()

    return {
        "steps": step,
        "steps_per_epoch": steps_per_epoch,
        "final_train_loss": final_loss,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "checkpoints": {"best_ema": str(output_dir / "best_ema.pt"),
                        "final": str(output_dir / "final.pt"), "last": str(last_path)},
        "dataset_hash": reader.dataset_hash, "split_hash": split.hash,
        "normalization_hash": stats.hash, "training_seed": seed,
        "named_generator_seeds": gen_seeds,
    }

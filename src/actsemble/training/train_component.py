"""Compatibility-component training loop.

Same-data contract: trains ONLY on the frozen dataset file (positives are
real windows; negatives are deterministic transformations of same-dataset
chunks). No simulator, no rollouts, no reward, no success labels.

High offline compatibility accuracy does not imply improved closed-loop
task success. Closed-loop evaluation is the decisive test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..components.action_chunk_compatibility import (
    ALL_NEGATIVE_TYPES,
    CompatibilityMLP,
    NegativeConfig,
    NegativeGenerator,
    window_negative_rng,
)
from ..data.normalization import MINMAX, Normalizer, compute_stats
from ..data.reader import DatasetReader
from ..data.validation import validate_dataset
from ..data.windows import enumerate_window_indices, extract_window, split_episodes
from ..seed import derive_seed, seed_everything, torch_generator
from ..types import EpisodeRecord
from ..utils.serialization import load_json, save_json
from .logging import TrainingLogger
from .train_diffusion_policy import make_policy_meta


def named_verifier_seeds(seed: int) -> dict[str, int]:
    """Dedicated derived seeds, one per stochastic stage (protocol §2).

    The negative-generation seed is separate and comes from
    ``training.negative_seed`` (it must stay fixed across verifier arms).
    """
    return {
        "verifier_init": derive_seed(seed, "verifier_init"),
        "dataloader_order": derive_seed(seed, "dataloader_order"),
    }


class CompatibilityDataset(Dataset):
    """Yields (obs history, positive chunk, negative chunks, negative types)."""

    def __init__(
        self,
        episodes: list[EpisodeRecord],
        normalizer: Normalizer,
        negative_generator: NegativeGenerator,
        *,
        obs_horizon: int,
        prediction_horizon: int,
        include_previous_action: bool,
        negatives_per_positive: int,
        negative_seed: int,
        alignment: str = "future_only",
        action_horizon: int | None = None,
    ):
        self.episodes = episodes
        self.normalizer = normalizer
        self.neg_gen = negative_generator
        self.obs_horizon = int(obs_horizon)
        self.prediction_horizon = int(prediction_horizon)
        self.include_previous_action = bool(include_previous_action)
        self.negatives_per_positive = int(negatives_per_positive)
        self.negative_seed = int(negative_seed)
        # Positive chunks must be extracted with the SAME window alignment as the
        # policy whose candidates this scorer will rank (future_only vs
        # diffusion_policy), else positives and scored candidates are misaligned.
        self.alignment = str(alignment)
        self.action_horizon = None if action_horizon is None else int(action_horizon)
        self.indices = enumerate_window_indices(
            episodes,
            alignment=self.alignment,
            obs_horizon=self.obs_horizon,
            prediction_horizon=self.prediction_horizon,
            action_horizon=self.action_horizon,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ei, t = self.indices[i]
        ep = self.episodes[ei]
        w = extract_window(
            ep,
            t,
            obs_horizon=self.obs_horizon,
            prediction_horizon=self.prediction_horizon,
            alignment=self.alignment,
        )
        obs = np.asarray(
            self.normalizer.normalize_state(w.obs_history), dtype=np.float32
        )
        if self.include_previous_action:
            prev = np.asarray(
                self.normalizer.normalize_action(w.prev_action_history),
                dtype=np.float32,
            )
            obs = np.concatenate([obs, prev], axis=1)
        negs, type_ids = [], []
        for r in range(self.negatives_per_positive):
            rng = window_negative_rng(self.negative_seed, ep.episode_id, t, r)
            neg, ntype = self.neg_gen.generate(
                self.episodes, ei, t, w.action_chunk, rng
            )
            negs.append(
                np.asarray(self.normalizer.normalize_action(neg), dtype=np.float32)
            )
            type_ids.append(ALL_NEGATIVE_TYPES.index(ntype))
        return {
            "obs_history": torch.from_numpy(obs),
            "positive_chunk": torch.from_numpy(
                np.asarray(
                    self.normalizer.normalize_action(w.action_chunk), dtype=np.float32
                )
            ),
            "negative_chunks": torch.from_numpy(np.stack(negs)),
            "negative_types": torch.tensor(type_ids, dtype=torch.long),
        }


def _forward_scores(model, batch, device):
    obs = batch["obs_history"].to(device).flatten(1)  # [B, H_o*feat]
    pos = batch["positive_chunk"].to(device).flatten(1)  # [B, H_p*A]
    negs = batch["negative_chunks"].to(device)  # [B, N, H_p, A]
    b, n = negs.shape[0], negs.shape[1]
    pos_scores = model(torch.cat([obs, pos], dim=1))  # [B]
    neg_in = torch.cat(
        [obs.unsqueeze(1).expand(b, n, -1).reshape(b * n, -1), negs.reshape(b * n, -1)],
        dim=1,
    )
    neg_scores = model(neg_in).reshape(b, n)  # [B, N]
    return pos_scores, neg_scores


def _loss(objective: str, pos_scores, neg_scores, margin: float):
    if objective == "binary_classification":
        logits = torch.cat([pos_scores, neg_scores.reshape(-1)])
        labels = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores.reshape(-1))]
        )
        return F.binary_cross_entropy_with_logits(logits, labels)
    if objective == "pairwise_ranking":
        diff = pos_scores.unsqueeze(1) - neg_scores  # [B, N]
        return F.relu(margin - diff).mean()
    raise ValueError(f"Unknown objective: {objective}")


@torch.no_grad()
def evaluate_compatibility(model, loader, device) -> dict:
    """Offline metrics: accuracies, per-type breakdown, score histograms."""
    model.eval()
    pairwise_correct = 0
    pairwise_total = 0
    pos_all, neg_all, types_all = [], [], []
    for batch in loader:
        pos_scores, neg_scores = _forward_scores(model, batch, device)
        pairwise_correct += int((pos_scores.unsqueeze(1) > neg_scores).sum().item())
        pairwise_total += int(neg_scores.numel())
        pos_all.append(pos_scores.cpu())
        neg_all.append(neg_scores.reshape(-1).cpu())
        types_all.append(batch["negative_types"].reshape(-1))
    pos = torch.cat(pos_all)
    neg = torch.cat(neg_all)
    types = torch.cat(types_all)
    hist_bins = np.linspace(0.0, 1.0, 21)
    per_type = {}
    for ti, tname in enumerate(ALL_NEGATIVE_TYPES):
        m = types == ti
        if m.any():
            per_type[tname] = {
                "count": int(m.sum()),
                "accuracy": float((neg[m] < 0.0).float().mean()),
            }
    pos_acc = float((pos > 0.0).float().mean())
    neg_acc = float((neg < 0.0).float().mean())
    bce = float(
        (
            F.binary_cross_entropy_with_logits(
                pos, torch.ones_like(pos), reduction="sum"
            )
            + F.binary_cross_entropy_with_logits(
                neg, torch.zeros_like(neg), reduction="sum"
            )
        )
        / (len(pos) + len(neg))
    )
    return {
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_accuracy": (pos_acc + neg_acc) / 2.0,
        "validation_loss": bce,
        "pairwise_ranking_accuracy": pairwise_correct / max(1, pairwise_total),
        "num_positives": int(len(pos)),
        "num_negatives": int(len(neg)),
        "per_negative_type": per_type,
        "score_histogram": {
            "bin_edges": hist_bins.tolist(),
            "positive": np.histogram(torch.sigmoid(pos).numpy(), bins=hist_bins)[
                0
            ].tolist(),
            "negative": np.histogram(torch.sigmoid(neg).numpy(), bins=hist_bins)[
                0
            ].tolist(),
        },
        "note": (
            "High offline compatibility accuracy does not imply improved "
            "closed-loop task success. Closed-loop evaluation is the decisive test."
        ),
    }


def train_component(
    *,
    component_cfg: dict,
    dataset_path: str | Path,
    output_dir: str | Path,
    max_steps: int | None = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    resume: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Training output directory is not empty: {output_dir}. "
            "Use a new directory or pass resume=True."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tcfg = component_cfg.get("training", {})
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device(device)
    gen_seeds = named_verifier_seeds(seed)

    reader = DatasetReader(dataset_path)
    validate_dataset(reader)
    split = split_episodes(
        reader.episode_ids,
        val_fraction=float(tcfg.get("val_fraction", 0.1)),
        seed=int(tcfg.get("split_seed", 0)),
    )
    stats = compute_stats(
        reader.episodes,
        method=str(component_cfg.get("normalization_method", MINMAX)),
    )
    normalizer = Normalizer(stats)
    # Reuse the policy meta builder for the shared contract fields
    # (horizons come from this component's config and must match the
    # policy's; enforced at system construction).
    shared_meta = make_policy_meta(reader, component_cfg, split.hash, stats)

    neg_cfg = NegativeConfig.from_dict(component_cfg.get("negatives", {}))
    neg_gen = NegativeGenerator(
        neg_cfg,
        action_low=np.asarray(shared_meta.action_low),
        action_high=np.asarray(shared_meta.action_high),
        prediction_horizon=shared_meta.prediction_horizon,
        obs_horizon=shared_meta.obs_horizon,
        alignment=shared_meta.extra.get("window_alignment", "future_only"),
    )

    ep_by_id = {ep.episode_id: ep for ep in reader.episodes}
    train_eps = [ep_by_id[i] for i in split.train_episode_ids]
    val_eps = [ep_by_id[i] for i in split.val_episode_ids]
    ds_kwargs = dict(
        obs_horizon=shared_meta.obs_horizon,
        prediction_horizon=shared_meta.prediction_horizon,
        include_previous_action=shared_meta.include_previous_action,
        # Positives MUST share the policy's chunk alignment, else the scorer sees
        # index 0 = a_t in training but a_{t-1} when ranking DP-aligned candidates.
        alignment=shared_meta.extra.get("window_alignment", "future_only"),
        action_horizon=shared_meta.action_horizon,
        negatives_per_positive=int(tcfg.get("negatives_per_positive", 1)),
        negative_seed=int(tcfg.get("negative_seed", 1234)),
    )
    train_ds = CompatibilityDataset(train_eps, normalizer, neg_gen, **ds_kwargs)
    val_ds = (
        CompatibilityDataset(val_eps, normalizer, neg_gen, **ds_kwargs)
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

    feat = shared_meta.state_dim + (
        shared_meta.action_dim if shared_meta.include_previous_action else 0
    )
    input_dim = (
        shared_meta.obs_horizon * feat
        + shared_meta.prediction_horizon * shared_meta.action_dim
    )
    model_cfg = component_cfg.get("model", {})
    # Dedicated init seed so weights are independent of other global draws.
    torch.manual_seed(gen_seeds["verifier_init"])
    model = CompatibilityMLP(
        input_dim,
        hidden=tuple(model_cfg.get("hidden", [512, 512, 256])),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("learning_rate", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-6)),
    )
    grad_clip = float(tcfg.get("gradient_clip_norm", 1.0))
    objective = str(tcfg.get("objective", "binary_classification"))
    margin = float(tcfg.get("ranking_margin", 1.0))
    total_steps = int(
        max_steps if max_steps is not None else tcfg.get("max_steps", 5000)
    )
    log_every = int(tcfg.get("log_every", 50))
    eval_every = int(tcfg.get("eval_every", max(1, min(500, total_steps))))
    checkpoint_every = tcfg.get("checkpoint_every")  # None disables snapshots
    checkpoint_dir = output_dir / "checkpoints"
    offline_history: list[dict] = []

    meta = {
        **shared_meta.to_dict(),
        "negatives": neg_cfg.to_dict(),
        "objective": objective,
        "component_type": "action_chunk_compatibility",
    }

    start_step = 0
    best_score = -float("inf")
    resumed_epoch_state = None
    last_path = output_dir / "last.pt"
    if resume and not last_path.exists():
        raise FileNotFoundError(f"Cannot resume: checkpoint not found: {last_path}")
    if resume:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ts = ckpt.get("train_state") or {}
        optimizer.load_state_dict(ts["optimizer"])
        start_step = int(ts["step"])
        best_score = float(ts.get("best_score", best_score))
        rng = ts.get("rng_state")
        if rng is None:
            raise ValueError(
                f"{last_path} has no saved rng_state; exact resume requires a newer checkpoint"
            )
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda" and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        resumed_epoch_state = (rng["loader_epoch"], int(rng["batches_into_epoch"]))
        history_path = output_dir / "offline_history.json"
        if history_path.exists():
            offline_history = load_json(history_path)

    logger = TrainingLogger(output_dir)
    save_json(
        {
            "component_config": component_cfg,
            "dataset_path": str(dataset_path),
            "split": split.to_dict(),
            "dataset_hash": reader.dataset_hash,
            "split_hash": split.hash,
            "normalization_hash": stats.hash,
            "device": str(device),
            "training_seed": seed,
            "named_generator_seeds": {
                **gen_seeds,
                "negative_generation": int(tcfg.get("negative_seed", 1234)),
            },
            "early_stopping": "disabled",
        },
        output_dir / "train_config.json",
    )

    def save(path: Path, *, with_train_state: bool) -> None:
        payload = {
            "kind": "actsemble_compatibility_component",
            "config": component_cfg,
            "meta": meta,
            "model_state": model.state_dict(),
        }
        if with_train_state:
            payload["train_state"] = {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
                "rng_state": {
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
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    model.train()
    step = start_step
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
        pos_scores, neg_scores = _forward_scores(model, batch, device)
        loss = _loss(objective, pos_scores, neg_scores, margin)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        step += 1
        final_train_loss = loss.item()

        if step % log_every == 0 or step == total_steps:
            with torch.no_grad():
                rank_acc = float((pos_scores.unsqueeze(1) > neg_scores).float().mean())
            logger.log(
                step, {"train/loss": final_train_loss, "train/ranking_acc": rank_acc}
            )
        if step % eval_every == 0 or step == total_steps:
            eval_loader = val_loader if val_loader is not None else loader
            metrics = evaluate_compatibility(model, eval_loader, device)
            model.train()
            tag = "val" if val_loader is not None else "trainset"
            logger.log(
                step,
                {
                    f"{tag}/positive_accuracy": metrics["positive_accuracy"],
                    f"{tag}/negative_accuracy": metrics["negative_accuracy"],
                    f"{tag}/pairwise_ranking_accuracy": metrics[
                        "pairwise_ranking_accuracy"
                    ],
                },
            )
            score = metrics["pairwise_ranking_accuracy"]
            if score >= best_score:
                best_score = score
                save(output_dir / "best.pt", with_train_state=False)
            save(last_path, with_train_state=True)

        # Interval snapshots + offline validation history for the protocol's
        # offline-only verifier selection (§8-§9). Also fires at the final
        # step even when total_steps is not a multiple of checkpoint_every.
        if checkpoint_every and (
            step % int(checkpoint_every) == 0 or step == total_steps
        ):
            snap_path = checkpoint_dir / f"step_{step:06d}.pt"
            save(snap_path, with_train_state=False)
            eval_loader = val_loader if val_loader is not None else loader
            metrics = evaluate_compatibility(model, eval_loader, device)
            model.train()
            offline_history.append(
                {
                    "step": step,
                    "checkpoint_path": str(snap_path),
                    "evaluated_on": "validation_episodes"
                    if val_loader is not None
                    else "train_episodes",
                    "metrics": metrics,
                }
            )
            save_json(offline_history, output_dir / "offline_history.json")

    save(last_path, with_train_state=True)
    save(output_dir / "final.pt", with_train_state=False)
    if not (output_dir / "best.pt").exists():
        save(output_dir / "best.pt", with_train_state=False)

    # Final offline evaluation report (episode-disjoint val when present).
    eval_loader = val_loader if val_loader is not None else loader
    final_metrics = evaluate_compatibility(model, eval_loader, device)
    final_metrics["evaluated_on"] = (
        "validation_episodes" if val_loader is not None else "train_episodes"
    )
    save_json(final_metrics, output_dir / "offline_eval.json")
    logger.close()

    return {
        "steps": step,
        "final_train_loss": final_train_loss,
        "best_ranking_accuracy": None if best_score == -float("inf") else best_score,
        "offline_eval": final_metrics,
        "offline_history": offline_history,
        "checkpoints": {
            "best": str(output_dir / "best.pt"),
            "final": str(output_dir / "final.pt"),
            "last": str(last_path),
        },
        "snapshots": [h["checkpoint_path"] for h in offline_history],
        "dataset_hash": reader.dataset_hash,
        "split_hash": split.hash,
        "normalization_hash": stats.hash,
        "training_seed": seed,
        "named_generator_seeds": gen_seeds,
    }

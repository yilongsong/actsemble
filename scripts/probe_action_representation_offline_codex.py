#!/usr/bin/env python
"""Teacher-forced slot diagnostics for an ACT checkpoint (development only).

Measures each decoder slot on real demonstration states. For absolute joint
targets, errors and vector gains are computed after subtracting the current
joint configuration, exposing the small residual hidden under the large
qpos-to-target identity component. For delta actions, the action itself is the
residual. This is an offline probe: it never imports or steps the simulator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.policies.loader import load_policy  # noqa: E402
from actsemble.utils.serialization import save_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-windows", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    policy = load_policy(args.checkpoint, device="cpu", use_ema=True)
    reader = DatasetReader(args.dataset)
    hp = policy.meta.prediction_horizon
    rows = []
    for ep in reader.episodes:
        for t in range(0, max(0, len(ep) - hp + 1)):
            rows.append((ep, t))
    rng = np.random.default_rng(20260719)
    if len(rows) > args.max_windows:
        rows = [rows[i] for i in rng.choice(len(rows), args.max_windows, replace=False)]

    states = np.stack([ep.state[t] for ep, t in rows]).astype(np.float32)
    targets = np.stack([ep.action[t : t + hp] for ep, t in rows]).astype(np.float32)
    obs_norm = np.asarray(
        policy.normalizer.normalize_state(states[:, None, :]), dtype=np.float32
    )
    preds = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            obs = torch.from_numpy(obs_norm[start : start + args.batch_size])
            z = torch.zeros(len(obs), policy.model.latent_dim)
            pred_norm = policy.model.decode(obs, z)
            preds.append(
                policy.normalizer.unnormalize_action(pred_norm).cpu().numpy()
            )
    pred = np.concatenate(preds, axis=0).astype(np.float64)
    targets = targets.astype(np.float64)

    absolute_joint = targets.shape[-1] == 7 and reader.metadata.controller == "pd_joint_pos"
    if absolute_joint:
        anchor = states[:, None, :7].astype(np.float64)
    else:
        anchor = np.zeros((len(rows), 1, targets.shape[-1]), dtype=np.float64)
    target_residual = targets - anchor
    pred_residual = pred - anchor

    slots = []
    eps = 1e-12
    for k in range(hp):
        tr = target_residual[:, k]
        pr = pred_residual[:, k]
        dot = np.sum(pr * tr, axis=1)
        t2 = np.sum(tr * tr, axis=1)
        p2 = np.sum(pr * pr, axis=1)
        slots.append(
            {
                "slot": k,
                "target_residual_norm_mean": float(np.linalg.norm(tr, axis=1).mean()),
                "target_residual_norm_median": float(np.median(np.linalg.norm(tr, axis=1))),
                "pred_residual_norm_mean": float(np.linalg.norm(pr, axis=1).mean()),
                "pred_residual_norm_median": float(np.median(np.linalg.norm(pr, axis=1))),
                "residual_gain_global": float(np.sum(dot) / (np.sum(t2) + eps)),
                "residual_cosine_mean": float(np.mean(dot / (np.sqrt(t2 * p2) + eps))),
                "raw_l1": float(np.mean(np.abs(pred[:, k] - targets[:, k]))),
                "anchor_copy_l1": float(np.mean(np.abs(anchor[:, 0] - targets[:, k]))),
                "pred_better_than_anchor_fraction": float(
                    np.mean(
                        np.mean(np.abs(pred[:, k] - targets[:, k]), axis=1)
                        < np.mean(np.abs(anchor[:, 0] - targets[:, k]), axis=1)
                    )
                ),
            }
        )

    # Directly test the off-by-one suspicion at slot 0: which captured target is
    # closer to current q, a_t or a_{t-1}? This is descriptive, not a label swap.
    off_by_one = None
    if absolute_joint:
        valid = [(ep, t) for ep, t in rows if t > 0]
        q = np.stack([ep.state[t, :7] for ep, t in valid]).astype(np.float64)
        cur = np.stack([ep.action[t] for ep, t in valid]).astype(np.float64)
        prev = np.stack([ep.action[t - 1] for ep, t in valid]).astype(np.float64)
        off_by_one = {
            "current_target_minus_q_norm_mean": float(np.linalg.norm(cur - q, axis=1).mean()),
            "previous_target_minus_q_norm_mean": float(np.linalg.norm(prev - q, axis=1).mean()),
            "previous_target_closer_fraction": float(
                np.mean(np.linalg.norm(prev - q, axis=1) < np.linalg.norm(cur - q, axis=1))
            ),
        }

    # Overlapping plans at s_t and the *real demonstrated* s_{t+1} predict the
    # same future actions: old slot k+1 and new slot k. Ground-truth overlap is
    # exact, so disagreement measures decoder/conditioning inconsistency without
    # any closed-loop or simulator-distribution confound.
    next_states = np.stack([ep.state[t + 1] for ep, t in rows]).astype(np.float32)
    next_obs_norm = np.asarray(
        policy.normalizer.normalize_state(next_states[:, None, :]), dtype=np.float32
    )
    next_preds = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            obs = torch.from_numpy(next_obs_norm[start : start + args.batch_size])
            z = torch.zeros(len(obs), policy.model.latent_dim)
            pred_norm = policy.model.decode(obs, z)
            next_preds.append(
                policy.normalizer.unnormalize_action(pred_norm).cpu().numpy()
            )
    next_pred = np.concatenate(next_preds, axis=0).astype(np.float64)
    if absolute_joint:
        next_anchor = next_states[:, None, :7].astype(np.float64)
    else:
        next_anchor = np.zeros((len(rows), 1, targets.shape[-1]), dtype=np.float64)
    overlap = []
    for k in range(hp - 1):
        old = pred[:, k + 1] - next_anchor[:, 0]
        fresh = next_pred[:, k] - next_anchor[:, 0]
        revision = np.linalg.norm(fresh - old, axis=1)
        old_norm = np.linalg.norm(old, axis=1)
        fresh_norm = np.linalg.norm(fresh, axis=1)
        cosine = np.sum(old * fresh, axis=1) / (old_norm * fresh_norm + eps)
        overlap.append(
            {
                "new_slot": k,
                "old_slot": k + 1,
                "revision_mean": float(revision.mean()),
                "revision_median": float(np.median(revision)),
                "revision_over_old_median": float(
                    np.median(revision / (old_norm + eps))
                ),
                "direction_cosine_mean": float(cosine.mean()),
                "direction_cosine_median": float(np.median(cosine)),
                "opposite_direction_fraction": float(np.mean(cosine < 0)),
            }
        )

    result = {
        "kind": "development_only_teacher_forced_slot_probe",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_windows": len(rows),
        "absolute_joint": absolute_joint,
        "off_by_one_descriptive": off_by_one,
        "slots": slots,
        "teacher_forced_overlap_consistency": overlap,
    }
    save_json(result, args.output)
    print(f"[offline-slot] wrote {args.output}")
    for row in slots[:8]:
        print(
            f"  k={row['slot']:>2}: target {row['target_residual_norm_mean']:.3f}, "
            f"pred {row['pred_residual_norm_mean']:.3f}, "
            f"gain {row['residual_gain_global']:.3f}, "
            f"cos {row['residual_cosine_mean']:.3f}, L1 {row['raw_l1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

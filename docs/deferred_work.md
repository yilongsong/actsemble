# Deferred work — known gaps to revisit

Tracked here so they are not forgotten. Most are "fine for **development** (the
`diagnostic` panel), needed before a **claim** (`final_test`, freeze-gated)."
Raised during the temporal-ensembling / ACT work (2026-07-17).

## Before any REPORTED claim (final_test, freeze-gated)
- **ACT ↔ checkpoint-selection protocol.** `scripts/run_protocol.py` hardcodes
  `train_diffusion_policy`; the ACT trainer (`training/train_act_policy.py`) has
  no `on_checkpoint` screening callback, no RNG-isolated screening, no
  confirmation workflow, no resume-trajectory preservation. ACT and diffusion
  therefore cannot yet be selected under identical rules. Wire ACT into
  `run_protocol` with the same screening/selection/resume before ACT-vs-diffusion
  is *claimed*. (docs/checkpoint_selection_protocol.md)
- **final_test 500 × ≥5 policy seeds.** All current temporal/ACT/replan results
  are on the `diagnostic` panel (300 ep, single seed) = development. A reported
  claim needs `final_test` (500) × ≥5 policy seeds, paired, freeze-gated.
  active_min has only `policy_seed_0` → train 4 more policy (and ACT) seeds.

## Model / training fidelity (for a canonical-reproduction benchmark)
- **Labeling.** Report as "state-conditioned **lightweight** ACT variant" and
  "**accelerated lightweight** Diffusion Policy variant", NOT canonical baselines.
- **ACT posterior collapse.** KL → ~7e-7 by 30k steps → the latent carries ~no
  style; effectively deterministic chunk-BC. Tolerable now (z=0 inference is
  deterministic *by design*; PushT RL demos are near-unimodal), but this
  checkpoint does not *exercise* ACT's multimodality mechanism. Levers: lower β /
  KL annealing / the padding mask (done, helps a little).
- **Scale to canonical** only if reproducing papers: ACT hidden 512 / ff 3200 / 7
  decoder layers (ours 256/512/4 — deliberately small for n=400, which would else
  overfit); DP channels 256/512/1024, emb 256, 100-step DDPM, cosine LR + warmup,
  AdamW betas (0.95, 0.999) (ours 128/256/512, emb 128, 10-step DDIM — the 10-step
  DDIM is a *feature*: it fits the real-time latency budget, see below). ACT is
  evaluated on EMA weights; original ACT uses raw weights. Param counts differ
  (~7.4M ACT vs ~16.8M diffusion) and diffusion does 10 U-Net evals vs ACT's 1
  decode, so equal training steps ≠ equal capacity/compute.
- **Window convention.** Repo-wide `data/windows.py` uses a future-only chunk from
  s_t, every timestep a sample. Canonical DP aligns the prediction horizon with
  the observation sequence and executes indices ~[H_o-1 .. H_o-1+H_a], padding only
  through H_a-1. Ours is internally consistent and *shared by both policies* (so
  internally fair), but not canonical and it causes more terminal-action
  replication (~20% of targets are padding). Changing it forces retraining the
  established diffusion policy.

## Method extensions
- **Latency-constrained benchmark mode.** The RULE is **ADOPTED** — see
  `docs/latency_rule.md` ("no-pause real-time contract": `H_a ≥ ⌈τ·f_c⌉`, report
  only latency-feasible replan frequencies, tag each result with (τ, platform)).
  PENDING is the *implementation*: a mode on `scripts/sweep_replan_frequency.py`
  that measures p95 τ per policy, computes `H_a_min`, evaluates each policy there,
  and reports "best closed-loop success under the contract" — so the benchmark
  rewards inference speed (ACT ~1.3 ms → 20 Hz; DDPM-100 ~170 ms → capped at 5 Hz).
- **Exact-ACT temporal weighting switch.** DONE — `temporal_ensemble` now has a
  `recency` knob (`recent` = freshest-weighted, default; `oldest` = the literal ACT
  convention `exp(-m·i)`, i=0 oldest). Faithful ACT replication ran via
  `scripts/replicate_act.py`.

## Pre-existing (from the framework review)
- **Fold oracle / pass@k into `evaluation/`** to reuse `ReplanningSystemBase`
  (needs the branching/rollout generalization — Tier-2-adjacent).

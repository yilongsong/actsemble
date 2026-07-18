# Deferred work — known gaps to revisit

Tracked here so they are not forgotten. Most are "fine for **development** (the
`diagnostic` / screening panels), needed before a **claim** (`final_test`,
freeze-gated)."

## Reference-fidelity review — ADDRESSED 2026-07-18
A rigorous review of the "canonical" reproductions found real deviations from the
references (not just labeling). Fixed this pass and retrained under the overnight
run (`scripts/overnight_canonical.py`). Coverage: `tests/test_fidelity_fixes.py`.

- **DP/flow window range.** `enumerate_window_indices(alignment="diffusion_policy",
  action_horizon=...)` now replicates the reference Diffusion-Policy
  `SequenceSampler` terminal range (pad_before `H_o-1`, pad_after `H_a-1`) instead
  of enumerating every timestep, so edge-replicated terminal actions are no longer
  over-represented (was ~18% of target slots, trained unmasked as the reference
  does). `future_only` is unchanged (lightweight reproducibility).
- **Execution-offset safety.** `build_system` derives the default execution offset
  from the policy's `window_alignment` meta (`H_o-1` for `diffusion_policy`), so
  ANY system using a DP-aligned checkpoint executes the action for time `t` — not a
  past-aligned one — even when the system config omits `execution.action_offset`.
  An explicit config value still wins.
- **Temporal-ensemble offset.** `TemporalEnsembleSystem` emits `chunk[offset+age]`
  (was `chunk[age]`), so temporal ensembling is correct for DP-aligned chunks.
- **Scorer / alignment contract.** `CompatibilityDataset` extracts positive chunks
  with the policy's `window_alignment`; `check_same_data` now compares
  `window_alignment` between policy and components (a future-only scorer cannot
  rank DP-aligned candidate chunks).
- **DP FiLM.** `cond_predict_scale: true` gives the official DP `out = scale*h + bias`
  (gated; the stabilized `(1+scale)*h + bias` stays the default so lightweight
  checkpoints keep their semantics). A `prediction_type` other than `epsilon` now
  raises instead of being silently ignored.
- **ACT is DETR-VAE-*style* (close to reference ACT, not a line-by-line port).**
  `arch: detr` = ReLU + **post-norm** layers, positional embeddings injected into
  the attention **query/key only**, a **zero-initialized decoder target** with
  learned query positions injected at every layer, **separate posterior/decoder
  observation projections**, a **final decoder LayerNorm**, **DETR xavier init**,
  **observation-only normalization**, and **episode-weighted train + (fixed) val**
  (was GELU + pre-norm + shared projection + queries-as-input + next_state-in-stats).
  Gated; `torch_builtin` (lightweight) stays the default. Takes effect on the ACT
  retrain. Remaining residuals below.
- **ACT data recipe.** `episode_sampling: true` weights **episodes** (not
  transitions) equally per epoch with a random start (reference ACT); `standardize`
  clips std to `>= 0.01` (reference ACT) so near-constant dims are not
  over-amplified. This also makes "epoch" mean #episodes — the reference budget unit.
- **Checkpoint-selection parity.** `evaluate_checkpoint_on_panel` (shared by
  screening AND confirmation) is now policy-agnostic (`load_policy`) and
  offset-aware (`build_system`), so ACT/flow screen + confirm identically to DP.
  `scripts/overnight_canonical.py` runs train → screen → confirm → 500-ep final per
  family and records the SELECTED epoch.

### Residual deviations (documented, not yet closed)
- DP inference defaults to **DDIM-16** (fast — latency contract); DDPM-100 is the
  reference sampler (`inference_sampler: ddpm, inference_steps: 100`). Training is
  DDPM-100 either way — a deliberate, disclosed acceleration.
- ACT is a **state-conditioned** DETR-VAE reproduction, NOT the vision-based ALOHA
  system and NOT a line-by-line port: no image backbone / image-token path; chunk
  16 (task-matched to PushT, vs ALOHA 100); batch 64 (vs 8); exact init *constants*
  for the CLS/query/pos embeddings are not matched (xavier on the weights is).
  Describe it as "state-conditioned ACT (DETR-VAE-style)", never "exact ACT". The
  chunk length also scopes our temporal-ensembling result to chunk-16 PushT — it is
  not a test of ACT's chunk-100 TE claim (`docs/findings_temporal_replan.md`).
- Flow matching is a **conditional rectified-flow baseline** over the DP U-Net —
  NOT a reproduction of pi0 (VLM + action expert) or PointFlowMatch (point-cloud
  SO(3) flow). Labeled as such everywhere.
- `run_protocol.py` still trains via `train_diffusion_policy` for the full
  freeze/verifier/integration path; the overnight driver is the multi-family
  screen+select path. Folding ACT/flow into `run_protocol` train-time screening is
  deferred (the post-hoc driver achieves the same selection under identical rules).
- **ACT posterior collapse** on near-unimodal PushT (KL -> ~0): the CVAE latent
  carries ~no style, so the model is effectively deterministic chunk-BC. Tolerable
  by design (z=0 inference), but PushT does not *exercise* ACT's multimodality —
  a multimodal task would.

## Before any REPORTED claim (final_test, freeze-gated)
- **final_test 500 x >= 5 policy seeds, paired.** The overnight run is single-seed
  (development-tier). A reported claim needs >= 5 policy seeds, paired, freeze-gated.
  active_min has only `policy_seed_0`.
- **Selected best epoch per config — TRACKED.** `overnight_{family}.json` records
  `selected_step`, `steps_per_epoch`, `selected_epoch`, and the full screening
  success-vs-epoch curve; every `train_config.json` records
  `steps_per_epoch`/`total_steps`. Keep this per (model x dataset x task) so the
  budget that actually won is auditable (not the max_epochs cap).

## Authoritative track — still open
- **Image / high-dim path (Phase 4).** Vision architecture is code-ready
  (`policies/vision/backbone.py`: self-contained ResNet18 + spatial-softmax +
  `ImageObsEncoder` for DP/Flow `global_cond`, `image_feature_map` tokens for ACT).
  DEFERRED: (a) wiring the vision encoder into the model forwards under
  `observation.mode: rgb`; (b) the image DATA pipeline (env rgb obs, image dataset
  collection/rendering, image training). torchvision RECONCILED:
  `torchvision==0.26.0+cu128` matches torch 2.11 and works; the self-contained
  ResNet18 is kept as a dependency-free fallback.
- **Multi-seed + larger data.** n=400 is below the canonical (large) architectures'
  data regime; for the benchmark, retrain at larger n with >= 5 seeds and the
  screening-selection protocol.

## Method extensions
- **Latency-constrained benchmark mode.** The RULE is ADOPTED (`docs/latency_rule.md`:
  no-pause real-time contract `H_a >= ceil(tau * f_c)`, report only latency-feasible
  replan frequencies, tag each result with (tau, platform)). PENDING is the
  *implementation*: a mode on `sweep_replan_frequency.py` that measures p95 tau per
  policy, computes `H_a_min`, and reports best closed-loop success under the contract.
- **Exact-ACT temporal weighting switch.** DONE — `temporal_ensemble` has a
  `recency` knob (`recent` freshest-weighted default; `oldest` = the literal ACT
  `exp(-m*i)` convention).

## Pre-existing (from the framework review)
- **Fold oracle / pass@k into `evaluation/`** to reuse `ReplanningSystemBase`
  (needs the branching/rollout generalization — Tier-2-adjacent).

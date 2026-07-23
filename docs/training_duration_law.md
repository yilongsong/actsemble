# Closed-loop-aware training budgets and checkpoint selection for robot imitation learning

**Full methods specification.** Written 2026-07-22. Revised after external
critique.

**Status labels used throughout.** Every statement is tagged:
`[MEASURED]` — observed, with the artifact it came from.
`[PLANNED]` — designed, not yet run.
`[HYPOTHESIS]` — a conjecture the design is built to test or refute.
`[RETRACTED]` — previously asserted here, now known false.

Nothing in this document is claim-grade. Every measured result below is single
seed, single dataset draw, and most are screening-tier (50 rollouts, ±13 points).

---

## 1. The problem

When you train an imitation policy on a fixed set of demonstrations, you must
choose how long to train. That choice is normally made by convention — "500
epochs", "300k steps" — and the convention is rarely reported as an
experimental variable.

It is one. Two conventions that sound equivalent are not:

- A fixed **epoch** budget holds the number of *passes over the data* constant.
  Because a pass over a small dataset is short, this gives small datasets
  proportionally **fewer weight updates**.
- A fixed **gradient step** budget holds the number of *weight updates*
  constant. This gives small datasets proportionally **more passes**.

Neither is wrong. They answer different questions, and a paper that does not say
which it used has not specified its experiment.

### 1.1 This blocked a real experiment `[MEASURED]`

We compared two demonstration-generation methods (`v1`, `v3`) at the same
episode count (25) under a fixed 10500 gradient steps:

| | v1 n=25 | v3 n=25 |
|---|---|---|
| transitions | 1652 | 2132 (+29%) |
| gradient steps | 10500 | 10500 |
| steps per epoch | 5 | 7 |
| **epochs completed** | **2100** | **1500** |
| final test (500 episodes) | **47.0%** CI [42.7, 51.4] | **25.0%** CI [21.4, 29.0] |

*Source:* `outputs/pickcube/teleop_v{1,3}/sweep_n25/diffusion/final_test.json`.

This is a valid **compute-matched** comparison: under equal weight updates, v1
wins decisively. It is **not** a converged-performance comparison, because v3
received 29% fewer passes and its screening curve was still rising when the
budget expired while v1's had plateaued. The two questions have different
answers and the experiment only answers one.

This distinction — compute-matched vs exposure-matched vs convergence-matched —
is itself a finding worth reporting, since results can reverse between them.

---

## 2. Definitions and notation

Every quantity used later, defined once. Units given explicitly.

### 2.1 Data

| symbol | meaning | units |
|---|---|---|
| `n` | demonstration **episodes** in the training set | count |
| `D` | **transitions** (state-action pairs) in the training set | count |
| `W` | **training windows** after horizon slicing (`≈ D`, one window per start index) | count |
| `M` | independent **episodes** contributing those windows (`= n`) | count |
| `v` | validation fraction held out of the training pool (0.1) | — |
| `B` | batch size (256) | count |

`D` and `M` are reported separately throughout. They are **not**
interchangeable: a dataset can gain transitions because the demonstrator moved
more slowly, in which case the extra windows are highly correlated and carry
little new information. Our own v1→v3 comparison has exactly this property.

### 2.2 Training

| symbol | meaning | units |
|---|---|---|
| `S_ep` | gradient steps per epoch `= floor((1-v)·W / B)`, drop_last | steps |
| `t` | gradient steps elapsed (**the primary time axis**) | steps |
| `T` | total gradient steps in a run | steps |
| `E` | epochs elapsed `= t / S_ep` | epochs |
| `H` | **pinned** learning-rate schedule horizon (see §5.1) | steps |
| `N` | model parameter count | count |

### 2.3 Evaluation

| symbol | meaning |
|---|---|
| `J(t)` | closed-loop success rate of the checkpoint at step `t` |
| `J̄(t)` | **expected** `J(t)` over dataset draw, training seed, and episode sampling |
| `J*` | `sup_t J̄(t)`, the best attainable expected success in a run |
| `δ` | acceptable absolute regret, in success **percentage points** |
| `t_δ` | `inf { t : J̄(t) ≥ J* − δ }` — **the primary target** (§6) |

---

## 3. Experimental system

### 3.1 Task and environment

| | |
|---|---|
| task | `PickCube-v1` (ManiSkill 3.0.1) |
| control mode | `pd_ee_delta_pose` |
| simulation backend | `physx_cuda` — **nondeterministic**; repeated evaluation of one checkpoint on one seed does not reproduce exactly |
| observation | 42-dim state vector |
| action | 7-dim (6 end-effector pose delta + 1 gripper) |
| episode horizon at evaluation | **160** control steps |
| success | `success_once`: cube center within **0.025 m** of the goal **and** robot static, at **any** step of the episode |

State layout used by all analysis code (0-indexed):
`[0:7]` arm joint positions, `[7:9]` gripper joints, `[18]` grasp flag,
`[19:22]` tool-center-point position, `[22:26]` TCP quaternion,
`[26:29]` goal position, `[29:32]` cube position, `[32:36]` cube quaternion.

**The 160-step horizon is load-bearing.** The task's registered horizon is 50,
which matched an earlier RL-generated demonstration bundle whose episodes
finished in 19–29 steps. The teleoperation-style demonstrator used here produces
38–110 step episodes, so a policy trained on it cannot finish inside 50 and
would score ≈0. `[MEASURED]` — this silently invalidated one full training
pipeline on 2026-07-21 before being caught.

### 3.2 Policy — Diffusion Policy (canonical)

Faithful to `real-stanford/diffusion_policy`, low-dimensional state variant.
Config: `configs/policies/state_diffusion_canonical.yaml`.

**Observation / action interface**

| | |
|---|---|
| observation history `H_o` | 2 frames |
| action prediction horizon `H_p` | 16 |
| action execution horizon `H_a` | 8 |
| window alignment | `diffusion_policy` — chunk aligned to observation start, execution begins at index `H_o − 1` |
| previous action in observation | no |

**Architecture**

| | |
|---|---|
| backbone | `conditional_unet_1d` |
| channels (down_dims) | `[256, 512, 1024]` |
| diffusion step-embedding dim | 256 |
| kernel size | 5 (GroupNorm `n_groups=8`) |
| conditioning | FiLM with `cond_predict_scale: true` (`out = scale·h + bias`) |

**Diffusion**

| | |
|---|---|
| training noise steps | 100 (DDPM) |
| beta schedule | cosine (`squaredcos_cap_v2`) |
| prediction type | epsilon |
| inference sampler | DDIM, **16** steps, `leading` spacing |
| temperature | 1.0 |

Inference sampler and step count are pinned in the checkpoint, the config, and
every evaluation artifact. Changing an inference parameter invalidates
**selection**, not merely the evaluation number.

**Optimization**

| | |
|---|---|
| optimizer | AdamW, betas `[0.95, 0.999]` |
| learning rate | 1e-4 |
| weight decay | 1e-6 |
| gradient clipping | 1.0 (global norm) |
| batch size | 256 |
| LR schedule | cosine, 500-step linear warmup, decaying to 0 at horizon `H` |
| EMA | diffusers power schedule: `power 0.75`, `inv_gamma 1.0`, `max 0.9999`, `min 0.0` |
| train/val split | **episode-level**, `val_fraction 0.1`, `split_seed 0` |

The split is by whole episode, not by window. Window-level splitting would place
overlapping windows from the same trajectory on both sides — they share up to
15 of 16 action frames — making validation loss optimistic and useless as a
signal. `[MEASURED]` — verified in `split_episodes`; the code is already correct
and an earlier claim in this document that it was not has been `[RETRACTED]`.

### 3.3 Demonstration generator

`scripts/generate_pickcube_teleop.py`. A scripted teleoperation-like policy with
injected human-plausible imperfection. Phases: `HOVER → DESCEND → CLOSE → LIFT →
CARRY/HOLD`.

Behavior worth stating, because both interact with dataset composition:

- **Alignment gate.** In `DESCEND` the hand holds its height while lateral error
  exceeds `align_tol`, then descends. This produces a hover segment.
- **Post-success hold.** The episode is truncated `hold` frames after success
  first registers. This controls how much "you have arrived, now stay" data
  exists.

**v1 generator arguments — the reference dataset for all results below**
(read from `outputs/pickcube/teleop_v1/teleop_100.h5.provenance.json`, not from
defaults):

```
align_tol 0.014      approach_bias 0.012   carry_cap 0.35     close_steps 4
descent_cap 0.3      grasp_gap 0.005       hold 4             hover_height 0.12
hover_tol 0.035      kp 6.0                kyaw 1.2           lift_clear 0.06
max_steps 110        n 100                 noise_alpha 0.85   noise_sigma 0.15
noise_taper 0.06     seed 0                yaw_cap 0.35       yaw_noise 0.03
```

Resulting pool: 100 episodes from 127 attempts, **6841 transitions**, episode
length 38–110 (mean 68).

**Every generator argument must be read from the provenance sidecar and passed
explicitly.** `[MEASURED]` — running the generator on its *defaults* silently
differed from v1 in `align_tol`, `noise_sigma`, `descent_cap` and `max_steps`,
turning an intended single-variable experiment into a six-variable one. See
§10.2.

### 3.4 Evaluation panels

A panel is a frozen, named bank of environment seeds. Panels are disjoint by
construction (different seed roots).

| panel | seed root | episodes | role | 95% CI half-width at J≈0.5 |
|---|---|---|---|---|
| `screening` | 2000 | 50 | per-snapshot curve during training | **±13.9 pt** |
| `confirmation` | 3500 | 200 | selects one checkpoint | ±6.9 pt |
| `final_test` | 1000 | 500 | **the quoted number** | ±4.4 pt |
| `diagnostic` | 20000 | 300 | recovery experiments, outside the frozen protocol | — |

The same panel seeds are used across all checkpoints within a run, so
checkpoint comparisons are **paired** on initial conditions.

---

## 4. The measurement object

We measure `J(t)`: closed-loop success as a function of gradient steps.

For this to be a single well-defined curve, **every candidate stopping point
must be a prefix of one training process.** A checkpoint at step 7500 must be
the same object whether the run was scheduled for 10,000 or 30,000 steps.

### 4.1 The horizon-dependent cosine breaks prefixes `[MEASURED]`

The original cosine schedule decayed over the *requested* budget:

```python
prog = (step - warmup) / max(1, total_steps - warmup)
return 0.5 * (1.0 + cos(pi * min(1.0, prog)))
```

Learning-rate multiplier at the **same gradient step** under different
scheduled horizons:

| step | horizon 10500 | horizon 15000 | ratio |
|---|---|---|---|
| 2000 | 0.946 | 0.974 | 1.03× |
| 5000 | 0.578 | 0.781 | 1.35× |
| **7500** | **0.206** | **0.527** | **2.56×** |
| 10000 | 0.006 | 0.266 | **43×** |

A longer run is therefore **not** an extension of a shorter one; its step-7500
checkpoint is a different policy trained under 2.6× the learning rate. Any
schedule that depends on the run horizon breaks the prefix requirement.

**What survives.** The three completed cells in §9.1 all ended at step **10500**
and therefore shared an identical schedule in step space. That result is not
corrupted.

### 4.2 Pinned cosine was tried and rejected `[MEASURED]` `[RETRACTED]`

The first fix pinned the cosine horizon to a fixed `H` independent of run length
(`lr_schedule_steps`, in `build_lr_scheduler`). This restores the prefix
property — every run follows the same curve — but introduces a **worse,
subtler** problem: it decays LR as a function of *absolute step*, and different
dataset sizes reach their optimum at different absolute steps. Each cell is
therefore measured at a **different point on the decay curve.**

LR multiplier at each cell's plausible optimum under `H = 30000`, for two guesses
of the size-scaling exponent `α` (`t* = 7500·(D/1652)^α`):

| cell | `D` | `t*` if α=0.21 | LR there | `t*` if α=0.5 | LR there |
|---|---|---|---|---|---|
| n=25 | 1652 | 7500 | **0.87** | 7500 | **0.87** |
| n=50 | 3294 | 8670 | 0.82 | 10591 | 0.74 |
| n=100 | 6841 | 10108 | 0.76 | 15262 | 0.50 |
| n=150 | 10300 | 11015 | 0.72 | 18727 | 0.32 |
| n=200 | 13700 | 11695 | **0.68** | 21598 | **0.19** |

**The larger the dataset, the lower the LR at its own optimum** — larger cells
are measured deeper into the anneal, and near the tail (α=0.5, n=200) they are
effectively frozen before they finish learning. The schedule throttles exactly
the cells whose optimum is latest, i.e. the large ones. Mechanism:

1. Decay is a function of absolute step `t`, fixed by `H`.
2. Larger `D` ⇒ optimum at larger `t` (the very law being measured).
3. Larger `t` ⇒ lower LR at the optimum.
4. A cell coasting at LR 0.2 cannot make the progress a cell at LR 0.9 can.
5. The design fails hardest exactly where the science is least certain: if `α`
   is larger than the censored 0.21 estimate (§9.2), large-`D` optima sit deep
   in the tail.

`lr_schedule_steps` remains in the code (it is the correct way to *deploy* an
annealed run at a chosen budget, §4.4) but is **not** used for the measurement
sweep.

### 4.3 The measurement schedule: constant LR `[PLANNED]`

The requirement is a schedule that is **horizon-independent** (prefixes stay
valid, §4.1) **and** **step-uniform** (no cell differentially throttled, §4.2).
Cosine-to-zero can satisfy the first but never the second. **Constant learning
rate satisfies both**, and is the schedule used for the sweep.

- Warmup is kept: a fixed 500-step linear ramp in *absolute* steps, identical
  for every run, so it is a genuine shared prefix.
- After warmup, LR is flat. Every checkpoint along one run is a true prefix, and
  every cell sits at the same LR (1.0) at its optimum.
- The measured object is explicitly the **constant-LR duration law**.

Constant LR is a labeled *variant* of the authoritative DP recipe (which
anneals). That is exactly what the two-stage protocol below is for. Every other
step-indexed mechanism already obeys the prefix rule: EMA is indexed by step
(power schedule), there is no augmentation and no run-length-dependent noise
schedule.

### 4.4 Two ways to measure the best duration `[PLANNED]`

"Best duration `T`" means: **train for `T` steps, stop, deploy — how good is the
resulting policy?** The deployed recipe anneals LR to zero at the end, and that
final settling phase is worth several points, so a policy *annealed over* `T`
steps differs from the checkpoint *sitting at* step `T` of a longer run. This
gives two measurement strategies.

**Approach 1 — one run, reuse snapshots (cheap; the sweep).** Train one long
run at constant LR; treat each saved checkpoint as the `T`-step policy.

```
n=25:  ONE run to H_max steps at constant LR
       evaluate snapshots: 5000 → 40%, 7500 → 47%, 10000 → 45%  →  best ≈ 7500
       cost: 1 run yields the whole curve
```

Valid only because constant LR makes the step-`T` checkpoint a faithful stand-in
for a `T`-length run. Measures an **un-annealed proxy** of the deployed policy.

**Approach 2 — one annealed run per duration (gold standard; spot-check only).**
For each candidate `T`, run a dedicated job whose cosine anneal ends exactly at
`T`, and take its **final** checkpoint.

```
n=25:  run to zero over  5000 → final 42%
       run to zero over  7500 → final 50%
       run to zero over 10000 → final 48%   →  best ≈ 7500
       cost: 1 run PER duration
```

Every measured policy is exactly what would ship for that budget. Deployment-
exact, no proxy — but ~40× the cost, because it cannot reuse snapshots (one full
run per duration instead of ~40 durations per run).

**How they combine.** Run Approach 1 at scale to map the curve and locate `t_δ`.
Run Approach 2 at **1–2 cells and 2–3 durations each** to check the annealed
final checkpoint peaks in the same place. If they agree, the cheap curve is
trusted everywhere; if they disagree, the proxy is broken and no shortcut
exists. **This transfer check is load-bearing** — it is what licenses using the
constant-LR curve in place of deployment-exact runs.

Deploy, in either case, by retraining with the cosine horizon set to the
selected `t_δ` (this is what `lr_schedule_steps` is for).

---

## 5. Protocol

### 5.1 Learning-rate schedule

**Measurement sweep (Approach 1):** `lr_scheduler: constant`,
`lr_warmup_steps: 500`, shared by every run (§4.3). Because the schedule is
horizon-independent, a cell may be run to any length and extended freely if
censored — there is no shared `H` to invalidate. `H_max` (the run length) is
chosen generously from the pilot (§9.4) only so the plateau is captured, not
because the schedule depends on it.

**Deployment / gold-standard check (Approach 2):** `lr_scheduler: cosine`,
`lr_warmup_steps: 500`, `lr_schedule_steps` set to the target budget so the
anneal ends exactly there. Used to train the shipped policy and, at 1–2 cells,
to verify the constant-LR optimum transfers (§4.4).

### 5.2 Snapshot schedule

`--snapshots K` sets the checkpoint interval to `T // K`, so every cell produces
the same number of points on its curve and incurs the same screening cost
regardless of dataset size. A fixed *step* interval would be coarse in epochs for
small `n` and fine for large `n`.

`[PLANNED]` Snapshots should be **denser in the transition region** than in the
plateau. Even spacing across a long horizon wastes resolution exactly where the
curve is flat. Proposed: logarithmic spacing, or two-stage — coarse pass, then
infill around the estimated `t_δ`.

### 5.3 Dataset chains — measuring the right uncertainty

**Today** `n = 25` means *the first 25 episodes of the pool*, always the same 25.
Multiple training seeds re-roll initialization and batch order but all see those
same 25 demonstrations. That measures training randomness, **not** the question
of interest: how much would the answer change if you had collected a *different*
25 demonstrations?

For small `n` the second effect is plausibly larger — those particular 25
episodes may happen to place the cube in easy positions.

`[PLANNED]` **3 randomized nested chains** from a 200-episode pool:

- Chain `c` fixes a random permutation `π_c` of the 200 episodes.
- `n`-subset of chain `c` = first `n` elements of `π_c`.
- Nesting holds **within** a chain (chain `c`'s 50 contains chain `c`'s 25), so
  the size axis is low-variance, while spread **across** chains estimates
  dataset-composition variance.
- 2 training seeds per (chain, size).

Requires regenerating the pool at `--n 200` with v1's exact arguments and
verifying the first 100 episodes reproduce **by hash** before any cells are
mixed.

### 5.4 External validation panel

`[PLANNED]` Generate ~50 demonstration episodes with the v1 generator and a
disjoint seed, used for training by **no** cell. Every cell computes its
validation signals on this same fixed set.

Fixes three problems with the current in-dataset 10% holdout:
1. At `n = 25` the holdout is ~2 episodes — far too noisy.
2. The holdout is a *different* set for every cell, so validation numbers are
   not comparable across `n`.
3. It costs training data from already-small datasets.

The external panel is not counted in `D`.

### 5.5 Signal battery

`[PLANNED]` Recorded **at every snapshot** (currently validation is computed on a
separate `eval_every: 1000` cadence that does not align with snapshots):

| signal | definition |
|---|---|
| train loss | the diffusion objective on training batches (**already recorded**) |
| validation loss | the same objective on the external panel (**already recorded, but on the in-dataset holdout at a misaligned cadence**) |
| held-out action error | ‖predicted action − demonstrated action‖ on the external panel, in metres and radians — interpretable, unlike the denoising loss |
| per-noise-level loss | the objective bucketed by diffusion timestep; low-noise buckets may matter more for control than the average |
| EMA-vs-raw divergence | ‖θ_EMA − θ‖ / ‖θ‖ — are the weights still moving |
| gradient norm | pre-clip global norm |
| update/weight ratio | ‖Δθ‖ / ‖θ‖ |
| LR multiplier | schedule position, for the record |
| closed-loop screening success | the expensive target being predicted |

---

## 6. Target definition

`[PLANNED]` The primary target is **not** the argmax of the success curve and
**not** a fraction of a fitted asymptote. It is a decision-theoretic quantity:

```
t_δ = inf { t : J̄(t) ≥ J* − δ }        J* = sup_u J̄(u)
```

in words: **the earliest checkpoint whose expected success is within `δ`
percentage points of the best attainable.** Report `δ ∈ {2, 5}`.

**Why not argmax.** `[MEASURED]` On the v1 `n=25` run, screening's argmax was
step 10500 (44%) while the 200-episode confirmation selected step 7500 — whose
screening score was 36%, one of the *worse* snapshots. With ±13.9-point screening
noise over a plateau spanning ~10 points, argmax over 21–40 snapshots is close to
a draw from the noise.

**Why not `t_95` of a fitted asymptote.** A monotone saturating fit cannot
represent genuine late degradation; the asymptote is poorly identified under
censoring; and "95% of raw success" behaves badly when `J(0) ≠ 0`. A relative
threshold also tolerates very different *absolute* regret on easy vs hard tasks,
and absolute regret is the operational quantity.

**`J̄` averages over** dataset draw (chain), training seed, and evaluation
episode sampling. Which of these is included must be stated with every reported
`t_δ`.

---

## 7. Statistical model

`[PLANNED]` A **joint hierarchical model**, not a two-stage fit-then-regress.

For checkpoint `k` in size cell `d`, family `f`, dataset chain `r`, seed `s`,
with `Y` successes out of `m` rollouts:

```
Y_{d,f,r,s,k} ~ BetaBinomial(m, p_{d,f,r,s}(t_k), φ)

log τ_{d,f,r,s} = a_f + α_f · log D_d + u_r + u_s
u_r ~ N(0, σ²_chain)        u_s ~ N(0, σ²_seed)
```

where `p(·)` is a saturating curve in `t` with time constant `τ`.

Rationale:
- A binomial/beta-binomial likelihood uses the **actual rollout counts**;
  least squares on success *fractions* discards them and mis-weights cells.
- Beta-binomial absorbs overdispersion from heterogeneous initial conditions.
- Curve-fitting uncertainty **propagates** into the scaling parameters instead
  of being discarded when a point estimate of `t_δ` is passed to a second stage.
- `σ²_chain` vs `σ²_seed` **separates dataset-composition variance from training
  variance** — the distinction §5.3 exists to measure.
- Censored cells are handled explicitly rather than silently.

**Do not pool seeds or chains before fitting.** Pooling erases the hierarchy and
produces overconfident intervals. Pooled curves may be *plotted*; the model keeps
the levels.

### 7.1 Model comparison — do not presuppose a power law

Fit and compare by **out-of-sample predictive error**, not in-sample fit:

| model | form |
|---|---|
| constant | `t_δ = A` |
| logarithmic | `t_δ = A + C·log D` |
| power law | `t_δ = A·D^α` |
| family-specific variants | `A_f`, `α_f` |

`[HYPOTHESIS]` If `α ≈ 0` and the constant-step model predicts as well, the
honest conclusion is *"a fixed optimizer-step budget is a strong default and a
fixed epoch budget systematically misallocates optimization"* — still useful, and
not to be dressed up as a power law.

### 7.2 On `N^γ` — removed `[RETRACTED]`

An earlier version of this document proposed `t = A_f · D^α · N^γ`. **This is
not identifiable** under the planned design: with one canonical model size per
family, `A_f · N_f^γ` is just another family-specific constant, and no amount of
dataset-size variation separates them.

Model size is therefore **removed** from the first study, which is explicitly
scoped to canonical fixed-size implementations. Varying width/depth within a
family is a separate study.

### 7.3 On the family constant `A_f`

`[HYPOTHESIS]` The sharp claim is that **`α` is shared across families and only
`A_f` differs**. If `α` differs by family that is a more interesting result.

`A_f` must **not** be interpreted as "learning signal extracted per step". It
absorbs optimizer, schedule, normalization, EMA settings, depth, action-horizon
parameterization, timestep sampling, loss scaling, and implementation quality. It
is an empirical family-and-recipe constant. Descriptive scaling is not mechanism.

### 7.4 Batch size

`E = t·B/D` is exact by definition, but `B` may also shift `t_δ` **in steps** by
changing gradient noise and updates-per-example. `[PLANNED]` ablate
`B ∈ {64, 128, 256, 512}` at 1–2 dataset sizes and test which unit aligns
saturation best: gradient steps, examples processed, epochs, or FLOPs. Until then
"steps are the invariant" is a hypothesis, not a result.

---

## 8. Design

### Phase 0 — pilot
Determine `H`: run the smallest and largest cells long enough to see the curve
flatten, under the **pinned** schedule.

### Phase 1 — reference layer (Diffusion Policy, PickCube)
| axis | values |
|---|---|
| `n` | 10, 25, 50, 75, 100, 125, 150, 200 |
| dataset chains | 3 |
| training seeds | 2 per (chain, size) |
| runs | **48** |
| snapshots | ~40 per run, denser in the transition region |

### Phase 2 — policy families
ACT and rectified flow, same grid, fewer chains/seeds. Question: is the *shape*
shared?

### Phase 3 — tasks and multi-task subsets
Multiple tasks; policies trained on subsets of tasks. Tests whether `D`
aggregating heterogeneous data still works and whether a difficulty term is
needed.

### 8.1 Extrapolation, not interpolation
`[PLANNED]` Fit on `n ≤ 100`; **predict `n = 125, 150, 200`.** Holding out
interior points tests only that the curve is smooth. Then fit on some tasks and
predict a held-out task.

### 8.2 Data-quantity ablation
`[PLANNED]` `D` and `M` are confounded in the natural pool. Break them:
(a) more episodes at ~fixed total windows; (b) more windows per episode at fixed
episode count. Without this, the defensible claim is conditional: *within a fixed
task, generator, and windowing procedure*, saturation scales with training
windows.

---

## 9. Results so far

All `[MEASURED]`, all single-seed, single-chain.

### 9.1 Dataset size (final test, 500 episodes)

| `n` | `D` | `S_ep` | selected step | selected epoch | confirmation | **final test** |
|---|---|---|---|---|---|---|
| 10 | 731 | 2 | — | — | — | screening 2–4% |
| 25 | 1652 | 5 | 7500 | 1500 | 48.0% | **47.0%** [42.7, 51.4] |
| 50 | 3294 | 10 | 8500 | 850 | 65.5% | **60.0%** [55.6, 64.2] |
| 100 | 6841 | ~21 | 10000 | ~476 | 82.0% | not run |

All four ended at step 10500 → identical LR schedule → mutually comparable.

### 9.2 Steps vs epochs

Across a **4× increase in data**: `t*` in steps moves 7500 → 10000 (**1.33×**);
in epochs 1500 → 476 (**0.32×**). Power-law fit to three points:
`t*_steps ∝ D^0.21`, `t*_epochs ∝ D^−0.79`.

**Caveats.** All three runs were capped at 10500 steps and `n=100`'s optimum sits
at 10000 — **right-censored**, so `α` is a lower bound. Single seed, single
chain, selection by 200-episode confirmation over a handful of candidates.

### 9.3 The 500-epoch default starves small datasets

| cell | best ≤ 500 epochs | best > 500 epochs |
|---|---|---|
| 25 | 28% | **42%** (+14) |
| 50 | 58% | **66%** (+8) |
| 100 | **82%, reached by epoch 500** | — |

Mechanism: `S_ep` grows linearly with `D`, so at fixed epochs `n=100` receives
~4× the updates of `n=25`.

### 9.4 Pilot: `n=25` to 3000 epochs

Run under the **old unpinned** schedule (15000-step horizon), so its curve is not
protocol-final; retained only as an indication of where saturation lies.

| step | epoch | train loss | val loss | success |
|---|---|---|---|---|
| 1000 | 200 | 0.0538 | **0.1622** ← min | 16% |
| 3000 | 600 | 0.0200 | 0.2817 | 28% |
| 6000 | 1200 | 0.0067 | 0.3421 | 40% |
| 8000 | 1600 | 0.0053 | 0.3737 | **46%** |
| 11000 | 2200 | 0.0017 | **0.4210** | 42% |

### 9.5 Offline signals do not predict the closed-loop optimum

**Training loss saturates ~700 epochs early.** It falls 0.988 → 0.0098 and is
flat by epoch ~825 while success is still less than 60% of its eventual peak.

**Validation loss is anti-correlated.** It rises monotonically 0.162 → 0.421
(2.6× worse) while success rises 16% → 46%. **Minimum-validation-loss early
stopping selects epoch 200 and a 16% policy, against a 46% optimum — a ~30-point
regret.**

This reproduces, in simulation and with a diffusion policy, the qualitative
observation from robomimic that validation loss is a poor checkpoint selector in
offline robot learning. The contribution here is the **quantified regret** under
a controlled protocol, and the open question of whether *any* cheap signal does
better (§5.5).

**A live trap in the codebase:** `best_ema.pt` is saved whenever validation loss
improves, so it currently holds the **step-1000, 16%** checkpoint. No reported
result uses it — selection is by closed-loop screening — but it is a loaded gun
for anyone who assumes "best" means best.

---

## 10. Hazards, each already encountered

| hazard | how it bit | mitigation |
|---|---|---|
| epochs vs steps | §1.1 | budget in either, report both |
| **schedule circularity** | §4.1, 43× LR ratio at step 10000 | horizon-independent schedule (§4.3) |
| **schedule throttles large cells** | pinned cosine decays LR at absolute steps, so late-optimum (large-`D`) cells are frozen near the tail (§4.2) | **constant LR** for the sweep; anneal only to deploy |
| right-censoring | `n=100` optimum at the budget edge (§9.2) | constant LR ⇒ any cell extends freely; require the optimum strictly interior; report censored cells |
| screening noise | argmax ≠ confirmation winner (§6) | `t_δ`, hierarchical fit, paired panel seeds |
| single seed / single chain | all §9 results | 3 chains × 2 seeds |
| **dataset identity drift** | generator defaults differed from v1 in 4 arguments, making a 1-variable test a 6-variable one | read provenance, pass args explicitly, verify subsets by hash |
| unintended carry-over | a lateral-noise-taper helper added for one variant kept running in the next, unnoticed | diff the recorded args of every pool before comparing |
| snapshot cadence | fixed step interval is coarse in epochs for small `n` | `--snapshots K` |
| selection ≠ evaluation | confirmation is optimistically biased | quote the 500-episode final test on a disjoint panel |
| nondeterminism | `physx_cuda` | never assume two runs derive the same episode outcomes |

### 10.1 `[RETRACTED]` claims previously made in this document
1. "Validation loss is never computed." **False** — computed every 1000 steps
   and logged as `val/loss`.
2. "The train/val split is window-level and leaks." **False** — it is
   episode-level via `split_episodes`.
3. "`t = A·D^α·N^γ`" — `N^γ` is unidentifiable and has been removed (§7.2).
4. "The v2 experiment shows the alignment gate does not help." **False** — that
   comparison differed in six variables, not one (§10.2).
5. "Pinned cosine (`lr_schedule_steps`) is the schedule fix." **Superseded** —
   pinned cosine keeps prefixes valid but throttles large-`D` cells at their
   optima (§4.2). The measurement schedule is **constant LR** (§4.3); pinned
   cosine is retained only for deployment and the transfer check (§4.4).

### 10.2 The v2 incident
An intended two-line change to the demonstrator was run on generator
**defaults**, which differed from the recorded v1 arguments in `align_tol`
(0.014→0.006), `noise_sigma` (0.15→0.05), `descent_cap` (0.3→0.16) and
`max_steps` (110→90). The resulting 44.2% measures a six-variable bundle. The
alignment gate remains untested.

---

## 11. Baselines and reporting

`[PLANNED]` Every stopping method is compared against:

1. fixed 500 epochs
2. fixed 10500 gradient steps
3. fixed examples processed (`B·t`)
4. final checkpoint of a large safe budget
5. minimum validation loss
6. training-loss convergence
7. plateau detector on screening success
8. oracle `t_δ` from the full curve

Reported for each:

- **closed-loop success regret** (points below the run's oracle-best) — the
  primary metric, not correlation
- training compute used
- **number of closed-loop rollouts consumed** — a method needing as many
  rollouts as exhaustive screening is not useful even if its pick is good
- fraction of runs stopped before the near-optimal region
- fraction of predicted budgets that were right-censored

`[PLANNED]` **Adaptive rollout allocation:** evaluate all checkpoints on a small
common panel, spend additional rollouts near the credible optimum, discard
dominated checkpoints, touch the final disjoint panel once. Phase 1 alone implies
~48 runs × 40 snapshots × 50 rollouts ≈ **96,000 screening rollouts** before
confirmation and final testing, so this materially changes feasibility.

---

## 12. What the paper claims

Not *"a training-duration law"* until a stable law is demonstrated across tasks.
Working framing: **closed-loop-aware training budgets and checkpoint selection
for robot imitation learning.**

1. Common budgeting conventions are not equivalent and can reverse conclusions.
2. Closed-loop saturation is far more stable in optimizer steps than in epochs.
3. A hierarchical model predicts a near-optimal budget for **unseen** dataset
   sizes.
4. Offline signals are evaluated by **checkpoint-selection regret**; validation
   loss is measurably harmful.
5. A combined static budget plus dynamic stopping rule reaches near-oracle
   success with substantially less training and rollout cost.

**Publication assessment.** PickCube + one policy family alone is a workshop
note — dismissible as an artifact of unit conversion. Multiple chains + several
tasks + three families supports a conference paper if the rule predicts unseen
sizes. A cheap stopping signal with low closed-loop regret is the strongest
version, because closed-loop evaluation is a primary bottleneck in robot
learning.

---

## 13. Open questions

1. Is `t*` in steps genuinely near-invariant, or does `α = 0.21` grow once the
   `n=100` censoring is removed?
2. Does **any** cheap signal predict the closed-loop optimum? Training loss and
   validation loss both fail (§9.5). Held-out action error and per-noise-level
   loss are untested.
3. Does the pinned-schedule optimum transfer to an annealed run (§4.2)?
4. Is the invariant gradient steps, examples processed, or compute (§7.4)?
5. For multi-task training, is `D` the summed transitions, or is a
   task-diversity term needed?
6. How large is dataset-composition variance relative to training variance at
   `n = 25`? Nothing currently measures it.

---

## 14. Related work

- **Kaplan et al., 2020**, *Scaling Laws for Neural Language Models*.
- **Hoffmann et al., 2022** (Chinchilla), *Training Compute-Optimal LLMs* —
  the structural template: fit a parametric surface, validate by prediction.
- **Muennighoff et al., 2023**, *Scaling Data-Constrained Language Models* —
  closest analogue: multi-epoch training on a fixed limited dataset and the
  decaying value of repeated passes.
- **Mandlekar et al., 2021** (robomimic), *What Matters in Learning from Offline
  Human Demonstrations* — documents that checkpoint selection and stopping
  criteria dominate outcomes in offline robot learning. §9.5 reproduces this
  qualitatively and quantifies the regret.
- **Lin, Hu, Sheng, Wen, You, Gao, 2024**, *Data Scaling Laws in Imitation
  Learning for Robotic Manipulation*, arXiv:2410.18647, ICLR 2025 — **verified
  by search**. Generalization vs number of environments, objects and
  demonstrations; >40,000 demonstrations, >15,000 real rollouts. Finding:
  environment/object **diversity** dominates, demonstrations-per-environment
  saturates. **Orthogonal axis** — data composition vs training duration — and
  complementary, since any such study must implicitly fix a duration policy.
- **Open X-Embodiment / RT-X, Octo, OpenVLA** — empirical data and model
  scaling, not closed-form duration rules.

Apart from the Lin et al. entry, these are from model knowledge with a January
2026 cutoff and **have not been verified**. A systematic review is required
before claiming novelty.

---

## 15. Plain language

Training a robot policy from demonstrations requires choosing how long to train,
and that choice is usually made by convention rather than measured. It matters
more than it looks: counting in epochs gives small datasets far fewer weight
updates than large ones, because a pass over a small dataset is short. We measured
this — 500 epochs was plenty for 100 demonstrations and clearly too few for 25.
Counting weight updates instead, the best stopping point barely moves as the
dataset grows: about 7,500 updates for 25 demonstrations and 10,000 for 100, even
though those are 1,500 and 476 epochs respectively.

The obvious way to avoid rollouts entirely would be to watch the loss on
held-out demonstrations and stop when it stops improving. We measured that too,
and it fails badly: held-out loss gets steadily *worse* while the robot gets
steadily *better*, so stopping at its minimum would hand you a policy that
succeeds 16% of the time instead of 46%.

The plan is to measure the best stopping point across eight dataset sizes, three
different random draws of which demonstrations you happen to have, and two
training seeds each; fit a model; and test it by predicting dataset sizes
deliberately excluded from the fit. Alongside that, record a battery of cheap
signals during training and ask whether *any* of them can tell you when to stop
without running the robot. The hard part is not the fitting — it is that robot
success rates are noisy enough that the apparently best checkpoint is often just
the luckiest one, so the protocol fits a curve to the whole run rather than
trusting any single measurement, and defines success in terms of how many points
you give up rather than where a peak appears to sit.

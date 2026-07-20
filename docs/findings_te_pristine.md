# Findings — Temporal Ensembling on Pristine Implementations + Action-Representation A/B

**2026-07-18. Development tier**: diagnostic panel, n=300 paired episodes,
single policy seed, PushT-v1 (`active_min` n=400), chunk 16, 20 Hz. NOT
frozen-protocol claims. Policies are the pristine baseline checkpoints
(faithful DETR ACT w/ fixed init 45.6%; re-selected DP 51.2%; flow 52.6%).
Supersedes the lightweight-policy TE study (`findings_temporal_replan.md`)
wherever they overlap. Run-to-run sim noise: `physx_cuda` repeat agreement
~96%/episode (sparse replan) and ~81% (dense) — all effects below are far
larger and within-run paired (McNemar).

## 1. Delta-EE space: the exact ACT-TE recipe inverts on all three families

Execution-scheme matrix (`rerun/{act,dp,flow}_te_replication`), exact ACT
temporal ensembling = query every step + oldest-weighted near-uniform
`w_i = exp(-0.01 i)` average; openloop = full executable chunk
(alignment-derived H_a: 16 for ACT, 15 for DP/flow):

| family | te (exact) | openloop | h8 (benchmark) | latest (dense) | te − openloop |
|---|---|---|---|---|---|
| ACT  | 16.3% | 25.0% | 41.3% | **48.0%** | **−8.7*** p=0.003 |
| Flow | 26.0% | 36.3% | 48.7% | **57.0%** | **−10.3*** p<0.001 |
| DP   | 29.0% | 41.7% | 51.0% | **55.7%** | **−12.7*** p<0.001 |

Aggregation variants vs `latest` (compute-matched, `rerun/*_te_variants`):

| family | mean | projection | medoid |
|---|---|---|---|
| ACT  | −20.3* | −29.0* | −14.0* |
| DP   | −10.7* | −13.0* | −4.7* |
| Flow | −12.0* | −13.0* | −4.0 (p=.065) |

Decay sweep (ACT): monotone 18.3% (uniform) → 47.0% (≈latest).

**Three regularities (delta space):**
1. The ACT paper's comparison (TE > full open-loop) **inverts with
   significance on every family**.
2. The H_a spectrum is monotone: openloop < h8 < latest — dense replanning
   without averaging wins everywhere (+7.0*/+8.7* over h8 for DP/flow, +6.7
   for ACT).
3. Family-dependent robustness, twice: open-loop chunk survival ranks
   DP > flow > ACT, and ensembling damage is ~2× worse for deterministic ACT
   than the generative samplers; medoid ≈ standalone for DP/flow but clearly
   harmful for ACT. Medoid hurting at all ⇒ the cost is **staleness**, not
   only mode-averaging.

## 2. Action-representation A/B (the "averaging deltas vs averaging targets" test)

Hypothesis tested: averaging **deltas** is destructive (opposing directions
cancel; Jensen norm-shrinkage integrates into position lag) while averaging
**absolute targets** (ACT's native space) is a benign low-pass — so TE's sign
might be representation-dependent. Counter-position (YS): mode-averaging is
parameterization-independent; it should hurt in both spaces.

**Dataset**: `data/active_min/subset_0400_jointpos.h5` — the SAME 400
episodes, states bit-identical, actions re-derived as absolute joint targets
by **glued drive-target capture** (teleport to each recorded state, one
dynamic step under the source `pd_ee_delta_pose` controller, read its
IK-computed `_target_qpos`; valid because `use_target=False` makes the target
a pointwise function of (qpos, action)). Two simpler conversions fail
informatively: naive `a_t = qpos_{t+1}` → 0% replay (PD chases targets;
~0.1 rad systematic lag); free open-loop capture → correct targets (0.0004
rad) but only 138/400 episodes survive GPU-PhysX open-loop replay.
Validation: one-step teleport tracking err mean 0.006 rad; open-loop
joint-target replay 52.5% (vs 34.5% for the source deltas — absolute targets
don't compound). ACT retrained on it via the standard pipeline: **500-ep
final 37.8%** [33.7, 42.1] (sel. ep 2700, 2.5 ms/query).

**Result** (`rerun_jointpos/act_te_replication`, n=300):

| scheme | ACT delta-EE | ACT jointpos |
|---|---|---|
| te (exact) | 16.3% | 14.7% |
| openloop | 25.0% | 21.0% |
| h8 | 41.3% | **36.7%** ← best |
| latest | **48.0%** ← best | 25.3% |
| te − openloop | −8.7* | **−6.3*** (p=0.023) |
| latest − h8 | +6.7* | **−11.4*** |
| latest − openloop | +23.0* | +4.3 (ns) |

**Verdict 1 — the averaging hypothesis is dead.** TE inverts in the native
absolute-joint space too. Averaging overlapping predictions is harmful in
both parameterizations (the counter-position was right): on this task the
dominant costs (staleness + mode conflict) are representation-independent.

**Verdict 2 — the *replanning* lever is representation-dependent (new
finding), mechanism MEASURED.** In delta space the H_a spectrum is monotone
toward dense; in absolute-target space it is NOT: h8 beats latest by 11.4
pts. Diagnostic (15 eps/arm, per-step joint displacement + command lead
`||a_t − qpos_t||`):

| | speed (rad/step) | executed lead (rad) | success |
|---|---|---|---|
| demos (captured targets) | 0.123 | 0.324 | — |
| latest | 0.053 (43%) | 0.148 (46%) | 3/15 |
| h8 | 0.062 (50%) | 0.167 (51%) | 9/15 |

**The mechanism, step by step.**

*The physical fact:* the low-level controller is a spring between the
commanded target and the actual position — force ∝ stretch (the ACT paper
says this itself: *"the amount of force applied is implicitly defined by the
difference between them, through the low-level PID controller"* — their
reason for recording LEADER, not follower, positions). A robot only moves if
the command is ahead of it, and only *presses* if the command is planted
beyond the resistance. (Our naive conversion — labels = achieved next
positions, the follower-positions mistake — shrank the stretch to ~⅓ and
reproduced the demo's contact slowdowns inside the command: 0% replay. The
achieved trajectory records the *outcome* of force, not the intent.)

*What a fresh command means in each space.* Freshness bundles two things:
new information (always good, both spaces) and RE-ANCHORING of the command
to the latest state. Delta: a command says "move BY δ" — the force rides
inside the command, the anchor is irrelevant, freshness is pure gain →
replan every step. Absolute: a command says "go TO X," where the policy
places X a small learned lead ahead of the position it just observed — the
force lives in the *stretch*, and every replan RESETS the stretch to that
constant lead. At a friction stall: "I'm still at q" → "go to q+0.15" →
insufficient force → still at q → same command, forever. The system never
gets angrier: no integrator.

*Why chunking rescues absolute targets:* the chunk is a pre-committed target
sequence marching on a fixed schedule, indifferent to the robot's actual
position. Stuck robot + marching targets = stretch grows step by step =
force ramps until friction breaks. Open-loop marching of absolute targets IS
integral action (wind-up); per-step replanning deletes exactly that term.

*The resulting geometry:* absolute space has a freshness↔wind-up TRADEOFF →
interior optimum: openloop 21.0 (all wind-up, no freshness) < latest 25.3
(all freshness, no wind-up — reactivity is worth something, but it cannot pay
for losing the integrator) < h8 36.7 (some of each). Delta space has no
tension (commands carry their own force) → corner optimum H_a=1. One line: **delta actions carry force in the command;
absolute actions carry it in the stretch — replanning refreshes the command
but resets the stretch.**

*The causal chain (twice corrected under review; measurements 2026-07-19).*
Facts, in order of establishment:
- Labels are CORRECT: glued capture stores true drive targets; dataset lead
  0.334 rad (full intent).
- Conditional lead heterogeneity is MINOR: neighbor-lead cancellation ratio
  0.87/0.82/0.77 at k=5/10/25 — local averaging explains ≤23% shrinkage,
  not the observed ~50%. (Kills "mode-averaging shrinks the lead" as
  primary.)
- **The network learned the press: on DEMO states, predicted lead 0.370 vs
  label 0.347 (ratio 1.07).** (Kills the loss-geometry/"L1 barely sees the
  residual" hypothesis outright — recorded here because it was previously
  asserted in this section.)
- The halved lead (0.148–0.167) appears ONLY on closed-loop rollout states.

**Observation-corrected mechanism (2026-07-19, videos + per-episode
traces; two earlier stories refuted and kept on record).**
- REFUTED: "anchor error off-manifold eats the lead" — the failing `latest`
  episodes run at FULL lead (0.316/0.339, ~zero stall steps). Also the
  earlier "halved executed lead (0.148)" measurement was
  **phase-contaminated**: post-success HOLDING steps (near-zero lead)
  dilute episode means; per-phase leads are full. (YS's symmetry objection
  — "consistent undershoot sounds like a bug" — was correct.)
- REFUTED (earlier): loss-geometry shrinkage (on-demo lead ratio 1.07) and
  primary heterogeneity-averaging (cancellation only 0.77–0.87).
- Secondary but real: **velocity conditioning** — zeroing qvel in the obs
  drops the absolute policy's predicted lead 24% (0.353→0.269) vs only 9%
  for the delta policy (1.523→1.385): the absolute map leans on the
  velocity channel ~3× harder.
- **Current account, backed by direct observation (videos,
  `rerun_jointpos/videos/`): PLAN CHURN AT ABSOLUTE-TARGET AMPLITUDE.**
  `latest` executes chunk[0] of a different plan every step; successive
  plans disagree modestly in direction; in absolute space each command is a
  full-lead-sized spring pull, so modest directional churn becomes violent,
  thrashy motion that over-manipulates the block off the goal (frames: block
  shoved past and rotated ~90° off target; contorted arm postures). `h8`
  executes one coherent plan per chunk — smooth, consistent pushes. Delta is
  benign under the same churn because its commands are STEP-sized: churn
  amplitude scales with command magnitude. DP-paper reconciliation: Chi et
  al. report position control (absolute) beating velocity control — but DP
  always executes chunked (Ta=8) and never runs dense-latest; our absolute
  cell is also fine at h8. No contradiction; the collapse lives in a cell
  the paper never visits.
Follow-ups (queued, with predictions under the churn account): (a)
**delta-joints** (= per-step residual; YS): latest fine (step-sized churn);
(b) **chunk-anchored residuals** (stencil stamped at the true pose once per
chunk): churn-free within chunks + anchor-correct at replans; (c)
**`pd_ee_pose`**: h8 fine (DP-paper-consistent), latest degraded (churn) —
if latest is fine there too, the churn account needs the coordinates axis.

**Scope**: within-representation orderings are the comparison; absolute
LEVELS are confounded (the jointpos policy is weaker overall, 37.8 vs 45.6
final — single seed, possibly the raw-radian action manifold or the
near-identity qpos→target shortcut; do not read cross-representation level
differences as a representation ranking).

## 3. Consequences

- **Execution scheme × action representation interact strongly.** "Dense
  replanning is the universal lever" (the lightweight-era headline) is a
  *delta-space* result. Any benchmark that fixes one execution scheme across
  representations mis-ranks methods; the schedule (H_a) must be tuned per
  representation — cheap, eval-only, and now demonstrably necessary.
- **The ACT paper's TE claim does not transfer to PushT/chunk-16 in either
  representation.** Still NOT a refutation of the paper (ALOHA, chunk 100,
  50 Hz, real sensor noise untested here); the honest statement is a scope
  one, now much stronger: two representations, three families, exact recipe.
- **A latency-rule nuance**: dense replanning being *feasible* under the
  no-pause contract (ACT 2.2–2.5 ms ≪ 50 ms) does not make it *desirable* —
  for absolute-target policies it is actively harmful. Feasibility and
  benefit separate.
- Retrieval/verifier-style selection over TE variants remains untested at
  the new operating points (Track A4/D1 follow-ups).

## 4. Artifacts

- Delta-space: `outputs/active_min/rerun/{act,dp,flow}_te_replication/`,
  `.../{act,dp,flow}_te_variants/`, `.../act_te_decay/`.
- Jointpos: `data/active_min/subset_0400_jointpos.h5` (+`.validation.json`),
  `outputs/active_min/rerun_jointpos/act/`,
  `outputs/active_min/rerun_jointpos/act_te_replication/`.
- Converter: `scripts/convert_action_representation.py` (glued
  drive-target capture; the two failed conversion methods documented in its
  docstring).
- Sim-nondeterminism measurement: repeat-eval agreement, recorded in
  `docs/recovery_scheme_and_oracle.md` §13 (A4-revised).

## 5. Follow-ups

1. ~~Rollout speed diagnostic~~ — DONE (table in §2): lead shrinkage ~2×
   confirmed; both schemes half-speed; success gap driven by force wind-up.
2. DP/flow retrains on jointpos (does the h8>latest flip hold for
   generative families?). [RUNNING 2026-07-19]
3. Multi-seed + final_test panels before any reported claim.
4. `pd_ee_pose` (absolute EE) — the third widely-used representation.

## 6. H_a curve, delta-EE ACT (2026-07-19): monotonicity refuted, top is flat

User-prompted check of "denser is always better." Fresh within-run sweep
(one launch, four arms, n=300 each, same panel, pristine delta ACT;
`outputs/active_min/rerun/act_horizon_sweep/`):

| H_a | replan Hz | success | vs h8 |
|---|---|---|---|
| 8 | 2.5 | 41.3% | — |
| 4 | 5 | 48.0% | +6.7* |
| 2 | 10 | **51.0%** | +9.7* (p=.001) |
| 1 | 20 | 47.0% | +5.7* (p=.046) |

Pairwise: h2−h1 +4.0 (p=.073, trend); h2−h4 and h4−h1 ns. Verdict: the
lever is "replan at ≥5 Hz", NOT "H_a=1"; the top is statistically flat
across h1–h4 with a nominal interior peak at h2. Cross-run stability: h8
reproduced exactly (41.3/41.3); h1 47.0 vs prior latest 48.0 (within the
characterized dense-replan repeat noise). The lightweight-DP sweep's top
was also nearly flat (h1 53.3 vs h2 52.0) — consistent saturation.
Deployability: the no-pause latency contract is satisfiable at h2–h4 at
zero measured cost (relevant for DP's 40 ms/query, which cannot honestly
run 20 Hz).

# Why Dense Replanning Fails for Absolute Joint Targets (and Wins for Delta-EE)

**2026-07-19. Development tier** (single policy seed, diagnostic panels), but
every claim below is backed by a named measurement, the pipeline is
bug-audited, and the account has now survived one round of **pre-registered
falsification — which it partly FAILED**: the velocity-shortcut *cause* was
refuted by the qvel-mask retrain (§5), while the trap phenomenon itself
replicated exactly, velocity-free. This document records both. Companion to
`findings_te_pristine.md`.

## 0. One-line answer (after both falsifiers)

At a family of "hard" states the absolute-joint policy **stops reading the
scene** (53× reduced response to block perturbations, §6.1) and its output
falls back to **echoing its own proprioception** ("stay at q") — a fallback
that survives removing the velocity input (§5) AND removing every stay-label
from training (§6.2), so it is model-structural, not imitated. Executing
"stay" freezes the plant, which freezes the policy's input, which — under
dense replanning (`latest`) — reproduces "stay" forever (median dwell ~69
steps); chunked execution (`h8`) escapes via the plan's own ramp (dwell
1–3). Delta-EE is immune twice: its scene-free fallback is a *moving*
command (nothing in its input equals its labels, so there is nothing to
copy), and small delta commands don't freeze the plant — no fixed point
exists. The representation determines the collapse target; the execution
scheme determines whether that target absorbs.

## 1. The phenomenon

Same architecture, same demos, same task; only action representation and
execution scheme vary (n=300 paired; probe replications at n=30):

| | latest (H_a=1) | h8 (H_a=8) | winner |
|---|---|---|---|
| delta-EE | **48.0%** | 41.3% | latest (+6.7) |
| delta-EE, **qvel-masked** | **48.7%** | 42.3% | latest (+6.4) |
| absolute-joint | 25.3% | **36.7%** | h8 (+11.4) |
| absolute-joint, **qvel-masked** | 21.0% | **35.0%** | h8 (+14.0) |

The masked rows complete the falsifier 2×2 (§5): removing the velocity
input entirely neither rescues absolute `latest` nor harms delta `latest` —
the latest-vs-h8 structure is locked to the representation, indifferent to
the velocity channel in both directions.

## 2. Bug audit (all clean)

| candidate bug | test | result |
|---|---|---|
| latest executes something other than chunk[0] | exact comparison, every step, deterministic policy | max diff **0.0** |
| policy nondeterminism | same obs, different generator seeds | max diff **0.0** |
| action normalization | label round-trip through the checkpoint normalizer | max err **2.4e-7** |
| clipping at (asymmetric) joint bounds | clip-hit fraction over all rollout steps | **0.0** |
| wrong labels in the converted dataset | one-step teleport tracking of captured drive targets | 0.006 rad mean |
| network failed to fit labels | predicted lead on demo states vs labels | ratio **1.07** |
| obs-mask leak (masked run) | qvel-scramble through checkpoint at inference | max diff **0.0** |

## 3. Theory graveyard (each died by measurement; newest last)

| theory | killed by |
|---|---|
| "PD-lag chasing": commanding achieved positions | that was the *naive conversion* (0% replay) — the trained dataset uses true drive targets (lead 0.334) |
| "loss geometry": L1 can't see the small lead residual | on-demo prediction ratio **1.07** — the network learned the lead |
| "anchor erosion off-manifold" | failing episodes run at FULL lead; random anchor error would *increase* norms (YS's symmetry objection) |
| "heterogeneity averaging" | neighbor-lead cancellation only 0.77–0.87 — explains ≤23% |
| "halved lead everywhere" (the measurement itself) | phase contamination: post-success HOLD steps diluted episode means |
| "plan churn thrash" as primary | real (videos) but the median failure mode is near-zero-lead parking, 69% of steps |
| **"velocity shortcut": network reads qvel, stationary ⇒ stay** | **the §5 kill-shot: qvel masked at train+inference (information-theoretically velocity-blind, H_o=1, no prev action) → latest 21.0% vs h8 35.0%; trap dwell 69 vs unmasked 68; trap fraction 69% in both. The 24%-vs-9% zero-v sensitivity and the 2.4× v-boost were real dependences of the unmasked network, but NOT load-bearing for the trap — v≈0 and trap geometry co-occur, and the input-channel intervention separates them.** |
| **"learned stay-regime": ≈q generalized from the 4.7% small-lead frames (H-A)** | **the §6.2 ablation: retrained on data with 0.00% stay-labels (all lead<0.15 frames excised) → trap output intact (0.023 vs local labels 0.309, 66% of latest steps). You cannot imitate a label you have never seen.** |

## 4. What is established (survives the falsifier)

All numbers below replicate on the **masked** policy unless noted
(`outputs/active_min/qvelmask_jointpos/autopsy/probe_qvelmask.json`).

**4.1 The trap, behaviorally.** Pre-success, `latest` spends **69%** of steps
commanding lead < 0.10 (median executed lead 0.049 vs dataset label mean
0.334); consecutive-trap dwell median **69 steps** (p90 83) vs `h8` median
**3** (150 separate entries — same trap exposure, immediate escape). Unmasked
values: 69%, 68, 1. Outcomes follow (6/30 vs 14/30).

**4.2 The escape ramp.** The predicted chunk at trap states still marches
out: ‖chunk[k] − q‖ = 0.031 → 0.194 over k = 0..7 (masked; unmasked
0.028 → 0.45). `h8` executes the ramp and re-enters healthy states; `latest`
executes only chunk[0] ("stay"), discards the ramp, and re-queries a state
that — with qvel masked — is *literally the same network input*.

**4.3 The stay-output is NOT local-label interpolation.** At trap states the
policy commands 0.032 while its 10 nearest demo states' **labels average
0.267** (dataset mean 0.334). The local label field says "move". Same
unmasked (0.034 vs 0.295). So no story of the form "the policy correctly
interpolates a stopped-label family" survives — the output sits ~8× below
the local field. It is best described as **collapse toward identity /
copy-proprioception**.

**4.4 The geometric signature.** Trap states' nearest demo neighbors sit at
**mean phase 0.738** of their episodes (healthy-state neighbors: 0.431), and
the plant is near-stationary there (rollout ‖v‖ 0.11 vs neighbor demo 2.03)
— the trap lives in late-phase-*like* configurations, reached mid-task
without success.

**4.5 Delta-EE inversion.** Unchanged from before: the delta policy at
stationary states still commands ~85% of demo magnitude; low-command steps
14% vs 69%; no absorbing point → dense replanning is pure freshness, +6.7.

**4.6 The amplifier, measured (freeze asymmetry).** Why does a stay-output
dominate rollouts (69% of steps) when stay-labels are only 4.7% of training
frames? Because the absorbing property is representation-asymmetric. From
the rollout traces: when the ABS policy emits a small command (<0.3× label
mean), per-step joint displacement is **0.0024 rad = 2% of healthy-step
motion** — the plant freezes, the next observation is (qvel-masked:
literally) identical, the deterministic policy returns the same output:
absorbing. When the DELTA policy emits a comparably small command, the
plant still moves at **42%** of healthy rate — the state changes, the next
query differs, the loop self-exits. In output space, "stop" is a fat ball
around q for absolute targets (anything inside the tracking deadband) and
essentially the exact point 0 for deltas. Equal shrinkage, trap only in
absolute coordinates.

**4.7 Where the stay-labels live (and don't).** Label magnitude conditioned
on plant speed, both datasets (same episodes): ABS stay-labels (<0.3× mean)
are 4.7% of frames at mean phase 0.86 — deceleration into the (trimmed)
endpoint, the hysteresis margin, and mid-episode stalls; the hold tail
itself was already trimmed. DELTA: 1.6%, and those "small" delta labels are
still moving commands. Note the raw "lead ≈ v/α" story is muddier than
first claimed: the slowest-|v| bin has ABOVE-average abs labels (1.29× —
reversals, where the target leads a stationary plant); the small-lead
family is specifically decelerations/stops, not all slow frames.

## 5. The falsifier: executed, and the causal claim failed

**Design** (pre-registered in the previous revision): hard-zero obs dims
7:14 (qvel) inside the model at BOTH entry points, train and inference
(`observation.mask_feature_ranges: [[7,14]]`, carried in the checkpoint;
invariance verified exact). With `history: 1` and no previous action, no
velocity information exists in the policy's input. Full protocol retrain
(same seed, budget, screening→confirmation selection; selected ep2900).

**Prediction** (velocity-shortcut account): trap disappears, `latest`
recovers to ≥ h8. **Result**: `latest` 21.0% [16.8, 26.0] vs `h8` 35.0%
[29.8, 40.6]; dwell/fraction/conditioning statistically indistinguishable
from unmasked (§4.1). **The account's causal channel is refuted.** Notable
secondary: masking *helped* overall (standalone final 41.4% vs 37.8%;
openloop 29.0% vs 21.0%) — the velocity channel was mildly harmful noise,
not the trap's engine.

Control arm (delta-EE + identical mask, n=300): latest 48.7% / h8 42.3% —
statistically indistinguishable from unmasked delta (48.0 / 41.3). Masking
is not generically helpful or harmful to dense replanning; the collapse is
representation-locked. Secondary: masked delta standalone final 41.4% vs
unmasked 45.6% (mild cost — for delta, velocity was signal; for absolute it
was noise: masked abs improved 37.8% → 41.4%), consistent with the original
9%-vs-24% velocity-sensitivity asymmetry.

## 6. Mechanism resolved to the model side (both discriminators executed)

The question was: why does the learned map output ≈q at a family of states,
~8–13× under its local label field? Two candidate accounts, two executed
experiments, one survivor.

**6.1 Scene-gain probe (executed).** At trap states, perturbing the BLOCK
(1 cm + 0.1 rad, sim-consistent) changes chunk[0] by **0.019**; at healthy
states, **1.001** — a **53× scene-deafness**. Perturbing qpos still moves
the output (gain 0.59). So the collapse is: *scene-conditioning shuts off;
what remains is a proprio-echo ≈ q*. (Delta control: its low-command regime
is also scene-reduced (6×) but its fallback output is 0.36 — a moving
command — and proprio-HYPERsensitive (gain 5.7): self-scrambling, not
absorbing. `gain_probe.json`.)

**6.2 No-stay-labels ablation (executed) — H-A ("learned stay-regime from
stop-aliased labels") REFUTED.** Retrained the unmasked abs-joints ACT on
`subset_0400_jointpos_nostay.h5`: every frame with label lead < 0.15
excised (**0.00% stay-labels** remain; −34.7% frames, 400 episodes → 368
segments; identical protocol). If ≈q were generalized from stay-labels, it
should vanish. Result: the ablated policy outputs **0.023** at trap states
vs neighbor labels **0.309** (13×), on **66%** of latest steps — the
fallback is intact in a network that has NEVER seen a stay-label. Arms
(n=300): latest 25.7 / h8 28.7 / openloop 24.0 — the ordering signature
flattens only because h8 dropped 8pt (the excised deceleration frames
carried real fine-positioning skill), not because latest recovered. Dwell
shifted (median 69 → 8, p90 still 82; steeper ramp 0.020→0.322): more
short visits, same absorbing core.

**Surviving account (H-B), each link measured:** at hard states the
network's scene-conditioning collapses (53×, §6.1) → the absolute head
falls back to the cheapest scene-free output, *copying the qpos present in
its own input* (0.023–0.034; robust to velocity masking AND to stay-label
removal) → executing ≈q freezes the plant (2% of healthy motion, §4.6) →
the next input is identical → deterministic policy, identical output:
absorbing under `latest` (65–69% of steps) → `h8`/openloop escape via the
later chunk slots, which always command motion (the ramp, present in all
three trained policies). Delta is immune twice: no input feature equals its
labels (nothing to copy; its degraded output is a 0.36 moving command), and
small delta commands do not freeze the plant (42% of healthy motion) — no
fixed point.

**Still open (one level down):** what makes those states "hard" — why
scene-conditioning collapses there (signature: late-phase-like geometry,
neighbor phase 0.70–0.74, stationary plant) — and whether the ≈q fallback
is literal input-copying circuitry or the L1 median under direction
ambiguity (whose center, for absolute targets, IS q). Matrix predictions,
now all H-B-based: **chunk-anchored residuals** (r = a_{t+k} − q_anchor)
re-introduce a copyable zero → trap predicted wherever anchor-copy is
cheap; **delta-JOINTS** (δ = a − q = the lead) keeps a
reachable-by-shrinkage zero → trap predicted at latest; **absolute-EE** has
a copyable tcp-pose input → trap at latest, fine at h8 (consistent with the
DP paper's position-control results, which are measured at chunked
execution only).

## 7. Caveats

Single policy seed per cell; diagnostic panels; ManiSkill `physx_cuda`
run-to-run noise ~±2–5pt (headline gaps larger; comparisons paired). The
n=30 probe outcome rates match the n=300 arms. Masked-vs-unmasked is not a
pure A/B (the observation changed); the within-run latest-vs-h8 contrast is
the load-bearing comparison, and the delta-mask control closes the square.

## 8. Artifacts

Unmasked: `outputs/active_min/rerun_jointpos/autopsy/` (autopsy.json,
trace npz, videos). Masked: `outputs/active_min/qvelmask_jointpos/`
(overnight_act.json, act_te_replication/, autopsy/probe_qvelmask.json,
autopsy/parking_latest_seed125647852.mp4 — watchable parking-mode episode
with live commanded-lead overlay, + lead-trace png; 4 of 8 episodes show
80+-step dwells).
Mask feature: `src/actsemble/policies/act/model.py` (obs_mask_ranges),
config `configs/policies/state_act_canonical_qvelmask.yaml`, tests
`tests/policies/test_act_obs_mask.py`. Probe scripts in session scratchpad;
conversion provenance in `scripts/convert_action_representation.py`.

# Actsemble system architecture — stages and tiers

How every Actsemble deployment system is structured so that (a) each
research-track item (A–G) is a small, swappable unit, (b) comparisons stay
apples-to-apples by construction, and (c) the design expands without
rewrites. This is the contract new systems are built against.

Companion to [checkpoint_selection_protocol.md](checkpoint_selection_protocol.md)
(how systems are *evaluated*) and [actsemble_research_tracks.md](actsemble_research_tracks.md)
(the *space* of systems). Taxonomy background lives in those; this file is the
*code* contract.

---

## 1. One interface, everywhere

Every system — from a bare frozen policy to a monitored recovery stack —
implements the same interface (`systems/interface.py`):

```
AutonomySystem:
    reset(*, episode_seed) -> None
    act(observation)       -> RobotAction     # called once per control step
    diagnostics()          -> dict
```

The evaluator only ever sees this interface, so **any** system — simple or
composite — runs on the same panels under the same protocol. That is what keeps
every comparison in the project clean.

Systems come in **two tiers**:

- **Tier 1 — a base pipeline.** One proposal source, a fixed-shape stage chain,
  one action out.
- **Tier 2 — a composite.** Coordinates several *child* `AutonomySystem`s,
  delegating the action to one at a time.

**The dividing test — where does the returned action come from?**
Out of *this* system's own stage-chain → **Tier 1**. Produced by a *delegated
child* → **Tier 2**. The test is mechanical (read `act()`) and recursive (a
Tier-2 child may itself be Tier-1 or Tier-2).

---

## 2. Tier 1 — the base pipeline

A single control-step decision is a fixed sequence of **stages**, each a small,
swappable seam. The base pipeline is `ReplanningSystemBase`
(`systems/interface.py`); every seam has a default that reproduces
candidate-zero, so a new mechanism overrides exactly the one seam it changes:

```python
class ReplanningSystemBase(AutonomySystem):          # the base pipeline
    def act(self, obs):
        self._history.append(self._frame(obs))
        if not self._queue:                          # replan when the execution queue drains
            self._replan()
        return RobotAction(self._queue.popleft())

    def _replan(self):
        ctx    = self._context()                     # §2.2  DecisionContext (systems/context.py)
        cands  = self._propose(ctx, record)          # Propose — samples K, records the §11 hash
        valid  = torch.isfinite(cands).all(dim=(1, 2))
        preds  = self._predict(cands, ctx)           # Predict — optional (Track C); default None
        scores = self._score(cands, preds, valid, ctx, record)   # Score — optional -> [K]; default None
        sel    = self._select(cands, scores, valid, ctx, record) # Select — default: argmax(scores) else 0
        h_a    = self._schedule(cands, sel, ctx)     # Schedule — default: fixed H_a
        self._enqueue(cands[sel][:h_a])
```

The base pipeline emits from a one-chunk execution queue (`_queue`); the
control-time cache (§2.3) is its generalization, **now realized** by the
temporal-execution variant (A4, `systems/temporal_ensemble.py`), which shares
the decision stages through `_decide` (Propose→…→Schedule) and swaps only the
execution/emission model.

### 2.1 The stages

| Stage | Signature (sketch) | Default | Tracks that live here |
|---|---|---|---|
| **Propose** | `(ctx) -> candidates [K,H_p,A]` | `policy.sample(K)` | A1, A6, G (pool) |
| **Predict** *(opt)* | `(candidates, ctx) -> rollouts` | none | C1–C2 |
| **Score** | `(candidates, preds, ctx) -> [K]` | uniform | A2, A3, A5, B*, C3–C5 |
| **Select** | `(candidates, scores, ctx) -> chunk` | argmax→candidate | A4, B6, C2, F4 (residual) |
| **Schedule** | `(chunk, ctx) -> horizon / cadence` | fixed `H_a` | A7, D |
| **Monitor** *(meta)* | `(ctx) -> signal` | none | E1–E3 |
| **Route** *(meta)* | `(signal) -> stage params/strategy` | none | A6, A7, E4 (scorer choice) |

Monitor/Route at this level only *configure this pipeline's own stages* (set K,
set horizon, pick which scorer to run). The action still exits this chain — so
adaptivity of this kind is Tier 1.

### 2.2 The decision context

Every stage receives one `DecisionContext` (`systems/context.py`):

- observation history (`H_o` frames, oldest first);
- **executed-action history** (past coherence — A5);
- replan / control-step index;
- the frozen **policy handle** (a Scorer may re-query it — A5 forward plausibility);
- the same-data **components** (verifier, dynamics model, …);
- the **prediction cache** (2.3);
- optional monitor **signal**.

The former `_select(candidates, valid, history, record)` fused Score+Select and
under-supplied context; the base now builds a `DecisionContext` and hands it to
every seam — which is what A5 (executed actions, policy handle) and C5
(components, predictions) need.

### 2.3 The prediction cache (the generalized queue)

Ordinary receding-horizon execution — *sample a chunk, execute `H_a`, discard,
replan* — is the **degenerate case** of a control-time-indexed cache that holds
one chunk and drains it `H_a` at a time. The base pipeline holds exactly that
one-chunk queue; **A4 (`TemporalEnsembleSystem`) realizes the general cache** — a
propose cadence (`replan_interval`) fills a control-time-indexed cache that
**retains overlapping predictions**, and the emission rule returns the action for
the current control-time from whatever the cache holds. Temporal ensembling (A4)
and temporal selection / chunk-switching (Track D, still future) are exactly *a
propose-cadence + an emission rule over this cache* — one policy, one chain,
action out of this chain → **Tier 1**. The emitted action may be a combined
(non-candidate) action; that is still Tier 1 — A4 offers `mean` (the classic ACT
average) plus the multimodal-safe `projection` (snap the weighted mean onto a
real prediction) and `medoid` rules, with `latest` (no ensembling) as the
compute-matched control. A note on fairness: because A4 changes the *propose
cadence* (it replans every step), it is **budget-matched** against the
receding-horizon baseline (§2.5), not candidate-identity-matched; the exact
same-plan property holds only *open-loop* (fixed observation) — in closed loop
the aggregation variants legitimately diverge as soon as their emitted actions
differ.

### 2.4 The Tier-1 contract — candidate identity (§11)

`Propose` records a SHA-256 of the K-candidate tensor. Because Propose is
untouched when you swap `Score` or `Select`, the K-tensor is **bitwise identical**
across those comparisons — every Score/Select experiment is a controlled
one-knob swap. Identity is a **prefix property**: once two systems select
different candidates their observations legitimately diverge; the invariant is
that candidate hashes match up to (and including) the first replan where the
selected indices differ. `tests/test_system_candidate_identity.py` enforces it,
and it must hold for every new Tier-1 system.

### 2.5 Fairness controls differ by stage

- **Score / Select** comparisons → candidate identity (same K-tensor).
- **Propose** comparisons (A1 / A6 / G-pool) → **budget-matched** (equal total
  samples, compute, parameters). You cannot hold proposals fixed while changing
  the proposer; the control is matched budget instead.

The stage a change lives in tells you which control applies.

### 2.6 Stage-first organization (the decided structure)

- **Stages are the swappable unit** — seam methods on the base pipeline
  (`_propose / _predict / _score / _select / _schedule`), each with a
  candidate-zero default.
- **A system overrides the seam(s) it changes** (a small subclass today); the
  **factory** (`systems/factory.py`) selects it from a config (`selection.type`
  + params, `configs/systems/*.yaml`).
- **An experiment is a one-knob swap** (`selection.type: verifier → dynamics_progress`);
  Propose is untouched, so the K-tensor stays identical — the clean comparison.
- **A "track / paradigm" is a named config recipe**, not a code module. Track C
  being *spread across* Predict + Score is honest: it *is* a dynamics model plus a
  scorer that consumes it. (A fuller data-driven propose×score×select registry is a
  natural later increment; the seams are what make it mechanical.)

---

## 3. Tier 2 — the composite / router

**A coordinator that owns child `AutonomySystem`s + a monitor + a gate + a handoff
protocol, and delegates the action to one child at a time.** The children never
talk to each other; the coordinator mediates everything.

```python
class Router(AutonomySystem):                 # children are AutonomySystems (Tier-1 or Tier-2)
    def act(self, obs):
        target = self.gate(self._signal(obs))            # which child owns control now
        if target != self._active:
            self.handoff(self._active, target, obs)      # transfer history / continuity
            self._active = target
        for name, child in self.children.items():
            child.observe(obs)                           # WARM-PAUSE: paused children keep observing
        return self.children[target].act(obs)            # DELEGATE — action comes from the child
```

### 3.1 Two modes

- **Asymmetric (supervisor).** A privileged **nominal** + a **fallback** — recovery
  (F) or escalation to a costlier system (E4). The nominal is **paused**, control
  **hands off** to the fallback, and after a while control **returns** to nominal.
  E4 and F share this shape; they differ only in trigger (*difficulty* vs
  *predicted failure*).
- **Symmetric (peer routing / MoE, G5).** Children are equals; the gate picks one
  per state; there is no privileged nominal and no "regain." Same coordinator
  machinery minus the pause/regain asymmetry.

The asymmetric mode is the one that drives the design, because it is where the
handoff concerns bite.

### 3.2 Handoff invariants (what a Tier-2 coordinator must guarantee)

1. **Single-actor** — exactly one child emits the physical action each step. The
   robot has one body; never two actions.
2. **Warm-pause** — a paused child keeps *ingesting observations* (updates its
   `H_o` history) but does not *emit*. Otherwise it wakes cold with a stale
   history and jerks on return.
3. **Continuity at the switch** — the incoming child's first action must not jump
   discontinuously from the last executed one (Track D3 / F re-entry; may need
   boundary inpainting or blending).
4. **Explicit re-entry condition** — *who declares the fallback finished?*
   Familiarity/support restored, or the fallback signalling completion. Owned by
   the monitor/gate, not the children.
5. **Hysteresis / minimum dwell** — the gate must not oscillate every step: a
   threshold to leave nominal, a lower threshold to return, and a dwell time.
6. **Reset** — on episode reset all children reset and control returns to nominal.

None of these exist in Tier 1 — which is exactly why routing/recovery cannot live
*inside* the base pipeline.

### 3.3 Fairness + metrics

Candidate identity is undefined across a router (children sample differently), so
Tier-2 comparisons are **budget-matched** (total samples, params, compute,
checkpoint-selection opportunities — the G-track "necessary controls"). The
question "does the recovery/escalation layer help?" is
**Tier-2(nominal+fallback) vs Tier-1(nominal alone)**, budget-matched, on the F/E
metrics: autonomous success, intervention rate, false-alarm rate,
failure-before-intervention.

---

## 4. What we commit to now

We build **Tier 1** now. We do **not** build Tier 2 yet. The single standing
rule that keeps Tier 2 possible without building it:

> **The base pipeline is a self-contained `AutonomySystem`. Whole-system routing
> and alternate-generator recovery are NOT allowed inside it** — they live in a
> future coordinator that composes base pipelines as children.

Tier 2 materialises only when we reach E-routing / F, as a small composite over
things we already have, evaluated through the identical panel machinery.

---

## 5. Where A–G lands

| Item | Tier | Stage / role |
|---|---|---|
| A1 multi-sample, A6 adaptive-K | 1 | Propose (A6 = Monitor→K) |
| A2 consensus, A3 policy-internal, A5 bidirectional | 1 | Score |
| A4 temporal ensembling | 1 | Select + cache (propose cadence) |
| A7 adaptive horizon | 1 | Schedule (Monitor→horizon) |
| B1–B6 learned selectors | 1 | Score (component) |
| C1–C2 dynamics / DREAM-switch | 1 | Predict (feeds Score/Select) |
| C3–C5 support / progress / uncertainty | 1 | Predict + Score |
| C6 oracle | 1 (diagnostic) | Score via privileged sim — not deployable |
| D1–D4 temporal execution | 1 | Schedule + Select over the cache |
| E1–E3 monitoring | 1 | Monitor (produces a signal) |
| E4 route **between scorers** | 1 | Route (within-stage) |
| F4 residual `a=a_π+Δ` | 1 | Select (post-select Correct modifier) |
| G1–G3 candidate pooling | 1 | Propose (multi-source) |
| **E4 route between *systems*** | **2** | delegate to nominal / world-model / recovery |
| **F1/F3 recovery policy, retrieval** | **2** | delegate to a different generator (handoff) |
| **G5 MoE gate** | **2** | delegate to π\_{m\*} |

The items that split (E4, F, G) resolve by the dividing test: Tier 1 when they
change a *knob*, Tier 2 when they change the *action source*.

---

## 6. Immediate build — A4 / A5 / C5 map onto Tier-1 stages

- **A5 bidirectional consistency** → override **`_score`**, reading
  `ctx.executed_actions` (`C_past`) and `ctx.policy` (`C_future`); the base's
  default `_select` does the fallback-aware argmax.
- **C5 dynamics uncertainty** → override **`_predict`** (a `StateDynamicsModel`
  component + an M-ensemble) and **`_score`**
  (`λ_p·progress − λ_s·support − λ_u·uncertainty`).
- **A4 temporal ensembling** → **DONE** (`systems/temporal_ensemble.py`): a
  temporal-execution variant that reuses `_decide` and replaces the queue with a
  control-time cache + an aggregation emission (`latest` / `mean` / `projection`
  / `medoid`), a propose cadence (`replan_interval`), and an EWMA age weight.
  Configs `configs/systems/temporal_*.yaml`; runner `scripts/compare_temporal.py`.

A5 / C5 ship as a config against the same base pipeline, keep candidate identity,
and are compared paired against candidate_zero / verifier / medoid / oracle on the
`diagnostic` panel — one knob at a time. A4 differs (a propose-cadence change), so
it is **budget-matched** against the receding-horizon baseline and additionally
**compute-matched** against `temporal_latest` to isolate the aggregation rule.

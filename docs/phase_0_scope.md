# Phase 0 scope: state-based mechanism validation

## What Phase 0 proves (if results are positive)

Phase 0 answers exactly one question:

> Given a frozen dataset of successful trajectories and a frozen diffusion
> policy trained on it, can additional models trained on the **same
> dataset**, assembled with the **unchanged** policy at runtime, raise
> closed-loop task success?

A positive result establishes the **system-level mechanism**: same-data
components carry information the policy's single sample does not exploit,
and runtime selection can convert that information into task success. The
compute-matched multi-sample control tells us whether the gain came from
the component or merely from drawing more samples.

## What Phase 0 does NOT prove

Positive state-based results establish the system-level mechanism, not
real-world applicability. Phase 0 provides **no evidence** about:

* visual perception capability;
* real-world deployability;
* sim-to-real transfer;
* robustness to perceptual uncertainty;
* success using RGB-only observations;
* benchmark-ready observation restrictions.

## Why privileged state is scientifically acceptable here

The hypothesis is about the *relationship between components trained on a
shared dataset and a frozen policy trained on that dataset* — not about
perception. Every model in the comparison (policy, component) receives the
same privileged observations, so the observation channel is a controlled
variable, not a confound. Low-dimensional state makes iteration ~100x
cheaper and removes representation-learning failures as an explanation for
negative results. If the mechanism does not work with clean, fully
observable state, there is no reason to expect it to work from pixels.

## Evidence required before moving to visual observations

1. A statistically defensible closed-loop improvement of
   `candidate_reranking_actsemble` over `standalone_diffusion` (paired
   bootstrap CI excluding zero at n ≥ 200 episodes) in at least one regime.
2. The improvement must **exceed** the multi-sample control's improvement —
   otherwise the effect is candidate quantity, not the component.
3. The effect should persist under at least one perturbed regime
   (recoverable disturbances), where selection has more room to matter.
4. Ideally, replication on a second state-based task (Phase 0B) to show
   the mechanism is not PushT-specific.

## Phase 1 migration (RGB + proprioception)

Planned changes, in order of appearance:

* dataset schema: replace the flat `state` array with observation groups
  (`rgb`, `proprio`, `previous_action`) — same episode alignment,
  hashing, validation, and provenance machinery;
* policy: add a small visual encoder feeding the same 1-D temporal U-Net;
* component: same compatibility formulation over encoded observations;
* systems, candidate identity, paired evaluation, perturbations: unchanged.

Everything downstream of `StateObservation` and the dataset reader is
already observation-agnostic; that boundary is the migration surface.

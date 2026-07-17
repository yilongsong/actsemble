# selector_baselines_v1

Experimental namespace for the **non-learned consensus selectors** — controls
for the Actsemble learned verifier. This directory holds the eventual frozen
specification and panels for the *formal* selector comparison.

**Nothing here modifies Phase 0A.** The implementation reads the frozen Phase 0A
seed-4 policy/verifier read-only and validates on unit/integration checks and a
development panel (`selector_development`, env_seed 9100) disjoint from every
Phase 0A panel. The Phase 0A frozen spec, checkpoints, verifier, final-test
outputs, and conclusion are untouched.

## Status

- Selectors implemented + unit/integration/dev-panel validated. See
  [`../../docs/consensus_selectors.md`](../../docs/consensus_selectors.md) and
  [`../../outputs/selector_baselines_v1/`](../../outputs/selector_baselines_v1/).
- **Not yet frozen for a formal comparison.** Before any scientific claim, a
  frozen `experiment_spec.yaml` and a new **untouched** final-test seed panel
  (disjoint from all existing panels) must be created here.

## Planned formal comparison

Systems (identical policy checkpoint, K=16, seeds, simulator, horizons):

1. candidate zero;
2. full-chunk medoid;
3. early-weighted medoid;
4. coordinate-median projection;
5. largest-cluster medoid;
6. verifier argmax.

Key scientific question: **generic self-consistency vs. a separately trained
same-data verifier** — does the learned verifier beat the best non-learned
consensus rule, and by how much?

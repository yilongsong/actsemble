# Adding components and selectors

The runtime contract is small; a new component touches four places:

1. **Model + runtime class** — `src/actsemble/components/<name>.py`.
   Implement the `LearnedComponent` surface (`dataset_hash`,
   `checkpoint_hash`, `reset()`) plus whatever scoring/prediction method
   your selector needs. Store the shared contract fields in the checkpoint
   meta (`dataset_hash`, `split_hash`, `normalization`, `state_dim`,
   `action_dim`, `obs_horizon`, `prediction_horizon`,
   `include_previous_action`) so `check_same_data` can enforce the
   same-data contract at assembly time.
2. **Trainer** — `src/actsemble/training/train_<name>.py`. Must read only
   the frozen dataset file. No simulator imports anywhere in its import
   graph (the no-sim test will catch violations). Reuse
   `make_policy_meta`, `split_episodes`, `compute_stats` so hashes match
   the policy's by construction.
3. **Config** — `configs/components/<name>.yaml`. Keep the
   `observation`/`action` sections identical to the policy config.
4. **Selector wiring** — extend `systems/factory.py` (a new
   `selection.type`) and, if the selection logic is more than an argmax,
   a new `ReplanningSystemBase` subclass overriding `_select`. `_select`
   receives the candidate tensor, a finite-validity mask, the raw
   observation history, and the per-replan diagnostics record; it returns
   an index and must never raise (record a fallback and return 0 instead).

Everything else — candidate sampling, seeds, paired evaluation, latency
accounting, comparison reports — comes for free.

## Component ideas (not implemented; do not build until Phase 0A concludes)

* **Familiarity model** — density/support estimate of the observation
  history alone (e.g., autoencoder reconstruction error, flow, or k-NN in
  normalized state space). Selector: veto candidates whose *predicted*
  successor states leave familiar territory, or gate replanning frequency.
* **Latent dynamics model** — same-data (s_t, a_t, s_{t+1}) regressor.
  Selector: roll each candidate chunk through the model and score
  imagined trajectories with any other component (composition is just a
  second entry in `components:` plus a selector that consumes both).
* **Trajectory retriever** — nearest-neighbor index over training windows
  (observation embedding -> chunk). Selector: score candidates by distance
  to retrieved expert chunks, or inject the retrieved chunk as candidate
  K+1 (still same-data; document the provenance in diagnostics).
* **Progress estimator** — monotone "phase" regressor trained on relative
  episode position (t / T is derivable from the dataset, not a reward).
  Selector: prefer candidates whose imagined successors advance phase.
  Careful: this skirts the no-dense-progress rule for *inputs*; the phase
  target is a training label derived from the dataset, never an
  observation input.
* **Recovery model** — train on windows where the *observation history*
  is perturbed (same deterministic transform machinery as negatives) but
  the target chunk is the demonstrated one; score candidates by proximity
  to the predicted recovery chunk.
* **Alternative candidate selectors** — over existing components:
  score-weighted softmax sampling (seeded), top-p filtering + random pick
  (tests whether argmax is too greedy), pairwise tournaments, or
  minimum-score vetoes with candidate-zero fallback. Selectors are pure
  functions of (candidates, scores, seed) and need no new training.

## Rules that always apply

Training inputs: the frozen dataset file, nothing else. No simulator, no
rollouts, no reward, no success labels, no failed trajectories, no
external data. The policy stays byte-identical. Any negative/positive
construction must be deterministic under the recorded seeds. Offline
metrics are diagnostics only —

> High offline accuracy does not imply improved closed-loop task success.
> Closed-loop evaluation is the decisive test.

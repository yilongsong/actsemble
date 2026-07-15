# Experiment contract

The rules that make the Actsemble comparison meaningful. Violating any of
these invalidates a run; most are enforced mechanically (noted inline).
Checkpoint selection, panel definitions, freezing, and final-test gating
are specified in [checkpoint_selection_protocol.md](checkpoint_selection_protocol.md).

## Same frozen dataset

* One dataset file per experiment; content-hashed (SHA-256 over all
  episode arrays + metadata). Enforced: every checkpoint stores
  `dataset_hash`, `split_hash`, and the full normalization statistics;
  `check_same_data` rejects mismatched assemblies at system construction;
  evaluation results embed the hashes and `compare_systems.py` refuses to
  compare across them.
* Success-only: the dataset contains only successful episodes. The
  auditable record (source success flags, rejection counts, conversion
  failures, original episode ids) is the private `*.provenance.json`
  sidecar — never model input. Enforced by `validate_dataset` +
  `validate_success_only_provenance` + tests.
* Reward, per-step success, failure, dense progress, termination reasons:
  never stored in episode arrays (schema forbids; validated).

## Same frozen policy

* All systems in a comparison load the same policy checkpoint file;
  equality is checked on the file hash (`policy_checkpoint_hash` in every
  result JSON; verified by `compare_systems.py` and the smoke test).
* Same EMA choice (`policy_weights_kind` recorded and compared), same
  normalization, same scheduler, same inference-step count, same sampler,
  same action horizon, same observation history. All of these live inside
  the checkpoint config, so loading the same file guarantees them; the
  eval result records them for audit.
* Systems never fine-tune, adapt, or write to the policy (inference only;
  `policy.frozen: true` required by the factory).

## Component training restrictions

* Inputs: the frozen dataset file. Nothing else.
* Forbidden: simulator interaction, gym environments, rollouts (online or
  offline), reward, success labels, privileged evaluation outcomes, failed
  trajectories, additional expert demonstrations, external datasets.
* Enforced: `tests/test_training_has_no_sim_dependency.py` runs both
  trainers in a subprocess where importing `mani_skill`, `gymnasium`, or
  `sapien` raises, and asserts the training import graph never touches
  `actsemble.sim`.
* Negatives are deterministic transformations of same-dataset chunks,
  seeded per (dataset seed, episode id, t, replica).

## Same controller and physics backend

* Demonstration conversion, training-data interpretation, and evaluation
  all use `pd_ee_delta_pose` on `physx_cuda`. The dataset records both;
  `verify_env_matches` fails loudly on any live-env mismatch (task id,
  controller, backend, action dim, bounds).
* Known, documented gap: the source demos were recorded in 1024 parallel
  GPU envs and GPU PhysX dynamics are batch-size dependent, so no
  single-env replay bitwise-reproduces them. The dataset stores the exact
  recorded transitions (state projection); evaluation runs the same
  backend at num_envs=1 (bitwise deterministic in-process).

## Paired rollout evaluation

* Episode i receives (env_seed, perturbation_seed, candidate_seed) derived
  from one root seed — identical for every system. Recorded in every
  result JSON; compared field-by-field before any report.
* Candidate-set identity: the candidate tensor for replan r of episode e
  depends only on (candidate_seed, e, r). With equal K, the multi-sample
  control and the Actsemble reranker receive bitwise-identical candidates;
  only the selection rule differs
  (`tests/test_system_candidate_identity.py`).
* Primary metric `success_once`, with `success_at_end`, timeouts,
  exceptions, fallbacks, clip rates, and latency always reported.
  Comparisons under 50 paired episodes are flagged as not statistically
  meaningful; smoke-test outcomes are never scientific evidence.

## Explicit compute controls

* `multisample_control` consumes exactly the same policy-sampling compute
  as the Actsemble system (same K per replan) with a component-free
  selection rule (seeded uniform random or first candidate). It is not
  claimed to be a strong selector.
* Latency (policy sampling, component scoring, total decision) is measured
  per replan and reported per system, so any success difference can be
  weighed against its compute cost.

## Perturbation rules

* Perturbations are seeded per episode and recoverable at the configured
  severities (`object_nudge` magnitudes are capped and workspace-clamped;
  mild severities must keep closed-loop success well above zero — verify
  per severity before trusting a regime).
* The system under evaluation never observes that a perturbation occurred,
  when, its direction, or its magnitude — only the resulting state.
* `object_nudge` uses privileged sim-state access internally; that access
  ends at the perturbation boundary.

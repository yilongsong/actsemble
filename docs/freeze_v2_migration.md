# Freeze v2 and evaluation-result migration

This document is the consolidated change record for the protocol, result,
runtime, resume, packaging, and audit changes introduced with freeze v2. It
describes compatibility and stale artifacts; it does not prescribe training
work.

## Protocol and result schemas

New freeze manifests identify themselves as
`actsemble_system_freeze_v2`. New evaluation results identify themselves as
`actsemble_evaluation_v2`. A v2 freeze refuses a result without the v2 result
identifier and verifies the result against the frozen policy, verifier when
used, data, environment, sampler, execution offset, fallback rule, source
tree, runtime, and panel.

The v2 result adds `normalization_hash`, `policy_config_hash`, `source`,
`runtime`, `environment_contract`, `sampler`, `selection_type`, `execution`,
and `fallback_rule`. It records both `fallback_episode_rate` and
`fallback_replan_rate`. The old `fallback_rate` key remains as an alias for
the replan-weighted rate. Mean latency aliases also remain for readers of old
reports.

Results are protected from accidental overwrite. `evaluate.py` and protocol
stages require an explicit `--force` to replace an existing output.

## Candidate and action identity

Every candidate tensor is hashed from its contiguous float32 bytes with the
full 64-character SHA-256 digest. Per-episode candidate digests are also full
SHA-256 values rather than truncated display identifiers. Executed-action
digests use full SHA-256 as well.

Candidate identity remains a prefix property. Paired systems must have equal
candidate hashes through the first replan where their selected indices differ.
Systems that never select differently must match at every replan.

## Fallback semantics

Selection first attempts candidate zero when a selector is absent, a component
raises, or no finite component score is available. If candidate zero itself is
non-finite, the shared replanning layer selects the first finite candidate and
records a fallback. If no candidate is finite, evaluation raises instead of
executing a non-finite action. The frozen rule string is
`candidate_zero_if_finite_else_first_valid; no_finite_candidate_raises`.

Fallback reporting now separates the fraction of episodes containing at least
one fallback from the fraction of replans that fell back. Selection-change and
action-clip rates are likewise aggregated over their natural denominators,
not averaged per episode.

## Verifier metric correction

Pairwise ranking accuracy now compares each positive score with every negative
belonging to that same positive. For a batch with `B` positives and `N`
negatives per positive, the metric contains `B*N` comparisons. The previous
flatten-and-truncate calculation compared only `B` entries and could cross
positive/negative groups.

Any selected-verifier record whose primary metric was calculated by the old
rule is stale as a selection audit. Recompute the offline validation history
with the corrected grouping and rerun offline verifier selection before using
that record in a new freeze. This correction does not introduce simulator
feedback into verifier selection.

## ACT objective and architecture record

The canonical ACT objective is the padding-masked elementwise L1 action loss
plus the configured KL term. Masking applies before reduction so padded action
positions do not contribute. Validation losses are example-weighted rather
than an unweighted mean of batch means. The DETR-style ACT path has separate
posterior and decoder observation projections, a final decoder normalization,
fixed sinusoidal positions when configured, and deterministic zero-latent
inference. Checkpoints remain structurally identifiable and load validation is
strict; results should retain the exact policy configuration and source hash so
objectives are not treated as interchangeable.

## Resume compatibility

Diffusion-policy and component checkpoints now persist all trajectory-driving
RNG state, data-loader epoch position, optimizer state, and current step. An
exact resume refuses a checkpoint without the new RNG state instead of silently
continuing along a different optimization trajectory.

Diffusion checkpoints also persist learning-rate scheduler state and its
planned total step count. Constant-rate runs can resume to a larger budget.
Cosine-scheduled runs require the same planned total step count because changing
the endpoint changes the entire schedule. ACT and flow training remain
one-shot and reject `--resume` explicitly. Training also refuses a non-empty
output directory unless the supported resume path is requested.

## Latency statistics

Evaluation performs one unmeasured warmup decision before recording episodes.
CUDA devices are synchronized around policy and component timing. Policy,
component, and end-to-end decision latency each report sample count, mean,
p95, and p99 in seconds. The retained `mean_policy_s`, `mean_component_s`, and
`mean_decision_s` fields mirror the nested means for compatibility.

## Source, runtime, and environment provenance

Claim-bearing freezes and results record the Git commit plus a content-sensitive
source-tree hash that includes tracked modifications and relevant untracked
source files. They also record dirty state and changed paths. Runtime provenance
includes Python implementation/version, platform, NumPy, PyTorch, CUDA runtime,
cuDNN, CUDA visibility, GPU names, byte order, and ManiSkill version where the
simulator is active.

The live environment contract records task, robot, controller, backend,
control frequency, action dimension, and action bounds. Freeze verification
checks these facts as well as the frozen source and core runtime versions.

## Packaging, commands, and test organization

The package now declares runtime dependencies used by video output, separates
simulator and analysis extras, and provides a pinned CUDA benchmark constraint
file under `requirements/`. Installed commands are
`actsemble-train-policy`, `actsemble-train-component`, and
`actsemble-verify-gpu`. Policy training dispatch is shared across diffusion,
ACT, and flow rather than duplicated in script entry points.

The previously ignored top-level `data/` pattern is now rooted as `/data/`, so
the importable `src/actsemble/data` package and `configs/data` are packaged and
tracked correctly. Tests are grouped by subsystem, including component
training, policy dispatch, protocol freezing, normalization, window alignment,
and policy-specific behavior. The default suite is simulator-independent;
`pytest -m sim` opts into live ManiSkill tests.

## GPU-host verification

Run `actsemble-verify-gpu` in the installed benchmark environment. It requires
PyTorch CUDA visibility, constructs the exact ManiSkill `physx_cuda` PushT
environment, performs reset and step calls, exercises the object-nudge state
mutation, synchronizes CUDA, and prints the observed hardware and environment
contract. It never falls back to CPU. Then run `python -m pytest -m sim -q` for
the marked repository integration tests.

A container or remote execution sandbox can have the correct Python packages
while lacking NVIDIA device or driver passthrough. Failure there establishes
only that CUDA is unavailable to that process; it does not establish that the
underlying host is misconfigured.

## Legacy and stale artifacts

Freeze v1 manifests remain historical records but cannot gate a new v2 final
result. Evaluation JSON without `actsemble_evaluation_v2`, exact sampler and
execution provenance, source/runtime provenance, or full candidate hashes is a
legacy result and should not be merged into a v2 claim set.

Results that use the old episode-averaged fallback, clipping, selection-change,
or latency summaries are stale for those aggregate fields. Results with
truncated candidate or action digests are stale for exact identity audits.
Verifier selections based on the old pairwise metric are stale as selection
records. Resume checkpoints without saved RNG state are loadable as model
artifacts where their policy loader permits it, but they are incompatible with
the exact-resume path. Existing installations should be reinstalled from the
current package metadata before relying on the new commands or tracked data
modules.

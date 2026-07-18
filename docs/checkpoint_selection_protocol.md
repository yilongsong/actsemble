# Checkpoint selection and evaluation protocol

How Actsemble selects the frozen policy checkpoint, the verifier
checkpoint, and produces unbiased closed-loop comparisons. Every stage is
driven by [scripts/run_protocol.py](../scripts/run_protocol.py) against an
experiment spec ([configs/protocol/default.yaml](../configs/protocol/default.yaml))
that is **frozen at init**: later stages read only the copy inside the
experiment directory, and a changed spec against an existing directory is
rejected — budgets, panels, and rules cannot drift after results are seen.

```bash
python scripts/run_protocol.py init --spec configs/protocol/default.yaml \
    --experiment-dir outputs/experiments/v1
python scripts/run_protocol.py all  --experiment-dir outputs/experiments/v1 --policy-seed 0
# ... repeat for every policy seed, then:
python scripts/run_protocol.py aggregate --experiment-dir outputs/experiments/v1
```

## Panels (§1) — `evaluation/panels.py`

Four disjoint frozen episode banks (plus one for dataset-size selection),
each episode carrying an environment seed, a policy-sampling seed, and a
perturbation seed derived from the panel root:

| panel | root seed | episodes | used for | never used for |
|---|---|---|---|---|
| screening | 2000 | 50 | cheap per-interval checkpoint evaluation | final selection alone |
| confirmation | 3500 | 200 | selecting the frozen policy | anything else |
| integration | 4500 | 10 | implementation checking | selection, tuning, claims |
| final_test | 1000 | 500/seed | **reported results only** | any decision whatsoever |
| dataset_size_development | 5500 | 200 | choosing n_demos | the final test |
| diagnostic | 20000 | 300 | oracle headroom, pass@k, selector development (non-protocol) | **any claim** |

The `diagnostic` panel lives in `DIAGNOSTIC_PANELS` (outside `DEFAULT_PANELS`,
so the frozen protocol is unchanged) and is checked disjoint from every
protocol bank; it backs privileged / upper-bound diagnostics and never
produces a reported result.

`assert_panels_disjoint` verifies pairwise disjointness **and**
disjointness from the demonstration-generation seeds (read from the
dataset provenance sidecar); the freeze stage runs it before any frozen
evaluation.

## Named randomness (§2)

Every stochastic stage has a dedicated generator with a derived, recorded
seed (`named_training_seeds` / `named_verifier_seeds`; the name→seed map
is written to each run's `train_config.json`): policy init, data-loader
order, diffusion noise, diffusion timesteps, validation, verifier init,
verifier loader order, negative generation, plus per-episode env /
policy-sampling / perturbation seeds from the panels and per-replan
candidate generators (§11). No stage relies on shared global RNG state.

## Screening cannot touch training (§3) — `utils/rng_state.py`

Training modules never import the simulator (enforced by test). Screening
runs through an `on_checkpoint` callback that the orchestrator injects
into the trainer; the trainer wraps every callback invocation in
`preserve_rng_states`, which snapshots and exactly restores Python, NumPy,
torch CPU, all torch CUDA states, and the named training generators.
`tests/test_rng_preservation.py` verifies end-to-end that training with
screening at any frequency — or disabled — produces bitwise-identical
final weights and loss trajectories.

## Policy training (§4) and selection (§5–§6)

Fixed budget (default 30,000 steps), **no early stopping**, full model
snapshot (raw + EMA + config + metadata) every `checkpoint_every` steps
(default 2,000) and at the final step. Screening (§5) evaluates each
snapshot's EMA weights with ONE chunk per replan on the identical
screening bank; a perfect score changes nothing. Confirmation (§6) takes
the top-5 screening checkpoints ∪ all within 0.10 of the screening best,
re-evaluates them on the confirmation panel, and selects by the
lexicographic rule (confirmation rate ↓, screening rate ↓, step ↑). The
winner is re-saved as `selected_policy.pt` with the complete screening +
confirmation history, panels, rule, training seed/budget, generator map,
and git commit embedded in its metadata (§17).

## Dataset-size selection (§7) — `scripts/select_dataset_size.py`

Nested deterministic subsets: one seeded permutation of all source
episodes; size n = its first n entries (`prepare_dataset.py
--subset-size`). The subset manifest and hash, plus the source bundle's
SHA-256, go into dataset metadata and provenance (in HDF5 metadata,
`subset_size = -1` encodes "all"). The driver runs the complete
train→screen→confirm pipeline per seed and size, evaluates each SELECTED
policy on the dataset-size-development panel, and picks the largest size
with mean success in the pre-declared 25–50% band. The final panel is
never touched.

## Verifier training and selection (§8–§9)

Same-data constraints unchanged (no simulator, rollouts, rewards, success
labels, failures, pretrained models, external data). Fixed budget,
snapshots at `checkpoint_every`, and an offline validation record
(`offline_history.json`: ranking accuracy, positive/negative/balanced
accuracy, BCE validation loss, per-negative-type breakdown) at every
interval on the episode-disjoint validation split. Selection
(`select-verifier`) is **entirely offline**: primary metric = validation
pairwise ranking accuracy (or `validation_loss`, lower-better),
secondary = balanced accuracy, then earliest step. The verifier receives
no simulator-derived selection signal of any kind; `selected_verifier.pt`
embeds the full offline history and audit fields.

## Freezing (§10) — `protocol/freeze.py`

`freeze.json` pins: selected policy path + file hash, EMA choice, sampler,
inference steps, temperature, all horizons, K, selected verifier path +
hash, both normalization hashes, dataset/subset/split hashes, selection
and fallback rules, env facts, and the integration/final panels. The
same-data contract and panel disjointness are re-verified at freeze time.
Integration and final-test refuse to run without it, and every final
result is verified field-by-field against it.

## Candidate-set identity (§11) — `systems/interface.py`

Per replan, the candidate generator seed is derived from exactly
(episode policy-sampling seed, **policy checkpoint hash**, replan index);
the sampled `[K, H_p, A]` tensor is SHA-256 hashed into the diagnostics
and result files. In paired-comparison mode all three systems sample the
same K-tensor and differ only in selection:

    standalone  → candidate zero
    control     → candidate zero (first_candidate; execution-identity control)
    actsemble   → highest verifier score (fallback: candidate zero)

Two implementation decisions, made deliberate and explicit:

1. **Checkpoint hash is folded into candidate seeds everywhere** (§11's
   derivation applied uniformly). Screening/confirmation therefore compare
   checkpoints under the identical seed *bank* but checkpoint-specific
   noise realizations — selection cannot overfit one noise stream.
2. **Closed-loop identity is a prefix property.** Once Actsemble selects a
   non-zero candidate its trajectory — hence its subsequent observations
   and tensors — legitimately diverges. The verified invariant
   (`reports.verify_candidate_identity`, enforced in every comparison and
   at integration): per episode, candidate hashes must be bitwise equal up
   to and including the first replan where the systems' selected indices
   differ; standalone vs control (which never differ) must match on every
   replan. Any earlier mismatch invalidates the paired comparison.

Preconditions discovered and handled during verification on physx_cuda:
`make_env` warms every new environment with throwaway contact episodes
(GPU PhysX lazily grows buffers, perturbing a fresh env's first episode at
~1e-7), and each system is evaluated in its **own freshly created env** so
all systems see bitwise-identical simulator histories.

## Integration (§12) — `protocol/integration.py`

Twelve named checks (checkpoint identity, hash compatibility, verifier
loading, dimensions, history reset, candidate identity, finite scores,
fault-injected fallback, logging, video, verifier-input closure — the
input layer is exactly sized for obs history + chunk, leaving no room for
simulator features — and a policy-weight digest before/after rollouts).
Exit is nonzero on any failure; success rates are recorded under
`success_rates_not_for_claims`.

## Final test (§13) — `protocol/final_test.py`

Gated on `freeze.json` **and** a fully passed integration report. Runs the
three systems in paired mode on the final panel for each regime declared
in the frozen spec, verifies every result against the freeze and candidate
identity, and writes per-system results + the comparison. Existing results
are never overwritten without `--force`; a changed system keeps its old
results under the old experiment version and needs a new version directory
with a fresh panel (§16, §17).

## Seeds and reporting (§14–§15) — `protocol/seed_report.py`

The trained model is the replication unit (≥5 policy seeds for claims;
verifier seeds recorded separately, paired by default). `aggregate`
reports per-seed Actsemble−standalone and Actsemble−control differences,
their mean, sample SD, Student-t 95% CI across seeds, and sign counts.
Per-episode stats (§15) are all in each result JSON: success rates with
Wilson CIs, episode length, timeout/exception/clipping/fallback rates,
policy and verifier latency, win/loss/tie and both-succeed/both-fail
counts, bootstrap CI for the paired difference, and the
candidate-selection-change frequency (fraction of replans where Actsemble
picked a non-zero candidate).

## Prohibited practices (§16) — mechanical guards

Frozen specs (no post-hoc budget/panel edits), freeze-gated evaluation,
no-overwrite-without-force, symmetric per-spec screening intervals,
RNG-guarded screening, panel disjointness, same-file policy loading with
hash verification, offline-only verifier selection, and candidate-identity
verification cover the §16 list mechanically; the remainder (e.g. "don't
train the verifier on policy failures") is enforced by the training-side
same-data constraints and the no-simulator import test.

## Audit trail (§17)

Selected policy and verifier checkpoints embed their full histories and
configuration (see above) plus the git commit (`utils/repo.py`; run
experiments from a committed tree — outside a repository the field is
recorded as null). Every evaluation result stores checkpoint hashes, the
full seed banks, per-episode outcomes and candidate hashes, simulator
version, controller, backend, and the complete config.

## Core interpretation (§19)

Policy selection may use held-out rollouts — rollout success is the
relevant measure of policy competence. The verifier may not: trained only
on successful demonstrations, selected only on demonstration-derived
offline validation. The final test therefore answers exactly: *does an
independently trained, same-data component improve the closed-loop
success of an already selected and frozen Diffusion Policy?*

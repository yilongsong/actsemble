# Actsemble

Assembling a robot autonomy system from a strong frozen action-generating
policy plus additional learned components trained on the **exact same
dataset** — and testing whether the assembled system beats the standalone
policy in closed-loop task success.

> **Warning — Phase 0 scope.** This initial version uses privileged
> low-dimensional simulator state to test the Actsemble mechanism quickly.
> Positive state-based results establish the system-level mechanism, **not
> real-world applicability**. It is not evidence that the same system works
> from real-world visual observations. Vision-based validation is a later
> phase (see [docs/phase_0_scope.md](docs/phase_0_scope.md)).

## The hypothesis

Given a frozen dataset **D** of successful state–action trajectories
(s_t, a_t, s_{t+1}) and a strong diffusion policy π_D trained on it, one or
more additional models M_D trained **only on the same dataset** can be
assembled with the **unchanged** policy into a system

    A = G(π_D, M_D^(1), ..., M_D^(k))

such that

    Success(A) > Success(π_D)

in closed-loop evaluation. The name **Actsemble** = *act* + *assemble*: the
action-generating policy stays frozen; extra same-data components and
runtime system logic are assembled around it.

## The three systems compared

| system | candidates per replan | selection rule | learned component |
|---|---|---|---|
| `standalone_diffusion` | 1 | candidate zero | none |
| `multisample_control` | K (16) | seeded uniform random | none |
| `candidate_reranking_actsemble` | K (16) | highest component score | action-chunk compatibility |

The multi-sample control is a **compute-matched sampling control**, not a
strong selector: it separates "benefit from sampling more policy outputs"
from "benefit from the learned component". All three use the *same frozen
policy checkpoint*, the *same EMA weights*, the *same normalization*, the
*same diffusion sampler settings*, and — for the control vs. Actsemble —
bitwise-**identical candidate sets** under paired seeds (verified by test).

The initial component is an **action-chunk compatibility model**
C_φ(s_{t−H_o+1:t}, a_{t:t+H_p−1}): an MLP trained to score how compatible a
candidate action chunk is with successful behavior in the dataset.
Positives are real demonstration windows; negatives are deterministic
transformations of same-dataset chunks (noise, scaling, offsets, temporal
shifts, chunks from other trajectories, reversals, discontinuities, ...).
It is **not** a reward model, success oracle, value function, or
simulator-trained critic — and:

> High offline compatibility accuracy does not imply improved closed-loop
> task success. Closed-loop evaluation is the decisive test.

## Phase 0 setup

* **Simulator** ManiSkill3 (3.0.1) / SAPIEN 3, task **PushT-v1**
  (`panda_stick` fixed-base arm pushing a T onto a target).
* **Controller** `pd_ee_delta_pose` — 6-D continuous end-effector delta
  pose in [−1, 1].
* **Physics backend** `physx_cuda` (matches the recorded demonstrations;
  single-env evaluation is bitwise deterministic in-process).
* **Observations** flat 31-D privileged state:
  `agent.qpos [0,7) | agent.qvel [7,14) | extra.tcp_pose [14,21) |
  extra.goal_pos [21,24) | extra.obj_pose [24,31)`.
  Rendered videos are for inspection only — images are never model inputs.
* **Policy** state-conditioned Diffusion Policy: 1-D temporal U-Net
  (FiLM-conditioned residual conv blocks) predicting added noise over an
  action chunk; DDPM training (100 steps, cosine schedule), DDIM inference
  (10 steps); obs history 2, prediction horizon 16, execution horizon 8;
  EMA weights for evaluation.
* **Demonstrations** the official ManiSkill PushT-v1 RL demo bundle
  (719/719 successful episodes). The bundle was recorded in 1024 parallel
  GPU envs; GPU PhysX dynamics are batch-size dependent, so single-env
  action replay does not reproduce them (measured ~1/8). The dataset is
  therefore exported by **state-projection replay**: each recorded env
  state is restored and the state observation computed from it, yielding
  exactly the recorded (s_t, a_t, s_{t+1}) transitions; success is
  re-evaluated from the final state with the task's own state-based
  success function. Recorded actions are clipped to the controller bounds
  at export (the controller clips identically at execution).

## Install

```bash
conda create -n actsemble python=3.11 -y
conda activate actsemble
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -e ".[dev,sim]"
python -m mani_skill.utils.download_demo "PushT-v1"
```

## Smoke test

```bash
python scripts/smoke_test.py
```

Runs the entire pipeline end-to-end on a few episodes with tiny models —
22 checks covering env instantiation, conversion, validation, training,
sampling determinism, all three systems, rollouts, videos, and hash
identity. It verifies **correctness, not success rate**, and exits nonzero
on failure.

## Full experiment workflow

```bash
# 1. Prepare the frozen dataset (719 successful episodes)
python scripts/prepare_dataset.py --config configs/data/pilot.yaml \
    --output data/push_t_pilot.h5

# 2. Inspect + validate + render a few demonstration videos
python scripts/inspect_dataset.py --dataset data/push_t_pilot.h5 \
    --video-dir outputs/dataset_videos

# 3. Train the diffusion policy (~30k steps)
python scripts/train_policy.py --config configs/policies/state_diffusion.yaml \
    --dataset data/push_t_pilot.h5 --output-dir outputs/policy_pilot

# 4. Train the compatibility component on the SAME dataset
python scripts/train_component.py \
    --config configs/components/action_chunk_compatibility.yaml \
    --dataset data/push_t_pilot.h5 --output-dir outputs/component_pilot

# 5. Evaluate the three systems on paired seeds (repeat per eval config)
python scripts/evaluate.py \
    --system-config configs/systems/standalone_diffusion.yaml \
    --policy-checkpoint outputs/policy_pilot/best_ema.pt \
    --eval-config configs/evaluation/nominal.yaml \
    --output outputs/pilot/eval_standalone_nominal.json

python scripts/evaluate.py \
    --system-config configs/systems/multisample_control.yaml \
    --policy-checkpoint outputs/policy_pilot/best_ema.pt \
    --eval-config configs/evaluation/nominal.yaml \
    --output outputs/pilot/eval_multisample_nominal.json

python scripts/evaluate.py \
    --system-config configs/systems/compatibility_reranking.yaml \
    --policy-checkpoint outputs/policy_pilot/best_ema.pt \
    --component-checkpoint outputs/component_pilot/best.pt \
    --eval-config configs/evaluation/nominal.yaml \
    --output outputs/pilot/eval_actsemble_nominal.json

# 6. Compare (verifies same task/dataset/policy/seeds before reporting)
python scripts/compare_systems.py \
    --standalone outputs/pilot/eval_standalone_nominal.json \
    --control outputs/pilot/eval_multisample_nominal.json \
    --actsemble outputs/pilot/eval_actsemble_nominal.json
```

Perturbed regimes: swap `--eval-config` for
`configs/evaluation/{action_noise,action_latency,observation_noise,object_nudge,combined_mild}.yaml`.
All use the same root seed, so environment seeds stay paired across
regimes and systems.

## Dataset schema (HDF5)

```
metadata/            project, schema version, simulator+version, task, robot,
                     obs mode, state dim+layout, controller, action dim+definition
                     (semantics/frame/units/bounds/scaling/clipping), control freq,
                     physics backend, source, seed, content hash
episodes/<ep_id>/    state [T,31] · previous_action [T,6] · action [T,6] ·
                     next_state [T,31] · step_index [T]
```

Alignment is exactly (s_t, a_t, s_{t+1}): `state[t]` is observed, `action[t]`
executed, `next_state[t]` results; `state[t+1] == next_state[t]` is validated.
The terminal state appears only as `next_state[T-1]` and has no action.
Observation histories pad left by replicating s_0; action chunks pad right by
replicating a_{T−1}; both paddings carry explicit masks (loss masking is
configurable, default off, matching standard Diffusion Policy).
Reward, success, failure, progress, and termination reasons are **never**
stored in episode arrays — success-only filtering lives in a private
`*.provenance.json` sidecar. Splits are episode-disjoint and hashed;
dataset, split, and normalization hashes are stored in every checkpoint and
enforced at assembly time.

## Checkpoint selection and final evaluation

Comparative experiments follow the frozen two-stage selection protocol —
fixed-budget training with interval snapshots, RNG-isolated screening
(panel seed 2000), confirmation (seed 3500) with lexicographic selection,
offline-only verifier selection, system freezing, integration checks
(seed 4500), and gated final tests (seed 1000, 500 episodes/seed, ≥5
policy seeds) aggregated across training seeds. See
[docs/checkpoint_selection_protocol.md](docs/checkpoint_selection_protocol.md);
driver: `scripts/run_protocol.py` with `configs/protocol/default.yaml`.

## Evaluation protocol

* Paired (environment, policy-sampling, perturbation) seeds per episode
  come from fixed disjoint panels — identical across systems; candidate
  tensors are hashed per replan and verified identical across systems up
  to the first selection divergence.
* Primary metric `success_once` (PushT terminates on success;
  `success_at_end` is recorded alongside).
* Reports: success rate with Wilson 95% CI, paired win/loss/tie, bootstrap
  CI for the paired success difference, timeout/exception/fallback/clip
  rates, policy/component/decision latency; results below 50 episodes are
  flagged as not statistically meaningful.
* Representative success/failure rollout videos are saved per system;
  `compare_systems.py` prints the seeds where systems disagree so those
  episodes can be re-rendered and inspected.

## Repository map

```
configs/            tasks, data, policies, components, systems, evaluation
scripts/            prepare/inspect/train/evaluate/compare/smoke_test
src/actsemble/
  data/             schema, writer, reader, validation, windows, normalization
  policies/         ActionChunkPolicy interface; diffusion model/scheduler/sampling
  components/       LearnedComponent interface; action-chunk compatibility
  systems/          standalone, multi-sample control, candidate reranking, factory
  sim/              env factory, adapters, demonstration source, rollout,
                    perturbations (action noise/latency, obs noise, object nudge)
  training/         policy + component training loops (no simulator imports)
  evaluation/       paired seeds, metrics, evaluator, reports, video
tests/              81 tests; `pytest` is sim-free, `pytest -m sim` adds env tests
docs/               phase_0_scope, experiment_contract, adding_components
```

## Current limitations

* Privileged 31-D state observations (deliberate for Phase 0) — no vision.
* One task (PushT-v1), one component type, K=16 candidate reranking only.
* Demonstrations converted by state projection: transitions reflect
  record-time GPU dynamics (1024 parallel envs), which single-env
  evaluation cannot bitwise reproduce; closed-loop control absorbs the
  gap, and backend/controller identity is enforced everywhere.
* `object_nudge` recoverability is verified empirically (mild severities
  keep closed-loop success well above zero), not by a privileged expert.
* Offline component metrics are reported per negative type, but negative
  realism is untested against real failure modes — by design, only
  closed-loop results count.

## From state to vision (later phases)

The observation contract is isolated behind `StateObservation` and the
dataset schema; Phase 1 swaps the state vector for RGB + proprioception +
previous actions while keeping the policy/component/system interfaces, the
dataset contract, the perturbation framework, and the paired evaluation
protocol unchanged. See [docs/phase_0_scope.md](docs/phase_0_scope.md).

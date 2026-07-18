# Non-learned consensus selectors

Deterministic self-consistency rules that select one action chunk from the `K`
chunks the frozen Diffusion Policy samples at a replanning step. They use **no
learned component** and serve as controls between "execute candidate zero" and
the learned action-chunk verifier: *do the gains from sampling `K` chunks come
from generic consensus, without training anything?*

All selectors receive the identical candidate tensor `[K, prediction_horizon,
action_dim]` that every paired system receives, reuse `ReplanningSystemBase`
unchanged (sampling, history, execution-horizon queueing, deterministic
candidate seeds, validity, hashing, reset, latency), and implement **only**
selection. They always return the index of a real policy-generated candidate —
never an average.

Implementation: [`src/actsemble/systems/consensus_selection.py`](../src/actsemble/systems/consensus_selection.py).
Configs: [`configs/systems/`](../configs/systems/) (`full_chunk_medoid`,
`early_weighted_medoid`, `coordinate_median_projection`, `largest_cluster_medoid`).

## Normalization

Distances are computed on the **policy/checkpoint action normalization**, so the
rules stay scale-valid for future tasks whose action dimensions have different
scales. For candidate `A` we use `Ã = normalizer.normalize_action(A)`. Selection
math runs on CPU in `float64` for exact, GPU-independent determinism. The raw
candidate tensor is never modified and the returned action is the original raw
candidate — normalization affects only *which index* is chosen.

## Distance metric

Squared-Euclidean over the normalized chunk, optionally weighted per timestep:

```
d(A_i, A_j) = Σ_τ w_τ · Σ_d ( Ã_i[τ,d] − Ã_j[τ,d] )²
```

`ChunkDistance` flattens chunks to `[K, prediction_horizon·action_dim]` and
applies a per-feature weight `w_feature[τ·A+d] = w_τ`, so one weighted reduction
serves both the uniform (`w_τ = 1`) and early-weighted variants.

## Selectors

**1. Full-chunk medoid** (`full_chunk_medoid`). Uniform weights. Pick the valid
candidate with the smallest summed distance to the other valid candidates:
`score_i = Σ_j d(A_i, A_j)`, `i* = argmin_i score_i`.

**2. Early-action-weighted medoid** (`early_weighted_medoid`). Same medoid rule
with `w_τ = exp(−λτ)`, normalized so `Σ_τ w_τ = 1`. Earlier actions weigh more,
because only the first execution-horizon actions run before replanning. Default
`early_weight_decay: λ = 0.25` (frozen default; never tuned on a final panel).

**3. Coordinate-wise median projected to a candidate**
(`coordinate_median_projection`). Compute the coordinate-wise median chunk
`M[τ,d] = median_k Ã_k[τ,d]` over valid candidates. `M` is **not executed** (it
may mix incompatible modes or be a chunk the policy never produced); instead
select the real candidate nearest to it: `i* = argmin_i d(A_i, M)` (uniform
normalized squared-Euclidean).

**4. Largest-cluster medoid** (`largest_cluster_medoid`). Cluster the valid
normalized chunks with a deterministic two-cluster k-medoids, take the largest
cluster, and execute its medoid. k-medoids: medoid 1 = candidate zero (or lowest
valid index if candidate zero is invalid), medoid 2 = valid candidate farthest
from medoid 1; alternate assignment / medoid recomputation until assignments are
stable or `max_iterations` (default 20). No randomized initialization. The
clustering is modular so a distance-threshold agglomerative variant can be added
later (`selection.clustering.algorithm`).

## Deterministic tie-breaking

- Medoid ties → **lowest candidate index** (`argmin` over ascending valid
  indices).
- Coordinate-median ties → lowest candidate index.
- Largest-cluster ties → larger size, then **lower mean within-cluster
  distance**, then the cluster containing the **lowest candidate index**; medoid
  ties within the chosen cluster → lowest index.
- All candidates identical → candidate zero.

## Invalid-candidate handling

A candidate is invalid if it contains NaN/inf, has the wrong shape (rejected
upstream in `_replan`), or violates candidate validity. Distances **exclude**
invalid candidates (their rows are zeroed only so the distance arithmetic never
touches NaN/inf; they are never selectable). Then:

- exactly one valid candidate → select it (recorded as a fallback if it isn't
  candidate zero);
- no valid candidate → **fall back to candidate zero**, `fallback=True`,
  `fallback_reason="no_valid_candidates"`;
- a selector never terminates an episode and never converts an invalid candidate
  into a valid one; the fallback reason is always recorded.

## Computational complexity

Per replan: `O(K² · prediction_horizon · action_dim)` for the pairwise distance
matrix (`K²` chunk comparisons), plus `O(K²)` for medoid scoring and, for
largest-cluster, `O(iterations · K²)`. For `K=16, H_p=16, A=6` the matrix is
`16·16·96` — sub-millisecond on CPU (~0.25 ms measured, vs ~18 ms for one policy
sampling step). The full pairwise matrix is **not** persisted by default in large
evaluations (summary statistics only); set `selection.diagnostic_mode: true` to
log it.

## Diagnostics (per replan)

Common: selector type, selected index, number of valid candidates, whether the
selection differs from candidate zero, selector latency, fallback + reason,
per-valid-candidate consensus/medoid score, minimum selected score, distance from
the selected candidate to candidate zero, mean and max pairwise candidate
distance. Early-weighted adds the timestep weights and `λ`; coordinate-median
adds every candidate's distance to the median; largest-cluster adds the cluster
assignment, sizes, selected cluster, selected medoid, within-cluster mean
distances, and iteration count.

## Adding another non-learned selector

1. Subclass `ConsensusSelector` in `consensus_selection.py`; implement
   `select(flat, valid_idx, dist, record, *, diag_mode)` returning a **global**
   candidate index, writing your diagnostics into `record`, and breaking ties by
   lowest index. Reuse `ChunkDistance` for distances.
2. Register it in `build_selector(...)` and add its name to `SELECTOR_TYPES`.
   `factory.build_system` and `SYSTEM_TYPES` pick it up automatically.
3. Add `configs/systems/<name>.yaml` (K=16, `components: []`, `selection.type:
   <name>`, no component-checkpoint field).
4. Add unit tests to `tests/systems/test_consensus_selectors.py` (determinism,
   no-mutation, invalid handling, and a handcrafted case with a known answer).

## Scientific protocol

These selectors are controls. A later **versioned** experiment
(`experiments/selector_baselines_v1/`) with a new frozen spec and a new untouched
final-test panel will compare candidate zero vs. the four consensus selectors
vs. the verifier argmax — the key question being *generic self-consistency
versus a separately trained same-data verifier*. The development-panel numbers
are diagnostic only and must not be used to select a method.

# scripts/ — what's durable, what's disposable

53 files accumulate fast during fast iteration. This exists so that a cleanup
pass is **mechanical rather than archaeological**: at any point you should be
able to delete every `probe` without reading it, and know that nothing breaks.

## The three tiers

| tier | meaning | deletion rule |
|---|---|---|
| **infra** | other code imports it, or the protocol depends on it | never delete casually; if ≥2 files import it, it belongs in `src/` |
| **driver** | a reusable entry point you'd run again on new data | keep while its task line is alive |
| **probe** | answered one question, once | **delete as soon as its finding is written down**, and record WHERE |

A probe's value is its *finding*, not its code. Once the finding is in
`docs/`, memory, or a config comment with the numbers, the script is a liability
— it implies a result is reproducible when the code around it has moved on.

## Rule of thumb going in

1. If you `import` it from a second file, it is no longer a script. Move it to
   `src/actsemble/` and give it a test.
2. A probe must name, in its docstring, where its answer is recorded. If it
   can't, it's not finished.
3. Prefer a flag on an existing driver over a new script.

## Load-bearing code currently in scripts/ (migrate to src/ at the next cleanup)

These are imported by other files, so they are already libraries in all but
location. Deleting or moving them breaks callers silently.

| file | imported by | should become |
|---|---|---|
| `evaluate.py` | 16 | `src/actsemble/evaluation/` entry point |
| `recovery_oracle_pickcube.py` | 9 | `src/actsemble/recovery/pickcube.py` |
| `train_component.py` | 8 | `src/actsemble/training/` |
| `compare_systems.py`, `compare_temporal.py` | 3 each | analysis module |
| `visualize_recovery_pickcube.py` | 3 | `src/actsemble/evaluation/viz/` |

## The 2026-07 PickCube recovery + dataset cluster

Added during the recovery/dataset work. Tiers and where each finding lives:

| file | tier | finding is recorded in | notes |
|---|---|---|---|
| `generate_pickcube_teleop.py` | **infra** | `configs/data/pickcube_teleop.yaml` | generates the DEFAULT pick-and-place dataset; deleting it makes the data unreproducible |
| `subset_dataset.py` | **driver** | — | nested subsets of a frozen dataset; needed for every n-demos sweep |
| `recovery_oracle_pickcube.py` | **infra** | memory `actsemble-recovery-oracle` | 9 importers — see migration table |
| `visualize_recovery_pickcube.py` | **infra** | — | 3 importers (annotate/filmstrip/grab helpers) |
| `overnight_canonical.py` | **driver** | — | train→screen→confirm→final for one family. **Name is legacy** (it described when it was first run); rename to `run_policy_pipeline.py`, keeping the `overnight_*.json` artifact prefix so existing records under `outputs/active_min/overnight/` stay mergeable |
| `analyze_ndemos_sweep.py` | **driver** | — | reads screening histories; works on partial sweeps |
| `visualize_dataset_comparison.py` | **driver** | — | replays a dataset by writing sim state; reusable for any dataset audit |
| `recovery_drive_height.py` | probe | drive curve: 2%→46% @ 0.14 m, dead zone 1% @ 0.08 m | superseded once rerun on the new policy |
| `recovery_height_ablation.py` | probe | **INVALIDATED** — see `outputs/.../invalidated_2026-07-21_dead_grasp_flag/README.md` | delete after the corrected rerun |
| `recovery_point_ablation.py` | probe | INVALIDATED (same cause) | delete |
| `recovery_target_comparison.py` | probe | INVALIDATED (same cause) | delete |
| `recovery_perfect_injection.py` | probe | **unanswered** — killed mid-run | keep until the 0.08 m dead-zone question is settled |
| `diagnose_pickcube_transport.py` | probe | memory `actsemble-pickcube-line` | delete |
| `visualize_joint_return.py` | probe | gripper-normalization bug (+1.0 = open) | finding is in `recovery.py` comments; delete |
| `visualize_reset_height.py` | probe | reset-height footage | delete with `recovery_drive_height.py` |
| `plot_recovery_height_curves.py` | probe | `reset_height_decision.png` | delete once the figure is in a doc |
| `failure_taxonomy.py` | probe | memory `actsemble-recovery-oracle` | grounded failure modes, both tasks |

**Deletable today: 4** (the three INVALIDATED ablations + `diagnose_pickcube_transport`).
Their results are already quarantined with a README explaining why they're wrong,
so the code only invites someone to re-run a known-broken measurement.

## Not classified

`phase0a*`, `probe_*_codex`, `sweep_*`, `selector_*`, `replicate_act.py`,
`passk_diagnostic.py` predate this file. They need an owner pass before a
cleanup — do not bulk-delete on the assumption they're probes.

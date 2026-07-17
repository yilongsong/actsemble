# Actsemble rollout-comparison dashboard (project standard)

A standardized, self-contained web interface for comparing closed-loop rollouts
across systems, experiments, and iterations. **This is the standard viewer for
the whole project** — any experiment that emits the manifest schema below renders
in the same interface. Iterate on the viewer (`_viewer.html`), not on per-experiment forks.

## View it

```bash
cd dashboard && python -m http.server 8000
# open http://localhost:8000/pushonly_selection.html
```

(Opening the `.html` directly via `file://` mostly works — the data is inlined —
but a local server is recommended so the `<video>` elements load reliably.)

Tabs: **Overview** (per-system success + CIs + paired contrasts + headroom),
**Failure modes** (outcome composition + recovery cross-tab), **Episodes**
(filter/sort every rollout; each shows a T-block trajectory sparkline, coverage
curve, auto-classified failure badge, metrics, and mp4 if rendered), **Compare**
(two systems side by side on the same seeds — e.g. base-fails vs oracle-recovers).

## Produce data

```bash
# rich rollout capture for ANY selector system (trajectory + video + classification)
python scripts/build_dashboard.py record \
  --system '{"name":"verifier_argmax","label":"Verifier","kind":"selector","color":"#6a4c93",
             "selection":{"type":"highest_component_score"},"num_candidates":16,
             "components":["outputs/pushonly_min/verifier_seed_0/selected_verifier.pt"]}' \
  --policy outputs/pushonly_min/policy_seed_0/selected_policy.pt --count 300 --videos 6

# oracle rollouts come from scripts/oracle_headroom.py (capture / video)

# assemble everything on disk into one dashboard html
python scripts/build_dashboard.py build --name pushonly_selection
```

`build` ingests: rich captures (`dashboard/captures/*.json`, base/oracle capture
jsons), paired-eval jsons (`outputs/.../compare/*.json`, success-only), and the
oracle headroom — then writes `dashboard/<name>.html` (+ `<name>.manifest.json`)
and copies referenced videos into `dashboard/videos/`.

## Manifest schema (the stable contract) — v1.0

```jsonc
{
  "schema_version": "1.0",
  "experiment": "pushonly_selection", "title": "...", "created": "...",
  "panel": {"name":"...", "env_seed":20000, "num_episodes":300},
  "goal_xy": [-0.156,-0.10], "goal_tolerance": 0.05,
  "world": {"x0":-0.46,"x1":0.24,"y0":-0.36,"y1":0.14},   // trajectory plot extent
  "failure_taxonomy": { "<type>": {"label":"...","color":"#..."}, ... },
  "mode_order": ["success","near_miss","misaligned","mispositioned","never_engaged","unknown"],
  "systems": [{
    "name","label","kind","color",
    "metrics": {"success_rate","success_count","n","wilson_ci","mean_steps","mean_final_cov"},
    "failure_counts": {"<type>": <int>, ...},
    "episodes": [{
      "env_seed", "success": <bool>, "failure_type": "<taxonomy key>",
      "final_cov","max_cov","steps","pos_error","obj_disp",
      "traj": {"obj":[[x,y],...~28], "cov":[...~28]} | null,   // downsampled; null = metrics-only
      "video": "videos/xxx.mp4" | null
    }, ...]
  }, ...],
  "contrasts": [{"a","b","delta","mcnemar_z","sig","a_wins","b_wins","n","note"}, ...],
  "highlights": [{"type":"headroom","oracle","cz","verifier","captured","k","note"}, ...],
  "recovery": {"title":"...","rows":[{"mode","count","recovered"}, ...]}   // optional
}
```

## Canonical failure taxonomy

Outcome classification (from the T-block trajectory), applied uniformly by every producer:

| type | rule |
|---|---|
| `success` | reached ≥90% goal-area coverage (episode terminates) |
| `never_engaged` | T-block moved <2 cm all episode |
| `near_miss` | max coverage ≥0.75 but ended <0.90 (got close, fell back) |
| `misaligned` | T-block within 5 cm of goal center but wrong orientation |
| `mispositioned` | T-block ended >5 cm from goal center |
| `unknown` | no trajectory captured (success-only source) |

The classifier lives in `scripts/build_dashboard.py:classify()` — one definition,
reused everywhere so labels are comparable across systems and experiments.

"""Across-training-seed aggregation (§14).

The trained model is the primary replication unit. For every policy seed
this computes Actsemble − standalone and Actsemble − control success
differences, then reports each seed-level difference, their mean, sample
standard deviation, a Student-t 95% confidence interval across seeds, and
sign counts. Episode-level paired analyses (the per-seed comparison.json
files) are secondary and conditional on the trained checkpoints. Never
report rollout episodes as independent training replications.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..utils.serialization import load_json, save_json

# Two-sided 95% Student-t critical values by degrees of freedom.
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
         15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
         25: 2.060, 30: 2.042}


def t_critical_95(df: int) -> float:
    if df <= 0:
        raise ValueError("Need at least 2 seeds for a confidence interval")
    keys = sorted(_T_95)
    return _T_95[min((k for k in keys if k >= df), default=keys[-1])]


def _seed_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    arr = np.asarray(diffs, dtype=np.float64)
    mean = float(arr.mean())
    out = {
        "per_seed_differences": diffs,
        "mean_difference": mean,
        "num_seeds": n,
        "positive_seeds": int((arr > 0).sum()),
        "negative_seeds": int((arr < 0).sum()),
        "zero_seeds": int((arr == 0).sum()),
    }
    if n >= 2:
        std = float(arr.std(ddof=1))
        half = t_critical_95(n - 1) * std / math.sqrt(n)
        out.update({
            "std_difference": std,
            "confidence_interval_95": [mean - half, mean + half],
            "ci_method": f"Student t, df={n - 1}",
        })
    else:
        out.update({"std_difference": None, "confidence_interval_95": None,
                    "ci_method": "unavailable (single seed)"})
    return out


def aggregate_experiment(experiment_dir: str | Path, *, regime: str = "nominal") -> dict:
    experiment_dir = Path(experiment_dir)
    per_seed = []
    for sdir in sorted(experiment_dir.glob("seed_*")):
        final_dir = sdir / "final_test" / regime
        if not final_dir.exists():
            continue
        rows = {name: load_json(final_dir / f"eval_{name}.json")
                for name in ("standalone", "control", "actsemble")}
        per_seed.append({
            "seed_dir": sdir.name,
            "policy_seed": int(sdir.name.split("_")[1]),
            "num_episodes": rows["standalone"]["num_episodes"],
            "success_rates": {n: r["success_rate"] for n, r in rows.items()},
            "actsemble_minus_standalone":
                rows["actsemble"]["success_rate"] - rows["standalone"]["success_rate"],
            "actsemble_minus_control":
                rows["actsemble"]["success_rate"] - rows["control"]["success_rate"],
        })
    if not per_seed:
        raise FileNotFoundError(
            f"No final-test results for regime {regime!r} under {experiment_dir}"
        )
    report = {
        "experiment_dir": str(experiment_dir),
        "regime": regime,
        "primary_results": {
            "actsemble_minus_standalone": _seed_stats(
                [s["actsemble_minus_standalone"] for s in per_seed]
            ),
            "actsemble_minus_control": _seed_stats(
                [s["actsemble_minus_control"] for s in per_seed]
            ),
        },
        "per_seed": per_seed,
        "note": ("Training seeds are the replication unit; episode-level paired "
                 "analyses are secondary and conditional on the trained checkpoints."),
    }
    save_json(report, experiment_dir / f"aggregate_{regime}.json")
    return report


def format_aggregate(report: dict) -> str:
    lines = ["=" * 74,
             f"Across-seed aggregate — regime {report['regime']}, "
             f"{len(report['per_seed'])} policy seed(s)",
             "=" * 74]
    for s in report["per_seed"]:
        rates = s["success_rates"]
        lines.append(
            f"{s['seed_dir']}: standalone {rates['standalone']:.1%}  "
            f"control {rates['control']:.1%}  actsemble {rates['actsemble']:.1%}  "
            f"(n={s['num_episodes']})"
        )
    for key, label in (("actsemble_minus_standalone", "Actsemble − standalone"),
                       ("actsemble_minus_control", "Actsemble − control")):
        st = report["primary_results"][key]
        ci = st["confidence_interval_95"]
        lines.append("")
        lines.append(f"{label}:")
        lines.append(f"  per-seed: {[f'{d:+.1%}' for d in st['per_seed_differences']]}")
        lines.append(f"  mean {st['mean_difference']:+.2%}"
                     + (f", sd {st['std_difference']:.2%}" if st["std_difference"] is not None else ""))
        if ci:
            lines.append(f"  95% CI across seeds [{ci[0]:+.2%}, {ci[1]:+.2%}] ({st['ci_method']})")
        lines.append(f"  seeds +/−/0: {st['positive_seeds']}/{st['negative_seeds']}/{st['zero_seeds']}")
    return "\n".join(lines)

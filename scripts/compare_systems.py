#!/usr/bin/env python
"""Compare standalone / multi-sample control / Actsemble evaluation results.

Verifies the runs are fairly comparable (same task, dataset, policy
checkpoint, controller, backend, and paired seeds) before reporting.

Usage:
    python scripts/compare_systems.py \
        --standalone outputs/eval_standalone.json \
        --control outputs/eval_multisample.json \
        --actsemble outputs/eval_actsemble.json \
        [--output outputs/comparison.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsemble.evaluation.reports import compare_systems, format_report
from actsemble.utils.serialization import load_json, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", required=True, help="baseline result JSON")
    parser.add_argument("--control", default=None, help="multi-sample control result JSON")
    parser.add_argument("--actsemble", default=None, help="Actsemble result JSON")
    parser.add_argument("--output", default=None, help="save the comparison report JSON")
    args = parser.parse_args()

    results = [load_json(args.standalone)]
    if args.control:
        results.append(load_json(args.control))
    if args.actsemble:
        results.append(load_json(args.actsemble))
    if len(results) < 2:
        print("Need at least two result files to compare", file=sys.stderr)
        return 2

    report = compare_systems(results, baseline_index=0)

    # Surface paired disagreements so failures can be re-rendered and inspected.
    base = results[0]
    for r in results[1:]:
        seeds_won = [
            s for s, (a, b) in zip(base["environment_seeds"],
                                   zip(r["successes"], base["successes"]))
            if a and not b
        ]
        seeds_lost = [
            s for s, (a, b) in zip(base["environment_seeds"],
                                   zip(r["successes"], base["successes"]))
            if b and not a
        ]
        report["systems"][r["system_name"]]["seeds_won_vs_baseline"] = seeds_won
        report["systems"][r["system_name"]]["seeds_lost_vs_baseline"] = seeds_lost

    print(format_report(report))
    for name, s in report["systems"].items():
        if s.get("seeds_won_vs_baseline"):
            print(f"{name} succeeded where baseline failed on env seeds: "
                  f"{s['seeds_won_vs_baseline']}")
        if s.get("seeds_lost_vs_baseline"):
            print(f"{name} failed where baseline succeeded on env seeds: "
                  f"{s['seeds_lost_vs_baseline']}")

    if args.output:
        save_json(report, args.output)
        print(f"\n[compare] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""n-demos sweep: success vs EPOCH, and does convergence fit in 500 epochs?

Two questions, one table:

  1. operating point -- which n lands the policy in the protocol's target band?
  2. is the max_epochs=500 budget actually enough? The canonical config trains
     a fixed number of EPOCHS, which silently gives smaller datasets fewer
     gradient steps (n=50 got 5000 vs n=100's 10500). Running every cell to a
     fixed STEP budget instead makes the small cells run far past epoch 500, so
     the screening curve can show whether anything is still improving up there.
     If every cell peaks before epoch 500, the assumption holds; if the small
     cells keep climbing past it, max_epochs=500 was undertraining them.

Reads each run's screening history (written DURING training) so this works on
partial sweeps.

    python scripts/analyze_ndemos_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = (0.40, 0.60)


def load_cell(run_dir: Path):
    hist = run_dir / "diffusion" / "screening" / "screening_history.json"
    if not hist.exists():
        return None
    rows = json.load(open(hist))
    cfg = run_dir / "diffusion" / "train_config.json"
    spe = None
    if cfg.exists():
        d = json.load(open(cfg))
        spe = (d.get("split") or {}).get("steps_per_epoch") or d.get("steps_per_epoch")
    if not spe:
        # infer: snapshots are on a fixed step interval; fall back to the
        # documented values rather than guessing wrong.
        spe = None
    return rows, spe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "outputs/pickcube/teleop_v1"))
    ap.add_argument("--steps-per-epoch", default="100:21,50:10,25:5,10:2",
                    help="n:steps_per_epoch pairs (reported by the trainer)")
    args = ap.parse_args()

    spe_map = {int(k): int(v) for k, v in
               (p.split(":") for p in args.steps_per_epoch.split(","))}
    root = Path(args.root)
    cells = {}
    for n in (10, 25, 50, 100):
        for cand in (root / f"sweep_n{n}", root / ("baseline" if n == 100 else f"baseline_n{n}")):
            got = load_cell(cand)
            if got:
                cells[n] = (got[0], cand)
                break
    if not cells:
        print("no screening histories found yet", file=sys.stderr)
        return 1

    print(f"{'n':>5} {'peak':>7} {'@step':>7} {'@epoch':>8} {'final':>7} "
          f"{'epochs run':>11}  in band?  curve (success % per snapshot)")
    for n in sorted(cells):
        rows, path = cells[n]
        spe = spe_map.get(n, 1)
        best = max(rows, key=lambda r: r["success_rate"])
        last = rows[-1]
        band = TARGET[0] <= best["success_rate"] <= TARGET[1]
        print(f"{n:>5} {100 * best['success_rate']:>6.0f}% {best['step']:>7} "
              f"{best['step'] / spe:>8.0f} {100 * last['success_rate']:>6.0f}% "
              f"{last['step'] / spe:>11.0f}  {'YES' if band else 'no ':>7}   "
              + " ".join(f"{100 * r['success_rate']:.0f}" for r in rows))

    print("\nDoes convergence fit inside 500 epochs?")
    for n in sorted(cells):
        rows, _ = cells[n]
        spe = spe_map.get(n, 1)
        ran = rows[-1]["step"] / spe
        if ran < 500:
            print(f"  n={n:<4} ran only {ran:.0f} epochs -- cannot answer")
            continue
        early = [r for r in rows if r["step"] / spe <= 500]
        late = [r for r in rows if r["step"] / spe > 500]
        be = max(early, key=lambda r: r["success_rate"])["success_rate"] if early else 0.0
        if not late:
            print(f"  n={n:<4} peak {100 * be:.0f}% by epoch 500; nothing ran past it")
            continue
        bl = max(late, key=lambda r: r["success_rate"])["success_rate"]
        gain = 100 * (bl - be)
        verdict = ("500 epochs was ENOUGH" if gain <= 2
                   else f"still improving past 500 (+{gain:.0f}pt)")
        print(f"  n={n:<4} best<=500ep {100 * be:.0f}%  best>500ep {100 * bl:.0f}%"
              f"  ({ran:.0f} epochs run)  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

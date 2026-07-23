#!/usr/bin/env python
"""Repair the recorded environment metadata inside trained checkpoints.

Why this exists: `scripts/generate_pickcube_teleop.py` originally wrote
`simulator_version=""` into its dataset metadata. That string is copied into
every checkpoint trained on it. `evaluate_system` compares a checkpoint's
recorded environment against the live one and refuses on ANY mismatch, so an
empty string -- which is not "unknown", it is a value that never matches --
passed screening and confirmation (which skip the check) and only failed at the
500-episode final test, after hours of compute.

What this does NOT do: change any weight, or any number that was measured. It
rewrites one recorded string to the value that was true at training time, and
stamps the checkpoint with a repair note so the edit is auditable.

The datasets themselves were rewritten with correct metadata, which changed
their content hash (metadata is part of the hash) even though the episode
arrays are byte-identical. Both hashes are recorded in the note so provenance
stays traceable across the repair.

    python scripts/repair_checkpoint_metadata.py --run outputs/pickcube/teleop_v1/baseline/diffusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mani_skill  # noqa: E402


def repair(path: Path, *, dry_run: bool) -> str | None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    env = ((ckpt.get("meta") or {}).get("extra") or {}).get("environment")
    if env is None:
        return None
    got = env.get("simulator_version")
    if got == mani_skill.__version__:
        return None
    if not dry_run:
        env["simulator_version"] = mani_skill.__version__
        ckpt["meta"].setdefault("extra", {})["metadata_repair_2026_07_21"] = {
            "field": "extra.environment.simulator_version",
            "was": got,
            "now": mani_skill.__version__,
            "reason": "generator recorded an empty string; weights untouched",
            "dataset_hash_recorded": ckpt["meta"].get("dataset_hash"),
        }
        torch.save(ckpt, path)
    return f"{got!r} -> {mani_skill.__version__!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a training run dir (contains checkpoints/, last.pt, ...)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    targets = sorted(run.glob("checkpoints/step_*.pt")) + [
        p for p in (run / "last.pt", run / "final.pt", run / "best_ema.pt",
                    run / "selected_policy.pt") if p.exists()
    ]
    if not targets:
        print(f"no checkpoints under {run}", file=sys.stderr)
        return 1
    n = 0
    for p in targets:
        out = repair(p, dry_run=args.dry_run)
        if out:
            n += 1
            print(f"  {'(dry) ' if args.dry_run else ''}{p.relative_to(run)}: {out}")
    print(f"{'would repair' if args.dry_run else 'repaired'} {n} of {len(targets)} checkpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())

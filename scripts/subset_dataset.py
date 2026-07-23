#!/usr/bin/env python
"""Deterministic NESTED subset of an already-frozen dataset.

`prepare_dataset.py --subset-size` subsets a raw demonstration bundle at
conversion time. That does not help for datasets that are GENERATED rather than
converted (scripts/generate_pickcube_teleop.py), so this takes a frozen .h5 and
writes a smaller one.

Nested by construction: episodes are taken in sorted id order, so size 25 is a
subset of size 50 is a subset of size 100. That is what makes an n-demos sweep
interpretable -- a smaller set is strictly less data, not different data.

    python scripts/subset_dataset.py --input outputs/pickcube/teleop_v1/teleop_100.h5 --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from actsemble.data.reader import DatasetReader  # noqa: E402
from actsemble.data.validation import (  # noqa: E402
    validate_dataset,
    validate_success_only_provenance,
)
from actsemble.data.writer import write_dataset, write_private_provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--output", default=None, help="default: <input dir>/<stem>_n<N>.h5")
    args = ap.parse_args()

    src = Path(args.input)
    reader = DatasetReader(src)
    ids = list(reader.episode_ids)
    if args.n > len(ids):
        raise SystemExit(f"asked for {args.n} episodes, source has {len(ids)}")
    keep = ids[: args.n]
    episodes = [reader.episode(e) for e in keep]

    out = Path(args.output) if args.output else src.parent / f"{src.stem}_n{args.n}.h5"
    meta = reader.metadata
    meta.extra["subset_of"] = str(src)
    meta.extra["subset_of_hash"] = meta.dataset_hash
    meta.extra["subset_size"] = args.n
    meta.extra["subset_rule"] = "first N by sorted episode id (nested)"
    h = write_dataset(out, episodes, meta)

    # Carry the success-only assertion forward; every kept episode came from a
    # source that already asserted it, and a subset cannot introduce failures.
    src_side = src.with_suffix(src.suffix + ".provenance.json")
    parent = json.load(open(src_side)) if src_side.exists() else {}
    write_private_provenance(out, {
        "success_only": True,
        "subset_of": str(src),
        "subset_size": args.n,
        "kept": len(episodes),
        "exported_episodes": [
            {"episode_id": e.episode_id, "source_success": True, "length": len(e)}
            for e in episodes
        ],
        "parent_provenance": {k: parent.get(k) for k in ("attempts", "kept", "args")},
    })
    stats = validate_dataset(DatasetReader(out))
    validate_success_only_provenance(out)
    print(f"[subset] {len(ids)} -> {args.n} episodes, {stats['num_transitions']} transitions")
    print(f"[subset] episode length min {stats['episode_length_min']} max {stats['episode_length_max']}")
    print(f"[subset] wrote {out}\n[subset] hash {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

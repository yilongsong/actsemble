#!/usr/bin/env python
"""WHICH reset height should recovery target? All curves on one axis.

Puts the teleport curves (what the policy would do given a perfectly specified
input) next to the PHYSICAL-DRIVE curve (what actually happens when the arm has
to travel there). The gap between them is the drive cost, and it is the whole
reason the decision is not "always reset next to the cube":

  teleport      : robot appears at the target -> no approach -> no contact risk.
                  Structurally HIDES the cost of near-grasp targets.
  physical drive: the arm must penetrate next to the cube to reach a near-grasp
                  target -> it can knock the cube (panel B) and it burns steps
                  from the episode budget (panel C).

Series in panel A, in increasing input completeness, so the U-shape can be read
as a function of how much of the policy's input was specified:
  zero-vel teleport (CONTROL, the original U)  <- recovery_height_ablation.json
  matched-vel teleport                          \
  perfect teleport (qpos+qvel+real prev frame)  /  perfect_injection.json
  physical drive (jaws open, joint-space)          drive_height.json
  no-recovery (let the failure play out)            hline from drive_height.json

    python scripts/plot_recovery_height_curves.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/pickcube/recovery_oracle"

# validated categorical palette (dataviz validator, light mode, all checks PASS:
# CVD dE 11.0 worst adjacent, normal-vision 18.7). Every series is ALSO direct-
# labelled and tabulated, which discharges the low-contrast warning on #E69F00.
C_ZERO = "#E69F00"      # zero-velocity teleport (control)
C_MATCHED = "#0072B2"   # matched-velocity teleport
C_PERFECT = "#009E73"   # perfect teleport
C_DRIVE = "#D55E00"     # physical drive
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"

# height landmarks, metres above the table (measured, see pickcube memory)
LANDMARKS = [(0.04, "cube top"), (0.168, "robot HOME"), (0.316, "demo max")]


def _label_end(ax, x, y, text, color):
    """Direct label at the right end of a series (never a number per point)."""
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=8.5, fontweight="bold", va="center")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", default=str(OUT / "recovery_height_ablation.json"))
    ap.add_argument("--injection", default=str(OUT / "perfect_injection.json"))
    ap.add_argument("--drive", default=str(OUT / "drive_height.json"))
    ap.add_argument("--out", default=str(OUT / "reset_height_decision.png"))
    args = ap.parse_args()

    def load(p):
        p = Path(p)
        return json.load(open(p)) if p.exists() else None

    ctrl, inj, drv = load(args.control), load(args.injection), load(args.drive)
    if ctrl is None and inj is None and drv is None:
        print("no result files yet", file=sys.stderr)
        return 1

    nrows = 3 if drv else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(8.2, 3.6 + 2.1 * (nrows - 1)),
                             sharex=True, gridspec_kw={"height_ratios": [3, 1.25, 1.25][:nrows]})
    axes = np.atleast_1d(axes)
    ax = axes[0]
    table = []   # (series, [(height, value)]) for the printed table

    def series(a, x, y, color, label, marker="o", ls="-", lw=2.0, alpha=1.0):
        a.plot(x, y, color=color, lw=lw, ls=ls, marker=marker, ms=5.5, alpha=alpha,
               markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3,
               clip_on=False)

    # ---- panel A: success vs reset height ------------------------------------
    if ctrl:
        x = [c["height"] for c in ctrl["curve"]]
        y = [100 * c["recovery"] for c in ctrl["curve"]]
        series(ax, x, y, C_ZERO, "teleport, zero velocity (control)", ls="--")
        _label_end(ax, x, y, "zero-vel", C_ZERO)
        table.append(("teleport zero-vel (control)", list(zip(x, y))))

    if inj:
        names = {"a_zero_vel": ("teleport, zero velocity", C_ZERO, "--"),
                 "b_matched_vel": ("teleport, matched velocity", C_MATCHED, "-"),
                 "c_perfect": ("teleport, PERFECT input (2 frames)", C_PERFECT, "-")}
        for arm, curve in inj["curves"].items():
            if arm == "a_zero_vel" and ctrl:
                continue          # already drawn from the larger control run
            lbl, col, ls = names.get(arm, (arm, MUTED, "-"))
            x = [c["height"] for c in curve]
            y = [100 * c["recovery"] for c in curve]
            series(ax, x, y, col, lbl, ls=ls)
            _label_end(ax, x, y, lbl.split(", ")[-1].replace(" input (2 frames)", ""), col)
            table.append((lbl, list(zip(x, y))))

    if drv:
        x = [c["height"] for c in drv["curve"]]
        y = [100 * c["success"] for c in drv["curve"]]
        series(ax, x, y, C_DRIVE, "PHYSICAL DRIVE (jaws open)", lw=2.6)
        _label_end(ax, x, y, "driven", C_DRIVE)
        table.append(("physical drive", list(zip(x, y))))
        base = 100 * drv["no_recovery"]
        ax.axhline(base, color=MUTED, lw=1.4, ls=":", zorder=2)
        ax.annotate(f"no recovery  {base:.0f}%", (0.0, base), xytext=(2, 4),
                    textcoords="offset points", color=MUTED, fontsize=8.5)

    for h, name in LANDMARKS:
        ax.axvline(h, color=GRID, lw=1.0, zorder=1)
        ax.annotate(name, (h, 1.005), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)

    ax.set_ylabel("success after reset (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=1)
    n_i = inj["n_scenes"] if inj else 0
    ax.set_title("Which reset height should recovery target?\n"
                 f"teleport = policy quality given the input · drive = what the arm can physically reach"
                 f"{f'  ({n_i} scenes)' if n_i else ''}",
                 fontsize=10.5, color=INK, loc="left", pad=18)

    # ---- panels B/C: what driving costs --------------------------------------
    if drv:
        xb = [c["height"] for c in drv["curve"]]
        series(axes[1], xb, [100 * c["drive_bump"] for c in drv["curve"]], C_DRIVE, "")
        axes[1].set_ylabel("cube knocked\nduring drive (cm)", fontsize=8.5, color=INK)
        axes[1].set_ylim(bottom=0)
        table.append(("cube bump during drive (cm)",
                      [(c["height"], 100 * c["drive_bump"]) for c in drv["curve"]]))

        series(axes[2], xb, [c["drive_steps"] for c in drv["curve"]], C_DRIVE, "")
        axes[2].set_ylabel("drive steps\n(taken from budget)", fontsize=8.5, color=INK)
        axes[2].set_ylim(bottom=0)
        table.append(("drive steps",
                      [(c["height"], c["drive_steps"]) for c in drv["curve"]]))
        for a in axes[1:]:
            for h, _ in LANDMARKS:
                a.axvline(h, color=GRID, lw=1.0, zorder=1)

    axes[-1].set_xlabel("reset height — fingertip z above table (m)", fontsize=9, color=INK)
    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.8, alpha=0.7, zorder=0)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=8.5, length=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, facecolor="white", bbox_inches="tight")

    # table view (accessibility relief + the numbers themselves)
    print(f"\nRESET-HEIGHT DECISION  ->  {args.out}\n")
    for name, pts in table:
        print(f"  {name}")
        print("    " + "".join(f"{h:>8.2f}" for h, _ in pts))
        print("    " + "".join(f"{v:>8.1f}" for _, v in pts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

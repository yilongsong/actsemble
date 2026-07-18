"""Failure-mode visualizations: base policy (no selection) vs rollout oracle.

Fig 1  failure modes WITHOUT oracle (base candidate_zero policy)
Fig 2  episodes the base policy FAILS but the oracle RECOVERS (selection headroom, made concrete)
Fig 3  failure modes OF the oracle (proposal-limited: no candidate ever wins)
"""

import json
import glob
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/active_min/oracle/capture")
FIG = Path("outputs/data_analysis")
FIG.mkdir(parents=True, exist_ok=True)
GOAL_YAW = (5 / 3) * np.pi

base = {
    e["env_seed"]: e for e in json.load(open(OUT / "base_0000_0300.json"))["episodes"]
}
orac = {}
for f in sorted(glob.glob(str(OUT / "oracle_*.json"))):
    for e in json.load(open(f))["episodes"]:
        orac[e["env_seed"]] = e

CATS = ["never engaged", "near miss (fell back)", "misaligned at goal", "mispositioned"]
CCOL = {
    "never engaged": "#8d99ae",
    "near miss (fell back)": "#e76f51",
    "misaligned at goal": "#e9c46a",
    "mispositioned": "#457b9d",
}


def categorize(ep):
    cov = np.asarray(ep["coverage"])
    obj = np.asarray(ep["obj_xy"])
    goal = np.asarray(ep["goal_xy"])
    disp = np.linalg.norm(obj - obj[0], axis=1).max()
    pos_err = np.linalg.norm(obj[-1] - goal)
    if disp < 0.02:
        return "never engaged"
    if cov.max() >= 0.75:
        return "near miss (fell back)"
    if pos_err <= 0.05:
        return "misaligned at goal"
    return "mispositioned"


def bar(ax, labels_counts, title):
    labels = [c for c in CATS if c in labels_counts]
    vals = [labels_counts[c] for c in labels]
    ax.bar(range(len(labels)), vals, color=[CCOL[c] for c in labels])
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("# episodes")
    ax.grid(alpha=0.25, axis="y", lw=0.5)


def draw_T_paths(ax, ep, color, lw=1.3, mark=True):
    obj = np.asarray(ep["obj_xy"])
    ax.plot(obj[:, 0], obj[:, 1], color=color, lw=lw, alpha=0.9)
    if mark:
        ax.plot(*obj[0], "o", ms=4, color="#111", alpha=0.7)
        ax.plot(*obj[-1], "x", ms=6, color=color, mew=2)


# ============================ FIG 1: base failure modes ============================
base_fail = {es: e for es, e in base.items() if not e["success_once"]}
cats1 = {}
for e in base_fail.values():
    c = categorize(e)
    cats1[c] = cats1.get(c, 0) + 1
goal = np.asarray(next(iter(base.values()))["goal_xy"])

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle(
    f"Failure modes WITHOUT oracle — base candidate_zero policy "
    f"({len(base_fail)}/{len(base)} episodes fail)",
    fontsize=13,
    fontweight="bold",
)
bar(ax[0], cats1, "(a) Failure-mode counts")

for es, e in base_fail.items():
    c = categorize(e)
    obj = np.asarray(e["obj_xy"])
    ax[1].plot(obj[-1, 0], obj[-1, 1], "o", ms=5, color=CCOL[c], alpha=0.6)
ax[1].plot(*goal, "*", ms=22, color="#e9c46a", mec="#7a5901", mew=1.2, zorder=10)
ax[1].add_patch(plt.Circle(goal, 0.05, fill=False, ls="--", ec="#7a5901", lw=1))
ax[1].set_title("(b) Where the T ends up (final position)")
ax[1].set_aspect("equal")
ax[1].set_xlabel("x (m)")
ax[1].set_ylabel("y (m)")
ax[1].grid(alpha=0.25, lw=0.5)
handles = [
    plt.Line2D([], [], marker="o", ls="", color=CCOL[c], label=c)
    for c in CATS
    if c in cats1
]
ax[1].legend(handles=handles, fontsize=7, loc="upper right")

for es, e in base_fail.items():
    c = categorize(e)
    cov = np.asarray(e["coverage"])
    ax[2].plot(np.linspace(0, 1, len(cov)), cov, color=CCOL[c], lw=0.5, alpha=0.25)
ax[2].axhline(0.9, ls="--", c="k", lw=1, label="success (0.90)")
ax[2].set_title("(c) Goal coverage over time (failures)")
ax[2].set_xlabel("normalized time")
ax[2].set_ylabel("goal-area coverage")
ax[2].set_ylim(0, 1)
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.25, lw=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fail1_base_modes.png", dpi=125)
plt.close(fig)

# ================== FIG 2: base fails, oracle recovers (headroom made concrete) ==================
recov = [
    es
    for es in orac
    if es in base and not base[es]["success_once"] and orac[es]["success_once"]
]
recov = sorted(
    recov, key=lambda es: -np.asarray(base[es]["coverage"]).max()
)  # base got closest first
pick = recov[:8]
fig, ax = plt.subplots(2, 4, figsize=(16.5, 8.4))
fig.suptitle(
    f"Base policy FAILS → Oracle RECOVERS (same init): selection headroom made concrete "
    f"({len(recov)} such episodes in the {len(orac)} shown)",
    fontsize=13,
    fontweight="bold",
)
for a, es in zip(ax.flat, pick):
    draw_T_paths(a, base[es], "#adb5bd", lw=1.6)  # base: grey, fails
    draw_T_paths(a, orac[es], "#2a9d8f", lw=1.6)  # oracle: teal, succeeds
    a.plot(*goal, "*", ms=18, color="#e9c46a", mec="#7a5901", mew=1.1, zorder=10)
    a.add_patch(plt.Circle(goal, 0.05, fill=False, ls="--", ec="#7a5901", lw=0.8))
    bc = np.asarray(base[es]["coverage"]).max()
    oc = np.asarray(orac[es]["coverage"]).max()
    a.set_title(f"seed {es}\nbase max cov {bc:.2f} · oracle {oc:.2f}", fontsize=8)
    a.set_aspect("equal")
    a.grid(alpha=0.25, lw=0.5)
    a.tick_params(labelsize=6)
for a in ax.flat[len(pick) :]:
    a.axis("off")
ax.flat[0].plot([], [], color="#adb5bd", lw=2, label="base (fails)")
ax.flat[0].plot([], [], color="#2a9d8f", lw=2, label="oracle (succeeds)")
ax.flat[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / "fail2_oracle_recovers.png", dpi=125)
plt.close(fig)

# ============================ FIG 3: oracle failure modes ============================
orac_fail = {es: e for es, e in orac.items() if not e["success_once"]}
cats3 = {}
for e in orac_fail.values():
    c = categorize(e)
    cats3[c] = cats3.get(c, 0) + 1

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
fig.suptitle(
    f"Failure modes OF the oracle — proposal-limited "
    f"({len(orac_fail)}/{len(orac)} oracle episodes still fail)",
    fontsize=13,
    fontweight="bold",
)
bar(ax[0], cats3, "(a) Oracle failure-mode counts")

for es, e in orac_fail.items():
    c = categorize(e)
    draw_T_paths(ax[1], e, CCOL[c], lw=0.8, mark=False)
    obj = np.asarray(e["obj_xy"])
    ax[1].plot(*obj[-1], "o", ms=4, color=CCOL[c], alpha=0.7)
ax[1].plot(*goal, "*", ms=22, color="#e9c46a", mec="#7a5901", mew=1.2, zorder=10)
ax[1].set_title("(b) Oracle-failure T-block paths")
ax[1].set_aspect("equal")
ax[1].set_xlabel("x (m)")
ax[1].set_ylabel("y (m)")
ax[1].grid(alpha=0.25, lw=0.5)


# proposal-limited evidence: fraction of replans with >=1 winning branch, fail vs success
def win_frac(e):
    rp = e["replans"]
    return np.mean([r["n_branch_success"] > 0 for r in rp]) if rp else 0.0


wf_fail = [win_frac(e) for e in orac_fail.values()]
wf_succ = [win_frac(e) for es, e in orac.items() if e["success_once"]]
ax[2].hist(
    wf_fail,
    bins=np.linspace(0, 1, 21),
    color="#c1121f",
    alpha=0.7,
    label=f"oracle FAIL (n={len(wf_fail)})",
)
ax[2].hist(
    wf_succ,
    bins=np.linspace(0, 1, 21),
    color="#2a9d8f",
    alpha=0.6,
    label=f"oracle SUCCESS (n={len(wf_succ)})",
)
ax[2].set_title(
    "(c) Fraction of replans with ≥1 winning candidate\n(≈0 for failures ⇒ proposal-limited)"
)
ax[2].set_xlabel("fraction of replans where some candidate reaches goal")
ax[2].set_ylabel("# episodes")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.25, lw=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fail3_oracle_modes.png", dpi=125)
plt.close(fig)

print(f"FIG1 base failures: {cats1}")
print(f"FIG2 recoverable (base-fail, oracle-success): {len(recov)}")
print(f"FIG3 oracle failures: {cats3}")
print("wrote fail1_base_modes.png, fail2_oracle_recovers.png, fail3_oracle_modes.png")

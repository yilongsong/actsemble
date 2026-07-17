"""Before/after visualization of the trailing-hold trim (full vs push-only dataset)."""
import json, sys
from pathlib import Path
import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/data_analysis"
stats = json.loads(Path(f"{OUT}/pushonly_stats.json").read_text())
info = {d["episode_id"]: d for d in stats["per_episode"]}

full = h5py.File("data/push_t_pilot.h5", "r")["episodes"]
eps = sorted(full.keys())
def st(e): return np.asarray(full[e]["state"])
def ac(e): return np.asarray(full[e]["action"])
goal = st(eps[0])[0, 21:23]

C_FULL, C_KEEP, C_CUT, C_REC = "#adb5bd", "#2a9d8f", "#c1121f", "#e76f51"

# ================= FIG A: trim summary =================
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))
fig.suptitle("Trailing-hold trim — full demos → push-only "
             f"(removed {stats['removed_fraction']*100:.0f}% of transitions, "
             f"{stats['recovery_episodes']} recovery episodes preserved)",
             fontsize=13, fontweight="bold")

orig_len = np.array([info[e]["orig_length"] for e in eps])
kept_len = np.array([info[e]["kept_length"] for e in eps])
first_s  = np.array([info[e]["first_success_state"] for e in eps])
recov    = np.array([info[e]["recovery"] for e in eps])

bins = np.arange(0, 102, 4)
ax[0].hist(orig_len, bins=bins, color=C_FULL, alpha=0.85, label=f"full (mean {orig_len.mean():.0f})")
ax[0].hist(kept_len, bins=bins, color=C_KEEP, alpha=0.85, label=f"push-only (mean {kept_len.mean():.0f})")
ax[0].set_title("(a) Episode length distribution"); ax[0].set_xlabel("timesteps")
ax[0].set_ylabel("# episodes"); ax[0].legend()

ax[1].scatter(first_s[~recov], kept_len[~recov], s=10, c=C_KEEP, alpha=0.5, label="normal")
ax[1].scatter(first_s[recov], kept_len[recov], s=22, c=C_REC, alpha=0.9,
              edgecolor="k", lw=0.3, label=f"recovery (n={recov.sum()})")
lim = max(first_s.max(), kept_len.max()) + 3
ax[1].plot([0, lim], [0, lim], "k:", lw=0.8, alpha=0.5, label="y=x")
ax[1].set_title("(b) Kept length vs first true-success step")
ax[1].set_xlabel("first success (90% coverage) step"); ax[1].set_ylabel("kept length")
ax[1].legend(fontsize=8)

nb, na = stats["transitions_before"], stats["transitions_after"]
ax[2].bar(["full", "push-only"], [nb, na], color=[C_FULL, C_KEEP])
for i, v in enumerate([nb, na]):
    ax[2].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
ax[2].set_title(f"(c) Total transitions\nremoved {nb-na:,} ({(nb-na)/nb*100:.1f}%)  ·  kept<18 steps: {stats['kept_below_18']}")
ax[2].set_ylabel("transitions"); ax[2].set_ylim(0, nb*1.12)
for a in ax: a.grid(alpha=0.25, lw=0.5, axis="y")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/trim_summary.png", dpi=125); plt.close(fig)

# ================= FIG B: what got removed =================
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
fig.suptitle("What the trim removes — the low-energy hold tail; the push (and recovery) is kept",
             fontsize=13, fontweight="bold")

# (a) example trajectories: center-distance to goal, mark true-success + cut
# pick a typical episode and a recovery episode
typical = eps[int(np.argmin(np.abs(kept_len - np.median(kept_len))))]
rec_eps = [e for e in eps if info[e]["recovery"]]
examples = [(typical, "typical", "#264653"), (rec_eps[0], "recovery", C_REC)] if rec_eps else [(typical, "typical", "#264653")]
for e, tag, col in examples:
    d = np.linalg.norm(st(e)[:, 24:26] - goal, axis=1)
    tt = np.arange(len(d))
    ax[0].plot(tt, d, color=col, lw=1.4, label=f"{tag} ({e})")
    k = info[e]["kept_length"]
    ax[0].axvline(k, color=col, ls="--", lw=1.2, alpha=0.8)
    fs = info[e]["first_success_state"]
    ax[0].plot(fs, d[fs], "o", color=col, ms=7, mec="k", mew=0.5)
ax[0].set_title("(a) T-block center-distance to goal\n● first true success   ┊ cut (kept end)")
ax[0].set_xlabel("timestep"); ax[0].set_ylabel("‖T−goal‖ (m)"); ax[0].legend(fontsize=8)

# (b) mean |action| per episode: full vs kept-portion
e_full = np.array([np.abs(ac(e)).mean() for e in eps])
e_keep = np.array([np.abs(ac(e)[:info[e]["kept_length"]]).mean() for e in eps])
e_drop = np.array([np.abs(ac(e)[info[e]["kept_length"]:]).mean()
                   if info[e]["kept_length"] < info[e]["orig_length"] else np.nan for e in eps])
bins = np.linspace(0, 0.9, 40)
ax[1].hist(e_full, bins=bins, color=C_FULL, alpha=0.8, label=f"full (μ={e_full.mean():.2f})")
ax[1].hist(e_keep, bins=bins, color=C_KEEP, alpha=0.75, label=f"kept/push (μ={e_keep.mean():.2f})")
ax[1].hist(e_drop[~np.isnan(e_drop)], bins=bins, color=C_CUT, alpha=0.6,
           label=f"removed/hold (μ={np.nanmean(e_drop):.2f})")
ax[1].set_title("(b) Mean |action| per episode\nhold tail is low-energy; kept push is higher-energy")
ax[1].set_xlabel("mean |normalized action|"); ax[1].set_ylabel("# episodes"); ax[1].legend(fontsize=8)

# (c) spatial: T-block path full (faint) vs kept (bold) for a sample
rng = np.random.default_rng(1)
sample = [eps[i] for i in rng.permutation(len(eps))[:35]]
for e in sample:
    xy = st(e)[:, 24:26]; k = info[e]["kept_length"]
    ax[2].plot(xy[:, 0], xy[:, 1], color=C_FULL, lw=0.7, alpha=0.5)
    ax[2].plot(xy[:k+1, 0], xy[:k+1, 1], color=C_KEEP, lw=1.1, alpha=0.8)
    ax[2].plot(*xy[0], "o", ms=3, color="#111", alpha=0.6)
ax[2].plot(*goal, "*", ms=20, color="#e9c46a", mec="#7a5901", mew=1.2, zorder=10, label="goal")
ax[2].plot([], [], color=C_FULL, lw=1.5, label="removed (hold)")
ax[2].plot([], [], color=C_KEEP, lw=1.5, label="kept (push)")
ax[2].set_aspect("equal"); ax[2].set_title("(c) T-block path: kept push vs removed hold")
ax[2].set_xlabel("x (m)"); ax[2].set_ylabel("y (m)"); ax[2].legend(fontsize=8)
for a in ax: a.grid(alpha=0.25, lw=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/trim_removed.png", dpi=125); plt.close(fig)
print("wrote trim_summary.png, trim_removed.png")

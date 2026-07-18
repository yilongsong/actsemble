"""Visualize PushT-v1 RL demonstration trajectories to characterize the data-generating process."""

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

OUT = "/tmp/claude-1001/-home-yilong-actsemble/202b653f-4245-4441-9f46-784ed7a47c11/scratchpad"
f = h5py.File("data/push_t_pilot.h5", "r")
g = f["episodes"]
eps = sorted(g.keys())


def st(e):
    return np.asarray(g[e]["state"])


def ac(e):
    return np.asarray(g[e]["action"])


TCP = slice(14, 16)  # tcp xy
OBJ = slice(24, 26)  # t-block xy
GOAL = slice(21, 23)  # goal xy
goal = st(eps[0])[0, GOAL]
lens = np.array([ac(e).shape[0] for e in eps])


def quat_yaw(q):  # q = [w,x,y,z] planar -> yaw about z
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def colored_path(ax, xy, lw=1.2, alpha=0.8, cmap="viridis"):
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t = np.linspace(0, 1, len(segs))
    lc = LineCollection(segs, cmap=cmap, array=t, lw=lw, alpha=alpha)
    ax.add_collection(lc)
    return lc


rng = np.random.default_rng(0)
sample = [eps[i] for i in rng.permutation(len(eps))[:45]]

# ============ FIGURE 1: spatial strategy ============
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
fig.suptitle(
    "PushT-v1 demonstrations — RL policy, state-projection export (719 successful demos)",
    fontsize=13,
    fontweight="bold",
)

# (a) end-effector paths
for e in sample:
    xy = st(e)[:, TCP]
    colored_path(ax[0], xy, lw=1.0, alpha=0.55)
    ax[0].plot(*xy[0], "o", ms=3, color="#111", zorder=5)
    ax[0].plot(*xy[-1], "x", ms=5, color="#c1121f", zorder=5, mew=1.6)
lc = colored_path(ax[0], st(sample[0])[:, TCP], lw=0.01, alpha=0)
cb = fig.colorbar(lc, ax=ax[0], fraction=0.046, pad=0.02)
cb.set_label("normalized time")
ax[0].set_title(
    f"(a) End-effector (stick) path — {len(sample)} demos\nblack ● start (nearly fixed) · red ✕ end"
)

# (b) T-block center paths + goal
for e in sample:
    xy = st(e)[:, OBJ]
    colored_path(ax[1], xy, lw=1.1, alpha=0.6)
    ax[1].plot(*xy[0], "o", ms=4, color="#111", zorder=5, alpha=0.7)
    ax[1].plot(*xy[-1], "s", ms=4, color="#2a9d8f", zorder=6)
ax[1].plot(
    *goal,
    "*",
    ms=22,
    color="#e9c46a",
    mec="#7a5901",
    mew=1.2,
    zorder=10,
    label="fixed goal",
)
ax[1].add_patch(plt.Circle(goal, 0.03, fill=False, ls="--", ec="#7a5901", lw=1))
ax[1].legend(loc="upper right", fontsize=8)
ax[1].set_title(
    "(b) T-block center path — same demos\nblack ● start (randomized) · teal ▪ end"
)

# (c) initialization distribution (all 719): object init position + yaw
obj0 = np.array([st(e)[0, 24:31] for e in eps])
yaw = quat_yaw(obj0[:, 3:])
q = ax[2].quiver(
    obj0[:, 0],
    obj0[:, 1],
    np.cos(yaw),
    np.sin(yaw),
    yaw,
    cmap="twilight",
    scale=22,
    width=0.005,
    alpha=0.8,
)
ax[2].plot(*goal, "*", ms=22, color="#e9c46a", mec="#7a5901", mew=1.2, zorder=10)
tcp0 = np.array([st(e)[0, 14:16] for e in eps])
ax[2].plot(
    tcp0[:, 0].mean(),
    tcp0[:, 1].mean(),
    "^",
    ms=11,
    color="#264653",
    mec="w",
    label="EE start (fixed)",
)
cb = fig.colorbar(q, ax=ax[2], fraction=0.046, pad=0.02)
cb.set_label("T-block init yaw (rad)")
ax[2].legend(loc="upper right", fontsize=8)
ax[2].set_title(
    "(c) Initial conditions, all 719 demos\narrow = T-block position + orientation"
)

for a in ax:
    a.set_aspect("equal")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.grid(alpha=0.25, lw=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/fig1_spatial.png", dpi=125)
plt.close(fig)

# ============ FIGURE 2: action character over time ============
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
fig.suptitle(
    "Action character over time — high-frequency RL control, not smooth motion",
    fontsize=13,
    fontweight="bold",
)
dim_names = ["Δx", "Δy", "Δz", "rot_x", "rot_y", "rot_z"]
palette = ["#264653", "#2a9d8f", "#e76f51", "#8338ec", "#f4a261", "#457b9d"]

# pick a long and a short demo
long_e = eps[int(np.argmax(lens == 100))]
short_e = eps[int(np.argmin(lens))]
for j, (e, title) in enumerate(
    [
        (long_e, f"(a) example demo (len {ac(long_e).shape[0]})"),
        (short_e, f"(b) fast demo (len {ac(short_e).shape[0]})"),
    ]
):
    a = ac(e)
    for d in range(6):
        ax[j].plot(a[:, d], color=palette[d], lw=1.1, alpha=0.9, label=dim_names[d])
    ax[j].axhline(1, ls=":", c="k", lw=0.6)
    ax[j].axhline(-1, ls=":", c="k", lw=0.6)
    ax[j].set_title(title)
    ax[j].set_xlabel("timestep")
    ax[j].set_ylabel("normalized action")
    ax[j].set_ylim(-1.15, 1.15)
    ax[j].grid(alpha=0.25, lw=0.5)
ax[0].legend(ncol=3, fontsize=8, loc="lower center")

# (c) progress-to-goal curves for many demos
prog_sample = [eps[i] for i in rng.permutation(len(eps))[:120]]
for e in prog_sample:
    d = np.linalg.norm(st(e)[:, OBJ] - goal, axis=1)
    ax[2].plot(np.linspace(0, 1, len(d)), d, color="#457b9d", lw=0.6, alpha=0.18)
# mean progress on common grid
grid = np.linspace(0, 1, 100)
M = np.array(
    [
        np.interp(
            grid,
            np.linspace(0, 1, len(st(e))),
            np.linalg.norm(st(e)[:, OBJ] - goal, axis=1),
        )
        for e in prog_sample
    ]
)
ax[2].plot(grid, M.mean(0), color="#c1121f", lw=2.5, label="mean")
ax[2].axhline(0.03, ls="--", c="#7a5901", lw=1, label="goal tolerance")
ax[2].set_title(
    "(c) T-block distance to goal — 120 demos\nstrategy persistence / monotonicity"
)
ax[2].set_xlabel("normalized time")
ax[2].set_ylabel("‖T-block − goal‖ (m)")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.25, lw=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/fig2_actions.png", dpi=125)
plt.close(fig)

f.close()
print("wrote fig1_spatial.png, fig2_actions.png")
print(f"long demo={long_e} short demo={short_e}")

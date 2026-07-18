"""Episode-disjoint splits and observation/action window extraction.

Window convention (decision point at step t of an episode with T
transitions, valid for every t in [0, T-1]):

* observation history  ``s_{t-H_o+1} .. s_t``  — padded at the episode
  start by replicating ``s_0``; ``obs_mask`` marks real steps.
* action chunk         ``a_t .. a_{t+H_p-1}``  — padded past the episode
  end by replicating ``a_{T-1}``; ``action_mask`` marks real steps.

Edge replication matches the original Diffusion Policy sampler; the
explicit masks let the loss ignore padded targets when configured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import EpisodeRecord
from ..utils.hashing import hash_json


@dataclass
class EpisodeSplit:
    train_episode_ids: list[str]
    val_episode_ids: list[str]

    @property
    def hash(self) -> str:
        return hash_json(
            {
                "train": sorted(self.train_episode_ids),
                "val": sorted(self.val_episode_ids),
            }
        )

    def to_dict(self) -> dict:
        return {
            "train_episode_ids": sorted(self.train_episode_ids),
            "val_episode_ids": sorted(self.val_episode_ids),
        }


def split_episodes(
    episode_ids: list[str], *, val_fraction: float, seed: int
) -> EpisodeSplit:
    """Deterministic episode-disjoint split.

    Entire episodes go to exactly one side; individual transitions are
    never split. With a single episode, it goes to train and validation is
    empty (callers must tolerate an empty validation set for smoke runs).
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    ids = sorted(episode_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("episode ids are not unique")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_val = int(round(len(ids) * val_fraction))
    n_val = min(n_val, len(ids) - 1)  # always keep at least one train episode
    val_ids = [ids[i] for i in perm[:n_val]]
    train_ids = [ids[i] for i in perm[n_val:]]
    assert set(train_ids).isdisjoint(val_ids)
    return EpisodeSplit(
        train_episode_ids=sorted(train_ids), val_episode_ids=sorted(val_ids)
    )


@dataclass
class Window:
    """One training window (all arrays raw / unnormalized)."""

    episode_id: str
    t: int
    obs_history: np.ndarray  # [H_o, state_dim]
    prev_action_history: np.ndarray  # [H_o, action_dim], aligned with obs_history
    obs_mask: np.ndarray  # [H_o] bool, True = real (not padded)
    action_chunk: np.ndarray  # [H_p, action_dim]
    action_mask: np.ndarray  # [H_p] bool, True = real (not padded)


def extract_window(
    episode: EpisodeRecord,
    t: int,
    *,
    obs_horizon: int,
    prediction_horizon: int,
    alignment: str = "future_only",
) -> Window:
    """``alignment`` sets where the action chunk starts relative to the decision
    step t: ``future_only`` (chunk = ``a_t .. a_{t+H_p-1}``, our default) or
    ``diffusion_policy`` (chunk = ``a_{t-H_o+1} .. a_{t-H_o+H_p}``, aligned to the
    observation-window start; the policy then executes from index ``H_o-1``,
    which is the action for time t). Both edge-pad + mask out-of-range indices."""
    T = len(episode)
    if not 0 <= t < T:
        raise IndexError(f"t={t} outside episode of length {T}")

    # Observation history with left edge-padding.
    obs_idx = np.arange(t - obs_horizon + 1, t + 1)
    obs_mask = obs_idx >= 0
    clipped = np.clip(obs_idx, 0, T - 1)
    obs_hist = episode.state[clipped]
    prev_action_hist = episode.previous_action[clipped]

    # Action chunk with edge-padding; start depends on the alignment convention.
    if alignment == "future_only":
        act_start = t
    elif alignment == "diffusion_policy":
        act_start = t - obs_horizon + 1
    else:
        raise ValueError(f"Unknown window alignment: {alignment!r}")
    act_idx = np.arange(act_start, act_start + prediction_horizon)
    action_mask = (act_idx >= 0) & (act_idx <= T - 1)
    action_chunk = episode.action[np.clip(act_idx, 0, T - 1)]

    return Window(
        episode_id=episode.episode_id,
        t=t,
        obs_history=obs_hist.astype(np.float32),
        prev_action_history=prev_action_hist.astype(np.float32),
        obs_mask=obs_mask,
        action_chunk=action_chunk.astype(np.float32),
        action_mask=action_mask,
    )


def enumerate_window_indices(
    episodes: list[EpisodeRecord],
    *,
    alignment: str = "future_only",
    obs_horizon: int = 1,
    prediction_horizon: int = 1,
    action_horizon: int | None = None,
) -> list[tuple[int, int]]:
    """All (episode_index, t) decision points across the given episodes.

    ``future_only`` enumerates every timestep ``t in [0, T-1]`` (the repo's
    original convention). ``diffusion_policy`` instead replicates the reference
    Diffusion-Policy ``SequenceSampler`` range: ``pad_before = H_o - 1`` (so the
    earliest window is ``t = 0``, left-padded) and ``pad_after = H_a - 1`` (so
    the latest window is padded by at most ``H_a - 1`` terminal actions). Without
    this cap, enumerating every timestep over-represents edge-replicated terminal
    actions — trained unmasked, that shifts the target distribution away from the
    reference. The cap requires ``action_horizon``; it is a no-op otherwise."""
    out: list[tuple[int, int]] = []
    for ei, ep in enumerate(episodes):
        T = len(ep)
        if alignment == "diffusion_policy" and action_horizon is not None:
            # right-pad slots at t = (t - H_o + H_p) - (T - 1); cap at H_a - 1
            last_t = min(
                T - 1,
                T
                + int(obs_horizon)
                + int(action_horizon)
                - int(prediction_horizon)
                - 2,
            )
            last_t = max(0, last_t)
        else:
            last_t = T - 1
        out.extend((ei, t) for t in range(0, last_t + 1))
    return out

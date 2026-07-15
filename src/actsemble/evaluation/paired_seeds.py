"""Paired evaluation seeds.

Every system evaluated under the same evaluation config receives the SAME
per-episode (environment seed, perturbation seed, candidate seed) triples,
derived deterministically from one root seed. Selector-level differences
are then the only source of outcome differences (up to simulator
determinism, which physx_cuda num_envs=1 provides in-process).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..seed import derive_seed


@dataclass(frozen=True)
class EpisodeSeeds:
    episode_index: int
    env_seed: int
    perturbation_seed: int
    candidate_seed: int


def paired_seeds(*, root_seed: int, num_episodes: int) -> list[EpisodeSeeds]:
    out = []
    for i in range(num_episodes):
        out.append(
            EpisodeSeeds(
                episode_index=i,
                env_seed=derive_seed(root_seed, "env", i),
                perturbation_seed=derive_seed(root_seed, "perturbation", i),
                candidate_seed=derive_seed(root_seed, "candidates", i),
            )
        )
    return out

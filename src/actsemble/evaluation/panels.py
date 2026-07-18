"""Fixed evaluation panels (checkpoint-selection protocol §1).

A panel is a named, frozen bank of evaluation episodes. Each episode
carries an environment-initialization seed, a policy-sampling seed, and a
perturbation seed, all derived deterministically from the panel's root
seed. The four protocol panels (screening / confirmation / integration /
final_test) must be pairwise disjoint and disjoint from the
demonstration-generation seeds; ``assert_panels_disjoint`` enforces this.

Panel roles:
* screening      — cheap periodic checkpoint evaluation (selection stage 1);
* confirmation   — accurate comparison of screened candidates (stage 2);
* integration    — implementation checking only; never used for selection
                   or claims;
* final_test     — reported results; evaluated only after system freezing;
* dataset_size_development — dataset-size selection (§7); disjoint from
                   the final test.

Diagnostic panels (``DIAGNOSTIC_PANELS``) live outside the frozen protocol
and never produce claims (oracle headroom, pass@k, selector development);
they are still checked disjoint from every protocol bank.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..seed import derive_seed

DEFAULT_PANELS = {
    "screening": {"env_seed": 2000, "num_episodes": 50},
    "confirmation": {"env_seed": 3500, "num_episodes": 200},
    "integration": {"env_seed": 4500, "num_episodes": 10},
    "final_test": {"env_seed": 1000, "num_episodes": 500},
    "dataset_size_development": {"env_seed": 5500, "num_episodes": 200},
}

# Diagnostic panels are NOT part of the frozen protocol and never produce a
# claim. They back privileged / upper-bound diagnostics (oracle headroom,
# pass@k, quick selector development). Kept out of DEFAULT_PANELS so the
# protocol's frozen-spec panel set is unchanged, but registered here so
# make_panel resolves them and assert_panels_disjoint can guard them against
# the protocol banks (and the demonstration seeds).
DIAGNOSTIC_PANELS = {
    "diagnostic": {"env_seed": 20000, "num_episodes": 300},
}

ALL_PANELS = {**DEFAULT_PANELS, **DIAGNOSTIC_PANELS}


@dataclass(frozen=True)
class Panel:
    name: str
    env_seed: int  # root of this panel's seed bank
    num_episodes: int

    def to_dict(self) -> dict:
        return {"name": self.name, "env_seed": self.env_seed, "num_episodes": self.num_episodes}


@dataclass(frozen=True)
class PanelEpisode:
    episode_index: int
    env_seed: int
    policy_sampling_seed: int
    perturbation_seed: int


def make_panel(name: str, spec: dict | None = None) -> Panel:
    spec = spec if spec is not None else ALL_PANELS[name]
    return Panel(name=name, env_seed=int(spec["env_seed"]), num_episodes=int(spec["num_episodes"]))


def load_panels(spec: dict | None = None) -> dict[str, Panel]:
    """Panels from an experiment spec's ``panels:`` mapping (defaults fill gaps)."""
    merged = {**DEFAULT_PANELS, **(spec or {})}
    return {name: make_panel(name, s) for name, s in merged.items()}


def panel_episodes(panel: Panel) -> list[PanelEpisode]:
    """The frozen episode bank of a panel. Same panel -> same bank, always."""
    out = []
    for i in range(panel.num_episodes):
        out.append(
            PanelEpisode(
                episode_index=i,
                env_seed=derive_seed(panel.env_seed, "env", i),
                policy_sampling_seed=derive_seed(panel.env_seed, "policy_sampling", i),
                perturbation_seed=derive_seed(panel.env_seed, "perturbation", i),
            )
        )
    return out


def assert_panels_disjoint(
    panels: dict[str, Panel], *, extra_seed_sets: dict[str, set[int]] | None = None
) -> None:
    """Fail loudly if any two panels (or a panel and e.g. the demonstration
    seeds) share an environment-initialization seed."""
    banks = {name: {e.env_seed for e in panel_episodes(p)} for name, p in panels.items()}
    for name, seeds in (extra_seed_sets or {}).items():
        banks[name] = set(int(s) for s in seeds)
    names = sorted(banks)
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = banks[a] & banks[b]
            if overlap:
                problems.append(f"{a} and {b} share {len(overlap)} env seed(s): "
                                f"{sorted(overlap)[:5]}")
    if problems:
        raise ValueError("Evaluation panels are not disjoint:\n  - " + "\n  - ".join(problems))

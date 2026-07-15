"""Fixed evaluation panels: determinism, structure, disjointness (§1)."""

import pytest

from actsemble.evaluation.panels import (
    DEFAULT_PANELS,
    Panel,
    assert_panels_disjoint,
    load_panels,
    panel_episodes,
)


def test_panel_banks_are_deterministic():
    p = Panel("screening", 2000, 50)
    a = panel_episodes(p)
    b = panel_episodes(p)
    assert a == b
    assert len(a) == 50
    assert len({e.env_seed for e in a}) == 50


def test_episode_seed_streams_are_independent():
    ep = panel_episodes(Panel("x", 123, 5))[0]
    assert len({ep.env_seed, ep.policy_sampling_seed, ep.perturbation_seed}) == 3


def test_default_panels_match_protocol_recipe():
    panels = load_panels()
    assert panels["screening"].env_seed == 2000 and panels["screening"].num_episodes == 50
    assert panels["confirmation"].env_seed == 3500 and panels["confirmation"].num_episodes == 200
    assert panels["integration"].env_seed == 4500 and panels["integration"].num_episodes == 10
    assert panels["final_test"].env_seed == 1000 and panels["final_test"].num_episodes == 500


def test_default_panels_are_pairwise_disjoint():
    assert_panels_disjoint(load_panels())


def test_spec_overrides_merge_with_defaults():
    panels = load_panels({"screening": {"env_seed": 7000, "num_episodes": 5}})
    assert panels["screening"].env_seed == 7000
    assert panels["final_test"].env_seed == DEFAULT_PANELS["final_test"]["env_seed"]


def test_overlap_with_demonstration_seeds_detected():
    panels = load_panels({"screening": {"env_seed": 2000, "num_episodes": 3}})
    demo_seeds = {panel_episodes(panels["screening"])[1].env_seed, 999999}
    with pytest.raises(ValueError, match="not disjoint"):
        assert_panels_disjoint(panels, extra_seed_sets={"demonstrations": demo_seeds})


def test_same_root_would_collide():
    with pytest.raises(ValueError, match="not disjoint"):
        assert_panels_disjoint(
            {"a": Panel("a", 42, 10), "b": Panel("b", 42, 10)}
        )

"""The evaluation horizon must be the one that was ASKED for.

Regression (2026-07-21): `--max-steps 160` on PickCube-v1 silently ran 50-step
episodes, because the horizon lives on ManiSkill's TimeLimitWrapper while
`env.spec.max_episode_steps` reads None -- which looks like "no limit". Every
episode was truncated mid-approach and would have scored 0%, with no error.
These tests use fake wrapper chains so they stay sim-free and fast.
"""

from __future__ import annotations

import pytest

from actsemble.sim.env_factory import effective_horizon


class _Leaf:
    env = None


class _Wrapper:
    def __init__(self, inner, limit=None):
        self.env = inner
        if limit is not None:
            self._max_episode_steps = limit


class _MisleadingSpec:
    """What ManiSkill actually presents: a spec claiming no limit, with the
    real limit buried on an inner wrapper."""

    max_episode_steps = None


def test_finds_limit_on_an_inner_wrapper():
    env = _Wrapper(_Wrapper(_Leaf(), limit=50))
    assert effective_horizon(env) == 50


def test_ignores_a_spec_that_claims_no_limit():
    env = _Wrapper(_Wrapper(_Leaf(), limit=50))
    env.spec = _MisleadingSpec()
    assert env.spec.max_episode_steps is None
    assert effective_horizon(env) == 50, "must not trust spec.max_episode_steps"


def test_returns_none_when_genuinely_unlimited():
    assert effective_horizon(_Wrapper(_Leaf())) is None


def test_takes_the_tightest_limit_when_several_wrappers_set_one():
    env = _Wrapper(_Wrapper(_Leaf(), limit=160), limit=50)
    assert effective_horizon(env) == 50, "the tightest wrapper is what truncates"


def test_terminates_on_a_self_referential_chain():
    node = _Wrapper(_Leaf())
    node.env = node
    assert effective_horizon(node) is None


@pytest.mark.parametrize("limit", [1, 50, 160, 500])
def test_roundtrips_any_limit(limit):
    assert effective_horizon(_Wrapper(_Leaf(), limit=limit)) == limit

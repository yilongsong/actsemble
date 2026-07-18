"""Policy subsystem: training dispatch across supported policy families."""

import pytest

from actsemble.training.factory import policy_trainer, resolve_policy_family
from actsemble.training.train_act_policy import train_act_policy
from actsemble.training.train_diffusion_policy import train_diffusion_policy
from actsemble.training.train_flow_policy import train_flow_policy


@pytest.mark.parametrize(
    ("config", "family", "trainer"),
    [
        ({"type": "diffusion"}, "diffusion", train_diffusion_policy),
        ({"model": {"type": "act"}}, "act", train_act_policy),
        ({"type": "flow"}, "flow", train_flow_policy),
    ],
)
def test_policy_family_dispatch(config, family, trainer):
    assert resolve_policy_family(config) == family
    assert policy_trainer(config) is trainer


def test_unknown_policy_family_rejected():
    with pytest.raises(ValueError, match="Unknown policy family"):
        resolve_policy_family({"type": "mystery"})

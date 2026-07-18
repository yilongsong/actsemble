"""The per-decision context passed to every pipeline stage.

One ``DecisionContext`` is built at each replan and handed to Propose /
Predict / Score / Select / Schedule (docs/system_architecture.md §2.2). It
carries everything a stage may need beyond the candidate tensor: the
observation history, the actions executed so far (past-coherence signals —
Track A5), the replan index, the frozen policy handle (so a Scorer may
re-query it — A5 forward plausibility), the assembled same-data components,
and an optional Monitor signal (within-stage adaptivity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DecisionContext:
    observation_history: np.ndarray          # [obs_horizon, feat], oldest first (raw)
    executed_actions: np.ndarray             # [n_executed, action_dim] committed so far (may be empty)
    replan_index: int
    policy: Any                              # frozen ActionChunkPolicy (a Scorer may re-query it)
    components: list = field(default_factory=list)   # assembled same-data components
    signal: dict | None = None               # Monitor output; None unless a Monitor stage is present

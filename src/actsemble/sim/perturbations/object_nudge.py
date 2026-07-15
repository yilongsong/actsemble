"""Mid-episode object nudge (privileged, recoverable).

At one seeded step inside a configured window, the manipulated object is
displaced by a small planar offset and yaw. Implementation uses privileged
simulator state access; the runtime system observes ONLY the resulting
state — never that a perturbation occurred, when, or how large.

Recoverability policy: magnitudes are capped (defaults: <= 3 cm, <= 15
deg) and the object position is clamped to stay near the workspace
center, so the task remains physically solvable within the episode
budget. Verify per severity that closed-loop success stays well above
zero before trusting a regime (see docs/experiment_contract.md).
"""

from __future__ import annotations

import numpy as np
import torch

from ...seed import derive_seed
from .base import NoOpPerturbation


class ObjectNudgePerturbation(NoOpPerturbation):
    name = "object_nudge"

    def __init__(
        self,
        *,
        actor_name: str = "Tee",
        max_translation: float = 0.03,
        max_yaw_degrees: float = 15.0,
        step_window: tuple[int, int] = (20, 50),
        position_clamp: float = 0.25,
        seed: int = 0,
    ):
        self.actor_name = actor_name
        self.max_translation = float(max_translation)
        self.max_yaw = float(np.deg2rad(max_yaw_degrees))
        self.step_window = (int(step_window[0]), int(step_window[1]))
        self.position_clamp = float(position_clamp)
        self.seed = int(seed)
        self._trigger_step = -1
        self._applied = False
        self._delta: np.ndarray | None = None

    def reset(self, *, episode_seed: int) -> None:
        rng = np.random.default_rng(derive_seed(self.seed, self.name, episode_seed))
        lo, hi = self.step_window
        self._trigger_step = int(rng.integers(lo, hi + 1))
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.5, 1.0) * self.max_translation
        yaw = rng.uniform(-self.max_yaw, self.max_yaw)
        self._delta = np.array([radius * np.cos(angle), radius * np.sin(angle), yaw])
        self._applied = False

    def before_step(self, env: object, step_index: int) -> None:
        if self._applied or step_index != self._trigger_step:
            return
        self._applied = True
        u = env.unwrapped
        state = u.get_state_dict()
        actors = state.get("actors", {})
        if self.actor_name not in actors:
            raise KeyError(
                f"object_nudge: actor {self.actor_name!r} not in env state "
                f"(have {sorted(actors.keys())})"
            )
        pose = actors[self.actor_name].clone()  # [1, 13]: pos(3) quat(wxyz 4) vel(3) angvel(3)
        dx, dy, dyaw = self._delta
        pose[0, 0] = torch.clamp(pose[0, 0] + dx, -self.position_clamp, self.position_clamp)
        pose[0, 1] = torch.clamp(pose[0, 1] + dy, -self.position_clamp, self.position_clamp)
        w, x, y, z = pose[0, 3], pose[0, 4], pose[0, 5], pose[0, 6]
        half = torch.tensor(dyaw / 2.0, dtype=pose.dtype)
        dw, dz = torch.cos(half), torch.sin(half)
        # Quaternion product (dw,0,0,dz) * (w,x,y,z): rotate about world z.
        pose[0, 3] = dw * w - dz * z
        pose[0, 4] = dw * x - dz * y
        pose[0, 5] = dw * y + dz * x
        pose[0, 6] = dw * z + dz * w
        actors[self.actor_name] = pose
        u.set_state_dict(state)

    @property
    def trigger_step(self) -> int:
        """Exposed for tests/diagnostics only — never to the system."""
        return self._trigger_step

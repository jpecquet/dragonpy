"""
Prey: kinematic targets for the dragonfly.

For v1, prey are passive scripted kinematic objects. Each prey owns its
world-frame position and velocity; the environment advances them each fast
tick by whatever update rule is attached.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Prey:
    position: np.ndarray               # (3,) world frame
    velocity: np.ndarray               # (3,) world frame
    radius:   float                    # spherical radius, nondim (for angular size)
    update:   Callable[["Prey", float, float], None] | None = None
    #         ^ update(self, t, dt) — mutates position/velocity in place.
    #           None means stationary or constant-velocity drift only.

    def step(self, t: float, dt: float) -> None:
        if self.update is not None:
            self.update(self, t, dt)
        else:
            self.position = self.position + self.velocity * dt

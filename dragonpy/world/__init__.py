"""
The outside world, from the standpoint of the dragonfly:
atmosphere (wind), gravity, and prey. v1 is a barren environment.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .prey import Prey
from .wind import still_air


@dataclass
class Environment:
    gravity_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0])
    )
    wind: Callable[[np.ndarray, float], np.ndarray] = still_air
    prey: list[Prey] = field(default_factory=list)

    def step_prey(self, t: float, dt: float) -> None:
        for p in self.prey:
            p.step(t, dt)

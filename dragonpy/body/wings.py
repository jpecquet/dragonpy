"""
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

@dataclass
class Wing:

    # geometric attributes
    span_ratio: float # ratio of spanwise wing length to body length
    aero_ratio: float # ratio of air mass displaced to body mass (rho*L*S / m)
    mass_ratio: float # ratio of wing mass to body mass
    drag_coeff: Callable[[float], float] # drag coefficient function of angle of attack
    lift_coeff: Callable[[float], float] # lift coefficient function of angle of attack

    # state variables
    euler_angle: np.ndarray # shape (3,)
    euler_rates: np.ndarray # shape (3,)

    @property
    def aero_center_radius(self) -> float:
        return 2/3 * self.span_ratio

    @property
    def force_coeff(self) -> float:
        return 0.5 * self.aero_ratio / self.span_ratio

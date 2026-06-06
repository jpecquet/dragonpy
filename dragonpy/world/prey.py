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


def cv_white_noise_update(
    qp: float, rng: np.random.Generator,
) -> Callable[["Prey", float, float], None]:
    """Stochastic constant-velocity update with white-noise acceleration.

    Returns a callable suitable for `Prey.update`. The motion is confined
    to the (x, z) plane (y is left untouched). Both axes evolve
    independently under the continuous-time SDE

        dx = v dt,    dv = sqrt(qp) dW,

    discretized exactly: over an interval dt, (Δx, Δv) is bivariate
    Gaussian with std(Δx) = sqrt(qp dt^3 / 3), std(Δv) = sqrt(qp dt),
    and correlation sqrt(3)/2. The intensity `qp` has nondimensional
    units of L_b^2 / T^3.
    """
    rho = np.sqrt(3.0) / 2.0           # corr(Δx, Δv) for the CV process
    sqrt_1_minus_rho2 = 0.5            # = sqrt(1 - 3/4)

    def update(prey: "Prey", t: float, dt: float) -> None:
        sd_x = np.sqrt(qp * dt ** 3 / 3.0)
        sd_v = np.sqrt(qp * dt)
        eta = rng.standard_normal(4)   # [η1_x, η2_x, η1_z, η2_z]

        dpos_x = sd_x * eta[0]
        dvel_x = sd_v * (rho * eta[0] + sqrt_1_minus_rho2 * eta[1])
        dpos_z = sd_x * eta[2]
        dvel_z = sd_v * (rho * eta[2] + sqrt_1_minus_rho2 * eta[3])

        prey.position[0] += prey.velocity[0] * dt + dpos_x
        prey.position[2] += prey.velocity[2] * dt + dpos_z
        prey.velocity[0] += dvel_x
        prey.velocity[2] += dvel_z

    return update

"""
Prey state estimator for the MBE controller.

Extended Kalman filter on a 2D constant-velocity prey. State convention:
    xi = [x_p, z_p, vx_p, vz_p]^T  (world frame)

The prediction step uses the exact discretization of a continuous-time CV
process with white-noise acceleration of (nondimensional) intensity `qp`,
matching the prey model in `dragonpy.world.prey.cv_white_noise_update`.
Updates handle bearing and range measurements separately, so the caller
can fuse either or both.

Bearings are scalar angles in the (x, z) plane,
    y_bearing = atan2(z_p - z_d, x_p - x_d),
where (x_d, z_d) is the observer's world position. Range is the scalar
Euclidean distance from observer to prey in the same plane. Both
measurement models are nonlinear; we linearize around the current mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreyStateEstimate:
    mean:       np.ndarray     # (4,)  [x, z, vx, vz]
    covariance: np.ndarray     # (4, 4)


class PreyStateEKF:
    """Discrete-time EKF for a 2D constant-velocity prey."""

    def __init__(
        self, qp: float, mean0: np.ndarray, cov0: np.ndarray,
    ) -> None:
        self.qp   = float(qp)
        self.mean = np.asarray(mean0, dtype=float).copy()
        self.cov  = np.asarray(cov0,  dtype=float).copy()

    def predict(self, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt

        # Process noise block per axis: rows/cols are (position, velocity).
        a = self.qp * dt ** 3 / 3.0
        b = self.qp * dt ** 2 / 2.0
        c = self.qp * dt

        # Map the two scalar axes into the [x, z, vx, vz] layout.
        Q = np.zeros((4, 4))
        Q[0, 0] = a; Q[0, 2] = b; Q[2, 0] = b; Q[2, 2] = c   # x-axis
        Q[1, 1] = a; Q[1, 3] = b; Q[3, 1] = b; Q[3, 3] = c   # z-axis

        self.mean = F @ self.mean
        self.cov  = F @ self.cov @ F.T + Q

    def update_bearing(
        self,
        bearing_angle: float,
        dragonfly_pos:  np.ndarray,   # (2,)  [x_d, z_d]
        R_bearing:      float,
    ) -> None:
        """Fuse a scalar bearing-angle measurement (radians)."""
        dx = self.mean[0] - dragonfly_pos[0]
        dz = self.mean[1] - dragonfly_pos[1]
        r2 = dx * dx + dz * dz
        if r2 < 1e-12:
            return                                  # singular geometry; skip
        H = np.zeros((1, 4))
        H[0, 0] = -dz / r2
        H[0, 1] =  dx / r2

        y_hat = np.arctan2(dz, dx)
        innov = bearing_angle - y_hat
        innov = (innov + np.pi) % (2.0 * np.pi) - np.pi
        self._apply_scalar_update(H, innov, R_bearing)

    def update_range(
        self,
        range_meas:    float,
        dragonfly_pos: np.ndarray,   # (2,)  [x_d, z_d]
        R_range:       float,
    ) -> None:
        """Fuse a scalar range measurement."""
        dx = self.mean[0] - dragonfly_pos[0]
        dz = self.mean[1] - dragonfly_pos[1]
        r  = float(np.hypot(dx, dz))
        if r < 1e-12:
            return
        H = np.zeros((1, 4))
        H[0, 0] = dx / r
        H[0, 1] = dz / r
        innov = range_meas - r
        self._apply_scalar_update(H, innov, R_range)

    def _apply_scalar_update(
        self, H: np.ndarray, innov: float, R: float,
    ) -> None:
        S = float((H @ self.cov @ H.T)[0, 0]) + R
        K = (self.cov @ H.T).reshape(4) / S
        self.mean = self.mean + K * innov
        self.cov  = (np.eye(4) - np.outer(K, H[0])) @ self.cov

    @property
    def estimate(self) -> PreyStateEstimate:
        return PreyStateEstimate(
            mean=self.mean.copy(), covariance=self.cov.copy(),
        )

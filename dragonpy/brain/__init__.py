"""
Brain layer: the logic that turns sensor readings into muscle commands.

Contract (convention, not enforced):
  reads  - `dragonfly.sensors.*.reading` and `dragonfly.wings` (immutable spec).
  writes - `dragonfly.stroke_patterns` and `dragonfly.wing_frequency`.
  never  - reads body kinematic state, wing_states, or wing_phases directly.

A brain is stateful (one instance per dragonfly) and is called once per
slow tick with the elapsed slow-tick dt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..body.muscles import StrokePattern

if TYPE_CHECKING:
    from ..dragonfly import Dragonfly


class Brain(ABC):
    @abstractmethod
    def update(self, dragonfly: "Dragonfly", dt: float) -> None: ...


class NullBrain(Brain):
    """No-op brain. Stroke patterns and frequency remain as initialized."""

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        pass


class HoverBrain(Brain):
    """Minimal rate controller for upright hover.

    Two feedback loops, both driven by the inertial sensor's body-frame
    velocity reading (which equals world-frame velocity while the body
    attitude is identity):

      d(sweep_amp)/dt    = -k_z * v_body_z
      d(feather_mean)/dt = +k_x * v_body_x

    Elevation is held at zero. All four wings receive identical patterns,
    so this brain is only meaningful when the four wings are geometrically
    symmetric (e.g. hinged at the same point with mirrored chiralities).
    """

    def __init__(
        self,
        sweep_amp_init: float,
        feather_amp: float,
        feather_phase: float,
        wing_frequency: float,
        k_z: float,
        k_x: float,
    ) -> None:
        self.sweep_amp    = sweep_amp_init
        self.feather_amp  = feather_amp
        self.feather_phase = feather_phase
        self.wing_frequency = wing_frequency
        self.feather_mean = 0.0
        self.k_z = k_z
        self.k_x = k_x
        self._installed = False

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        if not self._installed:
            dragonfly.wing_frequency = self.wing_frequency
            for p in dragonfly.stroke_patterns:
                p.stroke_plane_tilt = 0.0
                p.sweep_amp = self.sweep_amp
                p.sweep_mean = 0.0
                p.sweep_phase = 0.0
                p.elev_amp = 0.0
                p.elev_mean = 0.0
                p.elev_phase = 0.0
                p.feather_amp = self.feather_amp
                p.feather_mean = self.feather_mean
                p.feather_phase = self.feather_phase
            self._installed = True

        vel = dragonfly.sensors.inertial.reading.velocity
        self.sweep_amp    += -self.k_z * vel[2] * dt
        self.sweep_amp     = max(self.sweep_amp, 0.0)
        self.feather_mean += self.k_x * vel[0] * dt

        for p in dragonfly.stroke_patterns:
            p.sweep_amp = self.sweep_amp
            p.feather_mean = self.feather_mean


class InterceptBrain(Brain):
    """Hover until prey is visible, then pursue.

    Hover mode: identical to HoverBrain (sweep_amp and feather_mean rate
    controllers for altitude and station-keeping).

    Intercept mode (prey in compound eye FOV): high sweep amplitude, zero
    feather mean, stroke plane tilt rate-controlled to steer the velocity
    vector toward the prey bearing:

      d(tilt)/dt = k_tilt * (prey_elevation - vel_elevation)

    where elevation = atan2(z, x) in the xz body plane. Because this brain
    targets the point-mass model, stroke plane tilt doubles as effective body
    pitch.
    """

    def __init__(
        self,
        # hover parameters
        hover_sweep_amp: float,
        feather_amp: float,
        feather_phase: float,
        wing_frequency: float,
        k_z: float,
        k_x: float,
        # intercept parameters
        intercept_sweep_amp: float,
        intercept_feather_amp: float,
        k_tilt: float,
    ) -> None:
        self.hover_sweep_amp = hover_sweep_amp
        self.feather_amp     = feather_amp
        self.feather_phase   = feather_phase
        self.wing_frequency  = wing_frequency
        self.k_z = k_z
        self.k_x = k_x

        self.intercept_sweep_amp    = intercept_sweep_amp
        self.intercept_feather_amp  = intercept_feather_amp
        self.k_tilt = k_tilt

        # live state
        self.sweep_amp     = hover_sweep_amp
        self.feather_mean  = 0.0
        self.stroke_plane_tilt = 0.0
        self.mode: str     = "hover"
        self._installed    = False

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        if not self._installed:
            dragonfly.wing_frequency = self.wing_frequency
            for p in dragonfly.stroke_patterns:
                p.stroke_plane_tilt = 0.0
                p.sweep_amp = self.sweep_amp
                p.sweep_mean = 0.0
                p.sweep_phase = 0.0
                p.elev_amp = 0.0
                p.elev_mean = 0.0
                p.elev_phase = 0.0
                p.feather_amp = self.feather_amp
                p.feather_mean = 0.0
                p.feather_phase = self.feather_phase
            self._installed = True

        detections = dragonfly.sensors.eye.reading
        vel = dragonfly.sensors.inertial.reading.velocity

        if not detections:
            self._hover_update(vel, dragonfly, dt)
        else:
            self._intercept_update(detections[0], vel, dragonfly, dt)

    def _hover_update(
        self, vel: np.ndarray, dragonfly: "Dragonfly", dt: float,
    ) -> None:
        self.mode = "hover"
        self.sweep_amp += -self.k_z * vel[2] * dt
        self.sweep_amp  = max(self.sweep_amp, 0.0)
        self.feather_mean += self.k_x * vel[0] * dt
        self.stroke_plane_tilt = 0.0

        for p in dragonfly.stroke_patterns:
            p.sweep_amp = self.sweep_amp
            p.feather_mean = self.feather_mean
            p.feather_amp = self.feather_amp
            p.stroke_plane_tilt = 0.0

    def _intercept_update(
        self, prey, vel: np.ndarray, dragonfly: "Dragonfly", dt: float,
    ) -> None:
        self.mode = "intercept"
        bearing = prey.bearing

        speed = float(np.linalg.norm(vel[[0, 2]]))
        if speed > 1e-6:
            vel_elev  = np.arctan2(vel[2], vel[0])
        else:
            vel_elev  = 0.0
        prey_elev = np.arctan2(bearing[2], bearing[0])

        self.stroke_plane_tilt += self.k_tilt * (prey_elev - vel_elev) * dt

        self.sweep_amp    = self.intercept_sweep_amp
        self.feather_mean = 0.0

        for p in dragonfly.stroke_patterns:
            p.sweep_amp = self.sweep_amp
            p.feather_mean = 0.0
            p.feather_amp = self.intercept_feather_amp
            p.stroke_plane_tilt = self.stroke_plane_tilt


class StaticBrain(Brain):
    """Installs a fixed set of patterns once, then no-ops.

    Useful for open-loop hover / forward flight runs, and as a base for
    simple closed-loop controllers that override `update`.
    """

    def __init__(
        self,
        patterns: list[StrokePattern],
        wing_frequency: float,
    ) -> None:
        self._patterns = patterns
        self._frequency = wing_frequency
        self._installed = False

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        if self._installed:
            return
        for i, pat in enumerate(self._patterns):
            dragonfly.stroke_patterns[i] = pat
        dragonfly.wing_frequency = self._frequency
        self._installed = True

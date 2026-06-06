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
from .estimator import PreyStateEKF

if TYPE_CHECKING:
    from ..dragonfly import Dragonfly


class Brain(ABC):
    @abstractmethod
    def update(self, dragonfly: "Dragonfly", dt: float) -> None: ...


class NullBrain(Brain):
    """No-op brain. Stroke patterns and frequency remain as initialized."""

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        pass


def _apply_snapshot(dragonfly: "Dragonfly", snapshot: dict) -> None:
    """Write a snapshot dict's stroke-pattern fields onto every wing."""
    for p in dragonfly.stroke_patterns:
        for key, val in snapshot.items():
            setattr(p, key, val)


def _commit_through_queue(
    queue: list, snapshot: dict, n_delay: int, dragonfly: "Dragonfly",
) -> None:
    """Push the latest brain-commanded snapshot through a fixed-length
    delay queue and apply the oldest snapshot to the wings. Equivalent to
    an `n_delay`-tick FIFO on the muscle chain."""
    queue.append(snapshot)
    if len(queue) > n_delay:
        applied = queue.pop(0)
        _apply_snapshot(dragonfly, applied)


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
        stroke_plane_tilt: float = 0.0,
    ) -> None:
        self.sweep_amp    = sweep_amp_init
        self.feather_amp  = feather_amp
        self.feather_phase = feather_phase
        self.wing_frequency = wing_frequency
        self.feather_mean = 0.0
        self.k_z = k_z
        self.k_x = k_x
        self.stroke_plane_tilt = stroke_plane_tilt
        self._installed = False

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        if not self._installed:
            dragonfly.wing_frequency = self.wing_frequency
            for p in dragonfly.stroke_patterns:
                p.stroke_plane_tilt = self.stroke_plane_tilt
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
        hover_stroke_plane_tilt: float = 0.0,
        # intercept parameters
        intercept_sweep_amp: float = 0.0,
        intercept_feather_amp: float = 0.0,
        intercept_feather_phase: float = 0.0,
        k_tilt: float = 0.0,
        # muscle chain delay
        muscle_delay: float = 0.0,
    ) -> None:
        self.hover_sweep_amp = hover_sweep_amp
        self.feather_amp     = feather_amp
        self.feather_phase   = feather_phase
        self.wing_frequency  = wing_frequency
        self.k_z = k_z
        self.k_x = k_x
        self.hover_stroke_plane_tilt = hover_stroke_plane_tilt

        self.intercept_sweep_amp    = intercept_sweep_amp
        self.intercept_feather_amp  = intercept_feather_amp
        self.intercept_feather_phase = intercept_feather_phase
        self.k_tilt = k_tilt

        self.muscle_delay  = float(muscle_delay)
        self._cmd_queue: list[dict] = []
        self._n_delay: int = 0

        # live state
        self.sweep_amp     = hover_sweep_amp
        self.feather_mean  = 0.0
        self.stroke_plane_tilt = hover_stroke_plane_tilt
        self.mode: str     = "hover"
        self._installed    = False

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        if not self._installed:
            self._n_delay = max(0, round(self.muscle_delay / dt))
            dragonfly.wing_frequency = self.wing_frequency
            initial = self._snapshot()
            _apply_snapshot(dragonfly, initial)
            self._cmd_queue = [dict(initial) for _ in range(self._n_delay)]
            self._installed = True

        detections = dragonfly.sensors.eye.reading
        vel = dragonfly.sensors.inertial.reading.velocity

        if not detections:
            self._hover_update(vel, dt)
        else:
            self._intercept_update(detections[0], vel, dt)

        _commit_through_queue(
            self._cmd_queue, self._snapshot(), self._n_delay, dragonfly,
        )

    def _snapshot(self) -> dict:
        if self.mode == "hover":
            return {
                "stroke_plane_tilt": self.hover_stroke_plane_tilt,
                "sweep_amp":     self.sweep_amp,
                "sweep_mean":    0.0,
                "sweep_phase":   0.0,
                "elev_amp":      0.0,
                "elev_mean":     0.0,
                "elev_phase":    0.0,
                "feather_amp":   self.feather_amp,
                "feather_mean":  self.feather_mean,
                "feather_phase": self.feather_phase,
            }
        return {
            "stroke_plane_tilt": self.stroke_plane_tilt,
            "sweep_amp":     self.sweep_amp,
            "sweep_mean":    0.0,
            "sweep_phase":   0.0,
            "elev_amp":      0.0,
            "elev_mean":     0.0,
            "elev_phase":    0.0,
            "feather_amp":   self.intercept_feather_amp,
            "feather_mean":  0.0,
            "feather_phase": self.intercept_feather_phase,
        }

    def _hover_update(self, vel: np.ndarray, dt: float) -> None:
        self.mode = "hover"
        self.sweep_amp += -self.k_z * vel[2] * dt
        self.sweep_amp  = max(self.sweep_amp, 0.0)
        self.feather_mean += self.k_x * vel[0] * dt
        self.stroke_plane_tilt = self.hover_stroke_plane_tilt

    def _intercept_update(self, prey, vel: np.ndarray, dt: float) -> None:
        self.mode = "intercept"
        bearing = prey.bearing

        speed = float(np.linalg.norm(vel[[0, 2]]))
        if speed > 1e-6:
            vel_elev  = np.arctan2(vel[2], vel[0])
        else:
            vel_elev  = 0.0
        prey_elev = np.arctan2(bearing[2], bearing[0])

        # Wrap the bearing-error to (-pi, pi] so the controller always picks
        # the shortest angular path between velocity and prey bearing. Without
        # this, once velocity points backward (e.g. after over-rotating from
        # a step target jump), the raw subtraction of two arctan2's exceeds
        # ±pi and the controller drives gamma the wrong way around.
        elev_err = prey_elev - vel_elev
        elev_err = (elev_err + np.pi) % (2 * np.pi) - np.pi
        self.stroke_plane_tilt -= self.k_tilt * elev_err * dt
        # Clamp to anatomically plausible range. Without this the integrator
        # winds up indefinitely on step disturbances.
        self.stroke_plane_tilt = float(np.clip(
            self.stroke_plane_tilt, -np.pi / 2, np.pi / 2,
        ))

        self.sweep_amp    = self.intercept_sweep_amp
        self.feather_mean = 0.0


def _solve_intercept_time(
    r: np.ndarray, v_p: np.ndarray, s: float,
) -> float | None:
    """Smallest positive t with |r + v_p t| = s t, or None if unsolvable.

    Collision-course solve for an own-vehicle with speed `s` chasing a
    target whose relative position is `r` and velocity `v_p`. Caller
    chooses 2D or 3D by passing matching shapes.
    """
    A = s * s - float(v_p @ v_p)
    B = -2.0 * float(r @ v_p)
    C = -float(r @ r)
    if abs(A) < 1e-12:
        # Speed matches target speed: degenerate to linear in t.
        if abs(B) < 1e-12:
            return None
        t = -C / B
        return t if t > 0.0 else None
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    sd = np.sqrt(disc)
    candidates = [(-B - sd) / (2.0 * A), (-B + sd) / (2.0 * A)]
    positive = [t for t in candidates if t > 0.0]
    if not positive:
        return None
    return min(positive)


class MBEBrain(Brain):
    """Model-based estimation brain.

    Hover until prey is visible, then run an EKF on the prey state and
    steer onto the constant-bearing collision course implied by the
    estimate. The pursuit law mirrors the proportional controller:

      d(stroke_plane_tilt)/dt = -k_tilt * theta_error,

    but `theta_error` is taken between the velocity heading and the
    *lead bearing* (the angle to where the prey is predicted to be at
    the intercept time t_go), rather than between the velocity heading
    and the current prey bearing.

    Position bookkeeping: the brain dead-reckons its own world position
    from the inertial sensor's body-frame velocity, starting at
    `dragonfly_init_pos`. With point_mass=True the body frame coincides
    with the world frame, so this is exact in expectation.
    """

    def __init__(
        self,
        # hover parameters (match HoverBrain / InterceptBrain)
        hover_sweep_amp:           float,
        feather_amp:               float,
        feather_phase:             float,
        wing_frequency:            float,
        k_z:                       float,
        k_x:                       float,
        hover_stroke_plane_tilt:   float = 0.0,
        # intercept parameters
        intercept_sweep_amp:       float = 0.0,
        intercept_feather_amp:     float = 0.0,
        intercept_feather_phase:   float = 0.0,
        k_tilt:                    float = 0.0,
        # MBE parameters
        prey_state_prior_mean:     np.ndarray | None = None,    # (4,)
        prey_state_prior_cov:      np.ndarray | None = None,    # (4, 4)
        qp_assumed:                float = 0.0,
        R_bearing:                 float = 1e-4,
        use_range:                 bool  = False,
        R_range:                   float = 1e-2,
        prey_radius:               float = 0.0,
        dragonfly_init_pos:        np.ndarray | None = None,    # (3,)
        muscle_delay:              float = 0.0,
    ) -> None:
        self.hover_sweep_amp = hover_sweep_amp
        self.feather_amp     = feather_amp
        self.feather_phase   = feather_phase
        self.wing_frequency  = wing_frequency
        self.k_z = k_z
        self.k_x = k_x
        self.hover_stroke_plane_tilt = hover_stroke_plane_tilt

        self.intercept_sweep_amp     = intercept_sweep_amp
        self.intercept_feather_amp   = intercept_feather_amp
        self.intercept_feather_phase = intercept_feather_phase
        self.k_tilt = k_tilt

        # MBE state
        if prey_state_prior_mean is None or prey_state_prior_cov is None:
            raise ValueError(
                "MBEBrain requires prey_state_prior_mean and prey_state_prior_cov"
            )
        self.filter = PreyStateEKF(
            qp=qp_assumed,
            mean0=np.asarray(prey_state_prior_mean, dtype=float),
            cov0=np.asarray(prey_state_prior_cov,   dtype=float),
        )
        self.R_bearing  = float(R_bearing)
        self.use_range  = bool(use_range)
        self.R_range    = float(R_range)
        self.prey_radius = float(prey_radius)

        self.position_estimate = (
            np.zeros(3) if dragonfly_init_pos is None
            else np.asarray(dragonfly_init_pos, dtype=float).copy()
        )

        self.muscle_delay   = float(muscle_delay)
        self._cmd_queue: list[dict] = []
        self._n_delay: int  = 0

        # live state
        self.sweep_amp     = hover_sweep_amp
        self.feather_mean  = 0.0
        self.stroke_plane_tilt = hover_stroke_plane_tilt
        self.mode: str     = "hover"
        self._installed    = False
        self._lead_point   = np.zeros(2)   # (x, z) — for debugging/plotting

    def update(self, dragonfly: "Dragonfly", dt: float) -> None:
        if not self._installed:
            self._n_delay = max(0, round(self.muscle_delay / dt))
            dragonfly.wing_frequency = self.wing_frequency
            initial = self._snapshot()
            _apply_snapshot(dragonfly, initial)
            self._cmd_queue = [dict(initial) for _ in range(self._n_delay)]
            self._installed = True

        vel = dragonfly.sensors.inertial.reading.velocity

        # Dead reckoning: integrate own velocity into the position estimate.
        # With point_mass=True the inertial velocity is already in the world
        # frame, so this is just an Euler step.
        self.position_estimate = self.position_estimate + vel * dt

        # Prediction always runs, even when prey is not currently visible.
        self.filter.predict(dt)

        detections = dragonfly.sensors.eye.reading
        if detections:
            self._fuse_detection(detections[0])
            self._intercept_update(vel, dt)
        else:
            self._hover_update(vel, dt)

        _commit_through_queue(
            self._cmd_queue, self._snapshot(), self._n_delay, dragonfly,
        )

    def _snapshot(self) -> dict:
        if self.mode == "hover":
            return {
                "stroke_plane_tilt": self.hover_stroke_plane_tilt,
                "sweep_amp":     self.sweep_amp,
                "sweep_mean":    0.0,
                "sweep_phase":   0.0,
                "elev_amp":      0.0,
                "elev_mean":     0.0,
                "elev_phase":    0.0,
                "feather_amp":   self.feather_amp,
                "feather_mean":  self.feather_mean,
                "feather_phase": self.feather_phase,
            }
        return {
            "stroke_plane_tilt": self.stroke_plane_tilt,
            "sweep_amp":     self.sweep_amp,
            "sweep_mean":    0.0,
            "sweep_phase":   0.0,
            "elev_amp":      0.0,
            "elev_mean":     0.0,
            "elev_phase":    0.0,
            "feather_amp":   self.intercept_feather_amp,
            "feather_mean":  0.0,
            "feather_phase": self.intercept_feather_phase,
        }

    def _fuse_detection(self, prey) -> None:
        bearing = prey.bearing
        bearing_angle = float(np.arctan2(bearing[2], bearing[0]))
        own_xz = self.position_estimate[[0, 2]]
        self.filter.update_bearing(bearing_angle, own_xz, self.R_bearing)
        if self.use_range and self.prey_radius > 0.0:
            range_meas = self.prey_radius / np.tan(0.5 * prey.angular_size)
            self.filter.update_range(range_meas, own_xz, self.R_range)

    def _hover_update(self, vel: np.ndarray, dt: float) -> None:
        self.mode = "hover"
        self.sweep_amp += -self.k_z * vel[2] * dt
        self.sweep_amp  = max(self.sweep_amp, 0.0)
        self.feather_mean += self.k_x * vel[0] * dt
        self.stroke_plane_tilt = self.hover_stroke_plane_tilt

    def _intercept_update(self, vel: np.ndarray, dt: float) -> None:
        self.mode = "intercept"

        own_xz    = self.position_estimate[[0, 2]]
        prey_xz   = self.filter.mean[[0, 1]]
        prey_v_xz = self.filter.mean[[2, 3]]
        own_v_xz  = vel[[0, 2]]
        speed = float(np.linalg.norm(own_v_xz))

        # Lead point: predicted prey position at the collision-course
        # intercept time. Falls back to pure pursuit (aim at the current
        # estimated prey position) when no positive real solution exists
        # — e.g. while still hovering, or if speed < prey speed and prey
        # is fleeing radially.
        r = prey_xz - own_xz
        t_go = _solve_intercept_time(r, prey_v_xz, speed) if speed > 1e-6 else None
        lead_xz = prey_xz if t_go is None else prey_xz + prey_v_xz * t_go
        self._lead_point = lead_xz

        vel_elev  = np.arctan2(own_v_xz[1], own_v_xz[0]) if speed > 1e-6 else 0.0
        lead_elev = np.arctan2(lead_xz[1] - own_xz[1], lead_xz[0] - own_xz[0])
        elev_err  = lead_elev - vel_elev
        elev_err  = (elev_err + np.pi) % (2.0 * np.pi) - np.pi

        self.stroke_plane_tilt -= self.k_tilt * elev_err * dt
        self.stroke_plane_tilt = float(np.clip(
            self.stroke_plane_tilt, -np.pi / 2, np.pi / 2,
        ))

        self.sweep_amp    = self.intercept_sweep_amp
        self.feather_mean = 0.0


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

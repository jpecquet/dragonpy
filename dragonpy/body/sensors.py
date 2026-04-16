"""
Sensors: the firewall between ground truth and what the dragonfly knows.

Each sensor is a stateful module with `sample(sim)` that reads the
simulation and writes a typed `reading` on itself. The brain reads only
sensor readings, never sim state directly.

v1 sensor set:
  * InertialSensor — ideal proprioception (perfect knowledge of own state).
  * CompoundEye    — scans prey within FOV and range, returns TSDN-style
                     detections (bearing, angular size, angular velocity).
  * Ocelli, AirflowSensor, WingLoadSensor — stubs. Reading types defined,
                     `sample` is a no-op for v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..numerics.rotations import quat_to_matrix

if TYPE_CHECKING:
    from ..dynamics import Simulation


# ---------------------------------------------------------------------------
# Reading types


@dataclass
class InertialReading:
    velocity:         np.ndarray   # (3,) body frame
    angular_velocity: np.ndarray   # (3,) body frame
    gravity_body:     np.ndarray   # (3,) unit vector, gravity direction in body frame


@dataclass
class PreyDetection:
    bearing:          np.ndarray   # (3,) unit vector, body frame
    angular_size:     float        # radians; brain may invert this to estimate range
    angular_velocity: np.ndarray   # (3,) body frame, rate of bearing vector


@dataclass
class OcelliReading:
    gravity_body: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class AirflowReading:
    relative_wind_body: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class WingLoadReading:
    force_magnitude: np.ndarray = field(default_factory=lambda: np.zeros(4))


# ---------------------------------------------------------------------------
# Base class


class Sensor:
    """Base class. Subclasses implement `sample` and set `self.reading`."""

    reading: object

    def sample(self, sim: "Simulation") -> None:
        raise NotImplementedError

    def add_noise(self, reading):
        # v1 hook: no-op. Subclasses with noise models override.
        return reading


# ---------------------------------------------------------------------------
# Concrete sensors


class InertialSensor(Sensor):
    """Ideal proprioception. Gives the brain perfect knowledge of own state."""

    def __init__(self) -> None:
        self.reading = InertialReading(
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            gravity_body=np.array([0.0, 0.0, -1.0]),
        )

    def sample(self, sim: "Simulation") -> None:
        dfly = sim.dragonfly
        R_world_body = quat_to_matrix(dfly.attitude)
        R_body_world = R_world_body.T
        g_body = R_body_world @ sim.environment.gravity_direction
        self.reading = InertialReading(
            velocity=dfly.velocity.copy(),
            angular_velocity=dfly.angular_velocity.copy(),
            gravity_body=g_body,
        )


class CompoundEye(Sensor):
    """Prey-detecting eye with a forward-facing conical field of view.

    Returns one `PreyDetection` per prey that is (a) within `max_range` and
    (b) within `fov_half_angle` of the body +x axis.
    """

    def __init__(self, fov_half_angle: float, max_range: float) -> None:
        self.fov_half_angle = float(fov_half_angle)
        self.max_range = float(max_range)
        self.reading: list[PreyDetection] = []

    def sample(self, sim: "Simulation") -> None:
        dfly = sim.dragonfly
        R_world_body = quat_to_matrix(dfly.attitude)
        R_body_world = R_world_body.T
        v_self_world = R_world_body @ dfly.velocity
        omega_body = dfly.angular_velocity
        cos_fov = np.cos(self.fov_half_angle)

        detections: list[PreyDetection] = []
        for prey in sim.environment.prey:
            r_world = prey.position - dfly.position
            r_body = R_body_world @ r_world
            dist = float(np.linalg.norm(r_body))
            if dist > self.max_range or dist < 1e-12:
                continue
            bearing = r_body / dist
            if bearing[0] < cos_fov:      # forward-axis cone test
                continue

            v_rel_body = R_body_world @ (prey.velocity - v_self_world)
            dr_body = v_rel_body - np.cross(omega_body, r_body)
            # Angular velocity of the bearing unit vector in body frame.
            radial = bearing * np.dot(bearing, dr_body)
            ang_vel = (dr_body - radial) / dist

            detections.append(PreyDetection(
                bearing=bearing,
                angular_size=2.0 * np.arctan(prey.radius / dist),
                angular_velocity=ang_vel,
            ))
        self.reading = detections


class Ocelli(Sensor):
    """Stub. Reserved for a coarse horizon / attitude cue."""

    def __init__(self) -> None:
        self.reading = OcelliReading()

    def sample(self, sim: "Simulation") -> None:
        pass


class AirflowSensor(Sensor):
    """Stub. Reserved for antennal airspeed sensing."""

    def __init__(self) -> None:
        self.reading = AirflowReading()

    def sample(self, sim: "Simulation") -> None:
        pass


class WingLoadSensor(Sensor):
    """Stub. Reserved for per-wing strain / force proprioception."""

    def __init__(self) -> None:
        self.reading = WingLoadReading()

    def sample(self, sim: "Simulation") -> None:
        pass


# ---------------------------------------------------------------------------
# Container


@dataclass
class Sensors:
    inertial:  InertialSensor
    eye:       CompoundEye
    ocelli:    Ocelli
    airflow:   AirflowSensor
    wing_load: WingLoadSensor

    def sample_all(self, sim: "Simulation") -> None:
        self.inertial.sample(sim)
        self.eye.sample(sim)
        self.ocelli.sample(sim)
        self.airflow.sample(sim)
        self.wing_load.sample(sim)

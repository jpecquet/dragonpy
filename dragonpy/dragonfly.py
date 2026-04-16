"""
The Dragonfly: body plan + state + mind, assembled into one object.

Holds:
  - immutable body spec: wings, inertia.
  - brain and sensors.
  - muscle-layer state: stroke_patterns, wing_frequency, wing_phases,
                        wing_states (the last is cached observability).
  - rigid-body state: position (world), attitude (quat, body->world),
                      velocity (body), angular_velocity (body).
"""

from dataclasses import dataclass, field

import numpy as np

from .body.muscles import StrokePattern
from .body.sensors import Sensors
from .body.wings import Wing, WingState
from .brain import Brain
from .numerics.rotations import quat_identity


def _identity_wing_state() -> WingState:
    return WingState(
        R_hinge_from_wing=np.eye(3),
        omega_wing_in_hinge=np.zeros(3),
    )


@dataclass
class Dragonfly:
    wings:           list[Wing]
    sensors:         Sensors
    brain:           Brain
    stroke_patterns: list[StrokePattern]
    inertia_body:    np.ndarray              # (3,) diagonal inertia in body frame

    wing_frequency:  float = 0.0

    position:         np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude:         np.ndarray = field(default_factory=quat_identity)
    velocity:         np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    wing_phases: np.ndarray = field(default_factory=lambda: np.zeros(4))
    wing_states: list[WingState] | None = None

    def __post_init__(self) -> None:
        n = len(self.wings)
        if self.wing_states is None:
            self.wing_states = [_identity_wing_state() for _ in range(n)]
        assert len(self.stroke_patterns) == n
        assert len(self.wing_phases) == n
        assert len(self.wing_states) == n

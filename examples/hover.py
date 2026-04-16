"""
Hover test: four wings hinged at the COM, zero elevation, symmetric
patterns. Body attitude should stay at identity throughout; the only
degrees of freedom that should move are (x, z) in world frame.

The HoverBrain runs two rate controllers:
  sweep_amp    rate <- -k_z * vz_body
  feather_mean rate <- +k_x * vx_body
"""

import numpy as np

from dragonpy.body.muscles import StrokePattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing
from dragonpy.brain import HoverBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation, step_slow
from dragonpy.world import Environment


def flat_plate_cl(alpha):
    return np.sin(2.0 * alpha)


def flat_plate_cd(alpha):
    return 1.0 - np.cos(2.0 * alpha)


R_HINGE_RIGHT = np.array([[0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
R_HINGE_LEFT = np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])


def make_wing(chirality: int) -> Wing:
    R = R_HINGE_RIGHT if chirality == +1 else R_HINGE_LEFT
    return Wing(
        hinge_position=np.zeros(3),            # all hinged at COM
        hinge_orientation=R,
        chirality=chirality,
        span_ratio=0.8,
        mass_ratio=0.05,
        aero_ratio=0.15,
        lift_coeff=flat_plate_cl,
        drag_coeff=flat_plate_cd,
        n_elements=8,
    )


wings = [make_wing(+1), make_wing(-1), make_wing(+1), make_wing(-1)]

placeholder_pattern = StrokePattern(
    stroke_plane_tilt=0.0,
    sweep_amp=0.0, sweep_mean=0.0, sweep_phase=0.0,
    elev_amp=0.0,  elev_mean=0.0,  elev_phase=0.0,  elev_harmonic=2,
    feather_amp=0.0, feather_mean=0.0, feather_phase=0.0,
)

sensors = Sensors(
    inertial=InertialSensor(),
    eye=CompoundEye(fov_half_angle=np.pi / 2, max_range=20.0),
    ocelli=Ocelli(),
    airflow=AirflowSensor(),
    wing_load=WingLoadSensor(),
)

brain = HoverBrain(
    sweep_amp_init=0.45,
    feather_amp=0.7,
    feather_phase=np.pi / 2,
    wing_frequency=2.1,
    k_z=2.0,
    k_x=1.5,
)

dragonfly = Dragonfly(
    wings=wings,
    sensors=sensors,
    brain=brain,
    stroke_patterns=[StrokePattern(**placeholder_pattern.__dict__) for _ in range(4)],
    inertia_body=np.array([0.01, 0.05, 0.05]),
    position=np.array([0.0, 0.0, 5.0]),
    point_mass=True,
)

sim = Simulation(
    dragonfly=dragonfly,
    environment=Environment(),
    dt_fast=1.0 / (2.1 * 100),
    fast_per_slow=15,
)

T_END = 30.0
print(f"{'t':>7} {'x':>8} {'z':>8} {'vx':>9} {'vz':>9} "
      f"{'amp':>7} {'fmean':>8} {'pitch':>8}")
while sim.t < T_END:
    step_slow(sim)
    # Pitch from attitude quat (w, x, y, z): 2*asin(y) is small-angle pitch.
    q = dragonfly.attitude
    pitch = 2.0 * np.arcsin(np.clip(q[2], -1.0, 1.0))
    print(
        f"{sim.t:7.2f} "
        f"{dragonfly.position[0]:8.3f} "
        f"{dragonfly.position[2]:8.3f} "
        f"{dragonfly.velocity[0]:9.4f} "
        f"{dragonfly.velocity[2]:9.4f} "
        f"{brain.sweep_amp:7.3f} "
        f"{brain.feather_mean:8.4f} "
        f"{pitch:8.4f}"
    )

print()
print("final position:", dragonfly.position)
print("final velocity (body):", dragonfly.velocity)
print("final angular velocity (body):", dragonfly.angular_velocity)
print("attitude quat:", dragonfly.attitude)

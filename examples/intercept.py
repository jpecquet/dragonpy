"""
Intercept test: point-mass dragonfly detects a stationary prey and flies
toward it. Starts in hover, transitions to intercept when prey enters the
compound eye's FOV.

The InterceptBrain steers via stroke-plane tilt, which in the point-mass
model acts as effective body pitch.
"""

import numpy as np

from dragonpy.body.muscles import StrokePattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing
from dragonpy.brain import InterceptBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation, step_slow
from dragonpy.world import Environment
from dragonpy.world.prey import Prey


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
        hinge_position=np.zeros(3),
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

brain = InterceptBrain(
    hover_sweep_amp=0.45,
    feather_amp=0.7,
    feather_phase=np.pi / 2,
    wing_frequency=2.1,
    k_z=2.0,
    k_x=1.5,
    intercept_sweep_amp=1.0,
    intercept_feather_amp=0.7,
    k_tilt=5.0,
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

prey = Prey(
    position=np.array([10.0, 0.0, 5.0]),
    velocity=np.zeros(3),
    radius=0.05,
)

sim = Simulation(
    dragonfly=dragonfly,
    environment=Environment(prey=[prey]),
    dt_fast=1.0 / (2.1 * 100),
    fast_per_slow=15,
)

T_END = 30.0
print(f"{'t':>7} {'mode':>9} {'x':>8} {'z':>8} {'vx':>9} {'vz':>9} "
      f"{'tilt':>8} {'dist':>8}")
while sim.t < T_END:
    step_slow(sim)
    dist = float(np.linalg.norm(prey.position - dragonfly.position))
    print(
        f"{sim.t:7.2f} "
        f"{brain.mode:>9} "
        f"{dragonfly.position[0]:8.3f} "
        f"{dragonfly.position[2]:8.3f} "
        f"{dragonfly.velocity[0]:9.4f} "
        f"{dragonfly.velocity[2]:9.4f} "
        f"{brain.stroke_plane_tilt:8.4f} "
        f"{dist:8.3f}"
    )
    if dist < 0.2:
        print("\n*** prey captured ***")
        break

print()
print("final position:", dragonfly.position)
print("final velocity (body):", dragonfly.velocity)
print("distance to prey:", float(np.linalg.norm(prey.position - dragonfly.position)))

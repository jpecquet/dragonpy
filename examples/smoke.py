"""
Smoke test: build a dragonfly, install a static flapping pattern, and run
the simulation for a short time. Checks that nothing blows up and that the
state evolves as expected.

Not a correctness test. Just a pulse check on the assembled architecture.
"""

import numpy as np

from dragonpy.body.muscles import StrokePattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing
from dragonpy.brain import StaticBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation, run
from dragonpy.world import Environment


# --- flat-plate 2D airfoil coefficients ------------------------------------
def flat_plate_cl(alpha):
    return np.sin(2.0 * alpha)


def flat_plate_cd(alpha):
    return 1.0 - np.cos(2.0 * alpha)


# --- hinge orientations ----------------------------------------------------
# Right wing: hinge (+x_out, +y_fwd, +z_up) -> body (-y, +x, +z)
R_HINGE_RIGHT = np.array([
    [0.0,  1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0,  0.0, 1.0],
])
# Left wing: hinge (+x_out, +y_aft, +z_up) -> body (+y, -x, +z)
R_HINGE_LEFT = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
])


def make_wing(hinge_pos, chirality):
    R = R_HINGE_RIGHT if chirality == +1 else R_HINGE_LEFT
    return Wing(
        hinge_position=np.array(hinge_pos, dtype=float),
        hinge_orientation=R,
        chirality=chirality,
        span_ratio=0.8,
        mass_ratio=0.05,
        aero_ratio=0.15,
        lift_coeff=flat_plate_cl,
        drag_coeff=flat_plate_cd,
        n_elements=8,
    )


# --- build dragonfly -------------------------------------------------------
wings = [
    make_wing([ 0.20, -0.10, 0.10], +1),   # fore-right
    make_wing([ 0.20,  0.10, 0.10], -1),   # fore-left
    make_wing([-0.05, -0.10, 0.10], +1),   # hind-right
    make_wing([-0.05,  0.10, 0.10], -1),   # hind-left
]


def make_pattern(sweep_phase_offset: float) -> StrokePattern:
    return StrokePattern(
        stroke_plane_tilt=0.0,
        sweep_amp=1.0,
        sweep_mean=0.0,
        sweep_phase=sweep_phase_offset,
        elev_amp=0.2,
        elev_mean=0.0,
        elev_phase=0.0,
        elev_harmonic=2,
        feather_amp=0.7,
        feather_mean=0.0,
        feather_phase=np.pi / 2,
    )


# Fore pair in phase; hind pair lagging by ~half a cycle (typical dragonfly).
patterns = [
    make_pattern(0.0),
    make_pattern(0.0),
    make_pattern(np.pi),
    make_pattern(np.pi),
]

sensors = Sensors(
    inertial=InertialSensor(),
    eye=CompoundEye(fov_half_angle=np.pi / 2, max_range=20.0),
    ocelli=Ocelli(),
    airflow=AirflowSensor(),
    wing_load=WingLoadSensor(),
)

brain = StaticBrain(patterns=patterns, wing_frequency=2.1)

dragonfly = Dragonfly(
    wings=wings,
    sensors=sensors,
    brain=brain,
    stroke_patterns=[make_pattern(0.0) for _ in range(4)],  # overwritten on first brain tick
    inertia_body=np.array([0.01, 0.05, 0.05]),
    position=np.array([0.0, 0.0, 5.0]),
)

env = Simulation(
    dragonfly=dragonfly,
    environment=Environment(),
    dt_fast=1.0 / (2.1 * 100),   # 100 fast ticks per nominal wingbeat
    fast_per_slow=15,
)

# --- run and report --------------------------------------------------------
T_END = 2.0   # nondim time units (~sqrt(L/g) * 2)

print(f"{'t':>8} {'z':>10} {'vz':>10} {'|F|':>10} {'phase0':>10}")
while env.t < T_END:
    from dragonpy.dynamics import step_slow
    step_slow(env)
    print(
        f"{env.t:8.3f} "
        f"{dragonfly.position[2]:10.4f} "
        f"{dragonfly.velocity[2]:10.4f} "
        f"{np.linalg.norm(dragonfly.velocity):10.4f} "
        f"{dragonfly.wing_phases[0]:10.4f}"
    )

print()
print("final position:", dragonfly.position)
print("final velocity (body):", dragonfly.velocity)
print("final angular velocity (body):", dragonfly.angular_velocity)
print("attitude quat:", dragonfly.attitude)

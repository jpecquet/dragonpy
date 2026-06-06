"""
MBE vs proportional control: same stochastic prey realization, two
controllers, side-by-side capture metrics.

Scenario: the dragonfly starts at the origin, hovering. A prey appears
some distance away with a small drift velocity and constant-velocity +
white-noise acceleration dynamics. Both controllers see the same prey
realization (same RNG seed for the prey update). The dragonfly's wing /
body parameters are taken from `capture_study.py`, which is the
known-working configuration for stationary capture.

Proportional controller (InterceptBrain) steers on the current prey
bearing. MBE controller (MBEBrain) runs an EKF on the prey state and
steers on the lead bearing implied by a collision-course solve.
"""

import numpy as np

from dragonpy.body.muscles import StrokePattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing
from dragonpy.brain import InterceptBrain, MBEBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation, step_slow
from dragonpy.world import Environment
from dragonpy.world.prey import Prey, cv_white_noise_update


# ---------------------------------------------------------------------------
# Aero: Wang 2004 sinusoidal coefficients (matches §2.2.1 in the writeup).

CL_MAX, CD_MIN, CD_MAX = 1.2, 0.4, 2.4


def wang_cl(alpha):
    return CL_MAX * np.sin(2.0 * alpha)


def wang_cd(alpha):
    return CD_MIN + (CD_MAX - CD_MIN) * np.sin(alpha) ** 2


# ---------------------------------------------------------------------------
# Wing geometry.

R_HINGE_RIGHT = np.array([[ 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [ 0.0, 0.0, 1.0]])
R_HINGE_LEFT  = np.array([[ 0.0, -1.0, 0.0],
                          [ 1.0,  0.0, 0.0],
                          [ 0.0,  0.0, 1.0]])


def make_wing(chirality):
    return Wing(
        hinge_position=np.zeros(3),
        hinge_orientation=R_HINGE_RIGHT if chirality == +1 else R_HINGE_LEFT,
        chirality=chirality,
        span_ratio=0.75,
        mass_ratio=0.0,
        aero_ratio=0.025,
        lift_coeff=wang_cl,
        drag_coeff=wang_cd,
        n_elements=8,
    )


# ---------------------------------------------------------------------------
# Scenario.

SEED            = 42
QP              = 0.1                              # prey accel intensity (nondim)
PREY_INIT_POS   = np.array([6.0, 0.0, 0.0])
PREY_INIT_VEL   = np.array([0.0, 0.0, 1.0])
PREY_RADIUS     = 0.05
DRAGONFLY_INIT  = np.array([0.0, 0.0, 0.0])

WING_FREQUENCY    = 4.0
HIND_PHASE_OFFSET = np.pi / 2

SENSING_DELAY     = 1.0 / WING_FREQUENCY            # one wingbeat, per capture_study
HOVER_SWEEP_AMP   = np.radians(35)
FEATHER_AMP       = np.radians(60)
INTERCEPT_SWEEP_AMP   = np.radians(40)
INTERCEPT_FEATHER_AMP = np.radians(30)
K_TILT            = 2.0

T_END             = 20.0
CAPTURE_DIST      = 0.2
PRINT_EVERY_N     = 20

PLACEHOLDER_PATTERN = StrokePattern(
    stroke_plane_tilt=0.0,
    sweep_amp=0.0, sweep_mean=0.0, sweep_phase=0.0,
    elev_amp=0.0,  elev_mean=0.0,  elev_phase=0.0, elev_harmonic=2,
    feather_amp=0.0, feather_mean=0.0, feather_phase=0.0,
)


def build_sim(brain):
    """Fresh dragonfly + sim + prey. The prey RNG is seeded the same way
    for every call so both controllers see the same realization."""
    wings = [make_wing(c) for c in (+1, -1, +1, -1)]
    sensors = Sensors(
        inertial=InertialSensor(),
        eye=CompoundEye(
            fov_half_angle=np.pi,         # 360° vision, matches capture_study
            max_range=np.inf,
            delay=SENSING_DELAY,
        ),
        ocelli=Ocelli(),
        airflow=AirflowSensor(),
        wing_load=WingLoadSensor(),
    )
    dfly = Dragonfly(
        wings=wings,
        sensors=sensors,
        brain=brain,
        stroke_patterns=[
            StrokePattern(**PLACEHOLDER_PATTERN.__dict__) for _ in range(4)
        ],
        inertia_body=np.array([0.01, 0.05, 0.05]),
        position=DRAGONFLY_INIT.copy(),
        point_mass=True,
        wing_frequency=WING_FREQUENCY,
        wing_phases=np.array([0.0, 0.0, HIND_PHASE_OFFSET, HIND_PHASE_OFFSET]),
    )
    rng = np.random.default_rng(SEED)
    prey = Prey(
        position=PREY_INIT_POS.copy(),
        velocity=PREY_INIT_VEL.copy(),
        radius=PREY_RADIUS,
        update=cv_white_noise_update(QP, rng),
    )
    sim = Simulation(
        dragonfly=dfly,
        environment=Environment(prey=[prey]),
        dt_fast=1.0 / (WING_FREQUENCY * 100),
        fast_per_slow=15,
    )
    return sim, prey


def run_trial(label, brain):
    sim, prey = build_sim(brain)
    dfly = sim.dragonfly

    print(f"\n=== {label} ===")
    print(f"{'t':>6} {'mode':>9} {'x':>7} {'z':>7} {'vx':>7} {'vz':>7} "
          f"{'px':>7} {'pz':>7} {'dist':>7} {'tilt':>7}")
    step = 0
    captured = False
    min_dist = float("inf")
    min_dist_t = 0.0
    while sim.t < T_END:
        step_slow(sim)
        step += 1
        dist = float(np.linalg.norm(prey.position - dfly.position))
        if dist < min_dist:
            min_dist = dist
            min_dist_t = sim.t
        if step % PRINT_EVERY_N == 0:
            print(
                f"{sim.t:6.2f} {brain.mode:>9} "
                f"{dfly.position[0]:7.2f} {dfly.position[2]:7.2f} "
                f"{dfly.velocity[0]:7.3f} {dfly.velocity[2]:7.3f} "
                f"{prey.position[0]:7.2f} {prey.position[2]:7.2f} "
                f"{dist:7.3f} {brain.stroke_plane_tilt:7.3f}"
            )
        if dist < CAPTURE_DIST:
            captured = True
            print(f"*** captured at t={sim.t:.2f}, dist={dist:.3f} ***")
            break
    return {
        "captured":   captured,
        "t":          sim.t,
        "dist":       float(np.linalg.norm(prey.position - dfly.position)),
        "min_dist":   min_dist,
        "min_dist_t": min_dist_t,
        "final_pos":  dfly.position.copy(),
        "final_vel":  dfly.velocity.copy(),
    }


# ---------------------------------------------------------------------------
# Controllers.

proportional_brain = InterceptBrain(
    hover_sweep_amp=HOVER_SWEEP_AMP,
    feather_amp=FEATHER_AMP,
    feather_phase=np.pi / 2,
    wing_frequency=WING_FREQUENCY,
    k_z=1.0,
    k_x=1.0,
    intercept_sweep_amp=INTERCEPT_SWEEP_AMP,
    intercept_feather_amp=INTERCEPT_FEATHER_AMP,
    intercept_feather_phase=np.pi / 2,
    k_tilt=K_TILT,
)

# Known-but-noisy prior: ground-truth prey state plus a fixed Gaussian
# perturbation. The prior covariance equals the perturbation variance so the
# filter's initial uncertainty is calibrated against the actual error.
prior_rng     = np.random.default_rng(SEED + 1)
prior_pos_std = 0.2
prior_vel_std = 0.05
prior_perturb = np.array([
    prior_rng.normal(0, prior_pos_std),
    prior_rng.normal(0, prior_pos_std),
    prior_rng.normal(0, prior_vel_std),
    prior_rng.normal(0, prior_vel_std),
])
prior_mean = np.array([
    PREY_INIT_POS[0], PREY_INIT_POS[2],
    PREY_INIT_VEL[0], PREY_INIT_VEL[2],
]) + prior_perturb
prior_cov = np.diag([
    prior_pos_std ** 2, prior_pos_std ** 2,
    prior_vel_std ** 2, prior_vel_std ** 2,
])

mbe_brain = MBEBrain(
    hover_sweep_amp=HOVER_SWEEP_AMP,
    feather_amp=FEATHER_AMP,
    feather_phase=np.pi / 2,
    wing_frequency=WING_FREQUENCY,
    k_z=1.0,
    k_x=1.0,
    intercept_sweep_amp=INTERCEPT_SWEEP_AMP,
    intercept_feather_amp=INTERCEPT_FEATHER_AMP,
    intercept_feather_phase=np.pi / 2,
    k_tilt=K_TILT,
    prey_state_prior_mean=prior_mean,
    prey_state_prior_cov=prior_cov,
    qp_assumed=QP,
    R_bearing=1e-3,
    use_range=False,
    prey_radius=PREY_RADIUS,
    dragonfly_init_pos=DRAGONFLY_INIT.copy(),
    brake_range=3.0,
)


# ---------------------------------------------------------------------------
# Run both.

result_prop = run_trial("proportional", proportional_brain)
result_mbe  = run_trial("mbe",          mbe_brain)

print()
print("=== summary ===")
print(f"{'controller':>12} {'captured':>9} {'min_dist':>9} {'@t':>6}")
print(f"{'proportional':>12} {str(result_prop['captured']):>9} "
      f"{result_prop['min_dist']:>9.3f} {result_prop['min_dist_t']:>6.2f}")
print(f"{'mbe':>12} {str(result_mbe['captured']):>9} "
      f"{result_mbe['min_dist']:>9.3f} {result_mbe['min_dist_t']:>6.2f}")

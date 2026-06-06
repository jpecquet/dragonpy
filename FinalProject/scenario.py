"""
Shared scenario setup for the MBE study: wings, sensors, brains, prey
realization. Used by `run.py` and the plot scripts.

Two engagement geometries are defined in `SCENARIOS`:

- "vertical":   prey at (6, 0, 0) climbing at (0, 0, 1). Near-vertical
                line of sight; pure pursuit is close to a collision course.
- "horizontal": prey at (3, 0, 3) drifting at (1, 0, 0). Strong cross
                motion where lead pursuit should pay off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dragonpy.body.muscles import StrokePattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing
from dragonpy.brain import InterceptBrain, MBEBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation
from dragonpy.world import Environment
from dragonpy.world.prey import Prey, cv_white_noise_update


# ---------------------------------------------------------------------------
# Wang 2004 aero coefficients (matches §2.2.1 of the writeup).

CL_MAX, CD_MIN, CD_MAX = 1.2, 0.4, 2.4


def wang_cl(alpha):
    return CL_MAX * np.sin(2.0 * alpha)


def wang_cd(alpha):
    return CD_MIN + (CD_MAX - CD_MIN) * np.sin(alpha) ** 2


# ---------------------------------------------------------------------------
# Geometry and gains.

R_HINGE_RIGHT = np.array([[ 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [ 0.0, 0.0, 1.0]])
R_HINGE_LEFT  = np.array([[ 0.0, -1.0, 0.0],
                          [ 1.0,  0.0, 0.0],
                          [ 0.0,  0.0, 1.0]])

SEED            = 42
PREY_RADIUS     = 0.05
DRAGONFLY_INIT  = np.array([0.0, 0.0, 0.0])


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    prey_init_pos: np.ndarray
    prey_init_vel: np.ndarray
    qp: float = 0.1
    t_end: float = 20.0
    capture_dist: float = 0.35


SCENARIOS: dict[str, ScenarioConfig] = {
    "vertical": ScenarioConfig(
        name="vertical",
        prey_init_pos=np.array([6.0, 0.0, 0.0]),
        prey_init_vel=np.array([0.0, 0.0, 1.0]),
        qp=0.1,
        t_end=20.0,
    ),
    # In the horizontal case the dragonfly's chase has no gravity to
    # decelerate it; both controllers come close to capture but the
    # filter-convergence transient costs MBE its theoretical lead-pursuit
    # advantage. We stop at t=8 so the figures focus on the engagement
    # rather than any runaway phase. qp is reduced so the cross-flow
    # geometry remains visible through the noise.
    "horizontal": ScenarioConfig(
        name="horizontal",
        prey_init_pos=np.array([3.0, 0.0, 3.0]),
        prey_init_vel=np.array([1.0, 0.0, 0.0]),
        qp=0.05,
        t_end=8.0,
        capture_dist=0.75,
    ),
}

WING_FREQUENCY    = 4.0
HIND_PHASE_OFFSET = np.pi / 2

SENSING_DELAY     = 0.0
MUSCLE_DELAY      = 0.5 / WING_FREQUENCY
HOVER_SWEEP_AMP   = np.radians(35)
FEATHER_AMP       = np.radians(60)
INTERCEPT_SWEEP_AMP   = np.radians(40)
INTERCEPT_FEATHER_AMP = np.radians(30)
HOVER_TILT        = np.radians(30)
K_TILT            = 1.5

PRIOR_POS_STD     = 0.2
PRIOR_VEL_STD     = 0.2
R_BEARING         = 1e-2

PLACEHOLDER_PATTERN = StrokePattern(
    stroke_plane_tilt=0.0,
    sweep_amp=0.0, sweep_mean=0.0, sweep_phase=0.0,
    elev_amp=0.0,  elev_mean=0.0,  elev_phase=0.0, elev_harmonic=2,
    feather_amp=0.0, feather_mean=0.0, feather_phase=0.0,
)


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


def build_sim(brain, scenario: str = "vertical"):
    """Fresh dragonfly + sim + prey, seeded so both controllers see the
    same stochastic prey realization."""
    cfg = SCENARIOS[scenario]
    wings = [make_wing(c) for c in (+1, -1, +1, -1)]
    sensors = Sensors(
        inertial=InertialSensor(),
        eye=CompoundEye(
            fov_half_angle=np.pi, max_range=np.inf, delay=SENSING_DELAY,
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
        position=cfg.prey_init_pos.copy(),
        velocity=cfg.prey_init_vel.copy(),
        radius=PREY_RADIUS,
        update=cv_white_noise_update(cfg.qp, rng),
    )
    sim = Simulation(
        dragonfly=dfly,
        environment=Environment(prey=[prey]),
        dt_fast=1.0 / (WING_FREQUENCY * 100),
        fast_per_slow=15,
    )
    return sim, prey


def make_proportional_brain() -> InterceptBrain:
    return InterceptBrain(
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
        hover_stroke_plane_tilt=HOVER_TILT,
        muscle_delay=MUSCLE_DELAY,
    )


def make_prey_prior(scenario: str = "vertical") -> tuple[np.ndarray, np.ndarray]:
    """Known-but-noisy prior: ground-truth state with a fixed Gaussian
    perturbation. Returns (mean, cov)."""
    cfg = SCENARIOS[scenario]
    prior_rng = np.random.default_rng(SEED + 1)
    perturb = np.array([
        prior_rng.normal(0, PRIOR_POS_STD),
        prior_rng.normal(0, PRIOR_POS_STD),
        prior_rng.normal(0, PRIOR_VEL_STD),
        prior_rng.normal(0, PRIOR_VEL_STD),
    ])
    mean = np.array([
        cfg.prey_init_pos[0], cfg.prey_init_pos[2],
        cfg.prey_init_vel[0], cfg.prey_init_vel[2],
    ]) + perturb
    cov = np.diag([
        PRIOR_POS_STD ** 2, PRIOR_POS_STD ** 2,
        PRIOR_VEL_STD ** 2, PRIOR_VEL_STD ** 2,
    ])
    return mean, cov


def make_mbe_brain(scenario: str = "vertical") -> MBEBrain:
    cfg = SCENARIOS[scenario]
    mean, cov = make_prey_prior(scenario)
    return MBEBrain(
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
        hover_stroke_plane_tilt=HOVER_TILT,
        prey_state_prior_mean=mean,
        prey_state_prior_cov=cov,
        qp_assumed=cfg.qp,
        R_bearing=R_BEARING,
        use_range=True,
        R_range=0.1,
        prey_radius=PREY_RADIUS,
        dragonfly_init_pos=DRAGONFLY_INIT.copy(),
        muscle_delay=MUSCLE_DELAY,
    )

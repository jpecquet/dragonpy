"""
Parametric study for the interception problem.

Baseline parameters are defined in BASELINE. To sweep a parameter, override
it in the run_case() call.
"""

import numpy as np

from dragonpy.body.muscles import StrokePattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing
from dragonpy.brain import InterceptBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation, step_fast
from dragonpy.world import Environment
from dragonpy.world.prey import Prey


# Wang 2004 sinusoidal aero model (same as wang2007 validation case)
CL_MAX = 1.2
CD_MIN = 0.4
CD_MAX = 2.4


def wang_cl(alpha):
    return CL_MAX * np.sin(2.0 * alpha)


def wang_cd(alpha):
    return CD_MIN + (CD_MAX - CD_MIN) * np.sin(alpha) ** 2


R_HINGE_RIGHT = np.array([[0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
R_HINGE_LEFT = np.array([[0.0, -1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])

deg = np.radians

BASELINE = dict(
    # wing geometry
    aero_ratio=0.025,
    span_ratio=0.75,
    mass_ratio=0.0,

    # wing frequency: omega = 8*pi  =>  wing_frequency = omega/(2*pi) = 4
    wing_frequency=4.0,
    hind_phase_offset=np.pi,          # radians; phase lead of hind pair vs fore

    # hover kinematics
    hover_stroke_plane_tilt=deg(45),
    hover_sweep_amp=deg(35),
    hover_feather_amp=deg(60),
    hover_feather_phase=np.pi / 2,

    # intercept kinematics
    intercept_sweep_amp=deg(45),
    intercept_feather_amp=deg(30),
    intercept_feather_phase=np.pi / 2,

    # controller gains
    k_z=1.0,
    k_x=1.0,
    k_tilt=1.5,

    # prey
    prey_position=np.array([3.0, 0.0, 3.0]),

    # sensing
    sensing_delay=0.0,          # seconds; 0 = instantaneous
    fov_half_angle=np.pi / 2,   # radians; pi = 360 deg full FOV

    # simulation
    t_end=60.0,
    n_elements=8,
)


def run_case(label: str = "baseline", **overrides) -> dict:
    p = {**BASELINE, **overrides}

    def make_wing(chirality):
        R = R_HINGE_RIGHT if chirality == +1 else R_HINGE_LEFT
        return Wing(
            hinge_position=np.zeros(3),
            hinge_orientation=R,
            chirality=chirality,
            span_ratio=p["span_ratio"],
            mass_ratio=p["mass_ratio"],
            aero_ratio=p["aero_ratio"],
            lift_coeff=wang_cl,
            drag_coeff=wang_cd,
            n_elements=p["n_elements"],
        )

    wings = [make_wing(+1), make_wing(-1), make_wing(+1), make_wing(-1)]

    placeholder = StrokePattern(
        stroke_plane_tilt=0.0,
        sweep_amp=0.0, sweep_mean=0.0, sweep_phase=0.0,
        elev_amp=0.0, elev_mean=0.0, elev_phase=0.0, elev_harmonic=2,
        feather_amp=0.0, feather_mean=0.0, feather_phase=0.0,
    )

    sensors = Sensors(
        inertial=InertialSensor(),
        eye=CompoundEye(fov_half_angle=p["fov_half_angle"], max_range=np.inf, delay=p["sensing_delay"]),
        ocelli=Ocelli(),
        airflow=AirflowSensor(),
        wing_load=WingLoadSensor(),
    )

    brain = InterceptBrain(
        hover_sweep_amp=p["hover_sweep_amp"],
        feather_amp=p["hover_feather_amp"],
        feather_phase=p["hover_feather_phase"],
        wing_frequency=p["wing_frequency"],
        k_z=p["k_z"],
        k_x=p["k_x"],
        hover_stroke_plane_tilt=p["hover_stroke_plane_tilt"],
        intercept_sweep_amp=p["intercept_sweep_amp"],
        intercept_feather_amp=p["intercept_feather_amp"],
        intercept_feather_phase=p["intercept_feather_phase"],
        k_tilt=p["k_tilt"],
    )

    # Wing order: fore_R, fore_L, hind_R, hind_L
    hp = p["hind_phase_offset"]
    dragonfly = Dragonfly(
        wings=wings,
        sensors=sensors,
        brain=brain,
        stroke_patterns=[StrokePattern(**placeholder.__dict__) for _ in range(4)],
        inertia_body=np.array([0.01, 0.05, 0.05]),
        position=np.zeros(3),
        point_mass=True,
        wing_phases=np.array([0.0, 0.0, hp, hp]),
    )

    prey = Prey(
        position=p["prey_position"].copy(),
        velocity=np.zeros(3),
        radius=0.05,
    )

    wf = p["wing_frequency"]
    sim = Simulation(
        dragonfly=dragonfly,
        environment=Environment(prey=[prey]),
        dt_fast=1.0 / (wf * 100),
        fast_per_slow=15,
    )

    trajectory = [dragonfly.position.copy()]
    distances = [float(np.linalg.norm(prey.position - dragonfly.position))]
    times = [sim.t]
    dt_slow = sim.dt_fast * sim.fast_per_slow
    while sim.t < p["t_end"]:
        dragonfly.sensors.sample_all(sim)
        dragonfly.brain.update(dragonfly, dt_slow)
        for _ in range(sim.fast_per_slow):
            step_fast(sim)
            trajectory.append(dragonfly.position.copy())
            distances.append(float(np.linalg.norm(prey.position - dragonfly.position)))
            times.append(sim.t)

    distances = np.array(distances)
    times = np.array(times)
    i_min = int(np.argmin(distances))
    return dict(
        label=label,
        min_dist=float(distances[i_min]),
        t_min_dist=float(times[i_min]),
        t_final=sim.t,
        pos_final=dragonfly.position.copy(),
        vel_final=dragonfly.velocity.copy(),
        trajectory=np.array(trajectory),
        distances=distances,
        times=times,
        prey_position=p["prey_position"].copy(),
    )


def plot_trajectory_xz(result):
    import matplotlib.pyplot as plt
    from post.style import apply_matplotlib_style, resolve_style

    style = resolve_style()
    apply_matplotlib_style(style)

    traj = result["trajectory"]
    prey = result["prey_position"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(traj[:, 0], traj[:, 2], "-", color=style.trajectory_color,
            linewidth=1.4, label="dragonfly")
    ax.plot(traj[0, 0], traj[0, 2], "o", color=style.trajectory_color, markersize=5)
    ax.plot(prey[0], prey[2], "x", color=style.target_color,
            markersize=8, markeredgewidth=2, label="target")
    i_min = int(np.argmin(result["distances"]))
    ax.plot(traj[i_min, 0], traj[i_min, 2], "o", color=style.target_color,
            markersize=5, label="closest approach")

    ax.set_xlabel(r"$\tilde{X}$")
    ax.set_ylabel(r"$\tilde{Z}$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=style.font_size)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    delay = 1.0 / BASELINE["wing_frequency"]
    result = run_case("delay_1wb", sensing_delay=delay, t_end=10.0)
    print(f"{'Label':<20} {'min_dist':>10} {'t_min':>8}")
    print("-" * 40)
    print(f"{result['label']:<20} {result['min_dist']:10.4f} {result['t_min_dist']:8.2f}")
    plot_trajectory_xz(result)

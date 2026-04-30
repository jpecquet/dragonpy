"""
Reproduce the Wang & Russell 2007 tethered dragonfly validation case.

Runs one wingbeat with multi-harmonic kinematics extracted from experimental
data, then renders a stick-plot animation via post.plot_stick.
"""

import numpy as np

from dragonpy.body.muscles import StrokePattern, expand_pattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing, WingState, wing_wrench
from dragonpy.brain import StaticBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation, step_fast

# ---------------------------------------------------------------------------
# Dimensional morphology (from wang2007.md Table 1)
#   L = 51 mm, m = 620 mg, R_f = 46 mm, S_f = 380 mm²,
#   R_h = 43 mm, S_h = 440 mm², f = 33.4 Hz
# ---------------------------------------------------------------------------

OMEGA = 15.131323610234          # nondimensional angular frequency 2*pi*f*T0
WING_FREQ = OMEGA / (2.0 * np.pi)
STEPS_PER_WB = 200

# Aero coefficients (Wang 2004 sinusoidal model)
#   Cl(α) = CL_MAX · sin(2α)               → max Cl = CL_MAX at α = π/4
#   Cd(α) = CD_MIN + (CD_MAX - CD_MIN)·sin²α → min Cd = CD_MIN at α = 0
CL_MAX = 1.2
CD_MIN = 0.4
CD_MAX = 2.4


def wang_cl(alpha):
    return CL_MAX * np.sin(2.0 * alpha)


def wang_cd(alpha):
    return CD_MIN + (CD_MAX - CD_MIN) * np.sin(alpha) ** 2


# Hinge orientations (same as hover.py — +x spanwise outward)
R_HINGE_RIGHT = np.array([[ 0.0,  1.0,  0.0],
                           [-1.0,  0.0,  0.0],
                           [ 0.0,  0.0,  1.0]])
R_HINGE_LEFT  = np.array([[ 0.0, -1.0,  0.0],
                           [ 1.0,  0.0,  0.0],
                           [ 0.0,  0.0,  1.0]])

# ---------------------------------------------------------------------------
# Per-wing-pair parameters (from sim_wang2007.cfg, already in radians)
# ---------------------------------------------------------------------------

FORE = dict(
    lb0   = 0.901960784314,
    mu0   = 0.034537096774,
    gamma = 0.925024503557,
    phi_mean  = 0.004915964184,
    phi_harmonics = [
        (0.706037780510, -1.204374189276),
        (0.076305672724, -1.578010784884),
    ],
    psi_mean  = 0.587223636469,
    psi_harmonics = [
        (0.755307203563,  0.235331906025),
        (0.018516215488, -1.688892686728),
        (0.157978058365, -2.906146201606),
        (0.036079114685, -0.415303940399),
    ],
    cone_mean = 0.174532925199,
    cone_harmonics = [
        (0.017402712878,  2.548662663131),
        (0.018791087392,  0.576687642280),
        (0.028121320867, -1.811091059095),
        (0.016886077966,  2.037732674068),
        (0.000734766162, -0.581186734573),
    ],
)

HIND = dict(
    lb0   = 0.843137254902,
    mu0   = 0.037382258065,
    gamma = 0.767944870878,
    phi_mean  = 0.042130712253,
    phi_harmonics = [
        (0.785732254323, -0.817114757804),
        (0.115264353782, -0.898617794147),
    ],
    psi_mean  = 0.417743823427,
    psi_harmonics = [
        (0.768254858806,  0.603577493733),
        (0.238292054250,  0.397469590143),
        (0.083264876410, -1.274091872392),
        (0.030856373978, -2.508723333927),
    ],
    cone_mean = -0.087266462600,
    cone_harmonics = [
        (0.008452681927,  0.715910008531),
        (0.006424696603, -0.761359446904),
        (0.020308738443, -2.276028131709),
        (0.021255143230, -1.607522138327),
        (0.005689040323,  2.523625810292),
    ],
)


def make_wing(pair: dict, chirality: int) -> Wing:
    R = R_HINGE_RIGHT if chirality == +1 else R_HINGE_LEFT
    return Wing(
        hinge_position=np.zeros(3),
        hinge_orientation=R,
        chirality=chirality,
        span_ratio=pair["lb0"],
        mass_ratio=0.05,
        aero_ratio=pair["mu0"] / pair["lb0"],
        lift_coeff=wang_cl,
        drag_coeff=wang_cd,
        n_elements=5,
    )


def make_pattern(pair: dict) -> StrokePattern:
    return StrokePattern(
        stroke_plane_tilt=pair["gamma"],
        sweep_amp=0.0, sweep_mean=pair["phi_mean"], sweep_phase=0.0,
        elev_amp=0.0,  elev_mean=pair["cone_mean"], elev_phase=0.0,
        feather_amp=0.0, feather_mean=pair["psi_mean"], feather_phase=0.0,
        sweep_harmonics=pair["phi_harmonics"],
        feather_harmonics=pair["psi_harmonics"],
        elev_harmonics=pair["cone_harmonics"],
    )


# ---------------------------------------------------------------------------
# Build dragonfly  (order: fore_right, fore_left, hind_right, hind_left)
# ---------------------------------------------------------------------------

wing_specs = [
    (FORE, +1, "fore_right"),
    (FORE, -1, "fore_left"),
    (HIND, +1, "hind_right"),
    (HIND, -1, "hind_left"),
]

wings    = [make_wing(p, c) for p, c, _ in wing_specs]
patterns = [make_pattern(p) for p, _, _ in wing_specs]
names    = [n for _, _, n in wing_specs]

sensors = Sensors(
    inertial=InertialSensor(),
    eye=CompoundEye(fov_half_angle=np.pi / 2, max_range=20.0),
    ocelli=Ocelli(),
    airflow=AirflowSensor(),
    wing_load=WingLoadSensor(),
)

brain = StaticBrain(patterns=list(patterns), wing_frequency=WING_FREQ)

dragonfly = Dragonfly(
    wings=wings,
    sensors=sensors,
    brain=brain,
    stroke_patterns=[make_pattern(p) for p, _, _ in wing_specs],
    inertia_body=np.array([0.01, 0.05, 0.05]),
    point_mass=True,
    tethered=True,
    wing_phases=np.zeros(4),
)

dt_fast = 1.0 / (WING_FREQ * STEPS_PER_WB)
sim = Simulation(
    dragonfly=dragonfly,
    environment=__import__("dragonpy.world", fromlist=["Environment"]).Environment(),
    dt_fast=dt_fast,
    fast_per_slow=1,
)

# ---------------------------------------------------------------------------
# Run one wingbeat, recording wing vectors at each step
# ---------------------------------------------------------------------------

from dragonpy.body.wings import rot_x  # noqa: E402

T_WB = 2.0 * np.pi / OMEGA
n_steps = STEPS_PER_WB

time_arr = np.zeros(n_steps)
fz_total = np.zeros(n_steps)
wing_data = {name: {"e_r": [], "e_c": [], "e_s": []} for name in names}

_zero3 = np.zeros(3)

# Install patterns via StaticBrain
dragonfly.sensors.sample_all(sim)
dragonfly.brain.update(dragonfly, dt_fast)

for step_i in range(n_steps):
    step_fast(sim)
    time_arr[step_i] = sim.t
    fz_step = 0.0
    for i, (wing, name) in enumerate(zip(wings, names)):
        ws = dragonfly.wing_states[i]
        R_body_wing = wing.hinge_orientation @ ws.R_hinge_from_wing
        wing_data[name]["e_r"].append(R_body_wing[:, 0].copy())
        wing_data[name]["e_c"].append(wing.chirality * R_body_wing[:, 1].copy())
        tilt = wing.chirality * -dragonfly.stroke_patterns[i].stroke_plane_tilt
        e_s = wing.hinge_orientation @ rot_x(tilt) @ np.array([0.0, 0.0, 1.0])
        wing_data[name]["e_s"].append(e_s.copy())
        F_i, _ = wing_wrench(wing, ws, _zero3, _zero3, _zero3)
        fz_step += F_i[2]
    fz_total[step_i] = fz_step

# Stack into (N, 3) arrays
wings_dict = {}
for name in names:
    wings_dict[name] = {
        "e_r": np.array(wing_data[name]["e_r"]),
        "e_c": np.array(wing_data[name]["e_c"]),
        "e_s": np.array(wing_data[name]["e_s"]),
    }

# ---------------------------------------------------------------------------
# Hybrid Blender + matplotlib composite animation (dark mode)
# ---------------------------------------------------------------------------

from post.composite import render_hybrid_animation  # noqa: E402
from post.style import resolve_style  # noqa: E402

states_arr = np.zeros((n_steps, 6))

render_hybrid_animation(
    time_arr,
    states_arr,
    wings_dict,
    "wang2007_blender.mp4",
    wing_spans={n: FORE["lb0"] if "fore" in n else HIND["lb0"] for n in names},
    style=resolve_style(theme="dark"),
    center=[0.0, 0.0, 0.04],
    fps=30,
)

# ---------------------------------------------------------------------------
# XZ stick-plot animation (using compute_stick_endpoints directly)
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.animation as ani  # noqa: E402

from post.plot_stick import (  # noqa: E402
    STICK_LENGTH,
    compute_stick_endpoints,
    project_xz,
    wing_root_positions,
)
from post.style import apply_matplotlib_style, figure_size, resolve_style  # noqa: E402

style = resolve_style(theme="dark")
apply_matplotlib_style(style)

fore_lb0 = FORE["lb0"]
hind_lb0 = HIND["lb0"]
STATION = 2.0 / 3.0

fore_root, hind_root = wing_root_positions()
fore_wing = wings_dict["fore_right"]
hind_wing = wings_dict["hind_right"]

fore_centers  = np.zeros((n_steps, 2))
fore_leading  = np.zeros((n_steps, 2))
fore_trailing = np.zeros((n_steps, 2))
hind_centers  = np.zeros((n_steps, 2))
hind_leading  = np.zeros((n_steps, 2))
hind_trailing = np.zeros((n_steps, 2))

for i in range(n_steps):
    fore_centers[i], fore_leading[i], fore_trailing[i] = compute_stick_endpoints(
        fore_wing["e_r"][i], fore_wing["e_c"][i],
        fore_root, STICK_LENGTH, STATION, fore_lb0,
    )
    hind_centers[i], hind_leading[i], hind_trailing[i] = compute_stick_endpoints(
        hind_wing["e_r"][i], hind_wing["e_c"][i],
        hind_root, STICK_LENGTH, STATION, hind_lb0,
    )

fig, ax = plt.subplots(figsize=figure_size(0.6))

all_pts = np.vstack([fore_leading, fore_trailing, hind_leading, hind_trailing])
pad = 0.08
ax.set_xlim(float(np.min(all_pts[:, 0]) - pad), float(np.max(all_pts[:, 0]) + pad))
ax.set_ylim(float(np.min(all_pts[:, 1]) - pad), float(np.max(all_pts[:, 1]) + pad))
ax.set_aspect("equal")
ax.set_xlabel(r"$\tilde{X}$")
ax.set_ylabel(r"$\tilde{Z}$")
ax.set_title("Wang & Russell 2007 — right wing sticks (XZ)")
ax.grid(True, alpha=0.3)

for c_arr in (fore_centers, hind_centers):
    ax.plot(c_arr[:, 0], c_arr[:, 1], "-", color=style.muted_text_color, linewidth=0.5)
for root in (fore_root, hind_root):
    ax.plot(root[0], root[1], ".", color=style.muted_text_color, markersize=4)

le_marker = dict(
    marker="o", markersize=4,
    markerfacecolor=style.axes_facecolor,
    markeredgewidth=1.8, linestyle="none",
)

fore_stick, = ax.plot([], [], "-", color=style.body_color, linewidth=2.2)
hind_stick, = ax.plot([], [], "-", color=style.body_color, linewidth=2.2)
fore_le, = ax.plot([], [], markeredgecolor=style.body_color, **le_marker)
hind_le, = ax.plot([], [], markeredgecolor=style.body_color, **le_marker)
time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", color=style.text_color)

time_scale = OMEGA / (2.0 * np.pi)

fig.tight_layout()


def update(frame):
    fore_stick.set_data(
        [fore_trailing[frame, 0], fore_leading[frame, 0]],
        [fore_trailing[frame, 1], fore_leading[frame, 1]],
    )
    hind_stick.set_data(
        [hind_trailing[frame, 0], hind_leading[frame, 0]],
        [hind_trailing[frame, 1], hind_leading[frame, 1]],
    )
    fore_le.set_data([fore_leading[frame, 0]], [fore_leading[frame, 1]])
    hind_le.set_data([hind_leading[frame, 0]], [hind_leading[frame, 1]])
    time_text.set_text(r"$t/T_{wb} = %.2f$" % (time_arr[frame] * time_scale))
    return fore_stick, hind_stick, fore_le, hind_le, time_text


anim = ani.FuncAnimation(fig, update, frames=n_steps, interval=50, blit=False)

# ---------------------------------------------------------------------------
# Total vertical aerodynamic force (world +z) over one wingbeat
# ---------------------------------------------------------------------------

fig_fz, ax_fz = plt.subplots(figsize=figure_size(0.6))
ax_fz.plot(time_arr * time_scale, fz_total, color=style.body_color, linewidth=1.6)
ax_fz.axhline(0.0, color=style.muted_text_color, linewidth=0.5)
ax_fz.set_xlabel(r"$t/T_{wb}$")
ax_fz.set_ylabel(r"$\tilde{F}_z$ (world)")
ax_fz.set_title("Wang & Russell 2007 — total vertical aero force")
ax_fz.grid(True, alpha=0.3)
fig_fz.tight_layout()

plt.show()

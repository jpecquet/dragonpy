"""
3D wing motion check for Wang & Russell 2007 kinematics.

Each wing is drawn as a rectangular plate with its leading edge in red.
All four wings use the multi-harmonic stroke patterns from the experimental
data, with per-pair span and stroke-plane tilt.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

from dragonpy.body.muscles import StrokePattern, expand_pattern

# ── Wang 2007 parameters ──────────────────────────────────────────────────

R_HINGE_RIGHT = np.array([[ 0.0,  1.0,  0.0],
                           [-1.0,  0.0,  0.0],
                           [ 0.0,  0.0,  1.0]])
R_HINGE_LEFT  = np.array([[ 0.0, -1.0,  0.0],
                           [ 1.0,  0.0,  0.0],
                           [ 0.0,  0.0,  1.0]])

OMEGA = 15.131323610234
WING_FREQ = OMEGA / (2.0 * np.pi)
omega = OMEGA

FORE = dict(
    lb0   = 0.901960784314,
    mu0   = 0.034537096774,
    gamma = 0.925024503557,
    phi_mean  = 0.004915964184,
    phi_harmonics = [
        (0.706037780510, -1.204374189276),
        (0.076305672724, -1.578010784884),
    ],
    psi_mean  = 0.587223636469 ,
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
    psi_mean  = 0.417743823427 ,
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


# ── Wing layout ───────────────────────────────────────────────────────────

d = 0.15
WING_SPECS = [
    ("fore R", np.zeros(3), R_HINGE_RIGHT, +1, FORE),
    ("fore L", np.zeros(3), R_HINGE_LEFT,  -1, FORE),
    ("hind R", np.zeros(3), R_HINGE_RIGHT, +1, HIND),
    ("hind L", np.zeros(3), R_HINGE_LEFT,  -1, HIND),
]


def make_plate(span, chirality):
    chord = span * 0.25
    le_sign = chirality
    return np.array([
        [0.0,  le_sign * chord / 2, 0.0],
        [span, le_sign * chord / 2, 0.0],
        [span, -le_sign * chord / 2, 0.0],
        [0.0,  -le_sign * chord / 2, 0.0],
    ])


# ── Pre-compute frames ───────────────────────────────────────────────────

n_cycles = 2
frames_per_cycle = 60
n_frames = n_cycles * frames_per_cycle
phases = np.linspace(0, n_cycles * 2 * np.pi, n_frames, endpoint=False)

n_wings = len(WING_SPECS)
all_corners = np.zeros((n_frames, n_wings, 4, 3))

for fi, phase in enumerate(phases):
    for wi, (name, hinge_pos, R_bh, chirality, pair) in enumerate(WING_SPECS):
        pattern = make_pattern(pair)
        R_hw, _ = expand_pattern(pattern, phase, omega, chirality)
        R_bw = R_bh @ R_hw
        plate = make_plate(pair["lb0"], chirality)
        corners_body = (R_bw @ plate.T).T + hinge_pos
        all_corners[fi, wi] = corners_body

# ── Animation ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

WING_COLORS = ["#4488cc", "#cc8844", "#44cc88", "#cc44aa"]
WING_NAMES = [s[0] for s in WING_SPECS]

polys = []
le_lines = []
for wi in range(n_wings):
    poly = Poly3DCollection(
        [all_corners[0, wi]],
        alpha=0.4,
        facecolor=WING_COLORS[wi],
        edgecolor="k",
        linewidth=0.5,
    )
    ax.add_collection3d(poly)
    polys.append(poly)

    c = all_corners[0, wi]
    line, = ax.plot(
        [c[0, 0], c[1, 0]],
        [c[0, 1], c[1, 1]],
        [c[0, 2], c[1, 2]],
        color="red", linewidth=3,
    )
    le_lines.append(line)

# Body axes.
ax_len = 0.2
ax.quiver(0, 0, 0, ax_len, 0, 0, color="r", arrow_length_ratio=0.2, linewidth=1.5)
ax.quiver(0, 0, 0, 0, ax_len, 0, color="g", arrow_length_ratio=0.2, linewidth=1.5)
ax.quiver(0, 0, 0, 0, 0, ax_len, color="b", arrow_length_ratio=0.2, linewidth=1.5)

lim = max(FORE["lb0"], HIND["lb0"]) + 0.15
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_xlabel("x (forward)")
ax.set_ylabel("y (left)")
ax.set_zlabel("z (up)")
ax.set_title("Wang & Russell 2007 — wing motion check")
ax.set_aspect("equal")

legend_patches = [
    plt.Line2D([0], [0], color=WING_COLORS[i], lw=6, label=WING_NAMES[i])
    for i in range(n_wings)
] + [plt.Line2D([0], [0], color="red", lw=3, label="leading edge")]
ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)


def update(frame):
    for wi in range(n_wings):
        c = all_corners[frame, wi]
        polys[wi].set_verts([c])
        le_lines[wi].set_data_3d(
            [c[0, 0], c[1, 0]],
            [c[0, 1], c[1, 1]],
            [c[0, 2], c[1, 2]],
        )
    time_text.set_text(
        f"t/T_wb = {frame / frames_per_cycle:.2f}   "
        f"cycle {frame // frames_per_cycle + 1}/{n_cycles}"
    )
    return polys + le_lines + [time_text]


T_cycle = 1.0 / WING_FREQ
anim = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 * T_cycle / frames_per_cycle, blit=False,
)

plt.tight_layout()
plt.show()

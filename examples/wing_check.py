"""
Wing motion check: 3D animation of the four wings over a few wingbeat cycles.

Each wing is drawn as a rectangular plate with its leading edge in red.
Hinges are arranged in a square so the four wings are visually distinct.
Uses baseline parameters from the parametric study.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

from dragonpy.body.muscles import StrokePattern, expand_pattern
from dragonpy.body.wings import Wing

from examples.parametric import BASELINE, R_HINGE_RIGHT, R_HINGE_LEFT, deg

# ── Parameters ──────────────────────────────────────────────────────────────

span = BASELINE["span_ratio"]
chord = span * 0.25
wf = BASELINE["wing_frequency"]
omega = 2.0 * np.pi * wf

# Hinge layout: square, spaced so wings don't overlap visually.
d = 0.3
HINGE_SPECS = [
    ("fore R", np.array([ d, -d, 0.0]), R_HINGE_RIGHT, +1),
    ("fore L", np.array([ d,  d, 0.0]), R_HINGE_LEFT,  -1),
    ("hind R", np.array([-d, -d, 0.0]), R_HINGE_RIGHT, +1),
    ("hind L", np.array([-d,  d, 0.0]), R_HINGE_LEFT,  -1),
]

# Stroke pattern from BASELINE parametric study.
hover_pattern = StrokePattern(
    stroke_plane_tilt=BASELINE["hover_stroke_plane_tilt"],
    sweep_amp=BASELINE["hover_sweep_amp"],
    sweep_mean=0.0,
    sweep_phase=0.0,
    elev_amp=0.0,
    elev_mean=0.0,
    elev_phase=0.0,
    elev_harmonic=2,
    feather_amp=BASELINE["hover_feather_amp"],
    feather_mean=0.0,
    feather_phase=BASELINE["hover_feather_phase"],
)

# Wing plate corners in wing frame: span along +x, chord along +y.
# Leading edge is at +y for right wings, -y for left wings.
def make_plate(chirality):
    le_sign = chirality
    return np.array([
        [0.0,  le_sign * chord / 2, 0.0],   # root LE
        [span, le_sign * chord / 2, 0.0],   # tip LE
        [span, -le_sign * chord / 2, 0.0],  # tip TE
        [0.0,  -le_sign * chord / 2, 0.0],  # root TE
    ])

# ── Pre-compute frames ─────────────────────────────────────────────────────

n_cycles = 3
frames_per_cycle = 60
n_frames = n_cycles * frames_per_cycle
T_cycle = 1.0 / wf
phases = np.linspace(0, n_cycles * 2 * np.pi, n_frames, endpoint=False)

# For each frame, for each wing: transform plate corners to body frame.
# Shape: (n_frames, 4_wings, 4_corners, 3)
all_corners = np.zeros((n_frames, 4, 4, 3))

for fi, phase in enumerate(phases):
    for wi, (name, hinge_pos, R_bh, chirality) in enumerate(HINGE_SPECS):
        R_hw, _ = expand_pattern(hover_pattern, phase, omega, chirality)
        R_bw = R_bh @ R_hw
        plate = make_plate(chirality)
        corners_body = (R_bw @ plate.T).T + hinge_pos
        all_corners[fi, wi] = corners_body

# ── Animation ───────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

WING_COLORS = ["#4488cc", "#cc8844", "#44cc88", "#cc44aa"]
WING_NAMES = [s[0] for s in HINGE_SPECS]

polys = []
le_lines = []
for wi in range(4):
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

# Hinge markers.
for name, hinge_pos, _, _ in HINGE_SPECS:
    ax.plot(*hinge_pos, "ko", markersize=4)

# Body axes (small).
ax_len = 0.2
ax.quiver(0, 0, 0, ax_len, 0, 0, color="r", arrow_length_ratio=0.2, linewidth=1.5)
ax.quiver(0, 0, 0, 0, ax_len, 0, color="g", arrow_length_ratio=0.2, linewidth=1.5)
ax.quiver(0, 0, 0, 0, 0, ax_len, color="b", arrow_length_ratio=0.2, linewidth=1.5)

lim = span + 0.3
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_xlabel("x (forward)")
ax.set_ylabel("y (left)")
ax.set_zlabel("z (up)")
ax.set_title(f"Wing motion check — tilt={np.degrees(BASELINE['hover_stroke_plane_tilt']):.0f}°, "
              f"sweep={np.degrees(BASELINE['hover_sweep_amp']):.0f}°, "
              f"feather={np.degrees(BASELINE['hover_feather_amp']):.0f}°")
ax.set_aspect("equal")

legend_patches = [
    plt.Line2D([0], [0], color=WING_COLORS[i], lw=6, label=WING_NAMES[i])
    for i in range(4)
] + [plt.Line2D([0], [0], color="red", lw=3, label="leading edge")]
ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)


def update(frame):
    for wi in range(4):
        c = all_corners[frame, wi]
        polys[wi].set_verts([c])
        le_lines[wi].set_data_3d(
            [c[0, 0], c[1, 0]],
            [c[0, 1], c[1, 1]],
            [c[0, 2], c[1, 2]],
        )
    t = frame / frames_per_cycle / wf
    time_text.set_text(f"t = {t:.3f} T₀   cycle {frame // frames_per_cycle + 1}/{n_cycles}")
    return polys + le_lines + [time_text]


anim = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 * T_cycle / frames_per_cycle, blit=False,
)

plt.tight_layout()
plt.show()

"""
Wing motion check — XZ projection.

Shows the right fore and hind wing stick motion projected onto the body XZ
plane (sagittal view), using the same baseline hover pattern as wing_check.py.
Style follows post/plot_stick.py: 2D sticks with leading-edge markers and
muted center traces.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dragonpy.body.muscles import StrokePattern, expand_pattern
from dragonpy.body.wings import rot_x, rot_y, rot_z

from examples.parametric import BASELINE, R_HINGE_RIGHT, R_HINGE_LEFT, deg
from post.style import apply_matplotlib_style, figure_size, resolve_style

# ── Parameters ─────────────────────────────────────────────────────────────

span = BASELINE["span_ratio"]
chord = span * 0.25
wf = BASELINE["wing_frequency"]
omega = 2.0 * np.pi * wf

STICK_LENGTH = 0.12

d = 0.3
WING_SPECS = [
    ("fore", np.array([ d, -d, 0.0]), R_HINGE_RIGHT, +1),
    ("hind", np.array([-d, -d, 0.0]), R_HINGE_RIGHT, +1),
]

hover_pattern = StrokePattern(
    stroke_plane_tilt=np.pi / 2,
    sweep_amp=np.pi / 8,
    sweep_mean=0.0,
    sweep_phase=0.0,
    elev_amp=0.0,
    elev_mean=np.pi / 4,
    elev_phase=0.0,
    elev_harmonic=2,
    feather_amp=np.pi / 4,
    feather_mean=np.pi / 4,
    feather_phase=np.pi / 2,
)

STATION = 2.0 / 3.0

# ── Pre-compute frames ────────────────────────────────────────────────────

n_cycles = 3
frames_per_cycle = 60
n_frames = n_cycles * frames_per_cycle
phases = np.linspace(0, n_cycles * 2 * np.pi, n_frames, endpoint=False)

n_wings = len(WING_SPECS)
centers = np.zeros((n_wings, n_frames, 2))
leading = np.zeros((n_wings, n_frames, 2))
trailing = np.zeros((n_wings, n_frames, 2))


def project_xz(v):
    return np.array([v[0], v[2]])


for fi, phase in enumerate(phases):
    for wi, (name, hinge_pos, R_bh, chirality) in enumerate(WING_SPECS):
        R_hw, _ = expand_pattern(hover_pattern, phase, omega, chirality)
        R_bw = R_bh @ R_hw

        span_body = R_bw[:, 0]
        chord_body = R_bw[:, 1] * chirality

        root_xz = project_xz(hinge_pos)
        span_xz = project_xz(span_body)
        chord_xz = project_xz(chord_body)

        center = root_xz + STATION * span * span_xz

        chord_norm = np.linalg.norm(chord_xz)
        if chord_norm > 1e-9:
            chord_hat = chord_xz / chord_norm
        else:
            chord_hat = np.array([1.0, 0.0])

        centers[wi, fi] = center
        leading[wi, fi] = center + 0.25 * STICK_LENGTH * chord_hat
        trailing[wi, fi] = center - 0.75 * STICK_LENGTH * chord_hat

# ── Animation ──────────────────────────────────────────────────────────────

style = resolve_style(theme="light")
apply_matplotlib_style(style)

fig, ax = plt.subplots(figsize=figure_size(0.6))

all_pts = np.vstack([leading.reshape(-1, 2), trailing.reshape(-1, 2)])
pad = 0.08
ax.set_xlim(float(np.min(all_pts[:, 0]) - pad), float(np.max(all_pts[:, 0]) + pad))
ax.set_ylim(float(np.min(all_pts[:, 1]) - pad), float(np.max(all_pts[:, 1]) + pad))
ax.set_aspect("equal")
ax.set_xlabel(r"$X$ (forward)")
ax.set_ylabel(r"$Z$ (up)")
ax.set_title("Wing motion — XZ projection (right wings, hover baseline)")
ax.grid(True, alpha=0.3)

for wi in range(n_wings):
    ax.plot(
        centers[wi, :, 0],
        centers[wi, :, 1],
        "-",
        color=style.muted_text_color,
        linewidth=0.5,
    )

for wi, (name, hinge_pos, _, _) in enumerate(WING_SPECS):
    root_xz = project_xz(hinge_pos)
    ax.plot(root_xz[0], root_xz[1], ".", color=style.muted_text_color, markersize=4)

le_marker = dict(
    marker="o",
    markersize=4,
    markerfacecolor=style.axes_facecolor,
    markeredgewidth=1.8,
    linestyle="none",
)

WING_NAMES = [s[0] for s in WING_SPECS]
sticks = [ax.plot([], [], "-", color=style.body_color, linewidth=2.2)[0] for _ in range(n_wings)]
le_markers = [ax.plot([], [], markeredgecolor=style.body_color, **le_marker)[0] for _ in range(n_wings)]

time_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
    color=style.text_color,
)

legend_patches = [
    plt.Line2D([0], [0], color=style.muted_text_color, lw=1, label=WING_NAMES[i])
    for i in range(n_wings)
] + [
    plt.Line2D([0], [0], marker="o", color="none",
               markeredgecolor=style.body_color, markerfacecolor=style.axes_facecolor,
               markersize=4, markeredgewidth=1.8, label="leading edge"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=8)


def update(frame):
    artists = []
    for wi in range(n_wings):
        sticks[wi].set_data(
            [trailing[wi, frame, 0], leading[wi, frame, 0]],
            [trailing[wi, frame, 1], leading[wi, frame, 1]],
        )
        le_markers[wi].set_data(
            [leading[wi, frame, 0]],
            [leading[wi, frame, 1]],
        )
        artists.extend([sticks[wi], le_markers[wi]])
    t = frame / frames_per_cycle / wf
    time_text.set_text(r"$t/T_{wb} = %.2f$" % (frame / frames_per_cycle))
    artists.append(time_text)
    return tuple(artists)


T_cycle = 1.0 / wf
anim = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 * T_cycle / frames_per_cycle, blit=False,
)

plt.tight_layout()
plt.show()

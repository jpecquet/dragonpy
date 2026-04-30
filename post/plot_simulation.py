"""
3D flight animation of a dragonfly simulation.

Renders body (stick + head), wing planforms (composite-ellipse polygons),
trajectory trail, and velocity text using matplotlib.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from post.animation import save_animation
from post.style import apply_matplotlib_style, figure_size, resolve_style

# Body geometry (nondimensional)
_Lh = 0.1
_Lt = 0.25
_La = 1.0 - _Lh - _Lt

_WING_ROOT_DISTANCE = 0.1
_WING_ROOT_MIDPOINT_Z = 0.04
_WING_ROOT_TILT_DEG = 15.0

_half = 0.5 * _WING_ROOT_DISTANCE
_tilt = np.radians(_WING_ROOT_TILT_DEG)
_FW_X0 = _half * np.cos(_tilt)
_HW_X0 = -_half * np.cos(_tilt)
_FW_Z0 = _WING_ROOT_MIDPOINT_Z + _half * np.sin(_tilt)
_HW_Z0 = _WING_ROOT_MIDPOINT_Z - _half * np.sin(_tilt)

_A_XC = _HW_X0 - _La / 2
_T_XC = _A_XC + _La / 2 + _Lt / 2
_H_XC = _T_XC + _Lt / 2 + _Lh / 2

_RIGHT_Y = -0.04
_LEFT_Y = 0.04

_DEFAULT_LB0 = 0.75
_DEFAULT_ASPECT_RATIO = 5
_DEFAULT_ROOT_CHORD_RATIO = 4.0 / (np.pi * _DEFAULT_ASPECT_RATIO)

_FORCE_CENTER_FRACTION = 0.67
_FORCE_THRESHOLD = 1e-10

SIM_FPS = 30
SIM_BITRATE = 4000
SIM_DPI = 200
VIEWPORT_PADDING = 0.5


def _wing_offsets(wname):
    xo = _FW_X0 if "fore" in wname else _HW_X0
    zo = _FW_Z0 if "fore" in wname else _HW_Z0
    yo = _RIGHT_Y if "right" in wname else _LEFT_Y
    return np.array([xo, yo, zo], dtype=float)


def _wing_polygon_local(span, root_chord, n=24):
    eta = np.linspace(0.0, 1.0, n + 1)
    chord_scale = np.sqrt(np.clip(1.0 - (2.0 * eta - 1.0) ** 2, 0.0, None))
    x_le = 0.25 * root_chord * chord_scale
    x_te = -0.75 * root_chord * chord_scale
    y = span * eta
    leading = np.column_stack([x_le, y, np.zeros_like(y)])
    trailing = np.column_stack([x_te[-2::-1], y[-2::-1], np.zeros_like(y[:-1])])
    return np.vstack([leading, trailing])


def _wing_polygon_world(root, e_r, e_c, span, root_chord, n=24):
    poly = _wing_polygon_local(span, root_chord, n)
    return root + np.outer(poly[:, 0], e_c) + np.outer(poly[:, 1], e_r)


def _compute_viewport(states, padding=VIEWPORT_PADDING):
    positions = np.asarray(states)[:, 0:3]
    mins = positions.min(axis=0) - padding
    maxs = positions.max(axis=0) + padding
    extent = np.maximum(maxs - mins, 1.0)
    center = 0.5 * (mins + maxs)
    return center, extent


def animate_simulation(
    time,
    states,
    wings,
    outfile,
    *,
    wing_spans=None,
    style=None,
    show_axes=True,
    show_velocity=True,
    show_forces=False,
    force_scale=0.05,
    trail_length=-1,
    frame_step=1,
    fps=SIM_FPS,
    bitrate=SIM_BITRATE,
    dpi=SIM_DPI,
):
    """Render a 3D flight animation.

    Args:
        time: (N,) time array.
        states: (N, 6) array — [x, y, z, ux, uy, uz] per timestep.
        wings: dict mapping wing name to dict with 'e_r', 'e_c' arrays (N, 3),
               and optionally 'lift', 'drag' arrays (N, 3).
        outfile: output path (.mp4 or .gif).
        wing_spans: optional dict mapping wing name to span (lb0).
        style: StyleConfig (default: light theme).
        show_axes: draw axis labels and ticks.
        show_velocity: draw velocity text overlay.
        show_forces: draw lift/drag force vectors.
        force_scale: scaling factor for force vectors.
        trail_length: trajectory trail frames (<=0 = full trail).
        frame_step: render every Nth frame.
        fps: output framerate.
        bitrate: output bitrate (mp4).
        dpi: output resolution.
    """
    style = resolve_style(style)
    apply_matplotlib_style(style)

    states = np.asarray(states, dtype=float)
    time = np.asarray(time, dtype=float)
    if wing_spans is None:
        wing_spans = {}

    if frame_step > 1:
        idx = np.arange(0, len(time), frame_step)
        time = time[idx]
        states = states[idx]
        wings = {
            wname: {k: np.asarray(v)[idx] if np.ndim(v) >= 1 else v
                    for k, v in wdata.items()}
            for wname, wdata in wings.items()
        }

    n_frames = len(time)
    center, extent = _compute_viewport(states)

    fig = plt.figure(figsize=figure_size(0.65))
    ax = fig.add_subplot(111, projection="3d")

    half = 0.5 * extent
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])
    ax.set_box_aspect(extent / extent.max())

    if show_axes:
        ax.set_xlabel(r"$\tilde{X}$")
        ax.set_ylabel(r"$\tilde{Y}$")
        ax.set_zlabel(r"$\tilde{Z}$")
    else:
        ax.set_axis_off()

    ax.view_init(elev=30, azim=-60)

    trail_line, = ax.plot([], [], [], color=style.trajectory_color,
                          linewidth=style.trajectory_linewidth, alpha=0.7)

    body_line, = ax.plot([], [], [], color=style.body_color, linewidth=2.0,
                         solid_capstyle="round", zorder=2)
    head_scatter = ax.scatter([], [], [], s=30,
                              facecolors=style.axes_facecolor,
                              edgecolors=style.body_color,
                              linewidths=2.0, zorder=3)

    vel_text = None
    if show_velocity:
        vel_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                             fontsize=10, va="top", color=style.text_color)

    fig.tight_layout()

    _wing_artists = []

    def update(frame):
        nonlocal _wing_artists
        for a in _wing_artists:
            a.remove()
        _wing_artists.clear()

        xb = states[frame, 0:3]

        # Trajectory trail
        if trail_length <= 0:
            start = 0
        else:
            start = max(0, frame - trail_length + 1)
        if frame > 0:
            seg = states[start:frame + 1, 0:3]
            trail_line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        else:
            trail_line.set_data_3d([], [], [])

        # Body stick
        tail = xb + np.array([_A_XC - _La / 2, 0.0, 0.0])
        head = xb + np.array([_H_XC + _Lh / 2, 0.0, 0.0])
        body_line.set_data_3d([tail[0], head[0]], [tail[1], head[1]],
                              [tail[2], head[2]])
        head_scatter._offsets3d = ([head[0]], [head[1]], [head[2]])

        # Wings
        for wname, wdata in wings.items():
            e_r = np.asarray(wdata["e_r"][frame], dtype=float)
            e_c = np.asarray(wdata["e_c"][frame], dtype=float)
            nr, nc = np.linalg.norm(e_r), np.linalg.norm(e_c)
            if nr < 1e-12 or nc < 1e-12:
                continue
            e_r, e_c = e_r / nr, e_c / nc

            lb0 = float(wing_spans.get(wname, _DEFAULT_LB0))
            root_chord = _DEFAULT_ROOT_CHORD_RATIO * lb0
            root = xb + _wing_offsets(wname)

            verts = _wing_polygon_world(root, e_r, e_c, lb0, root_chord)
            poly = Poly3DCollection([verts],
                                    facecolors=style.wing_color,
                                    edgecolors=style.wing_edge_color,
                                    linewidths=0.6, alpha=0.75, zorder=1)
            ax.add_collection3d(poly)
            _wing_artists.append(poly)

            tip = root + lb0 * e_r
            spar = ax.plot([root[0], tip[0]], [root[1], tip[1]], [root[2], tip[2]],
                           color=style.wing_edge_color, linewidth=1.2, alpha=0.95,
                           zorder=2)[0]
            _wing_artists.append(spar)

            # Force vectors
            if show_forces:
                cp = root + _FORCE_CENTER_FRACTION * lb0 * e_r
                for key, color in [("lift", style.lift_color),
                                   ("drag", style.drag_color)]:
                    if key not in wdata:
                        continue
                    fvec = np.asarray(wdata[key][frame], dtype=float)
                    if np.linalg.norm(fvec) > _FORCE_THRESHOLD:
                        end = cp + force_scale * fvec
                        fline = ax.plot([cp[0], end[0]], [cp[1], end[1]],
                                        [cp[2], end[2]], color=color,
                                        linewidth=style.force_linewidth)[0]
                        _wing_artists.append(fline)

        if vel_text is not None:
            ux, uz = states[frame, 3], states[frame, 5]
            vel_text.set_text(f"$u_x$ = {ux:.2f}, $u_z$ = {uz:.2f}")

        return ()

    anim = ani.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

    save_animation(anim, outfile, fps=fps, bitrate=bitrate, dpi=dpi,
                   progress_label="Simulation")

    plt.close(fig)
    print(f"Saved: {outfile}")

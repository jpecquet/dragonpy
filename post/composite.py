"""
Hybrid Blender + matplotlib composite pipeline.

Renders matplotlib 3D axes with trajectory overlay, then composites
transparent Blender mesh frames on top, producing a final video.
"""

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.proj3d import proj_transform

from post.render_blender import find_blender, _extract_frame_data, _RENDER_SCRIPT
from post.style import apply_matplotlib_style, figure_size, resolve_style
from post.progress import pip_progress

_FORCE_CENTER_FRACTION = 0.67
_FORCE_THRESHOLD = 1e-10

_WING_ROOT_DISTANCE = 0.1
_WING_ROOT_MIDPOINT_Z = 0.04
_WING_ROOT_TILT_DEG = 15.0
_half = 0.5 * _WING_ROOT_DISTANCE
_tilt = np.radians(_WING_ROOT_TILT_DEG)
_FW_X0 = _half * np.cos(_tilt)
_HW_X0 = -_half * np.cos(_tilt)
_FW_Z0 = _WING_ROOT_MIDPOINT_Z + _half * np.sin(_tilt)
_HW_Z0 = _WING_ROOT_MIDPOINT_Z - _half * np.sin(_tilt)
_RIGHT_Y = -0.04
_LEFT_Y = 0.04

VIEWPORT_PADDING = 0.5


def _wing_offsets(wname):
    xo = _FW_X0 if "fore" in wname else _HW_X0
    zo = _FW_Z0 if "fore" in wname else _HW_Z0
    yo = _RIGHT_Y if "right" in wname else _LEFT_Y
    return np.array([xo, yo, zo], dtype=float)


def _compute_viewport(states, padding=VIEWPORT_PADDING):
    positions = np.asarray(states)[:, 0:3]
    mins = positions.min(axis=0) - padding
    maxs = positions.max(axis=0) + padding
    extent = np.maximum(maxs - mins, 1.0)
    center = 0.5 * (mins + maxs)
    return center, extent


def _major_tick_step(extent):
    return 0.5 if extent < 2.0 else 1.0


def compute_blender_ortho_scale(
    elevation, azimuth, center, extent, figsize, dpi, style,
    show_axes=True,
):
    """Compute ortho_scale for Blender to match the matplotlib 3D projection.

    Returns (ortho_scale, shift_x, shift_y) where shift values are
    Blender camera sensor shifts (fraction of sensor dimension).
    """
    apply_matplotlib_style(style)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=style.figure_facecolor)
    ax = fig.add_subplot(111, projection="3d", facecolor=style.axes_facecolor)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_proj_type("ortho")

    half = 0.5 * extent
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])
    try:
        ax.set_box_aspect(extent.tolist())
    except Exception:
        pass

    if show_axes:
        ax.set_xlabel(r"$\tilde{X}$", fontsize=10)
        ax.set_ylabel(r"$\tilde{Y}$", fontsize=10)
        ax.set_zlabel(r"$\tilde{Z}$", fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(_major_tick_step(extent[0])))
        ax.yaxis.set_major_locator(MultipleLocator(_major_tick_step(extent[1])))
        ax.zaxis.set_major_locator(MultipleLocator(_major_tick_step(extent[2])))
        fig.tight_layout()
    else:
        ax.set_axis_off()
        ax.set_position([0.0, 0.0, 1.0, 1.0])

    fig.canvas.draw()

    render_width = figsize[0] * dpi
    render_height = figsize[1] * dpi

    proj = ax.get_proj()

    azim_rad = math.radians(azimuth)
    cam_right = np.array([math.sin(azim_rad), -math.cos(azim_rad), 0.0])

    c = center
    cx, cy, _ = proj_transform(c[0], c[1], c[2], proj)
    center_disp = ax.transData.transform([(cx, cy)])[0]

    p1 = c + cam_right
    px, py, _ = proj_transform(p1[0], p1[1], p1[2], proj)
    p1_disp = ax.transData.transform([(px, py)])[0]

    pixels_per_world = np.linalg.norm(p1_disp - center_disp)
    ortho_scale = render_width / pixels_per_world

    offset_x = center_disp[0] - render_width / 2
    offset_y = center_disp[1] - render_height / 2

    shift_x = -offset_x / render_width
    shift_y = -offset_y / render_height

    plt.close(fig)
    return ortho_scale, shift_x, shift_y


def _render_mpl_frames(
    states, wings, wing_spans, style, center, extent,
    elevation, azimuth, figsize, dpi, output_dir,
    show_axes, show_velocity, show_forces, force_scale, trail_length,
):
    """Render matplotlib overlay frames (axes, grid, trail) to PNGs."""
    apply_matplotlib_style(style)
    n_frames = len(states)
    half = 0.5 * extent

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=style.figure_facecolor)
    ax = fig.add_subplot(111, projection="3d", facecolor=style.axes_facecolor)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_proj_type("ortho")

    def _setup():
        ax.set_xlim(center[0] - half[0], center[0] + half[0])
        ax.set_ylim(center[1] - half[1], center[1] + half[1])
        ax.set_zlim(center[2] - half[2], center[2] + half[2])
        try:
            ax.set_box_aspect(extent.tolist())
        except Exception:
            pass

        if show_axes:
            ax.set_xlabel(r"$\tilde{X}$", fontsize=10, color=style.text_color)
            ax.set_ylabel(r"$\tilde{Y}$", fontsize=10, color=style.text_color)
            ax.set_zlabel(r"$\tilde{Z}$", fontsize=10, color=style.text_color)
            ax.xaxis.set_major_locator(MultipleLocator(_major_tick_step(extent[0])))
            ax.yaxis.set_major_locator(MultipleLocator(_major_tick_step(extent[1])))
            ax.zaxis.set_major_locator(MultipleLocator(_major_tick_step(extent[2])))
            ax.tick_params(axis="x", which="major", pad=1)
            ax.tick_params(axis="y", which="major", pad=1)
            ax.tick_params(axis="z", which="major", pad=1)
            from matplotlib.colors import to_rgba
            pane_edge = to_rgba(style.axes_edge_color, alpha=0.8)
            grid_color = to_rgba(style.grid_color, alpha=0.45)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor(pane_edge)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["grid"]["color"] = grid_color
        else:
            ax.set_axis_off()
            ax.set_position([0.0, 0.0, 1.0, 1.0])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with pip_progress(n_frames, "Matplotlib", unit="frame") as progress:
        for frame in range(n_frames):
            ax.cla()
            _setup()

            xb = states[frame, 0:3]

            if trail_length <= 0:
                start = 0
            else:
                start = max(0, frame - trail_length + 1)
            if frame > 0:
                seg = states[start:frame + 1, 0:3]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        color=style.trajectory_color,
                        linewidth=style.trajectory_linewidth, alpha=0.7)

            if show_forces:
                for wname, wdata in wings.items():
                    lb0 = float(wing_spans.get(wname, 0.75))
                    cp = xb + _wing_offsets(wname) + _FORCE_CENTER_FRACTION * lb0 * np.asarray(wdata["e_r"][frame], dtype=float)
                    for key, color in [("lift", style.lift_color), ("drag", style.drag_color)]:
                        if key not in wdata:
                            continue
                        fvec = np.asarray(wdata[key][frame], dtype=float)
                        if np.linalg.norm(fvec) > _FORCE_THRESHOLD:
                            end = cp + force_scale * fvec
                            ax.plot([cp[0], end[0]], [cp[1], end[1]],
                                    [cp[2], end[2]], color=color, linewidth=2.0)

            if show_velocity:
                ux, uz = states[frame, 3], states[frame, 5]
                ax.text2D(0.02, 0.98, f"$u_x$ = {ux:.2f}, $u_z$ = {uz:.2f}",
                          transform=ax.transAxes, fontsize=10, va="top",
                          color=style.text_color)

            fig.savefig(
                output_dir / f"mpl_{frame:06d}.png",
                dpi=dpi, pad_inches=0,
            )
            progress.update(1)

    plt.close(fig)


def _run_blender_transparent(
    blender_path, states, wings, wing_spans,
    elevation, azimuth, ortho_scale, center, shift_x, shift_y,
    width, height, output_dir, n_workers=1,
):
    """Run Blender to render transparent-background frames."""
    frame_data = _extract_frame_data(states, wings, wing_spans)
    config = {
        "camera": {
            "elevation": elevation,
            "azimuth": azimuth,
            "ortho_scale": ortho_scale,
            "shift_x": shift_x,
            "shift_y": shift_y,
        },
        "center": list(center),
        "width": width,
        "height": height,
        "body_color": "#f2f5f7",
        "wing_color": "#8f9aa6",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / "config.json"
        data_file = tmpdir / "frame_data.json"

        with open(config_file, "w") as f:
            json.dump(config, f)
        with open(data_file, "w") as f:
            json.dump(frame_data, f)

        n_frames = len(states)
        n_workers = max(1, min(n_workers, 8))
        chunk_size = math.ceil(n_frames / n_workers)
        ranges = [
            (i * chunk_size, min((i + 1) * chunk_size, n_frames))
            for i in range(n_workers)
            if i * chunk_size < n_frames
        ]

        import time as _time
        from concurrent.futures import ProcessPoolExecutor

        def _run_chunk(start, end):
            cmd = [
                blender_path, "--background",
                "--python", str(_RENDER_SCRIPT),
                "--",
                "--data", str(data_file),
                "--config", str(config_file),
                "--output-dir", str(output_dir),
                "--start", str(start),
                "--end", str(end),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Blender failed (frames {start}-{end}): "
                    f"{result.stderr[-500:]}"
                )

        print(f"Rendering {n_frames} Blender frames ({n_workers} workers)...")

        if n_workers == 1:
            with pip_progress(n_frames, "Blender", unit="frame") as progress:
                _run_chunk(0, n_frames)
                progress.set(n_frames)
        else:
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [
                        executor.submit(_run_chunk, s, e) for s, e in ranges
                    ]
                    with pip_progress(n_frames, "Blender", unit="frame") as progress:
                        while not all(f.done() for f in futures):
                            rendered = len(list(output_dir.glob("frame_*.png")))
                            progress.set(rendered)
                            _time.sleep(0.5)
                        for f in futures:
                            f.result()
                        progress.set(n_frames)
            except (PermissionError, OSError):
                print("  Parallel render unavailable; falling back to 1 worker")
                with pip_progress(n_frames, "Blender", unit="frame") as progress:
                    _run_chunk(0, n_frames)
                    progress.set(n_frames)


def _assemble_composite_video(mpl_dir, blender_dir, output_file, fps):
    """Overlay transparent Blender frames on matplotlib frames via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(Path(mpl_dir) / "mpl_%06d.png"),
        "-framerate", str(fps),
        "-i", str(Path(blender_dir) / "frame_%06d.png"),
        "-filter_complex",
        "[1:v][0:v]scale2ref[fg][bg];"
        "[bg][fg]overlay=shortest=1:format=auto[tmp];"
        "[tmp]pad=ceil(iw/2)*2:ceil(ih/2)*2[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg overlay assembly failed: {result.stderr}")


def render_hybrid_animation(
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
    elevation=30.0,
    azimuth=-60.0,
    center=None,
    width=1950,
    height=1300,
    fps=30,
    frame_step=1,
    n_workers=1,
):
    """Render a hybrid Blender + matplotlib composite animation.

    Matplotlib renders 3D axes, grid, trajectory trail, and text overlays.
    Blender renders the dragonfly mesh with transparent background.
    The layers are composited via ffmpeg.

    Args:
        time: (N,) time array.
        states: (N, 6) array -- [x, y, z, ux, uy, uz] per timestep.
        wings: dict mapping wing name to dict with 'e_r', 'e_c' arrays (N, 3).
        outfile: output video path (.mp4).
        wing_spans: optional dict mapping wing name to span (lb0).
        style: StyleConfig (default: dark theme).
        show_axes: draw axis labels and ticks.
        show_velocity: draw velocity text overlay.
        show_forces: draw lift/drag force vectors.
        force_scale: scaling factor for force vectors.
        trail_length: trajectory trail frames (<=0 = full trail).
        elevation: camera elevation in degrees.
        azimuth: camera azimuth in degrees.
        center: camera target [x, y, z] (default: trajectory mean).
        width: render width in pixels.
        height: render height in pixels.
        fps: output framerate.
        frame_step: render every Nth frame.
        n_workers: number of parallel Blender processes.
    """
    blender_path = find_blender()
    if blender_path is None:
        raise RuntimeError(
            "Blender not found. Install Blender or add it to PATH."
        )

    style = resolve_style(style)
    states = np.asarray(states, dtype=float)
    time = np.asarray(time, dtype=float)
    if wing_spans is None:
        wing_spans = {}

    if frame_step > 1:
        idx = np.arange(0, len(time), frame_step)
        time = time[idx]
        states = states[idx]
        wings = {
            wname: {
                k: np.asarray(v)[idx] if np.ndim(v) >= 1 else v
                for k, v in wdata.items()
            }
            for wname, wdata in wings.items()
        }

    n_frames = len(time)
    if center is None:
        center = states[:, 0:3].mean(axis=0)
    center = np.asarray(center, dtype=float)

    _, extent = _compute_viewport(states)

    dpi = int(round(width / 6.5))
    figsize = (width / dpi, height / dpi)

    ortho_scale, shift_x, shift_y = compute_blender_ortho_scale(
        elevation, azimuth, center, extent, figsize, dpi, style,
        show_axes=show_axes,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mpl_dir = tmpdir / "mpl"
        blender_dir = tmpdir / "blender"

        _render_mpl_frames(
            states, wings, wing_spans, style, center, extent,
            elevation, azimuth, figsize, dpi, mpl_dir,
            show_axes, show_velocity, show_forces, force_scale, trail_length,
        )

        _run_blender_transparent(
            blender_path, states, wings, wing_spans,
            elevation, azimuth, ortho_scale, center.tolist(),
            shift_x, shift_y,
            width, height, blender_dir,
            n_workers=n_workers,
        )

        print("Assembling composite video...")
        _assemble_composite_video(mpl_dir, blender_dir, str(outfile), fps)

    print(f"Saved: {outfile}")

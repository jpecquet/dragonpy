"""
Blender 3D animation pipeline.

Orchestrates: serialize data → invoke Blender subprocess → assemble video.
"""

import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from post.hybrid_config import StyleConfig
from post.progress import pip_progress
from post.style import resolve_style

BLENDER_PATHS = {
    "Darwin": [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
    ],
    "Linux": [
        "/usr/bin/blender",
        "/snap/bin/blender",
    ],
    "Windows": [
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    ],
}

_RENDER_SCRIPT = Path(__file__).parent / "blender" / "render_dragonfly.py"


def find_blender() -> Optional[str]:
    try:
        result = subprocess.run(
            ["blender", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return "blender"
    except FileNotFoundError:
        pass

    system = platform.system()
    for path in BLENDER_PATHS.get(system, []):
        if Path(path).exists():
            return path
    return None


def _extract_frame_data(states, wings, wing_spans):
    wing_names = list(wings.keys())
    n_frames = len(states)
    wing_data = {}
    for wname in wing_names:
        wing_data[wname] = {
            "e_r": np.asarray(wings[wname]["e_r"]).tolist(),
            "e_c": np.asarray(wings[wname]["e_c"]).tolist(),
        }
    return {
        "n_frames": n_frames,
        "states": np.asarray(states).tolist(),
        "wing_names": wing_names,
        "wing_vectors": wing_data,
        "wing_lb0": {str(k): float(v) for k, v in wing_spans.items()},
    }


def _build_config(style, center, ortho_scale, width, height,
                  elevation, azimuth):
    return {
        "camera": {
            "elevation": elevation,
            "azimuth": azimuth,
            "ortho_scale": ortho_scale,
        },
        "center": list(center),
        "width": width,
        "height": height,
        "bg_color": style.figure_facecolor,
        "body_color": style.body_color,
        "wing_color": style.wing_color,
    }


def _assemble_video(frame_dir, output_file, framerate, n_frames):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(framerate),
        "-i", str(Path(frame_dir) / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def render_blender_animation(
    time,
    states,
    wings,
    outfile,
    *,
    wing_spans=None,
    style=None,
    elevation=30.0,
    azimuth=-60.0,
    ortho_scale=3.0,
    center=None,
    width=1950,
    height=1300,
    fps=30,
    frame_step=1,
    n_workers=1,
):
    """Render a Blender 3D animation of a dragonfly simulation.

    Args:
        time: (N,) time array.
        states: (N, 6) array — [x, y, z, ux, uy, uz] per timestep.
        wings: dict mapping wing name to dict with 'e_r', 'e_c' arrays (N, 3).
        outfile: output path (.mp4).
        wing_spans: optional dict mapping wing name to span (lb0).
        style: StyleConfig (default: dark theme).
        elevation: camera elevation in degrees.
        azimuth: camera azimuth in degrees.
        ortho_scale: orthographic camera scale (world units visible).
        center: camera target point [x, y, z] (default: trajectory mean).
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
        center = states[:, 0:3].mean(axis=0).tolist()

    config = _build_config(
        style, center, ortho_scale, width, height, elevation, azimuth
    )
    frame_data = _extract_frame_data(states, wings, wing_spans)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / "config.json"
        data_file = tmpdir / "frame_data.json"
        output_dir = tmpdir / "frames"
        output_dir.mkdir()

        with open(config_file, "w") as f:
            json.dump(config, f)
        with open(data_file, "w") as f:
            json.dump(frame_data, f)

        n_workers = max(1, min(n_workers, 8))
        chunk_size = math.ceil(n_frames / n_workers)
        ranges = [
            (i * chunk_size, min((i + 1) * chunk_size, n_frames))
            for i in range(n_workers)
            if i * chunk_size < n_frames
        ]

        import time as _time
        from concurrent.futures import ProcessPoolExecutor, as_completed

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
            return (start, end)

        print(f"Rendering {n_frames} Blender frames ({n_workers} workers)...")

        if n_workers == 1:
            with pip_progress(n_frames, "Blender", unit="frame") as progress:
                _run_chunk(0, n_frames)
                # Poll for rendered frames
                rendered = len(list(output_dir.glob("frame_*.png")))
                progress.set(rendered)
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

        print("Assembling video...")
        _assemble_video(output_dir, str(outfile), fps, n_frames)

    print(f"Saved: {outfile}")

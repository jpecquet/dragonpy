"""Shared animation save helpers."""

from pathlib import Path

import matplotlib.animation as ani

from post.progress import pip_progress


def save_animation(anim, outfile, fps, bitrate=2000, dpi=None, progress_label=None):
    """Save a matplotlib animation to mp4 or gif based on file extension."""
    progress = None
    progress_callback = None
    succeeded = False
    if progress_label:
        def _callback(frame_idx, total_frames):
            nonlocal progress
            total_frames = int(total_frames) if total_frames is not None else 0
            frame_idx = int(frame_idx)
            if progress is None:
                progress = pip_progress(total_frames, progress_label, unit="frame")
                progress.__enter__()
            elif progress.total != total_frames:
                progress.close(mark_complete=False)
                progress = pip_progress(total_frames, progress_label, unit="frame")
                progress.__enter__()
            progress.set(frame_idx + 1)
        progress_callback = _callback

    suffix = Path(outfile).suffix.lower()
    try:
        if suffix == ".mp4":
            writer = ani.FFMpegWriter(fps=fps, bitrate=bitrate)
            kwargs = {"writer": writer}
            if dpi is not None:
                kwargs["dpi"] = dpi
            if progress_callback is not None:
                kwargs["progress_callback"] = progress_callback
            anim.save(outfile, **kwargs)
            succeeded = True
            return
        if suffix == ".gif":
            kwargs = {"writer": "pillow", "fps": fps}
            if dpi is not None:
                kwargs["dpi"] = dpi
            if progress_callback is not None:
                kwargs["progress_callback"] = progress_callback
            anim.save(outfile, **kwargs)
            succeeded = True
            return
        raise ValueError(f"Unsupported animation format for '{outfile}'. Use .mp4 or .gif.")
    finally:
        if progress is not None:
            progress.close(mark_complete=succeeded)

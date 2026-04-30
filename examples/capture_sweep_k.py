"""
Sweep the pursuit gain k_tilt across multiple values, running a capture study
for each and generating the full set of half-disk plots (averaged-tilt plus
one panel per single tilt value).

Other parameters are held at the baseline defined alongside `make_setup` in
`examples/capture_study.py`: tau = T_wb, hind phase offset = pi/2, fov = 360
deg, baseline pursuit kinematics.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dragonpy.studies import run_capture_study
from examples.capture_plot import load_result, plot_half_disk
from examples.capture_study import WING_FREQUENCY, make_setup


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "capture"
T_WB = 1.0 / WING_FREQUENCY


def study_tag(k_tilt: float, sensing_delay: float, hind_phase: float) -> str:
    return (f"k{k_tilt:g}_tau{sensing_delay/T_WB:g}Twb"
            f"_hind{hind_phase/np.pi:g}pi_fov360")


def run_one(k_tilt: float, n_workers: int):
    sensing_delay = T_WB
    hind_phase = np.pi / 2
    radii  = np.arange(3.0, 12.0 + 1e-9, 1.0)
    thetas = np.deg2rad(np.arange(-90.0, 90.0 + 1e-9, 10.0))
    tilts  = np.deg2rad(np.arange(0.0, 360.0, 30.0))

    setup = make_setup(
        k_tilt=k_tilt,
        sensing_delay=sensing_delay,
        fov_half_angle=np.pi,
        hind_phase_offset=hind_phase,
    )

    n_total = len(radii) * len(thetas) * len(tilts)
    print(f"[k={k_tilt}] {n_total} trials, {n_workers} workers", flush=True)

    report_every = max(1, n_total // 20)

    def progress(idx, total, info):
        if idx % report_every == 0 or idx == total:
            pct = 100.0 * idx / total
            print(f"  [k={k_tilt}] {idx:4d}/{total} ({pct:3.0f}%)", flush=True)

    t0 = time.perf_counter()
    result = run_capture_study(
        setup=setup,
        radii=radii, thetas=thetas, tilts=tilts,
        max_wingbeats=1000.0,
        n_workers=n_workers,
        progress=progress,
    )
    elapsed = time.perf_counter() - t0
    n_timeouts = int((~result.crossed).sum())
    print(f"[k={k_tilt}] done in {elapsed:.1f}s "
          f"({elapsed / n_total * 1000:.0f} ms/trial wall, "
          f"{n_timeouts} timeouts)", flush=True)

    tag = study_tag(k_tilt, sensing_delay, hind_phase)
    out_dir = DATA_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "result.npz"
    np.savez(
        npz_path,
        radii=result.radii,
        thetas=result.thetas,
        tilts=result.tilts,
        final_position=result.final_position,
        final_velocity=result.final_velocity,
        final_time=result.final_time,
        crossed=result.crossed,
        target_position=result.target_position,
        wing_frequency=WING_FREQUENCY,
        T_wb=T_WB,
        k_tilt=k_tilt,
        sensing_delay=sensing_delay,
        hind_phase_offset=hind_phase,
        intercept_feather_amp=np.radians(30.0),
    )
    print(f"[k={k_tilt}] saved {npz_path}", flush=True)
    return result, out_dir


def make_plots_for(result, out_dir: Path, capture_radius: float):
    # Averaged over initial tilts.
    out = out_dir / f"half_disk_avg_rcap{capture_radius:g}.png"
    fig = plot_half_disk(result, capture_radius=capture_radius,
                         tilt=None, savepath=out)
    plt.close(fig)
    print(f"  saved {out.name}", flush=True)

    # One panel per initial tilt value.
    for tilt in result.tilts:
        deg = int(round(np.degrees(float(tilt))))
        out = out_dir / f"half_disk_tilt{deg:03d}_rcap{capture_radius:g}.png"
        fig = plot_half_disk(result, capture_radius=capture_radius,
                             tilt=float(tilt), savepath=out)
        plt.close(fig)
        print(f"  saved {out.name}", flush=True)


if __name__ == "__main__":
    n_workers = 6
    capture_radius = 0.5

    studies: list[tuple[float, object, Path]] = []

    # Run new studies.
    for k in (1.0, 3.0, 4.0):
        result, out_dir = run_one(k, n_workers)
        studies.append((k, result, out_dir))

    # Include the already-saved k=2 study for plot symmetry.
    k2_dir = DATA_ROOT / study_tag(2.0, T_WB, np.pi / 2)
    if (k2_dir / "result.npz").exists():
        studies.append((2.0, load_result(k2_dir / "result.npz"), k2_dir))

    # Generate the full plot set per study.
    for k, result, out_dir in sorted(studies, key=lambda t: t[0]):
        print(f"plotting k={k}...", flush=True)
        make_plots_for(result, out_dir, capture_radius=capture_radius)

    print("all done.")

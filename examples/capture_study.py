"""
Capture study: evaluate a pursuit controller's capture rate.

Sweeps target placements (R, theta in the xz plane) and initial stroke-plane
tilt for a fixed (gain, delay, kinematics) configuration. Each trial ends
when the dragonfly first crosses the circle of radius R around the origin,
or after `max_wingbeats / f` seconds. Capture rate is the fraction of trials
that finish within `capture_radius` of the target.

The function in `dragonpy.studies` is controller-agnostic; this script wires
it to the existing `InterceptBrain`. A higher-level study would loop over
gain / delay / kinematics and call `run_capture_study` for each point.

For multiprocessing, the `setup` callable must be picklable: we build it as
a `functools.partial` of the module-level `_build_simulation`.
"""

import os
import time
from functools import partial
from pathlib import Path

import numpy as np

from dragonpy.body.muscles import StrokePattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.body.wings import Wing
from dragonpy.brain import InterceptBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation
from dragonpy.studies import (
    capture_rate,
    captured_mask,
    run_capture_study,
    time_to_intercept,
)
from dragonpy.world import Environment
from dragonpy.world.prey import Prey


# --- Aero (Wang 2004 sinusoidal coefficients) -------------------------------

CL_MAX, CD_MIN, CD_MAX = 1.2, 0.4, 2.4


def wang_cl(alpha):
    return CL_MAX * np.sin(2.0 * alpha)


def wang_cd(alpha):
    return CD_MIN + (CD_MAX - CD_MIN) * np.sin(alpha) ** 2


# --- Wing geometry ----------------------------------------------------------

R_HINGE_RIGHT = np.array([[ 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [ 0.0, 0.0, 1.0]])
R_HINGE_LEFT  = np.array([[ 0.0, -1.0, 0.0],
                          [ 1.0,  0.0, 0.0],
                          [ 0.0,  0.0, 1.0]])

WING_FREQUENCY = 4.0


def make_wing(
    chirality: int,
    aero_ratio: float = 0.025,
    span_ratio: float = 0.75,
) -> Wing:
    return Wing(
        hinge_position=np.zeros(3),
        hinge_orientation=R_HINGE_RIGHT if chirality == +1 else R_HINGE_LEFT,
        chirality=chirality,
        span_ratio=span_ratio,
        mass_ratio=0.0,
        aero_ratio=aero_ratio,
        lift_coeff=wang_cl,
        drag_coeff=wang_cd,
        n_elements=8,
    )


# --- Per-trial Simulation factory -------------------------------------------
#
# Top-level so it can be pickled by multiprocessing. `make_setup` returns a
# functools.partial binding the per-study config; the per-trial inputs
# (`tilt`, `target_position`) are supplied by the study driver.

_PLACEHOLDER_PATTERN = StrokePattern(
    stroke_plane_tilt=0.0,
    sweep_amp=0.0, sweep_mean=0.0, sweep_phase=0.0,
    elev_amp=0.0,  elev_mean=0.0,  elev_phase=0.0,  elev_harmonic=2,
    feather_amp=0.0, feather_mean=0.0, feather_phase=0.0,
)


def _build_simulation(
    tilt: float,
    target_position: np.ndarray,
    *,
    k_tilt:                float,
    sensing_delay:         float,
    fov_half_angle:        float,
    hind_phase_offset:     float,
    intercept_feather_amp: float,
    aero_ratio:            float,
    span_ratio:            float,
) -> Simulation:
    wings = [
        make_wing(c, aero_ratio=aero_ratio, span_ratio=span_ratio)
        for c in (+1, -1, +1, -1)
    ]

    sensors = Sensors(
        inertial=InertialSensor(),
        eye=CompoundEye(
            fov_half_angle=fov_half_angle,
            max_range=np.inf,
            delay=sensing_delay,
        ),
        ocelli=Ocelli(),
        airflow=AirflowSensor(),
        wing_load=WingLoadSensor(),
    )

    brain = InterceptBrain(
        hover_sweep_amp=np.radians(35),
        feather_amp=np.radians(60),
        feather_phase=np.pi / 2,
        wing_frequency=WING_FREQUENCY,
        k_z=1.0,
        k_x=1.0,
        hover_stroke_plane_tilt=tilt,
        intercept_sweep_amp=np.radians(45),
        intercept_feather_amp=intercept_feather_amp,
        intercept_feather_phase=np.pi / 2,
        k_tilt=k_tilt,
    )

    dragonfly = Dragonfly(
        wings=wings,
        sensors=sensors,
        brain=brain,
        stroke_patterns=[
            StrokePattern(**_PLACEHOLDER_PATTERN.__dict__) for _ in range(4)
        ],
        inertia_body=np.array([0.01, 0.05, 0.05]),
        position=np.zeros(3),
        point_mass=True,
        wing_frequency=WING_FREQUENCY,
        wing_phases=np.array([0.0, 0.0, hind_phase_offset, hind_phase_offset]),
    )

    prey = Prey(
        position=target_position.astype(float).copy(),
        velocity=np.zeros(3),
        radius=0.05,
    )

    return Simulation(
        dragonfly=dragonfly,
        environment=Environment(prey=[prey]),
        dt_fast=1.0 / (WING_FREQUENCY * 100),
        fast_per_slow=15,
    )


def make_setup(
    *,
    k_tilt:                float,
    sensing_delay:         float,
    fov_half_angle:        float = np.pi,        # pi == full 360deg sphere
    hind_phase_offset:     float = np.pi / 2,
    intercept_feather_amp: float = np.radians(30),
    aero_ratio:            float = 0.025,
    span_ratio:            float = 0.75,
):
    """Bind a (gain, delay, kinematics) point and return a setup(tilt, target)
    callable. The result is a functools.partial — picklable for multiprocessing."""
    return partial(
        _build_simulation,
        k_tilt=k_tilt,
        sensing_delay=sensing_delay,
        fov_half_angle=fov_half_angle,
        hind_phase_offset=hind_phase_offset,
        intercept_feather_amp=intercept_feather_amp,
        aero_ratio=aero_ratio,
        span_ratio=span_ratio,
    )


# --- Driver -----------------------------------------------------------------

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = REPO_ROOT / "data" / "capture"

    T_wb = 1.0 / WING_FREQUENCY
    k_tilt = 2.0
    sensing_delay = T_wb         # tau = 1 wingbeat
    hind_phase_offset = np.pi / 2

    radii  = np.arange(3.0, 12.0 + 1e-9, 1.0)             # 3..12 in body lengths
    thetas = np.deg2rad(np.arange(-90.0, 90.0 + 1e-9, 10.0))   # -90..+90 step 10
    tilts  = np.deg2rad(np.arange(0.0, 360.0, 30.0))           # 0..330 step 30 (periodic)

    setup = make_setup(
        k_tilt=k_tilt,
        sensing_delay=sensing_delay,
        fov_half_angle=np.pi,                # 360deg vision
        hind_phase_offset=hind_phase_offset,
    )

    n_workers = max(1, (os.cpu_count() or 1))
    n_total = len(radii) * len(thetas) * len(tilts)
    print(f"grid: {len(radii)} R x {len(thetas)} theta x {len(tilts)} tilt = "
          f"{n_total} trials")
    print(f"controller: k_tilt={k_tilt}, tau={sensing_delay:.4f}s ({sensing_delay/T_wb:.2f} T_wb), "
          f"hind_phase_offset={hind_phase_offset/np.pi:.2f}pi")
    print(f"workers: {n_workers}")

    report_every = max(1, n_total // 40)

    def progress(idx, total, info):
        if idx % report_every == 0 or idx == total:
            flag = "hit" if info["crossed"] else "TIMEOUT"
            print(f"  [{idx:4d}/{total}] R={info['R']:.2f} "
                  f"theta={np.degrees(info['theta']):+6.1f}deg "
                  f"tilt={np.degrees(info['tilt']):5.1f}deg "
                  f"t={info['t_final']:6.2f}s {flag}", flush=True)

    t0 = time.perf_counter()
    result = run_capture_study(
        setup=setup,
        radii=radii, thetas=thetas, tilts=tilts,
        max_wingbeats=1000.0,
        n_workers=n_workers,
        progress=progress,
    )
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s "
          f"({elapsed / n_total * 1000:.0f} ms/trial wall, "
          f"{elapsed * n_workers / n_total * 1000:.0f} ms/trial CPU)")

    n_timeouts = int((~result.crossed).sum())
    if n_timeouts:
        print(f"WARNING: {n_timeouts}/{n_total} trials hit the timeout")

    print()
    print(f"{'capture_radius':>15} {'rate':>8} {'mean t_int':>12}")
    for cr in (0.1, 0.2, 0.5, 1.0, 2.0):
        rate = capture_rate(result, cr)
        ttis = time_to_intercept(result, cr)
        n_cap = int(captured_mask(result, cr).sum())
        mean_t = float(np.nanmean(ttis)) if n_cap else float("nan")
        print(f"{cr:>15.2f} {rate:>7.0%} ({n_cap:>4d}/{n_total})"
              f" {mean_t:>11.2f}s")

    tag = (f"k{k_tilt:g}_tau{sensing_delay/T_wb:g}Twb"
           f"_hind{hind_phase_offset/np.pi:g}pi_fov360")
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
        T_wb=T_wb,
        k_tilt=k_tilt,
        sensing_delay=sensing_delay,
        hind_phase_offset=hind_phase_offset,
    )
    print(f"saved {npz_path}")

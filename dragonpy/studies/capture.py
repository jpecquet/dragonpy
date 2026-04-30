"""
Capture study: sweep target placements (R, theta in the xz plane) and initial
stroke-plane tilt for a fixed controller, recording the final state when the
dragonfly first crosses the circle of radius R around its starting position.

The simulation core is controller-agnostic; this study expects the caller to
supply a `setup(tilt, target_position) -> Simulation` callable that builds a
fresh, configured Simulation per trial. Termination: the dragonfly's distance
from its starting position (in the xz plane) reaches R, or `max_wingbeats / f`
seconds elapse first.

Capture rate is post-processed from the raw final-state arrays by supplying a
capture radius — re-scoring is free, no need to rerun the sweep.

Conventions:
  * theta is measured in the xz plane from +x toward +z (standard polar with
    z as the "up-like" axis). target = R * (cos θ, 0, sin θ).
  * gravity is along -z, so theta -> -theta is NOT a free symmetry.
  * the eye is body-forward (+x); in point-mass mode the body frame is the
    world frame, so only theta in (-pi/2, +pi/2) is observable to the brain.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..dynamics import Simulation, step_fast


@dataclass
class CaptureStudyResult:
    radii:           np.ndarray   # (nR,)
    thetas:          np.ndarray   # (nTheta,)
    tilts:           np.ndarray   # (nTilt,)
    final_position:  np.ndarray   # (nR, nTheta, nTilt, 3)
    final_velocity:  np.ndarray   # (nR, nTheta, nTilt, 3)
    final_time:      np.ndarray   # (nR, nTheta, nTilt)
    crossed:         np.ndarray   # (nR, nTheta, nTilt) bool
    target_position: np.ndarray   # (nR, nTheta, 3)


def _target_xz(R: float, theta: float) -> np.ndarray:
    return np.array([R * np.cos(theta), 0.0, R * np.sin(theta)])


def _trial_worker(
    args: tuple,
) -> tuple[int, int, int, np.ndarray, np.ndarray, float, bool, np.ndarray]:
    """Top-level worker so it can be pickled by multiprocessing."""
    i, j, k, R, theta, tilt, setup, max_wingbeats = args
    target = _target_xz(R, theta)
    sim = setup(tilt, target.copy())
    pos, vel, t, did_cross = _run_one_trial(sim, R, max_wingbeats)
    return i, j, k, pos, vel, t, did_cross, target


def _run_one_trial(
    sim: Simulation, R: float, max_wingbeats: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    dfly = sim.dragonfly
    pos0 = dfly.position.copy()
    f = dfly.wing_frequency
    if f <= 0.0:
        # Brain may install frequency on first update; bail to a sensible
        # default rather than divide-by-zero. Caller can also set it pre-setup.
        raise ValueError(
            "dragonfly.wing_frequency must be positive at trial start; "
            "set it in the setup callable before returning the Simulation."
        )
    t_max = max_wingbeats / f
    R2 = R * R
    dt_slow = sim.dt_fast * sim.fast_per_slow

    crossed = False
    while sim.t < t_max:
        dfly.sensors.sample_all(sim)
        dfly.brain.update(dfly, dt_slow)
        for _ in range(sim.fast_per_slow):
            step_fast(sim)
            dx = dfly.position[0] - pos0[0]
            dz = dfly.position[2] - pos0[2]
            if dx * dx + dz * dz >= R2:
                crossed = True
                break
        if crossed:
            break

    return dfly.position.copy(), dfly.velocity.copy(), sim.t, crossed


def run_capture_study(
    setup:         Callable[[float, np.ndarray], Simulation],
    radii:         Sequence[float],
    thetas:        Sequence[float],
    tilts:         Sequence[float],
    max_wingbeats: float = 1000.0,
    n_workers:     int   = 1,
    progress:      Callable[[int, int, dict], None] | None = None,
) -> CaptureStudyResult:
    """Run one trial per (R, theta, tilt) cell and collect terminal states.

    `setup(tilt, target_position)` must return a fresh Simulation with the
    dragonfly initialized at the desired starting position (typically the
    origin) with zero velocity, and `wing_frequency` already set.

    With `n_workers > 1`, trials are dispatched to a multiprocessing pool;
    `setup` must then be picklable (a top-level function or a
    `functools.partial` of one — closures will not pickle).

    `progress(idx, total, info)` is called once per completed trial. With
    parallel workers, trials complete out of order, so `idx` is a counter
    of completed trials rather than a position in the (R, theta, tilt) grid.
    """
    radii  = np.asarray(radii,  dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    tilts  = np.asarray(tilts,  dtype=float)
    nR, nTh, nA = len(radii), len(thetas), len(tilts)

    final_pos = np.zeros((nR, nTh, nA, 3))
    final_vel = np.zeros((nR, nTh, nA, 3))
    final_t   = np.zeros((nR, nTh, nA))
    crossed   = np.zeros((nR, nTh, nA), dtype=bool)
    targets   = np.zeros((nR, nTh, 3))

    work = [
        (i, j, k, float(R), float(theta), float(tilt), setup, max_wingbeats)
        for i, R in enumerate(radii)
        for j, theta in enumerate(thetas)
        for k, tilt in enumerate(tilts)
    ]
    total = len(work)

    def _consume(it):
        for idx, (i, j, k, pos, vel, t, did_cross, target) in enumerate(it, 1):
            final_pos[i, j, k] = pos
            final_vel[i, j, k] = vel
            final_t[i, j, k]   = t
            crossed[i, j, k]   = did_cross
            targets[i, j]      = target
            if progress is not None:
                progress(idx, total, {
                    "R": float(radii[i]),
                    "theta": float(thetas[j]),
                    "tilt": float(tilts[k]),
                    "t_final": float(t),
                    "crossed": bool(did_cross),
                })

    if n_workers <= 1:
        _consume(_trial_worker(w) for w in work)
    else:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            _consume(pool.imap_unordered(_trial_worker, work))

    return CaptureStudyResult(
        radii=radii,
        thetas=thetas,
        tilts=tilts,
        final_position=final_pos,
        final_velocity=final_vel,
        final_time=final_t,
        crossed=crossed,
        target_position=targets,
    )


def captured_mask(
    result: CaptureStudyResult, capture_radius: float,
) -> np.ndarray:
    """(nR, nTheta, nTilt) bool: trial both reached the circle and finished
    within `capture_radius` of the target (in the xz plane)."""
    target = result.target_position[:, :, None, :]
    miss   = result.final_position - target
    dist   = np.linalg.norm(miss[..., [0, 2]], axis=-1)
    return result.crossed & (dist <= capture_radius)


def capture_rate(result: CaptureStudyResult, capture_radius: float) -> float:
    """Fraction of trials captured at the given capture radius."""
    return float(captured_mask(result, capture_radius).mean())


def time_to_intercept(
    result: CaptureStudyResult, capture_radius: float,
) -> np.ndarray:
    """(nR, nTheta, nTilt) array: final_time for captured trials, NaN else."""
    mask = captured_mask(result, capture_radius)
    return np.where(mask, result.final_time, np.nan)

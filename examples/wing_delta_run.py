"""
Long pursuit run for wing-kinematics-vs-hover analysis.

Generates one capture trajectory at the stationary-prey baseline (K=3,
baseline kinematics, all `BASELINE_KWARGS` defaults), with initial stroke-
plane tilt = 0 and target at R=50, theta=0 (straight ahead in +x). R is
deliberately well past the sweep grid (3..12) so the pursuit has time to
approach a gravity-balanced steady state.

Records, at every FAST tick:
  - body position and velocity (point-mass run, so attitude stays identity)
  - brain stroke-plane tilt gamma
  - per-wing R_hinge_from_wing
  - wing phases
plus the constant per-wing geometry (hinge orientation, span, chirality), so
wing-tip trajectories through the air can be reconstructed downstream and
compared to hover.

Output: data/wing_delta/<tag>.npz
"""

from pathlib import Path

import numpy as np

from dragonpy.dynamics import step_fast
from examples.capture_study import WING_FREQUENCY, make_setup


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = REPO_ROOT / "data" / "wing_delta"
T_WB      = 1.0 / WING_FREQUENCY


def run_long_pursuit(
    R:             float,
    theta:         float,
    tilt0:         float,
    k_tilt:        float = 3.0,
    max_wingbeats: float = 200.0,
):
    setup = make_setup(
        k_tilt=k_tilt,
        sensing_delay=T_WB,
        fov_half_angle=np.pi,
        hind_phase_offset=np.pi / 2,
        intercept_feather_amp=np.radians(30.0),
        aero_ratio=0.025,
        span_ratio=0.75,
    )
    target = np.array([R * np.cos(theta), 0.0, R * np.sin(theta)])
    sim = setup(tilt0, target.copy())
    dfly = sim.dragonfly

    f = dfly.wing_frequency
    if f <= 0.0:
        raise ValueError("wing_frequency must be positive")
    t_max = max_wingbeats / f
    R2 = R * R
    pos0 = dfly.position.copy()
    n_wings = len(dfly.wings)
    dt_slow = sim.dt_fast * sim.fast_per_slow

    times      = []
    positions  = []
    velocities = []
    gammas     = []
    Rhw        = [[] for _ in range(n_wings)]
    phases     = []

    def snapshot():
        times.append(float(sim.t))
        positions.append(dfly.position.copy())
        velocities.append(dfly.velocity.copy())
        gammas.append(float(getattr(dfly.brain, "stroke_plane_tilt", 0.0)))
        for i, ws in enumerate(dfly.wing_states):
            Rhw[i].append(ws.R_hinge_from_wing.copy())
        phases.append(dfly.wing_phases.copy())

    snapshot()
    crossed = False
    while sim.t < t_max:
        dfly.sensors.sample_all(sim)
        dfly.brain.update(dfly, dt_slow)
        for _ in range(sim.fast_per_slow):
            step_fast(sim)
            snapshot()
            dx = dfly.position[0] - pos0[0]
            dz = dfly.position[2] - pos0[2]
            if dx * dx + dz * dz >= R2:
                crossed = True
                break
        if crossed:
            break

    R_hinge_from_wing = np.stack(
        [np.stack(per_wing) for per_wing in Rhw], axis=1,
    )  # (N, n_wings, 3, 3)

    return {
        "t":                 np.array(times),
        "pos":               np.stack(positions),
        "vel":               np.stack(velocities),
        "gamma":             np.array(gammas),
        "R_hinge_from_wing": R_hinge_from_wing,
        "wing_phases":       np.stack(phases),
        "target":            target,
        "crossed":           np.array(crossed),
        "k_tilt":            np.array(k_tilt),
        "tilt0":             np.array(tilt0),
        "T_wb":              np.array(T_WB),
        "wing_frequency":    np.array(WING_FREQUENCY),
        "hinge_orientations": np.stack(
            [w.hinge_orientation for w in dfly.wings]
        ),  # (n_wings, 3, 3)
        "span_ratios": np.array([w.span_ratio for w in dfly.wings]),
        "chiralities": np.array([w.chirality for w in dfly.wings]),
    }


if __name__ == "__main__":
    R, theta, tilt0 = 50.0, 0.0, 0.0
    k_tilt = 3.0

    print(f"running pursuit: R={R}, theta={np.degrees(theta):+.0f}deg, "
          f"tilt0={np.degrees(tilt0):+.0f}deg, K={k_tilt}")
    out = run_long_pursuit(R, theta, tilt0, k_tilt=k_tilt)

    t_end = float(out["t"][-1])
    speed_xz = float(np.linalg.norm(out["vel"][-1, [0, 2]]))
    print(f"  finished: t={t_end:.2f}s ({t_end / T_WB:.1f} T_wb), "
          f"crossed={bool(out['crossed'])}")
    print(f"  final pos: {out['pos'][-1]}")
    print(f"  final speed (xz): {speed_xz:.3f}")
    print(f"  recorded {len(out['t'])} fast ticks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f"R{int(R)}_theta{int(round(np.degrees(theta)))}"
           f"_tilt{int(round(np.degrees(tilt0)))}_K{k_tilt:g}")
    npz = OUT_DIR / f"{tag}.npz"
    np.savez(npz, **out)
    print(f"  saved {npz}")

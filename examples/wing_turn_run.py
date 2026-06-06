"""
Sharp-turn run for maneuvering-kinematics analysis.

Pursues a stationary target initially at (50, 0, 0) under the same baseline
kinematics + K=3 controller as the long-pursuit run in `wing_delta_run.py`.
When the body enters a 1 BL trigger radius around the initial target
position, the target teleports up by Δz = +50 (matching `INITIAL_TARGET[0]`),
landing at (50, 0, 50) — perpendicular to the current motion direction in
the body x-z plane. This forces a ~90 degree upward turn from forward
cruise. The run continues until the body comes within `CAPTURE_RADIUS` of
the new target, or `MAX_WINGBEATS` elapses.

Records the same per-fast-tick state as `wing_delta_run.py`, plus the
target position history (so the jump moment is reconstructible) and the
fast-tick index of the jump.

Output: data/wing_delta/turn_R50_dz50_K3.npz
"""

from pathlib import Path

import numpy as np

from dragonpy.dynamics import step_fast
from examples.capture_study import WING_FREQUENCY, make_setup


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR   = REPO_ROOT / "data" / "wing_delta"
T_WB      = 1.0 / WING_FREQUENCY

INITIAL_TARGET       = np.array([50.0, 0.0, 0.0])
JUMP_DELTA           = np.array([50.0, 0.0, 20.0])  # +x and +z — modest 21° turn
JUMP_TRIGGER_RADIUS  = 1.0      # body lengths from initial target
CAPTURE_RADIUS       = 0.5      # body lengths from current target
MAX_WINGBEATS        = 300.0


def run_turn(
    initial_tilt:  float = 0.0,
    k_tilt:        float = 3.0,
    max_wingbeats: float = MAX_WINGBEATS,
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
    sim = setup(initial_tilt, INITIAL_TARGET.copy())
    dfly = sim.dragonfly
    prey = sim.environment.prey[0]

    f = dfly.wing_frequency
    if f <= 0.0:
        raise ValueError("wing_frequency must be positive")
    t_max = max_wingbeats / f
    dt_slow = sim.dt_fast * sim.fast_per_slow
    n_wings = len(dfly.wings)

    times      = []
    positions  = []
    velocities = []
    gammas     = []
    Rhw        = [[] for _ in range(n_wings)]
    phases     = []
    targets    = []

    jump_idx     = -1   # fast-tick index when the jump fires (-1 = not fired)
    captured     = False

    def snapshot():
        times.append(float(sim.t))
        positions.append(dfly.position.copy())
        velocities.append(dfly.velocity.copy())
        gammas.append(float(getattr(dfly.brain, "stroke_plane_tilt", 0.0)))
        for i, ws in enumerate(dfly.wing_states):
            Rhw[i].append(ws.R_hinge_from_wing.copy())
        phases.append(dfly.wing_phases.copy())
        targets.append(prey.position.copy())

    snapshot()

    R_jump_sq    = JUMP_TRIGGER_RADIUS * JUMP_TRIGGER_RADIUS
    R_cap_sq     = CAPTURE_RADIUS * CAPTURE_RADIUS

    while sim.t < t_max:
        dfly.sensors.sample_all(sim)
        dfly.brain.update(dfly, dt_slow)
        for _ in range(sim.fast_per_slow):
            step_fast(sim)
            snapshot()

            d_to_target = dfly.position - prey.position
            d2 = float(d_to_target @ d_to_target)

            if jump_idx < 0:
                d_to_initial = dfly.position - INITIAL_TARGET
                if float(d_to_initial @ d_to_initial) <= R_jump_sq:
                    prey.position = INITIAL_TARGET + JUMP_DELTA
                    jump_idx = len(times) - 1
                    print(f"  target jumped at t={sim.t:.3f}s "
                          f"(idx {jump_idx}), pos={dfly.position}, "
                          f"new target={prey.position}")
            else:
                if d2 <= R_cap_sq:
                    captured = True
                    break
        if captured:
            break

    R_hinge_from_wing = np.stack(
        [np.stack(per_wing) for per_wing in Rhw], axis=1,
    )

    return {
        "t":                 np.array(times),
        "pos":               np.stack(positions),
        "vel":               np.stack(velocities),
        "gamma":             np.array(gammas),
        "R_hinge_from_wing": R_hinge_from_wing,
        "wing_phases":       np.stack(phases),
        "target_pos":        np.stack(targets),
        "jump_idx":          np.array(jump_idx),
        "captured":          np.array(captured),
        "k_tilt":            np.array(k_tilt),
        "tilt0":             np.array(initial_tilt),
        "T_wb":              np.array(T_WB),
        "wing_frequency":    np.array(WING_FREQUENCY),
        "initial_target":    INITIAL_TARGET.copy(),
        "jump_delta":        JUMP_DELTA.copy(),
        "jump_trigger_radius": np.array(JUMP_TRIGGER_RADIUS),
        "capture_radius":    np.array(CAPTURE_RADIUS),
        "hinge_orientations": np.stack(
            [w.hinge_orientation for w in dfly.wings]
        ),
        "span_ratios": np.array([w.span_ratio for w in dfly.wings]),
        "chiralities": np.array([w.chirality for w in dfly.wings]),
    }


if __name__ == "__main__":
    print(f"running turn pursuit: initial target {INITIAL_TARGET}, "
          f"jump Δ={JUMP_DELTA}, K=3")
    out = run_turn()

    t_end = float(out["t"][-1])
    speed_xz = float(np.linalg.norm(out["vel"][-1, [0, 2]]))
    j = int(out["jump_idx"])
    print(f"  finished: t={t_end:.2f}s ({t_end / T_WB:.1f} T_wb), "
          f"captured={bool(out['captured'])}")
    if j >= 0:
        print(f"  jump at t={out['t'][j]:.3f}s ({out['t'][j] / T_WB:.1f} T_wb), "
              f"body pos={out['pos'][j]}")
    print(f"  final pos: {out['pos'][-1]}")
    print(f"  final speed (xz): {speed_xz:.3f}")
    print(f"  recorded {len(out['t'])} fast ticks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = (f"turn_dx{int(JUMP_DELTA[0])}_dz{int(JUMP_DELTA[2])}_K3.npz")
    npz = OUT_DIR / tag
    np.savez(npz, **out)
    print(f"  saved {npz}")

"""
Run both controllers on the shared scenario, recording per-slow-tick
time-series for downstream plotting. Saves npz files to FinalProject/data/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dragonpy.dynamics import step_slow

from scenario import (
    PREY_RADIUS, SCENARIOS,
    build_sim, make_mbe_brain, make_proportional_brain,
)


DATA_DIR = Path(__file__).resolve().parent / "data"


def _bearing_and_range(detections):
    if not detections:
        return np.nan, np.nan
    d = detections[0]
    bearing = float(np.arctan2(d.bearing[2], d.bearing[0]))
    rng = PREY_RADIUS / np.tan(0.5 * d.angular_size)
    return bearing, rng


def run_and_record(brain, brain_kind: str, scenario: str = "vertical") -> dict:
    """Run one trial, snapshot at each slow tick, return arrays."""
    sim, prey = build_sim(brain, scenario)
    dfly = sim.dragonfly
    cfg = SCENARIOS[scenario]
    t_end = cfg.t_end
    capture_dist = cfg.capture_dist

    rec: dict = {
        "t":           [],
        "dfly_pos":    [],
        "dfly_vel":    [],
        "prey_pos":    [],
        "prey_vel":    [],
        "tilt_cmd":    [],     # brain-commanded (pre-delay)
        "tilt":        [],     # wing-applied (post-delay)
        "sweep_amp_cmd": [],
        "sweep_amp":   [],
        "intercepting": [],
        "bearing_meas": [],
        "range_meas":   [],
    }
    if brain_kind == "mbe":
        rec.update({
            "filter_mean":       [],
            "filter_cov":        [],
            "position_estimate": [],
            "lead_point":        [],
        })

    def snap():
        rec["t"].append(sim.t)
        rec["dfly_pos"].append(dfly.position.copy())
        rec["dfly_vel"].append(dfly.velocity.copy())
        rec["prey_pos"].append(prey.position.copy())
        rec["prey_vel"].append(prey.velocity.copy())
        rec["tilt_cmd"].append(float(brain.stroke_plane_tilt))
        rec["sweep_amp_cmd"].append(float(brain.sweep_amp))
        applied = dfly.stroke_patterns[0]
        rec["tilt"].append(float(applied.stroke_plane_tilt))
        rec["sweep_amp"].append(float(applied.sweep_amp))
        rec["intercepting"].append(brain.mode == "intercept")
        b, r = _bearing_and_range(dfly.sensors.eye.reading)
        rec["bearing_meas"].append(b)
        rec["range_meas"].append(r)
        if brain_kind == "mbe":
            rec["filter_mean"].append(brain.filter.mean.copy())
            rec["filter_cov"].append(brain.filter.cov.copy())
            rec["position_estimate"].append(brain.position_estimate.copy())
            rec["lead_point"].append(np.asarray(brain._lead_point, dtype=float).copy())

    captured = False
    t_capture = float("nan")
    snap()                                        # initial state (t = 0)
    while sim.t < t_end:
        step_slow(sim)
        snap()
        dist = float(np.linalg.norm(prey.position - dfly.position))
        if dist < capture_dist:
            captured = True
            t_capture = float(sim.t)
            break

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["captured"]  = np.array(captured)
    out["t_capture"] = np.array(t_capture)
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        print(f"\n=== scenario: {scenario} ===")

        print("running proportional ...")
        prop = run_and_record(
            make_proportional_brain(), "proportional", scenario,
        )
        np.savez(DATA_DIR / f"{scenario}_proportional.npz", **prop)
        print(f"  captured={bool(prop['captured'])} "
              f"t_capture={float(prop['t_capture']):.2f}")

        print("running mbe ...")
        mbe = run_and_record(make_mbe_brain(scenario), "mbe", scenario)
        np.savez(DATA_DIR / f"{scenario}_mbe.npz", **mbe)
        print(f"  captured={bool(mbe['captured'])} "
              f"t_capture={float(mbe['t_capture']):.2f}")

    print(f"\nsaved to {DATA_DIR}")


if __name__ == "__main__":
    main()

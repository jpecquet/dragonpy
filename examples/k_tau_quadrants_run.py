"""Run the 3 missing-quadrant k-tau sweeps with 360-deg FOV."""

import numpy as np

from examples.k_tau_sweep import run_and_save
from examples.parametric import BASELINE


if __name__ == "__main__":
    wf = BASELINE["wing_frequency"]
    T_wb = 1.0 / wf

    ks = np.arange(0.0, 5.0 + 1e-9, 0.5)
    taus = np.arange(0.0, 3.0 * T_wb + 1e-9, T_wb / 4.0)
    t_end = 10.0

    fov = np.pi  # 360 deg full

    quadrants = [
        np.array([-3.0, 0.0, +3.0]),
        np.array([-3.0, 0.0, -3.0]),
        np.array([+3.0, 0.0, -3.0]),
    ]

    for prey in quadrants:
        run_and_save(prey, ks, taus, t_end, T_wb, fov_half_angle=fov)

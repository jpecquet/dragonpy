"""k-tau sweeps for x=+3, z=+/-3 with fore/hind phase offset = 0."""

import numpy as np

from examples.k_tau_sweep import run_and_save
from examples.parametric import BASELINE


if __name__ == "__main__":
    wf = BASELINE["wing_frequency"]
    T_wb = 1.0 / wf

    ks = np.arange(0.0, 5.0 + 1e-9, 0.5)
    taus = np.arange(0.0, 3.0 * T_wb + 1e-9, T_wb / 4.0)
    t_end = 10.0

    fov = np.pi

    preys = [
        np.array([+3.0, 0.0, +3.0]),
        np.array([+3.0, 0.0, -3.0]),
    ]
    phase_cases = [
        (3 * np.pi / 4, "_phase135"),
    ]
    for phase, suffix in phase_cases:
        for prey in preys:
            run_and_save(
                prey, ks, taus, t_end, T_wb,
                fov_half_angle=fov,
                extra_overrides={"hind_phase_offset": phase},
                tag_suffix=suffix,
            )

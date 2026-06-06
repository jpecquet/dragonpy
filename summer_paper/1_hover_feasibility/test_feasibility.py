"""
Validate the vectorized cycle-averaged force against the scalar reference
(which is built directly on the trusted dragonpy functions).

Run:  python summer_paper/1_hover_feasibility/test_feasibility.py
"""

import time

import numpy as np

from feasibility import (
    Params, cycle_averaged_force, cycle_averaged_force_scalar, replace,
)

# A spread of parameter points, including off-default gamma0/sigma0 and
# asymmetric pitch timing, so the rotation/chirality paths are all exercised.
CASES = [
    Params(),
    Params(gamma0=np.radians(40.0), sigma0=np.radians(120.0)),
    Params(Lw=0.65, phi1=np.radians(55.0), psi0=np.radians(-30.0),
           psi1=np.radians(20.0), delta0=np.radians(-70.0)),
    Params(Lw=0.85, Aw_over_mb=0.25, omega_star=20.0, psi0=np.radians(50.0),
           delta0=np.radians(150.0)),
    Params(omega_star=8.0, phi1=np.radians(5.0), Aw_over_mb=0.05),
    # Nonzero body velocity (forward flight) with off-default pitch/tilt.
    Params(gamma0=np.radians(25.0), psi0=np.radians(15.0),
           delta0=np.radians(110.0), v_body=(0.6, 0.0, 1.2)),
    Params(v_body=(0.4, -0.3, 0.9), sigma0=np.radians(90.0)),
]


def main():
    print("vectorized vs scalar cycle-averaged force")
    print("-" * 60)
    worst = 0.0
    for i, p in enumerate(CASES):
        a = cycle_averaged_force_scalar(p, n_phase=128)
        b = cycle_averaged_force(p, n_phase=128)
        err = float(np.max(np.abs(a - b)))
        worst = max(worst, err)
        print(f"  case {i}: scalar={a}  max|diff|={err:.2e}")
    print("-" * 60)
    print(f"worst abs error = {worst:.2e}")
    assert worst < 1e-10, "vectorized path diverges from scalar reference"

    # Speedup on a representative batch.
    p = Params()
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        cycle_averaged_force_scalar(p, n_phase=128)
    t_scalar = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n):
        cycle_averaged_force(p, n_phase=128)
    t_vec = time.perf_counter() - t0
    print(f"\n{n} evals @ n_phase=128:")
    print(f"  scalar     {t_scalar:7.3f} s  ({1e3*t_scalar/n:.2f} ms/eval)")
    print(f"  vectorized {t_vec:7.3f} s  ({1e3*t_vec/n:.2f} ms/eval)")
    print(f"  speedup    {t_scalar / t_vec:.0f}x")
    print("\nOK")


if __name__ == "__main__":
    main()

"""
Vertical aerodynamic force over one wingbeat: the unsteady force (with the body's
actual within-cycle velocity) vs the static assumption (v_body = 0) on which the
hover equilibrium is built.

Worst case for the discrepancy ("maximum wobble"): for the equilibrium |Fbar| = 1
the wing-speed scale s0*omega* is roughly fixed, while the body-velocity wobble
goes like impulse/mass ~ 1/omega*. So the ratio of body wobble to wing speed
scales like sqrt(C_psi * Aw*/m*) / omega*, maximized at the smallest omega* and
the largest Aw*/m* of the parameter ranges. We therefore take omega* = 8,
Aw*/m* = 0.25 (range extremes), gamma0 = 40 deg, C_psi held near max.

Reuses the equilibrium solve, point-mass integrator, and single-2/3-span-element
force from hover_drift.py.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import Params, REF_LW, STUDY_ELEMENT, N_WINGS, cycle_averaged_force, replace
from hover_drift import instant_force, simulate, solve_equilibrium

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
N_PHASE = 256

# Worst-case operating point for within-cycle wobble.
GAMMA0 = np.radians(40.0)
PSI1 = np.radians(51.0)
DELTA0 = np.pi / 2.0
# A*/m* and omega* deviate from the reference config (REF_A_OVER_M, REF_OMEGA_STAR)
# ON PURPOSE: the range extremes (max loading, min frequency) give the worst-case
# within-cycle wobble. Wing length stays at the reference value.
# A_OVER_M = 1.0 (total) is the per-wing A_w*/m* = 0.25 upper bound times N_w.
A_OVER_M, OMEGA_STAR, LW = 1.0, 8.0, REF_LW
SIGMA0 = np.pi

N_CYCLES = 20
SPC = 400          # steps per cycle (smooth force curve)


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p0 = Params(gamma0=GAMMA0, psi1=PSI1, delta0=DELTA0, A_over_m=A_OVER_M,
                omega_star=OMEGA_STAR, Lw=LW, sigma0=SIGMA0,
                element_span_fracs=STUDY_ELEMENT)
    psi0, phi1 = solve_equilibrium(p0)
    p = replace(p0, psi0=psi0, phi1=phi1)
    omega = 2.0 * np.pi * p.wing_frequency       # = omega_star

    t, pos, vel, period = simulate(p, n_cycles=N_CYCLES, steps_per_cycle=SPC)

    # Last full wingbeat: reconstruct vertical force with the actual v_body(t)
    # (unsteady) and with v_body = 0 (the static equilibrium assumption).
    seg = slice(-(SPC + 1), None)
    tt, vv = t[seg], vel[seg]
    phase_deg = np.linspace(0.0, 360.0, SPC + 1)
    Fz_unsteady = np.array([instant_force(p, omega * ti, omega, vi)[2]
                            for ti, vi in zip(tt, vv)])
    Fz_static = np.array([instant_force(p, omega * ti, omega, np.zeros(3))[2]
                          for ti in tt])

    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.axhline(1.0, color="0.75", lw=0.9, ls=":", zorder=0)
    ax.text(2.0, 1.0, "weight", color="0.5", fontsize=style.font_size - 2,
            va="bottom", ha="left")
    ax.plot(phase_deg, Fz_static, color="black", lw=1.6, ls="--",
            label=r"static ($v_b = 0$)")
    ax.plot(phase_deg, Fz_unsteady, color="black", lw=1.6,
            label=r"unsteady ($v_b \neq 0$)")
    ax.set_xlabel("wingbeat phase (deg)")
    ax.set_ylabel(r"vertical force $F_z$")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.legend(fontsize=style.font_size - 1, frameon=True, loc="best")

    out = OUT_DIR / "hover_force_wingbeat.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    # --- report numbers ---
    Fbar = cycle_averaged_force(p, N_PHASE)
    vz = vv[:, 2]
    wing_speed = (2.0 / 3.0) * LW * phi1 * omega
    print(f"worst case: omega*={OMEGA_STAR}, A*/m*={A_OVER_M} (Aw*/m*={A_OVER_M / N_WINGS}), gamma0=40 deg")
    print(f"equilibrium: psi0={np.degrees(psi0):.2f} deg, phi1={np.degrees(phi1):.2f} deg "
          f"(s0={2/3*LW*phi1:.3f}); Fbar_z={Fbar[2]:.4f}")
    print(f"body vz over the cycle: [{vz.min():+.4f}, {vz.max():+.4f}] "
          f"(wing-tip speed s0*omega = {wing_speed:.3f})")
    print(f"peak |Fz_unsteady - Fz_static| = {np.max(np.abs(Fz_unsteady - Fz_static)):.3f} "
          f"(peak static Fz = {Fz_static.max():.3f})")
    print(f"cycle-mean Fz: unsteady={Fz_unsteady.mean():.4f}, static={Fz_static.mean():.4f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

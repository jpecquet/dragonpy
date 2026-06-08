"""
Directional analog of C_psi: how the pitch pattern aims the cycle-averaged force
RELATIVE TO THE STROKE PLANE.

At the optimal (symmetric) pitch the cycle-averaged force points along the
stroke-plane normal, so hovering there needs a horizontal stroke plane. Asymmetric
pitch (psi0 != 0, or delta0 away from +-90) tilts the force off the normal, into
the stroke plane. With gamma0 = 0 the stroke-plane normal is +z and the in-plane
"forward" (leading-edge) direction is +x, and bilateral symmetry kills Fy, so the
tilt is

    beta(psi0, psi1, delta0) = atan2(Fx, Fz)   (deg, signed; + = toward +x).

This is gamma0-invariant: gamma0 rotates the whole averaged vector rigidly, so
beta is the force direction measured in the stroke-plane frame.

Test plot: beta vs psi0 (the knob that tilts the force off the normal). delta0
and psi1 only set the sign/scale, so their values are not annotated. On a second
axis we overlay C_psi^max(psi0), the best pitch-efficiency coefficient over the
pitch amplitude psi1 -- its optimum sits at the psi0-independent (psi1 ~ 51 deg,
delta0 = 90 deg), so steering the force off-normal via psi0 trades against the
attainable force magnitude.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    Params, REF_AW_OVER_MB, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, cycle_averaged_force, replace,
)
from feasibility_reduction import cpsi_of_pitch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE / "figures"
N_PHASE = 128
DELTA0 = np.radians(90.0)
PSI1 = np.radians(30.0)

# Reference amplitude (matches feasibility_reduction): s0 = 0.25 at Lw = 0.75
# (phi1 = s0 / ((2/3) Lw) = 0.5 rad).
AMP_NOM = dict(Aw_over_mb=REF_AW_OVER_MB, omega_star=REF_OMEGA_STAR,
               phi1=REF_S0 / (STUDY_SPAN_FRAC * REF_LW), Lw=REF_LW)

PSI0_RANGE = np.radians((-90.0, 90.0))


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = Params(gamma0=0.0, delta0=DELTA0, psi1=PSI1,
                  element_span_fracs=STUDY_ELEMENT, **AMP_NOM)

    n = 181
    psi0 = np.linspace(*PSI0_RANGE, n)
    beta = np.empty(n)
    fy_max = 0.0
    for i, p0 in enumerate(psi0):
        F = cycle_averaged_force(replace(base, psi0=p0), N_PHASE)
        beta[i] = np.degrees(np.arctan2(F[0], F[2]))
        fy_max = max(fy_max, abs(float(F[1])))

    # At psi0 = -90 the force points straight down with Fx at numerical -0, so
    # atan2 returns -180; the curve actually approaches it from +178 deg. Flip that
    # degenerate endpoint to +180 so beta stays continuous and monotone.
    if beta[0] < 0.0 < beta[1]:
        beta[0] += 360.0

    # C_psi^max(psi0): max pitch-efficiency over the pitch amplitude psi1. The
    # optimal phase is psi0-independent (delta0 = 90 deg; verified separately), so
    # we fix it and search psi1 only. Evaluated at the reference amplitude.
    psi1_search = np.radians(np.linspace(0.0, 60.0, 61))
    psi0_c = np.linspace(*PSI0_RANGE, 61)
    cpsi_max = np.empty_like(psi0_c)
    psi1_opt = np.empty_like(psi0_c)
    for i, p0 in enumerate(psi0_c):
        vals = np.array([cpsi_of_pitch(p0, p1, DELTA0) for p1 in psi1_search])
        k = int(np.argmax(vals))
        cpsi_max[i], psi1_opt[i] = vals[k], np.degrees(psi1_search[k])

    # Symmetric beta axis so beta = 0 sits at the vertical center of the plot.
    beta_lim = 180.0
    cpsi_lim = 1.5  # symmetric too, so C_psi = 0 aligns with beta = 0 (center)

    fig, ax = plt.subplots(figsize=(4.0, 2.0), constrained_layout=True)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    (l_beta,) = ax.plot(np.degrees(psi0), beta, color="black", lw=1.8,
                        label=r"$\beta$")
    ax.set_xlabel(r"$\psi_0$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    ax.set_xlim(*np.degrees(PSI0_RANGE))
    ax.set_ylim(-beta_lim, beta_lim)
    ax.set_xticks(np.arange(-90, 91, 30))  
    ax.set_yticks(np.arange(-180, 181, 90))  

    ax2 = ax.twinx()
    (l_cpsi,) = ax2.plot(np.degrees(psi0_c), cpsi_max, color="black", lw=1.8,
                         ls=":", label=r"$C_\psi^{\max}$")
    # Label sits over the tick range (top half of the axis), not the axis center.
    ax2.set_ylabel(r"$C_\psi^{\max}$", y=0.75)
    ax2.set_ylim(-cpsi_lim, cpsi_lim)            # zero centered, aligned with beta
    ax2.set_yticks(np.arange(0.0, cpsi_lim + 0.01, 0.5))  # positive ticks only

    ax.legend(handles=[l_beta, l_cpsi], loc="lower left",
              fontsize=style.font_size - 1, frameon=True)

    out = OUT_DIR / "force_direction_test.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    print(f"beta range: [{beta.min():.1f}, {beta.max():.1f}] deg")
    print(f"max |Fy| (bilateral-symmetry check): {fy_max:.2e}")
    print(f"C_psi^max range: [{cpsi_max.min():.3f}, {cpsi_max.max():.3f}] "
          f"(at psi0=0: {cpsi_max[len(psi0_c)//2]:.3f})")
    print(f"optimal psi1 range over psi0: "
          f"[{psi1_opt.min():.0f}, {psi1_opt.max():.0f}] deg (delta0=90 fixed)")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

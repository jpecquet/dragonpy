"""
Directional analog of C_{F^*}: how the pitch pattern aims the cycle-averaged force
RELATIVE TO THE STROKE PLANE.

At the optimal (symmetric) pitch the cycle-averaged force points along the
stroke-plane normal, so hovering there needs a horizontal stroke plane. Asymmetric
pitch (psi0 != 0) tilts the force off the normal, into the stroke plane. The tilt
angle beta is measured from the normal n, positive counterclockwise (x right, z up
at gamma0 = 0), so beta = -gamma for vertical hover at stroke-plane angle gamma.

This script uses the closed form of section 2 (eq:beta_psi0): beta depends on psi0
alone, independent of the pitch amplitude psi1 and phase delta0,

    beta(psi0) = atan2( (C_D,90 - C_D,0) sin 2psi0, 2 C_L,0 cos 2psi0 )

(delta0 > 0 branch). On a second axis we overlay the analytical C_{F^*}^max(psi0),
the best force coefficient over psi1 -- its optimum sits at the psi0-independent
psi1 ~ 51 deg -- so steering the force off-normal via psi0 trades against the
attainable force magnitude.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import CD_0, CD_90, CL_0
from cf_components_contour import cf_components

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
DELTA0 = np.radians(90.0)
PSI0_RANGE = np.radians((-90.0, 90.0))
PSI1_RANGE = np.radians((0.0, 90.0))


def beta_of_psi0(psi0):
    """Closed-form force direction beta(psi0) (eq:beta_psi0), degrees, CCW-positive.

    atan2 wraps at psi0 = +-90 (force straight down); np.unwrap + recentering on
    psi0 = 0 (where beta = 0 exactly) returns the continuous monotone branch that
    runs from -180 to +180. Expects a symmetric psi0 grid with an odd length.
    """
    b = np.arctan2((CD_90 - CD_0) * np.sin(2.0 * psi0),
                   2.0 * CL_0 * np.cos(2.0 * psi0))
    b = np.degrees(np.unwrap(b))
    return b - b[len(b) // 2]


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n = 181  # odd, so psi0 = 0 lands on a grid point (the beta = 0 anchor)
    psi0 = np.linspace(*PSI0_RANGE, n)
    beta = beta_of_psi0(psi0)

    # C_{F^*}^max(psi0): best analytical C_{F^*} = hypot(C_{F^*,n}, C_{F^*,s}) over the
    # pitch amplitude psi1. The optimal phase is psi0-independent (delta0 = 90 deg),
    # so we fix it and search psi1 only.
    psi1_search = np.linspace(*PSI1_RANGE, 91)
    psi0_c = np.linspace(*PSI0_RANGE, 61)
    cfstar_max = np.empty_like(psi0_c)
    psi1_opt = np.empty_like(psi0_c)
    for i, p0 in enumerate(psi0_c):
        vals = np.array([np.hypot(*cf_components(p0, p1, DELTA0)) for p1 in psi1_search])
        k = int(np.argmax(vals))
        cfstar_max[i], psi1_opt[i] = vals[k], np.degrees(psi1_search[k])

    # Symmetric beta axis so beta = 0 sits at the vertical center of the plot.
    beta_lim = 180.0
    cf_lim = 1.5  # symmetric too, so C_{F^*} = 0 aligns with beta = 0 (center)

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
    (l_cf,) = ax2.plot(np.degrees(psi0_c), cfstar_max, color="black", lw=1.8,
                       ls=":", label=r"$C_{F^\ast}^{\max}$")
    # Label sits over the tick range (top half of the axis), not the axis center.
    ax2.set_ylabel(r"$C_{F^\ast}^{\max}$", y=0.75)
    ax2.set_ylim(-cf_lim, cf_lim)                # zero centered, aligned with beta
    ax2.set_yticks(np.arange(0.0, cf_lim + 0.01, 0.5))  # positive ticks only

    ax.legend(handles=[l_beta, l_cf], loc="lower left",
              fontsize=style.font_size - 1, frameon=True)

    out = OUT_DIR / "force_direction_test.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    print(f"beta range: [{beta.min():.1f}, {beta.max():.1f}] deg (analytical, eq:beta_psi0)")
    print(f"beta at psi0 = 45 deg: {beta_of_psi0(np.array([-np.pi/4, 0, np.pi/4]))[-1]:.1f} deg "
          f"(closed form gives 90)")
    print(f"C_F*^max range: [{cfstar_max.min():.3f}, {cfstar_max.max():.3f}] "
          f"(at psi0=0: {cfstar_max[len(psi0_c)//2]:.3f})")
    print(f"optimal psi1 range over psi0: "
          f"[{psi1_opt.min():.0f}, {psi1_opt.max():.0f}] deg (delta0=90 fixed)")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""
Components of the force coefficient C_{F^*}: the stroke-normal (lift) part
C_{F^*,n} and the in-stroke (drag) part C_{F^*,s}, over the (psi0, psi1) pitch
plane at delta0 = 90 deg.

Companion to the C_{F^*} and C_{P^*} pitch contours (feasibility_reduction.py,
pitch_power_contour.py): same two-panel layout and styling. The difference is that
the two components are SIGNED (C_{F^*} = sqrt(C_{F^*,n}^2 + C_{F^*,s}^2)), so a
DIVERGING colormap centered at zero replaces the sequential one. The figure makes
the section-2 point visible: the in-stroke (drag) component C_{F^*,s} vanishes for
a symmetric stroke (psi0 = 0) -- the net force is then purely normal to the stroke
plane -- and grows with |psi0|, the drag contribution an inclined-stroke hover
(beta = -gamma) recruits.

    C_{F^*,n} = -2 C_L,0          <sin 2psi  cos tau |cos tau|>
    C_{F^*,s} = -(C_D,90 - C_D,0) <cos 2psi  cos tau |cos tau|>,
    psi(tau) = psi0 + psi1 sin(tau - delta0),  delta0 = 90 deg.

Output: cf_components.light.png in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import CD_0, CD_90, CL_0

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
CMAP = "RdBu_r"        # diverging: +ve red, -ve blue, white at zero (components are signed)
DELTA0 = np.pi / 2.0   # the optimal pitch phase used throughout section 2

# Parameter ranges (Table 1, extended), matching the C_{F^*}/C_{P^*} contours.
PSI0_RANGE = np.radians((-90.0, 90.0))
PSI1_RANGE = np.radians((0.0, 90.0))

# Wingbeat-average quadrature in the cycle phase tau = omega t.
_TAU = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
_W = np.cos(_TAU) * np.abs(np.cos(_TAU))   # cos tau |cos tau|


def cf_components(psi0, psi1, delta0):
    """Closed-form (C_{F^*,n}, C_{F^*,s}) of section 2 -- pitch only."""
    psi = psi0 + psi1 * np.sin(_TAU - delta0)
    c_n = -2.0 * CL_0 * np.mean(np.sin(2.0 * psi) * _W)
    c_s = -(CD_90 - CD_0) * np.mean(np.cos(2.0 * psi) * _W)
    return c_n, c_s


def grid_psi0_psi1(delta0, n=73):
    """(C_{F^*,n}, C_{F^*,s}) over (psi0, psi1) at fixed delta0; indexed [psi1, psi0]."""
    psi1 = np.linspace(*PSI1_RANGE, n)
    psi0 = np.linspace(*PSI0_RANGE, n)
    cn = np.empty((n, n))
    cs = np.empty((n, n))
    for j, ps1 in enumerate(psi1):
        for i, p0 in enumerate(psi0):
            cn[j, i], cs[j, i] = cf_components(p0, ps1, delta0)
    return psi0, psi1, cn, cs


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    psi0_ax, psi1_ax, cn, cs = grid_psi0_psi1(DELTA0)

    # Shared, symmetric 0.1-spaced levels so the two signed components are on one
    # diverging scale (white = 0) and directly comparable.
    vmax = np.ceil(max(np.abs(cn).max(), np.abs(cs).max()) * 10.0) / 10.0
    levels = np.round(np.arange(-vmax, vmax + 1e-9, 0.1), 2)
    line_kw = dict(levels=levels, colors="black", linewidths=0.3, alpha=0.35)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.4, 3.0), sharey=True, constrained_layout=True)

    P0, P1 = np.meshgrid(np.degrees(psi0_ax), np.degrees(psi1_ax))
    cf = ax1.contourf(P0, P1, cn, levels=levels, cmap=CMAP)
    ax1.contour(P0, P1, cn, **line_kw)
    ax1.set_xlabel(r"$\psi_0$ (deg)")
    ax1.set_ylabel(r"$\psi_1$ (deg)")
    ax1.set_title(r"(a)  $C_{F^\ast,n}$ at $\delta_0 = 90^\circ$",
                  fontsize=style.font_size)

    ax2.contourf(P0, P1, cs, levels=levels, cmap=CMAP)
    ax2.contour(P0, P1, cs, **line_kw)
    ax2.set_xlabel(r"$\psi_0$ (deg)")
    ax2.set_title(r"(b)  $C_{F^\ast,s}$ at $\delta_0 = 90^\circ$",
                  fontsize=style.font_size)

    fig.colorbar(cf, ax=[ax1, ax2], label=r"$C_{F^\ast,n},\ C_{F^\ast,s}$")
    out = OUT_DIR / "cf_components.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")

    # --- report numbers ---
    print(f"C_F*,n range: [{cn.min():.3f}, {cn.max():.3f}] "
          f"(max {cn.max():.3f} near psi0=0)")
    print(f"C_F*,s range: [{cs.min():.3f}, {cs.max():.3f}] "
          f"(antisymmetric in psi0; = 0 at psi0=0)")
    i0 = int(np.argmin(np.abs(psi0_ax)))
    print(f"|C_F*,s| at psi0=0: {np.abs(cs[:, i0]).max():.2e} (should be ~0)")
    print(f"shared symmetric color scale: +-{vmax:.1f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

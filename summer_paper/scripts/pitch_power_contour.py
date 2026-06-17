"""
Power-minimizing hover pitch: the C_{P^*} contour.

Companion to the C_{F^*} pitch-efficiency contour (feasibility_reduction.py,
pitch_efficiency.light.png): same two-panel layout and styling, but for the hover
power coefficient

    C_{P^*} = 2 sqrt(2) <C_D |cos tau|^3> / C_{F^*}^{3/2},

the flapping analog of the steady C_D / C_L^{3/2} (report section 2). It is
evaluated from the section-2 closed form (translating-element idealization), in
which both ingredients are functions of the pitch kinematics alone:

    psi(tau) = psi0 + psi1 sin(tau - delta0),
    C_D(psi) = C_D,0 sin^2 psi + C_D,90 cos^2 psi          (hover drag),
    C_{F^*}  = sqrt(C_{F^*,n}^2 + C_{F^*,s}^2),
        C_{F^*,n} = -2 C_L,0          <sin 2psi  cos tau |cos tau|>,
        C_{F^*,s} = -(C_D,90 - C_D,0) <cos 2psi  cos tau |cos tau|>.

Unlike C_{F^*} (a gain, maximized), C_{P^*} is a COST: the marked optimum is its
MINIMUM, and the colormap is reversed so the cheap region reads dark (as the
high-force region does in the C_{F^*} figure). C_{P^*} diverges as psi1 -> 0
(C_{F^*} -> 0, no net force to support the weight); the color scale is clipped
above and that corner is left to the 'over' color.

The two panels share the psi1 axis:
    (a) C_{P^*}(delta0, psi1) at psi0 = 0
    (b) C_{P^*}(psi0,  psi1) at the power-optimal delta0
`main` also reports the closed form's agreement with the full-arc numerical model
(power_efficiency.closed_form_cost), the analog of the C_{F^*} check.

Output: pitch_power.light.png in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    CD_0, CD_90, CL_0, Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR,
    STUDY_ELEMENT, replace,
)
from power_efficiency import PHI1_REF, closed_form_cost

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
CMAP = "RdPu_r"   # reversed: low cost (favorable) reads dark, as in the C_{F^*} figure
N_PHASE = 128

# Parameter ranges (Table 1), matching the C_{F^*} contour axes.
PSI0_RANGE = np.radians((-90.0, 90.0))
PSI1_RANGE = np.radians((0.0, 90.0))
DELTA0_RANGE = np.radians((-180.0, 180.0))

# Wingbeat-average quadrature in the cycle phase tau = omega t.
_TAU = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
_W = np.cos(_TAU) * np.abs(np.cos(_TAU))   # cos tau |cos tau|  (force weight)
_W3 = np.abs(np.cos(_TAU)) ** 3            # |cos tau|^3        (power weight)


def cfstar_analytic(psi0, psi1, delta0):
    """Closed-form force coefficient C_{F^*}(psi0, psi1, delta0) of section 2."""
    psi = psi0 + psi1 * np.sin(_TAU - delta0)
    c_n = -2.0 * CL_0 * np.mean(np.sin(2.0 * psi) * _W)
    c_s = -(CD_90 - CD_0) * np.mean(np.cos(2.0 * psi) * _W)
    return float(np.hypot(c_n, c_s))


def cpstar_analytic(psi0, psi1, delta0):
    """Closed-form hover power coefficient C_{P^*}(psi0, psi1, delta0).

    C_{P^*} = 2 sqrt(2) <C_D |cos tau|^3> / C_{F^*}^{3/2}; +inf where C_{F^*} -> 0.
    """
    psi = psi0 + psi1 * np.sin(_TAU - delta0)
    Cd = CD_0 * np.sin(psi) ** 2 + CD_90 * np.cos(psi) ** 2
    gP = np.mean(Cd * _W3)
    cf = cfstar_analytic(psi0, psi1, delta0)
    if cf < 1e-6:
        return np.inf
    return 2.0 * np.sqrt(2.0) * gP / cf ** 1.5


def cpstar_numeric(psi0, psi1, delta0):
    """Full-arc numerical C_{P^*} at the reference amplitude (the check reference).

    power_efficiency.closed_form_cost = 2 sqrt(2) <|cos|^3 C_D(alpha)> / C_{F^*}^{3/2}
    with alpha and C_{F^*} from the moving-arc model; the difference from
    cpstar_analytic is exactly the translating-element (fixed stroke tangent)
    idealization.
    """
    p = replace(Params(gamma0=0.0, element_span_fracs=STUDY_ELEMENT),
                psi0=psi0, psi1=psi1, delta0=delta0, phi1=PHI1_REF,
                A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR, Lw=REF_LW)
    return closed_form_cost(p, N_PHASE)


def grid_delta0_psi1(psi0, n=73):
    """C_{P^*} over (delta0, psi1) at fixed psi0; arrays indexed [psi1, delta0].

    Returns the analytical surface (plotted) and the full-arc numerical surface
    (agreement check only).
    """
    psi1 = np.linspace(*PSI1_RANGE, n)
    delta0 = np.linspace(*DELTA0_RANGE, n)
    cp = np.empty((n, n))
    cp_num = np.empty((n, n))
    for j, ps1 in enumerate(psi1):
        for i, d in enumerate(delta0):
            cp[j, i] = cpstar_analytic(psi0, ps1, d)
            cp_num[j, i] = cpstar_numeric(psi0, ps1, d)
    return delta0, psi1, cp, cp_num


def grid_psi0_psi1(delta0, n=73):
    """C_{P^*} over (psi0, psi1) at fixed delta0; arrays indexed [psi1, psi0]."""
    psi1 = np.linspace(*PSI1_RANGE, n)
    psi0 = np.linspace(*PSI0_RANGE, n)
    cp = np.empty((n, n))
    cp_num = np.empty((n, n))
    for j, ps1 in enumerate(psi1):
        for i, p0 in enumerate(psi0):
            cp[j, i] = cpstar_analytic(p0, ps1, delta0)
            cp_num[j, i] = cpstar_numeric(p0, ps1, delta0)
    return psi0, psi1, cp, cp_num


def _rel_agreement(ana, num, psi1_ax):
    """Max/mean |relative| difference over cells with psi1 >= 15 deg (away from the
    C_{F^*} -> 0 divergence, where both blow up and the ratio is ill-conditioned)."""
    keep = psi1_ax >= np.radians(15.0)
    a, b = ana[keep], num[keep]
    good = np.isfinite(a) & np.isfinite(b) & (b > 1e-9)
    rel = np.abs(a[good] - b[good]) / b[good]
    return rel.max(), rel.mean()


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # (a) at psi0 = 0; the power-optimal delta0 read off (a) fixes the (b) slice.
    delta0_ax, psi1_ax, cp_dp, cp_dp_num = grid_delta0_psi1(0.0)
    jmin, imin = np.unravel_index(np.nanargmin(cp_dp), cp_dp.shape)
    psi1_opt, delta0_opt = psi1_ax[jmin], abs(delta0_ax[imin])  # +-90 equal; report +90
    psi0_ax, _, cp_pp, cp_pp_num = grid_psi0_psi1(delta0_opt)
    # The global argmin of panel (b) sits in a large-|psi0|, low-psi1 corner -- a
    # cheap but NON-hover regime (the force is aimed far off the stroke normal,
    # i.e. mostly in-plane: a forward-flight thrust posture, not vertical hover).
    # The hover optimum is the psi0 = 0 column (vertical force at gamma = 0); we
    # mark that, matching the C_{F^*} figure's psi0 = 0 dot.
    j2, i2 = np.unravel_index(np.nanargmin(cp_pp), cp_pp.shape)

    # Color levels: 0.1-spaced from the floor of the global minimum up ~1 unit; the
    # divergent low-psi1 corner spills into the 'over' color (extend="max").
    cmin = min(np.nanmin(cp_dp), np.nanmin(cp_pp))
    lvl_min = np.floor(cmin * 10.0) / 10.0
    levels = np.round(np.arange(lvl_min, lvl_min + 1.0 + 1e-9, 0.1), 2)
    line_kw = dict(levels=levels, colors="black", linewidths=0.3, alpha=0.35)
    star_kw = dict(marker="o", color="white", ms=7, mec="black", mew=0.8,
                   linestyle="none")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.4, 3.0), sharey=True, constrained_layout=True)

    D0, P1a = np.meshgrid(np.degrees(delta0_ax), np.degrees(psi1_ax))
    cf = ax1.contourf(D0, P1a, cp_dp, levels=levels, cmap=CMAP, extend="max")
    ax1.contour(D0, P1a, cp_dp, **line_kw)
    ax1.plot(np.degrees(delta0_opt), np.degrees(psi1_opt), **star_kw)
    ax1.set_xlabel(r"$\delta_0$ (deg)")
    ax1.set_ylabel(r"$\psi_1$ (deg)")
    ax1.set_title(r"(a)  $C_{P^\ast}(\delta_0,\psi_1)$ at $\psi_0=0$",
                  fontsize=style.font_size)

    P0, P1b = np.meshgrid(np.degrees(psi0_ax), np.degrees(psi1_ax))
    ax2.contourf(P0, P1b, cp_pp, levels=levels, cmap=CMAP, extend="max")
    ax2.contour(P0, P1b, cp_pp, **line_kw)
    ax2.plot(0.0, np.degrees(psi1_opt), **star_kw)   # gamma=0 hover optimum (psi0=0)
    ax2.set_xlabel(r"$\psi_0$ (deg)")
    ax2.set_title(
        rf"(b)  $C_{{P^\ast}}(\psi_0,\psi_1)$ at $\delta_0={np.degrees(delta0_opt):.0f}^\circ$",
        fontsize=style.font_size)

    fig.colorbar(cf, ax=[ax1, ax2], label=r"$C_{P^\ast}$", extend="max")
    out = OUT_DIR / "pitch_power.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")

    # --- report numbers ---
    print(f"C_P* minimum {cp_dp[jmin, imin]:.3f} at "
          f"psi1={np.degrees(psi1_opt):.1f} deg, delta0={np.degrees(delta0_opt):.1f} deg "
          f"(psi0=0); analytical")
    print(f"  -- the C_F*-optimal pitch (psi1~51 deg) maximizes force, not efficiency; "
          f"the power optimum sits at higher psi1 (~76 deg), beyond the Table-1 "
          f"morphological range (psi1<=60 deg).")
    band = cp_dp[psi1_ax >= np.radians(15.0)]
    print(f"C_P* over (delta0,psi1) at psi0=0, psi1>=15 deg: "
          f"min {np.nanmin(cp_dp):.3f}, finite max {band[np.isfinite(band)].max():.3f} "
          f"(diverges as C_F* -> 0, e.g. delta0=0)")
    print(f"  panel (b) global min {cp_pp[j2, i2]:.3f} at psi0={np.degrees(psi0_ax[i2]):.0f}, "
          f"psi1={np.degrees(psi1_ax[j2]):.0f} deg is NON-hover (force off-vertical); "
          f"the marked psi0=0 hover optimum is {cp_pp[:, np.argmin(np.abs(psi0_ax))].min():.3f}")

    # --- agreement of the closed form with the full-arc numerical model ---
    mx_dp, mn_dp = _rel_agreement(cp_dp, cp_dp_num, psi1_ax)
    mx_pp, mn_pp = _rel_agreement(cp_pp, cp_pp_num, psi1_ax)
    print(f"analytical vs numerical C_P* over (delta0,psi1) grid (psi1>=15 deg): "
          f"max rel={mx_dp*100:.2f}%, mean rel={mn_dp*100:.2f}%")
    print(f"analytical vs numerical C_P* over (psi0,psi1) grid (psi1>=15 deg):  "
          f"max rel={mx_pp*100:.2f}%, mean rel={mn_pp*100:.2f}%")
    print(f"at optimum: analytical={cp_dp[jmin, imin]:.4f}, "
          f"numerical={cp_dp_num[jmin, imin]:.4f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

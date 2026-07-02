"""Force coefficient and force direction at nonzero advance ratio J.

Section 2's general case: body velocity along the stroke-plane normal,
J = <u*>/(s0* omega*). The instantaneous translating-element force has both
lift and drag in each stroke-plane component,

    C_{F^*,n} =  2 < sqrt(J^2 + cos^2 tau) ( |cos tau| C_L - J C_D ) >
    C_{F^*,s} = -2 < sqrt(J^2 + cos^2 tau) ( sigma J C_L + cos tau C_D ) >,

with sigma = sign(cos tau), alpha = sigma psi + atan(|cos tau|/J), and
psi(tau) = psi0 + psi1 sin(tau - delta0). This is the quadrature form of the
report's eq:cfn-general / eq:cfs-general; J = 0 reduces exactly to the hover
closed form (checked against cf_components_contour.cf_components at import
of nothing extra -- see the printed diagnostics).

Two figures, both at the section operating point delta0 = 90 deg:

    pitch_efficiency_J.light.png -- C_{F^*} over the (psi0, psi1) plane, one
        panel per J in [0, 1]: the panel-(b) slice of pitch_efficiency.light.png
        swept in advance ratio. The white dot tracks the hover optimum
        (psi0 = 0, psi1 = 51 deg) by continuation in J: psi0 = 0 stays a
        critical line by parity (C_{F^*,n} even, C_{F^*,s} odd in psi0), and
        the point slides down in psi1, turning from local maximum to saddle.
    force_direction_J.light.png -- force direction beta(psi0) with one curve
        per J, each at that J's tracked-optimum psi1 (beta is psi1-independent
        only at J = 0). Companion to force_direction_test.light.png, beta only.
    hover_optimum_J.light.png -- the tracked point's psi1 and C_{F^*} as
        functions of J (solid / dotted twin axes, force_direction_test style).

Output: light-mode figures in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cf_components_contour import cf_components
from feasibility import (
    CD_0, CD_90, CL_0, Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0,
    STUDY_ELEMENT, STUDY_SPAN_FRAC, cycle_averaged_force,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
CMAP = "RdPu"
DELTA0 = np.pi / 2.0            # the optimal pitch phase used throughout section 2
PSI1_HOVER = np.radians(51.0)   # hover optimum (section 2), delta0 = 90 deg
J_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

PSI0_RANGE = np.radians((-90.0, 90.0))
PSI1_RANGE = np.radians((0.0, 90.0))

# Wingbeat-average quadrature in the cycle phase tau = omega t. The grid count
# is a multiple of 4 so tau = pi/2, 3pi/2 (where sigma flips) land exactly on
# grid points, where sign(0) = 0 zeroes the direction-ambiguous lift term.
_TAU = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
_COS = np.cos(_TAU)
_SIG = np.sign(_COS)
_ABS = np.abs(_COS)

PHI1_REF = REF_S0 / (STUDY_SPAN_FRAC * REF_LW)


def cf_components_J(psi0, psi1, J, delta0=DELTA0):
    """(C_{F^*,n}, C_{F^*,s}) at advance ratio J -- general case of section 2."""
    psi = psi0 + psi1 * np.sin(_TAU - delta0)
    alpha = _SIG * psi + np.arctan2(_ABS, J)
    cl = CL_0 * np.sin(2.0 * alpha)
    cd = CD_0 * np.cos(alpha) ** 2 + CD_90 * np.sin(alpha) ** 2
    v = np.sqrt(J * J + _COS * _COS)
    c_n = 2.0 * np.mean(v * (_ABS * cl - J * cd))
    c_s = -2.0 * np.mean(v * (_SIG * J * cl + _COS * cd))
    return c_n, c_s


def grid_psi0_psi1(J, n=73):
    """C_{F^*} and C_{F^*,n} over (psi0, psi1) at fixed J, delta0 = 90 deg.

    Arrays indexed [psi1, psi0]. C_{F^*,n} feeds the zero contour separating
    propulsive (+n) from braking (-n) mean force.
    """
    psi1 = np.linspace(*PSI1_RANGE, n)
    psi0 = np.linspace(*PSI0_RANGE, n)
    cf = np.empty((n, n))
    cn = np.empty((n, n))
    for j, ps1 in enumerate(psi1):
        for i, p0 in enumerate(psi0):
            c_n, c_s = cf_components_J(p0, ps1, J)
            cf[j, i] = np.hypot(c_n, c_s)
            cn[j, i] = c_n
    return psi0, psi1, cf, cn


def track_hover_optimum(J, psi1_prev, n=721):
    """Continue the hover-optimal critical point to advance ratio J.

    psi0 = 0 is a critical line at every J (C_{F^*,n} is even and C_{F^*,s} odd
    in psi0, so C_{F^*} is even), so the hover optimum stays on it and only its
    psi1 moves: take the local maximum of C_{F^*}(0, psi1) nearest to the
    previous J's location. Returns (psi1, C_{F^*}, is_max) where is_max tells a
    surviving local maximum from a saddle (curvature across psi0).
    """
    psi1 = np.linspace(*PSI1_RANGE, n)
    f = np.array([np.hypot(*cf_components_J(0.0, p1, J)) for p1 in psi1])
    loc = np.flatnonzero((f[1:-1] >= f[:-2]) & (f[1:-1] >= f[2:])) + 1
    j = loc[np.argmin(np.abs(psi1[loc] - psi1_prev))]
    off_axis = np.hypot(*cf_components_J(np.radians(2.5), psi1[j], J))
    return psi1[j], f[j], off_axis < f[j]


def beta_of_psi0(psi0, J, psi1):
    """Continuous force-direction branch beta(psi0) at advance ratio J, degrees.

    beta = -atan2(C_{F^*,s}, C_{F^*,n}), unwrapped over the psi0 grid and
    anchored at psi0 = 0 where C_{F^*,s} = 0 by parity: beta(0) = 0 while the
    mean force keeps a +n component, jumping to +-180 once the ram drag
    (-2 J Cbar_D I_0) overturns it. Expects a symmetric grid with odd length.
    """
    cns, css = np.empty_like(psi0), np.empty_like(psi0)
    for i, p0 in enumerate(psi0):
        cns[i], css[i] = cf_components_J(p0, psi1, J)
    b = np.degrees(np.unwrap(-np.arctan2(css, cns)))
    mid = len(b) // 2
    b = b - b[mid]
    if cns[mid] <= 0.0:
        # Force along -n at psi0 = 0: anchor beta(0) = +180, on the psi0 > 0
        # branch (which approaches it continuously), and wrap the psi0 < 0
        # branch to [-180, 180) -- the wrap keeps +180, unlike the usual one.
        b = 180.0 - np.mod(-b, 360.0)
    return b, cns[mid]


def beta_numeric(psi0, J, psi1):
    """beta from the full-arc numerical model (agreement check only), degrees."""
    p = Params(A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR,
               phi1=PHI1_REF, Lw=REF_LW, gamma0=0.0,
               v_body=(0.0, 0.0, J * REF_S0 * REF_OMEGA_STAR),
               element_span_fracs=STUDY_ELEMENT,
               psi0=psi0, psi1=psi1, delta0=DELTA0)
    f = cycle_averaged_force(p, 128)
    # The model's x axis is -s of the report convention (its beta is measured
    # positive toward +x, the report's toward -s), hence the sign flip.
    return -np.degrees(np.arctan2(f[0], f[2]))


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- consistency: J = 0 must reduce to the hover closed form ---
    rng = np.random.default_rng(7)
    dmax = 0.0
    for p0, p1 in zip(rng.uniform(*PSI0_RANGE, 50), rng.uniform(*PSI1_RANGE, 50)):
        ref = cf_components(p0, p1, DELTA0)
        new = cf_components_J(p0, p1, 0.0)
        dmax = max(dmax, abs(ref[0] - new[0]), abs(ref[1] - new[1]))
    print(f"J=0 reduction to hover closed form: max|dC| = {dmax:.2e}")

    # ---------------------------------------------------------------------
    # Figure A: C_{F^*}(psi0, psi1) mosaic over J -- pitch_efficiency panel (b)
    # swept in advance ratio.
    grids = [grid_psi0_psi1(J) for J in J_VALUES]
    vmax = np.ceil(max(cf.max() for _, _, cf, _ in grids) * 5.0) / 5.0
    levels = np.round(np.arange(0.0, vmax + 1e-9, 0.2), 2)
    line_kw = dict(levels=levels, colors="black", linewidths=0.3, alpha=0.35)
    star_kw = dict(marker="o", color="white", ms=7, mec="black", mew=0.8,
                   linestyle="none")

    fig_a, axes = plt.subplots(2, 3, figsize=(6.4, 4.6), sharex=True,
                               sharey=True, constrained_layout=True)
    psi1_track = PSI1_HOVER
    for k, (ax, J, (psi0_ax, psi1_ax, cf, cn)) in enumerate(
            zip(axes.flat, J_VALUES, grids)):
        P0, P1 = np.meshgrid(np.degrees(psi0_ax), np.degrees(psi1_ax))
        im = ax.contourf(P0, P1, cf, levels=levels, cmap=CMAP)
        ax.contour(P0, P1, cf, **line_kw)
        # Propulsive/braking boundary: mean force flips from +n to -n. At
        # J = 0 the psi1 = 0 row is identically zero (no pitch oscillation,
        # no mean force), which would smear dotted artifacts along the
        # bottom edge -- mask the degenerate row.
        cn_plot = cn.copy()
        if np.abs(cn_plot[0]).max() < 1e-9:
            cn_plot[0] = np.nan
        ax.contour(P0, P1, cn_plot, levels=[0.0], colors="black",
                   linestyles=":", linewidths=1.0)
        psi1_track, cf_track, is_max = track_hover_optimum(J, psi1_track)
        ax.plot(0.0, np.degrees(psi1_track), **star_kw)
        ax.set_title(rf"({chr(ord('a') + k)})  $J = {J:.1f}$",
                     fontsize=style.font_size)
        if k >= 3:
            ax.set_xlabel(r"$\psi_0$ (deg)")
        if k % 3 == 0:
            ax.set_ylabel(r"$\psi_1$ (deg)")
        print(f"J = {J:.1f}: global C_F* max = {cf.max():.3f}; hover optimum "
              f"continues to psi1 = {np.degrees(psi1_track):.1f} deg, "
              f"C_F* = {cf_track:.3f} "
              f"({'local max' if is_max else 'saddle'})")
    fig_a.colorbar(im, ax=axes, label=r"$C_{F^\ast}$")
    out_a = OUT_DIR / "pitch_efficiency_J.light.png"
    fig_a.savefig(out_a, dpi=300, bbox_inches="tight")

    # ---------------------------------------------------------------------
    # Figure C: the tracked hover optimum's psi1 and C_{F^*} vs J, continued
    # on a fine J grid (each step seeded with the previous psi1).
    J_fine = np.linspace(0.0, 1.0, 101)
    psi1_c = np.empty_like(J_fine)
    cf_c = np.empty_like(J_fine)
    saddle_J = None
    prev = PSI1_HOVER
    for i, J in enumerate(J_fine):
        prev, cf_i, is_max = track_hover_optimum(J, prev)
        psi1_c[i], cf_c[i] = prev, cf_i
        if saddle_J is None and not is_max:
            saddle_J = J
    print(f"tracked optimum: psi1 {np.degrees(psi1_c[0]):.1f} -> "
          f"{np.degrees(psi1_c[-1]):.1f} deg, C_F* {cf_c[0]:.3f} -> "
          f"{cf_c[-1]:.3f}; local max -> saddle at J = {saddle_J:.2f}")

    fig_c, axc = plt.subplots(figsize=(4.0, 2.0), constrained_layout=True)
    (l_p1,) = axc.plot(J_fine, np.degrees(psi1_c), color="black", lw=1.8,
                       label=r"$\psi_1$")
    axc.set_xlabel(r"$J$")
    axc.set_ylabel(r"$\psi_1$ (deg)")
    axc.set_xlim(0.0, 1.0)
    axc.set_ylim(0.0, 60.0)
    axc2 = axc.twinx()
    (l_cf,) = axc2.plot(J_fine, cf_c, color="black", lw=1.8, ls=":",
                        label=r"$C_{F^\ast}$")
    axc2.set_ylabel(r"$C_{F^\ast}$")
    axc2.set_ylim(0.0, 1.5)
    axc.legend(handles=[l_p1, l_cf], loc="lower left",
               fontsize=style.font_size - 1, frameon=True)
    out_c = OUT_DIR / "hover_optimum_J.light.png"
    fig_c.savefig(out_c, dpi=200, bbox_inches="tight")

    # ---------------------------------------------------------------------
    # Figure B: beta(psi0), one curve per J, each at that J's tracked-optimum
    # psi1 (read off the figure-C continuation).
    n = 181  # odd, so psi0 = 0 lands on a grid point (the beta anchor)
    psi0 = np.linspace(*PSI0_RANGE, n)
    curve_colors = plt.cm.inferno(np.linspace(0.12, 0.82, len(J_VALUES)))

    fig_b, ax = plt.subplots(figsize=(4.0, 2.6), constrained_layout=True)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    for J, color in zip(J_VALUES, curve_colors):
        psi1_J = psi1_c[int(np.argmin(np.abs(J_fine - J)))]
        beta, cn0 = beta_of_psi0(psi0, J, psi1_J)
        # Break the line at wrap jumps: close the ending branch with the
        # boundary point's other representation (beta -+ 360, e.g. the J = 1
        # curve's left branch ends at (0, -180) while its right branch starts
        # at (0, +180)), then insert NaN so the branches are not joined.
        x, y = np.degrees(psi0), beta
        for c in reversed(np.flatnonzero(np.abs(np.diff(y)) > 180.0) + 1):
            x = np.insert(x, c, (x[c], np.nan))
            y = np.insert(y, c, (y[c] - np.sign(y[c] - y[c - 1]) * 360.0,
                                 np.nan))
        ax.plot(x, y, color=color, lw=1.6, label=rf"$J = {J:.1f}$")
        # Check away from psi0 = 45 deg, where beta = +-90 by symmetry in any model.
        i30 = np.searchsorted(psi0, np.radians(30.0))
        d_num = beta[i30] - beta_numeric(psi0[i30], J, psi1_J)
        print(f"J = {J:.1f}: psi1 = {np.degrees(psi1_J):.1f} deg, beta range "
              f"[{np.nanmin(beta):.1f}, {np.nanmax(beta):.1f}] deg, "
              f"C_F*_n(psi0=0) = {cn0:+.3f}, "
              f"analytic - numeric at psi0=30deg: {d_num:+.1f} deg")
    ax.set_xlabel(r"$\psi_0$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    ax.set_xlim(*np.degrees(PSI0_RANGE))
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_yticks(np.arange(-180, 181, 90))
    # The upper-left quadrant is empty: every curve is odd-symmetric, negative
    # (or wrapped to negative) for psi0 < 0.
    ax.legend(fontsize=style.font_size - 2, frameon=True, loc="upper left",
              handlelength=1.4, labelspacing=0.25)
    out_b = OUT_DIR / "force_direction_J.light.png"
    fig_b.savefig(out_b, dpi=200, bbox_inches="tight")

    print(f"wrote {out_a.relative_to(REPO_ROOT)}")
    print(f"wrote {out_b.relative_to(REPO_ROOT)}")
    print(f"wrote {out_c.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

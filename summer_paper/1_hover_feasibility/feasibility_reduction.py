"""
Dimensional reduction of the hover-feasibility map.

Section 2 leaves six parameters that set the cycle-averaged force F_a^*:

    Aw_over_mb  inverse wing loading
    s0          tip-excursion amplitude   (s0 = (2/3) Lw phi1, arc length)
    omega_star  nondim wingbeat frequency
    psi0        mean wing pitch
    psi1        wing pitch amplitude
    delta0      wing pitch phase

Numerically (see the CVs printed below) the force factors into an amplitude
"dynamic-pressure" group times a pitch-efficiency coefficient:

    F_a^*  ~  q^*  x  C_psi(psi0, psi1, delta0),
    q^* = (Aw/mb) s0^2 omega^*^2.

So the three amplitude parameters collapse onto a single axis q^*, and the only
genuinely multi-dimensional residual is the pitch-efficiency surface C_psi, which
is in turn dominated by (psi1, delta0) with weak psi0 dependence. With this
normalization C_psi is an effective lift coefficient: its ideal ceiling is the
section value C_L,0 = 1.5 (square-wave translating element).

This script draws that story in two figures:
    reduction_collapse.light.png  -- before/after collapse: F_a^* vs q^* (fans out)
                                     next to F_a^* vs q^* C_psi (lands on y=x), all
                                     points random in every parameter.
    pitch_efficiency.light.png    -- C_psi sliced two ways, sharing the psi1 axis:
                                     C_psi(delta0, psi1) at psi0=0 and
                                     C_psi(psi0, psi1) at the optimal delta0.

Output: light-mode figures in summer_paper/1_hover_feasibility/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from feasibility import (
    Params, STUDY_ELEMENT, STUDY_SPAN_FRAC, WEIGHT, avg_force_magnitude, replace,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE / "figures"
CMAP = "RdPu"
# Inferno trimmed at both ends: drop the near-black low end and the pale-yellow
# high end (which washes out against the white page).
SCATTER_CMAP = LinearSegmentedColormap.from_list(
    "inferno_trim", plt.cm.inferno(np.linspace(0.12, 0.82, 256)))
N_PHASE = 128
SPAN_FRAC_REF = STUDY_SPAN_FRAC

# Parameter ranges (Table 1).
AW_RANGE = (0.05, 0.25)
OMEGA_RANGE = (8.0, 20.0)
PHI1_RANGE = np.radians((2.0, 60.0))
LW_RANGE = (0.65, 0.85)
PSI0_RANGE = np.radians((-60.0, 60.0))
PSI1_RANGE = np.radians((0.0, 60.0))
DELTA0_RANGE = np.radians((-180.0, 180.0))

# Fixed REFERENCE amplitude at which C_psi = F_a^*/q^* is read off, making C_psi a
# function of pitch only. With the arc-length s0, F_a^*/q^* is amplitude-independent
# to ~2%, so C_psi here differs from a point's own F_a^*/q^* only by that small
# residual -- the genuine (non-tautological) width of the panel-(b) collapse. The
# reference excursion is s0 = 0.3, realized at Lw = 0.75, which fixes
# phi1 = s0 / ((2/3) Lw) = 0.6 rad (~34 deg).
S0_REF = 0.3
LW_REF = 0.75
AMP_NOM = dict(Aw_over_mb=0.15, omega_star=14.0,
               phi1=S0_REF / (SPAN_FRAC_REF * LW_REF), Lw=LW_REF)


def s0_of(Lw, phi1):
    # Arc-length (angular) excursion of the 2/3-span station, matching the report
    # definition s0 = (2/3) Lw phi1. The aerodynamic velocity scales with the
    # sweep ANGLE phi1, not sin(phi1); using phi1 here removes the
    # (phi1/sin phi1)^2 amplitude artifact, so F_a*/q* depends on pitch alone.
    return SPAN_FRAC_REF * Lw * phi1


def q_star(p):
    return p.Aw_over_mb * s0_of(p.Lw, p.phi1) ** 2 * p.omega_star ** 2


def cpsi_of_pitch(psi0, psi1, delta0):
    """Pitch-efficiency coefficient C_psi = F_a^*/q^*, at the nominal amplitude."""
    p = replace(Params(gamma0=0.0, element_span_fracs=STUDY_ELEMENT),
                psi0=psi0, psi1=psi1, delta0=delta0, **AMP_NOM)
    return avg_force_magnitude(p, N_PHASE) / q_star(p)


# ---------------------------------------------------------------------------
# Figure B data: pitch-efficiency C_psi sliced two ways through (psi0, psi1, delta0).

# 73 points over a 360 deg span gives a 5 deg step, so the delta0 grid lands
# exactly on +-90 and 0. C_psi is symmetric about delta0 = +-90 (machine
# precision); an off-grid step would put the sampled optimum a few degrees off
# 90 and make the filled contours look lopsided -- a sampling artifact, not
# physics.
def grid_delta0_psi1(psi0, n=73):
    """C_psi over (delta0, psi1) at fixed psi0; cpsi indexed [psi1, delta0]."""
    psi1 = np.linspace(*PSI1_RANGE, n)
    delta0 = np.linspace(*DELTA0_RANGE, n)
    cpsi = np.empty((n, n))
    for j, ps1 in enumerate(psi1):
        for i, d in enumerate(delta0):
            cpsi[j, i] = cpsi_of_pitch(psi0, ps1, d)
    return delta0, psi1, cpsi


def grid_psi0_psi1(delta0, n=73):
    """C_psi over (psi0, psi1) at fixed delta0; cpsi indexed [psi1, psi0]."""
    psi1 = np.linspace(*PSI1_RANGE, n)
    psi0 = np.linspace(*PSI0_RANGE, n)
    cpsi = np.empty((n, n))
    for j, ps1 in enumerate(psi1):
        for i, p0 in enumerate(psi0):
            cpsi[j, i] = cpsi_of_pitch(p0, ps1, delta0)
    return psi0, psi1, cpsi


# ---------------------------------------------------------------------------

def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figures inserted at native size)
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Single blade element at 2/3 span (overestimates force; see feasibility.py).
    base = Params(gamma0=0.0, element_span_fracs=STUDY_ELEMENT)
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------------
    # Figure A: variable separation as a before/after collapse.
    #
    # No operating point is privileged: every point is drawn at random in ALL
    # six parameters (psi0 included). For each point we form
    #   q*     amplitude group, a function of (Aw, s0, omega) only;
    #   C_psi  pitch efficiency, MEASURED AT A FIXED REFERENCE AMPLITUDE so it is a
    #          function of (psi0, psi1, delta0) only;
    #   F_a*   the actual cycle-averaged force at the point's OWN amplitude.
    # (a) F_a* vs q*      -- points fan out: amplitude alone does not set the force.
    # (b) F_a* vs q*C_psi -- points fall on y=x: force separates as q* x C_psi(pitch).
    # Because C_psi is taken at the reference amplitude and still predicts F at each
    # point's different amplitude, the collapse is a genuine test of separability.
    n_pts = 1500
    Aw = rng.uniform(*AW_RANGE, n_pts)
    om = rng.uniform(*OMEGA_RANGE, n_pts)
    ph = rng.uniform(*PHI1_RANGE, n_pts)
    lw = rng.uniform(*LW_RANGE, n_pts)
    p0v = rng.uniform(*PSI0_RANGE, n_pts)
    p1v = rng.uniform(*PSI1_RANGE, n_pts)
    d0v = rng.uniform(*DELTA0_RANGE, n_pts)
    qs = np.empty(n_pts)
    Fs = np.empty(n_pts)
    cpsi = np.empty(n_pts)
    for k in range(n_pts):
        p = replace(base, Aw_over_mb=Aw[k], omega_star=om[k], phi1=ph[k], Lw=lw[k],
                    psi0=p0v[k], psi1=p1v[k], delta0=d0v[k])
        qs[k] = q_star(p)
        Fs[k] = avg_force_magnitude(p, N_PHASE)
        cpsi[k] = cpsi_of_pitch(p0v[k], p1v[k], d0v[k])  # at the fixed reference amplitude
    pred = qs * cpsi
    resid = Fs / np.where(pred > 1e-9, pred, np.nan)
    cv_sep = float(np.nanstd(resid) / np.nanmean(resid))

    # The high-q* tail is sparse; show the dense body of the cloud.
    mask = qs <= 5.0
    qs, Fs, cpsi, pred = qs[mask], Fs[mask], cpsi[mask], pred[mask]

    # Width kept under the 6.5in report text column so it inserts at native size.
    fig_a, (axL, axR) = plt.subplots(1, 2, figsize=(6.3, 2.95), sharey=True,
                                     constrained_layout=True)
    sckw = dict(c=cpsi, s=7, cmap=SCATTER_CMAP, alpha=0.55, edgecolors="none",
                vmin=0.0, vmax=cpsi.max())
    axL.scatter(qs, Fs, **sckw)
    axL.set_xlabel(r"$q^\ast = (A_w^\ast/m_b^\ast)\, s_0^2\, \omega^{\ast 2}$")
    axL.set_ylabel(r"$F_a^\ast$")
    axL.set_xlim(left=0.0)
    axL.set_ylim(0, None)
    axL.set_title(r"(a)", fontsize=style.font_size)

    sc = axR.scatter(pred, Fs, **sckw)
    lim = float(max(pred.max(), Fs.max()))
    axR.plot([0, lim], [0, lim], color="black", lw=1.2, zorder=0)
    axR.set_xlabel(r"$q^\ast \, C_\psi(\psi_0,\psi_1,\delta_0)$")
    axR.set_xlim(left=0.0)
    axR.set_title(r"(b)", fontsize=style.font_size)

    fig_a.colorbar(sc, ax=[axL, axR], label=r"$F_a^\ast/q^\ast$")
    out_a = OUT_DIR / "reduction_collapse.light.png"
    fig_a.savefig(out_a, dpi=300, bbox_inches="tight")

    # ---------------------------------------------------------------------
    # Figure B: pitch-efficiency C_psi sliced two ways, sharing the psi1 axis.
    #   (a) C_psi(delta0, psi1) at psi0 = 0       -- pitch-timing lobes
    #   (b) C_psi(psi0,  psi1) at delta0 = optimum -- weak psi0 dependence
    # The optimum (psi1, delta0) is read off panel (a) and fixes the delta0 slice of
    # panel (b); both panels share one color scale and a single colorbar.
    delta0_ax, psi1_ax, cpsi_dp = grid_delta0_psi1(0.0)
    jmax, imax = np.unravel_index(np.argmax(cpsi_dp), cpsi_dp.shape)
    # +-90 are equal maxima (symmetry); report the +90 one to match the text.
    psi1_opt, delta0_opt = psi1_ax[jmax], abs(delta0_ax[imax])
    psi0_ax, _, cpsi_pp = grid_psi0_psi1(delta0_opt)

    # 0.1-spaced bins up to the section ceiling C_L,0 = 1.5 (the single 2/3-span
    # element pushes the peak C_psi to ~1.43, near that limit).
    levels = np.round(np.arange(0.0, 1.6, 0.1), 2)  # 0.0 .. 1.5, step 0.1
    line_kw = dict(levels=levels, colors="black", linewidths=0.3, alpha=0.35)
    star_kw = dict(marker="o", color="white", ms=7, mec="black", mew=0.8,
                   linestyle="none")

    fig_b, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.4, 3.0), sharey=True, constrained_layout=True)

    D0, P1a = np.meshgrid(np.degrees(delta0_ax), np.degrees(psi1_ax))
    cf = ax1.contourf(D0, P1a, cpsi_dp, levels=levels, cmap=CMAP)
    ax1.contour(D0, P1a, cpsi_dp, **line_kw)
    ax1.plot(np.degrees(delta0_opt), np.degrees(psi1_opt), **star_kw)
    ax1.set_xlabel(r"$\delta_0$ (deg)")
    ax1.set_ylabel(r"$\psi_1$ (deg)")
    ax1.set_title(r"(a)  $C_\psi(\delta_0,\psi_1)$ at $\psi_0=0$",
                  fontsize=style.font_size)

    P0, P1b = np.meshgrid(np.degrees(psi0_ax), np.degrees(psi1_ax))
    ax2.contourf(P0, P1b, cpsi_pp, levels=levels, cmap=CMAP)
    ax2.contour(P0, P1b, cpsi_pp, **line_kw)
    ax2.plot(0.0, np.degrees(psi1_opt), **star_kw)
    ax2.set_xlabel(r"$\psi_0$ (deg)")
    ax2.set_title(
        rf"(b)  $C_\psi(\psi_0,\psi_1)$ at $\delta_0={np.degrees(delta0_opt):.0f}^\circ$",
        fontsize=style.font_size)

    fig_b.colorbar(cf, ax=[ax1, ax2],
                   label=r"$C_\psi = (F_a^\ast/q^\ast)|_{\mathrm{ref}}$")
    out_b = OUT_DIR / "pitch_efficiency.light.png"
    fig_b.savefig(out_b, dpi=300, bbox_inches="tight")

    # --- report numbers ---
    print(f"separation collapse over {n_pts} random 6D points: "
          f"F_a*/(q* C_psi) CV = {cv_sep*100:.1f}%")
    print(f"C_psi range over (delta0,psi1) at psi0=0: "
          f"[{cpsi_dp.min():.3f}, {cpsi_dp.max():.3f}]")
    print(f"C_psi optimum at psi1={np.degrees(psi1_opt):.1f} deg, "
          f"delta0={np.degrees(delta0_opt):.1f} deg")
    print(f"C_psi range over (psi0,psi1) at delta0={np.degrees(delta0_opt):.0f} deg: "
          f"[{cpsi_pp.min():.3f}, {cpsi_pp.max():.3f}]")
    print(f"wrote {out_a.relative_to(REPO_ROOT)}")
    print(f"wrote {out_b.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

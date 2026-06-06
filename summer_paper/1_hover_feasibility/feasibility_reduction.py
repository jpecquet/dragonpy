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
    pitch_efficiency.light.png    -- C_psi(psi1, delta0) heatmap + C_psi vs psi0
                                     marginal.

Output: light-mode figures in summer_paper/1_hover_feasibility/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from feasibility import Params, WEIGHT, avg_force_magnitude, replace

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
SPAN_FRAC_REF = 2.0 / 3.0

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
# reference excursion is s0 = 0.5 (mid-range), realized at Lw = 0.75, which fixes
# phi1 = s0 / ((2/3) Lw) = 1.0 rad (~57 deg).
S0_REF = 0.5
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
    p = replace(Params(gamma0=0.0), psi0=psi0, psi1=psi1, delta0=delta0, **AMP_NOM)
    return avg_force_magnitude(p, N_PHASE) / q_star(p)


# ---------------------------------------------------------------------------
# Figure B data: pitch-efficiency surface C_psi(psi1, delta0) and the psi0 marginal.

def pitch_grid(psi0, n=56):
    psi1 = np.linspace(*PSI1_RANGE, n)
    delta0 = np.linspace(*DELTA0_RANGE, n)
    cpsi = np.empty((n, n))
    for j, d in enumerate(delta0):
        for i, ps1 in enumerate(psi1):
            cpsi[j, i] = cpsi_of_pitch(psi0, ps1, d)
    return psi1, delta0, cpsi


# ---------------------------------------------------------------------------

def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figures inserted at native size)
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = Params(gamma0=0.0)
    rng = np.random.default_rng(7)

    # --- pitch-efficiency surface at psi0 = 0 (drives both figures) ---
    psi1, delta0, cpsi0 = pitch_grid(0.0)
    # Locate the optimum (max C_psi) for the psi0 marginal line.
    jmax, imax = np.unravel_index(np.argmax(cpsi0), cpsi0.shape)
    psi1_opt, delta0_opt = psi1[imax], delta0[jmax]

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
    # Figure B: pitch-efficiency surface C_psi(psi1, delta0) + C_psi-vs-psi0 marginal.
    fig_b, (axh, axm) = plt.subplots(
        1, 2, figsize=(6.4, 3.0), gridspec_kw=dict(width_ratios=[3.0, 1.3]))

    D1, D0 = np.meshgrid(np.degrees(psi1), np.degrees(delta0))
    pcm = axh.pcolormesh(D1, D0, cpsi0, shading="auto", cmap=CMAP)
    fig_b.colorbar(pcm, ax=axh, label=r"$C_\psi = (F_a^\ast/q^\ast)|_{\mathrm{ref}}$")
    axh.plot(np.degrees(psi1_opt), np.degrees(delta0_opt), marker="*",
             color="black", ms=12, mfc="white", mew=1.2)
    axh.set_xlabel(r"$\psi_1$ (deg)")
    axh.set_ylabel(r"$\delta_0$ (deg)")
    axh.set_title(r"(a)  $C_\psi(\psi_1,\delta_0)$ at $\psi_0=0$",
                  fontsize=style.font_size)

    # Marginal: C_psi vs psi0 at the (psi1, delta0) optimum -- shows weak psi0.
    psi0_line = np.linspace(*PSI0_RANGE, 41)
    cpsi_line = np.array([cpsi_of_pitch(p0, psi1_opt, delta0_opt) for p0 in psi0_line])
    axm.plot(np.degrees(psi0_line), cpsi_line, color="black", lw=1.8)
    axm.set_xlabel(r"$\psi_0$ (deg)")
    axm.set_ylabel(r"$C_\psi$ at optimum")
    axm.set_ylim(0, None)
    axm.set_title(r"(b)  weak $\psi_0$", fontsize=style.font_size)

    fig_b.tight_layout()
    out_b = OUT_DIR / "pitch_efficiency.light.png"
    fig_b.savefig(out_b, dpi=300, bbox_inches="tight")

    # --- report numbers ---
    print(f"separation collapse over {n_pts} random 6D points: "
          f"F_a*/(q* C_psi) CV = {cv_sep*100:.1f}%")
    print(f"C_psi range over (psi1,delta0) at psi0=0: "
          f"[{cpsi0.min():.3f}, {cpsi0.max():.3f}]")
    print(f"C_psi optimum at psi1={np.degrees(psi1_opt):.1f} deg, "
          f"delta0={np.degrees(delta0_opt):.1f} deg")
    print(f"C_psi vs psi0 at optimum: [{cpsi_line.min():.3f}, {cpsi_line.max():.3f}] "
          f"(swing {cpsi_line.max() / cpsi_line.min():.2f}x)")
    print(f"wrote {out_a.relative_to(REPO_ROOT)}")
    print(f"wrote {out_b.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

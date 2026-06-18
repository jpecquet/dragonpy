"""C_psi over the body-velocity plane for the maneuvering equilibrium study.

Velocity is parameterized in the stroke-plane frame (gamma0-invariant, see
feasibility_velocity.py): advance ratio J = U*/(s0* omega*) as the radius and
the angle chi from the stroke-plane normal. The (J, chi) plane is then a disk,
and each scalar of interest becomes one polar map. Reconnaissance
(recon_velocity_cpsi.py) established the structure that makes this enough:

  - C(J, chi) is exactly symmetric under chi -> -chi at psi0 = 0, so we compute
    a half-disk and mirror it;
  - the optimal pitch phase stays pinned at delta0 = 90 deg over the whole disk,
    so only psi1 retunes with velocity.

Panels:
  (a) C = F_a^*/q^* at the hover-optimal pitch (psi0 = 0, psi1 = 51 deg,
      delta0 = 90 deg) -- how the hover wingbeat performs at velocity;
  (b) C^max with psi1 re-optimized at each (J, chi), delta0 = 90 deg;
  (c) the re-optimized psi1^* map.

Output: light-mode figure in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, N_WINGS, avg_force_magnitude,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
N_PHASE = 128
CMAP = "RdPu"
PSI1_CMAP = "BuPu"

PHI1_REF = REF_S0 / (STUDY_SPAN_FRAC * REF_LW)
Q_REF = (REF_A_OVER_M / N_WINGS) * (REF_S0 * REF_OMEGA_STAR) ** 2
PSI1_HOVER = np.radians(51.0)   # hover optimum (section 2), delta0 = 90 deg
DELTA0 = np.pi / 2

J_GRID = np.linspace(0.0, 0.8, 11)
CHI_HALF = np.radians(np.linspace(0.0, 180.0, 19))   # mirrored for display
PSI1_SEARCH = np.radians(np.arange(0.0, 61.0, 2.0))  # lift-branch search (see note)


def C_of(J, chi, psi1):
    U = J * REF_S0 * REF_OMEGA_STAR
    vb = (U * np.sin(chi), 0.0, U * np.cos(chi))
    p = Params(A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR,
               phi1=PHI1_REF, Lw=REF_LW, gamma0=0.0, v_body=vb,
               element_span_fracs=STUDY_ELEMENT,
               psi0=0.0, psi1=psi1, delta0=DELTA0)
    return avg_force_magnitude(p, N_PHASE) / Q_REF


def best_psi1(J, chi):
    """Line search over psi1 with parabolic refinement of the optimum."""
    C = np.array([C_of(J, chi, p1) for p1 in PSI1_SEARCH])
    i = int(np.argmax(C))
    if 0 < i < len(C) - 1:
        a, b, c = C[i - 1], C[i], C[i + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c)
        p1 = PSI1_SEARCH[i] + shift * (PSI1_SEARCH[1] - PSI1_SEARCH[0])
        return p1, C_of(J, chi, p1)
    return PSI1_SEARCH[i], C[i]


def half_disk_maps():
    nJ, nC = len(J_GRID), len(CHI_HALF)
    C_nom = np.empty((nJ, nC))
    C_max = np.empty((nJ, nC))
    psi1_opt = np.empty((nJ, nC))
    for i, J in enumerate(J_GRID):
        if J == 0.0:                       # center: chi is degenerate
            C_nom[i, :] = C_of(0.0, 0.0, PSI1_HOVER)
            p1, cm = best_psi1(0.0, 0.0)
            psi1_opt[i, :], C_max[i, :] = p1, cm
            continue
        for j, chi in enumerate(CHI_HALF):
            C_nom[i, j] = C_of(J, chi, PSI1_HOVER)
            psi1_opt[i, j], C_max[i, j] = best_psi1(J, chi)
    return C_nom, C_max, psi1_opt


def mirror(chi_half, Z):
    """Reflect a half-disk field (J, chi in [0, pi]) onto chi in [-pi, pi]."""
    chi_full = np.concatenate([-chi_half[:0:-1], chi_half])
    Z_full = np.concatenate([Z[:, :0:-1], Z], axis=1)
    return chi_full, Z_full


def setup_polar(ax, style):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels([r"$0^\circ$", r"$45^\circ$", r"$90^\circ$",
                        r"$135^\circ$", r"$180^\circ$", r"$-135^\circ$",
                        r"$-90^\circ$", r"$-45^\circ$"],
                       fontsize=style.font_size - 3)
    ax.set_rticks([0.4, 0.8])
    ax.set_yticklabels(["0.4", "0.8"], fontsize=style.font_size - 3)
    ax.set_rlabel_position(247.5)
    ax.grid(lw=0.4, alpha=0.5)


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    C_nom, C_max, psi1_opt = half_disk_maps()
    chi_full, C_nom_f = mirror(CHI_HALF, C_nom)
    _, C_max_f = mirror(CHI_HALF, C_max)
    _, psi1_f = mirror(CHI_HALF, np.degrees(psi1_opt))

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9),
                             subplot_kw=dict(projection="polar"),
                             constrained_layout=True)
    TH, R = np.meshgrid(chi_full, J_GRID)

    c_levels = np.arange(0.0, np.ceil(C_max_f.max() * 4) / 4 + 1e-9, 0.25)
    for ax, Z, title in [(axes[0], C_nom_f, "(a)"), (axes[1], C_max_f, "(b)")]:
        cf = ax.contourf(TH, R, Z, levels=c_levels, cmap=CMAP)
        ax.contour(TH, R, Z, levels=[1.0], colors="k", linewidths=0.7,
                   linestyles="--")
        setup_polar(ax, style)
        ax.set_title(title, fontsize=style.font_size)
    cb = fig.colorbar(cf, ax=axes[:2], shrink=0.85, pad=0.02, aspect=28)
    cb.set_label(r"$C_\psi$", fontsize=style.font_size - 1)
    cb.set_ticks(np.arange(0.0, c_levels[-1] + 1e-9, 1.0))
    cb.ax.tick_params(labelsize=style.font_size - 3)

    cf3 = axes[2].contourf(TH, R, psi1_f, levels=np.arange(0.0, 62.5, 2.5),
                           cmap=PSI1_CMAP)
    setup_polar(axes[2], style)
    axes[2].set_title("(c)", fontsize=style.font_size)
    cb3 = fig.colorbar(cf3, ax=axes[2], shrink=0.85, pad=0.02, aspect=28)
    cb3.set_label(r"$\psi_1^{\mathrm{opt}}$ (deg)", fontsize=style.font_size - 1)
    cb3.set_ticks(np.arange(0.0, 61.0, 10.0))
    cb3.ax.tick_params(labelsize=style.font_size - 3)

    out = OUT_DIR / "velocity_cpsi_maps.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")

    # --- report numbers ---
    iJ = {J: k for k, J in enumerate(J_GRID)}
    jC = {c: k for k, c in enumerate(np.degrees(CHI_HALF).round())}
    print("hover-optimal pitch at J=0:  C =", f"{C_nom[0, 0]:.3f}")
    for J in (0.4, 0.8):
        for chi in (0.0, 90.0, 180.0):
            i, j = iJ[J], jC[chi]
            print(f"  J={J}, chi={chi:5.1f}:  C_nom={C_nom[i, j]:.3f}  "
                  f"C_max={C_max[i, j]:.3f}  "
                  f"psi1*={np.degrees(psi1_opt[i, j]):.1f} deg")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

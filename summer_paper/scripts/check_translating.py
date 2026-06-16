"""
Additional check: reproduce a translating-element (small-angle, no-rotation)
reference result for the hover-feasibility boundary.

The reference model treats each blade element as purely translating, i.e. the
small-angle limit where the station excursion ~ (span fraction) * L_w * phi1. We
hold a fixed operating point and map F_a^* in the (A*/m_b*, omega*) plane
(total inverse wing loading; the model uses the per-wing A*/m_b* / N_w).

Fixed point:
    s_0* = 1.5      2/3-station excursion amplitude  (= (2/3) L_w phi1, arc length)
    phi1 = 10 deg   flapping amplitude (small angle)
    => L_w = 1.5 / ((2/3) phi1) ~ 12.9    (chosen so the 2/3 station hits s_0*)
    psi0 = 0 deg    mean pitch
    psi1 = 45 deg   pitch amplitude
    delta0 = 90 deg pitch phase (flip at stroke reversal)

Axes:
    A*/m_b*    total, [0.0075, 0.035]  (per-wing = / 4 = [0.001875, 0.00875])
    omega*     [5, 17]

Output: light-mode figure in summer_paper/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import CL_0, Params, WEIGHT, avg_force_magnitude, replace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
CMAP = "RdPu"
BOUNDARY_COLOR = "black"
N_PHASE = 128

# Fixed operating point. s_0* = 1.5 is, for every model, the displacement of the
# representative span station at SPAN_FRAC_REF = 2/3: the analytic form moves
# uniformly at s_0*, so it IS that station, while the blade-element wings are
# scaled so their 2/3 station has s_0* = 1.5 (tip displacement (3/2) s_0*, L_w
# larger by 3/2). This makes the single-element-at-2/3 vs analytic comparison
# apples-to-apples.
PHI1 = np.radians(10.0)
S0_STAR = 1.5
SPAN_FRAC_REF = 2.0 / 3.0
# Arc-length excursion s0 = (2/3) Lw phi1 (report definition). With this, the
# 2/3-station velocity amplitude is exactly s0*omega, matching the translating
# closed form (the sin(phi1) form left a ~0.5% velocity mismatch).
LW_BE = S0_STAR / (SPAN_FRAC_REF * PHI1)  # blade element: 2/3 station hits s_0*

BASE = Params(
    Lw=LW_BE,
    phi1=PHI1,
    psi0=0.0,
    psi1=np.radians(45.0),
    delta0=np.pi / 2,
    gamma0=0.0,
)


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figures inserted at native size)
    apply_matplotlib_style(style)

    n = 120
    aero = np.linspace(0.0075, 0.035, n)            # total inverse wing loading A*/m*
    omega = np.linspace(5.0, 17.0, n)

    def grid(profile, **overrides):
        Z = np.empty((len(omega), len(aero)))
        for j, w in enumerate(omega):
            for i, a in enumerate(aero):
                Z[j, i] = avg_force_magnitude(
                    replace(BASE, A_over_m=a, omega_star=w,
                            pitch_profile=profile, **overrides),
                    n_phase=N_PHASE)
        return Z

    # Blade-element wings use LW_BE so their 2/3 span station carries s_0*.
    Z_sin = grid("sinusoid", Lw=LW_BE)                                    # 8 elements
    Z_n1 = grid("sinusoid", Lw=LW_BE, element_span_fracs=(SPAN_FRAC_REF,))  # 1 elem @ 2/3 span

    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    # x-axis is the total inverse wing loading A*/m_b*, to match the plot we
    # compare against; the model divides it by N_w internally for the per-wing force.
    X, Y = np.meshgrid(aero, omega)

    # Simplified closed form: whole wing translating at tip speed s_0* omega*,
    #   F_a* = (C_L0 A*/4 m_b*) s_0*^2 omega*^2,  A* = total wing area (= X here).
    # No blade-element span integral, so it overestimates (<v^2> uses the tip
    # speed everywhere instead of the r^2-weighted span mean).
    Z_simp = (CL_0 * X / 4.0) * S0_STAR ** 2 * Y ** 2
    # Heatmap from the 1-element field; F_a^* = 1 boundaries overlaid.
    pcm = ax.pcolormesh(X, Y, Z_n1, shading="auto", cmap=CMAP)
    fig.colorbar(pcm, ax=ax, label=r"$F_a^*$ (1 element)")

    from matplotlib.lines import Line2D
    handles = []

    def add_contour(Z, color, ls, lw=2.0, label=None):
        if Z.min() < WEIGHT < Z.max():
            ax.contour(X, Y, Z, levels=[WEIGHT], colors=[color],
                       linewidths=lw, linestyles=ls)
            if label is not None:
                handles.append(Line2D([], [], color=color, lw=lw, ls=ls, label=label))

    add_contour(Z_simp, BOUNDARY_COLOR, "--", 2.0, label="simplified")
    add_contour(Z_n1, BOUNDARY_COLOR, "-", 2.0, label="1 element")
    add_contour(Z_sin, BOUNDARY_COLOR, ":", 2.0, label="8 element")

    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True,
                  fontsize=style.font_size - 1, title=r"$F_a^* = 1$ contour")

    ax.set_xlabel(r"$A^\ast/m^\ast$")
    ax.set_ylabel(r"$\omega^\ast$")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "translating_check.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"L_w blade elem = {LW_BE:.4f} "
          f"(2/3-span s_0* = {SPAN_FRAC_REF * LW_BE * PHI1:.4f})")
    print(f"F_a^* range (sinusoid): [{Z_sin.min():.4f}, {Z_sin.max():.4f}]")
    print(f"F_a^* range (analytic): [{Z_simp.min():.4f}, {Z_simp.max():.4f}]")
    print(f"F_a^* max  8el / 1el(2/3) / analytic: "
          f"{Z_sin.max():.4f} / {Z_n1.max():.4f} / {Z_simp.max():.4f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

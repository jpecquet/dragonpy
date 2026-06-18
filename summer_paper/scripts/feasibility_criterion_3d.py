"""
The hover-feasibility criterion as a 3-D surface.

Section 2 reduces feasibility to a single condition in the three amplitude
groups: hover needs the cycle-averaged force to carry the weight,

    F_a* = q* C_F* = 1,   q* = (1/4) (A*/m*) (s0* omega*)^2,

and the most generous pitch uses the maximum force coefficient C_F*^max (~1.43
for the single 2/3-span element of section 2), which sets the smallest excursion
that can hover. The boundary q* C_F*^max = 1 is a 2-D surface in
(omega*, A*/m*, s0*) space; the hover-capable region is ABOVE it (larger
excursion -> more force). Solving for the vertical axis,

    s0*  =  2 / (omega* sqrt(A*/m* C_F*^max)).

We draw that surface and shade the feasible region above it as a translucent
volume.

Output: light-mode figure in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402
from cf_components_contour import cf_components  # noqa: E402

OUT_DIR = HERE.parent / "figures"

# Parameter ranges (Table 1) for the two horizontal axes.
OMEGA_RANGE = (8.0, 20.0)
AW_RANGE = (0.2, 1.0)   # total inverse wing loading A*/m* (Table 1)
Z_TOP = 0.6  # top of the plotted box / feasible cloud

# Maximum force coefficient over the pitch parameters (psi0 = 0, delta0 = 90 deg,
# psi1 optimized); the single 2/3-span element of section 2 gives ~1.43. The
# hover criterion q* C_F*^max = 1 uses the best pitch, i.e. the smallest excursion
# that can carry the weight.
_PSI1 = np.linspace(0.0, np.pi / 2.0, 181)
CF_MAX = max(float(np.hypot(*cf_components(0.0, p1, np.pi / 2.0))) for p1 in _PSI1)


def s0_criterion(omega, aw):
    """s0* on the hover surface q* C_F*^max = 1: s0* = 2/(omega* sqrt(A*/m* C_F*^max))."""
    return 2.0 / (omega * np.sqrt(aw * CF_MAX))


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pink = plt.cm.RdPu(0.62)

    # Criterion surface s0* = 2/(omega* sqrt(A*/m*)) on q* = 1.
    n = 80
    omega = np.linspace(*OMEGA_RANGE, n)
    aw = np.linspace(*AW_RANGE, n)
    W, A = np.meshgrid(omega, aw)
    S0 = s0_criterion(W, A)

    fig = plt.figure(figsize=(4.5, 3.5))
    ax = fig.add_subplot(111, projection="3d")

    # Floor: the q* = 1 surface itself, single semi-transparent color (no
    # shading gradient, so it reads as one solid tint), with a subtle wireframe
    # on top to convey curvature.
    ax.plot_surface(W, A, S0, color=pink, linewidth=0, antialiased=True,
                    alpha=0.55, rcount=n, ccount=n, shade=False)
    ax.plot_wireframe(W, A, S0, color="0.35", linewidth=0.3, rcount=12,
                      ccount=12)

    # Feasible region (above the surface) as a translucent "cloud": four side
    # curtains that follow the curved floor up to Z_TOP, plus a flat top cap.
    polys = []
    for a_edge in AW_RANGE:                       # front/back edges (fixed aw)
        z_b = s0_criterion(omega, a_edge)
        polys.append([(x, a_edge, z) for x, z in zip(omega, z_b)]
                     + [(x, a_edge, Z_TOP) for x in omega[::-1]])
    for o_edge in OMEGA_RANGE:                    # left/right edges (fixed omega)
        z_b = s0_criterion(o_edge, aw)
        polys.append([(o_edge, a, z) for a, z in zip(aw, z_b)]
                     + [(o_edge, a, Z_TOP) for a in aw[::-1]])
    polys.append([                                # top cap
        (OMEGA_RANGE[0], AW_RANGE[0], Z_TOP),
        (OMEGA_RANGE[1], AW_RANGE[0], Z_TOP),
        (OMEGA_RANGE[1], AW_RANGE[1], Z_TOP),
        (OMEGA_RANGE[0], AW_RANGE[1], Z_TOP),
    ])
    ax.add_collection3d(
        Poly3DCollection(polys, facecolor=pink, edgecolor="none", alpha=0.12))

    ax.set_xlabel(r"$\omega^\ast$", labelpad=6)
    ax.set_ylabel(r"$A^\ast/m^\ast$", labelpad=8)
    ax.zaxis.set_rotate_label(False)  # take manual control of the z-label angle
    ax.set_zlabel(r"$s_0^\ast$", labelpad=6, rotation=90)  # underside to the right
    ax.set_xlim(*OMEGA_RANGE)
    ax.set_ylim(AW_RANGE[1], AW_RANGE[0])  # flipped
    ax.set_zlim(0.0, Z_TOP)
    ax.view_init(elev=22, azim=-128)

    out = OUT_DIR / "criterion_surface.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.45)

    print(f"C_F*^max = {CF_MAX:.4f}")
    print(f"s0* criterion range: [{S0.min():.3f}, {S0.max():.3f}]")
    print(f"example (omega*=8, A*/m*=1.0): s0* = {s0_criterion(8.0, 1.0):.3f}, "
          f"phi1 = {np.degrees(1.5 * s0_criterion(8.0, 1.0) / 0.75):.1f} deg")
    print(f"worst corner (omega*=8, A*/m*=0.2): s0* = {s0_criterion(8.0, 0.2):.3f}, "
          f"phi1 = {np.degrees(1.5 * s0_criterion(8.0, 0.2) / 0.75):.1f} deg")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

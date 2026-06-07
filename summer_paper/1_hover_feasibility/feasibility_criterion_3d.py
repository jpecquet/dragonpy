"""
The hover-feasibility criterion q* >= 1 as a 3-D surface.

Section 2 reduces feasibility to a single inequality in the three amplitude
groups,

    q* = (A_w*/m*) (s0* omega*)^2 >= 1,

obtained by taking the pitch coefficient C_psi ~ 1 (its many-element converged
value). Its boundary q* = 1 is a 2-D surface in (omega*, A_w*/m*, s0*) space; the
hover-capable region is ABOVE it (larger excursion -> more force). Solving for the
vertical axis,

    s0*  =  1 / (omega* sqrt(A_w*/m*)).

We draw that surface and shade the feasible region above it as a translucent
volume.

Output: light-mode figure in summer_paper/1_hover_feasibility/figures/.
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

OUT_DIR = HERE / "figures"

# Parameter ranges (Table 1) for the two horizontal axes.
OMEGA_RANGE = (8.0, 20.0)
AW_RANGE = (0.05, 0.25)
Z_TOP = 0.6  # top of the plotted box / feasible cloud


def s0_criterion(omega, aw):
    """s0* on the q* = 1 surface: q* = aw s0^2 omega^2 = 1."""
    return 1.0 / (omega * np.sqrt(aw))


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pink = plt.cm.RdPu(0.62)

    # Criterion surface s0* = 1/(omega* sqrt(A_w*/m*)) on q* = 1.
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
    ax.set_ylabel(r"$A_w^\ast/m^\ast$", labelpad=8)
    ax.zaxis.set_rotate_label(False)  # take manual control of the z-label angle
    ax.set_zlabel(r"$s_0^\ast$", labelpad=6, rotation=90)  # underside to the right
    ax.set_xlim(*OMEGA_RANGE)
    ax.set_ylim(AW_RANGE[1], AW_RANGE[0])  # flipped
    ax.set_zlim(0.0, Z_TOP)
    ax.view_init(elev=22, azim=-128)

    out = OUT_DIR / "criterion_surface.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.45)

    print(f"s0* criterion range: [{S0.min():.3f}, {S0.max():.3f}]")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

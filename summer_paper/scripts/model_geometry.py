"""
Model geometry definition figure for the Model Description section.

Single panel showing, in the fixed frame (x to the right, z up):
the point mass body at the origin (center-of-mass symbol), the stroke basis
(s, n) obtained by rotating (x, z) by the stroke plane angle gamma, the
lumped wing (slender ellipse) at position s(t) s along the stroke axis, the chord
direction c at pitch angle psi from the stroke plane normal n (dashed copy of
n translated to the wing hinge), and the gamma and psi arcs.

All directions are constructed from gamma and psi via the report's equations

    s = cos(gamma) x + sin(gamma) z
    n = -sin(gamma) x + cos(gamma) z
    c = -sin(psi) s + cos(psi) n

so the drawing is consistent with the text by construction.

Output: light-mode figure in summer_paper/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Wedge

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

GAMMA_DEG = 35.0  # stroke plane angle drawn
PSI_DEG = 30.0    # pitch angle drawn
S_T = 1.15        # wing hinge distance s(t) drawn


def arc_arrow(ax, center, radius, a0_deg, a1_deg, color, lw=1.0, ls="-",
              zorder=3):
    th = np.radians(np.linspace(a0_deg, a1_deg, 64))
    cx, cy = center
    ax.plot(cx + radius * np.cos(th), cy + radius * np.sin(th), color=color,
            lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)
    ax.annotate("", xy=(cx + radius * np.cos(th[-1]), cy + radius * np.sin(th[-1])),
                xytext=(cx + radius * np.cos(th[-5]), cy + radius * np.sin(th[-5])),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=7, shrinkA=0, shrinkB=0,
                                zorder=zorder))


def vec_arrow(ax, tip, color, lw=1.3, tail=(0.0, 0.0), zorder=4):
    ax.annotate("", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=zorder)


def com_symbol(ax, center, radius, color, zorder=8):
    """Standard center-of-mass marker: circle with alternating quadrants."""
    ax.add_patch(Circle(center, radius, fc="#ffffff", ec=color, lw=1.0,
                        zorder=zorder))
    for a0 in (0.0, 180.0):
        ax.add_patch(Wedge(center, radius, a0, a0 + 90.0, fc=color, ec=None,
                           zorder=zorder + 1))
    ax.add_patch(Circle(center, radius, fc="none", ec=color, lw=1.0,
                        zorder=zorder + 2))


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figure inserted at native size)
    apply_matplotlib_style(style)

    gamma = np.radians(GAMMA_DEG)
    psi = np.radians(PSI_DEG)

    # report equations: stroke basis from gamma, chord direction from psi
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])
    c_hat = -np.sin(psi) * s_hat + np.cos(psi) * n_hat
    hinge = S_T * s_hat

    muted = style.muted_text_color
    text = style.text_color

    fig, ax = plt.subplots(figsize=(3.4, 2.3))

    # fixed frame (background, muted)
    vec_arrow(ax, (1.05, 0.0), muted)
    vec_arrow(ax, (0.0, 1.05), muted)
    ax.text(1.12, 0.0, r"$\hat{x}$", color=muted, ha="left", va="center")
    ax.text(-0.07, 1.09, r"$\hat{z}$", color=muted, ha="right", va="center")

    # stroke basis
    vec_arrow(ax, tuple(1.75 * s_hat), text)
    vec_arrow(ax, tuple(1.0 * n_hat), text)
    ax.text(*(1.85 * s_hat), r"$\hat{s}$", color=text, ha="left", va="center")
    ax.text(*(1.13 * n_hat), r"$\hat{n}$", color=text, ha="center", va="center")

    # stroke plane angle, from z to n
    arc_arrow(ax, (0.0, 0.0), 0.50, 90.0, 90.0 + GAMMA_DEG, text)
    gmid = np.radians(90.0 + 0.5 * GAMMA_DEG)
    ax.text(0.66 * np.cos(gmid), 0.66 * np.sin(gmid), r"$\gamma$",
            color=text, ha="center", va="center")

    # flapping coordinate label along the stroke axis
    spos = 0.62 * s_hat + 0.10 * np.array([s_hat[1], -s_hat[0]])
    ax.text(*spos, r"$s(t)$", color=text, ha="left", va="top")

    # lumped wing: slender ellipse along the chord, hinge dot on the s axis
    chord_angle = np.degrees(np.arctan2(c_hat[1], c_hat[0]))
    ax.add_patch(Ellipse(tuple(hinge), width=0.84, height=0.11,
                         angle=chord_angle, fc=style.wing_color,
                         ec=style.wing_edge_color, lw=1.0, zorder=5))
    ax.plot(*hinge, marker="o", ms=5, color=text, zorder=7)

    # stroke plane normal translated to the hinge (psi reference), and chord
    ref = hinge + 0.72 * n_hat
    ax.plot([hinge[0], ref[0]], [hinge[1], ref[1]], ls="--", lw=0.9,
            color=muted, zorder=4)
    ax.text(*(hinge + 0.82 * n_hat), r"$\hat{n}$", color=muted, ha="center",
            va="center")
    vec_arrow(ax, tuple(hinge + 0.75 * c_hat), text, tail=tuple(hinge),
              zorder=6)
    perp = np.array([-c_hat[1], c_hat[0]])  # normal to the chord, away from n
    ax.text(*(hinge + 0.55 * c_hat + 0.15 * perp), r"$\hat{c}$", color=text,
            ha="right", va="center")

    # pitch angle, from the translated n to the chord direction
    n_angle = 90.0 + GAMMA_DEG
    arc_arrow(ax, tuple(hinge), 0.55, n_angle, n_angle + PSI_DEG, text)
    pmid = np.radians(n_angle + 0.5 * PSI_DEG)
    ax.text(hinge[0] + 0.68 * np.cos(pmid), hinge[1] + 0.68 * np.sin(pmid),
            r"$\psi$", color=text, ha="center", va="center")

    com_symbol(ax, (0.0, 0.0), 0.065, text)

    # point mass and lumped wing area annotations
    ax.text(-0.09, -0.08, r"$m$", color=text, ha="right", va="top")
    ax.text(*(hinge - 0.54 * c_hat + np.array([0.0, -0.02])),
            r"$\dfrac{A}{N}$", color=text, ha="left", va="top")

    ax.set_xlim(-0.85, 1.68)
    ax.set_ylim(-0.26, 1.48)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "model_geometry.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""
Stroke-frame velocity decomposition and flow angle for the Definitions
and Assumptions section of the controller design.

Single panel, same style as the velocity triangle figure: the lumped
wing chord at pitch psi from the stroke-plane normal, the wing velocity
v* decomposed into its stroke-frame components v_s* s_hat (flapping)
and <u*> n_hat (body velocity, along the stroke-plane normal by the
section's assumption), and the three signed angles measured with the
same sense (positive tilting toward -s): psi from n to c, chi from n to
v, and alpha = psi - chi from v to c.

The drawn configuration is the +s half-stroke (sin tau* < 0), where
both psi and chi are negative; chi is chosen so the resultant v*
matches the velocity triangle figure (alpha = psi - chi = 44 deg).

Output: light-mode figure in summer_paper/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

GAMMA_DEG = 35.0   # stroke plane angle drawn (matches figs 1-2)
PSI_DEG = -20.0    # pitch angle drawn (leading edge into the motion)
CHI_DEG = -64.0    # flow angle drawn (v* matches the velocity triangle fig)
V_MAG = 1.0        # wing velocity magnitude


def arc_arrow(ax, center, radius, a0_deg, a1_deg, color, lw=1.0, zorder=3):
    th = np.radians(np.linspace(a0_deg, a1_deg, 64))
    cx, cy = center
    ax.plot(cx + radius * np.cos(th), cy + radius * np.sin(th), color=color,
            lw=lw, solid_capstyle="round", zorder=zorder)
    ax.annotate("", xy=(cx + radius * np.cos(th[-1]), cy + radius * np.sin(th[-1])),
                xytext=(cx + radius * np.cos(th[-5]), cy + radius * np.sin(th[-5])),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=7, shrinkA=0, shrinkB=0,
                                zorder=zorder))


def vec_arrow(ax, tip, color, lw=1.4, tail=(0.0, 0.0), zorder=4):
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=zorder)


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figure inserted at native size)
    apply_matplotlib_style(style)

    gamma = np.radians(GAMMA_DEG)
    psi = np.radians(PSI_DEG)
    chi = np.radians(CHI_DEG)

    # stroke basis; psi and chi both measured from n, positive toward -s
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])
    c_hat = -np.sin(psi) * s_hat + np.cos(psi) * n_hat
    v_hat = -np.sin(chi) * s_hat + np.cos(chi) * n_hat
    v_vec = V_MAG * v_hat
    v_s = v_vec @ s_hat   # in-plane (flapping) component, > 0 on +s half-stroke
    v_n = v_vec @ n_hat   # normal component <u*>
    corner = v_s * s_hat

    alpha_deg = PSI_DEG - CHI_DEG
    print(f"alpha = psi - chi = {alpha_deg:.1f} deg, "
          f"J at mid-half-stroke = {v_n / v_s:.2f}")
    assert 0.0 < alpha_deg < 90.0 and v_s > 0.0 and v_n > 0.0

    muted = style.muted_text_color
    text = style.text_color
    n_deg = 90.0 + GAMMA_DEG
    v_deg = np.degrees(np.arctan2(v_hat[1], v_hat[0]))
    c_deg = np.degrees(np.arctan2(c_hat[1], c_hat[0]))

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    # stroke axis and stroke-plane normal reference
    ax.plot([-0.95 * s_hat[0], 1.35 * s_hat[0]],
            [-0.95 * s_hat[1], 1.35 * s_hat[1]], ls=":", lw=0.8, color=muted,
            zorder=1)
    ax.text(*(1.47 * s_hat), r"$\hat{s}$", color=muted, ha="left", va="center")
    ax.plot([0.0, 1.05 * n_hat[0]], [0.0, 1.05 * n_hat[1]], ls="--", lw=0.9,
            color=muted, zorder=1)
    ax.text(*(1.17 * n_hat), r"$\hat{n}$", color=muted, ha="center",
            va="center")

    # lumped wing at the hinge, chord direction at pitch psi from n
    ax.add_patch(Ellipse((0.0, 0.0), width=0.84, height=0.11, angle=c_deg,
                         fc=style.wing_color, ec=style.wing_edge_color, lw=1.0,
                         zorder=5))
    ax.plot(0.0, 0.0, marker="o", ms=5, color=text, zorder=7)
    vec_arrow(ax, 0.95 * c_hat, text, zorder=6)
    ax.text(*(1.07 * c_hat), r"$\hat{c}$", color=text, ha="right", va="center")

    # velocity decomposition in the stroke frame: v* = v_s s + <u*> n,
    # legs tip to tail with a right-angle mark at the corner
    vec_arrow(ax, corner, text)
    perp_s = np.array([s_hat[1], -s_hat[0]])
    ax.text(*(0.5 * corner + 0.13 * perp_s), r"$v_s^*\,\hat{s}$",
            color=text, ha="left", va="top")
    vec_arrow(ax, v_vec, text, tail=tuple(corner))
    u_mid = corner + 0.5 * v_n * n_hat
    ax.text(*(u_mid + 0.14 * s_hat), r"$\langle u^* \rangle\,\hat{n}$",
            color=text, ha="left", va="bottom")
    sq = 0.07
    sq_pts = [corner - sq * s_hat, corner - sq * s_hat + sq * n_hat,
              corner + sq * n_hat]
    ax.plot([p[0] for p in sq_pts], [p[1] for p in sq_pts], lw=0.8,
            color=muted, zorder=2)
    vec_arrow(ax, v_vec, text, lw=1.6)
    ax.text(*(v_vec + 0.12 * v_hat), r"$\vec{v}^*$", color=text, ha="left",
            va="bottom")

    # the three signed angles, all measured with the same sense:
    # chi from n to v, alpha from v to c, psi from n to c
    arc_arrow(ax, (0.0, 0.0), 0.47, n_deg, v_deg, text)
    ax.text(0.35 * np.cos(np.radians(75.0)), 0.35 * np.sin(np.radians(75.0)),
            r"$\chi$", color=text, ha="center", va="center")
    arc_arrow(ax, (0.0, 0.0), 0.62, v_deg, c_deg, text)
    amid = np.radians(0.5 * (v_deg + c_deg))
    ax.text(0.75 * np.cos(amid), 0.75 * np.sin(amid), r"$\alpha$",
            color=text, ha="center", va="center")
    arc_arrow(ax, (0.0, 0.0), 0.77, n_deg, c_deg, text)
    pmid = np.radians(0.5 * (n_deg + c_deg))
    ax.text(0.89 * np.cos(pmid), 0.89 * np.sin(pmid), r"$\psi$",
            color=text, ha="center", va="center")

    ax.set_xlim(-0.95, 1.48)
    ax.set_ylim(-0.66, 1.16)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "flow_angles.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

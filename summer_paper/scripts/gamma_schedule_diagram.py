"""
Stroke-plane schedule figure for level flight (eq:gamma-schedule).

Three panels at increasing forward speed (J = 0, 0.25, 0.6), each a
wingbeat overlay in the hover/pursuit-diagram style. At hover the stroke
plane holds the inclined-hover lean gamma = sigma_x gamma_hover = -45 deg
(sigma_x = +1, +x heading); as J grows the blend leans it toward the
velocity-aligned value gamma_u = -90 deg, at which the stroke-plane
normal n points along the flight direction (axial inflow). Each panel
draws the stroke axis s (dotted), the normal n (dashed), the body
velocity u* (length growing with J, schematic), and the wing over one
full wingbeat at psi1 = psi1_opt(J); the blended gamma is annotated
under each panel.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from maneuver_control import PSI1_LIM
from generalized_control import SCALE, gamma_schedule, slave_psi1

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

J_PANELS = (0.0, 0.25, 0.6)
PSI0_DEG = 20.0          # mean pitch drawn (schematic)
N_SNAP = 7               # snapshots per half-stroke, as in the diagram figures
S0_DRAW = 0.75           # drawn stroke amplitude
CHORD = 0.38             # drawn chord length
U_SCALE = 1.5            # drawn u* arrow length per unit J (schematic)


def vec_arrow(ax, tip, color, lw=1.4, tail=(0.0, 0.0), zorder=5):
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=zorder)


def draw_panel(ax, J, style):
    muted = style.muted_text_color
    text = style.text_color
    gamma, gamma_u, _ = gamma_schedule((J * SCALE, 0.0, 0.0), sx=1.0)
    psi1 = float(np.clip(slave_psi1(J), *PSI1_LIM))
    psi0 = np.radians(PSI0_DEG)
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])

    # stroke axis and stroke-plane normal, hover-diagram style
    ax.plot([-1.05 * s_hat[0], 1.05 * s_hat[0]],
            [-1.05 * s_hat[1], 1.05 * s_hat[1]], ls=":", lw=0.8, color=muted,
            zorder=1)
    ax.text(*(1.2 * s_hat), r"$\hat{s}$", color=muted, ha="center",
            va="center")
    ax.plot([0.0, 1.0 * n_hat[0]], [0.0, 1.0 * n_hat[1]], ls="--", lw=0.9,
            color=muted, zorder=1)
    ax.text(*(1.16 * n_hat), r"$\hat{n}$", color=muted, ha="center",
            va="center")

    # full wingbeat of wing snapshots: -s half behind and darker
    darker = tuple(0.8 * np.array(mcolors.to_rgb(style.wing_color)))
    tau_half = (np.arange(N_SNAP) + 0.5) * np.pi / N_SNAP
    for tau0, fc, zo in ((tau_half, darker, 2),
                         (tau_half + np.pi, style.wing_color, 3)):
        for tau in tau0:
            psi = psi0 + psi1 * np.sin(tau)
            c_hat = -np.sin(psi) * s_hat + np.cos(psi) * n_hat
            c_deg = np.degrees(np.arctan2(c_hat[1], c_hat[0]))
            center = S0_DRAW * np.cos(tau) * s_hat
            ax.add_patch(Ellipse(tuple(center), width=CHORD, height=0.05,
                                 angle=c_deg, fc=fc,
                                 ec=style.wing_edge_color, lw=0.9, zorder=zo))
    ax.plot(0.0, 0.0, marker="o", ms=5, color=text, zorder=7)

    # body velocity along +x, length growing with J (schematic)
    if J > 0.0:
        tip = np.array([U_SCALE * J, 0.0])
        vec_arrow(ax, tip, text, lw=1.6, zorder=6)
        ax.text(tip[0] + 0.02, -0.13, r"$\vec{u}^*$", color=text,
                ha="left", va="top", zorder=8)

    print(f"J = {J:g}: gamma = {np.degrees(gamma):.1f} deg "
          f"(gamma_u = {np.degrees(gamma_u):.0f} deg), "
          f"psi1_opt = {np.degrees(psi1):.0f} deg")

    ax.set_xlim(-1.25, 1.45)
    ax.set_ylim(-1.38, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figure inserted at native size)
    apply_matplotlib_style(style)

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5),
                             constrained_layout=True)
    for ax, J in zip(axes, J_PANELS):
        draw_panel(ax, J, style)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "gamma_schedule_level.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

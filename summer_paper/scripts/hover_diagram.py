"""
Control-law diagram for the hover use case.

Companion to the pursuit diagram, same style: the body at the origin
with a full-wingbeat overlay of wing snapshots (grey ellipses, the -s
half-stroke behind and darker), pinned at the hover stroke-plane angle
gamma_hover = -45 deg with psi1 = psi1_opt(0). The reference velocity is
zero, so it is not drawn: the body carries a residual drift velocity u,
and the force demand of eq:fdes composes tip to tail the weight
compensation z_hat with the braking term -K u, giving the resultant
<F*>_des that both supports the weight and opposes the drift.

Velocity and force arrow lengths are schematic (independent scales).

Output: light-mode figure in summer_paper/figures/.
"""

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

GAMMA_DEG = -45.0       # hover stroke-plane angle (pinned, sigma_x = +1)
PSI1_DEG = 51.0         # psi1_opt(0), hover pin value
PSI0_DEG = 20.0         # mean pitch drawn
U_DIR_DEG = 205.0       # residual drift direction
U_LEN = 0.42            # drawn drift arrow length (schematic)
A_SCALE = 1.2           # drawn length of -K u per unit drift length
Z_LEN = 0.8             # drawn weight-compensation leg
S0_DRAW = 0.75          # drawn stroke amplitude
CHORD = 0.38            # drawn chord length
N_SNAP = 7


def vec_arrow(ax, tip, color, lw=1.4, tail=(0.0, 0.0), zorder=5):
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=zorder)


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figure inserted at native size)
    apply_matplotlib_style(style)

    muted = style.muted_text_color
    text = style.text_color

    gamma = np.radians(GAMMA_DEG)
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])

    fig, ax = plt.subplots(figsize=(3.2, 2.85), constrained_layout=True)

    # stroke axis and stroke-plane normal, pursuit-diagram style
    line = 1.45 * S0_DRAW
    ax.plot([-line * s_hat[0], line * s_hat[0]],
            [-line * s_hat[1], line * s_hat[1]], ls=":", lw=0.8, color=muted,
            zorder=1)
    ax.text(*(1.58 * S0_DRAW * s_hat), r"$\hat{s}$", color=muted, ha="left",
            va="center")
    ax.plot([0.0, 0.92 * S0_DRAW * n_hat[0]],
            [0.0, 0.92 * S0_DRAW * n_hat[1]], ls="--", lw=0.9,
            color=muted, zorder=1)
    ax.text(*(1.04 * S0_DRAW * n_hat), r"$\hat{n}$", color=muted, ha="center",
            va="center")

    # full wingbeat of wing snapshots: -s half behind and darker
    darker = tuple(0.8 * np.array(mcolors.to_rgb(style.wing_color)))
    psi0 = np.radians(PSI0_DEG)
    psi1 = np.radians(PSI1_DEG)
    tau_half = (np.arange(N_SNAP) + 0.5) * np.pi / N_SNAP
    for tau0, fc, zo in ((tau_half, darker, 2),
                         (tau_half + np.pi, style.wing_color, 3)):
        for tau in tau0:
            psi = psi0 + psi1 * np.sin(tau)
            c_hat = -np.sin(psi) * s_hat + np.cos(psi) * n_hat
            c_deg = np.degrees(np.arctan2(c_hat[1], c_hat[0]))
            center = S0_DRAW * np.cos(tau) * s_hat
            ax.add_patch(Ellipse(tuple(center), width=CHORD, height=0.052,
                                 angle=c_deg, fc=fc,
                                 ec=style.wing_edge_color, lw=0.9, zorder=zo))
    ax.plot(0.0, 0.0, marker="o", ms=5, color=text, zorder=7)

    # residual drift velocity (u_r = 0 is not drawn)
    u_dir = np.radians(U_DIR_DEG)
    u_vec = U_LEN * np.array([np.cos(u_dir), np.sin(u_dir)])
    vec_arrow(ax, u_vec, text)
    u_base = u_vec - 0.11 * u_vec / np.linalg.norm(u_vec)
    ax.text(*(u_base + (-0.07, -0.08)), r"$\vec{u}^*$", color=text,
            ha="center", va="top")

    # force demand, tip to tail: the weight compensation, then the braking
    # term -K u, an amplified opposite copy of the drift
    err = -u_vec
    z_leg = np.array([0.0, Z_LEN])
    vec_arrow(ax, z_leg, text)
    ax.text(0.10, Z_LEN - 0.14, r"$\hat{z}$", color=text, ha="center",
            va="center", zorder=8)
    f_tip = z_leg + A_SCALE * err
    vec_arrow(ax, f_tip, text, tail=tuple(z_leg))
    # label set along the leg itself, kept upright-readable
    ang = np.degrees(np.arctan2(err[1], err[0]))
    ang_txt = ang - 180.0 if abs(ang) > 90.0 else ang
    along = err / np.linalg.norm(err)
    perp = np.array([-along[1], along[0]])
    pos = z_leg + 0.5 * A_SCALE * err + 0.05 * perp
    ax.text(*pos, r"$-K \vec{u}^*$", color=text,
            ha="center", va="bottom", rotation=ang_txt,
            rotation_mode="anchor", transform_rotates_text=True)
    vec_arrow(ax, f_tip, text)
    f_base = f_tip - 0.11 * f_tip / np.linalg.norm(f_tip)
    ax.text(f_tip[0] + 0.05, f_base[1],
            r"$\langle \vec{F}^* \rangle^\text{des}$",
            color=text, ha="left", va="center")

    ax.set_xlim(-1.02, 1.48)
    ax.set_ylim(-0.98, 1.20)
    ax.set_aspect("equal")
    ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "hover_diagram.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

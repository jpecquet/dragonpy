"""
Control-law diagram for the prey pursuit use case.

Single panel in the style of the hover beta figure: the body at the
origin with a full-wingbeat overlay of wing snapshots (grey ellipses,
the -s half-stroke behind and darker), drawn at the blended stroke-plane
angle of eq:gamma-schedule for the drawn advance ratio. The prey sits on
the line of sight (sparse dots); the reference velocity u_r = u_cmd
r_hat of eq:ur-pursuit points along it, while the body velocity u lags
below it (tail chase, prey moving along +x). The velocity error
u_r - u closes the velocity triangle, and the force demand of eq:fdes
is composed tip to tail from the weight compensation z_hat and the
amplified error K (u_r - u), giving the resultant <F*>_des.

Velocity and force arrow lengths are schematic (independent scales);
the stroke-plane geometry is computed from the report's equations at
J = 0.2 (gamma blend, slaved psi1 from fig:hover_opt_J).

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

J_DRAW = 0.2            # advance ratio drawn (matches the pursuit run)
JC = 0.35               # gamma blend constant (eq:gamma-schedule)
GAMMA_HOVER = -45.0     # deg, sigma_x = +1
PSI1_DEG = 41.0         # slaved psi1_opt(J = 0.2), fig:hover_opt_J
U_DIR_DEG = -8.0        # body velocity direction (lagging the line of sight)
LOS_DEG = 10.0          # line-of-sight bearing to the prey
U_LEN, UR_LEN = 0.85, 1.15   # drawn velocity arrow lengths (schematic)
PREY_DIST = 2.7
A_SCALE = 1.6           # drawn length of K (u_r - u) per unit error length
Z_LEN = 0.55            # drawn weight-compensation leg
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

    # stroke-plane angle from the gamma schedule: n aligned with u at full
    # blend, inclined hover plane at J = 0
    f = 1.0 - np.exp(-((J_DRAW / JC) ** 2))
    gamma_u = U_DIR_DEG - 90.0
    gamma = np.radians((1.0 - f) * GAMMA_HOVER + f * gamma_u)
    print(f"f(J={J_DRAW}) = {f:.3f}, gamma = {np.degrees(gamma):.1f} deg")
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])

    fig, ax = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)

    # stroke axis and stroke-plane normal, hover-beta figure style
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
    psi1 = np.radians(PSI1_DEG)
    tau_half = (np.arange(N_SNAP) + 0.5) * np.pi / N_SNAP
    for tau0, fc, zo in ((tau_half, darker, 2),
                         (tau_half + np.pi, style.wing_color, 3)):
        for tau in tau0:
            psi = psi1 * np.sin(tau)
            c_hat = -np.sin(psi) * s_hat + np.cos(psi) * n_hat
            c_deg = np.degrees(np.arctan2(c_hat[1], c_hat[0]))
            center = S0_DRAW * np.cos(tau) * s_hat
            ax.add_patch(Ellipse(tuple(center), width=CHORD, height=0.052,
                                 angle=c_deg, fc=fc,
                                 ec=style.wing_edge_color, lw=0.9, zorder=zo))
    ax.plot(0.0, 0.0, marker="o", ms=5, color=text, zorder=7)

    # prey on the line of sight, moving along +x
    los = np.radians(LOS_DEG)
    r_hat = np.array([np.cos(los), np.sin(los)])
    prey = PREY_DIST * r_hat
    ax.plot([0.0, prey[0]], [0.0, prey[1]], ls=(0, (1, 3)), lw=1.0,
            color=muted, zorder=1)
    ax.plot(*prey, marker="o", ms=5, color=text, zorder=6)
    ax.text(prey[0], prey[1] + 0.13, r"$\vec{x}^*_\text{prey}$", color=text,
            ha="center", va="bottom")
    vec_arrow(ax, prey + (0.38, 0.0), text, tail=tuple(prey))
    ax.text(prey[0] + 0.19, prey[1] - 0.12, r"$\vec{u}^*_\text{prey}$",
            color=text, ha="center", va="top")

    # velocity triangle: u lags the reference u_r = u_cmd r_hat (eq:ur-pursuit)
    u_dir = np.radians(U_DIR_DEG)
    u_vec = U_LEN * np.array([np.cos(u_dir), np.sin(u_dir)])
    ur_vec = UR_LEN * r_hat
    vec_arrow(ax, u_vec, text)
    u_base = u_vec - 0.11 * u_vec / np.linalg.norm(u_vec)
    ax.text(*(u_base + (-0.07, -0.08)), r"$\vec{u}^*$", color=text,
            ha="center", va="top")
    vec_arrow(ax, ur_vec, text)
    ur_base = ur_vec - 0.11 * r_hat
    ax.text(*(ur_base + (0.0, 0.10)), r"$\vec{u}^*_r$", color=text,
            ha="center", va="bottom")
    err = ur_vec - u_vec
    vec_arrow(ax, ur_vec, text, tail=tuple(u_vec))
    ax.text(*(u_vec + 0.5 * err + (0.12, -0.05)),
            r"$\vec{u}^*_r - \vec{u}^*$", color=text, ha="left", va="center")

    # force demand, tip to tail: the weight compensation, then K (u_r - u),
    # a parallel (amplified) copy of the velocity error
    z_leg = np.array([0.0, Z_LEN])
    vec_arrow(ax, z_leg, text)
    ax.text(0.10, Z_LEN - 0.06, r"$\hat{z}$", color=text, ha="center",
            va="center", zorder=8)
    f_tip = z_leg + A_SCALE * err
    vec_arrow(ax, f_tip, text, tail=tuple(z_leg))
    # label set along the leg itself so it cannot be read as belonging to
    # the resultant next to it (the leg is exactly parallel to u_r - u)
    ang = np.degrees(np.arctan2(err[1], err[0]))
    along = err / np.linalg.norm(err)
    perp = np.array([-along[1], along[0]])
    pos = z_leg + 0.5 * A_SCALE * err + 0.04 * perp
    ax.text(*pos, r"$K (\vec{u}^*_r - \vec{u}^*)$", color=text,
            ha="center", va="bottom", rotation=ang,
            rotation_mode="anchor", transform_rotates_text=True)
    vec_arrow(ax, f_tip, text)
    f_base = f_tip - 0.11 * f_tip / np.linalg.norm(f_tip)
    ax.text(f_tip[0] + 0.05, f_base[1],
            r"$\langle \vec{F}^* \rangle^\text{des}$",
            color=text, ha="left", va="center")

    ax.set_xlim(-0.85, 3.15)
    ax.set_ylim(-1.25, 1.32)
    ax.set_aspect("equal")
    ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pursuit_diagram.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

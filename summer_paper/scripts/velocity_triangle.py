"""
Velocity triangle and aerodynamic force directions for the Model
Description section.

Single panel, zoomed on one lumped wing (same style as the model geometry
figure): the wing chord section at pitch psi from the stroke-plane normal,
the velocity triangle v = sdot + u (flapping velocity along the stroke
axis, dotted, plus body velocity, added tip to tail), the signed angle of
attack alpha from v to the chord direction c, the chord-normal direction
p = c_perp (counterclockwise rotation of c), and the lift and drag
directions l = -d_perp and d = -v/|v|, with arrow lengths proportional to
C_L and C_D.

All quantities are computed from the report's equations (c from (gamma,
psi), alpha from atan2(v x c, v . c), C_L and C_D from eq:cl and eq:cd),
so the drawing is consistent with the text by construction. The drawn
configuration has the wing advancing along +s with the leading edge
pitched into the motion, giving 0 < alpha < pi/2 (quadrant I, C_L > 0).

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

GAMMA_DEG = 35.0            # stroke plane angle drawn (matches fig 1)
PSI_DEG = -20.0             # pitch angle drawn (leading edge into the motion)
SDOT = 0.646                # flapping velocity magnitude, along +s
U_MAG, U_ANG_DEG = 0.506, 95.0  # body velocity magnitude and world angle
CL0, CD0, CDPI2 = 1.5, 0.1, 2.0
F_SCALE = 0.52              # force arrow length per unit coefficient


def rot90(vec):
    """Counterclockwise rotation by pi/2."""
    return np.array([-vec[1], vec[0]])


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

    # report equations: stroke basis, chord direction, velocity composition
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])
    c_hat = -np.sin(psi) * s_hat + np.cos(psi) * n_hat
    sdot_vec = SDOT * s_hat
    u_ang = np.radians(U_ANG_DEG)
    u_vec = U_MAG * np.array([np.cos(u_ang), np.sin(u_ang)])
    v_vec = sdot_vec + u_vec
    v_hat = v_vec / np.linalg.norm(v_vec)
    d_hat = -v_hat
    l_hat = -rot90(d_hat)

    # angle of attack and coefficients, straight from the model
    cos_a = v_hat @ c_hat
    sin_a = v_hat[0] * c_hat[1] - v_hat[1] * c_hat[0]
    alpha_deg = np.degrees(np.arctan2(sin_a, cos_a))
    cl = 2.0 * CL0 * cos_a * sin_a
    cd = CD0 * cos_a ** 2 + CDPI2 * sin_a ** 2
    print(f"alpha = {alpha_deg:.1f} deg, C_L = {cl:.2f}, C_D = {cd:.2f}")
    assert 0.0 < alpha_deg < 90.0 and cl > 0.0

    muted = style.muted_text_color
    text = style.text_color
    v_deg = np.degrees(np.arctan2(v_hat[1], v_hat[0]))
    c_deg = np.degrees(np.arctan2(c_hat[1], c_hat[0]))

    fig, ax = plt.subplots(figsize=(3.55, 2.7))

    # stroke axis and lumped wing
    ax.plot([-0.95 * s_hat[0], 1.35 * s_hat[0]],
            [-0.95 * s_hat[1], 1.35 * s_hat[1]], ls=":", lw=0.8, color=muted,
            zorder=1)
    ax.text(*(1.47 * s_hat), r"$\hat{s}$", color=muted, ha="left", va="center")
    ax.add_patch(Ellipse((0.0, 0.0), width=0.84, height=0.11, angle=c_deg,
                         fc=style.wing_color, ec=style.wing_edge_color, lw=1.0,
                         zorder=5))
    ax.plot(0.0, 0.0, marker="o", ms=5, color=text, zorder=7)
    vec_arrow(ax, 0.75 * c_hat, text, zorder=6)
    ax.text(*(0.88 * c_hat), r"$\hat{c}$", color=text, ha="right", va="center")
    p_hat = rot90(c_hat)  # chord-normal direction
    vec_arrow(ax, 0.55 * p_hat, text, zorder=6)
    ax.text(*(0.68 * p_hat), r"$\hat{p}$", color=text, ha="center", va="center")

    # velocity triangle: flapping velocity, then body velocity tip to tail,
    # then the resultant wing velocity relative to the air
    vec_arrow(ax, sdot_vec, text)
    perp_s = np.array([s_hat[1], -s_hat[0]])
    ax.text(*(0.55 * sdot_vec + 0.13 * perp_s), r"$\vec{\dot{s}}$",
            color=text, ha="left", va="top")
    vec_arrow(ax, v_vec, text, tail=tuple(sdot_vec))
    u_mid = sdot_vec + 0.5 * u_vec
    perp_u = np.array([np.sin(u_ang), -np.cos(u_ang)])
    ax.text(*(u_mid + 0.13 * perp_u), r"$\vec{u}$", color=text, ha="left",
            va="bottom")
    vec_arrow(ax, v_vec, text, lw=1.6)
    ax.text(*(v_vec + 0.12 * v_hat), r"$\vec{v}$", color=text, ha="left",
            va="bottom")

    # angle of attack, from the velocity direction to the chord direction
    arc_arrow(ax, (0.0, 0.0), 0.50, v_deg, c_deg, text)
    amid = np.radians(0.5 * (v_deg + c_deg))
    ax.text(0.64 * np.cos(amid), 0.64 * np.sin(amid), r"$\alpha$",
            color=text, ha="center", va="center")

    # lift and drag, lengths proportional to the model coefficients
    lift_tip = F_SCALE * cl * l_hat
    drag_tip = F_SCALE * cd * d_hat
    vec_arrow(ax, lift_tip, style.lift_color, lw=1.6)
    ax.text(*(lift_tip + 0.14 * l_hat), r"$C_L\,\hat{\ell}$",
            color=style.lift_color, ha="right", va="center")
    vec_arrow(ax, drag_tip, style.drag_color, lw=1.6)
    ax.text(*(drag_tip + 0.15 * d_hat), r"$C_D\,\hat{d}$",
            color=style.drag_color, ha="center", va="top")

    ax.set_xlim(-1.18, 1.43)
    ax.set_ylim(-0.82, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "velocity_triangle.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

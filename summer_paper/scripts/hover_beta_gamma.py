"""
Force direction figure for hover at a horizontal vs inclined stroke plane.

Two panels. Left: gamma = 0, reference pitch kinematics (psi0 = 0), mean
force along n = vertical. Right: gamma = gamma_hover < 0 (stroke-plane
normal tilted toward +x), where hover still requires a vertical mean
force, now at the angle beta = -gamma_hover from n; the pitch trim
psi0_hover follows from the inverse of eq:beta-hover.

Each panel overlays the full wingbeat as wing snapshots in the grey
ellipse style (equal time increments, 7 per half-stroke as in the pitch
kinematics figure): the +s half-stroke in front in the standard wing
grey, the -s half-stroke behind it in a darker grey. The stroke axis s
(dotted) and normal n (dashed) are drawn as in the flow-angles figure,
with the mean force vector <F*> from the body and, on the right panel
only, the beta arc from n to the force direction.

The drawn force direction is computed from the hover closed forms
(eq:cfn, eq:cfs) at the trimmed psi0, and checked to be vertical.

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

CL0, DELTA_CD = 1.5, 1.9
GAMMA_HOVER_DEG = -45.0  # inclined stroke plane drawn (the report's choice)
PSI1_DEG = 51.0          # hover-optimal pitch amplitude
N_SNAP = 7               # snapshots per half-stroke, as in the pitch kinematics fig
S0_DRAW = 1.0            # drawn stroke amplitude
CHORD = 0.50             # drawn chord length

_TAU = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)


def hover_force_dir(psi0, psi1, gamma):
    """Unit mean-force direction in the fixed frame, from eq:cfn / eq:cfs."""
    psi = psi0 + psi1 * np.sin(_TAU)
    w = np.sin(_TAU) * np.abs(np.sin(_TAU))
    cfn = 2.0 * CL0 * np.mean(np.sin(2.0 * psi) * w)
    cfs = DELTA_CD * np.mean(np.cos(2.0 * psi) * w)
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])
    f = cfn * n_hat + cfs * s_hat
    assert cfn > 0.0
    return f / np.linalg.norm(f)


def arc_arrow(ax, radius, a0_deg, a1_deg, color, lw=1.0, zorder=6):
    th = np.radians(np.linspace(a0_deg, a1_deg, 64))
    ax.plot(radius * np.cos(th), radius * np.sin(th), color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)
    ax.annotate("", xy=(radius * np.cos(th[-1]), radius * np.sin(th[-1])),
                xytext=(radius * np.cos(th[-5]), radius * np.sin(th[-5])),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=7, shrinkA=0, shrinkB=0,
                                zorder=zorder))


def vec_arrow(ax, tip, color, lw=1.4, tail=(0.0, 0.0), zorder=5):
    ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=zorder)


def draw_panel(ax, gamma, psi0, psi1, style, show_beta):
    muted = style.muted_text_color
    text = style.text_color
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])

    # stroke axis and stroke-plane normal, flow-angles figure style
    ax.plot([-1.45 * s_hat[0], 1.45 * s_hat[0]],
            [-1.45 * s_hat[1], 1.45 * s_hat[1]], ls=":", lw=0.8, color=muted,
            zorder=1)
    ax.text(*(1.58 * s_hat), r"$\hat{s}$", color=muted, ha="left", va="center")
    ax.plot([0.0, 1.45 * n_hat[0]], [0.0, 1.45 * n_hat[1]], ls="--", lw=0.9,
            color=muted, zorder=1)
    ax.text(*(1.58 * n_hat), r"$\hat{n}$", color=muted, ha="center",
            va="center")

    # one full wingbeat of wing snapshots, equal time increments: the -s
    # half-stroke (sin tau > 0) behind and darker, the +s half in front
    darker = tuple(0.8 * np.array(mcolors.to_rgb(style.wing_color)))
    tau_half = (np.arange(N_SNAP) + 0.5) * np.pi / N_SNAP
    for tau0, fc, zo in ((tau_half, darker, 2),
                         (tau_half + np.pi, style.wing_color, 3)):
        for tau in tau0:
            psi = psi0 + psi1 * np.sin(tau)
            c_hat = -np.sin(psi) * s_hat + np.cos(psi) * n_hat
            c_deg = np.degrees(np.arctan2(c_hat[1], c_hat[0]))
            center = S0_DRAW * np.cos(tau) * s_hat
            ax.add_patch(Ellipse(tuple(center), width=CHORD, height=0.066,
                                 angle=c_deg, fc=fc,
                                 ec=style.wing_edge_color, lw=0.9, zorder=zo))
    ax.plot(0.0, 0.0, marker="o", ms=5, color=text, zorder=7)

    # mean force (vertical for hover), computed from the closed forms
    f_hat = hover_force_dir(psi0, psi1, gamma)
    vec_arrow(ax, 1.25 * f_hat, text, lw=2.0, zorder=6)
    side = -1.0 if show_beta else 1.0  # keep clear of the beta arc / n label
    ax.text(1.18 * f_hat[0] + 0.12 * side, 1.18 * f_hat[1],
            r"$\langle \vec{F}^* \rangle$", color=text,
            ha="left" if side > 0 else "right", va="center")

    # beta, from n to the mean force direction
    if show_beta:
        n_deg = np.degrees(np.arctan2(n_hat[1], n_hat[0]))
        f_deg = np.degrees(np.arctan2(f_hat[1], f_hat[0]))
        arc_arrow(ax, 0.60, n_deg, f_deg, text)
        bmid = np.radians(0.5 * (n_deg + f_deg))
        ax.text(0.74 * np.cos(bmid), 0.74 * np.sin(bmid), r"$\beta$",
                color=text, ha="center", va="center")

    ax.set_xlim(-1.62, 1.86)
    ax.set_ylim(-1.30, 1.72)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figure inserted at native size)
    apply_matplotlib_style(style)

    gamma = np.radians(GAMMA_HOVER_DEG)
    psi1 = np.radians(PSI1_DEG)
    # inverse of eq:beta-hover at beta = -gamma_hover
    psi0_hover = 0.5 * np.arctan(np.tan(-gamma) * 2.0 * CL0 / DELTA_CD)
    f_tilt = np.degrees(np.arctan2(hover_force_dir(psi0_hover, psi1, gamma)[0],
                                   hover_force_dir(psi0_hover, psi1, gamma)[1]))
    print(f"psi0_hover = {np.degrees(psi0_hover):.1f} deg "
          f"(beta = {-GAMMA_HOVER_DEG:.0f} deg); trimmed mean force is "
          f"{f_tilt:.2f} deg from vertical")
    assert abs(f_tilt) < 0.5

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.9),
                             constrained_layout=True)
    draw_panel(axes[0], 0.0, 0.0, psi1, style, show_beta=False)
    axes[0].set_title("(a)  Horizontal stroke plane", fontsize=style.font_size,
                      pad=12)
    draw_panel(axes[1], gamma, psi0_hover, psi1, style, show_beta=True)
    axes[1].set_title("(b)  Inclined stroke plane", fontsize=style.font_size,
                      pad=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "hover_beta_gamma.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

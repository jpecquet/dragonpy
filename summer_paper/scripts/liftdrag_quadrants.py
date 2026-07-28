"""
Verification figure for the lift/drag direction conventions

    d = -v/|v|,   l = -d_perp  (clockwise rotation of d by pi/2)
    C_L = 2 C_L0 (v . c)(v x c),   C_D = C_D0 (v . c)^2 + C_Dpi2 (v x c)^2

Four panels in a 2x2 grid, one per quadrant of the angle of attack alpha (the
signed angle from the velocity direction v to the chord direction c), drawn in
the (l, d) basis with v to the right: positive alpha in the top row,
negative alpha in the bottom row. Each panel shows the wing chord (hollow
dot = the leading edge, i.e. the c end), the alpha arc from v to the chord ray
(counterclockwise for alpha > 0, clockwise for alpha < 0), and the resulting
lift and drag forces: drag always along d, lift along +-l with the sign of
C_L = C_L0 sin 2 alpha.

Before drawing, the vector-math coefficient expressions are checked
numerically against the closed forms

    C_L = C_L0 sin 2 alpha
    C_D = C_D0 cos^2 alpha + C_Dpi2 sin^2 alpha

over a fine sweep of alpha in (-pi, pi].

Output: light-mode figure in summer_paper/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

CL0, CD0, CDPI2 = 1.5, 0.1, 2.0
ALPHA_DEG = 35.0  # |alpha| distance from the nearest quadrant boundary


def vector_coeffs(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C_L, C_D from the report's vector-math expressions, with v = +x and
    c at angle alpha from v (so v . c = cos alpha, v x c = sin alpha)."""
    dot = np.cos(alpha)
    cross = np.sin(alpha)
    return 2.0 * CL0 * dot * cross, CD0 * dot ** 2 + CDPI2 * cross ** 2


def verify(n: int = 1001) -> float:
    """Max |vector form - closed form| over alpha in (-pi, pi]."""
    alpha = np.linspace(-np.pi, np.pi, n)
    cl_vec, cd_vec = vector_coeffs(alpha)
    cl = CL0 * np.sin(2.0 * alpha)
    cd = CD0 * np.cos(alpha) ** 2 + CDPI2 * np.sin(alpha) ** 2
    return max(np.abs(cl_vec - cl).max(), np.abs(cd_vec - cd).max())


def arc_arrow(ax, radius, a0_deg, a1_deg, color, lw=1.0, ls="-", zorder=3):
    th = np.radians(np.linspace(a0_deg, a1_deg, 64))
    ax.plot(radius * np.cos(th), radius * np.sin(th), color=color, lw=lw,
            ls=ls, solid_capstyle="round", zorder=zorder)
    ax.annotate("", xy=(radius * np.cos(th[-1]), radius * np.sin(th[-1])),
                xytext=(radius * np.cos(th[-5]), radius * np.sin(th[-5])),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=7, shrinkA=0, shrinkB=0,
                                zorder=zorder))


def vec_arrow(ax, tip, color, lw=1.6, tail=(0.0, 0.0)):
    ax.annotate("", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=4)


def draw_panel(ax, alpha_deg, quad_label, range_label, style):
    alpha = np.radians(alpha_deg)
    cl, cd = vector_coeffs(np.array(alpha))

    muted = style.muted_text_color
    text = style.text_color

    # (l, d) basis reference lines (v to the right, l up, d to the left)
    ax.plot([0, 0], [-0.62, 1.05], ls=":", lw=0.8, color=muted, zorder=1)
    ax.plot([-1.05, 1.05], [0, 0], ls=":", lw=0.8, color=muted, zorder=1)
    ax.text(-0.07, 1.10, r"$\hat{\ell}$", color=muted, ha="right", va="top")
    ax.text(-1.03, -0.09, r"$\hat{d}$", color=muted, ha="left", va="top")

    # wing velocity direction and chord (hollow dot marks the leading edge,
    # the c end of the chord, at angle alpha from v)
    vec_arrow(ax, (0.95, 0.0), text, lw=1.3)
    ax.text(0.80, -0.13, r"$\vec{v}$", color=text, ha="center", va="top")
    chord = np.array([np.cos(alpha), np.sin(alpha)])
    ax.plot([-0.62 * chord[0], 0.62 * chord[0]],
            [-0.62 * chord[1], 0.62 * chord[1]],
            color=text, lw=2.6, solid_capstyle="round", zorder=5)
    ax.plot(*(0.62 * chord), marker="o", ms=6, mfc=style.axes_facecolor,
            mec=text, mew=1.3, zorder=6)

    # alpha arc from v to the chord ray, CCW for alpha > 0, CW for alpha < 0
    arc_arrow(ax, 0.40, 0.0, alpha_deg, text)
    amid = np.radians(0.5 * alpha_deg)
    ax.text(0.56 * np.cos(amid), 0.56 * np.sin(amid), r"$\alpha$",
            color=text, ha="center", va="center")

    # lift along +-l with the sign of C_L, drag always along d; the label of
    # an upward lift arrow sits beside the tip to keep clear of the l label
    scale = 0.42
    vec_arrow(ax, (0.0, scale * cl), style.lift_color)
    sgn_cl = ">" if cl > 0 else "<"
    if cl > 0:
        ax.text(0.10, scale * cl + 0.09, rf"$C_L {sgn_cl} 0$",
                color=style.lift_color, ha="left", va="bottom")
    else:
        ax.text(0.0, scale * cl - 0.13, rf"$C_L {sgn_cl} 0$",
                color=style.lift_color, ha="center", va="top")
    vec_arrow(ax, (-scale * cd, 0.0), style.drag_color)
    # drag label on whichever side of the axis the chord's left half is not
    left_y = -np.sign(chord[0]) * chord[1]
    ax.text(-scale * cd * 0.8 - 0.14, -0.14 * np.sign(left_y),
            r"$C_D > 0$", color=style.drag_color, ha="center",
            va="top" if left_y > 0 else "bottom")

    # quadrant numeral top left, alpha range top right as in the source sketch
    ax.text(-1.12, 1.42, quad_label, color=text, ha="left", va="top")
    ax.text(1.12, 1.42, range_label, color=text, ha="right", va="top")

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.06, 1.48)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    err = verify()
    status = "PASS" if err < 1e-12 else "FAIL"
    print(f"vector form vs closed form: max |dC| = {err:.3e}  [{status}]")
    if status == "FAIL":
        raise SystemExit("vector-math coefficient expressions do not match")

    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figures inserted at native size)
    apply_matplotlib_style(style)

    # positive alpha in the top row, negative in the bottom
    cases = [
        (ALPHA_DEG, "I", r"$0 < \alpha < \dfrac{\pi}{2}$"),
        (180.0 - ALPHA_DEG, "II", r"$\dfrac{\pi}{2} < \alpha < \pi$"),
        (ALPHA_DEG - 180.0, "III", r"$-\pi < \alpha < -\dfrac{\pi}{2}$"),
        (-ALPHA_DEG, "IV", r"$-\dfrac{\pi}{2} < \alpha < 0$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(4.4, 4.7))
    for ax, (alpha_deg, quad_label, range_label) in zip(axes.flat, cases):
        draw_panel(ax, alpha_deg, quad_label, range_label, style)
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.0,
                        wspace=0.12, hspace=0.15)

    # separators between the panels, midway between the drawn axes boxes
    fig.canvas.draw()
    p00 = axes[0, 0].get_position()
    x_sep = 0.5 * (p00.x1 + axes[0, 1].get_position().x0)
    y_sep = 0.5 * (p00.y0 + axes[1, 0].get_position().y1)
    fig.add_artist(plt.Line2D([x_sep, x_sep], [0.02, 0.98],
                              color=style.grid_color, lw=0.6,
                              transform=fig.transFigure))
    fig.add_artist(plt.Line2D([0.03, 0.97], [y_sep, y_sep],
                              color=style.grid_color, lw=0.6,
                              transform=fig.transFigure))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "liftdrag_quadrants.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

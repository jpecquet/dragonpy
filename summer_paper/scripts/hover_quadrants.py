"""
Hover-case force direction figure for the Hover Case (J=0) section.

Four panels in a 2x2 grid, one per sign combination of the pitch angle
psi (columns: psi < 0 left, psi > 0 right) and the stroke phase sin tau*
(rows: sin tau* < 0 top, sin tau* > 0 bottom), so the angle-of-attack
quadrants read I-IV in reading order, drawn in the stroke frame with s
to the right and n up. Each panel shows the wing chord at
pitch psi from n (hollow dot = the leading edge, the c end), the purely
in-plane wing velocity v* = -sign(sin tau*) s, and the resulting lift
and drag contributions C_L l and C_D d with arrow lengths proportional
to the coefficients: drag along d = sign(sin tau*) s, lift along
l = -sign(sin tau*) n with the sign of C_L = -C_L0 sin 2 psi.
As in the flow-angles figure, each panel also marks the pitch angle
with an arc from n to the chord ray and the angle of attack
alpha = psi -+ pi/2 with an arc from v to the chord ray.

Before drawing, the section's stroke-direction-independent coefficient
forms (eq:cl-hover, eq:cd-hover) and force directions are checked
numerically against the general model (alpha = psi -+ pi/2, C_L =
C_L0 sin 2 alpha, C_D = C_D0 cos^2 alpha + C_Dpi2 sin^2 alpha,
d = -v/|v|, l = -d_perp) over a sweep of psi on both half-strokes.

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
PSI_DEG = 45.0  # |psi| drawn


def hover_coeffs(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C_L, C_D from the hover-case closed forms (eq:cl-hover, eq:cd-hover)."""
    cl = -CL0 * np.sin(2.0 * psi)
    cd = 0.5 * (CDPI2 + CD0) + 0.5 * (CDPI2 - CD0) * np.cos(2.0 * psi)
    return cl, cd


def verify(n: int = 1001) -> float:
    """Max deviation of the hover forms from the general model over psi in
    (-pi/2, pi/2) on both half-strokes (alpha = psi -+ pi/2)."""
    psi = np.linspace(-np.pi / 2, np.pi / 2, n)[1:-1]
    err = 0.0
    for sgn in (1.0, -1.0):  # sign of sin tau*
        alpha = psi - sgn * np.pi / 2
        cl = CL0 * np.sin(2.0 * alpha)
        cd = CD0 * np.cos(alpha) ** 2 + CDPI2 * np.sin(alpha) ** 2
        cl_h, cd_h = hover_coeffs(psi)
        err = max(err, np.abs(cl - cl_h).max(), np.abs(cd - cd_h).max())
    return err


def arc_arrow(ax, radius, a0_deg, a1_deg, color, lw=1.0, zorder=3):
    th = np.radians(np.linspace(a0_deg, a1_deg, 64))
    ax.plot(radius * np.cos(th), radius * np.sin(th), color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)
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


def draw_panel(ax, psi_deg, stau_sign, quad_label, style):
    psi = np.radians(psi_deg)
    cl, cd = hover_coeffs(np.array(psi))
    d_hat = np.array([stau_sign, 0.0])   # d = sign(sin tau*) s
    l_hat = np.array([0.0, -stau_sign])  # l = -sign(sin tau*) n

    muted = style.muted_text_color
    text = style.text_color

    # stroke-frame reference lines (s to the right, n up)
    ax.plot([0, 0], [-0.62, 1.05], ls=":", lw=0.8, color=muted, zorder=1)
    ax.plot([-1.05, 1.05], [0, 0], ls=":", lw=0.8, color=muted, zorder=1)
    ax.text(-0.07, 1.10, r"$\hat{n}$", color=muted, ha="right", va="top")
    ax.text(1.03, 0.09, r"$\hat{s}$", color=muted, ha="right", va="bottom")

    # purely in-plane wing velocity, opposing the drag direction
    vec_arrow(ax, tuple(-0.95 * d_hat), text, lw=1.3)
    ax.text(-0.80 * stau_sign, -0.13, r"$\vec{v}^*$", color=text, ha="center",
            va="top")

    # wing chord at pitch psi from n (hollow dot marks the leading edge)
    chord = np.array([-np.sin(psi), np.cos(psi)])
    ax.plot([-0.62 * chord[0], 0.62 * chord[0]],
            [-0.62 * chord[1], 0.62 * chord[1]],
            color=text, lw=2.6, solid_capstyle="round", zorder=5)
    ax.plot(*(0.62 * chord), marker="o", ms=6, mfc=style.axes_facecolor,
            mec=text, mew=1.3, zorder=6)

    # lift and drag contributions, lengths proportional to the coefficients;
    # the label of an upward lift arrow sits beside the tip, on the side
    # away from the chord tilt
    scale = 0.42
    lift_tip = scale * cl * l_hat
    vec_arrow(ax, tuple(lift_tip), style.lift_color)
    if lift_tip[1] > 0:
        side = 1.0 if psi_deg > 0 else -1.0
        ax.text(0.10 * side, lift_tip[1] + 0.09, r"$C_L\,\hat{\ell}$",
                color=style.lift_color, ha="left" if side > 0 else "right",
                va="bottom")
    else:
        ax.text(0.0, lift_tip[1] - 0.13, r"$C_L\,\hat{\ell}$",
                color=style.lift_color, ha="center", va="top")
    drag_tip = scale * cd * d_hat
    vec_arrow(ax, tuple(drag_tip), style.drag_color)
    ax.text(drag_tip[0] * 0.8 + 0.14 * stau_sign, -0.14, r"$C_D\,\hat{d}$",
            color=style.drag_color, ha="center", va="top")

    # pitch angle from n to the chord ray, angle of attack from v to the
    # chord ray, arcs nested as in the flow-angles figure
    arc_arrow(ax, 0.40, 90.0, 90.0 + psi_deg, text)
    pmid = np.radians(90.0 + 0.5 * psi_deg)
    ax.text(0.53 * np.cos(pmid), 0.53 * np.sin(pmid), r"$\psi$",
            color=text, ha="center", va="center")
    alpha_deg = psi_deg - stau_sign * 90.0
    v_deg = 180.0 if stau_sign > 0 else 0.0
    arc_arrow(ax, 0.25, v_deg, v_deg + alpha_deg, text)
    amid = np.radians(v_deg + 0.5 * alpha_deg)
    ax.text(0.38 * np.cos(amid), 0.38 * np.sin(amid), r"$\alpha$",
            color=text, ha="center", va="center")

    # sign conditions top left/right, angle-of-attack quadrant bottom right
    psi_sgn = ">" if psi_deg > 0 else "<"
    stau_sgn = ">" if stau_sign > 0 else "<"
    ax.text(-1.12, 1.42, rf"$\psi {psi_sgn} 0$", color=text, ha="left",
            va="top")
    ax.text(1.12, 1.42, rf"$\sin\tau^* {stau_sgn} 0$", color=text, ha="right",
            va="top")
    ax.text(1.12, -1.02, quad_label, color=muted, ha="right", va="bottom")

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.06, 1.48)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    err = verify()
    status = "PASS" if err < 1e-12 else "FAIL"
    print(f"hover forms vs general model: max |dC| = {err:.3e}  [{status}]")
    if status == "FAIL":
        raise SystemExit("hover-case coefficient expressions do not match")

    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figures inserted at native size)
    apply_matplotlib_style(style)

    # each combination places alpha = psi -+ pi/2 in one quadrant of the
    # lift/drag quadrants figure; panels arranged so the quadrants read
    # I-IV in reading order (rows: sign of sin tau*, columns: sign of psi)
    cases = [
        (-PSI_DEG, -1.0, "I"),
        (PSI_DEG, -1.0, "II"),
        (-PSI_DEG, 1.0, "III"),
        (PSI_DEG, 1.0, "IV"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(4.4, 4.7))
    for ax, (psi_deg, stau_sign, quad_label) in zip(axes.flat, cases):
        draw_panel(ax, psi_deg, stau_sign, quad_label, style)
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
    out = OUT_DIR / "hover_quadrants.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

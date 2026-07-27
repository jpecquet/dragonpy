"""
Verification figure for the hover (J = 0) angle-of-attack expression

    alpha = sign(cos tau*) psi + pi/2                       (report eq:alpha, J = 0)

Four panels, one per sign combination of (cos tau*, psi), drawn in the stroke
basis (s to the right, n up). Each panel shows the wing chord (dot = the
psi-reference end, i.e. the ray at angle psi from n), the wing velocity
v* = s0* omega* cos tau* s, the alpha arc measured from v* to the chord ray
(counterclockwise on the +s stroke, clockwise on the -s stroke, per the two
branches alpha = psi - chi and alpha = chi - psi), and the resulting lift and
drag forces with the sign given by C_L = C_L0 sin 2 alpha (F_n* > 0 <=> +n)
and drag opposing the motion.

Before drawing, the closed form is checked numerically against the trusted
blade-element implementation (`feasibility._wing_force_vec` in translating
mode, which is exactly the report's 2D model): the instantaneous (F_s*, F_n*)
from

    F_n* = 1/2 (A_w*/m*) v*^2 C_L(alpha)
    F_s* = -1/2 (A_w*/m*) v*^2 C_D(alpha) sign(cos tau*)

must match the code at every phase, for several psi0 (all four sign combos).

Output: light-mode figure in summer_paper/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from feasibility import (  # noqa: E402
    R_HINGE_RIGHT, Params, _wing_force_vec, drag_coeff, lift_coeff, replace,
)
from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
PSI_DEG = 30.0  # |psi| drawn in the panels


def closed_form_forces(p: Params, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous (F_s*, F_n*) from the report's J = 0 closed form."""
    omega = 2.0 * np.pi * p.wing_frequency
    psi = p.psi0 + p.psi1 * np.sin(theta - p.delta0)
    sgn = np.sign(np.cos(theta))
    alpha = sgn * psi + np.pi / 2
    v = p.Lw * p.phi1 * omega * np.abs(np.cos(theta))  # translating-mode tip speed
    q = 0.5 * p.aw_over_m * v ** 2
    return -q * drag_coeff(alpha) * sgn, q * lift_coeff(alpha)


def verify(n_phase: int = 257) -> float:
    """Max |closed form - blade-element code| over phase, for several psi0."""
    base = Params(translating=True, gamma0=0.0, delta0=np.pi / 2,
                  psi1=np.radians(45.0))
    theta = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    worst = 0.0
    for psi0_deg in (-20.0, 0.0, 20.0):
        p = replace(base, psi0=np.radians(psi0_deg))
        omega = 2.0 * np.pi * p.wing_frequency
        # Right wing at gamma0 = 0: stroke tangent = +x, normal = +z, so the
        # body-frame (F_x, F_z) are the stroke-frame (F_s*, F_n*).
        F = _wing_force_vec(p, +1, R_HINGE_RIGHT, theta, omega)
        Fs, Fn = closed_form_forces(p, theta)
        worst = max(worst,
                    np.abs(F[:, 0] - Fs).max(),
                    np.abs(F[:, 2] - Fn).max(),
                    np.abs(F[:, 1]).max())
    return worst


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


def draw_panel(ax, stroke_sign, psi_deg, style, first):
    psi = np.radians(psi_deg)
    alpha_deg = stroke_sign * psi_deg + 90.0  # the expression under test
    cl = 1.5 * np.sin(2.0 * np.radians(alpha_deg))
    cd = 0.1 * np.cos(np.radians(alpha_deg)) ** 2 + 2.0 * np.sin(np.radians(alpha_deg)) ** 2

    muted = style.muted_text_color
    text = style.text_color

    # stroke basis reference lines (s horizontal, n vertical)
    ax.plot([0, 0], [-0.8, 1.05], ls=":", lw=0.8, color=muted, zorder=1)
    ax.plot([-1.05, 1.05], [0, 0], ls=":", lw=0.8, color=muted, zorder=1)
    ax.text(0.07, 1.02, r"$\hat{n}$", color=muted, ha="left", va="top")
    if first:
        ax.text(1.03, -0.09, r"$\hat{s}$", color=muted, ha="right", va="top")

    # wing chord; the dot marks the psi-reference end (ray at psi from n)
    chord = np.array([-np.sin(psi), np.cos(psi)])
    ax.plot([-0.62 * chord[0], 0.62 * chord[0]],
            [-0.62 * chord[1], 0.62 * chord[1]],
            color=text, lw=2.6, solid_capstyle="round", zorder=5)
    ax.plot(*(0.62 * chord), marker="o", ms=6, color=text, zorder=6)

    # wing velocity along the stroke axis
    vx = 0.95 * stroke_sign
    vec_arrow(ax, (vx, 0.0), text, lw=1.3)
    ax.text(0.68 * stroke_sign, -0.13, r"$\vec{v}^{\,*}$", color=text,
            ha="center", va="top")

    # alpha arc, always sweeping counterclockwise (positive alpha): from v*
    # to the chord ray on the +s stroke (alpha = psi - chi), from the chord
    # ray to v* on the -s stroke (alpha = chi - psi)
    if stroke_sign > 0:
        a0, a1 = 0.0, 90.0 + psi_deg
    else:
        a0, a1 = 90.0 + psi_deg, 180.0
    arc_arrow(ax, 0.40, a0, a1, text, lw=1.0)
    amid = np.radians(0.5 * (a0 + a1))
    ax.text(0.58 * np.cos(amid), 0.58 * np.sin(amid), r"$\alpha$",
            color=text, ha="center", va="center")

    # psi arc: from n to the chord ray (CCW positive), reaching one dot
    # radius past the dot centre so the arrowhead terminates on the leading
    # edge, drawn above everything
    dot_r = 0.067  # rendered radius of the 6 pt chord-end marker, data units
    arc_arrow(ax, 0.62 + dot_r, 90.0, 90.0 + psi_deg, muted, lw=0.9,
              zorder=10)
    pmid = np.radians(90.0 + 0.5 * psi_deg + np.sign(psi_deg) * 8.0)
    ax.text(0.90 * np.cos(pmid), 0.90 * np.sin(pmid), r"$\psi$",
            color=muted, ha="center", va="center")

    # lift (+-n per the sign of C_L = C_L0 sin 2 alpha) and drag (opposes v*)
    scale = 0.42
    vec_arrow(ax, (0.0, scale * cl), style.lift_color)
    vec_arrow(ax, (-scale * cd * stroke_sign, 0.0), style.drag_color)
    # lift label goes on whichever side of the arrow the chord is not
    side = np.sign(cl) * np.sign(psi_deg)
    ax.text(0.14 * side, scale * cl + 0.14 * np.sign(cl),
            r"$\vec{F}_L^{\,*}$", color=style.lift_color,
            ha="left" if side > 0 else "right", va="center")
    ax.text(-scale * cd * stroke_sign * 0.8, 0.14, r"$\vec{F}_D^{\,*}$",
            color=style.drag_color,
            ha="center", va="bottom")

    # case and formula result
    sgn_tau = ">" if stroke_sign > 0 else "<"
    sgn_psi = "+" if psi_deg > 0 else "-"
    quadrant = "I" if alpha_deg < 90.0 else "II"
    ax.text(0, -1.06, rf"$\cos\tau^* {sgn_tau} 0,\ \psi = {sgn_psi}{abs(psi_deg):.0f}^\circ$",
            color=text, ha="center", va="top")
    ax.text(0, -1.32,
            rf"$\alpha = {alpha_deg:.0f}^\circ$ ({quadrant})",
            color=text, ha="center", va="top")

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.62, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    err = verify()
    status = "PASS" if err < 1e-12 else "FAIL"
    print(f"closed form vs blade-element code: max |dF| = {err:.3e}  [{status}]")
    if status == "FAIL":
        raise SystemExit("closed-form alpha expression does not match the model")

    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figures inserted at native size)
    apply_matplotlib_style(style)

    # panels: forward stroke (cos tau* > 0) with psi < 0 / > 0, then the
    # return stroke (cos tau* < 0) with psi > 0 / < 0
    cases = [(+1, -PSI_DEG), (+1, +PSI_DEG), (-1, +PSI_DEG), (-1, -PSI_DEG)]

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.6))
    for i, (ax, (stroke_sign, psi_deg)) in enumerate(zip(axes, cases)):
        draw_panel(ax, stroke_sign, psi_deg, style, first=(i == 0))
    for x in (0.265, 0.5, 0.735):
        fig.add_artist(plt.Line2D([x, x], [0.06, 0.98], color=style.grid_color,
                                  lw=0.6, transform=fig.transFigure))
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.0,
                        wspace=0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "alpha_quadrants.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""
Definition figure for the force direction angle beta (eq:beta).

Single panel in the style of the hover_beta_gamma left panel: horizontal
stroke plane (gamma = 0), full-wingbeat overlay of wing snapshots. Drawn
at a positive mean pitch (psi0 = 20 deg, psi1 = psi1_opt), so the mean
force leans toward -s and beta, measured from the stroke-plane normal n
with the same signed sense as psi and chi (positive counterclockwise), is
positive. The mean force is resolved into its stroke-frame components
q* C_Fn n (lift color) and q* C_Fs s (drag color) with dashed projection
lines, illustrating the origin of the minus sign in eq:beta: here
C_Fs < 0 and beta > 0. The wing snapshots are drawn faded (alpha) so the
in-plane component arrow along the stroke axis stays visible.

The force direction is computed from the hover closed forms (eq:cfn,
eq:cfs); the drawn magnitude is schematic.

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
PSI0_DEG = 20.0          # mean pitch drawn (positive -> positive beta)
PSI1_DEG = 51.0          # hover-optimal pitch amplitude
N_SNAP = 7               # snapshots per half-stroke, as in hover_beta_gamma
S0_DRAW = 1.0            # drawn stroke amplitude
CHORD = 0.50             # drawn chord length
WING_ALPHA = 0.35        # fade so the in-plane component stays visible
F_LEN = 1.25             # drawn mean-force length (schematic)

_TAU = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)


def hover_cf(psi0, psi1):
    """Hover force coefficients (C_Fn, C_Fs) from eq:cfn / eq:cfs."""
    psi = psi0 + psi1 * np.sin(_TAU)
    w = np.sin(_TAU) * np.abs(np.sin(_TAU))
    cfn = 2.0 * CL0 * np.mean(np.sin(2.0 * psi) * w)
    cfs = DELTA_CD * np.mean(np.cos(2.0 * psi) * w)
    return cfn, cfs


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


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figure inserted at native size)
    apply_matplotlib_style(style)
    muted = style.muted_text_color
    text = style.text_color

    psi0 = np.radians(PSI0_DEG)
    psi1 = np.radians(PSI1_DEG)
    cfn, cfs = hover_cf(psi0, psi1)
    assert cfn > 0.0 and cfs < 0.0
    beta = -np.arctan(cfs / cfn)
    print(f"psi0 = {PSI0_DEG:.0f} deg, psi1 = {PSI1_DEG:.0f} deg: "
          f"C_Fn = {cfn:.3f}, C_Fs = {cfs:.3f}, "
          f"beta = {np.degrees(beta):.1f} deg")

    fig, ax = plt.subplots(figsize=(3.3, 2.9), constrained_layout=True)
    s_hat = np.array([1.0, 0.0])
    n_hat = np.array([0.0, 1.0])

    # stroke axis and stroke-plane normal, hover_beta_gamma style
    ax.plot([-1.45, 1.45], [0.0, 0.0], ls=":", lw=0.8, color=muted, zorder=1)
    ax.text(1.58, 0.0, r"$\hat{s}$", color=muted, ha="left", va="center")
    ax.plot([0.0, 0.0], [0.0, 1.45], ls="--", lw=0.9, color=muted, zorder=1)
    ax.text(0.0, 1.58, r"$\hat{n}$", color=muted, ha="center", va="center")

    # one full wingbeat of faded wing snapshots: -s half behind and darker
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
                                 ec=style.wing_edge_color, lw=0.9,
                                 alpha=WING_ALPHA, zorder=zo))
    ax.plot(0.0, 0.0, marker="o", ms=5, color=text, zorder=7)

    # mean force and its stroke-frame components (drawn length schematic)
    f_tip = F_LEN * np.array([cfs, cfn]) / np.hypot(cfn, cfs)
    fn_tip = np.array([0.0, f_tip[1]])
    fs_tip = np.array([f_tip[0], 0.0])
    for corner in (fn_tip, fs_tip):  # dashed projections completing the rectangle
        ax.plot([f_tip[0], corner[0]], [f_tip[1], corner[1]],
                ls=(0, (2, 2)), lw=0.7, color=muted, zorder=4)
    vec_arrow(ax, fn_tip, style.lift_color, lw=1.6, zorder=5)
    ax.text(0.09, fn_tip[1] - 0.05, r"$q^* C_{F^*_n} \hat{n}$",
            color=style.lift_color, ha="left", va="center")
    vec_arrow(ax, fs_tip, style.drag_color, lw=1.6, zorder=5)
    ax.text(fs_tip[0] - 0.02, -0.16, r"$q^* C_{F^*_s} \hat{s}$",
            color=style.drag_color, ha="center", va="top")
    vec_arrow(ax, f_tip, text, lw=2.0, zorder=6)
    ax.text(f_tip[0] - 0.07, f_tip[1] + 0.10, r"$\langle \vec{F}^* \rangle$",
            color=text, ha="right", va="center")

    # beta, from n to the mean force direction (positive, counterclockwise)
    f_deg = np.degrees(np.arctan2(f_tip[1], f_tip[0]))
    arc_arrow(ax, 0.60, 90.0, f_deg, text)
    bmid = np.radians(0.5 * (90.0 + f_deg))
    ax.text(0.74 * np.cos(bmid), 0.74 * np.sin(bmid), r"$\beta$",
            color=text, ha="center", va="center")

    ax.set_xlim(-1.55, 1.85)
    ax.set_ylim(-0.52, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "beta_definition.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

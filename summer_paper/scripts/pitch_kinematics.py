"""
Pitch kinematics illustration for the Wing Kinematics subsection.

Five panels: the sinusoidal pitch law psi = psi0 + psi1 cos(theta - delta0)
(theta = omega0 t + phi the flapping phase, psi0 = 0) for four values of the
pitch phase lead delta0, plus the square-wave simplification. Each panel shows
the two halves of the flapping cycle s = s0 cos(theta) as two rows of wing
snapshots along the stroke axis (s to the right): the top row is the second
half-stroke, theta in [pi, 2 pi], wing moving along +s; the bottom row is
the first half-stroke, theta in [0, pi], moving along -s. Arrows on the
dashed stroke lines indicate the direction of motion. Chord orientation is
computed from the report's equations (c = -sin(psi) s + cos(psi) n), with the
hollow dot marking the leading edge (the c end), as in the lift/drag quadrant
figure. Snapshots are taken at equal time increments (at the midpoints of
seven equal intervals per half-stroke, so the two rows tile the cycle without
repeating the reversal instants), crowding near the stroke reversals where
the flapping motion slows.

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

PSI1 = np.radians(45.0)  # pitch amplitude drawn (psi0 = 0)
N_SNAP = 7               # snapshots per half-stroke, centered in equal dt bins
THETA_TOP = (np.arange(N_SNAP) + 0.5) * np.pi / N_SNAP   # first half
THETA_BOT = THETA_TOP + np.pi                            # second half
CHORD_HALF = 0.26
ROW_GAP = 0.75    # vertical gap between the two half-stroke rows
LINE_END = 1.45   # dashed stroke line half-length
BLOCK_DX = 1.85   # panel column offset from figure centerline
BLOCK_DY = 2.1    # panel row pitch


def draw_chord(ax, hinge, psi, style):
    c_hat = np.array([-np.sin(psi), np.cos(psi)])  # chord direction in (s, n)
    head = hinge + CHORD_HALF * c_hat
    tail = hinge - CHORD_HALF * c_hat
    ax.plot([tail[0], head[0]], [tail[1], head[1]], color=style.text_color,
            lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot(*head, marker="o", ms=4.5, mfc=style.axes_facecolor,
            mec=style.text_color, mew=1.1, zorder=4)


def draw_half_row(ax, center, s_positions, psis, motion_sign, style):
    """One half-stroke: dashed stroke line with a motion arrow, chords."""
    x0, y0 = center
    muted = style.muted_text_color
    ax.plot([x0 - LINE_END, x0 + LINE_END], [y0, y0], ls=(0, (4, 3)), lw=0.8,
            color=muted, zorder=1)
    tip = x0 + motion_sign * (LINE_END + 0.22)
    ax.annotate("", xy=(tip, y0), xytext=(tip - motion_sign * 0.15, y0),
                arrowprops=dict(arrowstyle="-|>", color=muted, lw=0.8,
                                mutation_scale=9, shrinkA=0, shrinkB=0),
                zorder=1)
    for s, psi in zip(s_positions, psis):
        draw_chord(ax, np.array([x0 + s, y0]), psi, style)


def draw_block(ax, center, title, psi_first, psi_second, style):
    """One panel: title, second half-stroke (+s motion), first (-s motion)."""
    x0, y0 = center
    ax.text(x0, y0 + 0.52, title, color=style.text_color, ha="center",
            va="bottom")
    draw_half_row(ax, (x0, y0), np.cos(THETA_BOT), psi_second, +1, style)
    draw_half_row(ax, (x0, y0 - ROW_GAP), np.cos(THETA_TOP), psi_first, -1,
                  style)


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figure inserted at native size)
    apply_matplotlib_style(style)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    cases = [
        (0.0, r"$\delta_0 = 0$", (-BLOCK_DX, 0.0)),
        (np.pi / 2, r"$\delta_0 = \dfrac{\pi}{2}$", (BLOCK_DX, 0.0)),
        (-np.pi / 2, r"$\delta_0 = -\dfrac{\pi}{2}$", (-BLOCK_DX, -BLOCK_DY)),
        (np.pi, r"$\delta_0 = \pi$", (BLOCK_DX, -BLOCK_DY)),
    ]
    for delta0, title, center in cases:
        draw_block(ax, center, title,
                   PSI1 * np.cos(THETA_TOP - delta0),
                   PSI1 * np.cos(THETA_BOT - delta0), style)

    # square wave: psi0 + psi1 through the first half, psi0 - psi1 through
    # the second (the delta0 = pi/2 mid-stroke pitch held constant)
    draw_block(ax, (0.0, -2 * BLOCK_DY), "constant pitch",
               np.full(N_SNAP, PSI1), np.full(N_SNAP, -PSI1), style)

    ax.set_xlim(-3.55, 3.55)
    ax.set_ylim(-2 * BLOCK_DY - ROW_GAP - 0.42, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pitch_kinematics.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

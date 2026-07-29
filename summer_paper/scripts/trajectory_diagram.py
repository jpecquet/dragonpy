"""
Control-law diagram for the prescribed-trajectory use case.

Companion to the hover and pursuit diagrams, same style. The prescribed
path (trajectory color) descends from the upper left and levels off; the
reference point (dot) travels along it with velocity u_r, the path
tangent scaled by the speed schedule, attached to the point; the path
is solid where already traveled and thin/dashed downstream. The body is not constrained to the path (no position
feedback): it trails the reference with velocity u lagging the
leveling-off of the path, cutting slightly inside the bend. The
reference velocity is copied at the body (same color, linking the two
visually) to close the velocity triangle with the error u_r - u, and
the force demand of eq:fdes composes tip to tail the weight
compensation z_hat with the amplified error K (u_r - u).

The wing overlay is drawn at the blended stroke-plane angle of
eq:gamma-schedule for the drawn advance ratio J = 0.2, as in the
pursuit diagram. Velocity and force arrow lengths are schematic.

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

J_DRAW = 0.2            # advance ratio drawn (cruise of the trajectory run)
JC = 0.35               # gamma blend constant (eq:gamma-schedule)
GAMMA_HOVER = -45.0     # deg, sigma_x = +1
PSI1_DEG = 41.0         # slaved psi1_opt(J = 0.2), fig:hover_opt_J
U_DIR_DEG = 11.9        # body velocity direction (slightly climbing)
U_LEN, UR_LEN = 0.815, 1.0   # drawn velocity arrow lengths (schematic)
X_BODY, OFFSET = 1.2, 0.35    # body abscissa and its cut inside the bend
A_SCALE = 1.8           # drawn length of K (u_r - u) per unit error length
Z_LEN = 0.6             # drawn weight-compensation leg
S0_DRAW = 0.75          # drawn stroke amplitude
CHORD = 0.38            # drawn chord length
N_SNAP = 7


def path_z(x):
    """Prescribed path: descends from the upper left, bottoms out, and
    turns up onto a steady climb (softplus rise, bounded slope)."""
    return (0.37 - 0.8 * np.tanh(0.75 * (x + 1.1))
            + 0.45 * np.logaddexp(0.0, (x - 2.4) / 0.5))


def path_tangent(x):
    dz = (-0.6 / np.cosh(0.75 * (x + 1.1)) ** 2
          + 0.9 / (1.0 + np.exp(-(x - 2.4) / 0.5)))
    t = np.array([1.0, dz])
    return t / np.linalg.norm(t)


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
    traj = style.trajectory_color

    # stroke-plane angle from the gamma schedule at the body velocity
    f = 1.0 - np.exp(-((J_DRAW / JC) ** 2))
    gamma_u = U_DIR_DEG - 90.0
    gamma = np.radians((1.0 - f) * GAMMA_HOVER + f * gamma_u)
    print(f"f(J={J_DRAW}) = {f:.3f}, gamma = {np.degrees(gamma):.1f} deg")
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])

    body = np.array([X_BODY, path_z(X_BODY) + OFFSET])

    fig, ax = plt.subplots(figsize=(6.4, 2.9), constrained_layout=True)

    # prescribed path, solid up to the reference point (traveled), thin
    # and dashed downstream (upcoming); the reference point sits at the
    # bottom of the dip, its velocity arrow attached to it
    xs = np.linspace(-1.0, 4.1, 600)
    x_ref = xs[np.argmin(path_z(xs))] + 1.15   # well into the upward turn
    done = xs <= x_ref
    ax.plot(xs[done], path_z(xs[done]), color=traj, lw=2.2, zorder=1)
    ax.plot(xs[~done], path_z(xs[~done]), color=traj, lw=1.1, ls="--",
            zorder=1)
    ref = np.array([x_ref, path_z(x_ref)])
    r_tan = path_tangent(x_ref)
    ax.plot(*ref, marker="o", ms=5, color=traj, zorder=6)
    ur_vec = UR_LEN * r_tan
    vec_arrow(ax, ref + ur_vec, traj, tail=tuple(ref))
    ur_base = ref + ur_vec - 0.11 * r_tan
    ax.text(ur_base[0], ur_base[1] - 0.10, r"$\vec{u}^*_r$", color=traj,
            ha="center", va="top")

    # the body's own (achieved) trajectory: leaves the reference near its
    # entry point and drifts inside the bend as the tracking error builds,
    # arriving at the body tangent to u
    u_hat = np.array([np.cos(np.radians(U_DIR_DEG)),
                      np.sin(np.radians(U_DIR_DEG))])
    p0 = np.array([-0.95, path_z(-0.95) + 0.2])
    ctrl = np.array([p0, p0 + 0.85 * path_tangent(-0.95),
                     body - 0.75 * u_hat, body])
    t = np.linspace(0.0, 1.0, 200)[:, None]
    bez = ((1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1]
           + 3 * (1 - t) * t ** 2 * ctrl[2] + t ** 3 * ctrl[3])
    ax.plot(bez[:, 0], bez[:, 1], color="#ff7f0e", lw=1.8, zorder=1)

    # stroke axis and stroke-plane normal at the body, other-diagrams style
    line = 1.45 * S0_DRAW
    ax.plot([body[0] - line * s_hat[0], body[0] + line * s_hat[0]],
            [body[1] - line * s_hat[1], body[1] + line * s_hat[1]],
            ls=":", lw=0.8, color=muted, zorder=1)
    ax.text(*(body + 1.58 * S0_DRAW * s_hat), r"$\hat{s}$", color=muted,
            ha="left", va="center")
    ax.plot([body[0], body[0] + 0.92 * S0_DRAW * n_hat[0]],
            [body[1], body[1] + 0.92 * S0_DRAW * n_hat[1]], ls="--", lw=0.9,
            color=muted, zorder=1)
    ax.text(*(body + 1.04 * S0_DRAW * n_hat), r"$\hat{n}$", color=muted,
            ha="center", va="center")

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
            center = body + S0_DRAW * np.cos(tau) * s_hat
            ax.add_patch(Ellipse(tuple(center), width=CHORD, height=0.052,
                                 angle=c_deg, fc=fc,
                                 ec=style.wing_edge_color, lw=0.9, zorder=zo))
    ax.plot(*body, marker="o", ms=5, color=text, zorder=7)

    # velocity triangle at the body: u lags the reference velocity, which
    # is copied here (same color as at the reference point)
    u_dir = np.radians(U_DIR_DEG)
    u_vec = U_LEN * np.array([np.cos(u_dir), np.sin(u_dir)])
    vec_arrow(ax, body + u_vec, text, tail=tuple(body))
    head_mid = body + u_vec - 0.055 * u_vec / np.linalg.norm(u_vec)
    ax.text(head_mid[0], head_mid[1] - 0.11, r"$\vec{u}^*$", color=text,
            ha="center", va="top")
    vec_arrow(ax, body + ur_vec, traj, tail=tuple(body))
    err = ur_vec - u_vec
    vec_arrow(ax, body + ur_vec, text, tail=tuple(body + u_vec))
    ax.text(*(body + u_vec + 0.5 * err + (0.12, 0.0)),
            r"$\vec{u}^*_r - \vec{u}^*$", color=text, ha="left", va="center")

    # force demand, tip to tail: the weight compensation, then the
    # amplified error, parallel to u_r - u
    z_leg = body + (0.0, Z_LEN)
    vec_arrow(ax, z_leg, text, tail=tuple(body))
    ax.text(body[0] + 0.07, body[1] + Z_LEN - 0.045, r"$\hat{z}$", color=text,
            ha="center", va="center", zorder=8)
    f_tip = z_leg + A_SCALE * err
    vec_arrow(ax, f_tip, text, tail=tuple(z_leg))
    ang = np.degrees(np.arctan2(err[1], err[0]))
    ang_txt = ang - 180.0 if abs(ang) > 90.0 else ang
    along = err / np.linalg.norm(err)
    perp = np.array([-along[1], along[0]])
    pos = z_leg + 0.5 * A_SCALE * err + 0.10 * along + 0.05 * perp
    ax.text(*pos, r"$K (\vec{u}^*_r - \vec{u}^*)$", color=text,
            ha="center", va="bottom", rotation=ang_txt,
            rotation_mode="anchor", transform_rotates_text=True)
    vec_arrow(ax, f_tip, text, tail=tuple(body))
    f_dir = (f_tip - body) / np.linalg.norm(f_tip - body)
    f_base = f_tip - 0.11 * f_dir
    ax.text(f_tip[0] + 0.05, f_base[1],
            r"$\langle \vec{F}^* \rangle^\text{des}$",
            color=text, ha="left", va="center")

    ax.set_xlim(-1.05, 4.25)
    ax.set_ylim(-1.12, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "trajectory_diagram.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

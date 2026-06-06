"""
Turn-pursuit trajectory plot.

Loads the turn run from `data/wing_delta/turn_dx<DX>_dz<DZ>_K3.npz` and
renders a 2x2 figure summarizing the maneuver:
  - top-left:  body trajectory in the body x-z plane, with the initial and
               post-jump target positions marked and the jump moment
               highlighted on the path.
  - top-right: |v_xz| vs t.
  - bot-left:  stroke-plane tilt gamma vs t (degrees).
  - bot-right: heading error (bearing-to-current-target minus velocity
               elevation, wrapped to (-180°, 180°]) vs t.

Time-series panels mark t_jump with a vertical line.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.capture_study import WING_FREQUENCY
from post.style import apply_matplotlib_style, figure_size, resolve_style


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data" / "wing_delta"
OUT_DIR   = REPO_ROOT / "docs" / "_static" / "media" / "maneuvering"
T_WB      = 1.0 / WING_FREQUENCY

DATA_FILE = "turn_dx50_dz20_K3.npz"
OUT_FILE  = "turn_pursuit.dark.png"


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    data = np.load(DATA_DIR / DATA_FILE)
    t        = data["t"]
    pos      = data["pos"]
    vel      = data["vel"]
    gamma    = data["gamma"]
    tgt_seq  = data["target_pos"]
    j        = int(data["jump_idx"])
    initial_target = data["initial_target"]
    jump_delta     = data["jump_delta"]
    capture_radius = float(data["capture_radius"])
    new_target = initial_target + jump_delta

    speed_xz = np.linalg.norm(vel[:, [0, 2]], axis=1)

    bearing = tgt_seq - pos
    bearing_elev = np.arctan2(bearing[:, 2], bearing[:, 0])
    vel_elev_safe = np.where(speed_xz > 1e-6,
                             np.arctan2(vel[:, 2], vel[:, 0]),
                             0.0)
    heading_err = _wrap_to_pi(bearing_elev - vel_elev_safe)

    style = resolve_style(theme="dark")
    apply_matplotlib_style(style)

    w_fig, h_fig = figure_size(0.95, width_in=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(w_fig, h_fig), dpi=300)
    ax_xz, ax_v, ax_g, ax_h = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    pre_color   = "#4aa3ff"
    post_color  = "#facc15"
    jump_color  = "#ff6b6b"

    # --- xz trajectory ---
    ax_xz.plot(pos[:j+1, 0], pos[:j+1, 2], color=pre_color, lw=1.4, label="pre-jump")
    ax_xz.plot(pos[j:, 0],  pos[j:, 2],   color=post_color, lw=1.4, label="post-jump")
    ax_xz.scatter([initial_target[0]], [initial_target[2]],
                  marker="*", s=70, color=pre_color, edgecolor="none", zorder=5,
                  label="initial target")
    ax_xz.scatter([new_target[0]], [new_target[2]],
                  marker="*", s=70, color=post_color, edgecolor="none", zorder=5,
                  label="new target")
    ax_xz.scatter([pos[j, 0]], [pos[j, 2]],
                  marker="o", s=30, color=jump_color, zorder=6,
                  label="jump moment")
    th = np.linspace(0, 2 * np.pi, 96)
    ax_xz.plot(new_target[0] + capture_radius * np.cos(th),
               new_target[2] + capture_radius * np.sin(th),
               color=style.axes_edge_color, lw=0.5, ls=":")
    ax_xz.set_xlabel(r"$\tilde{x}$")
    ax_xz.set_ylabel(r"$\tilde{z}$")
    ax_xz.set_aspect("equal", adjustable="datalim")

    # --- time-series panels ---
    t_norm = t / T_WB
    t_jump_norm = t[j] / T_WB

    ax_v.plot(t_norm, speed_xz, color=pre_color, lw=1.2)
    ax_v.set_ylabel(r"$|\tilde V_{xz}|$")

    ax_g.plot(t_norm, np.degrees(gamma), color=pre_color, lw=1.2)
    ax_g.set_ylabel(r"$\gamma$ (°)")
    ax_g.axhline(0, color=style.axes_edge_color, lw=0.4, ls=":")

    ax_h.plot(t_norm, np.degrees(heading_err), color=pre_color, lw=1.2)
    ax_h.set_ylabel(r"$\theta_e$ (°)")
    ax_h.axhline(0, color=style.axes_edge_color, lw=0.4, ls=":")

    for ax in (ax_v, ax_g, ax_h):
        ax.set_xlabel(r"$t / T_\mathrm{wb}$")
        ax.axvline(t_jump_norm, color=jump_color, lw=0.8, ls="--")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0.0, t_norm[-1])

    ax_xz.grid(True, alpha=0.2)
    ax_xz.legend(loc="upper left", fontsize=style.font_size * 0.85,
                 frameon=False)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / OUT_FILE
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")
    print(f"  jump at t={t[j]:.2f}s ({t_jump_norm:.1f} T_wb)")
    print(f"  capture at t={t[-1]:.2f}s ({t_norm[-1]:.1f} T_wb), "
          f"final pos={pos[-1].round(3)}")


if __name__ == "__main__":
    main()

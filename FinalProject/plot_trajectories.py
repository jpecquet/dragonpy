"""
Trajectory comparison: dragonfly xz path for both controllers, plus the
common prey path. Time-stamp markers help line up the panels with the
time-series plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from post.style import apply_matplotlib_style, figure_size, resolve_style

from scenario import SCENARIOS


HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR  = HERE / "figs"


def _stamps(t, every: float):
    """Indices of slow ticks closest to multiples of `every`."""
    targets = np.arange(every, t[-1] + 1e-9, every)
    return [int(np.argmin(np.abs(t - tt))) for tt in targets]


def _plot_one(scenario: str, style):
    cfg = SCENARIOS[scenario]
    prop = dict(np.load(DATA_DIR / f"{scenario}_proportional.npz",
                        allow_pickle=True))
    mbe  = dict(np.load(DATA_DIR / f"{scenario}_mbe.npz",
                        allow_pickle=True))

    fig, ax = plt.subplots(figsize=figure_size(0.7, width_in=6.0))

    # Prey path: same for both controllers, so just use one.
    prey_xz = prop["prey_pos"][:, [0, 2]]
    ax.plot(prey_xz[:, 0], prey_xz[:, 1],
            color=style.target_color, ls="--", lw=1.5, label="prey")

    # Dragonfly paths.
    prop_xz = prop["dfly_pos"][:, [0, 2]]
    mbe_xz  = mbe ["dfly_pos"][:, [0, 2]]
    ax.plot(prop_xz[:, 0], prop_xz[:, 1],
            color=style.trajectory_color, lw=1.8, label="proportional")
    ax.plot(mbe_xz[:,  0], mbe_xz[:,  1],
            color=style.error_color,      lw=1.8, label="MBE")

    # Time-stamps every 1 nondim time unit on the dragonfly paths.
    every = 1.0
    for traj, t, color in (
        (prop_xz, prop["t"], style.trajectory_color),
        (mbe_xz,  mbe ["t"], style.error_color),
    ):
        idxs = _stamps(t, every)
        ax.plot(traj[idxs, 0], traj[idxs, 1], "o",
                color=color, markersize=3, zorder=5)

    # Start / capture markers.
    ax.plot(cfg.prey_init_pos[0], cfg.prey_init_pos[2], "x",
            color=style.target_color, markersize=8, mew=1.5, zorder=6)
    for rec, color, label in ((prop, style.trajectory_color, "prop capture"),
                              (mbe,  style.error_color,      "MBE capture")):
        if bool(rec["captured"]):
            ax.plot(rec["dfly_pos"][-1, 0], rec["dfly_pos"][-1, 2],
                    "*", color=color, markersize=12, mew=1.0, zorder=7,
                    label=label)

    ax.set_xlabel(r"$x / L_b$")
    ax.set_ylabel(r"$z / L_b$")
    ax.set_aspect("equal")
    ax.grid(True, color=style.grid_color, alpha=0.5)
    ax.legend(loc="best", fontsize=style.font_size - 1)

    fig.tight_layout()
    out = OUT_DIR / f"{scenario}_trajectories.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_matplotlib_style(resolve_style(theme="light"))
    style = resolve_style(theme="light")
    for scenario in SCENARIOS:
        _plot_one(scenario, style)


if __name__ == "__main__":
    main()

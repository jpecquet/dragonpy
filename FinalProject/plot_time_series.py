"""
Time-series of range, bearing, and stroke-plane angle gamma for both
controllers. For the MBE run, also plots filter estimates and the raw
delayed bearing/range measurements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from post.style import apply_matplotlib_style, figure_size, resolve_style

from scenario import SCENARIOS, WING_FREQUENCY


TLABEL = r"$t / T_\mathrm{wb}$"


HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR  = HERE / "figs"


def _true_bearing(dfly_pos, prey_pos):
    dx = prey_pos[:, 0] - dfly_pos[:, 0]
    dz = prey_pos[:, 2] - dfly_pos[:, 2]
    return np.arctan2(dz, dx)


def _true_range(dfly_pos, prey_pos):
    return np.linalg.norm(prey_pos[:, [0, 2]] - dfly_pos[:, [0, 2]], axis=1)


def _estimated_bearing(filter_mean, position_estimate):
    dx = filter_mean[:, 0] - position_estimate[:, 0]
    dz = filter_mean[:, 1] - position_estimate[:, 2]
    return np.arctan2(dz, dx)


def _estimated_range(filter_mean, position_estimate):
    dx = filter_mean[:, 0] - position_estimate[:, 0]
    dz = filter_mean[:, 1] - position_estimate[:, 2]
    return np.hypot(dx, dz)


def _plot_one(scenario: str, style):
    prop = dict(np.load(DATA_DIR / f"{scenario}_proportional.npz",
                        allow_pickle=True))
    mbe  = dict(np.load(DATA_DIR / f"{scenario}_mbe.npz",
                        allow_pickle=True))

    t_prop = prop["t"] * WING_FREQUENCY
    t_mbe  = mbe ["t"] * WING_FREQUENCY

    fig, axes = plt.subplots(
        1, 3, figsize=figure_size(0.32, width_in=10.0),
    )
    ax_r, ax_b, ax_g = axes

    # --- range ---
    ax_r.plot(t_prop, _true_range(prop["dfly_pos"], prop["prey_pos"]),
              color=style.trajectory_color, label="prop, true")
    ax_r.plot(t_mbe,  _true_range(mbe["dfly_pos"],  mbe["prey_pos"]),
              color=style.error_color, label="MBE, true")
    ax_r.plot(t_mbe, _estimated_range(mbe["filter_mean"],
                                         mbe["position_estimate"]),
              color=style.error_color, ls=":", lw=1.2, label="MBE, estimated")
    ax_r.set_ylabel(r"range $|\hat r| / L_b$")
    ax_r.set_xlabel(r"$t / T_\mathrm{wb}$")
    ax_r.grid(True, color=style.grid_color, alpha=0.5)
    ax_r.legend(loc="best", fontsize=style.font_size - 2)

    # --- bearing ---
    ax_b.plot(t_prop, _true_bearing(prop["dfly_pos"], prop["prey_pos"]),
              color=style.trajectory_color, label="prop, true")
    ax_b.plot(t_mbe,  _true_bearing(mbe["dfly_pos"],  mbe["prey_pos"]),
              color=style.error_color, label="MBE, true")
    ax_b.plot(t_mbe, _estimated_bearing(mbe["filter_mean"],
                                           mbe["position_estimate"]),
              color=style.error_color, ls=":", lw=1.2, label="MBE, estimated")
    ax_b.plot(t_mbe, mbe["bearing_meas"], ".",
              color=style.target_color, markersize=2.5, label="measurement")
    ax_b.set_ylabel(r"bearing $\theta_p$ (rad)")
    ax_b.set_xlabel(r"$t / T_\mathrm{wb}$")
    ax_b.grid(True, color=style.grid_color, alpha=0.5)
    ax_b.legend(loc="best", fontsize=style.font_size - 2)

    # --- stroke plane angle ---
    ax_g.plot(t_prop, prop["tilt"],
              color=style.trajectory_color, label="proportional")
    ax_g.plot(t_mbe,  mbe["tilt"],
              color=style.error_color, label="MBE")
    ax_g.set_ylabel(r"stroke plane angle $\gamma$ (rad)")
    ax_g.set_xlabel(r"$t / T_\mathrm{wb}$")
    ax_g.grid(True, color=style.grid_color, alpha=0.5)
    ax_g.legend(loc="best", fontsize=style.font_size - 2)

    fig.tight_layout()
    out = OUT_DIR / f"{scenario}_time_series.png"
    fig.savefig(out, dpi=180)
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

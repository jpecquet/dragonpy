"""
Filter consistency plots: error and 2-sigma bounds for each of the four
prey-state components (x_p, z_p, vx_p, vz_p). Uses the MAE 6760
plot_estimator helper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from post.style import apply_matplotlib_style, figure_size, resolve_style

from plot_estimator import plot_estimator
from scenario import SCENARIOS, WING_FREQUENCY


HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR  = HERE / "figs"


# Map from filter-state index to (true-state array, label).
COMPONENTS = [
    (0, "prey_pos", 0, r"$x_p$"),
    (1, "prey_pos", 2, r"$z_p$"),
    (2, "prey_vel", 0, r"$\dot x_p$"),
    (3, "prey_vel", 2, r"$\dot z_p$"),
]


def _plot_one(scenario: str):
    mbe = dict(np.load(DATA_DIR / f"{scenario}_mbe.npz", allow_pickle=True))
    t   = mbe["t"] * WING_FREQUENCY
    fm  = mbe["filter_mean"]
    fc  = mbe["filter_cov"]

    fig, axes = plt.subplots(
        2, 2, figsize=figure_size(0.85, width_in=6.5), sharex=True,
    )
    axes = axes.ravel()

    for ax, (idx, true_key, true_axis, label) in zip(axes, COMPONENTS):
        xest  = fm[:, idx]
        Pest  = fc[:, idx, idx]
        xtrue = mbe[true_key][:, true_axis]
        plot_estimator(
            t, xest, Pest, xtrue,
            plot_type="error", ax=ax, legend=False,
        )
        ax.set_ylabel(f"error in {label}")
        ax.set_xlabel("")                  # override plot_estimator's default

    axes[-1].set_xlabel(r"$t / T_\mathrm{wb}$")
    axes[-2].set_xlabel(r"$t / T_\mathrm{wb}$")

    # One shared legend at the top.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / f"{scenario}_filter_consistency.png"
    fig.savefig(out, dpi=180)
    print(f"saved {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_matplotlib_style(resolve_style(theme="light"))
    for scenario in SCENARIOS:
        _plot_one(scenario)


if __name__ == "__main__":
    main()

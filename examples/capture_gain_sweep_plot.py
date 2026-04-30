"""
Composite half-disk capture-map figures for the docs "Stationary Prey Capture"
gain-sweep section.

For each of K = 1, 2, 3, 4 (data under
`data/capture/k{K}_tau1Twb_hind0.5pi_fov360/result.npz`), render a 2x2 panel:

  +------+------+
  | K=1  | K=2  |
  +------+------+
  | K=3  | K=4  |
  +------+------+

Two figure series are produced (each in light and dark variants):

  1. avg:    capture rate averaged over initial stroke-plane tilts.
  2. per-tilt: same panel, sliced to one of the 12 initial-tilt grid values
     (0, 30, ..., 330 deg). One figure per tilt — the docs page exposes them
     via a tab-set so the reader can pick which tilt to inspect.

Outputs land at `docs/_static/media/capture/`.
"""

from pathlib import Path

import numpy as np

from examples.capture_plot import draw_half_disk, load_result


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "capture"
OUT_DIR   = REPO_ROOT / "docs" / "_static" / "media" / "capture"

GAINS = (1, 2, 3, 4)
CAPTURE_RADIUS = 0.5
PANEL_LAYOUT = [
    # (row, col, K)
    (0, 0, 1),
    (0, 1, 2),
    (1, 0, 3),
    (1, 1, 4),
]


def _result_path(k: int) -> Path:
    return DATA_ROOT / f"k{k}_tau1Twb_hind0.5pi_fov360" / "result.npz"


def plot_gain_grid(
    results: dict[int, "object"],
    tilt:    float | None,
    savepath: Path,
    theme:    str,
):
    import matplotlib.pyplot as plt

    from post.style import apply_matplotlib_style, figure_size, resolve_style

    style = resolve_style(theme=theme)
    apply_matplotlib_style(style)

    w, h = figure_size(1.0, width_in=9.5)
    fig, axes = plt.subplots(
        2, 2, figsize=(w, h), dpi=200,
        subplot_kw={"projection": "polar"},
    )

    mesh = None
    mode_label = ""
    for row, col, k in PANEL_LAYOUT:
        ax = axes[row, col]
        m, mode_label = draw_half_disk(
            ax, results[k], capture_radius=CAPTURE_RADIUS, tilt=tilt,
            style=style,
        )
        mesh = m
        ax.set_title(fr"$K = {k}$", fontsize=style.font_size, pad=8)

    # Single shared colorbar on the right.
    fig.subplots_adjust(left=0.05, right=0.86, top=0.93, bottom=0.05,
                        wspace=0.20, hspace=0.30)
    cax = fig.add_axes([0.89, 0.18, 0.020, 0.64])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(
        "capture rate" if tilt is None else "captured (1 = yes)",
        fontsize=style.font_size,
    )

    fig.savefig(savepath, dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {k: load_result(_result_path(k)) for k in GAINS}

    # Discover the tilt grid from any one of the results — they all share it.
    tilts = results[GAINS[0]].tilts
    tilt_degs = [int(round(np.degrees(float(t)))) for t in tilts]

    out = OUT_DIR / f"gain_sweep_avg_rcap{CAPTURE_RADIUS:g}.dark.png"
    plot_gain_grid(results, tilt=None, savepath=out, theme="dark")
    print(f"saved {out}")

    for tilt, deg in zip(tilts, tilt_degs):
        out = (OUT_DIR /
               f"gain_sweep_tilt{deg:03d}_rcap{CAPTURE_RADIUS:g}.dark.png")
        plot_gain_grid(results, tilt=float(tilt),
                       savepath=out, theme="dark")
        print(f"saved {out}")


if __name__ == "__main__":
    main()

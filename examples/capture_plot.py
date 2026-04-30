"""
Half-disk capture-rate heatmap.

Loads a `CaptureStudyResult` saved by `examples/capture_study.py` and renders
the capture map over (R, theta) in the (x, z) plane. The dragonfly sits at the
origin facing +x, gravity is along -z, so theta = 0 is straight ahead, theta =
+pi/2 is straight up, theta = -pi/2 is straight down.

Two reduction modes:
  * `tilt=None`        - average over all initial stroke-plane tilts. Each cell
                         is the fraction of tilt trials that captured.
  * `tilt=<float rad>` - slice to the nearest initial tilt. Each cell is binary
                         (hit / miss).
"""

import argparse
from pathlib import Path

import numpy as np

from dragonpy.studies import CaptureStudyResult, captured_mask


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = (
    REPO_ROOT / "data" / "capture" / "k2_tau1Twb_hind0.5pi_fov360" / "result.npz"
)


def load_result(path: Path) -> CaptureStudyResult:
    """Load a saved capture-study result from `examples/capture_study.py`."""
    d = np.load(path)
    return CaptureStudyResult(
        radii=d["radii"],
        thetas=d["thetas"],
        tilts=d["tilts"],
        final_position=d["final_position"],
        final_velocity=d["final_velocity"],
        final_time=d["final_time"],
        crossed=d["crossed"],
        target_position=d["target_position"],
    )


def _edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    d_left  = x[1]  - x[0]
    d_right = x[-1] - x[-2]
    interior = 0.5 * (x[:-1] + x[1:])
    return np.concatenate([[x[0] - 0.5 * d_left], interior, [x[-1] + 0.5 * d_right]])


def draw_half_disk(
    ax,
    result:         CaptureStudyResult,
    capture_radius: float,
    tilt:           float | None,
    *,
    style,
    cmap:           str = "inferno",
):
    """Draw the wedge (mesh + outline + theta ticks) onto an existing polar
    axes. Returns the QuadMesh and a short mode label.

    The caller owns the figure, colorbar, and title — this helper only paints
    the wedge so it can be reused across single-panel and multi-panel layouts.
    """
    captured = captured_mask(result, capture_radius)         # (nR, nTh, nTilt) bool

    if tilt is None:
        values = captured.mean(axis=2)                        # (nR, nTh) in [0, 1]
        mode_label = "avg over initial tilts"
    else:
        k = int(np.argmin(np.abs(result.tilts - float(tilt))))
        values = captured[:, :, k].astype(float)              # (nR, nTh) {0, 1}
        mode_label = f"initial tilt = {np.degrees(result.tilts[k]):.0f} deg"

    theta_edges = _edges(result.thetas)
    r_edges     = _edges(result.radii)

    mesh = ax.pcolormesh(
        theta_edges, r_edges, values,
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        shading="flat",
    )

    ax.set_thetamin(np.degrees(theta_edges[0]))
    ax.set_thetamax(np.degrees(theta_edges[-1]))
    ax.set_theta_zero_location("E")     # 0 deg at +x (east)
    ax.set_theta_direction(1)           # CCW so +theta is up (+z)

    ax.set_rlim(0.0, r_edges[-1])
    ax.set_rticks(result.radii[::2])
    ax.set_rlabel_position(180)

    theta_min_deg = np.degrees(theta_edges[0])
    theta_max_deg = np.degrees(theta_edges[-1])
    grid_lo = int(np.ceil(theta_min_deg / 30.0)) * 30
    grid_hi = int(np.floor(theta_max_deg / 30.0)) * 30
    theta_grid = np.arange(grid_lo, grid_hi + 1, 30)
    ax.set_thetagrids(theta_grid)
    ax.tick_params(labelsize=style.font_size - 2)

    ax.grid(False)
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
        spine.set_linewidth(0)
        spine.set_color("none")
    ax.tick_params(axis="both", which="both", length=0)

    outline_color = style.axes_edge_color
    arc_theta = np.linspace(theta_edges[0], theta_edges[-1], 240)
    r_inner = r_edges[0]
    r_outer = r_edges[-1]
    ax.plot(arc_theta, np.full_like(arc_theta, r_outer),
            color=outline_color, linewidth=0.8)
    ax.plot(arc_theta, np.full_like(arc_theta, r_inner),
            color=outline_color, linewidth=0.8)
    ax.plot([theta_edges[0],  theta_edges[0]],  [r_inner, r_outer],
            color=outline_color, linewidth=0.8)
    ax.plot([theta_edges[-1], theta_edges[-1]], [r_inner, r_outer],
            color=outline_color, linewidth=0.8)

    return mesh, mode_label


def plot_half_disk(
    result:         CaptureStudyResult,
    capture_radius: float,
    tilt:           float | None = None,
    *,
    savepath:       Path | str | None = None,
    title:          str | None = None,
    theme:          str = "dark",
    cmap:           str = "inferno",
):
    """Render the capture map over the half-disk in (x, z).

    `tilt=None` averages over the initial-stroke-plane-tilt axis (continuous
    fraction in [0, 1]). A scalar `tilt` (radians) slices to the nearest
    grid value (binary hit/miss).
    """
    import matplotlib.pyplot as plt

    from post.style import apply_matplotlib_style, figure_size, resolve_style

    style = resolve_style(theme=theme)
    apply_matplotlib_style(style)

    # With aspect=equal, the polar disk is a circle of diameter
    # min(axes_w, axes_h). To make the wedge fill the figure vertically we
    # need axes_w >= axes_h, i.e. figure slightly wider than tall.
    w, h = figure_size(0.9, width_in=5.0)
    fig, ax = plt.subplots(figsize=(w, h), dpi=300,
                           subplot_kw={"projection": "polar"})

    mesh, mode_label = draw_half_disk(
        ax, result, capture_radius, tilt, style=style, cmap=cmap,
    )

    # Polar axes' transAxes maps to the wedge bbox (after thetamin/max), so
    # axes-x = 0 is the wedge's left chord and axes-x = 1 is its outer edge.
    # Place the colorbar just left of the wedge.
    cb_axes_x  = -0.04
    cb_axes_w  = 0.025
    cax = ax.inset_axes([cb_axes_x, 0.0, cb_axes_w, 1.0])
    cbar = fig.colorbar(mesh, cax=cax)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cbar.set_label(
        "capture rate" if tilt is None else "captured (1 = yes)",
        fontsize=style.font_size,
    )

    # Center the title over [cbar (with its left-side tick labels and label)
    # + wedge]. Approximate the cbar's leftmost extent (label + tick text)
    # at axes-x ~ -0.18; right edge of composite is the wedge outer at 1.0.
    auto_title = (
        f"capture map ({mode_label}), "
        rf"$r_\mathrm{{cap}} = {capture_radius:g}$"
    )
    title_axes_x = 0.5 * (-0.18 + 1.0)
    ax.set_title(title or auto_title, fontsize=style.font_size, pad=14,
                 x=title_axes_x)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.05)
    if savepath is not None:
        fig.savefig(savepath, dpi=300)
    return fig


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                   help="path to result.npz produced by capture_study.py")
    p.add_argument("--capture-radius", type=float, default=0.5,
                   help="capture radius in body lengths")
    p.add_argument("--tilt-deg", type=float, default=None,
                   help="initial stroke-plane tilt in deg; omit to average over all")
    p.add_argument("--theme", choices=("light", "dark"), default="dark")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path; default: <npz_dir>/half_disk_<tag>.png")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = load_result(args.npz)
    tilt_rad = None if args.tilt_deg is None else np.radians(args.tilt_deg)

    if args.out is None:
        if args.tilt_deg is None:
            stem = f"half_disk_avg_rcap{args.capture_radius:g}"
        else:
            stem = f"half_disk_tilt{args.tilt_deg:.0f}_rcap{args.capture_radius:g}"
        out = args.npz.parent / f"{stem}.png"
    else:
        out = args.out

    plot_half_disk(
        result,
        capture_radius=args.capture_radius,
        tilt=tilt_rad,
        savepath=out,
        theme=args.theme,
    )
    print(f"saved {out}")

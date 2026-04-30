"""
k_tilt vs sensing-delay sweep for the prey-intercept problem.

For each (k_tilt, tau) pair, runs a fixed-time interception attempt and
records the closest approach between dragonfly and prey. Produces a heatmap
of min_dist over the (tau, k_tilt) grid.

Output layout (superparameter-aware):

    data/k_tau/<tag>/sweep.npz
    data/k_tau/<tag>/sweep.png

where <tag> encodes the superparameters used for the sweep (currently just
the prey start position, e.g. `prey_x3_z3`). Morphology and other
superparameters can be added to the tag as they are introduced.
"""

import os
import time

import numpy as np

from examples.parametric import BASELINE, run_case


DATA_ROOT = "data/k_tau"


def prey_tag(prey_xyz: np.ndarray) -> str:
    x, _, z = float(prey_xyz[0]), float(prey_xyz[1]), float(prey_xyz[2])
    return f"prey_x{x:g}_z{z:g}"


def output_dir(tag: str) -> str:
    d = os.path.join(DATA_ROOT, tag)
    os.makedirs(d, exist_ok=True)
    return d


def run_sweep(
    ks: np.ndarray,
    taus: np.ndarray,
    t_end: float = 10.0,
    prey_position: np.ndarray | None = None,
    fov_half_angle: float | None = None,
    extra_overrides: dict | None = None,
):
    min_dist = np.zeros((len(ks), len(taus)))
    t_min    = np.zeros((len(ks), len(taus)))
    overrides_common: dict = {"t_end": t_end}
    if prey_position is not None:
        overrides_common["prey_position"] = np.asarray(prey_position, dtype=float)
    if fov_half_angle is not None:
        overrides_common["fov_half_angle"] = float(fov_half_angle)
    if extra_overrides:
        overrides_common.update(extra_overrides)
    for i, k in enumerate(ks):
        for j, tau in enumerate(taus):
            r = run_case(
                f"k{k:g}_tau{tau:.4f}",
                k_tilt=float(k), sensing_delay=float(tau),
                **overrides_common,
            )
            min_dist[i, j] = r["min_dist"]
            t_min[i, j]    = r["t_min_dist"]
    return min_dist, t_min


def plot_sweep(
    ks, taus, min_dist, T_wb,
    savepath: str | None = None,
    title: str | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from post.style import apply_matplotlib_style, figure_size, resolve_style

    style = resolve_style()
    apply_matplotlib_style(style)

    def _edges(x):
        x = np.asarray(x, dtype=float)
        if len(x) == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5])
        d = x[1] - x[0]
        return np.concatenate([[x[0] - d / 2], (x[:-1] + x[1:]) / 2, [x[-1] + d / 2]])

    tau_over_T = taus / T_wb
    tau_edges = _edges(tau_over_T)
    k_edges   = _edges(ks)

    fig, ax = plt.subplots(figsize=figure_size(0.8))

    vmin = max(float(min_dist.min()), 1e-3)
    vmax = float(min_dist.max())
    mesh = ax.pcolormesh(
        tau_edges, k_edges, min_dist,
        cmap=style.landscape_colormap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="flat",
    )

    levels = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
    levels = levels[(levels >= vmin) & (levels <= vmax)]
    if levels.size:
        cs = ax.contour(
            tau_over_T, ks, min_dist,
            levels=levels,
            colors=style.landscape_contour_color,
            linewidths=0.8,
            alpha=0.7,
        )
        ax.clabel(cs, fmt="%.2g", fontsize=style.font_size - 2,
                  colors=style.landscape_contour_color)

    i_min, j_min = np.unravel_index(np.argmin(min_dist), min_dist.shape)
    ax.plot(
        tau_over_T[j_min], ks[i_min],
        marker="*",
        color=style.landscape_min_marker_color,
        markersize=16,
        markeredgecolor=style.text_color,
        markeredgewidth=0.8,
        linestyle="none",
        label=fr"min $d={min_dist[i_min, j_min]:.3f}$ at $(\tau/T_{{wb}}, k)=({tau_over_T[j_min]:.2f}, {ks[i_min]:.1f})$",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("closest approach $d_{\\min}$")

    ax.set_xlabel(r"$\tau\,/\,T_\mathrm{wb}$")
    ax.set_ylabel(r"$k_\mathrm{tilt}$")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=style.font_size - 2)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


def run_and_save(
    prey_position: np.ndarray, ks, taus, t_end: float, T_wb: float,
    fov_half_angle: float | None = None,
    extra_overrides: dict | None = None,
    tag_suffix: str = "",
):
    tag = prey_tag(prey_position) + tag_suffix
    outdir = output_dir(tag)
    t0 = time.perf_counter()
    min_dist, t_min = run_sweep(
        ks, taus, t_end=t_end, prey_position=prey_position,
        fov_half_angle=fov_half_angle,
        extra_overrides=extra_overrides,
    )
    elapsed = time.perf_counter() - t0
    print(f"[{tag}] sweep wall: {elapsed:.1f}s")

    npz_path = os.path.join(outdir, "sweep.npz")
    png_path = os.path.join(outdir, "sweep.png")
    np.savez(
        npz_path,
        ks=ks, taus=taus, min_dist=min_dist, t_min=t_min,
        T_wb=T_wb, t_end=t_end, prey_position=np.asarray(prey_position, dtype=float),
    )
    title = fr"prey $(x, z) = ({prey_position[0]:g}, {prey_position[2]:g})$"
    plot_sweep(ks, taus, min_dist, T_wb, savepath=png_path, title=title)
    print(f"[{tag}] saved {npz_path} + {png_path}")


if __name__ == "__main__":
    wf = BASELINE["wing_frequency"]
    T_wb = 1.0 / wf

    ks   = np.arange(0.0, 5.0 + 1e-9, 0.5)
    taus = np.arange(0.0, 3.0 * T_wb + 1e-9, T_wb / 4.0)
    t_end = 10.0

    print(f"T_wb = {T_wb:.4f}")
    print(f"k grid (n={len(ks)}):   {ks.tolist()}")
    print(f"tau grid (n={len(taus)}): {[round(t, 4) for t in taus]}")

    baseline_prey = BASELINE["prey_position"]
    x0, z0 = float(baseline_prey[0]), float(baseline_prey[2])
    prey_cases = [
        np.array([x0,     0.0, z0    ]),
        np.array([2 * x0, 0.0, z0    ]),
        np.array([x0,     0.0, 2 * z0]),
        np.array([2 * x0, 0.0, 2 * z0]),
    ]

    for prey in prey_cases:
        run_and_save(prey, ks, taus, t_end, T_wb)

"""
2x2 grid of k_tilt vs sensing-delay maps over the 4 stationary-prey quadrants
(x-tilde = +/-3, z-tilde = +/-3).

Output PNGs land under `docs/_static/media/pursuit/` in both light and dark
variants.
"""

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "k_tau"
OUT_DIR = REPO_ROOT / "docs" / "_static" / "media" / "pursuit"
OUT_STEM = "pursuit_k_tau_stationary"


QUADRANTS = [
    # (row, col, x_tilde, z_tilde)
    (0, 0, -3, +3),
    (0, 1, +3, +3),
    (1, 0, -3, -3),
    (1, 1, +3, -3),
]


def _edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    d = x[1] - x[0]
    return np.concatenate([[x[0] - d / 2], (x[:-1] + x[1:]) / 2, [x[-1] + d / 2]])


def _load(xt: int, zt: int, tag_suffix: str = ""):
    path = DATA_ROOT / f"prey_x{xt:g}_z{zt:g}{tag_suffix}" / "sweep.npz"
    d = np.load(path)
    return d["ks"], d["taus"], d["min_dist"], float(d["T_wb"])


def plot_grid(panels, layout, T_wb, savepath, theme, width_in=8.5, ratio=0.9):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from post.style import apply_matplotlib_style, figure_size, resolve_style

    style = resolve_style(theme=theme)
    apply_matplotlib_style(style)

    all_vals = np.concatenate([m.ravel() for _, _, m in panels.values()])
    vmin = max(float(all_vals.min()), 1e-3)
    vmax = float(all_vals.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)

    nrows = max(r for r, _, _, _ in layout) + 1
    ncols = max(c for _, c, _, _ in layout) + 1

    w, h = figure_size(ratio, width_in=width_in)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h),
                             sharex=True, sharey=True, squeeze=False)

    mesh = None
    for row, col, key, title in layout:
        ax = axes[row, col]
        ks, taus, min_dist = panels[key]
        tau_over_T = taus / T_wb
        mesh = ax.pcolormesh(
            _edges(tau_over_T), _edges(ks), min_dist,
            cmap="inferno",
            norm=norm,
            shading="flat",
        )
        ax.set_title(title, fontsize=style.font_size)
        if row == nrows - 1:
            ax.set_xlabel(r"$\tau\,/\,T_\mathrm{wb}$")
        if col == 0:
            ax.set_ylabel(r"$K$")

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.7, pad=0.02, aspect=40)
    cbar.set_label(r"$\min(d/L)$")

    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_panels(items):
    """items: list of (key, xt, zt, tag_suffix). Returns (panels, T_wb)."""
    panels = {}
    T_wb = None
    for key, xt, zt, suf in items:
        ks, taus, min_dist, T_wb_q = _load(xt, zt, suf)
        mask = ks > 0
        panels[key] = (ks[mask], taus, min_dist[mask, :])
        T_wb = T_wb_q
    return panels, T_wb


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: 2x2 quadrants, default phase (pi).
    items = [((xt, zt), xt, zt, "") for _, _, xt, zt in QUADRANTS]
    panels, T_wb = _build_panels(items)
    layout = [
        (r, c, (xt, zt),
         fr"$(\tilde{{x}}, \tilde{{z}}) = ({xt:+d}, {zt:+d})$")
        for r, c, xt, zt in QUADRANTS
    ]
    for theme in ("light", "dark"):
        out = OUT_DIR / f"{OUT_STEM}.{theme}.png"
        plot_grid(panels, layout, T_wb, str(out), theme)
        print(f"saved {out}")

    # Figure 2: 4x2 phase comparison — rows = phase offset, cols = z-sign at x=+3.
    phase_rows = [
        (0, "_phase0", r"\Delta\phi = 0"),
        (1, "_phase90", r"\Delta\phi = \pi/2"),
        (2, "_phase135", r"\Delta\phi = 3\pi/4"),
        (3, "", r"\Delta\phi = \pi"),
    ]
    phase_items = []
    phase_layout = []
    for row, suf, phase_label in phase_rows:
        for col, zt in enumerate((+3, -3)):
            key = (suf, +3, zt)
            phase_items.append((key, +3, zt, suf))
            phase_layout.append((
                row, col, key,
                fr"${phase_label}$,  "
                fr"$(\tilde{{x}}, \tilde{{z}}) = (+3, {zt:+d})$",
            ))
    p_panels, T_wb_p = _build_panels(phase_items)
    for theme in ("light", "dark"):
        out = OUT_DIR / f"pursuit_k_tau_phase.{theme}.png"
        plot_grid(p_panels, phase_layout, T_wb_p, str(out), theme,
                  width_in=8.5, ratio=1.7)
        print(f"saved {out}")

    # Figure 3: target-distance 2x2 — rows = z-sign, cols = target distance.
    distance_cells = [
        # (row, col, xt, zt)
        (0, 0, +3, +3),
        (0, 1, +6, +6),
        (1, 0, +3, -3),
        (1, 1, +6, -6),
    ]
    d_items = [((xt, zt), xt, zt, "") for _, _, xt, zt in distance_cells]
    d_panels, T_wb_d = _build_panels(d_items)
    d_layout = [
        (r, c, (xt, zt),
         fr"$(\tilde{{x}}, \tilde{{z}}) = ({xt:+d}, {zt:+d})$")
        for r, c, xt, zt in distance_cells
    ]
    for theme in ("light", "dark"):
        out = OUT_DIR / f"pursuit_k_tau_distance.{theme}.png"
        plot_grid(d_panels, d_layout, T_wb_d, str(out), theme)
        print(f"saved {out}")


if __name__ == "__main__":
    main()

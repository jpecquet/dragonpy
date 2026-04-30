"""
XZ-plane trajectory plot for a tau/T_wb=1 controller pursuing stationary
prey at near (x, z = +3, +/-3) and distant (x, z = +6, +/-6) targets,
comparing tilt gains K=2 and K=3 at the baseline fore/hind phase offset (pi).

Each trajectory is trimmed at its closest-approach point. Outputs land under
`docs/_static/media/pursuit/` in both light and dark variants.
"""

from pathlib import Path

import numpy as np

from examples.parametric import BASELINE, run_case


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "_static" / "media" / "pursuit"
OUT_STEM = "pursuit_k_tau_distance_trajectories"

K_VALUES = [
    (2.0, "--", r"$K = 2$"),
    (3.0, "-",  r"$K = 3$"),
]


def plot_trajectories(runs, savepath, theme):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from post.style import apply_matplotlib_style, figure_size, resolve_style

    style = resolve_style(theme=theme)
    apply_matplotlib_style(style)

    w, h = figure_size(1.0, width_in=6.5)
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")

    target_colors = {
        (+3.0, +3.0): "tab:blue",
        (+6.0, +6.0): "tab:orange",
        (+3.0, -3.0): "tab:red",
        (+6.0, -6.0): "tab:green",
    }

    start = runs[0]["result"]["trajectory"][0]
    ax.plot(start[0], start[2], "o", color=style.text_color, markersize=6)

    seen_targets = set()
    for entry in runs:
        r = entry["result"]
        ls = entry["linestyle"]
        prey = r["prey_position"]
        key = (float(prey[0]), float(prey[2]))
        color = target_colors[key]

        if key not in seen_targets:
            ax.plot(prey[0], prey[2], "x", color=color,
                    markersize=9, markeredgewidth=2)
            seen_targets.add(key)

        traj = r["trajectory"]
        i_min = int(np.argmin(r["distances"]))
        seg = traj[: i_min + 1]
        ax.plot(seg[:, 0], seg[:, 2], linestyle=ls, color=color,
                linewidth=style.trajectory_linewidth)

    handles = [
        Line2D([0], [0], color=style.text_color, linestyle=ls,
               linewidth=style.trajectory_linewidth, label=label)
        for _, ls, label in K_VALUES
    ]

    ax.set_title(r"$\tau/T_\mathrm{wb} = 1$")
    ax.set_xlabel(r"$\tilde{X}$")
    ax.set_ylabel(r"$\tilde{Z}$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        borderaxespad=0.0,
        fontsize=style.font_size,
    )
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wf = BASELINE["wing_frequency"]
    T_wb = 1.0 / wf

    preys = [
        np.array([+3.0, 0.0, +3.0]),
        np.array([+6.0, 0.0, +6.0]),
        np.array([+3.0, 0.0, -3.0]),
        np.array([+6.0, 0.0, -6.0]),
    ]

    runs = []
    for k, linestyle, _ in K_VALUES:
        for prey in preys:
            r = run_case(
                f"k{k:g}_tau1Twb_x{prey[0]:g}_z{prey[2]:g}",
                k_tilt=float(k),
                sensing_delay=T_wb,
                prey_position=prey,
                t_end=15.0,
                fov_half_angle=np.pi,
            )
            runs.append({"result": r, "linestyle": linestyle, "k": k})
            print(f"K={k:g}  prey ({prey[0]:+g}, {prey[2]:+g}): "
                  f"min_dist={r['min_dist']:.3f} at t={r['t_min_dist']:.2f}s")

    out = OUT_DIR / f"{OUT_STEM}.dark.png"
    plot_trajectories(runs, str(out), "dark")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""
Representative-trajectory comparison figures for the docs page.

For each sweep, this module:

  1. Picks a contrast trial (R, theta, tilt) where the boolean capture vector
     across sweep values has the most disagreement, lightly biased toward
     mid-range R and small |theta| for clarity.
  2. Re-runs the simulation at that trial for every sweep value, recording
     position, velocity, and the brain's stroke-plane tilt at each slow tick.
  3. Renders a 6-panel comparison: xz trajectory, distance, speed, heading
     error, gamma, gamma_dot.

The output PNG goes under `docs/_static/media/capture/<sweep_name>_traj.dark.png`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dragonpy.dynamics import step_fast
from dragonpy.studies import captured_mask
from examples.capture_plot import load_result
from examples.capture_study import WING_FREQUENCY, make_setup


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "capture"
OUT_DIR   = REPO_ROOT / "docs" / "_static" / "media" / "capture"

CAPTURE_RADIUS = 0.5
T_WB = 1.0 / WING_FREQUENCY

BASELINE_KWARGS: dict[str, Any] = dict(
    k_tilt=3.0,
    sensing_delay=T_WB,
    fov_half_angle=np.pi,
    hind_phase_offset=np.pi / 2,
    intercept_feather_amp=np.radians(30.0),
    aero_ratio=0.025,
    span_ratio=0.75,
)


# --- Contrast-trial selection -------------------------------------------------

def pick_contrast_cell(value_data_dirs: list[str], capture_radius: float = 0.5):
    """Find the (R_idx, theta_idx, tilt_idx) cell whose boolean capture vector
    across sweep values is most informative (high cross-value variance), with
    light bias toward mid-range R and small |theta|.

    Returns: (R_value, theta_value, tilt_value, captured_per_value).
    """
    masks = []
    radii = thetas = tilts = None
    for d in value_data_dirs:
        result = load_result(DATA_ROOT / d / "result.npz")
        masks.append(captured_mask(result, capture_radius))  # (nR, nTh, nTilt)
        radii, thetas, tilts = result.radii, result.thetas, result.tilts

    M = np.stack(masks, axis=0)            # (nValues, nR, nTh, nTilt)
    p = M.mean(axis=0)                      # mean across values
    variance = p * (1.0 - p)                # bool variance per cell

    # Mid-R / small-|theta| bias (small penalties so they only break ties).
    R_mid    = 0.5 * (radii[0] + radii[-1])
    R_range  = max(radii[-1] - radii[0], 1e-9)
    R_pen    = np.abs(radii - R_mid) / R_range          # in [0, 0.5]
    th_pen   = np.abs(thetas) / np.pi                   # in [0, 0.5]
    score = (
        variance
        - 0.05 * R_pen[:, None, None]
        - 0.05 * th_pen[None, :, None]
    )
    idx = np.unravel_index(int(np.argmax(score)), score.shape)
    return (
        float(radii[idx[0]]),
        float(thetas[idx[1]]),
        float(tilts[idx[2]]),
        M[:, idx[0], idx[1], idx[2]].astype(bool),
    )


# --- Trajectory recording ----------------------------------------------------

def _target_xz(R: float, theta: float) -> np.ndarray:
    return np.array([R * np.cos(theta), 0.0, R * np.sin(theta)])


@dataclass
class Trace:
    t:        np.ndarray   # (N,)   seconds
    pos:      np.ndarray   # (N, 3) body position
    vel:      np.ndarray   # (N, 3) body velocity
    gamma:    np.ndarray   # (N,)   stroke-plane tilt at each slow tick (radians)
    target:   np.ndarray   # (3,)   target position
    crossed:  bool         # did the trial reach the R-circle


def run_with_trace(
    setup_fn: Callable[[float, np.ndarray], "object"],
    R: float,
    theta: float,
    tilt: float,
    max_wingbeats: float = 1000.0,
) -> Trace:
    """Re-run a single capture trial and record full trajectory."""
    target = _target_xz(R, theta)
    sim = setup_fn(tilt, target.copy())
    dfly = sim.dragonfly
    f = dfly.wing_frequency
    if f <= 0.0:
        raise ValueError("wing_frequency must be positive")
    t_max = max_wingbeats / f
    R2 = R * R
    dt_slow = sim.dt_fast * sim.fast_per_slow
    pos0 = dfly.position.copy()

    times    = [float(sim.t)]
    positions = [dfly.position.copy()]
    velocities = [dfly.velocity.copy()]
    gammas   = [float(getattr(dfly.brain, "stroke_plane_tilt", 0.0))]

    crossed = False
    while sim.t < t_max:
        dfly.sensors.sample_all(sim)
        dfly.brain.update(dfly, dt_slow)
        for _ in range(sim.fast_per_slow):
            step_fast(sim)
            dx = dfly.position[0] - pos0[0]
            dz = dfly.position[2] - pos0[2]
            if dx * dx + dz * dz >= R2:
                crossed = True
                break
        times.append(float(sim.t))
        positions.append(dfly.position.copy())
        velocities.append(dfly.velocity.copy())
        gammas.append(float(getattr(dfly.brain, "stroke_plane_tilt", 0.0)))
        if crossed:
            break

    return Trace(
        t=np.array(times),
        pos=np.stack(positions),
        vel=np.stack(velocities),
        gamma=np.array(gammas),
        target=target,
        crossed=crossed,
    )


# --- Plotting ----------------------------------------------------------------

def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


# 4-color palette: blue, purple, red, gold. For 3-value sweeps the middle
# (purple) is dropped so the endpoints span the full hue range.
_PALETTE_4 = ["#4aa3ff", "#c084fc", "#ff6b6b", "#facc15"]
_PALETTE_3 = ["#4aa3ff", "#ff6b6b", "#facc15"]


def _palette(n: int) -> list[str]:
    if n == 3:
        return _PALETTE_3
    return _PALETTE_4[:n]


def plot_trajectory_panel(
    title:    str,
    contrast: tuple[float, float, float],
    traces:   list[tuple[str, Trace, bool]],
    savepath: Path,
    theme:    str = "dark",
) -> None:
    """Render the 6-panel comparison figure: xz traj, distance, speed,
    heading error, gamma, gamma_dot. `traces` is [(label, Trace, captured), ...].
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from post.style import apply_matplotlib_style, figure_size, resolve_style

    style = resolve_style(theme=theme)
    apply_matplotlib_style(style)

    w, h = figure_size(1.2, width_in=6.5)
    fig, axes = plt.subplots(3, 2, figsize=(w, h), dpi=300)

    n = len(traces)
    colors = _palette(n)

    target = traces[0][1].target

    ax_xz   = axes[0, 0]
    ax_dist = axes[0, 1]
    ax_spd  = axes[1, 0]
    ax_herr = axes[1, 1]
    ax_g    = axes[2, 0]
    ax_gd   = axes[2, 1]

    # Mark target on the xz panel.
    ax_xz.scatter([target[0]], [target[2]], marker="*", s=80,
                  color=style.text_color, zorder=10)

    for i, (label, tr, captured) in enumerate(traces):
        c = colors[i]
        t_norm = tr.t / T_WB
        d_xz = np.linalg.norm(tr.pos[:, [0, 2]] - target[[0, 2]][None, :], axis=1)
        speed = np.linalg.norm(tr.vel[:, [0, 2]], axis=1)

        v_heading      = np.arctan2(tr.vel[:, 2], tr.vel[:, 0])
        target_heading = np.arctan2(target[2] - tr.pos[:, 2],
                                    target[0] - tr.pos[:, 0])
        h_err = _wrap_to_pi(target_heading - v_heading)

        # gamma_dot: trim the truncated final slow tick to avoid finite-diff
        # spike at termination.
        if tr.t.size >= 3 and tr.crossed:
            t_g = tr.t[:-1]
            gamma_g = tr.gamma[:-1]
        else:
            t_g = tr.t
            gamma_g = tr.gamma
        gamma_dot = (
            np.gradient(gamma_g, t_g) if t_g.size >= 2 else np.zeros_like(gamma_g)
        )
        t_g_norm = t_g / T_WB

        ax_xz.plot(tr.pos[:, 0], tr.pos[:, 2], color=c, lw=1.5)
        marker_face = c if captured else "none"
        ax_xz.scatter([tr.pos[-1, 0]], [tr.pos[-1, 2]],
                      marker="o", s=28,
                      facecolor=marker_face, edgecolor=c, lw=1.2, zorder=9)

        ax_dist.plot(t_norm, d_xz, color=c, lw=1.5)
        ax_spd .plot(t_norm, speed, color=c, lw=1.5)
        ax_herr.plot(t_norm, np.degrees(h_err), color=c, lw=1.5)
        ax_g   .plot(t_norm, np.degrees(tr.gamma), color=c, lw=1.5)
        ax_gd  .plot(t_g_norm, np.degrees(gamma_dot), color=c, lw=1.5)

    # xz trajectory cosmetics.
    ax_xz.set_xlabel(r"$\tilde{x}$")
    ax_xz.set_ylabel(r"$\tilde{z}$")
    ax_xz.set_aspect("equal", adjustable="datalim")
    ax_xz.set_title(r"trajectory $(\tilde{x}, \tilde{z})$",
                    fontsize=style.font_size)
    th = np.linspace(0, 2 * np.pi, 128)
    ax_xz.plot(target[0] + CAPTURE_RADIUS * np.cos(th),
               target[2] + CAPTURE_RADIUS * np.sin(th),
               color=style.axes_edge_color, lw=0.6, ls=":")

    panels = [
        (ax_dist, r"distance $\tilde{d}$",                       r"$\tilde{d}$"),
        (ax_spd,  r"speed $\tilde{u}$",                          r"$\tilde{u}$"),
        (ax_herr, r"heading error $\theta_t - \theta_v$ (°)",    r"$\theta_t - \theta_v$ (°)"),
        (ax_g,    r"stroke-plane tilt $\gamma$ (°)",             r"$\gamma$ (°)"),
        (ax_gd,   r"$\dot\gamma$ (° / s)",                        r"$\dot\gamma$ (° / s)"),
    ]
    for ax, title_str, ylabel in panels:
        ax.set_xlabel(r"$t / T_\mathrm{wb}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title_str, fontsize=style.font_size)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(left=0.0)

    # Clamp the gamma panel to a single revolution. Trials whose controller
    # runs away (γ unwrapped past ±180°) will exit the visible range — that's
    # the intended cue.
    ax_g.set_ylim(-180.0, 180.0)

    # Top-of-figure legend, single row.
    handles = [
        Line2D([0], [0], color=c, lw=2.0, label=label)
        for (label, _, _), c in zip(traces, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        fontsize=style.font_size,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(savepath, dpi=300)
    plt.close(fig)


# --- Per-sweep driver --------------------------------------------------------

def render_sweep_trajectory(
    sweep_name:       str,
    section_title:    str,
    value_labels:     list[str],
    value_overrides:  list[dict],
    value_data_dirs:  list[str],
) -> Path:
    """Pick the contrast trial for the sweep, re-run all values at that trial,
    plot the comparison, and return the saved PNG path."""
    R, theta, tilt, captured_per_value = pick_contrast_cell(
        value_data_dirs, capture_radius=CAPTURE_RADIUS,
    )
    traces: list[tuple[str, Trace, bool]] = []
    for label, overrides, captured in zip(
        value_labels, value_overrides, captured_per_value,
    ):
        kwargs = {**BASELINE_KWARGS, **overrides}
        setup_fn = make_setup(**kwargs)
        tr = run_with_trace(setup_fn, R, theta, tilt)
        traces.append((label, tr, bool(captured)))

    out = OUT_DIR / f"{sweep_name}_traj.dark.png"
    plot_trajectory_panel(section_title, (R, theta, tilt), traces, out)
    print(f"saved {out.name}  "
          f"(R={R:.0f}, theta={np.degrees(theta):+.0f}°, "
          f"tilt={np.degrees(tilt):+.0f}°)")
    return out

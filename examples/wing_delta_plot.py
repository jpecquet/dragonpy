"""
Wing-tip-path analysis: pursuit kinematics in the air frame vs. hover.

Loads the long-pursuit run from `data/wing_delta/R50_theta0_tilt0_K3.npz`,
extracts wingbeat cycles at increasing body speed, and renders a 2x2 figure
to `docs/_static/media/maneuvering/wing_tip_loops.dark.png`.

Layout:
  Rows: top view (body x-y plane) | side view (body x-z plane)
  Cols: air-frame tip path over two cycles, anchored at the body position at
          the start of the window (so each cycle's V·T forward drift is visible
          as a scallop, but the inter-sample bulk drift over 50 BL is hidden)
        drift-subtracted tip path (single cycle, recentered at hinge origin)

The dragonfly is point-mass in this run, so the body never rotates and the
world frame coincides with the air rest frame (no wind in the sim). The wing
tip in world frame is therefore `body_position + R_body_from_hinge @
R_hinge_from_wing @ (span_ratio, 0, 0)`.

Hover reference: a HoverBrain run with 45-degree tilted stroke plane,
converged to gravity-balance and zero horizontal velocity by the rate
controllers on `sweep_amp` (vertical) and `feather_mean` (horizontal). The
converged stroke pattern is used as the hover reference loop.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dragonpy.body.muscles import StrokePattern, expand_pattern
from dragonpy.body.sensors import (
    AirflowSensor, CompoundEye, InertialSensor, Ocelli, Sensors, WingLoadSensor,
)
from dragonpy.brain import HoverBrain
from dragonpy.dragonfly import Dragonfly
from dragonpy.dynamics import Simulation, step_slow
from dragonpy.world import Environment
from examples.capture_study import WING_FREQUENCY, make_wing
from post.style import apply_matplotlib_style, figure_size, resolve_style


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data" / "wing_delta"
OUT_DIR   = REPO_ROOT / "docs" / "_static" / "media" / "maneuvering"
T_WB      = 1.0 / WING_FREQUENCY

DATA_FILE       = "R50_theta0_tilt0_K3.npz"
OUT_TIP_FILE    = "wing_tip_loops.dark.png"
OUT_AOA_FILE    = "wing_aoa.dark.png"

REFERENCE_WING  = 0      # right forewing
SPAN_FRACTION   = 0.7    # representative blade element (70% span)

HOVER_TILT_DEG  = 45.0
HOVER_DURATION  = 50.0 / WING_FREQUENCY   # 50 wingbeats
HOVER_K_Z       = 1.0
HOVER_K_X       = 1.0

# Intercept-mode kinematic constants. These match the InterceptBrain config
# wired up in `examples.capture_study` / `examples.capture_trajectory_plot`
# (BASELINE_KWARGS) — keep in sync if either side changes.
INTERCEPT_SWEEP_AMP     = np.radians(45.0)
INTERCEPT_FEATHER_AMP   = np.radians(30.0)
INTERCEPT_FEATHER_PHASE = np.pi / 2


def converge_hover_pattern(verbose: bool = True) -> StrokePattern:
    """Run HoverBrain to convergence at a 45-degree tilted stroke plane.

    Builds a point-mass dragonfly with the same wing geometry as the capture
    study, gives it HoverBrain (rate controllers on sweep amplitude for
    vertical balance and feather mean for horizontal balance), runs from
    rest, and returns the converged StrokePattern.
    """
    placeholder = StrokePattern(
        stroke_plane_tilt=0.0,
        sweep_amp=0.0, sweep_mean=0.0, sweep_phase=0.0,
        elev_amp=0.0,  elev_mean=0.0,  elev_phase=0.0,  elev_harmonic=2,
        feather_amp=0.0, feather_mean=0.0, feather_phase=0.0,
    )
    wings = [make_wing(c) for c in (+1, -1, +1, -1)]
    sensors = Sensors(
        inertial=InertialSensor(),
        eye=CompoundEye(fov_half_angle=np.pi, max_range=np.inf, delay=T_WB),
        ocelli=Ocelli(),
        airflow=AirflowSensor(),
        wing_load=WingLoadSensor(),
    )
    brain = HoverBrain(
        sweep_amp_init=np.radians(35.0),
        feather_amp=np.radians(60.0),
        feather_phase=np.pi / 2,
        wing_frequency=WING_FREQUENCY,
        k_z=HOVER_K_Z,
        k_x=HOVER_K_X,
        stroke_plane_tilt=np.radians(HOVER_TILT_DEG),
    )
    dragonfly = Dragonfly(
        wings=wings,
        sensors=sensors,
        brain=brain,
        stroke_patterns=[StrokePattern(**placeholder.__dict__) for _ in range(4)],
        inertia_body=np.array([0.01, 0.05, 0.05]),
        position=np.zeros(3),
        point_mass=True,
        wing_frequency=WING_FREQUENCY,
    )
    sim = Simulation(
        dragonfly=dragonfly,
        environment=Environment(),
        dt_fast=1.0 / (WING_FREQUENCY * 100),
        fast_per_slow=15,
    )
    while sim.t < HOVER_DURATION:
        step_slow(sim)

    if verbose:
        print(
            f"hover converged: t={sim.t:.2f} s "
            f"({sim.t / T_WB:.1f} T_wb), "
            f"v=({dragonfly.velocity[0]:+.4f}, {dragonfly.velocity[2]:+.4f}), "
            f"sweep_amp={np.degrees(brain.sweep_amp):.2f}°, "
            f"feather_mean={np.degrees(brain.feather_mean):+.2f}°, "
            f"tilt={np.degrees(brain.stroke_plane_tilt):.1f}°"
        )

    return StrokePattern(
        stroke_plane_tilt=float(brain.stroke_plane_tilt),
        sweep_amp=float(brain.sweep_amp),
        sweep_mean=0.0,
        sweep_phase=0.0,
        elev_amp=0.0,
        elev_mean=0.0,
        elev_phase=0.0,
        elev_harmonic=2,
        feather_amp=float(brain.feather_amp),
        feather_mean=float(brain.feather_mean),
        feather_phase=float(brain.feather_phase),
    )


def hover_tip_loop(
    hover_pattern:     StrokePattern,
    hinge_orientation: np.ndarray,
    span_ratio:        float,
    chirality:         int,
    n_phase:           int = 240,
) -> np.ndarray:
    """Tip trajectory of one wing under `hover_pattern`, body at origin."""
    phases = np.linspace(0.0, 2 * np.pi, n_phase, endpoint=False)
    omega = 2 * np.pi * WING_FREQUENCY
    span_axis_wing = np.array([span_ratio, 0.0, 0.0])
    tips = np.empty((n_phase, 3))
    for i, phase in enumerate(phases):
        R_hw, _ = expand_pattern(hover_pattern, float(phase), omega, chirality)
        R_bw = hinge_orientation @ R_hw
        tips[i] = R_bw @ span_axis_wing
    return tips


def intercept_pattern(gamma: float) -> StrokePattern:
    """Stroke pattern installed by InterceptBrain in pursuit mode at the given
    stroke-plane tilt `gamma`."""
    return StrokePattern(
        stroke_plane_tilt=float(gamma),
        sweep_amp=INTERCEPT_SWEEP_AMP,
        sweep_mean=0.0, sweep_phase=0.0,
        elev_amp=0.0, elev_mean=0.0, elev_phase=0.0,
        elev_harmonic=2,
        feather_amp=INTERCEPT_FEATHER_AMP,
        feather_mean=0.0,
        feather_phase=INTERCEPT_FEATHER_PHASE,
    )


def aoa_and_speed(
    R_hw:           np.ndarray,  # (..., 3, 3)
    omega_wh:       np.ndarray,  # (..., 3)
    v_body:         np.ndarray,  # (..., 3)
    hinge_orient:   np.ndarray,  # (3, 3)
    span_ratio:     float,
    span_fraction:  float,
    chirality:      int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-steady angle of attack and air-relative speed at one blade element.

    Mirrors the in-wing-frame velocity decomposition used in `wing_wrench`:
    rotate the air-relative element velocity into wing frame, then the
    chord-plane components (y_le, z) define alpha and |V|. Returns shapes
    matching the leading dims of the inputs.
    """
    R_bw = hinge_orient @ R_hw
    omega_wing_rel_body = (hinge_orient @ omega_wh[..., None])[..., 0]
    r = span_fraction * span_ratio
    wing_x_body = R_bw[..., :, 0]
    r_from_hinge_body = r * wing_x_body
    v_elem_body = v_body + np.cross(omega_wing_rel_body, r_from_hinge_body)
    v_rel_wing = np.einsum("...ji,...j->...i", R_bw, v_elem_body)
    vy_le = chirality * v_rel_wing[..., 1]
    vz = v_rel_wing[..., 2]
    V = np.sqrt(vy_le * vy_le + vz * vz)
    alpha = np.arctan2(-vz, vy_le)
    return alpha, V


def hover_aoa_cycle(
    hover_pattern:  StrokePattern,
    hinge_orient:   np.ndarray,
    span_ratio:     float,
    span_fraction:  float,
    chirality:      int,
    n_phase:        int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AoA and |V_rel| at the chosen blade element over one hover cycle.
    Returns (phase_norm in [0,1), alpha [rad], V [body lengths / s])."""
    phases = np.linspace(0.0, 2 * np.pi, n_phase, endpoint=False)
    omega_phase = 2 * np.pi * WING_FREQUENCY
    R_hw = np.empty((n_phase, 3, 3))
    omega_wh = np.empty((n_phase, 3))
    for i, phase in enumerate(phases):
        R_hw[i], omega_wh[i] = expand_pattern(
            hover_pattern, float(phase), omega_phase, chirality,
        )
    v_body = np.zeros((n_phase, 3))
    alpha, V = aoa_and_speed(
        R_hw, omega_wh, v_body, hinge_orient, span_ratio, span_fraction, chirality,
    )
    return phases / (2 * np.pi), alpha, V


def pursuit_cycle_aoa(
    R_hw_seq:       np.ndarray,  # (M, 3, 3) saved
    gamma_seq:      np.ndarray,  # (M,) saved stroke_plane_tilt
    phase_seq:      np.ndarray,  # (M,) saved wing phase
    vel_seq:        np.ndarray,  # (M, 3) saved body velocity
    hinge_orient:   np.ndarray,
    span_ratio:     float,
    span_fraction:  float,
    chirality:      int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AoA and |V_rel| over one pursuit cycle. Reconstructs omega_wh by
    re-expanding the intercept-mode pattern at the saved gamma + phase."""
    omega_phase = 2 * np.pi * WING_FREQUENCY
    M = len(phase_seq)
    omega_wh = np.empty((M, 3))
    for i in range(M):
        pat = intercept_pattern(gamma_seq[i])
        _, omega_wh[i] = expand_pattern(
            pat, float(phase_seq[i]), omega_phase, chirality,
        )
    alpha, V = aoa_and_speed(
        R_hw_seq, omega_wh, vel_seq, hinge_orient, span_ratio, span_fraction, chirality,
    )
    phase_norm = ((phase_seq - phase_seq[0]) / (2 * np.pi)) % 1.0
    return phase_norm, alpha, V


def cycle_bounds(wing_phase: np.ndarray) -> list[tuple[int, int]]:
    """Return [(i0, i1), ...] per full 2pi cycle in monotonic `wing_phase`."""
    p = wing_phase / (2 * np.pi)
    k_lo = int(np.ceil(p[0]))
    k_hi = int(np.floor(p[-1]))
    bounds = []
    for k in range(k_lo, k_hi):
        i0 = int(np.searchsorted(p, k))
        i1 = int(np.searchsorted(p, k + 1))
        if i1 > i0 + 4:
            bounds.append((i0, i1))
    return bounds


def render_tip_loops(
    bounds, pick, labels, colors, hover_color, hover_tip,
    tip_world, pos, t, n_cycles, style, out_path,
) -> None:
    w_fig, h_fig = figure_size(0.95, width_in=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(w_fig, h_fig), dpi=300)
    ax_xy_raw, ax_xy_sub = axes[0, 0], axes[0, 1]
    ax_xz_raw, ax_xz_sub = axes[1, 0], axes[1, 1]

    def plot_loop(ax, xy, color, lw, label=None):
        closed = np.vstack([xy, xy[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw, label=label)

    plot_loop(ax_xy_raw, hover_tip[:, [0, 1]], hover_color, 1.6, label="hover")
    plot_loop(ax_xy_sub, hover_tip[:, [0, 1]], hover_color, 1.6, label="hover")
    plot_loop(ax_xz_raw, hover_tip[:, [0, 2]], hover_color, 1.6)
    plot_loop(ax_xz_sub, hover_tip[:, [0, 2]], hover_color, 1.6)

    for c, k, label in zip(colors, pick, labels):
        i0, i1 = bounds[k]
        i1_two = bounds[min(k + 1, n_cycles - 1)][1] if k < n_cycles - 1 else i1
        tip_two_anchored = tip_world[i0:i1_two] - pos[i0][None, :]

        tip_one = tip_world[i0:i1]
        t_one   = t[i0:i1]
        dt = t_one[-1] - t_one[0]
        if dt <= 0:
            continue
        v_cycle = (pos[i1 - 1] - pos[i0]) / dt
        tip_sub = (
            tip_one
            - v_cycle[None, :] * (t_one[:, None] - t_one[0])
            - pos[i0][None, :]
        )

        ax_xy_raw.plot(tip_two_anchored[:, 0], tip_two_anchored[:, 1],
                       color=c, lw=1.0, alpha=0.95, label=label)
        ax_xz_raw.plot(tip_two_anchored[:, 0], tip_two_anchored[:, 2],
                       color=c, lw=1.0, alpha=0.95)
        plot_loop(ax_xy_sub, tip_sub[:, [0, 1]], c, 1.3, label=label)
        plot_loop(ax_xz_sub, tip_sub[:, [0, 2]], c, 1.3)

    for ax in (ax_xy_raw, ax_xy_sub):
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{y}$")
    for ax in (ax_xz_raw, ax_xz_sub):
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{z}$")
    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")

    handles, leg_labels = ax_xy_sub.get_legend_handles_labels()
    fig.legend(
        handles, leg_labels, loc="upper center",
        ncol=len(leg_labels), bbox_to_anchor=(0.5, 0.995), frameon=False,
        fontsize=style.font_size,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def render_aoa(
    bounds, pick, labels, colors, hover_color,
    R_hw_all, wing_phases, vel, gamma,
    hinge_orient, span_ratios, chiralities,
    hover_pattern, w, style, out_path,
) -> None:
    span_ratio = float(span_ratios[w])
    chirality  = int(chiralities[w])

    phase_h, alpha_h, V_h = hover_aoa_cycle(
        hover_pattern, hinge_orient[w], span_ratio, SPAN_FRACTION, chirality,
    )

    w_fig, h_fig = figure_size(0.85, width_in=6.5)
    fig, axes = plt.subplots(2, 1, figsize=(w_fig, h_fig), dpi=300, sharex=True)
    ax_a, ax_v = axes

    # AoA at low |V| is geometrically undefined (angle relative to a
    # near-zero vector). Mask alpha where |V| < 5% of the cycle peak so the
    # plot doesn't show meaningless wrap artifacts at stroke reversal.
    def mask_low_V(alpha, V):
        thresh = 0.05 * float(np.max(V))
        return np.where(V > thresh, alpha, np.nan)

    ph_h = np.append(phase_h, 1.0)
    al_h = np.append(alpha_h, alpha_h[0])
    Vh   = np.append(V_h, V_h[0])
    al_h_masked = mask_low_V(al_h, Vh)
    ax_a.plot(ph_h, np.degrees(al_h_masked), color=hover_color, lw=1.6, label="hover")
    ax_v.plot(ph_h, Vh, color=hover_color, lw=1.6)

    for c, k, label in zip(colors, pick, labels):
        i0, i1 = bounds[k]
        ph, alpha, V = pursuit_cycle_aoa(
            R_hw_all[i0:i1, w],
            gamma[i0:i1],
            wing_phases[i0:i1, w],
            vel[i0:i1],
            hinge_orient[w], span_ratio, SPAN_FRACTION, chirality,
        )
        order = np.argsort(ph)
        alpha_masked = mask_low_V(alpha[order], V[order])
        ax_a.plot(ph[order], np.degrees(alpha_masked),
                  color=c, lw=1.2, alpha=0.95, label=label)
        ax_v.plot(ph[order], V[order], color=c, lw=1.2, alpha=0.95)

    ax_a.set_ylabel(r"$\alpha$ (°)")
    ax_v.set_ylabel(r"$|\tilde{V}_\mathrm{air}|$")
    ax_v.set_xlabel(r"cycle phase $\phi/2\pi$")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0.0, 1.0)
    ax_a.axhline(0.0, color=style.axes_edge_color, lw=0.6, ls=":")

    handles, leg_labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles, leg_labels, loc="upper center",
        ncol=len(leg_labels), bbox_to_anchor=(0.5, 0.995), frameon=False,
        fontsize=style.font_size,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    data = np.load(DATA_DIR / DATA_FILE)
    t              = data["t"]
    pos            = data["pos"]
    vel            = data["vel"]
    gamma          = data["gamma"]
    R_hw_all       = data["R_hinge_from_wing"]      # (N, n_wings, 3, 3)
    wing_phases    = data["wing_phases"]            # (N, n_wings)
    hinge_orient   = data["hinge_orientations"]     # (n_wings, 3, 3)
    span_ratios    = data["span_ratios"]
    chiralities    = data["chiralities"]

    w = REFERENCE_WING
    span_ratio = float(span_ratios[w])
    chirality  = int(chiralities[w])
    R_bw_w = hinge_orient[w] @ R_hw_all[:, w]
    tip_body  = R_bw_w @ np.array([span_ratio, 0.0, 0.0])
    tip_world = pos + tip_body

    bounds = cycle_bounds(wing_phases[:, w])
    n_cycles = len(bounds)
    if n_cycles < 5:
        raise RuntimeError(f"too few full cycles ({n_cycles}) for analysis")

    cycle_speed = np.array([
        float(np.linalg.norm(np.mean(vel[i0:i1, [0, 2]], axis=0)))
        for (i0, i1) in bounds
    ])

    pick = sorted({
        int(0.05 * n_cycles),
        int(0.30 * n_cycles),
        int(0.60 * n_cycles),
        n_cycles - 2,
    })
    pick = [max(0, min(n_cycles - 2, k)) for k in pick]

    print(f"cycles: {n_cycles}; speed range: "
          f"{cycle_speed.min():.2f} .. {cycle_speed.max():.2f}")
    print(f"picked indices: {pick}")
    for k in pick:
        i0, i1 = bounds[k]
        print(f"  cycle {k}: t={t[i0]:.2f}..{t[i1-1]:.2f} s, "
              f"|V|={cycle_speed[k]:.2f}")

    hover_pattern = converge_hover_pattern()
    hover_tip = hover_tip_loop(
        hover_pattern, hinge_orient[w], span_ratio, chirality,
    )

    style = resolve_style(theme="dark")
    apply_matplotlib_style(style)

    cmap = plt.cm.plasma
    n_pick = len(pick)
    colors = [cmap(0.15 + 0.7 * i / max(1, n_pick - 1)) for i in range(n_pick)]
    hover_color = "#4aa3ff"

    labels = [rf"$\tilde u={cycle_speed[k]:.2f}$" for k in pick]

    render_tip_loops(
        bounds, pick, labels, colors, hover_color, hover_tip,
        tip_world, pos, t, n_cycles, style,
        out_path=OUT_DIR / OUT_TIP_FILE,
    )
    render_aoa(
        bounds, pick, labels, colors, hover_color,
        R_hw_all, wing_phases, vel, gamma,
        hinge_orient, span_ratios, chiralities,
        hover_pattern, w, style,
        out_path=OUT_DIR / OUT_AOA_FILE,
    )


if __name__ == "__main__":
    main()

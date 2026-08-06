"""Reproduce the quantitative figures and metrics of report1 from the backend.

Design maps (mean-force analysis): cf_components, pitch_efficiency,
force_direction_test, pitch_efficiency_J, hover_optimum_J, force_direction_J.
Closed-loop results: analytic_trim_hover, trajectory_gains[_drawn],
analytic_trim_pursuit. Usage: python figures.py [out_dir] [names...]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from summer_paper.backend import sim  # noqa: E402
from summer_paper.backend import trajectory as traj  # noqa: E402
from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = ROOT / "summer_paper" / "figures"

PSI0_RANGE = np.radians((-90.0, 90.0))
PSI1_RANGE = np.radians((0.0, 90.0))
DELTA0_RANGE = np.radians((-180.0, 180.0))
J_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

C1, C2, C3 = "black", "#b2182b", "#2166ac"
S0_OVER_PHI1 = sim.SPAN_FRAC * sim.REF.Lw
STAR = dict(marker="o", color="white", ms=7, mec="black", mew=0.8,
            linestyle="none")


def _save(fig, name, dpi=200):
    out = OUT_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Mean-force analysis (delta0 = 90 deg operating point unless noted).

def grid_op(J, n=73):
    """C_Fn, C_Fs, C_F* over (psi0, psi1), indexed [psi1, psi0]."""
    psi0 = np.linspace(*PSI0_RANGE, n)
    psi1 = np.linspace(*PSI1_RANGE, n)
    I0, In, Is = sim.averages(J, psi1)
    cn = 2.0 * np.outer(In, np.cos(2.0 * psi0)) - 2.0 * J * sim.CDBAR * I0
    cs = -2.0 * np.outer(Is, np.sin(2.0 * psi0))
    return psi0, psi1, np.hypot(cn, cs), cn, cs


def track_optimum(J, psi1_prev, n=721):
    """Continue the hover-optimal point along psi0 = 0 to advance ratio J."""
    psi1 = np.linspace(*PSI1_RANGE, n)
    I0, In, _ = sim.averages(J, psi1)
    f = np.abs(2.0 * In - 2.0 * J * sim.CDBAR * I0)
    loc = np.flatnonzero((f[1:-1] >= f[:-2]) & (f[1:-1] >= f[2:])) + 1
    j = loc[np.argmin(np.abs(psi1[loc] - psi1_prev))]
    off = np.hypot(*sim.cf_op(np.radians(2.5), psi1[j], J))
    return psi1[j], f[j], float(off) < f[j]


def optimum_continuation(J_fine):
    psi1_c, cf_c = np.empty_like(J_fine), np.empty_like(J_fine)
    prev, saddle_J = np.radians(51.0), None
    for i, J in enumerate(J_fine):
        prev, cf_i, is_max = track_optimum(J, prev)
        psi1_c[i], cf_c[i] = prev, cf_i
        if saddle_J is None and not is_max:
            saddle_J = J
    return psi1_c, cf_c, saddle_J


def beta_curve(psi0, J, psi1):
    """Continuous force-direction branch beta(psi0) in degrees (odd grid)."""
    I0, In, Is = sim.averages(J, psi1)
    cn = 2.0 * float(In) * np.cos(2.0 * psi0) - 2.0 * J * sim.CDBAR * I0
    cs = -2.0 * float(Is) * np.sin(2.0 * psi0)
    b = np.degrees(np.unwrap(-np.arctan2(cs, cn)))
    mid = len(b) // 2
    b = b - b[mid]
    if cn[mid] <= 0.0:
        b = 180.0 - np.mod(-b, 360.0)   # anchor beta(0) = +180 (force along -n)
    return b, cn[mid]


def fig_cf_components(style):
    psi0, psi1, _, cn, cs = grid_op(0.0)
    vmax = np.ceil(max(np.abs(cn).max(), np.abs(cs).max()) * 10.0) / 10.0
    levels = np.round(np.arange(-vmax, vmax + 1e-9, 0.1), 2)
    line_kw = dict(levels=levels, colors="black", linewidths=0.3, alpha=0.35)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True,
                                   constrained_layout=True)
    P0, P1 = np.meshgrid(np.degrees(psi0), np.degrees(psi1))
    cf = ax1.contourf(P0, P1, cn, levels=levels, cmap="RdBu_r")
    ax1.contour(P0, P1, cn, **line_kw)
    ax1.set_xlabel(r"$\psi_0$ (deg)")
    ax1.set_ylabel(r"$\psi_1$ (deg)")
    ax1.set_title(r"(a)  $C_{F^\ast,n}$ at $\delta_0 = 90^\circ$",
                  fontsize=style.font_size)
    ax2.contourf(P0, P1, cs, levels=levels, cmap="RdBu_r")
    ax2.contour(P0, P1, cs, **line_kw)
    ax2.set_xlabel(r"$\psi_0$ (deg)")
    ax2.set_title(r"(b)  $C_{F^\ast,s}$ at $\delta_0 = 90^\circ$",
                  fontsize=style.font_size)
    fig.colorbar(cf, ax=[ax1, ax2], label=r"$C_{F^\ast,n},\ C_{F^\ast,s}$")
    print(f"C_F*,n range [{cn.min():.3f}, {cn.max():.3f}], "
          f"C_F*,s range [{cs.min():.3f}, {cs.max():.3f}]")
    _save(fig, "cf_components.light.png", dpi=300)


def fig_pitch_efficiency(style):
    n = 73
    delta0 = np.linspace(*DELTA0_RANGE, n)
    psi1 = np.linspace(*PSI1_RANGE, n)
    cf_dp = np.array([[np.hypot(*sim.cf_quad(0.0, p1, d)) for d in delta0]
                      for p1 in psi1])
    jmax, imax = np.unravel_index(np.argmax(cf_dp), cf_dp.shape)
    psi1_o, delta0_o = psi1[jmax], abs(delta0[imax])
    psi0, _, cf_pp, _, _ = grid_op(0.0, n)

    levels = np.round(np.arange(0.0, 1.6, 0.1), 2)
    line_kw = dict(levels=levels, colors="black", linewidths=0.3, alpha=0.35)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True,
                                   constrained_layout=True)
    D0, P1 = np.meshgrid(np.degrees(delta0), np.degrees(psi1))
    cf = ax1.contourf(D0, P1, cf_dp, levels=levels, cmap="RdPu")
    ax1.contour(D0, P1, cf_dp, **line_kw)
    ax1.plot(np.degrees(delta0_o), np.degrees(psi1_o), **STAR)
    ax1.set_xlabel(r"$\delta_0$ (deg)")
    ax1.set_ylabel(r"$\psi_1$ (deg)")
    ax1.set_title(r"(a)  $C_{F^\ast}(\delta_0,\psi_1)$ at $\psi_0=0$",
                  fontsize=style.font_size)
    P0, P1 = np.meshgrid(np.degrees(psi0), np.degrees(psi1))
    ax2.contourf(P0, P1, cf_pp, levels=levels, cmap="RdPu")
    ax2.contour(P0, P1, cf_pp, **line_kw)
    ax2.plot(0.0, np.degrees(psi1_o), **STAR)
    ax2.set_xlabel(r"$\psi_0$ (deg)")
    ax2.set_title(rf"(b)  $C_{{F^\ast}}(\psi_0,\psi_1)$ at "
                  rf"$\delta_0={np.degrees(delta0_o):.0f}^\circ$",
                  fontsize=style.font_size)
    fig.colorbar(cf, ax=[ax1, ax2], label=r"$C_{F^\ast}$")
    print(f"C_F* optimum {cf_dp[jmax, imax]:.3f} at "
          f"psi1={np.degrees(psi1_o):.1f} deg, "
          f"delta0={np.degrees(delta0_o):.1f} deg")
    _save(fig, "pitch_efficiency.light.png", dpi=300)


def fig_force_direction(style):
    n = 181
    psi0 = np.linspace(*PSI0_RANGE, n)
    b = np.arctan2(sim.DCD * np.sin(2.0 * psi0),
                   2.0 * sim.CL0 * np.cos(2.0 * psi0))
    beta = np.degrees(np.unwrap(b))
    beta -= beta[n // 2]
    psi0_c, _, cf, _, _ = grid_op(0.0, 61)
    cf_max = cf.max(axis=0)

    fig, ax = plt.subplots(figsize=(4.0, 2.4), constrained_layout=True)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    (l_beta,) = ax.plot(np.degrees(psi0), beta, color="black", lw=1.8,
                        label=r"$\beta$")
    ax.set_xlabel(r"$\psi_0$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    ax.set_xlim(*np.degrees(PSI0_RANGE))
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax2 = ax.twinx()
    (l_cf,) = ax2.plot(np.degrees(psi0_c), cf_max, color="black", lw=1.8,
                       ls=":", label=r"$C_{F^\ast}^{\max}$")
    ax2.set_ylabel(r"$C_{F^\ast}^{\max}$", y=0.75)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks(np.arange(0.0, 1.51, 0.5))
    ax.legend(handles=[l_beta, l_cf], loc="lower right",
              fontsize=style.font_size - 1, frameon=True)
    print(f"beta range [{beta.min():.1f}, {beta.max():.1f}] deg; "
          f"C_F*^max at psi0=0: {cf_max[len(psi0_c) // 2]:.3f}")
    _save(fig, "force_direction_test.light.png")


def fig_maps_J(style):
    grids = [grid_op(J) for J in J_VALUES]
    vmax = np.ceil(max(g[2].max() for g in grids) * 5.0) / 5.0
    levels = np.round(np.arange(0.0, vmax + 1e-9, 0.2), 2)
    line_kw = dict(levels=levels, colors="black", linewidths=0.3, alpha=0.35)

    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.6), sharex=True,
                             sharey=True, constrained_layout=True)
    prev = np.radians(51.0)
    for k, (ax, J, (psi0, psi1, cf, cn, _)) in enumerate(
            zip(axes.flat, J_VALUES, grids)):
        P0, P1 = np.meshgrid(np.degrees(psi0), np.degrees(psi1))
        im = ax.contourf(P0, P1, cf, levels=levels, cmap="RdPu")
        ax.contour(P0, P1, cf, **line_kw)
        cn_plot = cn.copy()
        if np.abs(cn_plot[0]).max() < 1e-9:   # degenerate psi1 = 0 row at J = 0
            cn_plot[0] = np.nan
        ax.contour(P0, P1, cn_plot, levels=[0.0], colors="black",
                   linestyles=":", linewidths=1.0)
        prev, cf_track, is_max = track_optimum(J, prev)
        ax.plot(0.0, np.degrees(prev), **STAR)
        ax.set_title(rf"({chr(ord('a') + k)})  $J = {J:.1f}$",
                     fontsize=style.font_size)
        if k >= 3:
            ax.set_xlabel(r"$\psi_0$ (deg)")
        if k % 3 == 0:
            ax.set_ylabel(r"$\psi_1$ (deg)")
        print(f"J = {J:.1f}: C_F* max = {cf.max():.3f}; optimum at "
              f"psi1 = {np.degrees(prev):.1f} deg, C_F* = {cf_track:.3f} "
              f"({'local max' if is_max else 'saddle'})")
    fig.colorbar(im, ax=axes, label=r"$C_{F^\ast}$")
    _save(fig, "pitch_efficiency_J.light.png", dpi=300)


def fig_optimum_J(style):
    J_fine = np.linspace(0.0, 1.0, 101)
    psi1_c, cf_c, saddle_J = optimum_continuation(J_fine)
    print(f"tracked optimum: psi1 {np.degrees(psi1_c[0]):.1f} -> "
          f"{np.degrees(psi1_c[-1]):.1f} deg, C_F* {cf_c[0]:.3f} -> "
          f"{cf_c[-1]:.3f}; saddle at J = {saddle_J:.2f}")

    fig, ax = plt.subplots(figsize=(4.4, 2.7), constrained_layout=True)
    (l1,) = ax.plot(J_fine, np.degrees(psi1_c), color="black", lw=1.8,
                    label=r"$\psi_1^{\mathrm{opt}}$")
    (l2,) = ax.plot(J_fine,
                    np.degrees(psi1_c[0]) - np.degrees(np.arctan(J_fine)),
                    color="black", lw=1.4, ls="--",
                    label=r"$\psi_1^{\mathrm{opt}}(0) - \tan^{-1} J$")
    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$\psi_1$ (deg)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 60.0)
    ax2 = ax.twinx()
    (l3,) = ax2.plot(J_fine, cf_c, color="black", lw=1.8, ls=":",
                     label=r"$C_{F^\ast}$")
    ax2.set_ylabel(r"$C_{F^\ast}$")
    ax2.set_ylim(0.0, 1.5)
    ax.legend(handles=[l1, l2, l3], loc="lower left",
              fontsize=style.font_size - 1, frameon=True)
    _save(fig, "hover_optimum_J.light.png")


def fig_beta_J(style):
    J_fine = np.linspace(0.0, 1.0, 101)
    psi1_c, _, _ = optimum_continuation(J_fine)
    n = 181
    psi0 = np.linspace(*PSI0_RANGE, n)
    colors = plt.cm.inferno(np.linspace(0.12, 0.82, len(J_VALUES)))

    fig, ax = plt.subplots(figsize=(4.0, 2.6), constrained_layout=True)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    for J, color in zip(J_VALUES, colors):
        psi1_J = psi1_c[int(np.argmin(np.abs(J_fine - J)))]
        beta, cn0 = beta_curve(psi0, J, psi1_J)
        x, y = np.degrees(psi0), beta
        for c in reversed(np.flatnonzero(np.abs(np.diff(y)) > 180.0) + 1):
            x = np.insert(x, c, (x[c], np.nan))
            y = np.insert(y, c, (y[c] - np.sign(y[c] - y[c - 1]) * 360.0,
                                 np.nan))
        ax.plot(x, y, color=color, lw=1.6, label=rf"$J = {J:.1f}$")
        print(f"J = {J:.1f}: psi1 = {np.degrees(psi1_J):.1f} deg, beta range "
              f"[{np.nanmin(beta):.1f}, {np.nanmax(beta):.1f}] deg, "
              f"C_F*_n(0) = {cn0:+.3f}")
    ax.set_xlabel(r"$\psi_0$ (deg)")
    ax.set_ylabel(r"$\beta$ (deg)")
    ax.set_xlim(*np.degrees(PSI0_RANGE))
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_yticks(np.arange(-180, 181, 90))
    ax.legend(fontsize=style.font_size - 2, frameon=True, loc="upper left",
              handlelength=1.4, labelspacing=0.25)
    _save(fig, "force_direction_J.light.png")


# ---------------------------------------------------------------------------
# Closed-loop results.

def _trace_grid(fig):
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.45])
    axT = fig.add_subplot(gs[:, 0])
    rows = []
    for i in range(4):
        rows.append(fig.add_subplot(gs[i, 1], sharex=rows[0] if rows else None))
    return axT, rows


def _finish_rows(rows, style, tmax, zero_lines=True):
    for ax in rows:
        if zero_lines:
            ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(0.0, tmax)
        ax.legend(fontsize=style.font_size - 4, frameon=True,
                  loc="upper right", ncol=3, handlelength=1.3,
                  labelspacing=0.2, columnspacing=0.8)
    for ax in rows[:-1]:
        ax.tick_params(labelbottom=False)
    rows[3].set_xlabel(r"time ($\sqrt{L/g}$)")


def _controls_rows(rows, run):
    t = run["t"]
    rows[2].plot(t, np.degrees(run["gamma"]), color=C1, lw=1.4,
                 label=r"$\gamma$")
    rows[2].plot(t, np.degrees(run["psi0"]), color=C2, lw=1.4, ls="--",
                 label=r"$\psi_0$")
    rows[2].plot(t, np.degrees(run["psi1"]), color=C3, lw=1.4, ls=":",
                 label=r"$\psi_1$")
    rows[2].set_ylabel("angles (deg)")
    rows[3].plot(t, S0_OVER_PHI1 * run["phi1"], color=C1, lw=1.4,
                 label=r"$s_0^*$")
    rows[3].set_ylabel(r"$s_0^*$")


def run_hover():
    run = sim.simulate(lambda t, s: (np.zeros(3), True), traj.T_HOVER,
                       trim="analytic", v0=traj.V0)
    ep = np.hypot(run["x"], run["z"])
    sp = np.hypot(run["vx"], run["vz"])
    print(f"hover: max |x| = {ep.max():.3f}, final |x| = {ep[-1]:.3f}, "
          f"final speed = {sp[-1]:.4f}")
    return run


def fig_hover(run, style):
    fig = plt.figure(figsize=(6.5, 4.6), constrained_layout=True)
    axT, rows = _trace_grid(fig)
    t = run["t"]
    axT.plot(run["x"], run["z"], color=C1, lw=1.4)
    axT.plot(run["x"][0], run["z"][0], "o", color=C1, ms=4, mfc="white")
    axT.plot(run["x"][-1], run["z"][-1], "o", color=C1, ms=4)
    axT.set_xlabel(r"$x^*$ (body lengths)")
    axT.set_ylabel(r"$z^*$ (body lengths)")
    axT.set_title("(a) trajectory", fontsize=style.font_size)
    axT.set_aspect("equal", adjustable="datalim")

    rows[0].plot(t, run["x"], color=C1, lw=1.4, label=r"$x^*$")
    rows[0].plot(t, run["z"], color=C2, lw=1.4, ls="--", label=r"$z^*$")
    rows[0].set_ylabel("position")
    rows[0].set_title("(b) time traces", fontsize=style.font_size)
    rows[1].plot(t, run["vx"], color=C1, lw=1.4, label=r"$u^*_x$")
    rows[1].plot(t, run["vz"], color=C2, lw=1.4, ls="--", label=r"$u^*_z$")
    rows[1].set_ylabel("velocity")
    _controls_rows(rows, run)
    _finish_rows(rows, style, traj.T_HOVER)
    _save(fig, "analytic_trim_hover.light.png")


def run_trajectory(path, v_cruise):
    ref = traj.Reference(path, v_cruise, traj.TAPER, 0.0)
    print(f"path: {ref.total:.2f} body lengths; cruise {v_cruise}, "
          f"taper {traj.TAPER}, reference ends at t = {ref.duration():.2f}")

    def u_ref(t, s):
        _, v2, _, done = ref.sample(t)
        return np.array([v2[0], 0.0, v2[1]]), done

    run = sim.simulate(u_ref, ref.duration() + traj.T_PAD, trim="newton")
    p_ref = np.array([ref.sample(t)[0] for t in run["t"]])
    v_ref = np.array([ref.sample(t)[1] for t in run["t"]])
    run["ex"], run["ez"] = p_ref[:, 0] - run["x"], p_ref[:, 1] - run["z"]
    run["vx_ref"], run["vz_ref"] = v_ref[:, 0], v_ref[:, 1]
    ep = np.hypot(run["ex"], run["ez"])
    follow = run["t"] <= ref.duration()
    print(f"max |e_p| following = {ep[follow].max():.3f}, "
          f"rms = {np.sqrt(np.mean(ep[follow] ** 2)):.3f}, "
          f"final |e_p| = {ep[-1]:.3f}")
    return run


def fig_trajectory(run, path, style, tag=""):
    fig = plt.figure(figsize=(6.5, 4.6), constrained_layout=True)
    axT, rows = _trace_grid(fig)
    t = run["t"]
    axT.plot(path[:, 0], path[:, 1], color="0.65", lw=2.6, alpha=0.6,
             label="reference", zorder=0)
    axT.plot(run["x"], run["z"], color=C1, lw=1.2, label="achieved")
    axT.plot(run["x"][0], run["z"][0], "o", color=C1, ms=4, mfc="white")
    axT.plot(path[-1, 0], path[-1, 1], "+", color=C2, ms=9, mew=1.6, zorder=5)
    axT.set_xlabel(r"$x^*$ (body lengths)")
    axT.set_ylabel(r"$z^*$ (body lengths)")
    axT.set_title("(a) trajectory", fontsize=style.font_size)
    axT.set_aspect("equal", adjustable="datalim")
    axT.legend(fontsize=style.font_size - 3, frameon=True, loc="upper left")

    rows[0].plot(t, run["ex"], color=C1, lw=1.4, label=r"$e_x$")
    rows[0].plot(t, run["ez"], color=C2, lw=1.4, ls="--", label=r"$e_z$")
    rows[0].set_ylabel("error")
    rows[0].set_title("(b) time traces", fontsize=style.font_size)
    rows[1].plot(t, run["vx_ref"], color=C1, lw=2.4, alpha=0.3)
    rows[1].plot(t, run["vx"], color=C1, lw=1.0, label=r"$u^*_x$")
    rows[1].plot(t, run["vz_ref"], color=C2, lw=2.4, alpha=0.3)
    rows[1].plot(t, run["vz"], color=C2, lw=1.0, ls="--", label=r"$u^*_z$")
    rows[1].set_ylabel("velocity")
    _controls_rows(rows, run)
    _finish_rows(rows, style, t[-1])
    _save(fig, f"trajectory_gains{tag}.light.png")


def run_pursuit():
    def u_ref(t, s):
        r = traj.prey_pos(t) - s[0:3]
        rng = np.hypot(r[0], r[2])
        return traj.V_CMD * np.array([r[0], 0.0, r[2]]) / rng, False

    def stop(t, s):
        r = traj.prey_pos(t) - s[0:3]
        return np.hypot(r[0], r[2]) < traj.R_CAP

    run = sim.simulate(u_ref, traj.T_MAX, trim="analytic", stop=stop)
    t = run["t"]
    run["px"] = traj.PREY_P0[0] + traj.PREY_V[0] * t
    run["pz"] = traj.PREY_P0[2] + traj.PREY_V[2] * t
    run["rng"] = np.hypot(run["px"] - run["x"], run["pz"] - run["z"])
    t_cap = t[-1] if run["rng"][-1] < traj.R_CAP else None
    sp = np.hypot(run["vx"], run["vz"])
    cap = f"{t_cap:.2f}" if t_cap is not None else "none"
    print(f"pursuit: capture t = {cap}, intercept "
          f"({run['x'][-1]:.2f}, {run['z'][-1]:.2f}), "
          f"peak speed = {sp.max():.3f}")
    return run


def fig_pursuit(run, style):
    fig = plt.figure(figsize=(6.5, 4.6), constrained_layout=True)
    axT, rows = _trace_grid(fig)
    t = run["t"]
    for j in np.linspace(0, len(t) - 1, 8).astype(int):
        axT.plot([run["x"][j], run["px"][j]], [run["z"][j], run["pz"][j]],
                 color="0.75", lw=0.5, zorder=0)
    axT.plot(run["px"], run["pz"], color=C2, lw=1.6, ls="--", label="prey")
    axT.plot(run["x"], run["z"], color=C1, lw=1.6, label="dragonfly")
    axT.plot(run["x"][0], run["z"][0], "o", color=C1, ms=4, mfc="white")
    axT.plot(run["px"][0], run["pz"][0], "s", color=C2, ms=4, mfc="white")
    axT.plot(run["x"][-1], run["z"][-1], "*", color=C2, ms=11, zorder=5)
    axT.set_xlabel(r"$x^*$ (body lengths)")
    axT.set_ylabel(r"$z^*$ (body lengths)")
    axT.set_title("(a) pursuit", fontsize=style.font_size)
    axT.set_aspect("equal", adjustable="datalim")
    axT.legend(fontsize=style.font_size - 3, frameon=True, loc="lower right")

    rows[0].plot(t, run["rng"], color=C1, lw=1.4, label="range")
    rows[0].axhline(traj.R_CAP, color=C2, lw=0.9, ls=":")
    rows[0].set_ylabel("range")
    rows[0].set_ylim(0.0, None)
    rows[0].set_title("(b) time traces", fontsize=style.font_size)
    rows[1].plot(t, run["vx"], color=C1, lw=1.4, label=r"$u^*_x$")
    rows[1].plot(t, run["vz"], color=C2, lw=1.4, ls="--", label=r"$u^*_z$")
    rows[1].set_ylabel("velocity")
    _controls_rows(rows, run)
    _finish_rows(rows, style, float(t[-1]))
    _save(fig, "analytic_trim_pursuit.light.png")


# ---------------------------------------------------------------------------

FIGS = ("cf_components", "pitch_efficiency", "force_direction", "maps_J",
        "optimum_J", "beta_J", "hover", "trajectory", "pursuit")


def main(argv):
    global OUT_DIR
    names = list(argv)
    if names and names[0] not in FIGS:
        OUT_DIR = Path(names.pop(0))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    only = set(names)

    def want(name):
        return not only or name in only

    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)

    print(f"psi1_opt(0) = {np.degrees(sim.PSI1_HOVER):.2f} deg; "
          f"K T* = 1 (K = {sim.REF.omega_star / (2 * np.pi):.3f})")
    if want("cf_components"):
        fig_cf_components(style)
    if want("pitch_efficiency"):
        fig_pitch_efficiency(style)
    if want("force_direction"):
        fig_force_direction(style)
    if want("maps_J"):
        fig_maps_J(style)
    if want("optimum_J"):
        fig_optimum_J(style)
    if want("beta_J"):
        fig_beta_J(style)
    if want("hover"):
        fig_hover(run_hover(), style)
    if want("trajectory"):
        course = traj.primitive_path()
        fig_trajectory(run_trajectory(course, traj.V_CRUISE), course, style)
        drawn = traj.drawn_path()
        if drawn is not None:
            fig_trajectory(run_trajectory(drawn, traj.V_DRAWN), drawn, style,
                           tag="_drawn")
    if want("pursuit"):
        fig_pursuit(run_pursuit(), style)


if __name__ == "__main__":
    main(sys.argv[1:])

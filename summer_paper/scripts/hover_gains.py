"""Hover recovery from a velocity perturbation under the velocity law.

Exercises the report's control law (sec:control-law) in its hover regime:
u_r = 0, the body starts at the origin with velocity (0.2, 0.2), and the
force demand is K (u_r - u) plus weight compensation (eq:fdes). The
commanded reference velocity is zero throughout, so once the body settles
near rest (J below a small threshold) gamma and psi1 pin to their hover
values and the recovery is carried by the two hover control variables
(s0*, psi0) -- table tab:hover in action.

    hover_gains.light.png -- left, the (x*, z*) trajectory; right, time
        traces of the body state and all four control variables.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import REF_LW, REF_OMEGA_STAR, STUDY_SPAN_FRAC
from maneuver_control import trim_fast, PSI1_LIM
from generalized_control import (
    GAMMA_H, N_PHASE, gamma_schedule, make_p, slave_psi1,
)
from hover_drift import GRAVITY, instant_force

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

K = REF_OMEGA_STAR / (2.0 * np.pi)  # one-beat velocity-law gain, K T* = 1 (eq:fdes)
V0 = (0.2, 0.0, 0.2)               # initial body velocity perturbation
T_SIM = 20.0
J_PIN = 0.05                       # settled-near-rest threshold for the pin
GAMMA_HOVER = -GAMMA_H             # +x heading default (sigma_x = +1)
PSI1_HOVER = float(np.clip(slave_psi1(0.0), *PSI1_LIM))


def allocate(v, F_des, warm):
    """Control-law allocation: gamma/psi1 schedule with hover pin + trim.

    The reference velocity is zero throughout, so the pin engages whenever
    the body has settled near rest (sec:control-law)."""
    gamma, _, J = gamma_schedule(v, sx=1.0)   # latched commanded heading
    if J < J_PIN:
        gamma, ps1 = GAMMA_HOVER, PSI1_HOVER
    else:
        ps1 = float(np.clip(slave_psi1(J), *PSI1_LIM))
    u0 = warm if warm is not None else (np.radians(20.0), 0.0)
    phi1, psi0, _ = trim_fast(gamma, ps1, tuple(v), F_des, u0, N_PHASE)
    return (phi1, psi0, gamma, ps1)


def simulate(T=T_SIM):
    """Unsteady closed-loop hover hold from the perturbed initial condition."""
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)

    s = np.zeros(7)
    s[3:6] = V0
    u = None
    last_ctrl = -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz",
                           "phi1", "psi0", "gamma", "psi1")}

    def deriv(state, u):
        v = state[3:6]
        p = make_p(u[0], u[1], u[2], u[3], (0.0, 0.0, 0.0))
        F = instant_force(p, state[6], omega, v)
        d = np.empty(7)
        d[0:3] = v
        d[3:6] = F + GRAVITY
        d[6] = omega
        return d

    for i in range(n + 1):
        t = i * dt
        if t - last_ctrl >= period - 1e-9:
            e_v = -s[3:6]                        # u_r = 0
            F_des = K * e_v - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u = allocate(s[3:6], F_des, warm)
            last_ctrl = t
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["phi1"].append(u[0]); log["psi0"].append(u[1])
        log["gamma"].append(u[2]); log["psi1"].append(u[3])
        if i < n:
            k1 = deriv(s, u)
            k2 = deriv(s + 0.5 * dt * k1, u)
            k3 = deriv(s + 0.5 * dt * k2, u)
            k4 = deriv(s + dt * k3, u)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(val) for k, val in log.items()}


def figure(run, style):
    c1, c2, c3 = "black", "#b2182b", "#2166ac"
    fig = plt.figure(figsize=(6.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.45])

    # left: trajectory
    axT = fig.add_subplot(gs[:, 0])
    axT.plot(run["x"], run["z"], color=c1, lw=1.4)
    axT.plot(run["x"][0], run["z"][0], "o", color=c1, ms=4, mfc="white")
    axT.plot(run["x"][-1], run["z"][-1], "o", color=c1, ms=4)
    axT.set_xlabel(r"$x^*$ (body lengths)")
    axT.set_ylabel(r"$z^*$ (body lengths)")
    axT.set_title("(a) trajectory", fontsize=style.font_size)
    axT.set_aspect("equal", adjustable="datalim")

    # right: time traces
    rows = []
    for i in range(4):
        ax = fig.add_subplot(gs[i, 1], sharex=rows[0] if rows else None)
        rows.append(ax)
    t = run["t"]

    rows[0].plot(t, run["x"], color=c1, lw=1.4, label=r"$x^*$")
    rows[0].plot(t, run["z"], color=c2, lw=1.4, ls="--", label=r"$z^*$")
    rows[0].set_ylabel("position")
    rows[0].set_title("(b) time traces", fontsize=style.font_size)

    rows[1].plot(t, run["vx"], color=c1, lw=1.4, label=r"$u^*_x$")
    rows[1].plot(t, run["vz"], color=c2, lw=1.4, ls="--", label=r"$u^*_z$")
    rows[1].set_ylabel("velocity")

    rows[2].plot(t, np.degrees(run["gamma"]), color=c1, lw=1.4,
                 label=r"$\gamma$")
    rows[2].plot(t, np.degrees(run["psi0"]), color=c2, lw=1.4, ls="--",
                 label=r"$\psi_0$")
    rows[2].plot(t, np.degrees(run["psi1"]), color=c3, lw=1.4, ls=":",
                 label=r"$\psi_1$")
    rows[2].set_ylabel("angles (deg)")

    s0 = STUDY_SPAN_FRAC * REF_LW * run["phi1"]
    rows[3].plot(t, s0, color=c1, lw=1.4, label=r"$s_0^*$")
    rows[3].set_ylabel(r"$s_0^*$")
    rows[3].set_xlabel(r"time ($\sqrt{L/g}$)")

    for ax in rows:
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(0.0, T_SIM)
        ax.legend(fontsize=style.font_size - 4, frameon=True, loc="upper right",
                  ncol=3, handlelength=1.3, labelspacing=0.2, columnspacing=0.8)
    for ax in rows[:-1]:
        ax.tick_params(labelbottom=False)

    out = OUT_DIR / "hover_gains.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"K = {K}; v0 = ({V0[0]}, {V0[2]}); "
          f"gamma_hover = {np.degrees(GAMMA_HOVER):.0f} deg, "
          f"psi1_hover = {np.degrees(PSI1_HOVER):.1f} deg")
    run = simulate()
    out = figure(run, style)
    ep = np.hypot(run["x"], run["z"])
    sp = np.hypot(run["vx"], run["vz"])
    m = len(run["t"]) // 2
    print(f"max |x| = {ep.max():.3f}, final |x| = {ep[-1]:.3f}, "
          f"final speed = {sp[-1]:.3f}, "
          f"|x| range over 2nd half = [{ep[m:].min():.3f}, {ep[m:].max():.3f}]")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

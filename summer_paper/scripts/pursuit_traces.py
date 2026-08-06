"""Prey pursuit under the report's control law, with control-variable traces.

Pure-pursuit guidance closed through the control law of sec:control-law: the
force demand uses only the velocity term of eq:fdes, with the reference
velocity defined as the desired closing speed times the line-of-sight
direction to the prey,

    u_r = u_cmd * r_hat,      F_des = K_d (u_r - u) + z_hat.

The prey moves at constant velocity; interception is declared at range 0.5
body lengths. Companion to hover_gains.py (same layout): left, the pursuit
geometry; right, time traces of range, body velocity and all four control
variables.

Output: pursuit_traces.light.png in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import REF_LW, REF_OMEGA_STAR, STUDY_SPAN_FRAC
from generalized_control import make_p, proposed_allocate
from hover_drift import GRAVITY, instant_force

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

KD = REF_OMEGA_STAR / (2.0 * np.pi)  # one-beat velocity gain, K T* = 1 (report control law)
V_CMD = 1.5                         # desired closing speed
R_CAP = 0.5                         # capture radius (body lengths)
T_MAX = 22.0
PREY_P0 = (4.0, 0.0, 4.0)
PREY_V = (1.0, 0.0, 0.0)


def prey_pos(t):
    return np.array([PREY_P0[0] + PREY_V[0] * t, 0.0,
                     PREY_P0[2] + PREY_V[2] * t])


def simulate():
    """Unsteady closed-loop pursuit; control once per wingbeat."""
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T_MAX / dt)

    s = np.zeros(7)
    u = None
    last_ctrl = -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz", "px", "pz", "rng",
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

    t_cap = None
    for i in range(n + 1):
        t = i * dt
        pp = prey_pos(t)
        r = pp - s[0:3]
        rng = float(np.hypot(r[0], r[2]))
        if t - last_ctrl >= period - 1e-9:
            v = s[3:6]
            los = np.array([r[0], 0.0, r[2]]) / rng
            u_r = V_CMD * los                     # closing speed along the LOS
            F_des = KD * (u_r - v) - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u = proposed_allocate(v, F_des, u_r, np.zeros(3), warm=warm)
            last_ctrl = t
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["px"].append(pp[0]); log["pz"].append(pp[2]); log["rng"].append(rng)
        log["phi1"].append(u[0]); log["psi0"].append(u[1])
        log["gamma"].append(u[2]); log["psi1"].append(u[3])
        if rng < R_CAP:
            t_cap = t
            break
        if i < n:
            k1 = deriv(s, u)
            k2 = deriv(s + 0.5 * dt * k1, u)
            k3 = deriv(s + 0.5 * dt * k2, u)
            k4 = deriv(s + dt * k3, u)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(val) for k, val in log.items()}, t_cap


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run, t_cap = simulate()
    t = run["t"]
    c1, c2, c3 = "black", "#b2182b", "#2166ac"

    fig = plt.figure(figsize=(6.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.45])

    # left: pursuit geometry with line-of-sight rays
    axT = fig.add_subplot(gs[:, 0])
    idx = np.linspace(0, len(t) - 1, 8).astype(int)
    for j in idx:
        axT.plot([run["x"][j], run["px"][j]], [run["z"][j], run["pz"][j]],
                 color="0.75", lw=0.5, zorder=0)
    axT.plot(run["px"], run["pz"], color=c2, lw=1.6, ls="--", label="prey")
    axT.plot(run["x"], run["z"], color=c1, lw=1.6, label="dragonfly")
    axT.plot(run["x"][0], run["z"][0], "o", color=c1, ms=4, mfc="white")
    axT.plot(run["px"][0], run["pz"][0], "s", color=c2, ms=4, mfc="white")
    axT.plot(run["x"][-1], run["z"][-1], "*", color=c2, ms=11, zorder=5)
    axT.set_xlabel(r"$x^*$ (body lengths)")
    axT.set_ylabel(r"$z^*$ (body lengths)")
    axT.set_title("(a) pursuit", fontsize=style.font_size)
    axT.set_aspect("equal", adjustable="datalim")
    axT.legend(fontsize=style.font_size - 3, frameon=True, loc="lower right")

    # right: time traces
    rows = []
    for i in range(4):
        ax = fig.add_subplot(gs[i, 1], sharex=rows[0] if rows else None)
        rows.append(ax)

    rows[0].plot(t, run["rng"], color=c1, lw=1.4, label="range")
    rows[0].axhline(R_CAP, color=c2, lw=0.9, ls=":")
    rows[0].set_ylabel("range")
    rows[0].set_ylim(0.0, None)
    rows[0].set_title("(b) time traces", fontsize=style.font_size)

    rows[1].plot(t, run["vx"], color=c1, lw=1.4, label=r"$u^*_x$")
    rows[1].plot(t, run["vz"], color=c2, lw=1.4, ls="--", label=r"$u^*_z$")
    rows[1].axhline(0.0, color="0.85", lw=0.8, zorder=0)
    rows[1].set_ylabel("velocity")

    rows[2].plot(t, np.degrees(run["gamma"]), color=c1, lw=1.4,
                 label=r"$\gamma$")
    rows[2].plot(t, np.degrees(run["psi0"]), color=c2, lw=1.4, ls="--",
                 label=r"$\psi_0$")
    rows[2].plot(t, np.degrees(run["psi1"]), color=c3, lw=1.4, ls=":",
                 label=r"$\psi_1$")
    rows[2].axhline(0.0, color="0.85", lw=0.8, zorder=0)
    rows[2].set_ylabel("angles (deg)")

    s0 = STUDY_SPAN_FRAC * REF_LW * run["phi1"]
    rows[3].plot(t, s0, color=c1, lw=1.4, label=r"$s_0^*$")
    rows[3].axhline(0.0, color="0.85", lw=0.8, zorder=0)
    rows[3].set_ylabel(r"$s_0^*$")
    rows[3].set_xlabel(r"time ($\sqrt{L/g}$)")

    for ax in rows:
        ax.set_xlim(0.0, float(t[-1]))
        ax.legend(fontsize=style.font_size - 4, frameon=True,
                  loc="upper right", ncol=3, handlelength=1.3,
                  labelspacing=0.2, columnspacing=0.8)
    for ax in rows[:-1]:
        ax.tick_params(labelbottom=False)

    out = OUT_DIR / "pursuit_traces.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    sp = np.hypot(run["vx"], run["vz"])
    print(f"Kd = {KD}, u_cmd = {V_CMD}, capture radius = {R_CAP}")
    print(f"capture at t = {t_cap:.2f}, intercept "
          f"({run['x'][-1]:.2f}, {run['z'][-1]:.2f}); "
          f"peak speed = {sp.max():.3f} (J = {sp.max() / 3.5:.2f}); "
          f"gamma range [{np.degrees(run['gamma'].min()):.0f}, "
          f"{np.degrees(run['gamma'].max()):.0f}] deg, "
          f"psi1 range [{np.degrees(run['psi1'].min()):.0f}, "
          f"{np.degrees(run['psi1'].max()):.0f}] deg")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""Prescribed-trajectory tracking: Kp, Kd, Kp+Kd, and Kp+Kd+feedforward.

The reference path is an upward bend: a horizontal run, a circular arc of
radius R bending up by 45 deg, and a straight 45-deg climb, traversed with a
trapezoidal speed schedule (linear ramps from/to rest over the taper distance,
constant cruise speed in between). The reference acceleration a_r carries the
tangential (ramp) and centripetal (arc) demand exactly.

Four outer-loop cases close the report's control law (sec:control-law):
    kp    -- a_des = Kp e_p                      (position feedback only)
    kd    -- a_des = Kd e_v                      (velocity feedback only)
    kpkd  -- a_des = Kp e_p + Kd e_v             (no feedforward)
    ff    -- a_des = a_r + Kp e_p + Kd e_v       (full law, eq:fdes)

Companion to hover_gains.py (same layout): left, the (x*, z*) trajectory over
the reference path; right, time traces of tracking error, velocity and all
four control variables.
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

KP, KD = 2.25, 2.7                 # outer-loop gains (as in the hover section)
T_SIM = 12.0
HOLD_PIN_RADIUS = 0.3              # sec:control-law hover pin
GAMMA_HOVER = -GAMMA_H             # +x heading (sigma_x = +1 throughout)
PSI1_HOVER = float(np.clip(slave_psi1(0.0), *PSI1_LIM))

# Path geometry: horizontal run, radius-R arc up by BEND, straight climb.
L1, R, BEND, L2 = 1.5, 2.0, np.radians(45.0), 1.5
V_CRUISE, TAPER = 0.7, 0.3

CASES = [
    ("kp",   KP,  0.0, False, r"$K_p$ only"),
    ("kd",   0.0, KD,  False, r"$K_d$ only"),
    ("kpkd", KP,  KD,  False, r"$K_p + K_d$"),
    ("ff",   KP,  KD,  True,  r"$K_p + K_d + \vec{a}^*_r$"),
]


# ---------------------------------------------------------------------------
# Analytical reference: exact position/tangent/curvature + trapezoid speed.

L_ARC = R * BEND
TOTAL = L1 + L_ARC + L2
ACC = V_CRUISE ** 2 / (2.0 * TAPER)          # ramp acceleration (= v^2 / 2d)
T_RAMP = V_CRUISE / ACC
T_CRUISE = (TOTAL - 2.0 * TAPER) / V_CRUISE
T_END = 2.0 * T_RAMP + T_CRUISE              # reference exhausted


def _geom(s):
    """Position, unit tangent, and dT/ds at arc length s (2D, (x, z))."""
    if s <= L1:
        return np.array([s, 0.0]), np.array([1.0, 0.0]), np.zeros(2)
    if s <= L1 + L_ARC:
        th = (s - L1) / R
        p = np.array([L1 + R * np.sin(th), R * (1.0 - np.cos(th))])
        return p, np.array([np.cos(th), np.sin(th)]), \
            np.array([-np.sin(th), np.cos(th)]) / R
    d = s - L1 - L_ARC
    e = np.array([np.cos(BEND), np.sin(BEND)])
    p0 = np.array([L1 + R * np.sin(BEND), R * (1.0 - np.cos(BEND))])
    return p0 + d * e, e, np.zeros(2)


def _speed(tau):
    """Trapezoid speed magnitude, its time-rate, and arc length at time tau."""
    if tau <= T_RAMP:
        return ACC * tau, ACC, 0.5 * ACC * tau * tau
    if tau <= T_RAMP + T_CRUISE:
        return V_CRUISE, 0.0, TAPER + V_CRUISE * (tau - T_RAMP)
    u = tau - T_RAMP - T_CRUISE
    return max(V_CRUISE - ACC * u, 0.0), -ACC, \
        TAPER + (TOTAL - 2.0 * TAPER) + V_CRUISE * u - 0.5 * ACC * u * u


def reference(t):
    """(p_ref, v_ref, a_ref, done) as 3-vectors (y = 0)."""
    if t >= T_END:
        p, _, _ = _geom(TOTAL)
        return np.array([p[0], 0.0, p[1]]), np.zeros(3), np.zeros(3), True
    vmag, dvdt, s = _speed(t)
    p, T, dTds = _geom(min(s, TOTAL))
    v = vmag * T
    a = dvdt * T + vmag ** 2 * dTds
    return (np.array([p[0], 0.0, p[1]]), np.array([v[0], 0.0, v[1]]),
            np.array([a[0], 0.0, a[1]]), False)


# ---------------------------------------------------------------------------

def allocate(v, F_des, done, e_p_norm, warm):
    """Control-law allocation: schedule while following, pin once settled.

    The stroke-plane lean sign is latched at sigma_x = +1 (the commanded
    heading is +x throughout), per sec:control-law."""
    if done and e_p_norm < HOLD_PIN_RADIUS:
        gamma, ps1 = GAMMA_HOVER, PSI1_HOVER
    else:
        gamma, _, J = gamma_schedule(v, sx=1.0)
        ps1 = float(np.clip(slave_psi1(J), *PSI1_LIM))
    u0 = warm if warm is not None else (np.radians(20.0), 0.0)
    phi1, psi0, _ = trim_fast(gamma, ps1, tuple(v), F_des, u0, N_PHASE)
    return (phi1, psi0, gamma, ps1)


def simulate(Kp, Kd, ff, T=T_SIM):
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)

    s = np.zeros(7)
    u = None
    last_ctrl = -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz", "ex", "ez",
                           "vx_ref", "vz_ref",
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
        p_ref, v_ref, a_ref, done = reference(t)
        e_p = p_ref - s[0:3]
        e_v = v_ref - s[3:6]
        if t - last_ctrl >= period - 1e-9:
            a_des = Kp * e_p + Kd * e_v + (a_ref if ff else 0.0)
            F_des = a_des - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u = allocate(s[3:6], F_des, done,
                         float(np.hypot(e_p[0], e_p[2])), warm)
            last_ctrl = t
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["ex"].append(e_p[0]); log["ez"].append(e_p[2])
        log["vx_ref"].append(v_ref[0]); log["vz_ref"].append(v_ref[2])
        log["phi1"].append(u[0]); log["psi0"].append(u[1])
        log["gamma"].append(u[2]); log["psi1"].append(u[3])
        if i < n:
            k1 = deriv(s, u)
            k2 = deriv(s + 0.5 * dt * k1, u)
            k3 = deriv(s + 0.5 * dt * k2, u)
            k4 = deriv(s + dt * k3, u)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(val) for k, val in log.items()}


def figure(tag, run, style):
    c1, c2, c3 = "black", "#b2182b", "#2166ac"
    fig = plt.figure(figsize=(6.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.45])

    # left: trajectory over the reference path
    axT = fig.add_subplot(gs[:, 0])
    s_grid = np.linspace(0.0, TOTAL, 200)
    P = np.array([_geom(si)[0] for si in s_grid])
    axT.plot(P[:, 0], P[:, 1], color="0.65", lw=2.6, alpha=0.6,
             label="reference", zorder=0)
    axT.plot(run["x"], run["z"], color=c1, lw=1.2, label="achieved")
    axT.plot(run["x"][0], run["z"][0], "o", color=c1, ms=4, mfc="white")
    axT.plot(P[-1, 0], P[-1, 1], "+", color=c2, ms=9, mew=1.6, zorder=5)
    axT.set_xlabel(r"$x^*$ (body lengths)")
    axT.set_ylabel(r"$z^*$ (body lengths)")
    axT.set_title("(a) trajectory", fontsize=style.font_size)
    axT.set_aspect("equal", adjustable="datalim")
    axT.legend(fontsize=style.font_size - 3, frameon=True, loc="upper left")

    # right: time traces
    rows = []
    for i in range(4):
        ax = fig.add_subplot(gs[i, 1], sharex=rows[0] if rows else None)
        rows.append(ax)
    t = run["t"]

    rows[0].plot(t, run["ex"], color=c1, lw=1.4, label=r"$e_x$")
    rows[0].plot(t, run["ez"], color=c2, lw=1.4, ls="--", label=r"$e_z$")
    rows[0].set_ylabel("error")
    rows[0].set_title("(b) time traces", fontsize=style.font_size)

    rows[1].plot(t, run["vx_ref"], color=c1, lw=2.4, alpha=0.3)
    rows[1].plot(t, run["vx"], color=c1, lw=1.0, label=r"$u^*_x$")
    rows[1].plot(t, run["vz_ref"], color=c2, lw=2.4, alpha=0.3)
    rows[1].plot(t, run["vz"], color=c2, lw=1.0, ls="--", label=r"$u^*_z$")
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
        ax.legend(fontsize=style.font_size - 4, frameon=True,
                  loc="upper right", ncol=3, handlelength=1.3,
                  labelspacing=0.2, columnspacing=0.8)
    for ax in rows[:-1]:
        ax.tick_params(labelbottom=False)

    out = OUT_DIR / f"trajectory_gains_{tag}.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"path: {L1} + arc(R={R}, {np.degrees(BEND):.0f} deg) + {L2} "
          f"= {TOTAL:.2f} body lengths; cruise {V_CRUISE}, taper {TAPER}, "
          f"reference ends at t = {T_END:.2f}")
    print(f"gains: Kp = {KP}, Kd = {KD}")
    for tag, Kp, Kd, ff, label in CASES:
        run = simulate(Kp, Kd, ff)
        out = figure(tag, run, style)
        ep = np.hypot(run["ex"], run["ez"])
        follow = run["t"] <= T_END
        print(f"{label:>22}: max |e_p| following = {ep[follow].max():.3f}, "
              f"rms = {np.sqrt(np.mean(ep[follow] ** 2)):.3f}, "
              f"final |e_p| = {ep[-1]:.3f}")
        print(f"  wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

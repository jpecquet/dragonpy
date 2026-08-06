"""Hover and pursuit test cases with the ANALYTICAL normal-inflow trim.

The report's control law meets the force demand by a damped Newton solve on the
simulated cycle-averaged force at the measured (generally oblique) inflow
(sec:force_demand). This script swaps that inner solve for the analytical
normal-inflow expressions of sec:maf: the split trim of eq:trim-split, with
C_Fn, C_Fs from eq:cfn-general-op / eq:cfs-general-op (I_0, I_n, I_s evaluated
by quadrature over the wingbeat), the direction demand inverted for psi0 in
closed form (harmonic equation), and s0* explicit from the magnitude demand.
psi1 is slaved to the analytic argmax of C_Fn at psi0 = 0 (same definition as
eq:psi1-slave); the gamma schedule is unchanged.

The closed-loop unsteady simulations (RK4 on the true instantaneous force,
control once per wingbeat) are identical between the two controllers -- only
the once-per-beat (s0*, psi0) allocation differs. The hover case runs to
t = 10. Outputs are STANDALONE analytical-trim figures, in the exact style of
the numerical-trim figures of hover_gains.py and pursuit_traces.py:

    analytic_trim_hover.light.png   -- hover recovery (hover_gains.py case)
    analytic_trim_pursuit.light.png -- prey pursuit (pursuit_traces.py case)

The numerical-vs-analytical comparison is reported in the printed metrics. A
third allocator, `pursuit_allocate_selfJ`, replaces the frozen reference-J of
eq:trim-split with a scalar fixed point on the J the wing actually sees
(J = U_n / (s0* omega*) at the trimmed s0*); its closed-loop metrics are
printed but it is not drawn.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, STUDY_SPAN_FRAC,
    CL_0, CD_0, CD_90, cycle_averaged_force,
)
from maneuver_control import trim_fast, PHI1_LIM, PSI0_LIM, PSI1_LIM
from generalized_control import (
    N_PHASE, SCALE, gamma_schedule, make_p, slave_psi1, proposed_allocate,
    _flight_dir,
)
from hover_drift import GRAVITY, instant_force
import hover_gains
import pursuit_traces

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

T_HOVER = 10.0                     # hover-case horizon (hover_gains uses 20)

DCD = CD_90 - CD_0                 # Delta C_D
CDBAR = 0.5 * (CD_90 + CD_0)       # C_D bar
S0_OVER_PHI1 = STUDY_SPAN_FRAC * REF_LW   # s0* = (2/3) Lw phi1

# Midpoint phase grid: avoids sin(tau) = 0 exactly, where the J = 0
# integrands are 0/0 (they are finite in the limit).
_NTAU = 2048
_TAU = (np.arange(_NTAU) + 0.5) * (2.0 * np.pi / _NTAU)
_ST = np.sin(_TAU)

_PSI1_GRID = np.radians(np.arange(5.0, 90.01, 0.25))


def averages(J, psi1):
    """I_0, I_n, I_s of eqs cfn/cfs-general-op; psi1 may be an array."""
    psi1 = np.asarray(psi1, float)
    st = _ST
    s2 = st * st
    den = np.sqrt(J * J + s2)
    c = np.cos(2.0 * psi1[..., None] * st)
    s = np.sin(2.0 * psi1[..., None] * st)
    I0 = float(den.mean())
    In = (J * (2.0 * CL_0 * s2 + 0.5 * DCD * (J * J - s2)) / den * c).mean(-1) \
        - ((CL_0 * (J * J - s2) - DCD * J * J) * st / den * s).mean(-1)
    Is = (J * (CL_0 * (J * J - s2) + DCD * s2) / den * c).mean(-1) \
        + ((2.0 * CL_0 * J * J - 0.5 * DCD * (J * J - s2)) * st / den * s).mean(-1)
    return I0, In, Is


def psi1_opt(J):
    """Analytic psi1 slave: argmax of C_Fn at psi0 = 0, i.e. argmax I_n."""
    _, In, _ = averages(J, _PSI1_GRID)
    return float(np.clip(_PSI1_GRID[int(np.argmax(In))], *PSI1_LIM))


def analytic_trim(gamma, psi1, J, F_des):
    """(phi1, psi0) from the split normal-inflow trim (eq:trim-split).

    Direction: the harmonic equation a sin(2 psi0) + b cos(2 psi0) = c from
    C_Fn F_s^des = C_Fs F_n^des, solved in closed form; of the two roots, keep
    the one with positive projection on the demand and the smaller |psi0|.
    Magnitude: q* = F^des / C_F*, then s0* (and phi1) explicit."""
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    fxz = np.array([F_des[0], F_des[2]])
    Fn, Fs = float(fxz @ n_hat), float(fxz @ s_hat)
    Fmag = float(np.hypot(Fn, Fs))
    if Fmag < 1e-9:
        return PHI1_LIM[0], 0.0

    I0, In, Is = averages(J, np.asarray(psi1))
    In, Is = float(In), float(Is)

    a, b, c = Is * Fn, In * Fs, J * CDBAR * I0 * Fs
    R = np.hypot(a, b)
    best = None
    if R < 1e-12:
        cands = [0.0]
    else:
        asn = np.arcsin(np.clip(c / R, -1.0, 1.0))
        ph = np.arctan2(b, a)
        cands = [asn - ph, np.pi - asn - ph]
    for t in cands:
        t = (t + np.pi) % (2.0 * np.pi) - np.pi
        CFn = 2.0 * In * np.cos(t) - 2.0 * J * CDBAR * I0
        CFs = -2.0 * Is * np.sin(t)
        proj = CFn * Fn + CFs * Fs
        cand = (proj, -abs(t), t, CFn, CFs)
        if best is None or (proj > 0) > (best[0] > 0) or \
           ((proj > 0) == (best[0] > 0) and cand[1] > best[1]):
            best = cand
    _, _, t, CFn, CFs = best
    psi0 = float(np.clip(0.5 * t, *PSI0_LIM))
    CF = max(np.hypot(CFn, CFs), 1e-9)
    q_star = Fmag / CF
    s0 = np.sqrt(4.0 * q_star / REF_A_OVER_M) / REF_OMEGA_STAR
    phi1 = float(np.clip(s0 / S0_OVER_PHI1, *PHI1_LIM))
    return phi1, psi0


# ---------------------------------------------------------------------------
# Allocators. Same schedules and pin logic as the numerical controllers; only
# the (phi1, psi0) allocation is analytic.

PSI1_HOVER_AN = psi1_opt(0.0)


def hover_allocate_an(v, F_des, warm=None):
    gamma, _, J = gamma_schedule(v, sx=1.0)
    if J < hover_gains.J_PIN:
        gamma, ps1 = hover_gains.GAMMA_HOVER, PSI1_HOVER_AN
    else:
        ps1 = psi1_opt(J)
    phi1, psi0 = analytic_trim(gamma, ps1, J, F_des)
    return (phi1, psi0, gamma, ps1)


def hover_allocate_num(v, F_des, warm=None):
    return hover_gains.allocate(v, F_des, warm)


def pursuit_allocate_an(v, F_des, v_ref, warm=None):
    d = _flight_dir(v, v_ref, np.zeros(3))
    speed = max(np.hypot(v[0], v[2]), 1e-3 * SCALE)
    v_sched = (speed * d[0], 0.0, speed * d[1])
    gamma, _, Js = gamma_schedule(v_sched)
    ps1 = psi1_opt(Js)
    phi1, psi0 = analytic_trim(gamma, ps1, Js, F_des)
    return (phi1, psi0, gamma, ps1)


def pursuit_allocate_num(v, F_des, v_ref, warm=None):
    return proposed_allocate(v, F_des, v_ref, np.zeros(3), warm=warm)


def pursuit_allocate_selfJ(v, F_des, v_ref, warm=None):
    """Analytic trim with J self-consistent at the trimmed amplitude.

    Instead of freezing J at the reference wingbeat velocity (eq:trim-split),
    the trim's J is iterated to the advance ratio the wing actually sees,
    J = U_n / (s0* omega*) with U_n the axial inflow component and s0* the
    trimmed amplitude (a scalar fixed point, converges in a few steps). The
    schedules (gamma, psi1) stay on the reference J as in the report. This
    removes the frozen-J error; the oblique-inflow error of the gamma-blend
    transition band remains."""
    d = _flight_dir(v, v_ref, np.zeros(3))
    speed = max(np.hypot(v[0], v[2]), 1e-3 * SCALE)
    v_sched = (speed * d[0], 0.0, speed * d[1])
    gamma, _, Jref = gamma_schedule(v_sched)
    ps1 = psi1_opt(Jref)
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])
    Un = abs(v[0] * n_hat[0] + v[2] * n_hat[1])
    J = Jref
    for _ in range(20):
        phi1, psi0 = analytic_trim(gamma, ps1, J, F_des)
        Jn = Un / (S0_OVER_PHI1 * phi1 * REF_OMEGA_STAR)
        if abs(Jn - J) < 1e-6:
            break
        J = 0.5 * (J + Jn)
    return (phi1, psi0, gamma, ps1)


# ---------------------------------------------------------------------------
# Closed-loop simulations, numerics identical to hover_gains / pursuit_traces.

def _deriv(state, u, omega):
    v = state[3:6]
    p = make_p(u[0], u[1], u[2], u[3], (0.0, 0.0, 0.0))
    F = instant_force(p, state[6], omega, v)
    d = np.empty(7)
    d[0:3] = v
    d[3:6] = F + GRAVITY
    d[6] = omega
    return d


def _rk4(s, u, dt, omega):
    k1 = _deriv(s, u, omega)
    k2 = _deriv(s + 0.5 * dt * k1, u, omega)
    k3 = _deriv(s + 0.5 * dt * k2, u, omega)
    k4 = _deriv(s + dt * k3, u, omega)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_hover(allocate, T=T_HOVER):
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)
    s = np.zeros(7)
    s[3:6] = hover_gains.V0
    u, last_ctrl = None, -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz",
                           "phi1", "psi0", "gamma", "psi1")}
    for i in range(n + 1):
        t = i * dt
        if t - last_ctrl >= period - 1e-9:
            F_des = hover_gains.K * (-s[3:6]) - GRAVITY   # u_r = 0
            warm = (u[0], u[1]) if u is not None else None
            u = allocate(s[3:6], F_des, warm)
            last_ctrl = t
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["phi1"].append(u[0]); log["psi0"].append(u[1])
        log["gamma"].append(u[2]); log["psi1"].append(u[3])
        if i < n:
            s = _rk4(s, u, dt, omega)
    return {k: np.array(v) for k, v in log.items()}


def simulate_pursuit(allocate):
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(pursuit_traces.T_MAX / dt)
    s = np.zeros(7)
    u, last_ctrl = None, -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz", "px", "pz", "rng",
                           "phi1", "psi0", "gamma", "psi1")}
    t_cap = None
    for i in range(n + 1):
        t = i * dt
        pp = pursuit_traces.prey_pos(t)
        r = pp - s[0:3]
        rng = float(np.hypot(r[0], r[2]))
        if t - last_ctrl >= period - 1e-9:
            los = np.array([r[0], 0.0, r[2]]) / rng
            u_r = pursuit_traces.V_CMD * los
            F_des = pursuit_traces.KD * (u_r - s[3:6]) - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u = allocate(s[3:6], F_des, u_r, warm=warm)
            last_ctrl = t
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["px"].append(pp[0]); log["pz"].append(pp[2]); log["rng"].append(rng)
        log["phi1"].append(u[0]); log["psi0"].append(u[1])
        log["gamma"].append(u[2]); log["psi1"].append(u[3])
        if rng < pursuit_traces.R_CAP:
            t_cap = t
            break
        if i < n:
            s = _rk4(s, u, dt, omega)
    return {k: np.array(v) for k, v in log.items()}, t_cap


# ---------------------------------------------------------------------------
# Static trim comparison: same demand, same schedules, trim side by side.

def static_table():
    print("static trim to weight-up, numerical (Newton, oblique inflow) vs "
          "analytical (normal-inflow split):")
    print(f"{'case':>10} | {'s0 num':>7} {'s0 an':>7} | {'psi0 num':>8} "
          f"{'psi0 an':>7} | {'psi1 num':>8} {'psi1 an':>7} | {'|dF| an':>7}")
    cases = [("hover", 0.0), ("fwd 0.2", 0.2), ("fwd 0.4", 0.4),
             ("fwd 0.6", 0.6), ("fwd 0.8", 0.8)]
    warm = None
    for name, Jx in cases:
        v = (Jx * SCALE, 0.0, 0.0)
        un = hover_allocate_num(v, -GRAVITY, warm)
        warm = (un[0], un[1])
        ua = hover_allocate_an(v, -GRAVITY)
        Fa = cycle_averaged_force(make_p(*ua, v), N_PHASE)
        dF = np.hypot(Fa[0], Fa[2] - 1.0)
        print(f"{name:>10} | {S0_OVER_PHI1 * un[0]:7.3f} {S0_OVER_PHI1 * ua[0]:7.3f} | "
              f"{np.degrees(un[1]):8.1f} {np.degrees(ua[1]):7.1f} | "
              f"{np.degrees(un[3]):8.1f} {np.degrees(ua[3]):7.1f} | {dF:7.3f}")


# ---------------------------------------------------------------------------
# Figures: standalone analytical-trim runs, exact same style as the
# numerical-trim figures (hover_gains.figure and pursuit_traces.main).

def figure_hover(run, style):
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

    s0 = S0_OVER_PHI1 * run["phi1"]
    rows[3].plot(t, s0, color=c1, lw=1.4, label=r"$s_0^*$")
    rows[3].set_ylabel(r"$s_0^*$")
    rows[3].set_xlabel(r"time ($\sqrt{L/g}$)")

    for ax in rows:
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(0.0, T_HOVER)
        ax.legend(fontsize=style.font_size - 4, frameon=True, loc="upper right",
                  ncol=3, handlelength=1.3, labelspacing=0.2, columnspacing=0.8)
    for ax in rows[:-1]:
        ax.tick_params(labelbottom=False)

    out = OUT_DIR / "analytic_trim_hover.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_pursuit(run, style):
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
    rows[0].axhline(pursuit_traces.R_CAP, color=c2, lw=0.9, ls=":")
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

    s0 = S0_OVER_PHI1 * run["phi1"]
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

    out = OUT_DIR / "analytic_trim_pursuit.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"analytic psi1_opt(0) = {np.degrees(PSI1_HOVER_AN):.1f} deg "
          f"(numerical slave: {np.degrees(hover_gains.PSI1_HOVER):.1f} deg)")
    static_table()

    print("\nhover recovery (numerical)...")
    hov_num = simulate_hover(hover_allocate_num)
    print("hover recovery (analytical)...")
    hov_an = simulate_hover(hover_allocate_an)
    out1 = figure_hover(hov_an, style)
    for tag, run in (("num", hov_num), ("an ", hov_an)):
        ep = np.hypot(run["x"], run["z"])
        sp = np.hypot(run["vx"], run["vz"])
        print(f"  {tag}: max |x| = {ep.max():.3f}, final |x| = {ep[-1]:.3f}, "
              f"final speed = {sp[-1]:.4f}")
    print(f"wrote {out1.relative_to(REPO_ROOT)}")

    print("\npursuit (numerical)...")
    pur_num, cap_num = simulate_pursuit(pursuit_allocate_num)
    print("pursuit (analytical)...")
    pur_an, cap_an = simulate_pursuit(pursuit_allocate_an)
    out2 = figure_pursuit(pur_an, style)
    print("pursuit (analytical, self-consistent J)...")
    pur_sj, cap_sj = simulate_pursuit(pursuit_allocate_selfJ)
    for tag, run, cap in (("num", pur_num, cap_num), ("an ", pur_an, cap_an),
                          ("sJ ", pur_sj, cap_sj)):
        sp = np.hypot(run["vx"], run["vz"])
        cap_s = f"{cap:.2f}" if cap is not None else "none"
        print(f"  {tag}: capture t = {cap_s}, intercept "
              f"({run['x'][-1]:.2f}, {run['z'][-1]:.2f}), "
              f"peak speed = {sp.max():.3f}")
    print(f"wrote {out2.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

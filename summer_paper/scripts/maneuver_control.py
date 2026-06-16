"""
Maneuvering flight: power-optimal trim over the velocity plane (report1, section 3.2).

This is part A of the maneuvering controller: the velocity-dependent equilibrium
that the controller is built around. With no body drag, steady straight flight at
*any* body velocity needs the same net force as hover -- the weight, pointing up.
What changes with speed is only that the wing sees inflow, so the controls that
produce that force change. We resolve the control redundancy by minimizing the
cycle-averaged aerodynamic power (the operative form of the cost of endurance
P*, which differs only by the constant weight-support normalization).

Control vector (delta0 pinned at +90 deg, the optimum everywhere):
    u = (phi1, psi0, gamma0, psi1)
Two of these (phi1, psi0) are the fast force-tracking handles inherited from the
hover controller; the other two (gamma0, psi1) are the redundant DOF we set by
power-optimality. Splitting the allocation this way -- slow power-optimal schedule
(gamma0*, psi1*)(v) plus fast (phi1, psi0) feedback -- keeps the inner loop a 2x2
inversion rather than a per-step nonlinear program.

Body velocity is given in the (planar) world frame v = (vx, 0, vz); with attitude
frozen the body does not rotate, so the body frame the force law uses coincides
with the world frame. We also report the stroke-frame parameterization of the
inflow, advance ratio J = |v|/(s0* omega*) and angle chi from the stroke-plane
normal, to connect to the C_psi(J, chi) maps of section 3.1.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import numpy as np

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, cycle_averaged_force, replace,
)
from power_efficiency import cycle_averaged_power, cost_of_endurance

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = HERE.parent / "figures"
N_PHASE = 128
WEIGHT_UP = np.array([0.0, 0.0, 1.0])
DELTA0 = np.pi / 2.0

# Control bounds (Table 1 ranges).
PHI1_LIM = np.radians((2.0, 60.0))
PSI0_LIM = np.radians((-60.0, 60.0))
PSI1_LIM = np.radians((5.0, 60.0))
GAMMA0_LIM = np.radians((0.0, 70.0))


def make_p(phi1, psi0, gamma0, psi1, v_body):
    """Reference-morphology Params at control u and body velocity v_body."""
    return Params(
        Lw=REF_LW, A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR,
        gamma0=float(gamma0), phi1=float(phi1), psi0=float(psi0),
        psi1=float(psi1), delta0=DELTA0, sigma0=np.pi,
        v_body=(float(v_body[0]), float(v_body[1]), float(v_body[2])),
        element_span_fracs=STUDY_ELEMENT)


def _force_xz(phi1, psi0, gamma0, psi1, v_body, n_phase=N_PHASE):
    F = cycle_averaged_force(make_p(phi1, psi0, gamma0, psi1, v_body), n_phase)
    return np.array([F[0], F[2]])


def trim_fast(gamma0, psi1, v_body, F_des, u0, n_phase=N_PHASE, iters=40):
    """Solve (phi1, psi0) so the cycle-averaged force matches F_des (x,z).

    2x2 damped Newton with finite-difference Jacobian, the velocity-aware
    generalization of the hover (phi1, psi0) trim. `u0 = (phi1, psi0)` seeds the
    iteration (warm-started by continuation along velocity). Returns
    (phi1, psi0, converged)."""
    target = np.array([F_des[0], F_des[2]])
    phi1, psi0 = u0
    ephi, epsi = np.radians(0.5), np.radians(0.5)
    converged = False
    for _ in range(iters):
        F = _force_xz(phi1, psi0, gamma0, psi1, v_body, n_phase)
        r = F - target
        if np.max(np.abs(r)) < 1e-6:
            converged = True
            break
        Jp = (_force_xz(phi1 + ephi, psi0, gamma0, psi1, v_body, n_phase) - F) / ephi
        Js = (_force_xz(phi1, psi0 + epsi, gamma0, psi1, v_body, n_phase) - F) / epsi
        Jm = np.column_stack([Jp, Js])
        try:
            step = np.linalg.solve(Jm, -r)
        except np.linalg.LinAlgError:
            break
        # damp + clip into the admissible box
        step = np.clip(step, -np.radians(15.0), np.radians(15.0))
        phi1 = float(np.clip(phi1 + step[0], *PHI1_LIM))
        psi0 = float(np.clip(psi0 + step[1], *PSI0_LIM))
    return phi1, psi0, converged


def power_optimal_trim(v_body, F_des=WEIGHT_UP, n_phase=N_PHASE,
                       n_gamma=11, n_psi1=11, u_hint=None):
    """Power-optimal control u = (phi1, psi0, gamma0, psi1) for force F_des at v_body.

    Resolves the 2-DOF control redundancy by minimizing cycle-averaged power over
    the schedule DOF (gamma0, psi1) while (phi1, psi0) are trimmed to hit F_des at
    each grid node. Returns (u, power, feasible, info)."""
    gammas = np.linspace(*GAMMA0_LIM, n_gamma)
    psi1s = np.linspace(*PSI1_LIM, n_psi1)
    seed = (np.radians(20.0), np.radians(-20.0)) if u_hint is None else (u_hint[0], u_hint[1])
    best = None
    for g in gammas:
        # warm-start phi1,psi0 across psi1 within a gamma column
        u0 = seed
        for ps1 in psi1s:
            phi1, psi0, ok = trim_fast(g, ps1, v_body, F_des, u0, n_phase)
            if not ok:
                continue
            u0 = (phi1, psi0)
            P = cycle_averaged_power(make_p(phi1, psi0, g, ps1, v_body), n_phase)
            if best is None or P < best[1]:
                best = ((phi1, psi0, g, ps1), P)
    if best is None:
        return None, np.inf, False, "no feasible trim"
    return best[0], best[1], True, "ok"


def stroke_frame(v_body, gamma0):
    """Advance ratio J and inflow angle chi (from the stroke-plane normal).

    Velocity scale is the reference wing speed s0* omega*. The stroke-plane normal
    is n = cos(gamma0) z - sin(gamma0) x (report convention); chi is measured from
    n toward the in-plane leading-edge (+x) direction."""
    s0 = STUDY_SPAN_FRAC * REF_LW * (REF_S0 / (STUDY_SPAN_FRAC * REF_LW))  # = REF_S0
    scale = REF_S0 * REF_OMEGA_STAR
    v = np.asarray(v_body, float)
    speed = np.hypot(v[0], v[2])
    J = speed / scale
    if speed < 1e-12:
        return 0.0, 0.0
    n = np.array([-np.sin(gamma0), np.cos(gamma0)])      # (x,z) of stroke normal
    t = np.array([np.cos(gamma0), np.sin(gamma0)])       # in-plane leading-edge dir
    vxz = np.array([v[0], v[2]])
    chi = np.arctan2(np.dot(vxz, t), np.dot(vxz, n))
    return J, chi


def trim_line(js, direction, n_gamma=7, n_psi1=9):
    """Power-optimal trims along a velocity ray, warm-started by continuation.

    `js` are signed advance ratios; `direction` is the unit body-velocity vector
    in (x, z). Returns dict of arrays: phi1, psi0, gamma0, psi1, P (aero power),
    Pmuscle (= P + Vz*W, adding the climb work)."""
    scale = REF_S0 * REF_OMEGA_STAR
    out = {k: [] for k in ("phi1", "psi0", "gamma0", "psi1", "P", "Pmuscle")}
    u_hint = None
    for J in js:
        v = (J * scale * direction[0], 0.0, J * scale * direction[1])
        u, P, ok, _ = power_optimal_trim(v, n_gamma=n_gamma, n_psi1=n_psi1, u_hint=u_hint)
        if not ok:
            for k in out:
                out[k].append(np.nan)
            continue
        u_hint = u
        out["phi1"].append(u[0]); out["psi0"].append(u[1])
        out["gamma0"].append(u[2]); out["psi1"].append(u[3])
        out["P"].append(P); out["Pmuscle"].append(P + v[2])  # +Vz*W (W=1)
    return {k: np.array(v) for k, v in out.items()}


def figure_trim_manifold():
    import matplotlib.pyplot as plt
    from post.style import apply_matplotlib_style, resolve_style
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Jf = np.linspace(0.0, 0.9, 16)          # forward flight (chi ~ 90)
    Jv = np.linspace(-0.4, 1.0, 22)         # vertical: <0 descend, >0 climb (chi ~ 0/180)
    fwd = trim_line(Jf, (1.0, 0.0))
    vert = trim_line(Jv, (0.0, 1.0))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)

    axA.plot(Jf, fwd["P"], color="black", lw=1.8, label="forward")
    axA.plot(Jv, vert["P"], color="black", lw=1.8, ls="--", label="vertical (aero)")
    axA.plot(Jv, vert["Pmuscle"], color="0.55", lw=1.5, ls=":",
             label="vertical (+ climb work)")
    axA.axvline(0.0, color="0.85", lw=0.8, zorder=0)
    axA.set_xlabel(r"advance ratio $J$ (climb $>0$)")
    axA.set_ylabel(r"aerodynamic power $\bar{P}^\ast$")
    axA.set_ylim(0, None)
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="upper left")
    axA.set_title("(a)", fontsize=style.font_size)

    axB.plot(Jf, np.degrees(fwd["psi0"]), color="black", lw=1.8, label=r"$\psi_0$")
    axB.plot(Jf, np.degrees(fwd["psi1"]), color="black", lw=1.8, ls="--", label=r"$\psi_1$")
    axB.plot(Jf, np.degrees(fwd["phi1"]), color="black", lw=1.8, ls=":", label=r"$\phi_1$")
    axB.plot(Jf, np.degrees(fwd["gamma0"]), color="0.6", lw=1.5, ls="-.", label=r"$\gamma_0$")
    axB.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    axB.set_xlabel(r"advance ratio $J$ (forward)")
    axB.set_ylabel("control (deg)")
    axB.legend(fontsize=style.font_size - 3, frameon=True, loc="center left", ncol=2)
    axB.set_title("(b)", fontsize=style.font_size)

    out = OUT_DIR / "maneuver_trim.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"forward: psi0 {np.degrees(fwd['psi0'][0]):.0f} -> {np.degrees(fwd['psi0'][-1]):.0f} deg, "
          f"gamma0 max {np.degrees(np.nanmax(fwd['gamma0'])):.1f} deg, "
          f"P {fwd['P'][0]:.2f} -> {fwd['P'][-1]:.2f}")
    print(f"vertical gamma0 range: [{np.degrees(np.nanmin(vert['gamma0'])):.0f}, "
          f"{np.degrees(np.nanmax(vert['gamma0'])):.0f}] deg")


# ---------------------------------------------------------------------------
# Part C: closed-loop maneuvers.
#
# Controller = velocity-scheduled hover controller. Outer loop: a first-order
# velocity tracker with feedforward, a_des = a_ref + Kv (v_ref - v); the required
# net force is F_des = a_des - g (g = (0,0,-1)). Inner loop (dynamic inversion):
# hold gamma0 = 0 (power-optimal), re-optimize psi1 for power, and trim (phi1,psi0)
# to hit F_des at the current body velocity. The control is updated once per
# wingbeat (the loop bandwidth is ~10x below omega*), held across the beat, and
# the UNSTEADY instantaneous force is integrated -- the same design/verification
# split as the hover controller.

from hover_drift import GRAVITY, instant_force  # noqa: E402

KV = 2.0                                  # velocity-loop gain (1/T0); ~omega* / 7
PSI1_ALLOC = np.radians(np.linspace(20.0, 60.0, 7))


def allocate(v_body, F_des, warm=None):
    """Power-optimal (phi1, psi0, gamma0=0, psi1) producing F_des at v_body.

    gamma0 fixed horizontal (power-optimal for forward/climb); psi1 swept for
    minimum power with (phi1, psi0) trimmed to F_des at each node; warm-started."""
    u0 = (warm[0], warm[1]) if warm is not None else (np.radians(20.0), np.radians(-10.0))
    best = None
    for ps1 in PSI1_ALLOC:
        phi1, psi0, ok = trim_fast(0.0, ps1, v_body, F_des, u0)
        if not ok:
            continue
        u0 = (phi1, psi0)
        P = cycle_averaged_power(make_p(phi1, psi0, 0.0, ps1, v_body))
        if best is None or P < best[1]:
            best = ((phi1, psi0, 0.0, ps1), P)
    if best is None:                      # saturated: keep the last trim attempt
        phi1, psi0, _ = trim_fast(0.0, np.radians(45.0), v_body, F_des, u0)
        return (phi1, psi0, 0.0, np.radians(45.0))
    return best[0]


def _rc(t, a, b):
    """Raised-cosine 0->1 ramp on [a,b] and its time derivative."""
    if t <= a:
        return 0.0, 0.0
    if t >= b:
        return 1.0, 0.0
    s = (t - a) / (b - a)
    return 0.5 * (1.0 - np.cos(np.pi * s)), 0.5 * np.pi / (b - a) * np.sin(np.pi * s)


def ref_dash(t, vc=1.75, acc=(1.0, 4.0), dec=(9.0, 12.0)):
    """Forward dash: accelerate to vc, cruise, decelerate. Returns (v_ref, a_ref)."""
    up, dup = _rc(t, *acc)
    dn, ddn = _rc(t, *dec)
    f, df = up - dn, dup - ddn
    return np.array([vc * f, 0.0, 0.0]), np.array([vc * df, 0.0, 0.0])


def ref_pullup(t, vc=1.4, turn=(2.0, 7.0)):
    """Pull-up: rotate the velocity from forward (vc x) to climb (vc z)."""
    s, ds = _rc(t, *turn)
    th, dth = 0.5 * np.pi * s, 0.5 * np.pi * ds
    v = vc * np.array([np.cos(th), 0.0, np.sin(th)])
    a = vc * dth * np.array([-np.sin(th), 0.0, np.cos(th)])
    return v, a


def simulate_maneuver(ref, v0, T, ctrl_per_beat=1):
    """Unsteady closed-loop run. Control re-allocated every `ctrl_per_beat` beats."""
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)
    ctrl_dt = ctrl_per_beat * period

    s = np.zeros(7)
    s[3:6] = v0
    u = None
    last_ctrl = -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz", "phi1", "psi0", "psi1",
                           "vx_ref", "vz_ref")}

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
        if t - last_ctrl >= ctrl_dt - 1e-9:
            v = s[3:6]
            v_ref, a_ref = ref(t)
            a_des = a_ref + KV * (v_ref - v)
            F_des = a_des - GRAVITY
            u = allocate(v, F_des, warm=u)
            last_ctrl = t
        v_ref, _ = ref(t)
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["phi1"].append(u[0]); log["psi0"].append(u[1]); log["psi1"].append(u[3])
        log["vx_ref"].append(v_ref[0]); log["vz_ref"].append(v_ref[2])
        if i < n:
            k1 = deriv(s, u)
            k2 = deriv(s + 0.5 * dt * k1, u)
            k3 = deriv(s + 0.5 * dt * k2, u)
            k4 = deriv(s + dt * k3, u)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(v) for k, v in log.items()}


def figure_maneuvers():
    import matplotlib.pyplot as plt
    from post.style import apply_matplotlib_style, resolve_style
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dash = simulate_maneuver(ref_dash, (0.0, 0.0, 0.0), 14.0)
    pull = simulate_maneuver(ref_pullup, (1.4, 0.0, 0.0), 10.0)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)

    # (a) forward dash: velocity tracking (commanded vs achieved, with ripple)
    axA.plot(dash["t"], dash["vx_ref"], color="0.6", lw=2.4, label="commanded")
    axA.plot(dash["t"], dash["vx"], color="black", lw=1.0, label="achieved")
    axA.set_xlabel(r"time ($\sqrt{L/g}$)")
    axA.set_ylabel(r"forward velocity $v_x^\ast$")
    axA.set_xlim(0, 14)
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="upper right")
    axA.set_title("(a) forward dash", fontsize=style.font_size)

    # (b) pull-up: path in the sagittal plane
    axB.plot(pull["x"], pull["z"], color="black", lw=1.4)
    axB.plot(pull["x"][0], pull["z"][0], "o", color="black", ms=4, mfc="white")
    # velocity arrows along the path
    idx = np.linspace(0, len(pull["t"]) - 1, 9).astype(int)
    for j in idx:
        axB.annotate("", xy=(pull["x"][j] + 0.18 * pull["vx"][j],
                             pull["z"][j] + 0.18 * pull["vz"][j]),
                     xytext=(pull["x"][j], pull["z"][j]),
                     arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))
    axB.set_xlabel(r"$x$ (body lengths)")
    axB.set_ylabel(r"$z$ (body lengths)")
    axB.set_title("(b) pull-up", fontsize=style.font_size)
    axB.set_aspect("equal", adjustable="datalim")

    out = OUT_DIR / "maneuver_control.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    # tracking error metrics
    dash_err = np.abs(dash["vx"] - dash["vx_ref"])
    sp_ref = np.hypot(pull["vx_ref"], pull["vz_ref"])
    sp = np.hypot(pull["vx"], pull["vz"])
    print(f"dash: peak |vx-vx_ref| = {dash_err.max():.3f}, "
          f"cruise ripple (std last 2 T0) = "
          f"{np.std(dash['vx'][(dash['t']>6)&(dash['t']<8)]):.4f}")
    print(f"pull-up: peak speed error = {np.max(np.abs(sp - sp_ref)):.3f}, "
          f"psi0 range [{np.degrees(pull['psi0'].min()):.0f}, "
          f"{np.degrees(pull['psi0'].max()):.0f}] deg, "
          f"psi1 range [{np.degrees(pull['psi1'].min()):.0f}, "
          f"{np.degrees(pull['psi1'].max()):.0f}] deg")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main():
    print("power-optimal straight-flight trim over body velocity (F = weight up)\n")
    print(f"{'case':>16} {'vx':>5} {'vz':>5} {'J':>5} {'chi':>6} | "
          f"{'phi1':>5} {'psi0':>6} {'gam0':>5} {'psi1':>5} | {'Pbar*':>7} {'P*':>6}")
    scale = REF_S0 * REF_OMEGA_STAR
    cases = [
        ("hover", 0.0, 0.0),
        ("fwd J=0.3", 0.3 * scale, 0.0),
        ("fwd J=0.6", 0.6 * scale, 0.0),
        ("climb J=0.2", 0.0, 0.2 * scale),
        ("climb J=0.4", 0.0, 0.4 * scale),
        ("descend J=0.3", 0.0, -0.3 * scale),
    ]
    u_hint = None
    for name, vx, vz in cases:
        v = (vx, 0.0, vz)
        u, P, ok, info = power_optimal_trim(v, u_hint=u_hint)
        if not ok:
            print(f"{name:>16} {vx:5.2f} {vz:5.2f}  ---  infeasible ({info})")
            continue
        phi1, psi0, g, ps1 = u
        J, chi = stroke_frame(v, g)
        Pstar = cost_of_endurance(make_p(*u, v))
        print(f"{name:>16} {vx:5.2f} {vz:5.2f} {J:5.2f} {np.degrees(chi):6.1f} | "
              f"{np.degrees(phi1):5.1f} {np.degrees(psi0):6.1f} {np.degrees(g):5.1f} "
              f"{np.degrees(ps1):5.1f} | {P:7.4f} {Pstar:6.3f}")
    print()
    figure_trim_manifold()
    print()
    figure_maneuvers()


if __name__ == "__main__":
    main()

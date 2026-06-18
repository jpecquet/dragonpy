"""
Proposed generalized (maneuvering) controller -- feasibility check and comparison
to the current gamma=0 power-optimal controller (report1, Generalized Controller).

THE PROPOSAL (Jean's, report1 sec:general-controller):
  - inner loop like hover: phi1 -> magnitude, psi0 -> direction; delta0 = 90 deg
  - psi1 slaved to advance ratio J, always giving MAXIMUM C_F* (not power-optimal)
  - gamma scheduled to align the stroke-plane normal n with the flight direction:
        gamma(J) = f(J) gamma_f + (1 - f(J)) gamma_h
    with gamma_h = 40 deg (hover value) at J = 0, and gamma_f the tilt that puts
    n along the body velocity (axial inflow, chi = 0) in the high-J limit;
    f monotone, f(0) = 0, f(inf) = 1. Sign of gamma_h flips with flight
    direction to remove the +x / -x asymmetry.

THE CURRENT CONTROLLER (maneuver_control.allocate): gamma = 0 (horizontal stroke
plane, so forward flight is in-plane wind chi ~ 90 deg), psi1 re-optimized for
MINIMUM power, (phi1, psi0) trimmed to weight-up.

This model has NO body drag, so steady straight flight at ANY velocity needs the
same net force as hover: the weight, pointing straight up. Both controllers must
therefore trim the cycle-averaged force to (0, 0, 1); they differ only in how the
two redundant DOF (gamma, psi1) are scheduled. We compare them on:
  feasibility (can it trim within the Table-1 control box?), the psi0 steering
  headroom (the current controller's known weakness), C_F*, and power P*.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import numpy as np

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, N_WINGS, avg_force_magnitude, cycle_averaged_force,
)
from maneuver_control import (
    make_p, trim_fast, stroke_frame, WEIGHT_UP, DELTA0,
    PHI1_LIM, PSI0_LIM, PSI1_LIM, KV, ref_dash, ref_pullup,
)
from hover_drift import GRAVITY, instant_force
from power_efficiency import cycle_averaged_power, cost_of_endurance

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = HERE.parent / "figures"
N_PHASE = 128
SCALE = REF_S0 * REF_OMEGA_STAR              # wing-speed velocity scale s0* omega*
GAMMA_H = np.radians(40.0)                    # hover stroke-plane angle (fixed param)
JC = 0.35                                     # blend half-scale in advance ratio


def q_star(p):
    """Flapping dynamic-pressure scale q* = (A*/m*/4)(s0* omega*)^2 at this phi1.

    s0* = (2/3) Lw phi1 is the actual sweep excursion, so C_F* = |F|/q* is the
    force per unit flapping dynamic pressure: at hover it peaks ~1.43, and values
    ABOVE that flag force the inflow supplies beyond what the sweep alone could
    (the gliding/fixed-wing assist of fast axial flight)."""
    s0 = (2.0 / 3.0) * p.Lw * p.phi1
    return (p.A_over_m / N_WINGS) * (s0 * p.omega_star) ** 2


# ---------------------------------------------------------------------------
# Proposed controller: gamma schedule + psi1 slaving + (phi1, psi0) trim.

PHI1_REF = REF_S0 / (STUDY_SPAN_FRAC * REF_LW)        # reference sweep amplitude
PSI1_GRID = np.radians(np.arange(4.0, 90.1, 2.0))     # psi1 search, Table-1 range


def slave_psi1(J, chi):
    """psi1 that maximizes the stroke-NORMAL (weight-supporting) force at psi0=0.

    This is the lift branch / hover-connected C_F* peak. Maximizing raw |F|
    instead would, in fast axial inflow, select the psi1 ~ 90 deg DRAG branch,
    whose cycle-mean force points backward along the inflow (NEGATIVE normal
    force) and cannot support weight. Evaluated gamma0-invariantly in the
    (J, chi) stroke frame (n = +z there), so it depends only on the inflow."""
    U = J * SCALE
    vb = (U * np.sin(chi), 0.0, U * np.cos(chi))
    best = (PSI1_GRID[0], -np.inf)
    for ps1 in PSI1_GRID:
        p = Params(A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR, phi1=PHI1_REF,
                   Lw=REF_LW, gamma0=0.0, v_body=vb, element_span_fracs=STUDY_ELEMENT,
                   psi0=0.0, psi1=float(ps1), delta0=DELTA0)
        Fn = cycle_averaged_force(p, N_PHASE)[2]       # +z = stroke normal at gamma0=0
        if Fn > best[1]:
            best = (float(ps1), Fn)
    return best[0]


def blend(J, Jc=JC):
    """f(J) in [0,1], monotone, f(0)=0, f(inf)=1. Gaussian ramp on |J|."""
    return 1.0 - np.exp(-(J / Jc) ** 2)


def gamma_schedule(v_body, gamma_h=GAMMA_H, Jc=JC):
    """Scheduled stroke-plane angle aligning n with the velocity as J grows.

    n(gamma) = (-sin gamma, cos gamma) in (x, z); aligning n with the unit
    velocity gives gamma_f = atan2(-vhat_x, vhat_z), used unrestricted (no
    descent clamp): forward flight -> gamma_f = -/+90 deg, climb -> 0, descent
    -> -/+180 deg (n pointing down along the fall, axial inflow). In descent the
    symmetric-pitch force then points down along n and psi0 (now allowed to
    +/-90 deg) flips it back up to oppose gravity.

    The inclined hover plane leans toward +x by default (gamma_h_signed
    = -gamma_h = -40 deg, i.e. n leaning +x); only a commanded -x heading flips
    it to +40 deg. Hover, climb, descent and +x forward flight therefore all
    start from gamma = -40 deg at J = 0. Returns (gamma, gamma_f, J)."""
    v = np.asarray(v_body, float)
    speed = np.hypot(v[0], v[2])
    J = speed / SCALE
    sx = 1.0 if v[0] >= -1e-12 else -1.0       # commanded fore/aft; +x default
    gh = -gamma_h * sx                          # +x -> -40 deg, -x -> +40 deg
    if speed < 1e-9:
        return gh, gh, 0.0
    vhat = np.array([v[0], v[2]]) / speed
    gamma_f = float(np.arctan2(-vhat[0], vhat[1]))
    f = blend(J, Jc)
    return f * gamma_f + (1.0 - f) * gh, gamma_f, J


def proposed_trim(v_body, warm=None, Jc=JC):
    """Proposed-controller equilibrium trim to weight-up at body velocity v_body.

    gamma from the schedule, psi1 slaved to max C_F* at the operating (J, chi),
    (phi1, psi0) trimmed to (0,0,1). Returns an info dict."""
    gamma, gamma_f, J = gamma_schedule(v_body, Jc=Jc)
    Js, chi = stroke_frame(v_body, gamma)             # inflow seen at this gamma
    ps1 = float(np.clip(slave_psi1(Js, chi), *PSI1_LIM))
    u0 = warm if warm is not None else (np.radians(25.0), np.radians(20.0))
    phi1, psi0, ok = trim_fast(gamma, ps1, v_body, WEIGHT_UP, u0, N_PHASE)
    p = make_p(phi1, psi0, gamma, ps1, v_body)
    F = np.array(avg_force_xz(p))
    res = float(np.hypot(F[0], F[1] - 1.0))
    feasible = ok and res < 1e-3
    return dict(gamma=gamma, gamma_f=gamma_f, J=J, chi=chi, psi1=ps1,
                phi1=phi1, psi0=psi0, feasible=feasible, residual=res,
                Cpsi=avg_force_magnitude(p, N_PHASE) / q_star(p),
                P=cycle_averaged_power(p, N_PHASE),
                Pstar=cost_of_endurance(p, N_PHASE),
                headroom=np.degrees(PSI0_LIM[1]) - abs(np.degrees(psi0)))


# ---------------------------------------------------------------------------
# Current controller: gamma = 0, psi1 power-optimal, (phi1, psi0) trim.

PSI1_POWER = np.radians(np.linspace(20.0, 90.0, 15))


def avg_force_xz(p):
    from feasibility import cycle_averaged_force
    F = cycle_averaged_force(p, N_PHASE)
    return F[0], F[2]


def current_trim(v_body, warm=None):
    """Current-controller trim: gamma=0, psi1 swept for MIN power, (phi1,psi0) trim."""
    u0 = warm if warm is not None else (np.radians(25.0), np.radians(-15.0))
    best = None
    for ps1 in PSI1_POWER:
        phi1, psi0, ok = trim_fast(0.0, ps1, v_body, WEIGHT_UP, u0, N_PHASE)
        if not ok:
            continue
        u0 = (phi1, psi0)
        P = cycle_averaged_power(make_p(phi1, psi0, 0.0, ps1, v_body), N_PHASE)
        if best is None or P < best[-1]:
            best = (phi1, psi0, ps1, P)
    if best is None:
        return dict(gamma=0.0, J=np.hypot(v_body[0], v_body[2]) / SCALE, psi1=np.nan,
                    phi1=np.nan, psi0=np.nan, feasible=False, residual=np.inf,
                    Cpsi=np.nan, P=np.inf, Pstar=np.inf, headroom=np.nan, chi=np.nan)
    phi1, psi0, ps1, P = best
    p = make_p(phi1, psi0, 0.0, ps1, v_body)
    _, chi = stroke_frame(v_body, 0.0)
    return dict(gamma=0.0, J=np.hypot(v_body[0], v_body[2]) / SCALE, chi=chi, psi1=ps1,
                phi1=phi1, psi0=psi0, feasible=True, residual=0.0,
                Cpsi=avg_force_magnitude(p, N_PHASE) / q_star(p), P=P,
                Pstar=cost_of_endurance(p, N_PHASE),
                headroom=np.degrees(PSI0_LIM[1]) - abs(np.degrees(psi0)))


# ---------------------------------------------------------------------------
# Sweeps and reporting.

def sweep(trim_fn, js, direction):
    """Run a trim function along a signed advance-ratio ray (warm-started)."""
    out = []
    warm = None
    for J in js:
        v = (J * SCALE * direction[0], 0.0, J * SCALE * direction[1])
        r = trim_fn(v, warm=warm)
        if r["feasible"]:
            warm = (r["phi1"], r["psi0"])
        out.append(r)
    return out


def _row(name, r):
    g = np.degrees(r["gamma"]); ps1 = np.degrees(r["psi1"])
    ph = np.degrees(r["phi1"]); ps0 = np.degrees(r["psi0"])
    chi = np.degrees(r.get("chi", np.nan))
    flag = " " if r["feasible"] else "X"
    print(f"  {name:>9} {flag} gam{g:6.1f} chi{chi:6.1f} | phi1{ph:5.1f} psi0{ps0:6.1f} "
          f"psi1{ps1:5.1f} | C_F*{r['Cpsi']:5.2f} hdrm{r['headroom']:5.1f} "
          f"P{r['P']:7.3f} P*{r['Pstar']:5.2f}")


def envelope(trim_fn, direction, Jmax=1.2, dJ=0.05):
    """Largest advance ratio along `direction` that stays feasible (contiguous)."""
    J = 0.0
    warm = None
    last = 0.0
    while J <= Jmax + 1e-9:
        v = (J * SCALE * direction[0], 0.0, J * SCALE * direction[1])
        r = trim_fn(v, warm=warm)
        if not r["feasible"]:
            break
        warm = (r["phi1"], r["psi0"])
        last = J
        J += dJ
    return last


def report():
    print(__doc__.split("\n\n")[0])
    print(f"\nvelocity scale s0* omega* = {SCALE:.3f};  "
          f"gamma_h = {np.degrees(GAMMA_H):.0f} deg, Jc = {JC}\n")

    cases = [
        ("hover",        (0.0, 0.0)),
        ("fwd J=0.2",    (0.2, 0.0)),
        ("fwd J=0.4",    (0.4, 0.0)),
        ("fwd J=0.6",    (0.6, 0.0)),
        ("fwd J=0.8",    (0.8, 0.0)),
        ("climb J=0.2",  (0.0, 0.2)),
        ("climb J=0.4",  (0.0, 0.4)),
        ("descend J=0.3",(0.0, -0.3)),
    ]
    for name, (jx, jz) in cases:
        v = (jx * SCALE, 0.0, jz * SCALE)
        print(f"{name}:")
        _row("PROPOSED", proposed_trim(v))
        _row("CURRENT", current_trim(v))

    print("\nFeasible advance-ratio envelope (contiguous from J=0):")
    for label, d in [("forward +x", (1.0, 0.0)), ("climb +z", (0.0, 1.0)),
                     ("descend -z", (0.0, -1.0))]:
        jp = envelope(proposed_trim, d)
        jc = envelope(current_trim, d)
        print(f"  {label:>11}:  proposed J<={jp:.2f}   current J<={jc:.2f}")


# ---------------------------------------------------------------------------
# Comparison figure: forward-flight schedules and costs, both controllers.

def figure_compare():
    import matplotlib.pyplot as plt
    from post.style import apply_matplotlib_style, resolve_style
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    c1, c2 = "black", "#b2182b"             # primary (solid), secondary (dashed)

    def arr(rows, key, deg=False):
        v = np.array([r[key] if r["feasible"] else np.nan for r in rows], float)
        return np.degrees(v) if deg else v

    def plot_row(axrow, Jf, rows, titles):
        # steering
        axrow[0].plot(Jf, arr(rows, "gamma", True), color=c1, lw=1.8, label=r"$\gamma$")
        axrow[0].plot(Jf, arr(rows, "psi0", True), color=c2, lw=1.8, ls="--", label=r"$\psi_0$")
        axrow[0].axhline(-90, color="0.7", lw=0.8, ls=":")
        axrow[0].axhline(90, color="0.7", lw=0.8, ls=":")
        axrow[0].set_ylabel("direction controls (deg)")
        axrow[0].legend(fontsize=style.font_size - 3, frameon=True, loc="best")
        # magnitude
        axrow[1].plot(Jf, arr(rows, "phi1", True), color=c1, lw=1.8, label=r"$\phi_1$")
        axrow[1].plot(Jf, arr(rows, "psi1", True), color=c2, lw=1.8, ls="--", label=r"$\psi_1$")
        axrow[1].axhline(90, color="0.7", lw=0.8, ls=":")
        axrow[1].set_ylabel("magnitude controls (deg)")
        axrow[1].legend(fontsize=style.font_size - 3, frameon=True, loc="best")
        # cost
        l1, = axrow[2].plot(Jf, arr(rows, "P"), color=c1, lw=1.8, label=r"$\langle P^* \rangle$")
        ax2b = axrow[2].twinx()
        l2, = ax2b.plot(Jf, arr(rows, "Cpsi"), color=c2, lw=1.8, ls="--", label=r"$C_{F^*}$")
        ax2b.set_ylabel(r"$C_{F^*}$", fontsize=style.font_size - 1)
        axrow[2].set_ylabel(r"aero power $\langle P^* \rangle$")
        axrow[2].legend(handles=[l1, l2], fontsize=style.font_size - 3, frameon=True, loc="best")
        for a, ti in zip(axrow, titles):
            a.set_xlim(0.0, 1.0)
            a.set_title(ti, fontsize=style.font_size)

    Jf = np.linspace(0.0, 1.0, 21)
    fwd = sweep(proposed_trim, Jf, (1.0, 0.0))
    Jc = np.linspace(0.0, 1.0, 21)
    clb = sweep(proposed_trim, Jc, (0.0, 1.0))

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.6), constrained_layout=True)
    plot_row(axes[0], Jf, fwd, ["(a) steering", "(b) magnitude", "(c) cost"])
    plot_row(axes[1], Jc, clb, ["(d) steering", "(e) magnitude", "(f) cost"])
    for a in axes[1]:
        a.set_xlabel(r"advance ratio $J$")
    for axrow, lbl in zip(axes, ("forward", "climb")):
        axrow[0].annotate(lbl, xy=(-0.42, 0.5), xycoords="axes fraction",
                          rotation=90, va="center", ha="center", weight="bold",
                          fontsize=style.font_size)

    out = OUT_DIR / "generalized_control_compare.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Closed-loop maneuvers under the proposed controller (unsteady simulation).
#
# Outer loop: first-order velocity tracker with feedforward,
# a_des = a_ref + Kv (v_ref - v), F_des = a_des - g. Inner loop (proposed
# allocation): schedule gamma and slave psi1 on the flight direction/speed,
# then trim (phi1, psi0) to F_des at the current velocity. Control updated once
# per wingbeat and held; the UNSTEADY instantaneous force is integrated.

def _flight_dir(*vecs):
    """Unit (x,z) flight direction from the first non-negligible vector."""
    for vec in vecs:
        nrm = np.hypot(vec[0], vec[2])
        if nrm > 1e-6:
            return np.array([vec[0], vec[2]]) / nrm
    return np.array([1.0, 0.0])


def proposed_allocate(v, F_des, v_ref, a_ref, warm=None):
    """Proposed controls (phi1, psi0, gamma, psi1) for force F_des at velocity v.

    gamma and psi1 are scheduled on the flight direction/speed (feedforward);
    (phi1, psi0) are trimmed to F_des at the actual velocity (feedback). Near
    rest the direction falls back to v_ref then a_ref so the schedule stays
    on the commanded heading."""
    d = _flight_dir(v, v_ref, a_ref)
    speed = max(np.hypot(v[0], v[2]), 1e-3 * SCALE)   # keep heading defined near rest
    v_sched = (speed * d[0], 0.0, speed * d[1])
    gamma, _, _ = gamma_schedule(v_sched)
    Js, chi = stroke_frame(v_sched, gamma)
    ps1 = float(np.clip(slave_psi1(Js, chi), *PSI1_LIM))
    u0 = warm if warm is not None else (np.radians(20.0), np.radians(0.0))
    phi1, psi0, _ = trim_fast(gamma, ps1, tuple(v), F_des, u0, N_PHASE)
    return (phi1, psi0, gamma, ps1)


def simulate_proposed(ref, v0, T, ctrl_per_beat=1):
    """Unsteady closed-loop run with the proposed controller."""
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)
    ctrl_dt = ctrl_per_beat * period

    s = np.zeros(7)
    s[3:6] = v0
    u = None
    last_ctrl = -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz", "gamma", "psi0",
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
            warm = (u[0], u[1]) if u is not None else None
            u = proposed_allocate(v, F_des, v_ref, a_ref, warm=warm)
            last_ctrl = t
        v_ref, _ = ref(t)
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["gamma"].append(u[2]); log["psi0"].append(u[1])
        log["vx_ref"].append(v_ref[0]); log["vz_ref"].append(v_ref[2])
        if i < n:
            k1 = deriv(s, u)
            k2 = deriv(s + 0.5 * dt * k1, u)
            k3 = deriv(s + 0.5 * dt * k2, u)
            k4 = deriv(s + dt * k3, u)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(val) for k, val in log.items()}


def figure_maneuver():
    import matplotlib.pyplot as plt
    from post.style import apply_matplotlib_style, resolve_style
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dash = simulate_proposed(ref_dash, (0.0, 0.0, 0.0), 14.0)
    pull = simulate_proposed(ref_pullup, (1.4, 0.0, 0.0), 10.0)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.6), constrained_layout=True)

    # (a) forward dash: velocity tracking
    axA.plot(dash["t"], dash["vx_ref"], color="0.6", lw=2.4, label="commanded")
    axA.plot(dash["t"], dash["vx"], color="black", lw=1.0, label="achieved")
    axA.set_xlabel(r"time ($\sqrt{L/g}$)")
    axA.set_ylabel(r"forward velocity $v_x^\ast$")
    axA.set_xlim(0, 14)
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="upper right")
    axA.set_title("(a) forward dash", fontsize=style.font_size)

    # (b) pull-up: sagittal-plane path with velocity arrows
    axB.plot(pull["x"], pull["z"], color="black", lw=1.4)
    axB.plot(pull["x"][0], pull["z"][0], "o", color="black", ms=4, mfc="white")
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

    out = OUT_DIR / "gen_maneuver.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    dash_err = np.abs(dash["vx"] - dash["vx_ref"])
    sp_ref = np.hypot(pull["vx_ref"], pull["vz_ref"])
    sp = np.hypot(pull["vx"], pull["vz"])
    print(f"dash: peak |vx-vx_ref| = {dash_err.max():.3f}; "
          f"pull-up: peak speed error = {np.max(np.abs(sp - sp_ref)):.3f}, "
          f"gamma {np.degrees(pull['gamma'].min()):.0f}..{np.degrees(pull['gamma'].max()):.0f} deg")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def ref_loop(t, R=1.5, t0=1.0, T=26.0):
    """Closed circular flight loop, started/stopped from rest.

    The body traces a circle of radius R centred at (0, R): forward, up,
    backward, down, back to the origin. Arc-phase sigma sweeps 0 -> 2pi with an
    eased (smoothstep) schedule so velocity is zero at both ends. The velocity
    direction (cos sigma, sin sigma) rotates through all four quadrants, so both
    v_x and v_z reverse sign twice. Returns (v_ref, a_ref)."""
    if t <= t0 or t >= t0 + T:
        return np.zeros(3), np.zeros(3)
    s = (t - t0) / T
    sig = 2.0 * np.pi * (3.0 * s ** 2 - 2.0 * s ** 3)
    sig_d = 2.0 * np.pi * (6.0 * s - 6.0 * s ** 2) / T
    sig_dd = 2.0 * np.pi * (6.0 - 12.0 * s) / T ** 2
    cs, sn = np.cos(sig), np.sin(sig)
    v = R * sig_d * np.array([cs, 0.0, sn])
    a = R * np.array([sig_dd * cs - sig_d ** 2 * sn, 0.0,
                      sig_dd * sn + sig_d ** 2 * cs])
    return v, a


def figure_loop():
    import matplotlib.pyplot as plt
    from post.style import apply_matplotlib_style, resolve_style
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    T_sim = 30.0
    run = simulate_proposed(ref_loop, (0.0, 0.0, 0.0), T_sim)
    t = run["t"]
    vx_ref = np.array([ref_loop(tt)[0][0] for tt in t])
    vz_ref = np.array([ref_loop(tt)[0][2] for tt in t])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
    cx, cz = "black", "#b2182b"

    # (a) velocity components vs time: commanded (light) vs achieved
    axA.plot(t, vx_ref, color=cx, lw=2.4, alpha=0.3)
    axA.plot(t, run["vx"], color=cx, lw=1.0, label=r"$v_x^\ast$")
    axA.plot(t, vz_ref, color=cz, lw=2.4, alpha=0.3)
    axA.plot(t, run["vz"], color=cz, lw=1.0, ls="--", label=r"$v_z^\ast$")
    axA.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    axA.set_xlabel(r"time ($\sqrt{L/g}$)")
    axA.set_ylabel(r"body velocity")
    axA.set_xlim(0, 30)
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="upper right")
    axA.set_title("(a) velocity tracking", fontsize=style.font_size)

    # (b) path with velocity arrows
    axB.plot(run["x"], run["z"], color="black", lw=1.4)
    axB.plot(run["x"][0], run["z"][0], "o", color="black", ms=4, mfc="white")
    idx = np.linspace(0, len(t) - 1, 13).astype(int)
    for j in idx:
        axB.annotate("", xy=(run["x"][j] + 0.4 * run["vx"][j],
                             run["z"][j] + 0.4 * run["vz"][j]),
                     xytext=(run["x"][j], run["z"][j]),
                     arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
    axB.set_xlabel(r"$x$ (body lengths)")
    axB.set_ylabel(r"$z$ (body lengths)")
    axB.set_title("(b) flight path", fontsize=style.font_size)
    axB.set_aspect("equal", adjustable="datalim")

    out = OUT_DIR / "gen_loop.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    ex = np.abs(run["vx"] - vx_ref); ez = np.abs(run["vz"] - vz_ref)
    print(f"loop: peak |vx err|={ex.max():.3f}, peak |vz err|={ez.max():.3f}; "
          f"vx in [{run['vx'].min():.2f},{run['vx'].max():.2f}], "
          f"vz in [{run['vz'].min():.2f},{run['vz'].max():.2f}]")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Prey capture: pure-pursuit guidance driving the velocity-tracking controller.
# At each control update the commanded velocity points along the line of sight
# to the moving prey, at a fixed speed exceeding the prey's, so the body closes
# in. Interception is declared at range r_cap = 0.5 L (docs/research/pursuit).

PREY_P0 = (2.5, 0.0, 2.0)
PREY_V = (0.35, 0.0, 0.0)


def prey_pos(t, p0=PREY_P0, vp=PREY_V):
    return np.array([p0[0] + vp[0] * t, 0.0, p0[2] + vp[2] * t])


def simulate_pursuit(v_cmd=0.7, T=22.0, r_cap=0.5, ctrl_per_beat=1):
    """Pure-pursuit interception of a moving prey by the proposed controller."""
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)
    ctrl_dt = ctrl_per_beat * period

    s = np.zeros(7)
    u = None
    last_ctrl = -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz", "px", "pz", "rng")}

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
        if t - last_ctrl >= ctrl_dt - 1e-9:
            v = s[3:6]
            los = np.array([r[0], 0.0, r[2]]) / rng
            v_ref = v_cmd * los                       # command: aim at prey, speed v_cmd
            F_des = KV * (v_ref - v) - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u = proposed_allocate(v, F_des, v_ref, np.zeros(3), warm=warm)
            last_ctrl = t
        log["t"].append(t)
        log["x"].append(s[0]); log["z"].append(s[2])
        log["vx"].append(s[3]); log["vz"].append(s[5])
        log["px"].append(pp[0]); log["pz"].append(pp[2]); log["rng"].append(rng)
        if rng < r_cap:
            t_cap = t
            break
        if i < n:
            k1 = deriv(s, u)
            k2 = deriv(s + 0.5 * dt * k1, u)
            k3 = deriv(s + 0.5 * dt * k2, u)
            k4 = deriv(s + dt * k3, u)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(val) for k, val in log.items()}, t_cap


def figure_pursuit():
    import matplotlib.pyplot as plt
    from post.style import apply_matplotlib_style, resolve_style
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log, t_cap = simulate_pursuit(v_cmd=0.7)
    cz = "#b2182b"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.8), constrained_layout=True)

    # (a) pursuit geometry: paths + line-of-sight rays
    idx = np.linspace(0, len(log["t"]) - 1, 8).astype(int)
    for j in idx:
        axA.plot([log["x"][j], log["px"][j]], [log["z"][j], log["pz"][j]],
                 color="0.75", lw=0.5, zorder=0)
    axA.plot(log["px"], log["pz"], color=cz, lw=1.6, ls="--", label="prey")
    axA.plot(log["x"], log["z"], color="black", lw=1.6, label="dragonfly")
    axA.plot(log["x"][0], log["z"][0], "o", color="black", ms=4, mfc="white")
    axA.plot(log["px"][0], log["pz"][0], "s", color=cz, ms=4, mfc="white")
    axA.plot(log["x"][-1], log["z"][-1], "*", color=cz, ms=11, zorder=5)
    axA.set_xlabel(r"$x$ (body lengths)")
    axA.set_ylabel(r"$z$ (body lengths)")
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="lower right")
    axA.set_title("(a) pursuit", fontsize=style.font_size)
    axA.set_aspect("equal", adjustable="datalim")

    # (b) range to prey vs time
    axB.plot(log["t"], log["rng"], color="black", lw=1.6)
    axB.axhline(0.5, color=cz, lw=0.9, ls=":")
    axB.set_xlabel(r"time ($\sqrt{L/g}$)")
    axB.set_ylabel(r"range to prey (body lengths)")
    axB.set_ylim(0, None)
    axB.set_title("(b) closure", fontsize=style.font_size)

    out = OUT_DIR / "gen_pursuit.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"pursuit: capture at t={t_cap:.1f} (range 0.5), "
          f"intercept ({log['x'][-1]:.2f}, {log['z'][-1]:.2f})")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main():
    report()
    figure_compare()
    figure_maneuver()
    figure_loop()
    figure_pursuit()


if __name__ == "__main__":
    main()

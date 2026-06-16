"""
How close to hover does the *static* equilibrium actually get under the real
(unsteady) dynamics?

We pick the "perfect" hover parameters from the cycle-averaged equilibrium of
section 2: at a fixed stroke-plane angle gamma0, with C_psi held near its maximum
(psi1 ~ 51 deg, delta0 = 90 deg), solve for

    psi0  such that the cycle-averaged force is vertical  (F_x = 0), and
    phi1  such that its magnitude equals the weight       (|F| = 1).

That balances gravity *on cycle average*, with the body held at rest (v_body = 0).
The real problem is unsteady: within each wingbeat the body accelerates, so its
velocity -- and hence the air-relative velocity at the wing -- is no longer zero,
and the instantaneous force no longer integrates to exactly (0,0,1). We integrate
the point-mass equations of motion (translation only, attitude frozen) with these
fixed open-loop parameters and watch the positional drift.

Force model: single blade element at 2/3 span (the study convention), reused from
`feasibility` via `_wing_force_vec`; the integrator mirrors
`dynamics._deriv_point_mass` (dpos=vel, dvel=F+g, dphase=omega) so it is the same
physics the full simulator would run, just specialized to the 2/3-span element.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, STUDY_ELEMENT,
    R_HINGE_RIGHT, R_HINGE_LEFT, _CHIRALITY,
    _wing_force_vec, cycle_averaged_force, replace,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
N_PHASE = 256          # phase samples for the equilibrium cycle-average
GRAVITY = np.array([0.0, 0.0, -1.0])

# Wing inertia. Each of the 4 wings carries mass MASS_RATIO (wing mass / body
# mass); body mass = 1 is the normalization unit, so the total mass the aero must
# carry is M = 1 + 4*MASS_RATIO. COM_FRAC places the wing centre of mass along the
# span (1/2 = uniform rectangular wing); MASS_RATIO = 0 recovers the massless model.
#
# Range from Wakeling (1997) m_w/m_b over 37 Anisoptera specimens (data/morphology/
# wakeling1997, forewing+hindwing pooled): min 0.010, max 0.026, mean ~0.014. The
# lone 0.003 forewing (LD1) is dropped -- the README flags that forewing table as
# poorly digitized. Total-mass factor M = m/m_b spans 1.03-1.10 in the same data.
MASS_RATIO = 0.014
MASS_RATIO_MIN = 0.010
MASS_RATIO_MAX = 0.026
COM_FRAC = 0.5


def total_mass(mass_ratio):
    return 1.0 + 4.0 * mass_ratio

# Operating point. gamma0 is the stroke-plane angle; C_psi is held near its max
# by psi1 ~ 51 deg, delta0 = 90 deg (psi0-independent optimum). Morphology and
# frequency are fixed at mid-range; phi1 and psi0 are solved for hover.
GAMMA0 = np.radians(40.0)
PSI1 = np.radians(51.0)
DELTA0 = np.pi / 2.0
A_OVER_M, OMEGA_STAR, LW = REF_A_OVER_M, REF_OMEGA_STAR, REF_LW
SIGMA0 = np.pi          # fore/hind phase shift (report default)

# Per-wing hinge frames (same order as feasibility: fore_R, fore_L, hind_R,
# hind_L). The fore/hind phase offsets are built per call from p.sigma0.
HINGES = (R_HINGE_RIGHT, R_HINGE_LEFT, R_HINGE_RIGHT, R_HINGE_LEFT)


def base_params(**over):
    return replace(Params(gamma0=GAMMA0, psi1=PSI1, delta0=DELTA0,
                          A_over_m=A_OVER_M, omega_star=OMEGA_STAR, Lw=LW,
                          sigma0=SIGMA0, element_span_fracs=STUDY_ELEMENT), **over)


# ---------------------------------------------------------------------------
# Equilibrium solve: cycle-averaged force = (0, 0, 1).

def _bisect(f, lo, hi, target, increasing, iters=60):
    """Bisection root of f(x) = target; `increasing` flags the sign of df/dx."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = (f(mid) < target)
        if below == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_equilibrium(p, mass_ratio=MASS_RATIO):
    """Return (psi0, phi1) giving cycle-averaged force (0, 0, M).

    With wing mass the aero must carry the total weight M = 1 + 4*mass_ratio, not
    just the body's 1; the direction solve (F_x = 0) is unaffected by mass.
    """
    weight = total_mass(mass_ratio)
    for _ in range(8):
        # Magnitude: |F| increases with phi1.
        phi1 = _bisect(
            lambda v: float(np.linalg.norm(cycle_averaged_force(replace(p, phi1=v), N_PHASE))),
            0.05, 3.0, target=weight, increasing=True)
        p = replace(p, phi1=phi1)
        # Direction: F_x decreases as psi0 grows. With the report convention
        # (+gamma0 tilts the normal toward -x) the hover root sits at psi0 < 0,
        # so bracket [-70, 10] deg (mirror of the old [-10, 70]).
        psi0 = _bisect(
            lambda v: float(cycle_averaged_force(replace(p, psi0=v), N_PHASE)[0]),
            np.radians(-70.0), np.radians(10.0), target=0.0, increasing=False)
        p = replace(p, psi0=psi0)
    return p.psi0, p.phi1


# ---------------------------------------------------------------------------
# Unsteady point-mass integration with the fixed equilibrium parameters.

def instant_force(p, phase, omega, vel):
    """Total body-frame aero force at one instant (single 2/3-span element)."""
    pv = replace(p, v_body=(float(vel[0]), float(vel[1]), float(vel[2])))
    offsets = (0.0, 0.0, p.sigma0, p.sigma0)   # fore/hind phase from p.sigma0
    F = np.zeros(3)
    for ch, off, H in zip(_CHIRALITY, offsets, HINGES):
        F += _wing_force_vec(pv, ch, H, np.array([phase - off]), omega)[0]
    return F


# ---------------------------------------------------------------------------
# Wing-inertia kinematics: the mass-weighted wing centre-of-mass offset in the
# body frame, and its phase derivative. With attitude frozen the body recoils to
# hold the system COM on its aero+gravity trajectory, so the body picks up the
# kinematic offset -S/M (position) and -omega*dS/M (velocity), S = sum_i m_w s_i.

def wing_com_sum(p, phase, mass_ratio):
    """Mass-weighted wing-COM offset S and dS/dphase, summed over the 4 wings.

    s_i = r_cm * xhat_i, with xhat_i the spanwise axis in the body frame
        xhat = H @ Rx(tilt) @ Rz(sweep) @ [1,0,0],   r_cm = COM_FRAC * Lw.
    Feather rotates about the span axis itself, so it leaves s_i unchanged. The
    sweep/tilt sign conventions mirror feasibility._wing_force_vec. Returns
    (S, dS_dphase), each (3,), already weighted by mass_ratio.
    """
    r_cm = COM_FRAC * p.Lw
    offsets = (0.0, 0.0, p.sigma0, p.sigma0)   # fore/hind phase from p.sigma0
    S = np.zeros(3)
    dS = np.zeros(3)
    for ch, off, H in zip(_CHIRALITY, offsets, HINGES):
        tilt = ch * p.gamma0   # +gamma0 -> normal toward -x (mirrors feasibility._wing_force_vec)
        sweep = ch * p.phi1 * np.sin(phase - off)
        dsweep = ch * p.phi1 * np.cos(phase - off)        # d(sweep)/d(phase)
        cs, ss = np.cos(sweep), np.sin(sweep)
        ct, st = np.cos(tilt), np.sin(tilt)
        xhat = np.array([cs, ct * ss, st * ss])
        dxhat = dsweep * np.array([-ss, ct * cs, st * cs])
        S += mass_ratio * r_cm * (H @ xhat)
        dS += mass_ratio * r_cm * (H @ dxhat)
    return S, dS


def simulate(p, n_cycles=30, steps_per_cycle=200, mass_ratio=MASS_RATIO):
    """Unsteady hover with wing inertia (translation only, attitude frozen).

    The system COM follows  M Rcom'' = F_aero + M g.  The body translation that
    feeds the aero is recovered by subtracting the wing-COM offset,
        r_body = Rcom - S/M,    v_body = Vcom - omega*dS/M,
    so the returned trajectory is the BODY's. Integrated state is
    [Rcom(3), Vcom(3), phase]; initial conditions put the body at rest at the
    origin, so mass_ratio = 0 reproduces the massless run exactly.
    """
    omega = 2.0 * np.pi * p.wing_frequency       # = omega_star
    period = 2.0 * np.pi / omega
    dt = period / steps_per_cycle
    n_steps = int(n_cycles * steps_per_cycle)
    M = total_mass(mass_ratio)

    def body_from_com(state):
        S, dS = wing_com_sum(p, state[6], mass_ratio)
        return state[0:3] - S / M, state[3:6] - omega * dS / M

    def deriv(state):
        _, v_body = body_from_com(state)
        d = np.empty(7)
        d[0:3] = state[3:6]                                  # dRcom = Vcom
        d[3:6] = instant_force(p, state[6], omega, v_body) / M + GRAVITY
        d[6] = omega
        return d

    # Body at rest at origin: r_body(0)=0, v_body(0)=0 => Rcom=S/M, Vcom=omega*dS/M.
    S0, dS0 = wing_com_sum(p, 0.0, mass_ratio)
    s = np.zeros(7)
    s[0:3] = S0 / M
    s[3:6] = omega * dS0 / M
    t = np.empty(n_steps + 1)
    pos = np.empty((n_steps + 1, 3))
    vel = np.empty((n_steps + 1, 3))
    r0, v0 = body_from_com(s)
    t[0], pos[0], vel[0] = 0.0, r0, v0
    for i in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        r_body, v_body = body_from_com(s)
        t[i + 1], pos[i + 1], vel[i + 1] = (i + 1) * dt, r_body, v_body
    return t, pos, vel, period


def last_cycle_p2p(t, arr, period):
    """Peak-to-peak of each column of `arr` over the final full wingbeat."""
    seg = arr[t >= (t[-1] - period)]
    return seg.max(axis=0) - seg.min(axis=0)


def inertia_bob_p2p(p, mass_ratio, n=400):
    """Pure wing-inertia body bob: peak-to-peak of the kinematic offset -S/M.

    This is the body motion wing inertia forces regardless of the aero dynamics
    (it is independent of omega*), so it isolates effect (2) from the aero-
    fluctuation bob, effect (1)."""
    M = total_mass(mass_ratio)
    phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    S = np.array([wing_com_sum(p, ph, mass_ratio)[0] for ph in phases]) / M
    return S.max(axis=0) - S.min(axis=0)


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p0 = base_params()
    psi0, phi1 = solve_equilibrium(p0)
    p = replace(p0, psi0=psi0, phi1=phi1)
    Fbar = cycle_averaged_force(p, N_PHASE)

    t, pos, _vel, period = simulate(p, n_cycles=30, steps_per_cycle=200)
    cycles = t / period

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                   constrained_layout=True)
    axA.axhline(0.0, color="0.8", lw=0.8)
    axA.plot(cycles, pos[:, 0], color="black", lw=1.4, label=r"$x$ (fore/aft)")
    axA.plot(cycles, pos[:, 2], color="black", lw=1.4, ls="--", label=r"$z$ (vertical)")
    axA.set_xlabel("wingbeat cycles")
    axA.set_ylabel("position (body lengths)")
    axA.legend(fontsize=style.font_size - 1, frameon=True, loc="best")

    axB.plot(pos[:, 0], pos[:, 2], color="black", lw=1.0)
    axB.plot(pos[0, 0], pos[0, 2], "o", color="black", ms=4)
    axB.set_xlabel(r"$x$ (body lengths)")
    axB.set_ylabel(r"$z$ (body lengths)")
    axB.set_title("path", fontsize=style.font_size)
    axB.set_aspect("equal", adjustable="datalim")

    out = OUT_DIR / "hover_drift.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    # --- wing-inertia decomposition: effect (1) aero fluctuation vs (2) inertia ---
    M = total_mass(MASS_RATIO)
    full_p2p = last_cycle_p2p(t, pos, period)            # both effects, coupled
    inertia_p2p = inertia_bob_p2p(p, MASS_RATIO)         # effect (2) alone (-S/M)
    # Effect (1) alone: re-solve hover for massless wings and integrate at m_w = 0.
    psi0_0, phi1_0 = solve_equilibrium(p0, mass_ratio=0.0)
    p_massless = replace(p0, psi0=psi0_0, phi1=phi1_0)
    t0, pos0, _v0, period0 = simulate(p_massless, n_cycles=30, steps_per_cycle=200,
                                      mass_ratio=0.0)
    aero_p2p = last_cycle_p2p(t0, pos0, period0)

    # --- report numbers ---
    print(f"equilibrium @ gamma0={np.degrees(GAMMA0):.0f} deg, m_w={MASS_RATIO} "
          f"(M={M:.2f}): psi0={np.degrees(psi0):.2f} deg, "
          f"phi1={np.degrees(phi1):.2f} deg (s0={2/3*LW*phi1:.3f})")
    print(f"cycle-avg force Fbar = [{Fbar[0]:+.2e}, {Fbar[1]:+.2e}, {Fbar[2]:+.4f}] "
          f"(target [0,0,{M:.2f}])")
    drift = pos[-1] - pos[0]
    print(f"net drift over {cycles[-1]:.0f} cycles: "
          f"dx={drift[0]:+.4f}, dy={drift[1]:+.4f}, dz={drift[2]:+.4f} body lengths")
    print(f"  mean drift speed: {np.linalg.norm(drift)/t[-1]:.2e} L/T0")
    print("within-cycle body bob, peak-to-peak (body lengths):")
    print(f"  vertical  z: aero-only={aero_p2p[2]:.2e}  inertia-only={inertia_p2p[2]:.2e}  "
          f"full={full_p2p[2]:.2e}")
    print(f"  fore/aft  x: aero-only={aero_p2p[0]:.2e}  inertia-only={inertia_p2p[0]:.2e}  "
          f"full={full_p2p[0]:.2e}")
    print(f"  inertia/aero ratio: z={inertia_p2p[2]/aero_p2p[2]:.2f}  "
          f"x={inertia_p2p[0]/aero_p2p[0]:.2f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

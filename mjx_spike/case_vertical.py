"""
Passive-stability test case (user-specified):

    m_w = 0.02, Aw/m = 0.15, omega* = 14, Lw = 0.75,
    stroke plane HORIZONTAL in the world  (=> gamma0 = 90 deg body-relative),
    body initially VERTICAL (pitched up 90 deg: longitudinal +x -> world up),
    forewing hinges at x = +0.1, hindwing hinges at x = -0.1 (on the long axis),
    standard symmetric hover trim (psi0=0, psi1=51, delta0=90, sigma0=180),
    phi1 solved so the cycle-averaged force carries the total weight M = 1.08.

Hypothesis: passively (attitude) stable. We integrate the free 6-DOF dynamics
from the nominal vertical attitude and from a +15 deg pitch perturbation, and
watch whether the body returns to vertical (stable) or tumbles (unstable).
"""

import sys
from pathlib import Path

import numpy as np
import jax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "summer_paper" / "1_hover_feasibility"))
sys.path.insert(0, str(HERE.parents[0]))

import articulated as A           # noqa: E402
import roll_articulated as R      # noqa: E402
from feasibility import Params, replace, cycle_averaged_force  # noqa: E402

# Two-winged insect emulated by 4 in-phase wings at one hinge station, each with
# the (halved) area below. Long wings (Lw=6) -> large flapping counter-torque.
MW = 0.064
M_TOTAL = 1.0 + 4.0 * MW          # body + 4 wings = 1.256
GAMMA0 = np.radians(90.0)         # horizontal stroke plane with a vertical body
LW = 6.0
OMEGA = 10.8
AW = 0.0345                       # per wing
PSI0 = 0.0
PHI1 = 0.2675                     # fixed; s0 = (2/3) Lw phi1 = 1.07
SIGMA0 = 0.0                      # in-phase (the 4 wings act as one pair)
ROOT_FORE = 0.1                   # same hinge station for all four wings
ROOT_HIND = 0.1
# Slender-body inertia: thin rod, length = 1 body length, mass 1, along x.
# (Tiny next to the long wings' inertia, which MuJoCo carries via the wing bodies.)
THORAX_INERTIA = (1e-3, 1.0 / 12.0, 1.0 / 12.0)
DT = 0.0002
N_CYCLES = 30


def solve_phi1(cfg, M):
    """phi1 so the cycle-averaged |force| = M, via the trusted numpy oracle."""
    base = Params(Lw=cfg.Lw, Aw_over_mb=cfg.Aw, omega_star=cfg.omega,
                  gamma0=cfg.gamma0, psi0=PSI0, psi1=cfg.psi1, delta0=cfg.delta0,
                  sigma0=cfg.sigma0, element_span_fracs=(2.0 / 3.0,))
    lo, hi = 0.05, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = np.linalg.norm(cycle_averaged_force(replace(base, phi1=mid), 256))
        lo, hi = (mid, hi) if f < M else (lo, mid)
    return 0.5 * (lo + hi)


def pitch_quat(deg):
    """Body->world quat for a pitch-up rotation of `deg` about +y (right axis).
    deg=90 sends body +x (forward) to world up (-z)."""
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, np.sin(a), 0.0])


def body_x_tilt_deg(quat):
    """Angle of the body longitudinal axis from world-up, per timestep."""
    w, x, y, z = quat.T
    bx_z = 2.0 * (x * z - w * y)        # world-z component of body +x axis
    return np.degrees(np.arccos(np.clip(-bx_z, -1.0, 1.0)))   # up = -z


def run(cfg, phi1, init_deg, label):
    wings = A.default_wings(root_fore=ROOT_FORE, root_hind=ROOT_HIND)
    m, mx, info = R.build(cfg, wings, massless=False, dt=DT, dof='free',
                          mass_ratio=MW, thorax_inertia=THORAX_INERTIA)
    # ensure wing mass = MW
    thid = m.body("thorax").id
    n_steps = int(N_CYCLES * (2 * np.pi / cfg.omega) / DT)
    roll = R.make_rollout(mx, info, cfg, n_steps, thid, 'free',
                          phi1=phi1, psi0=PSI0, init_quat=pitch_quat(init_deg))
    pos, quat = jax.jit(roll)()
    pos, quat = np.asarray(pos), np.asarray(quat)
    tilt = body_x_tilt_deg(quat)
    spc = int((2 * np.pi / cfg.omega) / DT)
    alt = -pos[:, 2]                      # world up = -z
    horiz = np.hypot(pos[:, 0], pos[:, 1])

    # Per-cycle PEAK tilt: the oscillation envelope. Growing -> unstable,
    # roughly constant -> neutral, shrinking -> asymptotically stable.
    ncyc = len(tilt) // spc
    peak = np.array([tilt[i * spc:(i + 1) * spc].max() for i in range(ncyc)])
    first, last = peak[:ncyc // 3].mean(), peak[-ncyc // 3:].mean()
    if last > 1.3 * first + 1:
        verdict = "envelope GROWS -> unstable"
    elif last < 0.7 * first:
        verdict = "envelope DECAYS -> asymptotically stable"
    else:
        verdict = "envelope ~bounded -> marginal / neutral"

    print(f"  [{label}]  start tilt {init_deg-90:+.0f} deg from vertical, {ncyc} cycles")
    env = "  ".join(f"c{i*3}:{peak[i*3]:.0f}" for i in range(ncyc // 3))
    print(f"    per-cycle peak tilt (deg):  {env}")
    print(f"    peak tilt: max {tilt.max():.0f} deg; first-third avg {first:.0f} -> "
          f"last-third avg {last:.0f}   => {verdict}")
    print(f"    altitude change: {alt[-1]-alt[0]:+.2f} L   horizontal drift: {horiz[-1]:.2f} L")


def solve_psi1_hover(M):
    """Feather amplitude psi1 (deg<51) so cycle-avg |F| = M, with phi1 fixed."""
    def Fmag(psi1):
        p = Params(Lw=LW, Aw_over_mb=AW, omega_star=OMEGA, gamma0=GAMMA0,
                   phi1=PHI1, psi0=PSI0, psi1=psi1, delta0=np.pi / 2.0,
                   sigma0=SIGMA0, element_span_fracs=(2.0 / 3.0,))
        return np.linalg.norm(cycle_averaged_force(p, 256))
    lo, hi = np.radians(0.5), np.radians(51.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if Fmag(mid) < M else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    psi1 = solve_psi1_hover(M_TOTAL)
    cfg = A.Cfg(gamma0=GAMMA0, psi1=psi1, delta0=np.pi / 2.0,
                omega=OMEGA, Aw=AW, Lw=LW, sigma0=SIGMA0)
    Fbar = cycle_averaged_force(
        Params(Lw=cfg.Lw, Aw_over_mb=cfg.Aw, omega_star=cfg.omega, gamma0=cfg.gamma0,
               phi1=PHI1, psi0=PSI0, psi1=cfg.psi1, delta0=cfg.delta0,
               sigma0=cfg.sigma0, element_span_fracs=(2.0 / 3.0,)), 256)
    Fmag = np.linalg.norm(Fbar)
    print("Vertical-body passive-stability test (two-wing emulation: 4 in-phase "
          f"wings at x={ROOT_FORE}):")
    print(f"  Lw={LW}, omega={OMEGA}, Aw/m={AW}/wing, m_w={MW}, s0={2/3*LW*PHI1:.3f}, "
          f"M={M_TOTAL}")
    print(f"  phi1={PHI1} rad ({np.degrees(PHI1):.1f} deg) fixed; "
          f"hover-trim psi1={np.degrees(psi1):.2f} deg -> |Fbar|={Fmag:.3f} "
          f"(F/W={Fmag/M_TOTAL:.2f})  Fbar(body)={np.round(Fbar,4)}\n")
    run(cfg, PHI1, init_deg=90.0, label="nominal vertical")
    print()
    run(cfg, PHI1, init_deg=105.0, label="perturbed +15deg")


if __name__ == "__main__":
    main()

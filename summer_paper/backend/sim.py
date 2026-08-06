"""Flapping-flight point-mass model and controller of report1 (streamlined core).

Planar (x, z) point mass, weight 1, driven by N lumped wings (single blade
element at 2/3 span, quasi-steady C_L/C_D). Controls u = (phi1, psi0, gamma,
psi1) with delta0 = 90 deg fixed; s0* = (2/3) Lw phi1. Provides the mean-force
model (numerical arc + analytic normal-inflow forms), the gamma/psi1 schedules,
the two trims (analytic split, damped Newton), and the once-per-beat closed-loop
simulator. numpy only.
"""

from dataclasses import dataclass

import numpy as np

CL0, CD0, CD90 = 1.5, 0.1, 2.0
DCD, CDBAR = CD90 - CD0, 0.5 * (CD90 + CD0)

GRAVITY = np.array([0.0, 0.0, -1.0])
DELTA0 = np.pi / 2.0
SPAN_FRAC = 2.0 / 3.0
N_WINGS = 4                      # 2 pairs (fore/hind), each lumping left+right
N_PHASE = 128

PHI1_LIM = np.radians((2.0, 60.0))
PSI0_LIM = np.radians((-90.0, 90.0))
PSI1_LIM = np.radians((5.0, 90.0))


@dataclass(frozen=True)
class Morphology:
    Lw: float = 0.75
    A_over_m: float = 0.6
    omega_star: float = 14.0


REF = Morphology()
REF_S0 = 0.25
SCALE = REF_S0 * REF.omega_star  # fixed reference wingbeat velocity (s0* omega*)^ref
GAMMA_H = np.radians(45.0)       # |gamma_hover|; sign set by the latched heading
JC = 0.35                        # gamma-blend half-scale in advance ratio
J_PIN = 0.05                     # settled-near-rest threshold for the hover pin

_HINGE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Wing force: right fore wing on the moving arc; left wings mirror (x, z equal,
# y cancels), hind pair leads by pi.

def _rot_x(a):
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1),
                     np.stack([z, s, c], -1)], -2)


def _rot_z(a):
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _wing_force(u, theta, v_body, m):
    """Body-frame force of one wing over phases theta (T,) -> (T, 3)."""
    phi1, psi0, gamma, psi1 = u
    omega = m.omega_star
    sweep = phi1 * np.sin(theta)
    sweep_d = phi1 * omega * np.cos(theta)
    feather = psi0 + np.pi / 2.0 + psi1 * np.sin(theta - DELTA0)
    feather_d = psi1 * omega * np.cos(theta - DELTA0)

    R_tilt = _rot_x(np.full_like(theta, gamma))
    R_bw = _HINGE @ (R_tilt @ (_rot_z(sweep) @ _rot_x(feather)))
    omega_zyx = np.stack([np.cos(sweep) * feather_d, np.sin(sweep) * feather_d,
                          sweep_d], -1)
    omega_body = np.einsum("ij,tj->ti", _HINGE @ R_tilt[0], omega_zyx)
    r = SPAN_FRAC * m.Lw * R_bw[:, :, 0]
    v_elem = np.cross(omega_body, r) + np.asarray(v_body, float)
    v_w = np.einsum("tk,tkj->tj", v_elem, R_bw)

    vy, vz = v_w[:, 1], v_w[:, 2]
    V2 = vy * vy + vz * vz
    alpha = np.arctan2(-vz, vy)
    Cl = CL0 * np.sin(2.0 * alpha)
    Cd = CD0 * np.cos(alpha) ** 2 + CD90 * np.sin(alpha) ** 2
    V = np.where(V2 > 1e-24, np.sqrt(V2), 1.0)
    dy, dz = -vy / V, -vz / V
    s = 0.5 * (m.A_over_m / N_WINGS) * V2
    Fy = s * (Cl * dz + Cd * dy)
    Fz = s * (-Cl * dy + Cd * dz)
    F_w = np.stack([np.zeros_like(Fy), Fy, Fz], -1)
    return np.einsum("tk,tjk->tj", F_w, R_bw)


def instant_force(u, phase, v_body, m=REF):
    """Total instantaneous aero force (fore at phase, hind at phase - pi)."""
    F = 2.0 * _wing_force(u, np.array([phase, phase - np.pi]), v_body, m).sum(0)
    F[1] = 0.0
    return F


def cycle_force(u, v_body, m=REF, n=N_PHASE):
    """Cycle-averaged total aero force at fixed body velocity."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    F = 4.0 * _wing_force(u, theta, v_body, m).mean(0)
    F[1] = 0.0
    return F


# ---------------------------------------------------------------------------
# Analytic normal-inflow mean force (report phase tau; psi = psi0 + psi1
# cos(tau - delta0), inflow J along the stroke normal).

_TAU = (np.arange(2048) + 0.5) * (2.0 * np.pi / 2048)   # midpoint: no 0/0 at J=0
_ST = np.sin(_TAU)
_PSI1_GRID = np.radians(np.arange(5.0, 90.01, 0.25))


def averages(J, psi1):
    """I_0, I_n, I_s of eq:cfn/cfs-general-op; psi1 may be an array."""
    psi1 = np.asarray(psi1, float)
    st, s2 = _ST, _ST * _ST
    den = np.sqrt(J * J + s2)
    c = np.cos(2.0 * psi1[..., None] * st)
    s = np.sin(2.0 * psi1[..., None] * st)
    I0 = float(den.mean())
    In = (J * (2.0 * CL0 * s2 + 0.5 * DCD * (J * J - s2)) / den * c).mean(-1) \
        - ((CL0 * (J * J - s2) - DCD * J * J) * st / den * s).mean(-1)
    Is = (J * (CL0 * (J * J - s2) + DCD * s2) / den * c).mean(-1) \
        + ((2.0 * CL0 * J * J - 0.5 * DCD * (J * J - s2)) * st / den * s).mean(-1)
    return I0, In, Is


def cf_op(psi0, psi1, J):
    """(C_Fn, C_Fs) at the operating point delta0 = 90 deg; psi1 array-capable."""
    I0, In, Is = averages(J, psi1)
    return (2.0 * In * np.cos(2.0 * psi0) - 2.0 * J * CDBAR * I0,
            -2.0 * Is * np.sin(2.0 * psi0))


def cf_quad(psi0, psi1, delta0, J=0.0):
    """(C_Fn, C_Fs) by direct quadrature, any delta0 (eq:cfn/cfs-general)."""
    psi = psi0 + psi1 * np.cos(_TAU - delta0)
    alpha = psi - np.arctan2(_ST, J)
    Cl = CL0 * np.sin(2.0 * alpha)
    Cd = CD0 * np.cos(alpha) ** 2 + CD90 * np.sin(alpha) ** 2
    v = np.sqrt(J * J + _ST * _ST)
    return (-2.0 * np.mean(v * (_ST * Cl + J * Cd)),
            -2.0 * np.mean(v * (J * Cl - _ST * Cd)))


def psi1_opt(J):
    """psi1 maximizing C_Fn at psi0 = 0 (eq:psi1-slave)."""
    _, In, _ = averages(J, _PSI1_GRID)
    return float(np.clip(_PSI1_GRID[int(np.argmax(In))], *PSI1_LIM))


PSI1_HOVER = psi1_opt(0.0)
GAMMA_HOVER = -GAMMA_H           # sigma_x = +1 lean


# ---------------------------------------------------------------------------
# Trims.

def analytic_trim(gamma, psi1, J, F_des, m=REF):
    """(phi1, psi0) from the split normal-inflow trim (eq:trim-split)."""
    n_hat = np.array([-np.sin(gamma), np.cos(gamma)])
    s_hat = np.array([np.cos(gamma), np.sin(gamma)])
    fxz = np.array([F_des[0], F_des[2]])
    Fn, Fs = float(fxz @ n_hat), float(fxz @ s_hat)
    Fmag = float(np.hypot(Fn, Fs))
    if Fmag < 1e-9:
        return PHI1_LIM[0], 0.0

    I0, In, Is = averages(J, np.asarray(psi1))
    In, Is = float(In), float(Is)
    # Direction: a sin(2 psi0) + b cos(2 psi0) = c; keep the root with positive
    # projection on the demand and the smaller |psi0|.
    a, b, c = Is * Fn, In * Fs, J * CDBAR * I0 * Fs
    R = np.hypot(a, b)
    if R < 1e-12:
        cands = [0.0]
    else:
        asn = np.arcsin(np.clip(c / R, -1.0, 1.0))
        ph = np.arctan2(b, a)
        cands = [asn - ph, np.pi - asn - ph]
    best = None
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
    s0 = np.sqrt(4.0 * (Fmag / CF) / m.A_over_m) / m.omega_star
    phi1 = float(np.clip(s0 / (SPAN_FRAC * m.Lw), *PHI1_LIM))
    return phi1, psi0


def newton_trim(gamma, psi1, v_body, F_des, u0, m=REF, n=N_PHASE, iters=40,
                tol=1e-6):
    """(phi1, psi0, ok) matching the simulated mean force to F_des (x, z)."""
    target = np.array([F_des[0], F_des[2]])
    phi1, psi0 = u0
    eps = np.radians(0.5)

    def fxz(p1, p0):
        F = cycle_force((p1, p0, gamma, psi1), v_body, m, n)
        return np.array([F[0], F[2]])

    ok = False
    for _ in range(iters):
        F = fxz(phi1, psi0)
        r = F - target
        if np.max(np.abs(r)) < tol:
            ok = True
            break
        Jm = np.column_stack([(fxz(phi1 + eps, psi0) - F) / eps,
                              (fxz(phi1, psi0 + eps) - F) / eps])
        try:
            step = np.linalg.solve(Jm, -r)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -np.radians(15.0), np.radians(15.0))
        phi1 = float(np.clip(phi1 + step[0], *PHI1_LIM))
        psi0 = float(np.clip(psi0 + step[1], *PSI0_LIM))
    return phi1, psi0, ok


# ---------------------------------------------------------------------------
# Schedules and allocation (sec:control-law).

def gamma_schedule(v, sx=1.0):
    """(gamma, J): blend from sigma_x gamma_hover to the velocity-aligned tilt."""
    speed = np.hypot(v[0], v[2])
    J = speed / SCALE
    gh = -GAMMA_H * sx
    if speed < 1e-9:
        return gh, 0.0
    gamma_u = float(np.arctan2(-v[0] / speed, v[2] / speed))
    f = 1.0 - np.exp(-(J / JC) ** 2)
    return f * gamma_u + (1.0 - f) * gh, J


def allocate(v, F_des, done=False, sx=1.0, warm=None, trim="analytic", m=REF):
    """Controls (phi1, psi0, gamma, psi1) meeting F_des at body velocity v."""
    gamma, J = gamma_schedule(v, sx)
    if done and J < J_PIN:
        gamma, ps1 = -GAMMA_H * sx, PSI1_HOVER
    else:
        ps1 = psi1_opt(J)
    if trim == "analytic":
        phi1, psi0 = analytic_trim(gamma, ps1, J, F_des, m)
    else:
        u0 = warm if warm is not None else (np.radians(20.0), 0.0)
        phi1, psi0, _ = newton_trim(gamma, ps1, tuple(v), F_des, u0, m)
    return (phi1, psi0, gamma, ps1)


# ---------------------------------------------------------------------------
# Closed-loop simulator: control once per wingbeat, RK4 on the unsteady force.
# State s = [x, y, z, vx, vy, vz, phase].

def _deriv(s, u, m):
    F = instant_force(u, s[6], s[3:6], m)
    d = np.empty(7)
    d[0:3] = s[3:6]
    d[3:6] = F + GRAVITY
    d[6] = m.omega_star
    return d


def rk4_step(s, u, dt, m):
    k1 = _deriv(s, u, m)
    k2 = _deriv(s + 0.5 * dt * k1, u, m)
    k3 = _deriv(s + 0.5 * dt * k2, u, m)
    k4 = _deriv(s + dt * k3, u, m)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(u_ref, T, trim="analytic", stop=None, v0=(0.0, 0.0, 0.0), sx=1.0,
             K=None, m=REF):
    """Closed-loop run of the velocity law F_des = K (u_ref - u) + z_hat.

    u_ref(t, s) -> (reference velocity (3,), done); stop(t, s) -> bool ends the
    run after logging. Returns arrays t, x, z, vx, vz, phi1, psi0, gamma, psi1.
    """
    omega = m.omega_star
    K = omega / (2.0 * np.pi) if K is None else K   # K T* = 1
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)

    s = np.zeros(7)
    s[3:6] = v0
    u, last = None, -1e9
    log = {k: [] for k in ("t", "x", "z", "vx", "vz",
                           "phi1", "psi0", "gamma", "psi1")}
    for i in range(n + 1):
        t = i * dt
        v_ref, done = u_ref(t, s)
        if t - last >= period - 1e-9:
            F_des = K * (np.asarray(v_ref, float) - s[3:6]) - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u = allocate(s[3:6], F_des, done=done, sx=sx, warm=warm,
                         trim=trim, m=m)
            last = t
        for k, val in zip(("t", "x", "z", "vx", "vz"),
                          (t, s[0], s[2], s[3], s[5])):
            log[k].append(val)
        for k, val in zip(("phi1", "psi0", "gamma", "psi1"), u):
            log[k].append(val)
        if stop is not None and stop(t, s):
            break
        if i < n:
            s = rk4_step(s, u, dt, m)
    return {k: np.array(v) for k, v in log.items()}

"""
Real-time dragonfly flight simulation for the interactive GUI.

This is the live, mutable-morphology counterpart of the (frozen-reference) closed
-loop runs in ``summer_paper/scripts/generalized_control.py``. It reuses the
validated point-mass force model from ``feasibility.py`` (single 2/3-span element,
report1 conventions) and re-expresses the report's *generalized controller*
parametrized by a mutable :class:`Config` so the user can change morphology, gains
and follow speed on the fly:

  * inner allocation (unchanged from the report): stroke-plane angle ``gamma`` is
    scheduled on the flight direction/advance ratio, the pitch amplitude ``psi1``
    is slaved to the force-maximizing (lift-branch) value, and the two hover
    handles ``(phi1, psi0)`` are trimmed by a 2x2 damped-Newton step to the
    commanded cycle-averaged force;
  * outer loop: the report's pure velocity tracker, a_des = Kd (v_ref - v) --
    a single velocity gain, no position feedback, no acceleration feedforward.
    The drawn path supplies v_ref; at the path end v_ref = 0, so the body damps
    to a hover near (not pinned to) the final point. The exposed gain ``Kd`` is
    this loop's.

The schedule's advance-ratio scale and the ``psi1`` slave are evaluated against
the fixed reference wing speed (``REF_S0 * REF_OMEGA_STAR``), exactly as in the
report scripts; the inner trim and the integrated dynamics use the live ``Config``
morphology. See module ``generalized_control.py`` for the canonical (reference)
versions of every function below.

numpy only; runs on the project ``dragonpy`` env.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
REPO_ROOT = HERE.parents[2]
for _p in (str(SCRIPTS), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feasibility import (  # noqa: E402
    Params, cycle_averaged_force, replace, _wing_force_vec, _CHIRALITY,
    R_HINGE_RIGHT, R_HINGE_LEFT, STUDY_ELEMENT, STUDY_SPAN_FRAC,
    REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0,
)

# ---------------------------------------------------------------------------
# Constants mirrored from the report scripts (maneuver_control / generalized).

GRAVITY = np.array([0.0, 0.0, -1.0])
DELTA0 = np.pi / 2.0
SIGMA0 = np.pi
HINGES = (R_HINGE_RIGHT, R_HINGE_LEFT, R_HINGE_RIGHT, R_HINGE_LEFT)

# Control bounds (Table 1 ranges).
PHI1_LIM = np.radians((2.0, 60.0))
PSI0_LIM = np.radians((-90.0, 90.0))
PSI1_LIM = np.radians((5.0, 90.0))

# Fixed reference wing-speed velocity scale s0* omega* used by the gamma schedule
# and the psi1 slave (as in generalized_control.py). PHI1_REF is the reference
# sweep amplitude that realizes REF_S0 at REF_LW.
SCALE = REF_S0 * REF_OMEGA_STAR
PHI1_REF = REF_S0 / (STUDY_SPAN_FRAC * REF_LW)
GAMMA_H = np.radians(45.0)        # inclined-hover stroke-plane angle magnitude
                                  # (|gamma_hover|, report reference config)
JC = 0.35                         # gamma-blend half-scale in advance ratio

N_PHASE_TRIM = 48                 # phase samples for the per-beat trim/slave
N_PSI1_GRID = 19                  # psi1 search nodes for the slave


# ---------------------------------------------------------------------------
# Live configuration (mutated by the GUI; read under the sim lock).

@dataclass
class Config:
    # morphology (Table 1)
    Lw: float = REF_LW                 # wing length ratio   [0.65, 0.85]
    A_over_m: float = REF_A_OVER_M     # inverse wing loading [0.2, 1.0]
    omega_star: float = REF_OMEGA_STAR # wingbeat frequency   [8, 20]

    # outer-loop (velocity tracker) gain
    Kd: float = 2.0                    # velocity gain (report KV; ~omega*/7)

    # follow speed: body lengths per sqrt(L/g) at which a drawn path is traced
    v_follow: float = 1.0

    # velocity taper: arc-length distance (body lengths) over which the follow
    # speed ramps linearly up from rest at the path start and back down at the end
    taper: float = 0.3

    # playback: simulated time units per wall-clock second
    time_scale: float = 1.0


# ---------------------------------------------------------------------------
# Morphology-parametrized physics + controller (faithful to generalized_control).

def make_p(cfg: Config, phi1, psi0, gamma0, psi1, v_body):
    return Params(
        Lw=cfg.Lw, A_over_m=cfg.A_over_m, omega_star=cfg.omega_star,
        gamma0=float(gamma0), phi1=float(phi1), psi0=float(psi0),
        psi1=float(psi1), delta0=DELTA0, sigma0=SIGMA0,
        v_body=(float(v_body[0]), float(v_body[1]), float(v_body[2])),
        element_span_fracs=STUDY_ELEMENT)


def _force_xz(cfg, phi1, psi0, gamma0, psi1, v_body, n_phase):
    F = cycle_averaged_force(make_p(cfg, phi1, psi0, gamma0, psi1, v_body), n_phase)
    return np.array([F[0], F[2]])


def trim_fast(cfg, gamma0, psi1, v_body, F_des, u0, n_phase=N_PHASE_TRIM, iters=30):
    """Solve (phi1, psi0) so the cycle-averaged force matches F_des (x, z).

    2x2 damped Newton with finite-difference Jacobian (maneuver_control.trim_fast),
    warm-started from u0 = (phi1, psi0). Returns (phi1, psi0, converged)."""
    target = np.array([F_des[0], F_des[2]])
    phi1, psi0 = u0
    ephi = epsi = np.radians(0.5)
    converged = False
    for _ in range(iters):
        F = _force_xz(cfg, phi1, psi0, gamma0, psi1, v_body, n_phase)
        r = F - target
        if np.max(np.abs(r)) < 1e-5:
            converged = True
            break
        Jp = (_force_xz(cfg, phi1 + ephi, psi0, gamma0, psi1, v_body, n_phase) - F) / ephi
        Js = (_force_xz(cfg, phi1, psi0 + epsi, gamma0, psi1, v_body, n_phase) - F) / epsi
        Jm = np.column_stack([Jp, Js])
        try:
            step = np.linalg.solve(Jm, -r)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -np.radians(15.0), np.radians(15.0))
        phi1 = float(np.clip(phi1 + step[0], *PHI1_LIM))
        psi0 = float(np.clip(psi0 + step[1], *PSI0_LIM))
    return phi1, psi0, converged


_PSI1_GRID = np.radians(np.linspace(5.0, 90.0, N_PSI1_GRID))


def slave_psi1(J):
    """psi1 maximizing the stroke-normal (weight-supporting) force at psi0=0.

    Lift branch / hover-connected C_F* peak at an ASSUMED AXIAL inflow (chi = 0),
    a pure function of J, reference morphology -- exactly
    generalized_control.slave_psi1 but on a coarser grid for real-time use."""
    U = J * SCALE
    vb = (0.0, 0.0, U)
    best_ps1, best_Fn = _PSI1_GRID[0], -np.inf
    cfg_ref = Config(Lw=REF_LW, A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR)
    for ps1 in _PSI1_GRID:
        Fn = cycle_averaged_force(
            make_p(cfg_ref, PHI1_REF, 0.0, 0.0, float(ps1), vb), N_PHASE_TRIM)[2]
        if Fn > best_Fn:
            best_ps1, best_Fn = float(ps1), Fn
    return best_ps1


# Fixed hover pitch amplitude: the force-maximizing psi1 at zero advance ratio
# (the hover-connected C_F* peak). Held in hover/hold mode so psi1 does not jitter
# with the residual hover velocity.
PSI1_HOVER = float(np.clip(slave_psi1(0.0), *PSI1_LIM))

# Pin gamma/psi1 to the hover values only once genuinely settled near the hold
# point: position error (body lengths) below this radius. With velocity-only
# feedback the body damps to rest wherever it is rather than flying back, so a
# body that settles farther out simply keeps scheduling -- which at J ~ 0
# approaches the hover values anyway. Position error, not the
# wingbeat-oscillating velocity, is the gate, because it is a stable
# (non-jittering) pin/unpin signal; and as the body settles the scheduled
# gamma/psi1 already approach the hover values, so the pin is continuous.
HOLD_PIN_RADIUS = 0.3


def blend(J, Jc=JC):
    return 1.0 - np.exp(-(J / Jc) ** 2)


def gamma_schedule(v_body, sign_x, gamma_h=GAMMA_H, Jc=JC):
    """Scheduled stroke-plane angle aligning n with the velocity as J grows.

    generalized_control.gamma_schedule, but the inclined-hover *sign* (which way
    the plane leans, +/- gamma_h) is taken from `sign_x` -- the COMMANDED heading
    -- rather than from the (jittering) actual velocity, so it does not flip while
    hovering. The alignment angle gamma_f and the blend J still follow the actual
    velocity `v_body`, which is what makes following track the real inflow."""
    v = np.asarray(v_body, float)
    speed = np.hypot(v[0], v[2])
    J = speed / SCALE
    gh = -gamma_h * sign_x
    if speed < 1e-9:
        return gh
    vhat = np.array([v[0], v[2]]) / speed
    gamma_f = float(np.arctan2(-vhat[0], vhat[1]))
    f = blend(J, Jc)
    return f * gamma_f + (1.0 - f) * gh


def allocate(cfg, v, F_des, sign_x, hold=False, warm=None):
    """Proposed controls (phi1, psi0, gamma, psi1) for force F_des at velocity v.

    gamma and psi1 are scheduled on the ACTUAL velocity (feedforward inflow),
    except the inclined-hover sign, which comes from the COMMANDED heading
    `sign_x` so the stroke plane does not flip on hover velocity jitter. (phi1,
    psi0) are trimmed to F_des at the actual velocity (feedback).

    In hover/hold mode (`hold=True`) gamma and psi1 are held at their fixed hover
    values -- the inclined hover stroke plane and the hover-optimal pitch
    amplitude -- instead of being scheduled on the residual hover velocity, so
    they stay put while holding; only (phi1, psi0) keep trimming to hold position.
    Returns (phi1, psi0, gamma, psi1, ok)."""
    if hold:
        gamma = -GAMMA_H * sign_x
        ps1 = PSI1_HOVER
    else:
        gamma = gamma_schedule(v, sign_x)
        Js = float(np.hypot(v[0], v[2])) / SCALE
        ps1 = float(np.clip(slave_psi1(Js), *PSI1_LIM))
    u0 = warm if warm is not None else (np.radians(20.0), 0.0)
    phi1, psi0, ok = trim_fast(cfg, gamma, ps1, tuple(v), F_des, u0)
    return phi1, psi0, gamma, ps1, ok


def instant_force(cfg, u, phase, vel):
    """Total body-frame aero force at one instant (hover_drift.instant_force)."""
    omega = cfg.omega_star
    p = make_p(cfg, u[0], u[1], u[2], u[3], (float(vel[0]), float(vel[1]), float(vel[2])))
    offsets = (0.0, 0.0, p.sigma0, p.sigma0)
    F = np.zeros(3)
    for ch, off, H in zip(_CHIRALITY, offsets, HINGES):
        F += _wing_force_vec(p, ch, H, np.array([phase - off]), omega)[0]
    return F


# ---------------------------------------------------------------------------
# Path smoothing and the time-parametrized reference.

def smooth_path(points, anchor, ds=0.04, sigma=3.0):
    """Resample a freehand polyline to uniform spacing and Gaussian-smooth it.

    `points` is a list of world (x, z) vertices; `anchor` (the body position at
    release) is forced to be the exact start. Returns an (M, 2) array, or None
    if the path is too short to follow."""
    pts = [anchor] + [p for p in points]
    P = np.asarray(pts, float)
    # drop consecutive duplicates
    keep = np.ones(len(P), bool)
    keep[1:] = np.any(np.abs(np.diff(P, axis=0)) > 1e-6, axis=1)
    P = P[keep]
    if len(P) < 2:
        return None
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 0.05:
        return None
    m = max(8, int(total / ds))
    su = np.linspace(0.0, total, m)
    x = np.interp(su, s, P[:, 0])
    z = np.interp(su, s, P[:, 1])
    end_raw = np.array([x[-1], z[-1]])    # pre-smoothing endpoints, pinned below
    # Gaussian smoothing with edge padding (replicates endpoints).
    rad = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.convolve(np.pad(x, rad, mode="edge"), k, "valid")
    z = np.convolve(np.pad(z, rad, mode="edge"), k, "valid")
    out = np.column_stack([x, z])
    out[0] = anchor                       # exact start (last commanded position)
    out[-1] = end_raw                     # exact end (where the user released)
    return out


class Reference:
    """Time-parametrized tracking target along a smoothed path at a fixed speed.

    sample(t) returns (p_ref(2), v_ref(2), a_ref(2), done). The arc-length speed
    follows a trapezoid with LINEAR ramps (constant acceleration): it ramps up
    from rest over the taper distance `taper`, cruises at `speed`, then ramps
    linearly back to rest over `taper` at the end. There is no velocity step at
    the path ends (a step would demand an unbounded transient force and saturate
    the trim). v_ref is the tangent times this speed; a_ref is the reference
    acceleration (tangential + centripetal), used only to sense the commanded
    heading at a path start -- it is NOT fed forward to the force demand. After
    the end it holds (v=a=0)."""

    def __init__(self, path, speed, taper, t0):
        self.t0 = t0
        v = max(float(speed), 1e-6)
        d = max(float(taper), 1e-3)    # taper (accel/decel) distance; guard zero
        if path is None or len(path) < 2:
            self.hold = True
            self.p_end = np.asarray(path[0] if path is not None else (0.0, 0.0), float)
            self.T3 = 0.0
            return
        self.hold = False
        self.P = np.asarray(path, float)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.S = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = float(self.S[-1])
        T = np.gradient(self.P, self.S, axis=0)
        Tn = np.linalg.norm(T, axis=1, keepdims=True)
        self.T = T / np.where(Tn > 1e-9, Tn, 1.0)
        self.dTds = np.gradient(self.T, self.S, axis=0)
        self.p_end = self.P[-1].copy()
        # linear-ramp trapezoidal speed schedule: constant accel a reaches the
        # cruise speed v over the taper distance d (a = v^2 / 2d).
        self.a = v * v / (2.0 * d)
        if self.total >= 2.0 * d:                  # room for a cruise segment
            self.vp = v
            self.d_acc = d
            self.Lc = self.total - 2.0 * d
        else:                                      # too short: triangular, lower peak
            self.vp = float(np.sqrt(self.a * self.total))
            self.d_acc = self.total / 2.0
            self.Lc = 0.0
        self.t_a = self.vp / self.a                # ramp duration
        self.t_c = self.Lc / self.vp if self.vp > 0 else 0.0
        self.T3 = 2.0 * self.t_a + self.t_c        # total follow duration

    def duration(self):
        return self.T3

    def _speed(self, tau):
        """Speed magnitude vmag(tau) and its time-rate dvmag/dt, plus arc-length s."""
        a, vp = self.a, self.vp
        if tau <= self.t_a:                        # linear ramp up (accel +a)
            vmag, dvdt = a * tau, a
            s = 0.5 * a * tau * tau
        elif tau <= self.t_a + self.t_c:           # cruise
            vmag, dvdt = vp, 0.0
            s = self.d_acc + vp * (tau - self.t_a)
        else:                                      # linear ramp down (accel -a)
            u = tau - self.t_a - self.t_c
            vmag, dvdt = max(vp - a * u, 0.0), -a
            s = self.d_acc + self.Lc + vp * u - 0.5 * a * u * u
        return vmag, dvdt, s

    def sample(self, t):
        if self.hold:
            return self.p_end, np.zeros(2), np.zeros(2), True
        tau = t - self.t0
        if tau >= self.T3:
            return self.p_end, np.zeros(2), np.zeros(2), True
        vmag, dvdt, s = self._speed(tau)
        s = min(s, self.total)
        p = np.array([np.interp(s, self.S, self.P[:, 0]),
                      np.interp(s, self.S, self.P[:, 1])])
        T = np.array([np.interp(s, self.S, self.T[:, 0]),
                      np.interp(s, self.S, self.T[:, 1])])
        dTds = np.array([np.interp(s, self.S, self.dTds[:, 0]),
                         np.interp(s, self.S, self.dTds[:, 1])])
        v = vmag * T
        a = dvdt * T + vmag ** 2 * dTds   # tangential + centripetal reference accel
        return p, v, a, False


# ---------------------------------------------------------------------------
# Real-time simulator: integrates the unsteady dynamics, control once per beat.

class Simulator:
    """Continuously-running point-mass flight sim with the proposed controller.

    State [x, y, z, vx, vy, vz, phase]; planar (y == 0). Control (phi1, psi0,
    gamma, psi1) is re-allocated once per wingbeat and held; the unsteady
    instantaneous force is RK4-integrated. Thread-safe: the background loop holds
    `lock` while stepping and snapshotting; the HTTP layer holds it to read the
    snapshot and to push commands."""

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.lock = threading.Lock()
        self.reset()

    # -- commands (called under external lock via the server) ----------------

    def reset(self):
        self.s = np.zeros(7)
        self.t = 0.0
        self.u = None
        self.last_ctrl = -1e9
        self.commanded = np.zeros(2)            # last commanded (x, z)
        self.sched_sign = 1.0                   # held commanded heading sign (+x)
        self.ref = Reference(None, self.cfg.v_follow, self.cfg.taper, 0.0)
        self._allocate()

    def set_path(self, points):
        """Install a new smoothed path starting from the ACTUAL body position.

        Anchoring at the body's instantaneous position at mouse release -- not
        at the last commanded point -- matters under the velocity-only tracker:
        the body settles with a positional offset from the previous endpoint,
        and a path anchored at the old commanded point would begin with a
        spurious catch-up transient (called under the sim lock, so reading the
        live state is safe)."""
        path = smooth_path(points, (float(self.s[0]), float(self.s[2])))
        if path is None:
            return None
        self.ref = Reference(path, self.cfg.v_follow, self.cfg.taper, self.t)
        return path

    # -- internals -----------------------------------------------------------

    def _allocate(self):
        v = self.s[3:6]
        p = self.s[0:3]
        p_ref, v_ref, a_ref, done = self.ref.sample(self.t)
        self.commanded = p_ref.copy()
        # Stroke-plane lean sign from the COMMANDED heading (commanded velocity,
        # then commanded accel at a path start), held through hover at the last
        # commanded sign. Using the command -- not the actual, jittering velocity
        # -- keeps the stroke plane from flipping sign while hovering.
        for vec in (v_ref, a_ref):
            if abs(vec[0]) > 1e-6:
                self.sched_sign = 1.0 if vec[0] >= 0.0 else -1.0
                break
        # outer loop: the report's pure velocity tracker -- a single gain, no
        # position feedback, no acceleration feedforward. e_p is not fed back;
        # it only gates the hover pin below.
        e_p = np.array([p_ref[0] - p[0], 0.0, p_ref[1] - p[2]])
        e_v = np.array([v_ref[0] - v[0], 0.0, v_ref[1] - v[2]])
        a_des = self.cfg.Kd * e_v
        F_des = a_des - GRAVITY
        # Pin gamma/psi1 to hover values only when holding AND settled near the
        # hold point; otherwise keep scheduling so an off-track body can recover.
        hold_pin = done and float(np.hypot(e_p[0], e_p[2])) < HOLD_PIN_RADIUS
        warm = (self.u[0], self.u[1]) if self.u is not None else None
        phi1, psi0, gamma, psi1, ok = allocate(
            self.cfg, v, F_des, self.sched_sign, hold=hold_pin, warm=warm)
        self.u = (phi1, psi0, gamma, psi1)
        self.ctrl_ok = ok

    def _deriv(self, state):
        v = state[3:6]
        F = instant_force(self.cfg, self.u, state[6], v)
        d = np.empty(7)
        d[0:3] = v
        d[3:6] = F + GRAVITY
        d[6] = self.cfg.omega_star
        return d

    def step(self, dt):
        """Advance one fixed integration step (control updated once per beat)."""
        period = 2.0 * np.pi / self.cfg.omega_star
        if self.t - self.last_ctrl >= period - 1e-12:
            self._allocate()
            self.last_ctrl = self.t
        k1 = self._deriv(self.s)
        k2 = self._deriv(self.s + 0.5 * dt * k1)
        k3 = self._deriv(self.s + 0.5 * dt * k2)
        k4 = self._deriv(self.s + dt * k3)
        self.s = self.s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.s[1] = 0.0
        self.s[4] = 0.0                  # keep strictly planar
        self.t += dt

    def snapshot(self):
        """Current render state (called under lock)."""
        following = (not self.ref.hold) and (self.t - self.ref.t0) < self.ref.duration()
        speed = float(np.hypot(self.s[3], self.s[5]))
        return {
            "t": self.t,
            "x": float(self.s[0]), "z": float(self.s[2]),
            "vx": float(self.s[3]), "vz": float(self.s[5]),
            "J": speed / SCALE,
            "phase": float(self.s[6] % (2.0 * np.pi)),
            "phi1": float(self.u[0]), "psi0": float(self.u[1]),
            "gamma": float(self.u[2]), "psi1": float(self.u[3]),
            "delta0": DELTA0,
            "cx": float(self.commanded[0]), "cz": float(self.commanded[1]),
            "following": bool(following),
            "ok": bool(getattr(self, "ctrl_ok", True)),
        }

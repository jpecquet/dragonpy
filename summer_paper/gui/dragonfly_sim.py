"""Real-time dragonfly flight simulation for the interactive GUI.

Thin, mutable-morphology layer over the report1 backend
(``summer_paper/backend``): all physics, schedules and trims come from
``backend.sim``, path smoothing and the speed schedule from
``backend.trajectory``. This module keeps only what is GUI-specific:

  * a mutable :class:`Config` (live morphology, outer-loop gain, follow speed);
  * the per-beat allocation with the GUI tweaks -- the inclined-hover lean sign
    follows the COMMANDED heading (so it does not flip on hover jitter), and
    gamma/psi1 pin to their hover values once settled near a hold point;
  * the real-time RK4 :class:`Simulator` (drawn-path following, pure pursuit).

The gamma-schedule and psi1-slave run on the fixed reference advance-ratio
scale, exactly as in the report; the trim and the integrated dynamics use the
live morphology. numpy only.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from summer_paper.backend import sim as core  # noqa: E402
from summer_paper.backend.trajectory import R_CAP, Reference, smooth_path  # noqa: E402

GRAVITY = core.GRAVITY
DELTA0 = core.DELTA0

N_PHASE_TRIM = 48        # phase samples for the per-beat real-time trim
TRIM_TOL = 1e-5
TRIM_ITERS = 30

# Pin gamma/psi1 to the hover values only once genuinely settled near the hold
# point (position error is a stable pin/unpin signal; the scheduled values
# approach the hover ones as the body settles, so the pin is continuous).
HOLD_PIN_RADIUS = 0.3


@dataclass
class Config:
    # morphology (Table 1)
    Lw: float = core.REF.Lw                  # wing length ratio   [0.65, 0.85]
    A_over_m: float = core.REF.A_over_m      # inverse wing loading [0.2, 1.0]
    omega_star: float = core.REF.omega_star  # wingbeat frequency   [8, 20]

    # outer-loop velocity gain (report control law: K T* = 1 -> ~2.23)
    Kd: float = 2.0

    # follow speed: body lengths per sqrt(L/g) at which a drawn path is traced
    v_follow: float = 1.0

    # accel/decel distance (body lengths) of the trapezoidal speed schedule
    taper: float = 0.3

    # playback: simulated time units per wall-clock second
    time_scale: float = 1.0

    def morphology(self):
        return core.Morphology(self.Lw, self.A_over_m, self.omega_star)


def allocate(cfg, v, F_des, sign_x, hold=False, warm=None):
    """GUI controls (phi1, psi0, gamma, psi1, ok) for force F_des at velocity v.

    gamma/psi1 scheduled on the actual velocity with the lean sign from the
    commanded heading `sign_x`; held at the hover values when `hold`. (phi1,
    psi0) Newton-trimmed to F_des at the live morphology."""
    if hold:
        gamma, ps1 = -core.GAMMA_H * sign_x, core.PSI1_HOVER
    else:
        gamma, J = core.gamma_schedule(v, sx=sign_x)
        ps1 = core.psi1_opt(J)
    u0 = warm if warm is not None else (np.radians(20.0), 0.0)
    phi1, psi0, ok = core.newton_trim(
        gamma, ps1, tuple(v), F_des, u0, m=cfg.morphology(),
        n=N_PHASE_TRIM, iters=TRIM_ITERS, tol=TRIM_TOL)
    return phi1, psi0, gamma, ps1, ok


class Simulator:
    """Continuously-running point-mass flight sim under the report controller.

    State [x, y, z, vx, vy, vz, phase]; planar (y == 0). Control re-allocated
    once per wingbeat and held; the unsteady instantaneous force is
    RK4-integrated. Thread-safe: the background loop holds `lock` while
    stepping; the HTTP layer holds it to read snapshots and push commands."""

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
        self.sched_sign = 1.0                   # latched commanded heading sign
        self.ref = Reference(None, self.cfg.v_follow, self.cfg.taper, 0.0)
        self.target = None                      # active pursuit target (or None)
        self.captured_t = -1e9                  # time of the last capture
        self._allocate()

    def set_path(self, points):
        """Install a new smoothed path anchored at the ACTUAL body position."""
        path = smooth_path(points, (float(self.s[0]), float(self.s[2])))
        if path is None:
            return None
        self.target = None                      # a drawn path replaces a pursuit
        self.ref = Reference(path, self.cfg.v_follow, self.cfg.taper, self.t)
        return path

    def set_target(self, x, z, vx, vz):
        """Install a moving point target: pure pursuit at the follow speed."""
        self.target = dict(p0=np.array([float(x), float(z)]),
                           v=np.array([float(vx), float(vz)]), t0=self.t)

    def clear_target(self):
        """Cancel an active pursuit and hold at the current body position."""
        if self.target is not None:
            self.target = None
            self._hold_here()

    # -- internals -----------------------------------------------------------

    def _target_pos(self, t):
        tg = self.target
        return tg["p0"] + tg["v"] * (t - tg["t0"])

    def _hold_here(self):
        pos = (float(self.s[0]), float(self.s[2]))
        self.ref = Reference([pos], self.cfg.v_follow, self.cfg.taper, self.t)
        self.commanded = np.array(pos)

    def _allocate(self):
        v = self.s[3:6]
        p = self.s[0:3]
        if self.target is not None:
            # pure pursuit: follow speed along the line of sight (bearing only)
            pt = self._target_pos(self.t)
            r = np.array([pt[0] - p[0], pt[1] - p[2]])
            rng = float(np.hypot(r[0], r[1]))
            los = r / rng if rng > 1e-9 else np.array([1.0, 0.0])
            p_ref, v_ref, a_ref, done = pt, self.cfg.v_follow * los, np.zeros(2), False
        else:
            p_ref, v_ref, a_ref, done = self.ref.sample(self.t)
        self.commanded = np.asarray(p_ref, float).copy()
        # Lean sign from the COMMANDED heading (velocity, then accel at a path
        # start), latched through hover so the stroke plane does not flip.
        for vec in (v_ref, a_ref):
            if abs(vec[0]) > 1e-6:
                self.sched_sign = 1.0 if vec[0] >= 0.0 else -1.0
                break
        # outer loop: the report's velocity law; e_p only gates the hover pin
        e_p = np.array([p_ref[0] - p[0], p_ref[1] - p[2]])
        e_v = np.array([v_ref[0] - v[0], 0.0, v_ref[1] - v[2]])
        F_des = self.cfg.Kd * e_v - GRAVITY
        hold_pin = done and float(np.hypot(*e_p)) < HOLD_PIN_RADIUS
        warm = (self.u[0], self.u[1]) if self.u is not None else None
        phi1, psi0, gamma, psi1, ok = allocate(
            self.cfg, v, F_des, self.sched_sign, hold=hold_pin, warm=warm)
        self.u = (phi1, psi0, gamma, psi1)
        self.ctrl_ok = ok

    def _deriv(self, state, m):
        v = state[3:6]
        F = core.instant_force(self.u, state[6], v, m)
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
        m = self.cfg.morphology()
        k1 = self._deriv(self.s, m)
        k2 = self._deriv(self.s + 0.5 * dt * k1, m)
        k3 = self._deriv(self.s + 0.5 * dt * k2, m)
        k4 = self._deriv(self.s + dt * k3, m)
        self.s = self.s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.s[1] = 0.0
        self.s[4] = 0.0                  # keep strictly planar
        self.t += dt
        if self.target is not None:      # capture check, every integration step
            pt = self._target_pos(self.t)
            if np.hypot(pt[0] - self.s[0], pt[1] - self.s[2]) < R_CAP:
                self.captured_t = self.t
                self.target = None
                self._hold_here()

    def snapshot(self):
        """Current render state (called under lock)."""
        following = (not self.ref.hold) and (self.t - self.ref.t0) < self.ref.duration()
        speed = float(np.hypot(self.s[3], self.s[5]))
        tg = self.target
        pt = self._target_pos(self.t) if tg is not None else (0.0, 0.0)
        return {
            "pursuing": tg is not None,
            "tx": float(pt[0]), "tz": float(pt[1]),
            "tvx": float(tg["v"][0]) if tg is not None else 0.0,
            "tvz": float(tg["v"][1]) if tg is not None else 0.0,
            "captured": bool(self.t - self.captured_t < 1.5),
            "t": self.t,
            "x": float(self.s[0]), "z": float(self.s[2]),
            "vx": float(self.s[3]), "vz": float(self.s[5]),
            "J": speed / core.SCALE,
            "phase": float(self.s[6] % (2.0 * np.pi)),
            "phi1": float(self.u[0]), "psi0": float(self.u[1]),
            "gamma": float(self.u[2]), "psi1": float(self.u[3]),
            "delta0": DELTA0,
            "cx": float(self.commanded[0]), "cz": float(self.commanded[1]),
            "following": bool(following),
            "ok": bool(getattr(self, "ctrl_ok", True)),
        }

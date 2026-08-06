"""Reference trajectories for the report1 test cases (and the GUI).

Path sources (freehand smoothing, the grid-snapped primitive course, the saved
GUI drawing), the trapezoidal-speed Reference that turns a polyline into
u_ref(t), and the hover / prey-pursuit case definitions. numpy only.
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

# hover case
V0 = (0.2, 0.0, 0.2)
T_HOVER = 10.0

# prescribed-trajectory cases
V_CRUISE, V_DRAWN, TAPER = 0.5, 0.35, 0.3
T_PAD = 4.6

# prey-pursuit case
PREY_P0 = (4.0, 0.0, 4.0)
PREY_V = (1.0, 0.0, 0.0)
V_CMD = 1.5
R_CAP = 0.5
T_MAX = 22.0


def prey_pos(t):
    return np.array([PREY_P0[0] + PREY_V[0] * t, 0.0,
                     PREY_P0[2] + PREY_V[2] * t])


def smooth_path(points, anchor, ds=0.04, sigma=3.0):
    """Uniformly resample and Gaussian-smooth a freehand (x, z) polyline.

    `anchor` is forced to be the exact start; the raw endpoint is pinned.
    Returns an (M, 2) array, or None if the path is too short."""
    P = np.asarray([anchor] + list(points), float)
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
    su = np.linspace(0.0, total, max(8, int(total / ds)))
    x = np.interp(su, s, P[:, 0])
    z = np.interp(su, s, P[:, 1])
    end_raw = np.array([x[-1], z[-1]])
    rad = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.convolve(np.pad(x, rad, mode="edge"), k, "valid")
    z = np.convolve(np.pad(z, rad, mode="edge"), k, "valid")
    out = np.column_stack([x, z])
    out[0], out[-1] = anchor, end_raw
    return out


class Reference:
    """Tracking target along a polyline at a trapezoidal speed schedule.

    Linear ramps from/to rest over the taper distance `taper`, constant cruise
    at `speed` in between (triangular if the path is too short). sample(t)
    returns (p_ref(2), v_ref(2), a_ref(2), done); after the end it holds."""

    def __init__(self, path, speed, taper, t0):
        self.t0 = t0
        v = max(float(speed), 1e-6)
        d = max(float(taper), 1e-3)
        if path is None or len(path) < 2:
            self.hold = True
            self.p_end = np.asarray(path[0] if path is not None else (0.0, 0.0), float)
            self.total, self.T3 = 0.0, 0.0
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
        self.a = v * v / (2.0 * d)
        if self.total >= 2.0 * d:
            self.vp, self.d_acc, self.Lc = v, d, self.total - 2.0 * d
        else:
            self.vp = float(np.sqrt(self.a * self.total))
            self.d_acc, self.Lc = self.total / 2.0, 0.0
        self.t_a = self.vp / self.a
        self.t_c = self.Lc / self.vp if self.vp > 0 else 0.0
        self.T3 = 2.0 * self.t_a + self.t_c

    def duration(self):
        return self.T3

    def _speed(self, tau):
        a, vp = self.a, self.vp
        if tau <= self.t_a:
            return a * tau, a, 0.5 * a * tau * tau
        if tau <= self.t_a + self.t_c:
            return vp, 0.0, self.d_acc + vp * (tau - self.t_a)
        u = tau - self.t_a - self.t_c
        return (max(vp - a * u, 0.0), -a,
                self.d_acc + self.Lc + vp * u - 0.5 * a * u * u)

    def sample(self, t):
        if self.hold or t - self.t0 >= self.T3:
            return self.p_end, np.zeros(2), np.zeros(2), True
        vmag, dvdt, s = self._speed(t - self.t0)
        s = min(s, self.total)
        p = np.array([np.interp(s, self.S, self.P[:, 0]),
                      np.interp(s, self.S, self.P[:, 1])])
        T = np.array([np.interp(s, self.S, self.T[:, 0]),
                      np.interp(s, self.S, self.T[:, 1])])
        dTds = np.array([np.interp(s, self.S, self.dTds[:, 0]),
                         np.interp(s, self.S, self.dTds[:, 1])])
        return p, vmag * T, dvdt * T + vmag ** 2 * dTds, False


def primitive_path(ds=0.04):
    """Grid-snapped test course: lines and tangent arcs (level start, 45-deg
    climb, tight loop, vertical descent, two sharp corners, quarter turn)."""
    R1, RL, RC = 1.0, 0.4, 0.25
    e = np.sqrt(2.0) / 2.0
    t1 = (1.0 + np.sqrt(2.0)) * RL * e
    cy = 1.0 + RL * (1.0 + np.sqrt(2.0))
    segs = []

    def line(p0, p1):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / ds)) + 1)
        s = np.linspace(0.0, 1.0, n)[:, None]
        segs.append(p0 + s * (p1 - p0))

    def arc(c, r, a0, a1):
        n = max(2, int(np.ceil(np.radians(abs(a1 - a0)) * r / ds)) + 1)
        th = np.radians(np.linspace(a0, a1, n))
        segs.append(np.column_stack([c[0] + r * np.cos(th),
                                     c[1] + r * np.sin(th)]))

    line((0.0, 0.0), (2.0 - np.sqrt(2.0), 0.0))
    arc((2.0 - np.sqrt(2.0), R1), R1, -90.0, -45.0)
    line((2.0 - e, 1.0 - e), (2.0 + t1, 1.0 + t1))
    arc((2.0 + RL, cy), RL, -45.0, 180.0)
    line((2.0, cy), (2.0, -1.0 + RC))
    arc((2.0 - RC, -1.0 + RC), RC, 0.0, -90.0)
    line((2.0 - RC, -1.0), (-1.0 + RC, -1.0))
    arc((-1.0 + RC, -1.0 + RC), RC, -90.0, -180.0)
    line((-1.0, -1.0 + RC), (-1.0, 0.0))
    arc((0.0, 0.0), 1.0, 180.0, 90.0)
    line((0.0, 1.0), (1.0, 1.0))

    assert max(np.linalg.norm(b[0] - a[-1])
               for a, b in zip(segs, segs[1:])) < 1e-9
    return np.vstack([s if i == 0 else s[1:] for i, s in enumerate(segs)])


def drawn_path(file=DATA / "drawn_path.json"):
    """The committed GUI freehand drawing, smoothed and shifted to the origin
    (None if absent)."""
    if not Path(file).exists():
        return None
    with open(file) as f:
        d = json.load(f)
    path = smooth_path([tuple(p) for p in d["points"]], tuple(d["anchor"]))
    return None if path is None else path - path[0]

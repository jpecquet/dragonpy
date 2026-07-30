"""Prescribed-trajectory tracking under the velocity law.

The reference path is a closed-course test trajectory idealized from a GUI
drawing (data/drawn_path.json) into grid-snapped primitives -- lines and
tangent circular arcs covering several regimes: a level start easing into a
45-deg climb, a tight loop (radius 0.4, 225-deg arc through its own
crossing), a vertical descent, two sharp corners (radius 0.25) framing a
level run, and a gentle quarter-turn (radius 1, centered on the origin) onto
a level finish. All polygon control vertices sit on the integer grid: (0,0),
(1,0), (2,1) [loop crossing], (2,-1), (-1,-1), (-1,1), (1,1).

A second case replays the committed GUI freehand drawing
(data/drawn_path.json, re-smoothed with the GUI's own smooth_path and
shifted to the origin) -- a dragonfly-shaped closed course -- when the file
is present.

The polyline is traversed with the GUI Reference's trapezoidal speed
schedule (linear ramps from/to rest over the taper distance, constant cruise
speed in between), which defines the reference velocity u_r(t); the force
demand is K (u_r - u) plus weight compensation (eq:fdes), with no position
or acceleration terms. The reference position is still computed, but only as
a diagnostic (the tracking-error trace); it is not fed back.

Companion to hover_gains.py (same layout):

    trajectory_gains.light.png       -- the primitive course: left, the
        (x*, z*) trajectory over the reference path; right, time traces of
        tracking error, velocity and all four control variables.
    trajectory_gains_drawn.light.png -- same layout for the freehand
        drawing (only if data/drawn_path.json exists).

Runs on the project env (numpy + matplotlib only).
"""

import json
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
from summer_paper.gui.dragonfly_sim import Reference, smooth_path  # noqa: E402

OUT_DIR = HERE.parent / "figures"
DRAWN_PATH = HERE.parent / "data" / "drawn_path.json"

K = REF_OMEGA_STAR / (2.0 * np.pi)  # one-beat velocity-law gain, K T* = 1 (eq:fdes)
T_PAD = 4.6                        # sim time past the reference end (settling)
J_PIN = 0.05                       # settled-near-rest threshold for the pin
GAMMA_HOVER = -GAMMA_H             # +x heading (sigma_x = +1 throughout)
PSI1_HOVER = float(np.clip(slave_psi1(0.0), *PSI1_LIM))

# Speed schedule (canonical figure values; the GUI's live settings are
# recorded in the dump but not used here). The drawn course is flown at a
# reduced cruise: its near-vertical plunges into the lower wing lobes
# saturate the descent trim at 0.5, and at the one-beat gain the
# departure persists down to 0.4.
V_CRUISE, TAPER = 0.5, 0.3
V_DRAWN = 0.35

SQ2 = np.sqrt(2.0)


def primitive_path(ds=0.04):
    """The test path as grid-snapped lines and tangent arcs (module doc)."""
    R1, RL, RC = 1.0, 0.4, 0.25   # ease-in / loop / sharp-corner radii
    e = SQ2 / 2.0
    t1 = (1.0 + SQ2) * RL * e     # loop tangency overshoot past (2, 1)
    cy = 1.0 + RL * (1.0 + SQ2)   # loop center height (tangency-derived)
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

    line((0.0, 0.0), (2.0 - SQ2, 0.0))              # level start
    arc((2.0 - SQ2, R1), R1, -90.0, -45.0)          # ease into the 45-deg climb
    line((2.0 - e, 1.0 - e), (2.0 + t1, 1.0 + t1))  # climb through the crossing
    arc((2.0 + RL, cy), RL, -45.0, 180.0)           # tight loop over the top
    line((2.0, cy), (2.0, -1.0 + RC))               # vertical descent
    arc((2.0 - RC, -1.0 + RC), RC, 0.0, -90.0)      # sharp bottom-right corner
    line((2.0 - RC, -1.0), (-1.0 + RC, -1.0))       # level run left
    arc((-1.0 + RC, -1.0 + RC), RC, -90.0, -180.0)  # sharp bottom-left corner
    line((-1.0, -1.0 + RC), (-1.0, 0.0))            # climb the left wall
    arc((0.0, 0.0), 1.0, 180.0, 90.0)               # gentle quarter turn
    line((0.0, 1.0), (1.0, 1.0))                    # level finish

    gaps = [np.linalg.norm(b[0] - a[-1]) for a, b in zip(segs, segs[1:])]
    assert max(gaps) < 1e-9, "primitive chain is not continuous"
    return np.vstack([s if i == 0 else s[1:] for i, s in enumerate(segs)])


def drawn_path():
    """(M, 2) polyline of the committed GUI drawing, or None if absent."""
    if not DRAWN_PATH.exists():
        return None
    with open(DRAWN_PATH) as f:
        d = json.load(f)
    path = smooth_path([tuple(p) for p in d["points"]], tuple(d["anchor"]))
    print(f"reference: GUI drawing ({DRAWN_PATH.relative_to(REPO_ROOT)}, "
          f"drawn at v_follow = {d['v_follow']:.2f}, "
          f"taper = {d['taper']:.2f})")
    return path - path[0]                        # start at the origin


# ---------------------------------------------------------------------------

def allocate(v, F_des, done, warm):
    """Control-law allocation: schedule while following, pin once settled.

    The pin engages once the reference velocity has vanished (trajectory
    exhausted) and the body has settled near rest (sec:control-law). The
    stroke-plane lean sign is latched at sigma_x = +1 (the commanded
    heading is +x throughout)."""
    gamma, _, J = gamma_schedule(v, sx=1.0)
    if done and J < J_PIN:
        gamma, ps1 = GAMMA_HOVER, PSI1_HOVER
    else:
        ps1 = float(np.clip(slave_psi1(J), *PSI1_LIM))
    u0 = warm if warm is not None else (np.radians(20.0), 0.0)
    phi1, psi0, _ = trim_fast(gamma, ps1, tuple(v), F_des, u0, N_PHASE)
    return (phi1, psi0, gamma, ps1)


def simulate(ref, T):
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
        p2, v2, _, done = ref.sample(t)
        p_ref = np.array([p2[0], 0.0, p2[1]])
        v_ref = np.array([v2[0], 0.0, v2[1]])
        e_p = p_ref - s[0:3]                     # diagnostic only
        if t - last_ctrl >= period - 1e-9:
            e_v = v_ref - s[3:6]
            F_des = K * e_v - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u = allocate(s[3:6], F_des, done, warm)
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


def figure(run, path, style, tag=""):
    c1, c2, c3 = "black", "#b2182b", "#2166ac"
    fig = plt.figure(figsize=(6.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.45])

    # left: trajectory over the reference path
    axT = fig.add_subplot(gs[:, 0])
    axT.plot(path[:, 0], path[:, 1], color="0.65", lw=2.6, alpha=0.6,
             label="reference", zorder=0)
    axT.plot(run["x"], run["z"], color=c1, lw=1.2, label="achieved")
    axT.plot(run["x"][0], run["z"][0], "o", color=c1, ms=4, mfc="white")
    axT.plot(path[-1, 0], path[-1, 1], "+", color=c2, ms=9, mew=1.6, zorder=5)
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
        ax.set_xlim(0.0, t[-1])
        ax.legend(fontsize=style.font_size - 4, frameon=True,
                  loc="upper right", ncol=3, handlelength=1.3,
                  labelspacing=0.2, columnspacing=0.8)
    for ax in rows[:-1]:
        ax.tick_params(labelbottom=False)

    out = OUT_DIR / f"trajectory_gains{tag}.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def run_case(path, tag, style, v_cruise):
    ref = Reference(path, v_cruise, TAPER, 0.0)
    print(f"path: {ref.total:.2f} body lengths; cruise {v_cruise}, "
          f"taper {TAPER}, reference ends at t = {ref.duration():.2f}")
    run = simulate(ref, ref.duration() + T_PAD)
    out = figure(run, path, style, tag)
    ep = np.hypot(run["ex"], run["ez"])
    follow = run["t"] <= ref.duration()
    print(f"max |e_p| following = {ep[follow].max():.3f}, "
          f"rms = {np.sqrt(np.mean(ep[follow] ** 2)):.3f}, "
          f"final |e_p| = {ep[-1]:.3f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"K = {K}")

    print("reference: primitive test path (lines + arcs)")
    run_case(primitive_path(), "", style, V_CRUISE)
    drawn = drawn_path()
    if drawn is not None:
        run_case(drawn, "_drawn", style, V_DRAWN)


if __name__ == "__main__":
    main()

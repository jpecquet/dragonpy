"""
Gain sensitivity study for the velocity law (sec:control-law, sec:results).

Repeats the hover recovery of hover_gains.py (velocity perturbation
(0.2, 0.2), u_r = 0) across a sweep of gains K, expressed as K T* with
T* = 2 pi / omega* the wingbeat period, and checks the discrete-loop
model of sec:control-law against the closed-loop simulation:

    contraction -- the velocity error measured at the control instants
        should contract by (1 - K T*) each beat while the trim is
        unsaturated (signed estimate: projection of each beat's error
        onto the previous one);
    settling    -- time for the beat-sampled error to fall below 5% of
        the initial perturbation and stay there; theory
        T* ln(0.05) / ln|1 - K T*|, floored at one beat;
    drift       -- accumulated position offset |x(T)|, first-order
        prediction |u(0)| / K (no position feedback);
    saturation  -- fraction of transient beats (error above a floor)
        where the trim fails to converge or sits at a kinematic bound.

    gain_sweep.light.png -- the four metrics against K T*, with the
        theory overlays, the one-beat gain K T* = 1, the predicted
        stability bound K T* = 2, and the report's K = 2.7 marked.

Runs on the project env (numpy + matplotlib only).
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import REF_OMEGA_STAR
from maneuver_control import trim_fast, PHI1_LIM, PSI0_LIM, PSI1_LIM
from generalized_control import (
    GAMMA_H, N_PHASE, gamma_schedule, make_p, slave_psi1,
)
from hover_drift import GRAVITY, instant_force

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"

T_STAR = 2.0 * np.pi / REF_OMEGA_STAR   # wingbeat period, time units
K_REPORT = 2.7                          # the gain used throughout the report
V0 = (0.2, 0.0, 0.2)                    # initial velocity perturbation
T_SIM = 20.0
J_PIN = 0.05                            # settled-near-rest threshold for the pin
GAMMA_HOVER = -GAMMA_H                  # +x heading default (sigma_x = +1)
PSI1_HOVER = float(np.clip(slave_psi1(0.0), *PSI1_LIM))

SETTLE_FRAC = 0.05                      # settling threshold on |e| / |e(0)|
E_FLOOR = 0.02                          # transient floor for beat statistics
U_BLOWUP = 5.0                          # abort speed for unstable runs
BOUND_EPS = np.radians(0.05)            # "at a kinematic bound" tolerance

# K T* grid: across the deadbeat point and the predicted stability bound,
# denser near the bound, with the report's gain included exactly.
KT_GRID = np.unique(np.concatenate([
    np.arange(0.2, 1.9, 0.15),
    np.arange(1.9, 2.75, 0.1),
    [K_REPORT * T_STAR],
]))


def allocate(v, F_des, warm):
    """hover_gains allocation + saturation flag (trim failed or at a bound)."""
    gamma, _, J = gamma_schedule(v, sx=1.0)
    if J < J_PIN:
        gamma, ps1 = GAMMA_HOVER, PSI1_HOVER
    else:
        ps1 = float(np.clip(slave_psi1(J), *PSI1_LIM))
    u0 = warm if warm is not None else (np.radians(20.0), 0.0)
    phi1, psi0, ok = trim_fast(gamma, ps1, tuple(v), F_des, u0, N_PHASE)
    at_bound = (phi1 <= PHI1_LIM[0] + BOUND_EPS
                or phi1 >= PHI1_LIM[1] - BOUND_EPS
                or psi0 <= PSI0_LIM[0] + BOUND_EPS
                or psi0 >= PSI0_LIM[1] - BOUND_EPS)
    return (phi1, psi0, gamma, ps1), (not ok) or at_bound


def simulate(K, T=T_SIM):
    """Closed-loop hover recovery at gain K; per-beat error and saturation log."""
    omega = REF_OMEGA_STAR
    period = 2.0 * np.pi / omega
    dt = period / 120.0
    n = int(T / dt)

    s = np.zeros(7)
    s[3:6] = V0
    u = None
    last_ctrl = -1e9
    beats = {"e": [], "sat": []}
    blowup = False

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
        if t - last_ctrl >= period - 1e-9:
            e_v = -s[3:6]                        # u_r = 0
            F_des = K * e_v - GRAVITY
            warm = (u[0], u[1]) if u is not None else None
            u, sat = allocate(s[3:6], F_des, warm)
            beats["e"].append(e_v.copy())
            beats["sat"].append(sat)
            last_ctrl = t
        if np.linalg.norm(s[3:6]) > U_BLOWUP:
            blowup = True
            break
        if i < n:
            k1 = deriv(s, u)
            k2 = deriv(s + 0.5 * dt * k1, u)
            k3 = deriv(s + 0.5 * dt * k2, u)
            k4 = deriv(s + dt * k3, u)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {"e": np.array(beats["e"]), "sat": np.array(beats["sat"]),
            "x_final": s[0:3].copy(), "blowup": blowup}


def metrics(run):
    """Per-beat contraction, settling time, drift, and saturation fraction."""
    e = run["e"]
    sat = run["sat"]
    mag = np.linalg.norm(e, axis=1)
    e0 = mag[0]

    # signed contraction: least-squares fit of e_{n+1} = r e_n over transient
    # beats (|e_n|^2-weighted, so near-floor beats do not distort it)
    idx = [i for i in range(len(e) - 1) if mag[i] > E_FLOOR]
    if idx:
        num = sum(float(np.dot(e[i + 1], e[i])) for i in idx)
        den = sum(float(np.dot(e[i], e[i])) for i in idx)
        contraction = num / den
    else:
        contraction = np.nan

    # settling: first beat below threshold that stays below (beat-sampled)
    settle = np.nan
    if not run["blowup"]:
        th = SETTLE_FRAC * e0
        below = mag < th
        for i in range(len(mag)):
            if below[i:].all():
                settle = i * T_STAR
                break

    drift = np.nan if run["blowup"] else float(np.linalg.norm(run["x_final"]))
    transient = mag > E_FLOOR
    sat_frac = float(sat[transient].mean()) if transient.any() else 0.0
    return contraction, settle, drift, sat_frac


def figure(kts, table, style):
    text = style.text_color
    muted = style.muted_text_color
    theory_c = "#b2182b"
    kt_report = K_REPORT * T_STAR
    contr, settle, drift, satf = (np.array([row[j] for row in table])
                                  for j in range(4))

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True,
                             constrained_layout=True)
    (axA, axB), (axC, axD) = axes
    kt_th = np.linspace(0.02, 2.75, 400)

    def decorate(ax):
        ax.axvspan(2.0, 2.78, color="0.93", zorder=0)
        ax.axvline(1.0, color=muted, lw=0.8, ls=":", zorder=1)
        ax.set_xlim(0.0, 2.78)

    # (a) per-beat contraction
    decorate(axA)
    axA.axhline(0.0, color="0.85", lw=0.8, zorder=1)
    axA.plot(kt_th, 1.0 - kt_th, color=theory_c, lw=1.2, ls="--",
             label=r"$1 - K T^*$", zorder=2)
    axA.plot(kts, contr, "o", color=text, ms=4, label="measured", zorder=3)
    axA.plot(kt_report, contr[np.argmin(np.abs(kts - kt_report))], "o",
             mfc="none", mec=text, ms=9, mew=1.2, zorder=4)
    axA.annotate(rf"$K = {K_REPORT}$",
                 xy=(kt_report, contr[np.argmin(np.abs(kts - kt_report))]),
                 xytext=(kt_report + 0.12, 0.32), color=text,
                 arrowprops=dict(arrowstyle="-", color=muted, lw=0.7))
    axA.set_ylabel("per-beat contraction")
    axA.set_title("(a) velocity-error contraction", fontsize=style.font_size)
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="lower left",
               handlelength=1.6, labelspacing=0.3)

    # (b) settling time
    decorate(axB)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_th = np.log(SETTLE_FRAC) / np.log(np.abs(1.0 - kt_th))
    t_th = T_STAR * np.maximum(1.0, n_th)
    t_th[kt_th >= 2.0] = np.nan
    axB.plot(kt_th, t_th, color=theory_c, lw=1.2, ls="--",
             label="discrete model", zorder=2)
    axB.plot(kts, settle, "o", color=text, ms=4, label="measured", zorder=3)
    axB.plot(kt_report, settle[np.argmin(np.abs(kts - kt_report))], "o",
             mfc="none", mec=text, ms=9, mew=1.2, zorder=4)
    axB.set_ylabel(r"settling time ($\sqrt{L/g}$)")
    axB.set_ylim(0.0, 8.0)
    axB.set_title("(b) settling to 5%", fontsize=style.font_size)
    axB.legend(fontsize=style.font_size - 3, frameon=True, loc="upper center",
               handlelength=1.6, labelspacing=0.3)

    # (c) accumulated drift; the discrete model integrates the piecewise-
    # linear velocity decay: |x| = |u0| T* (2 - KT*) / (2 KT*) = |u0| (1/K - T*/2)
    decorate(axC)
    drift_th = np.linalg.norm(V0) * T_STAR * (2.0 - kt_th) / (2.0 * kt_th)
    drift_th[kt_th >= 2.0] = np.nan
    axC.plot(kt_th, drift_th, color=theory_c,
             lw=1.2, ls="--", label="discrete model", zorder=2)
    axC.plot(kts, drift, "o", color=text, ms=4, label="measured", zorder=3)
    axC.plot(kt_report, drift[np.argmin(np.abs(kts - kt_report))], "o",
             mfc="none", mec=text, ms=9, mew=1.2, zorder=4)
    axC.set_ylabel("final offset (body lengths)")
    axC.set_ylim(0.0, 0.8)
    axC.set_xlabel(r"$K T^*$")
    axC.set_title("(c) position drift", fontsize=style.font_size)
    axC.legend(fontsize=style.font_size - 3, frameon=True, loc="upper right",
               handlelength=1.6, labelspacing=0.3)

    # (d) saturation fraction
    decorate(axD)
    axD.plot(kts, satf, "o", color=text, ms=4, zorder=3)
    axD.plot(kt_report, satf[np.argmin(np.abs(kts - kt_report))], "o",
             mfc="none", mec=text, ms=9, mew=1.2, zorder=4)
    axD.set_ylabel("saturated beats (fraction)")
    axD.set_ylim(-0.04, 1.04)
    axD.set_xlabel(r"$K T^*$")
    axD.set_title("(d) trim saturation", fontsize=style.font_size)

    out = OUT_DIR / "gain_sweep.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    quick = "--test" in sys.argv
    kts = np.array([K_REPORT * T_STAR]) if quick else KT_GRID
    print(f"T* = {T_STAR:.4f}; sweeping {len(kts)} gains, "
          f"KT* in [{kts[0]:.2f}, {kts[-1]:.2f}]")
    table = []
    for kt in kts:
        K = kt / T_STAR
        t0 = time.time()
        run = simulate(K)
        m = metrics(run)
        table.append(m)
        drift_th = np.linalg.norm(V0) * (1.0 / K - T_STAR / 2.0)
        print(f"  KT* = {kt:.3f} (K = {K:.2f}): contraction = {m[0]:+.3f} "
              f"(theory {1 - kt:+.3f}), settle = {m[1]:.2f}, "
              f"drift = {m[2]:.3f} (theory {drift_th:.3f}), "
              f"sat = {m[3]:.2f}, blowup = {run['blowup']} "
              f"[{time.time() - t0:.1f} s]")
    if quick:
        return
    out = figure(kts, table, style)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

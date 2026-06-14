"""
Proportional hover control with the two selected variables.

Section 2 left two control handles at a fixed stroke plane (gamma0), frequency
(omega*), and C_psi-optimal pitch (psi1, delta0):

    phi1  -- stroke amplitude  -> force MAGNITUDE  (q* ~ phi1^2)  -> vertical force
    psi0  -- mean wing pitch   -> force DIRECTION  (beta(psi0))   -> horizontal force

With attitude frozen this is a square 2-input / 2-DOF (x = fore/aft, z = vertical)
problem, and the control authority is nearly triangular at the hover trim:
    dFz/dphi1 ~ +5.8 (phi1 is almost pure lift),
    dFx/dpsi0 ~ -2.1, with dFz/dpsi0 ~ -1.0 (tilting psi0 also sheds some lift).
So we pair amplitude->altitude and mean-pitch->station-keeping, each a single-axis
constant-gain loop, and let the strong/fast phi1 loop mop up the psi0 lift cross-term.

We use PURE PROPORTIONAL on position (no rate term, no integrator). There is passive
aerodynamic damping (dFz/dvz ~ -0.57, dFx/dvx ~ -0.85: a body moving through the air
feels a velocity-opposing force), so the loop is stable on its own -- lightly damped
(zeta ~ 0.2 at this bandwidth), so it returns with some ringing.

    phi1 = phi1_trim - g_phi * (z - z_ref)
    psi0 = psi0_trim - g_psi * (x - x_ref)

The gains g_phi, g_psi are designed ONCE at the reference config (target omega_n via
g = omega_n^2 / effectiveness) and then held FIXED. main() runs the reference step
response and then a robustness sweep: the same fixed gains on off-nominal morphology
(omega*, Aw/m, Lw, gamma0 across the Table-1 ranges), trimming each plant to hover.

Operating point: counterstroking (sigma0 = 180 deg), massless body (m_w = 0): at
counterstroking wing inertia cancels, so it is dropped for a clean point mass.

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hover_drift import base_params, solve_equilibrium, instant_force, GRAVITY
from feasibility import cycle_averaged_force, replace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE / "figures"

OMEGA_N = 1.5               # target closed-loop natural frequency (1/T0) at the design point
PHI1_LIM = np.radians((2.0, 60.0))
PSI0_LIM = np.radians((-60.0, 60.0))

X0, Z0 = 0.10, -0.10        # initial (x, z) offset, body lengths; command is the origin
N_CYCLES = 40
SPC = 200


def effectiveness(p):
    """Diagonal control effectiveness and passive aero damping at p's trim.

    Finite-difference the cycle-averaged force: bz = dFz/dphi1, bx = dFx/dpsi0
    (the handles), and cz = -dFz/dvz, cx = -dFx/dvx (passive damping, positive when
    the force opposes the velocity)."""
    F0 = cycle_averaged_force(p, 256)
    dphi, dpsi, dv = np.radians(1.0), np.radians(1.0), 0.05
    bz = (cycle_averaged_force(replace(p, phi1=p.phi1 + dphi), 256)[2] - F0[2]) / dphi
    bx = (cycle_averaged_force(replace(p, psi0=p.psi0 + dpsi), 256)[0] - F0[0]) / dpsi
    cz = -(cycle_averaged_force(replace(p, v_body=(0, 0, dv)), 256)[2] - F0[2]) / dv
    cx = -(cycle_averaged_force(replace(p, v_body=(dv, 0, 0)), 256)[0] - F0[0]) / dv
    return bz, bx, cz, cx


def trim(p):
    """Return p re-solved to the hover trim (psi0, phi1) for its morphology."""
    psi0, phi1 = solve_equilibrium(p, mass_ratio=0.0)
    return replace(p, psi0=psi0, phi1=phi1)


def simulate_control(p, g_phi, g_psi, n_cycles=N_CYCLES, steps_per_cycle=SPC):
    """Closed-loop P-only point-mass run from (X0, Z0) to the origin, fixed gains."""
    omega = 2.0 * np.pi * p.wing_frequency
    period = 2.0 * np.pi / omega
    dt = period / steps_per_cycle
    n_steps = int(n_cycles * steps_per_cycle)
    phi1_0, psi0_0 = p.phi1, p.psi0

    def actuate(pos):
        phi1 = np.clip(phi1_0 - g_phi * pos[2], *PHI1_LIM)   # amplitude -> altitude
        psi0 = np.clip(psi0_0 - g_psi * pos[0], *PSI0_LIM)   # mean pitch -> fore/aft
        return phi1, psi0

    def deriv(s):
        pos, vel, phase = s[0:3], s[3:6], s[6]
        phi1, psi0 = actuate(pos)
        F = instant_force(replace(p, phi1=phi1, psi0=psi0), phase, omega, vel)
        d = np.empty(7)
        d[0:3] = vel
        d[3:6] = F + GRAVITY
        d[6] = omega
        return d

    s = np.array([X0, 0.0, Z0, 0.0, 0.0, 0.0, 0.0])
    t = np.empty(n_steps + 1)
    pos = np.empty((n_steps + 1, 3))
    t[0], pos[0] = 0.0, s[0:3]
    for i in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i + 1], pos[i + 1] = (i + 1) * dt, s[0:3]
    return t, pos, period, steps_per_cycle


def settling_cycles(pos_axis, spc, e0, tol_frac=0.05):
    """Cycles for the per-cycle-sampled position to stay within tol_frac*|e0|.

    Sampling once per wingbeat strips the within-cycle bob so the metric reflects
    the macro return, not the residual hover oscillation."""
    per_cycle = pos_axis[::spc]
    bad = np.where(np.abs(per_cycle) > tol_frac * abs(e0))[0]
    return float(bad[-1]) if len(bad) else 0.0


def damping_ratios(g_phi, g_psi, eff):
    """Closed-loop omega_n and zeta per axis for fixed gains on a plant (eff)."""
    bz, bx, cz, cx = eff
    wn_z = np.sqrt(max(g_phi * bz, 0.0))
    wn_x = np.sqrt(max(g_psi * bx, 0.0))      # g_psi and bx share sign -> product > 0
    zz = cz / (2 * wn_z) if wn_z > 0 else np.inf
    zx = cx / (2 * wn_x) if wn_x > 0 else np.inf
    return wn_z, zz, wn_x, zx


def step_figure(configs, out_path, g_phi, g_psi, font_size, match_omega_n=False):
    """Overlay P-only step responses (x*, z*) for a list of (label, overrides, ls).

    Each plant is re-trimmed to hover. With match_omega_n=False every config uses
    the same fixed gains (a robustness view). With match_omega_n=True the gains are
    retuned per config to hit OMEGA_N on both axes, so the curves share a bandwidth
    and the only remaining difference is the damping ratio -- the right convention
    when the figure's job is to isolate the passive damping (e.g. vs gamma0)."""
    fig, (axX, axZ) = plt.subplots(1, 2, figsize=(6.0, 3.0), sharex=True,
                                   constrained_layout=True)
    for lab, over, ls in configs:
        p = trim(base_params(sigma0=np.pi, **over))
        if match_omega_n:
            bz, bx, _cz, _cx = effectiveness(p)
            gp, gs = OMEGA_N ** 2 / bz, OMEGA_N ** 2 / bx
        else:
            gp, gs = g_phi, g_psi
        t, pos, period, _spc = simulate_control(p, gp, gs)
        cyc = t / period
        axX.plot(cyc, pos[:, 0], color="black", lw=1.5, ls=ls, label=lab)
        axZ.plot(cyc, pos[:, 2], color="black", lw=1.5, ls=ls, label=lab)
    for ax, lab in [(axX, r"$x^*$"), (axZ, r"$z^*$")]:
        ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("Wingbeat cycles")
        ax.set_ylabel(lab)
        ax.set_xlim(0, N_CYCLES)
    axZ.legend(fontsize=font_size - 2, frameon=True, loc="upper right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- design the fixed gains at the reference config ---
    p_ref = trim(base_params(sigma0=np.pi))
    bz, bx, cz, cx = effectiveness(p_ref)
    g_phi = OMEGA_N ** 2 / bz
    g_psi = OMEGA_N ** 2 / bx               # bx < 0 -> g_psi < 0 -> restoring
    print(f"reference trim phi1*={np.degrees(p_ref.phi1):.1f} deg, "
          f"psi0*={np.degrees(p_ref.psi0):.1f} deg; bz={bz:.2f} bx={bx:.2f} "
          f"cz={cz:.2f} cx={cx:.2f}")
    print(f"fixed P gains: g_phi={g_phi:.3f}, g_psi={g_psi:.3f} (rad/L); "
          f"design omega_n={OMEGA_N}, zeta z={cz/(2*OMEGA_N):.2f} x={cx/(2*OMEGA_N):.2f}")

    # --- step response: wing loading (main driver of passive damping) ---
    wing_loading = [
        (r"$A_w^*/m^* = 0.05$", dict(Aw_over_mb=0.05), "--"),
        (r"$A_w^*/m^* = 0.15$", {}, "-"),
        (r"$A_w^*/m^* = 0.25$", dict(Aw_over_mb=0.25), ":"),
    ]
    out = step_figure(wing_loading, OUT_DIR / "hover_control_step.light.png",
                      g_phi, g_psi, style.font_size)
    print(f"wrote {out.relative_to(REPO_ROOT)}")

    # --- step response: stroke-plane angle (damping anisotropy) ---
    stroke_plane = [
        (r"$\gamma_0 = 0^\circ$", dict(gamma0=0.0), "--"),
        (r"$\gamma_0 = 40^\circ$", {}, "-"),
    ]
    out_g = step_figure(stroke_plane, OUT_DIR / "hover_control_step_gamma.light.png",
                        g_phi, g_psi, style.font_size, match_omega_n=True)
    print(f"wrote {out_g.relative_to(REPO_ROOT)}")

    # --- robustness: same fixed gains, off-nominal morphology ---
    configs = [
        ("reference", {}),
        ("omega*=8", dict(omega_star=8.0)),
        ("omega*=20", dict(omega_star=20.0)),
        ("Aw/m=0.05", dict(Aw_over_mb=0.05)),
        ("Aw/m=0.25", dict(Aw_over_mb=0.25)),
        ("Lw=0.65", dict(Lw=0.65)),
        ("Lw=0.85", dict(Lw=0.85)),
        ("gamma0=20", dict(gamma0=np.radians(20.0))),
        ("gamma0=60", dict(gamma0=np.radians(60.0))),
    ]
    print("\nrobustness (fixed reference gains, plant re-trimmed to hover):")
    print(f"{'config':>11} {'wn_z':>5} {'z_z':>5} {'wn_x':>5} {'z_x':>5} "
          f"{'set_x':>6} {'set_z':>6} {'|end|':>7} {'stable':>7}")
    for name, over in configs:
        p = trim(base_params(sigma0=np.pi, **over))
        eff = effectiveness(p)
        wn_z, zz, wn_x, zx = damping_ratios(g_phi, g_psi, eff)
        t, pos, period, spc = simulate_control(p, g_phi, g_psi)
        sx = settling_cycles(pos[:, 0], spc, X0)
        sz = settling_cycles(pos[:, 2], spc, Z0)
        end = np.hypot(pos[-1, 0], pos[-1, 2])
        stable = np.all(np.abs(pos[:, [0, 2]]) < 1.0)   # bounded, never runs away
        print(f"{name:>11} {wn_z:5.2f} {zz:5.2f} {wn_x:5.2f} {zx:5.2f} "
              f"{sx:6.1f} {sz:6.1f} {end:7.4f} {'yes' if stable else 'NO':>7}")


if __name__ == "__main__":
    main()

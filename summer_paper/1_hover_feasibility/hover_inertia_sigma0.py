"""
Within-cycle body oscillation vs the fore/hind phase shift sigma0.

The Unsteady Dynamics section weighs two sources of within-wingbeat body motion at
an otherwise-hovering operating point:

    (1) aero fluctuation -- the cycle-averaged force hovers, but the instantaneous
        aerodynamic force does not, so the body accelerates back and forth;
    (2) wing inertia -- the wings carry mass, so the body recoils to hold the
        system centre of mass on its smooth aero+gravity trajectory.

Both depend on the fore/hind phase shift sigma0, which the model does NOT fix:
counterstroking (sigma0 = 180 deg) puts the fore and hind pairs in antiphase, so
their aero force peaks interleave (smoother total force) AND their wing-COM momenta
cancel (no net recoil). In phase (sigma0 = 0) both add. This script sweeps sigma0
from 0 to 180 deg at the reference configuration and plots the overall within-cycle
body oscillation amplitude for massless wings (m_w = 0) and for m_w = 0.02, so the
gap between the curves is what wing inertia adds on top of the aero fluctuation.

Metric: oscillation amplitude = half the largest chord of the settled within-cycle
(x, z) body path. A raw single-cycle peak-to-peak is noisy (the loop is slightly
non-elliptical and wanders cycle to cycle) and, for m_w > 0, rides on the secular
drift (the coupled hover slowly translates as the body's velocity rectifies the V^2
aero force -- reported as drift/cyc, sharp at low sigma0). So each coordinate is
least-squares fit over the last cycles to a linear drift plus a few wingbeat
harmonics, and the amplitude is read from the harmonic part alone. The 2nd harmonic
is kept because at counterstroking the fundamental cancels and the whole residual
oscillation is at 2*omega.

Reference configuration: gamma0 = 40 deg inclined hover, Aw/mb = 0.15, omega* = 14,
Lw = 0.75, C_psi held near max (psi1 = 51 deg, delta0 = 90 deg); psi0, phi1 solved
for hover at each wing mass. Wing mass m_w = 0.02 per wing (mid Wakeling Anisoptera
range).

Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hover_drift import base_params, solve_equilibrium, simulate
from feasibility import replace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE / "figures"
MASS_RATIO = 0.02            # per-wing wing/body mass (mid Wakeling Anisoptera range)
SIGMA0_DEG = np.arange(0.0, 180.1, 10.0)
N_CYCLES = 16               # integration length (transient settles by ~cycle 12)
SPC = 200                   # integration steps per wingbeat
N_HARM = 2                  # wingbeat harmonics kept in the oscillation fit
FIT_CYCLES = 4              # cycles fitted, at the end of the run


def oscillation_amplitude(t, pos, omega, period):
    """Amplitude of the settled within-cycle body oscillation in the (x, z) plane.

    Least-squares fit each coordinate over the last FIT_CYCLES cycles to a linear
    drift plus N_HARM wingbeat harmonics, reconstruct one clean period from the
    harmonics alone (drift dropped), and return half its largest chord. The
    amplitude is stable to ~1% across fit windows, so it reads the genuine settled
    oscillation rather than a transient or the secular drift."""
    m = t >= t[-1] - FIT_CYCLES * period
    tt = t[m]
    cols = [np.ones_like(tt), tt]
    for k in range(1, N_HARM + 1):
        cols += [np.cos(k * omega * tt), np.sin(k * omega * tt)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), pos[m][:, [0, 2]], rcond=None)
    th = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    rec = np.zeros((720, 2))
    for k in range(1, N_HARM + 1):
        rec += (np.outer(np.cos(k * th), coef[2 * k])
                + np.outer(np.sin(k * th), coef[2 * k + 1]))
    d = rec[:, None, :] - rec[None, :, :]
    return 0.5 * float(np.sqrt((d * d).sum(-1)).max())


def run_case(sigma0, mass_ratio):
    """Settled body-oscillation amplitude and secular drift for one hover.

    Solves the hover for this wing mass, integrates the coupled dynamics, and reads
    the settled oscillation (harmonic fit) plus the net drift over the last cycle.
    Both returned in body lengths."""
    p = base_params(sigma0=sigma0)
    psi0, phi1 = solve_equilibrium(p, mass_ratio=mass_ratio)
    p = replace(p, psi0=psi0, phi1=phi1)
    omega = 2.0 * np.pi * p.wing_frequency
    t, pos, _v, period = simulate(p, n_cycles=N_CYCLES, steps_per_cycle=SPC,
                                  mass_ratio=mass_ratio)
    amp = oscillation_amplitude(t, pos, omega, period)
    last = pos[t >= t[-1] - period]
    drift = float(np.linalg.norm(last[-1] - last[0]))
    return amp, drift


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sig = np.radians(SIGMA0_DEG)
    amp0 = np.array([run_case(s, 0.0)[0] for s in sig])
    res2 = [run_case(s, MASS_RATIO) for s in sig]
    amp2 = np.array([a for a, _ in res2])
    drift2 = np.array([d for _, d in res2])

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    ax.plot(SIGMA0_DEG, amp0 * 1e3, color="black", ls="-", lw=1.8,
            label=r"$m_w^* = 0$")
    ax.plot(SIGMA0_DEG, amp2 * 1e3, color="black", ls=":", lw=1.8,
            label=fr"$m_w^* = {MASS_RATIO}$")
    ax.set_xlabel(r"$\sigma_0$ (deg)")
    ax.set_ylabel(r"Body oscillation amplitude ($10^{-3}\,L$)")
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_ylim(0, None)
    ax.legend(fontsize=style.font_size - 1, frameon=True, loc="upper right")

    out = OUT_DIR / "inertia_vs_aero_sigma0.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    # --- report numbers ---
    print(f"reference config (gamma0=40 deg); oscillation amplitude = half largest "
          f"chord of the settled body path ({N_HARM}-harmonic fit) [1e-3 L]")
    print(f"{'sigma0':>7} {'mw=0':>8} {'mw=0.02':>9} {'ratio':>7} "
          f"{'drift/cyc[L]':>13}")
    for sd, a0, a2, dr in zip(SIGMA0_DEG, amp0, amp2, drift2):
        print(f"{sd:7.0f} {a0*1e3:8.2f} {a2*1e3:9.2f} {a2/a0:7.2f} {dr:13.4f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

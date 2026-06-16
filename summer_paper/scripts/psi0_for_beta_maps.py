"""psi0 required to reach a target force direction beta, over the velocity disk.

For each body velocity (advance ratio J, angle chi from the stroke-plane
normal) we invert the monotonic beta(psi0) law by bisection over the Table-1
range psi0 in [-60, 60] deg, at delta0 = 90 deg and the reference
configuration. One polar panel per target beta; hatched cells mark velocities
where no psi0 in range reaches the target (loss of directional authority).

Two modes (CLI arg, default both):
  hover   -- psi1 fixed at the hover optimum (51 deg);
  retuned -- psi1 re-optimized for force magnitude at each (J, chi) at psi0=0
             (the psi1^opt field of velocity_cpsi_maps.py), asking whether
             amplitude retuning also buys back directional authority.

Output: light-mode figure(s) in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, cycle_averaged_force,
)
from velocity_cpsi_maps import best_psi1, setup_polar

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
N_PHASE = 128
CMAP = "RdBu_r"

PHI1_REF = REF_S0 / (STUDY_SPAN_FRAC * REF_LW)
PSI1_HOVER = np.radians(51.0)
DELTA0 = np.pi / 2
PSI0_MAX = np.radians(60.0)     # Table 1 range

BETA_TARGETS = [0.0, 15.0, 30.0, 45.0]   # degrees
J_GRID = np.linspace(0.0, 0.8, 11)
CHI_FULL = np.radians(np.arange(-180.0, 181.0, 10.0))


def beta_of(J, chi, psi0, psi1):
    U = J * REF_S0 * REF_OMEGA_STAR
    p = Params(A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR,
               phi1=PHI1_REF, Lw=REF_LW, gamma0=0.0,
               v_body=(U * np.sin(chi), 0.0, U * np.cos(chi)),
               element_span_fracs=STUDY_ELEMENT,
               psi0=psi0, psi1=psi1, delta0=DELTA0)
    f = cycle_averaged_force(p, N_PHASE)
    return np.arctan2(f[0], f[2])


def psi0_for(J, chi, psi1, beta_target, b_lo, b_hi, iters=22):
    """Invert beta(psi0) = beta_target by bisection; NaN if out of reach.

    beta is decreasing in psi0, so the bracket is [b_hi, b_lo] in beta with
    psi0 endpoints [-PSI0_MAX, +PSI0_MAX].
    """
    if not (b_lo <= beta_target <= b_hi):
        return np.nan
    lo, hi = -PSI0_MAX, PSI0_MAX
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if beta_of(J, chi, mid, psi1) > beta_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def psi1_at(J, chi, mode, cache):
    """Wing pitch amplitude for the given mode at this velocity.

    'retuned' uses the magnitude-optimal psi1 at psi0 = 0, which is symmetric
    in chi (mirror symmetry), so it is cached by |chi|.
    """
    if mode == "hover":
        return PSI1_HOVER
    key = (round(J, 6), round(abs(chi), 6))
    if key not in cache:
        cache[key] = best_psi1(J, abs(chi))[0]
    return cache[key]


def compute_maps(mode):
    nJ, nC, nT = len(J_GRID), len(CHI_FULL), len(BETA_TARGETS)
    psi0 = np.full((nT, nJ, nC), np.nan)
    cache = {}
    for i, J in enumerate(J_GRID):
        chis = [0.0] if J == 0.0 else CHI_FULL
        for j, chi in enumerate(chis):
            psi1 = psi1_at(J, chi, mode, cache)
            b_hi = beta_of(J, chi, -PSI0_MAX, psi1)
            b_lo = beta_of(J, chi, PSI0_MAX, psi1)
            for t, bt in enumerate(BETA_TARGETS):
                psi0[t, i, j] = psi0_for(J, chi, psi1, np.radians(bt),
                                         b_lo, b_hi)
        if J == 0.0:
            psi0[:, i, :] = psi0[:, i, :1]   # center is chi-degenerate
    return psi0


def main(mode):
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    psi0 = np.degrees(compute_maps(mode))

    fig, axes = plt.subplots(1, len(BETA_TARGETS), figsize=(7.6, 2.5),
                             subplot_kw=dict(projection="polar"),
                             constrained_layout=True)
    TH, R = np.meshgrid(CHI_FULL, J_GRID)
    levels = np.arange(-60.0, 61.0, 5.0)

    for ax, bt, Z in zip(axes, BETA_TARGETS, psi0):
        Zm = np.ma.masked_invalid(Z)
        cf = ax.contourf(TH, R, Zm, levels=levels, cmap=CMAP)
        unreachable = np.isnan(Z).astype(float)
        if unreachable.any():
            ax.contourf(TH, R, unreachable, levels=[0.5, 1.5], colors="none",
                        hatches=["////"])
            ax.contour(TH, R, unreachable, levels=[0.5], colors="0.3",
                       linewidths=0.6)
        setup_polar(ax, style)
        ax.set_title(rf"$\beta = {bt:.0f}^\circ$", fontsize=style.font_size)
    cb = fig.colorbar(cf, ax=axes, shrink=0.8, pad=0.015, aspect=30)
    cb.set_label(r"$\psi_0$ required (deg)", fontsize=style.font_size - 1)
    cb.set_ticks(np.arange(-60.0, 61.0, 20.0))
    cb.ax.tick_params(labelsize=style.font_size - 3)

    suffix = "" if mode == "hover" else f"_{mode}"
    out = OUT_DIR / f"psi0_for_beta_maps{suffix}.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")

    print(f"mode: {mode} (psi1 " +
          ("fixed at hover optimum)" if mode == "hover" else
           "re-optimized at each velocity)"))
    for bt, Z in zip(BETA_TARGETS, psi0):
        frac = 100.0 * np.isnan(Z[1:, :]).mean()   # exclude degenerate center
        print(f"  beta = {bt:4.0f} deg:  unreachable over {frac:.1f}% of the "
              f"disk (J > 0)")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["hover", "retuned"]
    for m in modes:
        main(m)
        plt.close("all")

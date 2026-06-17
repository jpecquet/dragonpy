"""
Map the hover-feasibility envelope over the reduced parameter set.

After section 2's reduction (drop gamma0, sigma0; collapse Lw, phi1 -> s0) six
parameters remain that set the cycle-averaged force F_a^*:

    s0          tip-excursion amplitude   (here s0 = (2/3) Lw phi1, Lw = 0.75)
    A_over_m    inverse wing loading (total; per-wing = A_over_m / N_w)
    omega_star  nondim wingbeat frequency
    psi0        mean wing pitch
    psi1        wing pitch amplitude
    delta0      wing pitch phase

This script sweeps physically meaningful 2D slices, draws F_a^* as a heatmap,
and overlays the feasibility boundary F_a^* = 1 (force = weight). The feasible side is
where the cycle-averaged force can be tilted (via gamma0) to carry the weight.

Output: light-mode figure in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    Params, STUDY_ELEMENT, STUDY_SPAN_FRAC, WEIGHT, avg_force_magnitude, replace,
)

# White at the minimum (low force) deepening to magenta (high force); a black
# F_a^* = 1 boundary reads clearly against it.
CMAP = "RdPu"
BOUNDARY_COLOR = "black"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
LW_FIXED = 0.75               # s0 is realized by sweeping phi1 at this Lw
SPAN_FRAC_REF = STUDY_SPAN_FRAC  # s0 is the displacement of the 2/3-span station
N_PHASE = 128             # phase samples for the cycle average (smooth -> coarse ok)


def grid_magnitude(base, x_attr, x_vals, y_attr, y_vals):
    """F_a^* over a 2D grid; returns Z with shape (len(y_vals), len(x_vals))."""
    Z = np.empty((len(y_vals), len(x_vals)))
    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            p = replace(base, **{x_attr: xv, y_attr: yv})
            Z[j, i] = avg_force_magnitude(p, n_phase=N_PHASE)
    return Z


def s0_axis(phi1_vals, Lw=LW_FIXED):
    # Arc-length excursion of the 2/3-span representative station, matching the
    # report definition s0 = (2/3) Lw phi1 (the sweep angle, not sin(phi1)).
    return SPAN_FRAC_REF * Lw * phi1_vals


def panel(ax, X, Y, Z, xlabel, ylabel, title, style):
    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=CMAP)
    # Feasibility boundary F_a^* = 1 (force = weight).
    if Z.min() < WEIGHT < Z.max():
        cs = ax.contour(X, Y, Z, levels=[WEIGHT], colors=[BOUNDARY_COLOR],
                        linewidths=2.0)
        ax.clabel(cs, fmt={WEIGHT: r"$F_a^* = 1$"}, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=style.font_size + 1)
    return pcm


def main():
    style = resolve_style(theme="light")
    style.font_size = 11  # match the 11pt report body (figures inserted at native size)
    apply_matplotlib_style(style)
    # Single blade element at 2/3 span (overestimates force; see feasibility.py).
    base = Params(gamma0=0.0, element_span_fracs=STUDY_ELEMENT)

    n = 60
    omega = np.linspace(8.0, 20.0, n)
    aero = np.linspace(0.2, 1.0, n)        # total inverse wing loading A*/m*
    phi1 = np.radians(np.linspace(2.0, 60.0, n))
    s0 = s0_axis(phi1)
    psi0 = np.radians(np.linspace(-90.0, 90.0, n))
    psi1 = np.radians(np.linspace(0.0, 90.0, n))
    delta0 = np.radians(np.linspace(-180.0, 180.0, n))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) tip excursion s0 vs frequency -- the two dominant force drivers.
    Z = grid_magnitude(base, "omega_star", omega, "phi1", phi1)
    pcm = panel(axes[0, 0], *np.meshgrid(omega, s0), Z,
                r"$\omega^\ast$", r"$s_0^*$",
                r"(a)  excursion vs frequency", style)
    fig.colorbar(pcm, ax=axes[0, 0], label=r"$F_a^*$")

    # (b) inverse wing loading vs frequency, in the total A*/m* of Table 1.
    Z = grid_magnitude(base, "omega_star", omega, "A_over_m", aero)
    pcm = panel(axes[0, 1], *np.meshgrid(omega, aero), Z,
                r"$\omega^\ast$", r"$A^\ast/m^\ast$",
                r"(b)  wing loading vs frequency", style)
    fig.colorbar(pcm, ax=axes[0, 1], label=r"$F_a^*$")

    # (c) pitch geometry: mean pitch vs pitch amplitude.
    Z = grid_magnitude(base, "psi0", psi0, "psi1", psi1)
    pcm = panel(axes[1, 0], *np.meshgrid(np.degrees(psi0), np.degrees(psi1)), Z,
                r"$\psi_0$ (deg)", r"$\psi_1$ (deg)",
                r"(c)  pitch geometry", style)
    fig.colorbar(pcm, ax=axes[1, 0], label=r"$F_a^*$")

    # (d) pitch timing: pitch phase vs pitch amplitude.
    Z = grid_magnitude(base, "delta0", delta0, "psi1", psi1)
    pcm = panel(axes[1, 1], *np.meshgrid(np.degrees(delta0), np.degrees(psi1)), Z,
                r"$\delta_0$ (deg)", r"$\psi_1$ (deg)",
                r"(d)  pitch timing", style)
    fig.colorbar(pcm, ax=axes[1, 1], label=r"$F_a^*$")

    fig.suptitle(r"Hover feasibility: cycle-averaged force vs weight "
                 r"($F_a^* = 1$ boundary in highlight)", y=1.00)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "feasibility_envelope.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

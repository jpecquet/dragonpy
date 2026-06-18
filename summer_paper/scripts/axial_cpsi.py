"""Axial-inflow force coefficient and force-maximizing pitch vs advance ratio.

Stroke-plane-normal (axial) flight, chi = 0: the body moves along the stroke
normal, so in the gamma0-invariant (J, chi) frame the inflow is
v_body = (0, 0, J s0* omega*). At psi0 = 0 the cycle-mean force lies along the
normal and C_F* = |F| / q* is read at the reference wingbeat. We show
  (a) C_F*(J) at the fixed hover-optimal pitch (the force falling as the inflow
      washes out the angle of attack) and at the pitch re-optimized for maximum
      lift-branch force (partly recovering it), and
  (b) that force-maximizing amplitude psi1_max(J).

Figure for report1 sec:spn-force (fig:axial_cpsi).
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from velocity_cpsi_maps import C_of, PSI1_HOVER
from generalized_control import slave_psi1

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    J = np.linspace(0.0, 0.9, 19)
    C_nom = np.array([C_of(j, 0.0, PSI1_HOVER) for j in J])
    psi1_opt = np.array([slave_psi1(j, 0.0) for j in J])
    C_max = np.array([C_of(j, 0.0, p) for j, p in zip(J, psi1_opt)])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.6), constrained_layout=True)
    c1, c2 = "black", "#b2182b"

    axA.plot(J, C_nom, color=c1, lw=1.8, label=r"hover-optimal $\psi_1$")
    axA.plot(J, C_max, color=c2, lw=1.8, ls="--", label=r"re-optimized $\psi_1$")
    axA.axhline(1.0, color="0.7", lw=0.8, ls=":")
    axA.set_xlabel(r"advance ratio $J$")
    axA.set_ylabel(r"$C_{F^*}$")
    axA.set_ylim(0, None)
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="best")
    axA.set_title("(a)", fontsize=style.font_size)

    axB.plot(J, np.degrees(psi1_opt), color=c2, lw=1.8)
    axB.set_xlabel(r"advance ratio $J$")
    axB.set_ylabel(r"force-maximizing $\psi_1^{\max}$ (deg)")
    axB.set_ylim(0, None)
    axB.set_title("(b)", fontsize=style.font_size)

    out = OUT_DIR / "axial_cpsi.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    i4 = int(np.argmin(np.abs(J - 0.4)))
    print(f"C_F* hover-opt: {C_nom[0]:.2f} (hover) -> {C_nom[i4]:.2f} (J=0.4); "
          f"re-opt {C_max[i4]:.2f} (J=0.4); "
          f"psi1_max {np.degrees(psi1_opt[0]):.0f} -> {np.degrees(psi1_opt[-1]):.0f} deg")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

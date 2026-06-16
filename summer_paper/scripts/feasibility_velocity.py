"""
Does the F_a^* = q^* C_psi separation survive nonzero body velocity?

The hover result (feasibility_reduction.py) factors the cycle-averaged force into
an amplitude group q^* = (Aw/mb) s0^2 omega^2 times a pitch coefficient C_psi.
Here we add a body velocity and ask whether that factorization holds in forward /
maneuvering flight.

Framing (chosen so the result is gamma0-invariant -- verified numerically):
the body velocity is set relative to the stroke-plane normal n. Its magnitude is
the advance ratio J = U / (s0 omega), i.e. body speed over mean wing-tip speed,
and its direction is the angle chi from n:

    v_body = U (sin chi, 0, cos chi),   U = J s0 omega,   at gamma0 = 0 (n = +z).

chi = 0 is head-on (wind perpendicular to the stroke plane), chi = 90 deg is
grazing (wind in the stroke plane). Tying U to the wing speed (via J) keeps every
velocity in the problem proportional to s0 omega, so the q^* magnitude scaling is
preserved and only the dimensionless C can change with (J, chi).

Two panels:
  (a) C = F_a^*/q^* at the reference amplitude and nominal pitch, vs J, one curve
      per chi -- how forward speed and direction reshape the force coefficient.
  (b) separation residual: CV of F_a^*/q^* over random amplitude (Aw, omega, phi1,
      Lw) at fixed pitch, vs J -- where the factorization still holds.

Output: light-mode figure in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, N_WINGS, avg_force_magnitude,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
N_PHASE = 128
SPAN_FRAC_REF = STUDY_SPAN_FRAC

# Reference amplitude (matches feasibility_reduction): s0 = 0.25 at Lw = 0.75.
# AW_REF is the per-wing inverse wing loading (= total REF_A_OVER_M / N_WINGS);
# kept per-wing because q_star below is written in per-wing A_w*/m*.
AW_REF, OMEGA_REF, LW_REF, S0_REF = REF_A_OVER_M / N_WINGS, REF_OMEGA_STAR, REF_LW, REF_S0
PHI1_REF = S0_REF / (SPAN_FRAC_REF * LW_REF)            # = 0.5 rad
# Nominal (hover) pitch.
PITCH_NOM = dict(psi0=0.0, psi1=np.radians(45.0), delta0=np.pi / 2)

# Ranges for the amplitude-only random draw of panel (b).
AW_RANGE = (0.05, 0.25)
OMEGA_RANGE = (8.0, 20.0)
PHI1_RANGE = np.radians((2.0, 60.0))
LW_RANGE = (0.65, 0.85)

CHIS = [0, 45, 90, 135, 180]   # degrees from the stroke-plane normal
LINE_CMAP = LinearSegmentedColormap.from_list(
    "inferno_trim", plt.cm.inferno(np.linspace(0.12, 0.82, 256)))


def s0_of(Lw, phi1):
    return SPAN_FRAC_REF * Lw * phi1


def q_star(Aw, Lw, phi1, omega):
    return Aw * s0_of(Lw, phi1) ** 2 * omega ** 2


def v_body_for(J, chi, s0, omega):
    """Body velocity at advance ratio J and angle chi from the normal (gamma0=0)."""
    U = J * s0 * omega
    return (U * np.sin(chi), 0.0, U * np.cos(chi))


def C_ref(J, chi, **pitch):
    """C = F_a^*/q^* at the reference amplitude and given pitch."""
    s0 = s0_of(LW_REF, PHI1_REF)
    vb = v_body_for(J, chi, s0, OMEGA_REF)
    p = Params(A_over_m=N_WINGS * AW_REF, omega_star=OMEGA_REF, phi1=PHI1_REF, Lw=LW_REF,
               gamma0=0.0, v_body=vb, element_span_fracs=STUDY_ELEMENT, **pitch)
    return avg_force_magnitude(p, N_PHASE) / q_star(AW_REF, LW_REF, PHI1_REF, OMEGA_REF)


def collapse_cv(J, chi, rng, n=200, **pitch):
    """CV of F_a^*/q^* over random amplitude (Aw, omega, phi1, Lw) at fixed pitch.

    At fixed (J, chi, pitch) a perfect q^* factorization would give CV = 0; the
    residual measures how much the body velocity breaks the amplitude scaling.
    """
    Aw = rng.uniform(*AW_RANGE, n)
    om = rng.uniform(*OMEGA_RANGE, n)
    ph = rng.uniform(*PHI1_RANGE, n)
    lw = rng.uniform(*LW_RANGE, n)
    ratio = np.empty(n)
    for k in range(n):
        s0 = s0_of(lw[k], ph[k])
        vb = v_body_for(J, chi, s0, om[k])
        p = Params(A_over_m=N_WINGS * Aw[k], omega_star=om[k], phi1=ph[k], Lw=lw[k],
                   gamma0=0.0, v_body=vb, element_span_fracs=STUDY_ELEMENT, **pitch)
        ratio[k] = avg_force_magnitude(p, N_PHASE) / q_star(Aw[k], lw[k], ph[k], om[k])
    return float(ratio.std() / ratio.mean())


def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)

    colors = LINE_CMAP(np.linspace(0.0, 1.0, len(CHIS)))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 2.95),
                                   constrained_layout=True)

    # (a) C vs J at nominal pitch, one curve per chi.
    J_fine = np.linspace(0.0, 1.0, 41)
    for chi_deg, col in zip(CHIS, colors):
        chi = np.radians(chi_deg)
        C = np.array([C_ref(J, chi, **PITCH_NOM) for J in J_fine])
        axA.plot(J_fine, C, color=col, lw=1.8, label=rf"${chi_deg}^\circ$")
    axA.axhline(C_ref(0.0, 0.0, **PITCH_NOM), color="0.6", lw=0.8, ls=":")
    axA.set_xlabel(r"advance ratio $J = U^\ast/(s_0^\ast \omega^\ast)$")
    axA.set_ylabel(r"$C = F_a^\ast/q^\ast$")
    axA.set_xlim(0, 1.0)
    axA.set_ylim(0, None)
    axA.set_title("(a)", fontsize=style.font_size)
    axA.legend(title=r"$\chi$", fontsize=style.font_size - 3,
               title_fontsize=style.font_size - 3, frameon=True, ncol=2)

    # (b) separation residual CV vs J, one curve per chi.
    J_coarse = np.linspace(0.0, 1.0, 11)
    for chi_deg, col in zip(CHIS, colors):
        chi = np.radians(chi_deg)
        cv = np.array([collapse_cv(J, chi, rng, **PITCH_NOM) for J in J_coarse])
        axB.plot(J_coarse, 100.0 * cv, color=col, lw=1.8, marker="o", ms=3)
    axB.set_xlabel(r"advance ratio $J$")
    axB.set_ylabel(r"separation residual CV (\%)")
    axB.set_xlim(0, 1.0)
    axB.set_ylim(0, None)
    axB.set_title("(b)", fontsize=style.font_size)

    out = OUT_DIR / "velocity_separation.light.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")

    # --- report numbers ---
    print("C = F_a*/q* at nominal pitch (reference amplitude):")
    for chi_deg in CHIS:
        chi = np.radians(chi_deg)
        print(f"  chi={chi_deg:3d}:  J=0 -> {C_ref(0.0, chi, **PITCH_NOM):.3f},  "
              f"J=0.3 -> {C_ref(0.3, chi, **PITCH_NOM):.3f},  "
              f"J=0.6 -> {C_ref(0.6, chi, **PITCH_NOM):.3f}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

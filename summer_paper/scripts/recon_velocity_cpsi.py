"""Reconnaissance for the maneuvering-flight C_psi study (not a report figure).

Questions, each of which changes what is worth plotting in the report:
  1. Is C(J, chi) symmetric under chi -> -chi?  (halves the velocity disk)
  2. Does the (psi1, delta0) optimum move with (J, chi), and by how much?
  3. Does beta(psi0) stay a one-variable law at velocity, or does it pick up
     a strong (J, chi) dependence?
  4. Is C(J, chi) low-order in the velocity components (u, w)/(s0 omega)?
     (if yes, the whole 2-D map compresses to a few fit coefficients)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, N_WINGS, cycle_averaged_force,
)

N_PHASE = 128
PHI1_REF = REF_S0 / (STUDY_SPAN_FRAC * REF_LW)
S0_REF = REF_S0
PITCH_NOM = dict(psi0=0.0, psi1=np.radians(45.0), delta0=np.pi / 2)
Q_REF = (REF_A_OVER_M / N_WINGS) * (S0_REF * REF_OMEGA_STAR) ** 2


def force_vec(J, chi, psi0, psi1, delta0):
    U = J * S0_REF * REF_OMEGA_STAR
    vb = (U * np.sin(chi), 0.0, U * np.cos(chi))
    p = Params(A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR,
               phi1=PHI1_REF, Lw=REF_LW, gamma0=0.0, v_body=vb,
               element_span_fracs=STUDY_ELEMENT,
               psi0=psi0, psi1=psi1, delta0=delta0)
    return cycle_averaged_force(p, N_PHASE)


def C_of(J, chi, **pitch):
    return np.linalg.norm(force_vec(J, chi, **pitch)) / Q_REF


def beta_of(J, chi, **pitch):
    f = force_vec(J, chi, **pitch)
    return np.degrees(np.arctan2(f[0], f[2]))  # angle from stroke normal (+z)


print("=== 1. chi -> -chi symmetry at nominal pitch ===")
for J in (0.3, 0.6):
    for chi_deg in (30, 60, 90, 120, 150):
        cp = C_of(J, np.radians(chi_deg), **PITCH_NOM)
        cm = C_of(J, np.radians(-chi_deg), **PITCH_NOM)
        print(f"  J={J}, chi=+-{chi_deg:3d}:  C+={cp:.4f}  C-={cm:.4f}  "
              f"rel diff={abs(cp-cm)/cp:.2e}")

print("\n=== 2. (psi1, delta0) optimum vs (J, chi), psi0=0 ===")
psi1_grid = np.radians(np.linspace(20.0, 70.0, 26))
d0_grid = np.radians(np.linspace(50.0, 130.0, 33))
for J in (0.0, 0.3, 0.6):
    for chi_deg in (0, 90, 180):
        if J == 0.0 and chi_deg != 0:
            continue
        chi = np.radians(chi_deg)
        Cmap = np.array([[C_of(J, chi, psi0=0.0, psi1=p1, delta0=d0)
                          for d0 in d0_grid] for p1 in psi1_grid])
        i, j = np.unravel_index(np.argmax(Cmap), Cmap.shape)
        print(f"  J={J}, chi={chi_deg:3d}:  psi1*={np.degrees(psi1_grid[i]):5.1f} deg, "
              f"delta0*={np.degrees(d0_grid[j]):5.1f} deg,  Cmax={Cmap[i, j]:.3f}")

print("\n=== 3. beta(psi0) at velocity (nominal psi1, delta0) ===")
psi0s = np.radians([-20.0, 0.0, 20.0])
hdr = "  psi0:      " + "".join(f"{np.degrees(p):>9.0f}" for p in psi0s)
print(hdr + "   (beta in deg, angle of mean force from stroke normal)")
for J in (0.0, 0.3, 0.6):
    for chi_deg in (0, 90, 180):
        if J == 0.0 and chi_deg != 0:
            continue
        chi = np.radians(chi_deg)
        betas = [beta_of(J, chi, psi0=p, psi1=PITCH_NOM["psi1"],
                         delta0=PITCH_NOM["delta0"]) for p in psi0s]
        print(f"  J={J}, chi={chi_deg:3d}: " +
              "".join(f"{b:9.1f}" for b in betas))

print("\n=== 4. polynomial order of C in velocity components (u, w) ===")
rng = np.random.default_rng(3)
Js = rng.uniform(0.0, 0.8, 150)
chis = rng.uniform(-np.pi, np.pi, 150)
u, w = Js * np.sin(chis), Js * np.cos(chis)
C = np.array([C_of(J, chi, **PITCH_NOM) for J, chi in zip(Js, chis)])
for deg in (2, 3, 4):
    cols = [u**a * w**b for a in range(deg + 1) for b in range(deg + 1 - a)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, C, rcond=None)
    res = C - A @ coef
    print(f"  degree {deg}: rms residual = {res.std():.4f} "
          f"({100 * res.std() / C.mean():.2f}% of mean C)")

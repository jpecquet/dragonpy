"""
Power efficiency of the hover wingbeat (report1, stationary-flight section).

C_psi measures how much cycle-averaged FORCE the wingbeat makes per unit dynamic
pressure -- it is an effective lift coefficient. It is *not* an energy efficiency:
a wing that aims its force off the stroke-plane normal (nonzero psi0, needed to
hover with an inclined stroke plane) loses C_psi, yet may be no more costly to run.
This script adds the missing energetic measure, following the two Wang papers in
the repo root.

Aerodynamic power, no wing inertia
----------------------------------
With massless wings and the quasi-steady force law, the only mechanical cost is
the work each element does against aerodynamic drag (lift is perpendicular to the
air-relative velocity, so it does no work; the pitch-axis torque term is dropped
with the wing inertia, consistent with Wang 2008 and the negative-pitch-power
result of Bergou 2007). For one element the instantaneous power is

    P_elem = |F_D| * V = (1/2) rho A_w V^2 C_D(alpha) * V = (1/2) rho A_w V^3 C_D,

with V the in-plane air-relative speed already used in the force law. Summed over
wings/elements and cycle-averaged, in the repo's units (mass = g = L = 1, so the
weight is 1 and the velocity scale is sqrt(gL) = 1) this is

    Pbar* = < sum (1/2)(A_w*/m*) V^3 C_D >        [units of weight * sqrt(gL)].

Cost of endurance (Wang 2008, eq. 1)
------------------------------------
To turn power into a dimensionless *efficiency* we normalize by the power needed
to support the weight at the reference speed U_ref = sqrt(2 m g / rho A), exactly
as Wang 2008. In our nondimensional groups U_ref* = sqrt(2 / (A*/m*)), so

    P* = Pbar* / U_ref* = Pbar* * sqrt( (A*/m*) / 2 ).

This is the same quantity that for a steadily translating wing supporting its
weight by lift reduces to the familiar P* = C_D / C_L^(3/2) (verified in main()).
Imposing the hover constraint F_a* = q* C_psi = 1 (so q* = 1/C_psi) collapses the
amplitude scale and leaves a pure function of the kinematics,

    P* = 2 sqrt(2) * < |cos(phase)|^3 C_D(alpha) > / C_psi^(3/2),

the flapping analog of C_D / C_L^(3/2): the drag-power shape factor in the
numerator plays the role of C_D, and C_psi plays the role of C_L. This closed
form is checked against the direct numerical integral in main().

Main result
-----------
Sweeping the stroke-plane angle gamma0, with the wingbeat re-trimmed to hover at
each gamma0 (psi0 aims the force vertical, phi1 sets |F| = 1, pitch held at the
C_psi optimum psi1 = 51 deg, delta0 = 90 deg), C_psi falls by ~25% from gamma0 =
0 to 40 deg while P* stays essentially flat (in fact improves slightly): the
inclined stroke spends on drag what it loses on the force coefficient, so hover
is no less efficient with an inclined stroke plane. This is the quasi-steady
counterpart of Wang 2004 (specific power roughly independent of stroke-plane
angle up to ~60 deg) and of Wang 2008 (near-optimal flapping cost).

Output: power_efficiency.light.png in summer_paper/figures/.
Runs on the project env (numpy + matplotlib only).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from feasibility import (
    Params, REF_A_OVER_M, REF_LW, REF_OMEGA_STAR, REF_S0, STUDY_ELEMENT,
    STUDY_SPAN_FRAC, N_WINGS, R_HINGE_RIGHT, R_HINGE_LEFT, _CHIRALITY,
    _rot_x, _rot_z, drag_coeff, lift_coeff, avg_force_magnitude,
    cycle_averaged_force, replace,
)
from feasibility_reduction import q_star, cpsi_of_pitch
from hover_drift import base_params, solve_equilibrium, _bisect

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

OUT_DIR = HERE.parent / "figures"
N_PHASE = 256
HINGES = (R_HINGE_RIGHT, R_HINGE_LEFT, R_HINGE_RIGHT, R_HINGE_LEFT)


# ---------------------------------------------------------------------------
# Aerodynamic power. Mirrors feasibility._wing_force_vec exactly through the
# velocity/angle-of-attack computation, then forms the drag power instead of the
# force vector. Translating mode is not needed here and is omitted.

def _wing_power_vec(p, chirality, H, theta, omega):
    """Cycle of total aerodynamic power for one wing over phases `theta` (T,).

    Power = sum_elements |F_D| * V = sum (1/2)(A_w*/m*)/N * V^3 * C_D(alpha), with
    V the in-plane air-relative speed (same V^2 the force law uses). Lift does no
    work (perpendicular to V), so only C_D enters. Returns shape (T,).
    """
    if p.element_span_fracs is not None:
        fracs = np.asarray(p.element_span_fracs, dtype=float)
    else:
        fracs = (np.arange(p.n_elements) + 0.5) / p.n_elements
    N = len(fracs)

    sweep = chirality * (p.phi1 * np.sin(theta))
    sweep_d = chirality * (p.phi1 * omega * np.cos(theta))
    if p.pitch_profile == "square":
        feather = chirality * (p.psi0 + np.pi / 2 + p.psi1 * np.sign(np.sin(theta - p.delta0)))
        feather_d = np.zeros_like(theta)
    elif p.pitch_profile == "sinusoid":
        feather = chirality * (p.psi0 + np.pi / 2 + p.psi1 * np.sin(theta - p.delta0))
        feather_d = chirality * (p.psi1 * omega * np.cos(theta - p.delta0))
    else:
        raise ValueError(f"unknown pitch_profile {p.pitch_profile!r}")
    tilt = chirality * p.gamma0

    R_tilt = _rot_x(np.full_like(theta, tilt))
    R_bw = H @ (R_tilt @ (_rot_z(sweep) @ _rot_x(feather)))

    cp, sp = np.cos(sweep), np.sin(sweep)
    omega_zyx = np.stack([cp * feather_d, sp * feather_d, sweep_d], -1)
    omega_wh = np.einsum("ij,tj->ti", R_tilt[0], omega_zyx)
    omega_rel_body = np.einsum("ij,tj->ti", H, omega_wh)
    r = fracs * p.Lw
    wing_x_body = R_bw[:, :, 0]
    r_from_hinge = r[None, :, None] * wing_x_body[:, None, :]
    v_elem = np.cross(omega_rel_body[:, None, :], r_from_hinge)
    v_elem = v_elem + np.asarray(p.v_body, dtype=float)[None, None, :]

    v_rel_wing = np.einsum("tnk,tkj->tnj", v_elem, R_bw)
    vy_le = chirality * v_rel_wing[:, :, 1]
    vz = v_rel_wing[:, :, 2]
    V2 = vy_le * vy_le + vz * vz
    alpha = np.arctan2(-vz, vy_le)
    Cd = drag_coeff(alpha)

    power = 0.5 * p.aw_over_m / N * V2 ** 1.5 * Cd        # (T, N)
    return power.sum(axis=1)                              # (T,)


def cycle_averaged_power(p, n_phase=N_PHASE):
    """Cycle-averaged aerodynamic power Pbar* (units of weight * sqrt(gL))."""
    omega = 2.0 * np.pi * p.wing_frequency
    phases = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    offsets = (0.0, 0.0, p.sigma0, p.sigma0)
    P = 0.0
    for ch, off, H in zip(_CHIRALITY, offsets, HINGES):
        P += _wing_power_vec(p, ch, H, phases - off, omega).sum()
    return P / n_phase


def cost_of_endurance(p, n_phase=N_PHASE):
    """Wang-2008 dimensionless aerodynamic cost of endurance P* = Pbar*/U_ref*.

    U_ref* = sqrt(2/(A*/m*)) is the weight-support reference speed in repo units.
    Lower P* is more efficient; 1/P* is the efficiency.
    """
    return cycle_averaged_power(p, n_phase) * np.sqrt(p.A_over_m / 2.0)


# ---------------------------------------------------------------------------
# Reference: steady translating wing supporting its weight by lift. Minimizing
# P* = C_D/C_L^(3/2) over angle of attack is the classical efficient steady
# motion (Wang 2008); its minimum is the natural yardstick for the flapping cost.

def steady_cost_min():
    a = np.linspace(np.radians(1.0), np.radians(89.0), 4000)
    cl, cd = lift_coeff(a), drag_coeff(a)
    P = cd / cl ** 1.5
    k = int(np.argmin(P))
    return P[k], a[k]


def _reference_alpha(p, theta):
    """Angle of attack of the right-fore 2/3-span element over phases `theta`.

    Honors p.pitch_profile (sinusoid or square). The feather rotation is about
    the span axis, so its rate adds no velocity to a span-station element and only
    the pitch ANGLE (not its rate) affects alpha -- hence the square wave differs
    from the sinusoid only through the chord orientation."""
    omega = 2.0 * np.pi * p.wing_frequency
    ch, H = 1, R_HINGE_RIGHT
    sweep = ch * (p.phi1 * np.sin(theta))
    sweep_d = ch * (p.phi1 * omega * np.cos(theta))
    if p.pitch_profile == "square":
        feather = ch * (p.psi0 + np.pi / 2 + p.psi1 * np.sign(np.sin(theta - p.delta0)))
    else:
        feather = ch * (p.psi0 + np.pi / 2 + p.psi1 * np.sin(theta - p.delta0))
    R_tilt = _rot_x(np.full_like(theta, ch * p.gamma0))
    R_bw = H @ (R_tilt @ (_rot_z(sweep) @ _rot_x(feather)))
    omega_zyx = np.stack([np.zeros_like(theta), np.zeros_like(theta), sweep_d], -1)
    omega_rel_body = np.einsum("ij,tj->ti", H, np.einsum("ij,tj->ti", R_tilt[0], omega_zyx))
    v_elem = np.cross(omega_rel_body, (STUDY_SPAN_FRAC * p.Lw) * R_bw[:, :, 0])
    v_rel_wing = np.einsum("tk,tkj->tj", v_elem, R_bw)
    return np.arctan2(-v_rel_wing[:, 2], ch * v_rel_wing[:, 1])


def closed_form_cost(p, n_phase=N_PHASE):
    """P* = 2 sqrt(2) <|cos phase|^3 C_D> / C_psi^(3/2) (single-element hover).

    Independent re-derivation used to corroborate the direct integral; valid for
    both pitch profiles.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    alpha = _reference_alpha(p, theta)
    gP = np.mean(np.abs(np.cos(theta)) ** 3 * drag_coeff(alpha))
    cpsi = avg_force_magnitude(p, n_phase) / q_star(p)
    return 2.0 * np.sqrt(2.0) * gP / cpsi ** 1.5


def halfstroke_split(p, n_phase=4000):
    """Per-half-stroke (mean |alpha|, share of cycle drag-power), heavier half first.

    Half-strokes split at the sweep reversal (sign of cos theta, where the square
    wave flips). Returns ((|alpha|_power, powshare_power), (|alpha|_return, ...)),
    the 'power' stroke being the one with the larger mean angle of attack."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    alpha = _reference_alpha(p, theta)
    powden = np.abs(np.cos(theta)) ** 3 * drag_coeff(alpha)
    A = np.cos(theta) >= 0.0
    halves = [(np.degrees(np.abs(alpha[m]).mean()), powden[m].sum() / powden.sum())
              for m in (A, ~A)]
    return tuple(sorted(halves, reverse=True))


PHI1_REF = REF_S0 / (STUDY_SPAN_FRAC * REF_LW)   # reference amplitude (C_psi read-off)


def drag_weight_fraction(p, n_phase=720):
    """Fraction of the cycle-averaged vertical force carried by drag (vs lift).

    The quasi-steady analog of Wang 2004's "drag supports weight" result: at a
    horizontal stroke the two half-strokes' drag cancels vertically (fraction 0);
    tilting the stroke gives drag a net upward component. Computed on the right
    fore wing (the four wings contribute identically to F_z by symmetry)."""
    omega = 2.0 * np.pi * p.wing_frequency
    ch, H = 1, R_HINGE_RIGHT
    th = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    sweep = ch * (p.phi1 * np.sin(th))
    sweep_d = ch * (p.phi1 * omega * np.cos(th))
    feather = ch * (p.psi0 + np.pi / 2 + p.psi1 * np.sin(th - p.delta0))
    feather_d = ch * (p.psi1 * omega * np.cos(th - p.delta0))
    R_tilt = _rot_x(np.full_like(th, ch * p.gamma0))
    R_bw = H @ (R_tilt @ (_rot_z(sweep) @ _rot_x(feather)))
    oz = np.stack([np.cos(sweep) * feather_d, np.sin(sweep) * feather_d, sweep_d], -1)
    orb = np.einsum("ij,tj->ti", H, np.einsum("ij,tj->ti", R_tilt[0], oz))
    v = np.cross(orb, (STUDY_SPAN_FRAC * p.Lw) * R_bw[:, :, 0])
    vw = np.einsum("tk,tkj->tj", v, R_bw)
    vy, vz = ch * vw[:, 1], vw[:, 2]
    V2 = vy * vy + vz * vz
    alpha = np.arctan2(-vz, vy)
    sc = 0.5 * p.aw_over_m * V2
    safe = np.sqrt(np.where(V2 > 1e-12, V2, 1.0))
    dle, dz = -vy / safe, -vz / safe
    lle, lz = dz, -dle
    Fl = np.stack([np.zeros_like(dle), ch * sc * lift_coeff(alpha) * lle, sc * lift_coeff(alpha) * lz], -1)
    Fd = np.stack([np.zeros_like(dle), ch * sc * drag_coeff(alpha) * dle, sc * drag_coeff(alpha) * dz], -1)
    Lz = np.einsum("tk,tjk->tj", Fl, R_bw)[:, 2].mean()
    Dz = np.einsum("tk,tjk->tj", Fd, R_bw)[:, 2].mean()
    return Dz / (Lz + Dz)


def solve_psi0_vertical(p, n_phase=128):
    """Mean pitch psi0 that aims the cycle-averaged force vertical (F_x = 0).

    The direction-only half of the hover trim. P* (closed form) and C_psi are
    amplitude-independent, so phi1 is held at the reference value and only the
    direction constraint is enforced -- much cheaper than the full trim, and the
    psi0 root is insensitive to phi1 (beta depends on psi0, not amplitude)."""
    return _bisect(
        lambda v: float(cycle_averaged_force(replace(p, psi0=v), n_phase)[0]),
        np.radians(-70.0), np.radians(20.0), target=0.0, increasing=False, iters=50)


def power_optimal_hover(gamma0, n_phase=128, profile="sinusoid"):
    """Minimum-power hover at fixed stroke-plane angle gamma0.

    Free pitch DOF are (psi1, delta0); psi0 is pinned by the vertical-force
    constraint and phi1 by |F| = 1 (the latter cancels from P*). Returns
    (P*_min, C_psi, psi0, psi1, delta0). Coarse grid then a psi1 refine."""
    def cost(psi1, delta0):
        p = replace(Params(element_span_fracs=STUDY_ELEMENT), gamma0=gamma0,
                    psi1=psi1, delta0=delta0, phi1=PHI1_REF,
                    A_over_m=REF_A_OVER_M, omega_star=REF_OMEGA_STAR, Lw=REF_LW,
                    pitch_profile=profile)
        p = replace(p, psi0=solve_psi0_vertical(p, n_phase))
        return closed_form_cost(p, n_phase), p

    # psi1 is bounded by the Table-1 morphological range [0, 60] deg; the
    # unconstrained power optimum sits even higher (~75 deg), so the constrained
    # optimum rides the upper bound -- noted in the report. A square wave flips at
    # the stroke reversal (delta0 = 90); off-90 flips mid-stroke and is
    # unphysical, so only the sinusoid searches delta0.
    PSI1_MAX = np.radians(60.0)
    delta_grid = np.radians([90.0]) if profile == "square" else np.radians([75.0, 90.0, 105.0])
    psi1_grid = np.radians(np.linspace(8.0, 60.0, 14))
    best = (np.inf, None)
    for d in delta_grid:
        for ps1 in psi1_grid:
            c, p = cost(ps1, d)
            if c < best[0]:
                best = (c, p, ps1, d)
    # refine psi1 around the grid optimum at the winning delta0
    _, p, ps1_opt, d_opt = best
    lo, hi = ps1_opt - np.radians(4.0), min(ps1_opt + np.radians(4.0), PSI1_MAX)
    for ps1 in np.linspace(max(lo, np.radians(2.0)), hi, 9):
        c, pp = cost(ps1, d_opt)
        if c < best[0]:
            best = (c, pp, ps1, d_opt)
    c, p, ps1_opt, d_opt = best
    cpsi = avg_force_magnitude(p, n_phase) / q_star(p)
    return c, cpsi, p.psi0, ps1_opt, d_opt


# ---------------------------------------------------------------------------

def main():
    style = resolve_style(theme="light")
    style.font_size = 11
    apply_matplotlib_style(style)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    P_steady, a_steady = steady_cost_min()

    # --- consistency check: direct integral vs closed form, at the reference hover ---
    p_chk = base_params(gamma0=0.0)
    psi0, phi1 = solve_equilibrium(p_chk, mass_ratio=0.0)
    p_chk = replace(p_chk, psi0=psi0, phi1=phi1)
    P_direct = cost_of_endurance(p_chk)
    P_closed = closed_form_cost(p_chk)
    print(f"steady-flight minimum cost  P* = C_D/C_L^(3/2) = {P_steady:.4f} "
          f"at alpha = {np.degrees(a_steady):.1f} deg")
    print(f"check @ gamma0=0 hover: P*_direct={P_direct:.4f}  "
          f"P*_closed={P_closed:.4f}  (rel diff {abs(P_direct/P_closed-1)*100:.2f}%)")
    # same check for the square profile (direct integral honors pitch_profile)
    p_sq = replace(p_chk, pitch_profile="square")
    psi0_sq, phi1_sq = solve_equilibrium(p_sq, mass_ratio=0.0)
    p_sq = replace(p_sq, psi0=psi0_sq, phi1=phi1_sq)
    print(f"check @ gamma0=0 hover (square): P*_direct={cost_of_endurance(p_sq):.4f}  "
          f"P*_closed={closed_form_cost(p_sq):.4f}")

    # --- sweep stroke-plane angle ---
    # Two hover families at each gamma0, both re-trimmed (psi0 vertical, |F|=1):
    #   force-optimal  -- pitch pinned at the C_psi optimum (psi1=51, delta0=90),
    #                     the strategy the rest of the report uses;
    #   power-optimal  -- pitch (psi1, delta0) chosen to minimize P* in range.
    gammas = np.radians(np.linspace(0.0, 60.0, 13))
    gdeg = np.degrees(gammas)
    Pforce = np.empty_like(gammas)     # P* of the C_psi-optimal (force-optimal) pitch
    cpsi = np.empty_like(gammas)
    dragf = np.empty_like(gammas)
    psi0s = np.empty_like(gammas)
    Ppow = np.empty_like(gammas)       # P* of the power-optimal sinusoid pitch
    psi1pow = np.empty_like(gammas)
    Ppow_sq = np.empty_like(gammas)    # P* of the power-optimal square pitch
    for i, g in enumerate(gammas):
        p = base_params(gamma0=float(g))
        psi0, phi1 = solve_equilibrium(p, mass_ratio=0.0)
        p = replace(p, psi0=psi0, phi1=phi1)
        Pforce[i] = cost_of_endurance(p)
        cpsi[i] = cpsi_of_pitch(psi0, p.psi1, p.delta0)
        dragf[i] = drag_weight_fraction(p)
        psi0s[i] = psi0
        Ppow[i], _c, _p0, ps1, _d = power_optimal_hover(float(g))
        psi1pow[i] = ps1
        Ppow_sq[i] = power_optimal_hover(float(g), profile="square")[0]

    # --- figure: (a) cost of endurance vs gamma0, (b) drag share of weight ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 2.9), constrained_layout=True)

    axA.axhline(P_steady, color="0.6", lw=1.0, ls=(0, (4, 3)))
    axA.text(1.5, P_steady, r"steady $C_D/C_L^{3/2}$", va="bottom", ha="left",
             fontsize=style.font_size - 2, color="0.4")
    axA.plot(gdeg, Pforce, color="black", lw=1.8, ls=":",
             label=r"force-optimal pitch ($C_\psi^{\max}$)")
    axA.plot(gdeg, Ppow, color="black", lw=1.8,
             label=r"power-optimal, sinusoid")
    axA.plot(gdeg, Ppow_sq, color="black", lw=1.6, ls=(0, (3, 1, 1, 1)),
             label=r"power-optimal, square")
    axA.set_xlabel(r"$\gamma_0$ (deg)")
    axA.set_ylabel(r"cost of endurance $P^\ast$")
    axA.set_xlim(0, 60)
    axA.set_ylim(0, None)
    axA.legend(fontsize=style.font_size - 3, frameon=True, loc="upper left")
    axA.set_title("(a)", fontsize=style.font_size)

    axB.plot(gdeg, 100.0 * dragf, color="black", lw=1.8)
    axB.plot(63.0, 76.0, "o", color="black", ms=5, mfc="white", mew=1.0)
    axB.annotate("Wang 2004\n(76% at $63^\\circ$)", xy=(63.0, 76.0),
                 xytext=(33.0, 82.0), fontsize=style.font_size - 3,
                 va="center", arrowprops=dict(arrowstyle="-", lw=0.6))
    axB.set_xlabel(r"$\gamma_0$ (deg)")
    axB.set_ylabel(r"drag share of vertical force (\%)")
    axB.set_xlim(0, 65)
    axB.set_ylim(0, 90)
    axB.set_title("(b)", fontsize=style.font_size)

    out = OUT_DIR / "power_efficiency.light.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")

    # --- report numbers ---
    print("\ngamma0 psi0   C_psi  dragF%  P*_force 1/P*   P*_pow  psi1_pow  P*f/steady")
    for g, p0, c, df, Pf, Pp, ps1 in zip(gdeg, psi0s, cpsi, dragf, Pforce, Ppow, psi1pow):
        if round(g) % 10 == 0:
            print(f"{g:5.0f} {np.degrees(p0):5.1f}  {c:5.3f}  {100*df:5.1f}  "
                  f"{Pf:7.3f} {1/Pf:6.3f}  {Pp:6.3f}  {np.degrees(ps1):6.1f}   {Pf/P_steady:6.2f}")
    i40 = int(np.argmin(np.abs(gdeg - 40.0)))
    print(f"\nforce-optimal hover, gamma0 0 -> 40 deg:  "
          f"C_psi {cpsi[0]:.3f} -> {cpsi[i40]:.3f} ({(cpsi[i40]/cpsi[0]-1)*100:+.0f}%),  "
          f"P* {Pforce[0]:.3f} -> {Pforce[i40]:.3f} ({(Pforce[i40]/Pforce[0]-1)*100:+.0f}%)")
    print(f"power-optimal hover, gamma0 0 -> 40 deg:  "
          f"P* {Ppow[0]:.3f} -> {Ppow[i40]:.3f} ({(Ppow[i40]/Ppow[0]-1)*100:+.0f}%)")
    print(f"max-force vs min-power pitch at gamma0=0: "
          f"C_psi-opt psi1=51 deg gives P*={Pforce[0]:.3f}; "
          f"power-opt psi1={np.degrees(psi1pow[0]):.0f} deg gives P*={Ppow[0]:.3f} "
          f"({(Ppow[0]/Pforce[0]-1)*100:+.0f}%)")
    print(f"drag share of vertical force: {100*dragf[0]:.0f}% at gamma0=0 -> "
          f"{100*dragf[-1]:.0f}% at gamma0=60 deg (cf. Wang 2004: 76% at 63 deg)")
    print(f"flapping cost / steady-optimal: power-optimal "
          f"[{(Ppow/P_steady).min():.2f}, {(Ppow/P_steady).max():.2f}] over gamma0")

    # --- square-wave pitch: cheaper everywhere, and it feathers the return ---
    print("\nsquare-wave power-optimal pitch:")
    print(f"  P* {Ppow_sq[0]:.3f} -> {Ppow_sq[i40]:.3f} (gamma0 0 -> 40 deg, "
          f"{(Ppow_sq[i40]/Ppow_sq[0]-1)*100:+.0f}%); vs sinusoid {Ppow[0]:.3f} -> {Ppow[i40]:.3f}")
    print(f"  cheaper than sinusoid by "
          f"{(1-Ppow_sq[0]/Ppow[0])*100:.0f}% at gamma0=0, "
          f"{(1-Ppow_sq[i40]/Ppow[i40])*100:.0f}% at gamma0=40 deg; "
          f"vs steady ideal {Ppow_sq[0]/P_steady:.2f}x at gamma0=0")
    for g in (0.0, 40.0):
        _P, _c, _p0, ps1, _d = power_optimal_hover(np.radians(g), profile="square")
        psq = replace(Params(element_span_fracs=STUDY_ELEMENT), gamma0=np.radians(g),
                      psi1=ps1, delta0=np.pi / 2, phi1=PHI1_REF, A_over_m=REF_A_OVER_M,
                      omega_star=REF_OMEGA_STAR, Lw=REF_LW, pitch_profile="square")
        psq = replace(psq, psi0=solve_psi0_vertical(psq))
        (aP, sP), (aR, sR) = halfstroke_split(psq)
        print(f"  gamma0={g:.0f}: power stroke <|aoa|>={aP:.0f} deg ({100*sP:.0f}% of power), "
              f"return stroke <|aoa|>={aR:.0f} deg ({100*sR:.0f}% of power)")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

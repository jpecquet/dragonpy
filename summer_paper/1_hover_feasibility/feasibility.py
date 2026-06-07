"""
Hover-feasibility core (report1 section 2).

The feasibility question is: for which morphology/kinematics parameters does the
cycle-averaged aerodynamic force reach at least the body weight? Outside that
set hover -- and therefore any maneuvering anchored on hover -- is impossible.

The central simplification is that this average is a one-dimensional quadrature,
not a dynamics simulation. At hover the body is stationary (v_body = 0,
omega_body = 0) in still air, so every blade element's air-relative velocity is
produced purely by wing motion. The cycle-averaged force is then

    Fbar = (1 / 2pi) integral_0^2pi  sum_wings  F_wing(state(phi)) dphi,

and both ingredients already exist in dragonpy: `expand_pattern` turns
(pattern, phase) into a wing pose, and `wing_force_point_mass` turns a pose into
a force with v_body = wind = 0. Each parameter point costs microseconds.

Units follow the rest of the repo: L0 = body length, T0 = sqrt(L0/g), body
mass = 1, so g = 1 and the weight to clear is exactly 1. A frequency parameter
omega* = omega sqrt(L0/g) is the nondimensional angular wingbeat; the muscle
layer wants wing_frequency = omega* / 2pi.

Parameter naming mirrors Table 1 of the report:

    Lw          wing length ratio            L_w / L         (-> Wing.span_ratio)
    Aw_over_mb  inverse wing loading         A_w rho L / m_b (-> Wing.aero_ratio)
    omega_star  nondim wingbeat frequency    omega sqrt(L/g)
    gamma0      stroke-plane angle                            (-> stroke_plane_tilt)
    phi1        flapping amplitude                            (-> sweep_amp)
    psi0        mean wing pitch                               (-> feather_mean)
    psi1        wing pitch amplitude                          (-> feather_amp)
    delta0      wing pitch phase                              (-> feather_phase)
    sigma0      fore/hind phase shift                         (-> hind phase offset)

Runs on the project env (numpy only).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dragonpy.body.muscles import StrokePattern, expand_pattern  # noqa: E402
from dragonpy.body.wings import Wing, WingState, wing_force_point_mass  # noqa: E402


# ---------------------------------------------------------------------------
# Aerodynamic coefficients, exactly as specified in report1 section 1.

CL_0 = 1.5
CD_0 = 0.1
CD_90 = 2.0


def lift_coeff(alpha: np.ndarray) -> np.ndarray:
    return CL_0 * np.sin(2.0 * alpha)


def drag_coeff(alpha: np.ndarray) -> np.ndarray:
    return CD_0 * np.cos(alpha) ** 2 + CD_90 * np.sin(alpha) ** 2


# Hinge orientations: +x spanwise outward, +z dorsal (see wings.py docstring).
R_HINGE_RIGHT = np.array([[0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
R_HINGE_LEFT = np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])

WEIGHT = 1.0  # body mass = 1, g = 1

# Representative-station model for report1 section 2. The whole study lumps each
# wing into a SINGLE blade element at 2/3 span (SPAN_FRAC_REF). Pass it as
# element_span_fracs=STUDY_ELEMENT. This deliberately OVERESTIMATES the true
# span-integrated force: the force ~ <V^2>, whose r^2-weighted span mean is 1/3,
# while a single 2/3 station gives (2/3)^2 = 4/9 -- a ~4/3 high bias (see
# check_translating.py). We accept it because it makes q* = (Aw/mb) s0^2 omega^2
# an EXACT velocity scale -- s0 = (2/3) Lw phi1 is precisely this station's
# arc-length excursion -- which is what the F_a* = q* C_psi separation rests on.
STUDY_SPAN_FRAC = 2.0 / 3.0
STUDY_ELEMENT = (STUDY_SPAN_FRAC,)


# ---------------------------------------------------------------------------
# Parameters.

@dataclass(frozen=True)
class Params:
    """The nine parameters of Table 1 (defaults near the middle of each range)."""

    # morphology
    Lw: float = 0.75            # [0.65, 0.85]
    Aw_over_mb: float = 0.15    # [0.05, 0.25]

    # kinematics
    omega_star: float = 14.0    # [8, 20]
    gamma0: float = 0.0         # stroke-plane angle (rad)
    phi1: float = np.radians(30.0)   # flapping amplitude (rad)
    psi0: float = 0.0                # mean pitch (rad)
    psi1: float = np.radians(45.0)   # pitch amplitude (rad)
    delta0: float = np.pi / 2        # pitch phase (rad); +-pi/2 flips at reversal
    sigma0: float = np.pi            # fore/hind phase shift (rad)

    # Body velocity in the body frame (units of sqrt(gL); g = L = 1). Every
    # element's air-relative velocity gains this constant term (wind held at
    # zero), so a nonzero v_body is forward/maneuvering flight rather than hover.
    # Stored as a tuple so Params stays frozen/hashable. Defaults to rest.
    v_body: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Pitch profile over the stroke:
    #   "sinusoid" : feather = psi0 + pi/2 + psi1 sin(phi + delta0)  (default)
    #   "square"   : feather = psi0 + pi/2 + psi1 sign(sin(phi + delta0))
    #                a bang-bang flip -- constant +-psi1 on each half-stroke,
    #                pitching instantly at stroke reversal (zero pitch-rate in
    #                between, hence no feather rotational-velocity term).
    pitch_profile: str = "sinusoid"

    n_elements: int = 8

    # Optional explicit blade-element span fractions (each in [0, 1], equally
    # weighted). When set it overrides `n_elements` and the default midpoint
    # rule -- e.g. (2/3,) is a single element evaluated at 2/3 span.
    element_span_fracs: tuple[float, ...] | None = None

    # Purely translating element: the whole wing area moves at the uniform tip
    # stroke speed (no spanwise gradient, no rotational sweep/feather velocity);
    # the sinusoidal feather only sets the angle of attack. This is the
    # numerically-evaluated counterpart of the simplified closed form, with the
    # real C_L(alpha)/C_D(alpha) instead of a constant C_L,0.
    translating: bool = False

    @property
    def wing_frequency(self) -> float:
        return self.omega_star / (2.0 * np.pi)


def make_wing(p: Params, chirality: int) -> Wing:
    R = R_HINGE_RIGHT if chirality == +1 else R_HINGE_LEFT
    return Wing(
        hinge_position=np.zeros(3),     # hinge offset is irrelevant to the force-only path
        hinge_orientation=R,
        chirality=chirality,
        span_ratio=p.Lw,
        mass_ratio=0.0,
        aero_ratio=p.Aw_over_mb,
        lift_coeff=lift_coeff,
        drag_coeff=drag_coeff,
        n_elements=p.n_elements,
    )


def make_pattern(p: Params) -> StrokePattern:
    """One stroke pattern shared by all four wings (chirality handled downstream).

    Elevation is held at zero: Table 1 carries no figure-8 parameter, so the
    stroke is a pure sweep + feather oscillation in the (tilted) stroke plane.
    """
    return StrokePattern(
        stroke_plane_tilt=p.gamma0,
        sweep_amp=p.phi1, sweep_mean=0.0, sweep_phase=0.0,
        elev_amp=0.0, elev_mean=0.0, elev_phase=0.0, elev_harmonic=2,
        feather_amp=p.psi1, feather_mean=p.psi0, feather_phase=p.delta0,
    )


# Wing order: fore_R, fore_L, hind_R, hind_L.
_CHIRALITY = (+1, -1, +1, -1)


def cycle_averaged_force_scalar(p: Params, n_phase: int = 256) -> np.ndarray:
    """Reference implementation built on the trusted dragonpy functions.

    Loops over phase, expanding the pattern and summing per-wing forces. Kept as
    a correctness oracle for the vectorized path below (see `test_feasibility`).
    Only the sinusoidal pitch profile is supported here (the muscle layer's
    StrokePattern has no square-wave mode).
    """
    if p.pitch_profile != "sinusoid":
        raise NotImplementedError(
            "scalar reference only supports pitch_profile='sinusoid'")
    if p.element_span_fracs is not None:
        raise NotImplementedError(
            "scalar reference only supports the default midpoint span stations")
    if p.translating:
        raise NotImplementedError("scalar reference does not support translating mode")
    wings = [make_wing(p, c) for c in _CHIRALITY]
    pattern = make_pattern(p)
    offsets = np.array([0.0, 0.0, p.sigma0, p.sigma0])
    omega = 2.0 * np.pi * p.wing_frequency

    phases = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    v_body = np.asarray(p.v_body, dtype=float)
    zero = np.zeros(3)

    F_sum = np.zeros(3)
    for phi in phases:
        for wing, off in zip(wings, offsets):
            R_hw, w_hinge = expand_pattern(pattern, float(phi + off), omega, wing.chirality)
            ws = WingState(R_hinge_from_wing=R_hw, omega_wing_in_hinge=w_hinge)
            F_sum += wing_force_point_mass(wing, ws, v_body, zero)
    return F_sum / n_phase


# ---------------------------------------------------------------------------
# Vectorized path: the same physics with the phase loop pushed into numpy.

def _rot_x(a: np.ndarray) -> np.ndarray:
    """Batched rotation about x; `a` shape (T,) -> (T, 3, 3)."""
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack([
        np.stack([o, z, z], -1),
        np.stack([z, c, -s], -1),
        np.stack([z, s, c], -1),
    ], -2)


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack([
        np.stack([c, -s, z], -1),
        np.stack([s, c, z], -1),
        np.stack([z, z, o], -1),
    ], -2)


def _wing_force_vec(p: Params, chirality: int, H: np.ndarray,
                    theta: np.ndarray, omega: float) -> np.ndarray:
    """Cycle of forces for one wing over phases `theta` (T,), body frame -> (T,3).

    Mirrors muscles.expand_pattern + wings.wing_force_point_mass with
    v_body = wind = 0 and elevation held at zero, vectorized over phase. The
    scalar reference above is the source of truth for these expressions.
    """
    if p.element_span_fracs is not None:
        fracs = np.asarray(p.element_span_fracs, dtype=float)
    else:
        fracs = (np.arange(p.n_elements) + 0.5) / p.n_elements
    N = len(fracs)
    # Stroke angles (sweep_mean = sweep_phase = 0; elevation = 0).
    sweep_raw = p.phi1 * np.sin(theta)
    sweep_d_raw = p.phi1 * omega * np.cos(theta)
    if p.pitch_profile == "square":
        # Bang-bang pitch: piecewise constant +-psi1, instant flip at reversal,
        # so the feather rotational velocity is zero between flips.
        feather_raw = p.psi0 + np.pi / 2 + p.psi1 * np.sign(np.sin(theta + p.delta0))
        feather_d_raw = np.zeros_like(theta)
    elif p.pitch_profile == "sinusoid":
        feather_raw = p.psi0 + np.pi / 2 + p.psi1 * np.sin(theta + p.delta0)
        feather_d_raw = p.psi1 * omega * np.cos(theta + p.delta0)
    else:
        raise ValueError(f"unknown pitch_profile {p.pitch_profile!r}")

    # Chirality (sweep, feather, tilt flip on the left wing; elevation = 0).
    sweep = -chirality * sweep_raw
    sweep_d = -chirality * sweep_d_raw
    feather = chirality * feather_raw
    feather_d = chirality * feather_d_raw
    tilt = -chirality * p.gamma0

    if p.translating:
        # Planform held at stroke centre (sweep angle 0) and the feather adds no
        # rotational velocity; sweep_d is kept only to set the translation speed.
        sweep = np.zeros_like(theta)
        feather_d = np.zeros_like(theta)

    # R_hinge_from_wing = Rx(tilt) @ Rz(sweep) @ Ry(0) @ Rx(feather).
    R_zyx = _rot_z(sweep) @ _rot_x(feather)
    R_tilt = _rot_x(np.full_like(theta, tilt))
    R_hw = R_tilt @ R_zyx

    R_bw = H @ R_hw                                                      # (T,3,3)

    if p.translating:
        # Uniform translation at the tip stroke speed along the stroke tangent;
        # one lumped element carries the whole wing area (no spanwise gradient).
        wing_x_body = R_bw[:, :, 0]                                      # (T,3)
        tangent = np.cross(np.array([0.0, 0.0, 1.0]), wing_x_body)       # (T,3)
        U = p.Lw * sweep_d                                               # tip speed (T,)
        v_elem = (U[:, None] * tangent)[:, None, :]                      # (T,1,3)
        N = 1
    else:
        # omega_zyx with elevation = 0: [cos(sweep)*fd, sin(sweep)*fd, sweep_d].
        cp, sp = np.cos(sweep), np.sin(sweep)
        omega_zyx = np.stack([cp * feather_d, sp * feather_d, sweep_d], -1)  # (T,3)
        omega_wh = np.einsum("ij,tj->ti", R_tilt[0], omega_zyx)             # tilt const
        omega_rel_body = np.einsum("ij,tj->ti", H, omega_wh)               # (T,3)
        r = fracs * p.Lw                                                   # (N,)
        wing_x_body = R_bw[:, :, 0]                                         # (T,3)
        r_from_hinge = r[None, :, None] * wing_x_body[:, None, :]           # (T,N,3)
        v_elem = np.cross(omega_rel_body[:, None, :], r_from_hinge)         # (T,N,3)

    # Body translation through still air adds a constant body-frame term to every
    # element's air-relative velocity (wind = 0); zero for hover.
    v_elem = v_elem + np.asarray(p.v_body, dtype=float)[None, None, :]

    v_rel_wing = np.einsum("tnk,tkj->tnj", v_elem, R_bw)                 # body -> wing

    vy_le = chirality * v_rel_wing[:, :, 1]
    vz = v_rel_wing[:, :, 2]
    V2 = vy_le * vy_le + vz * vz
    alpha = np.arctan2(-vz, vy_le)

    Cl = lift_coeff(alpha)
    Cd = drag_coeff(alpha)

    safeV = np.where(V2 > 1e-24, np.sqrt(V2), 1.0)
    d_le, d_z = -vy_le / safeV, -vz / safeV
    l_le, l_z = d_z, -d_le

    scale = 0.5 * p.Aw_over_mb / N * V2
    F_le = scale * (Cl * l_le + Cd * d_le)
    Fz_wing = scale * (Cl * l_z + Cd * d_z)
    Fy_wing = chirality * F_le
    F_elem_wing = np.stack([np.zeros_like(F_le), Fy_wing, Fz_wing], -1)  # (T,N,3)

    F_elem_body = np.einsum("tnk,tjk->tnj", F_elem_wing, R_bw)           # wing -> body
    return F_elem_body.sum(axis=1)                                      # (T,3)


def cycle_averaged_force(p: Params, n_phase: int = 256) -> np.ndarray:
    """Cycle-averaged total aerodynamic force (body frame), in units of weight.

    Vectorized over phase. The fore pair runs at phase phi; the hind pair leads
    by sigma0. Averaging over a full period makes each wing's contribution
    independent of its own phase offset -- the sigma0-invariance the report
    claims.
    """
    omega = 2.0 * np.pi * p.wing_frequency
    phases = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    offsets = (0.0, 0.0, p.sigma0, p.sigma0)
    hinges = (R_HINGE_RIGHT, R_HINGE_LEFT, R_HINGE_RIGHT, R_HINGE_LEFT)

    F_sum = np.zeros(3)
    for chirality, off, H in zip(_CHIRALITY, offsets, hinges):
        F_sum += _wing_force_vec(p, chirality, H, phases + off, omega).sum(axis=0)
    return F_sum / n_phase


def avg_force_magnitude(p: Params, n_phase: int = 256) -> float:
    return float(np.linalg.norm(cycle_averaged_force(p, n_phase)))


def is_feasible(p: Params, n_phase: int = 256) -> bool:
    """Can the cycle-averaged force be aimed to carry the weight?

    The criterion is on magnitude alone: gamma0 rotates the whole averaged
    vector, so any |Fbar| >= weight can be oriented straight up.
    """
    return avg_force_magnitude(p, n_phase) >= WEIGHT


__all__ = [
    "Params", "cycle_averaged_force", "cycle_averaged_force_scalar",
    "avg_force_magnitude", "is_feasible",
    "make_wing", "make_pattern", "lift_coeff", "drag_coeff", "WEIGHT", "replace",
    "STUDY_SPAN_FRAC", "STUDY_ELEMENT",
]

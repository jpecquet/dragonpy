"""
Wing module: spec, pose, and quasi-steady blade-element aerodynamics.

Frame conventions (all right-handed):
  world: z up; gravity defaults to -z.
  body:  COM-attached; +x forward, +y left, +z up.
  hinge: fixed rotation off body, one per wing, always right-handed with
         +x = spanwise outward (away from body centerline) and +z = dorsal.
         The third axis is whatever right-handedness forces: on the right
         wing that puts +y toward the leading edge, on the left wing it
         puts +y toward the trailing edge. A symmetric stroke pattern must
         therefore be sign-flipped on the left wing; the muscle layer
         handles this via `Wing.chirality`, and the aero code below is
         chirality-agnostic.
  wing:  hinge rotated by ZYX intrinsic Euler triple (sweep, elevation, feather).
         Sweep is the primary flapping oscillation, about hinge +z. In the
         untilted reference this axis is dorsal, so sweep traces a horizontal
         arc (the stroke plane = body x-y). Tilting the stroke plane (a
         muscle-layer parameter) rotates this axis toward horizontal, at
         which point sweep becomes the intuitive "up-and-down flap" of
         forward flight.  Elevation is about the once-rotated +y (figure-8
         out-of-plane deviation). Feather is about the twice-rotated +x
         (wing pitch about the long axis).

Units: length L0 = body length, time T0 = sqrt(L0/g), mass m = body mass.
In these units g = 1 and body mass = 1. Frequencies are in 1/T0.

Planform: rectangular with uniform chord for v1. The chord is implicit in
`aero_ratio`; an explicit `chord_ratio` will be added when non-rectangular
planforms are introduced.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Wing:
    """Immutable geometric and aerodynamic spec of one wing."""

    hinge_position:    np.ndarray   # (3,) body frame, hinge relative to COM
    hinge_orientation: np.ndarray   # (3,3) R_body_from_hinge
    chirality:         int          # +1 right wing, -1 left wing

    span_ratio: float               # spanwise length / body length
    mass_ratio: float               # wing mass / body mass
    aero_ratio: float               # rho * L0 * S / m  (aero force scale)

    # Sectional coefficients as functions of angle of attack (radians).
    # Must accept and return numpy arrays (vectorized over blade elements).
    lift_coeff: Callable[[np.ndarray], np.ndarray]
    drag_coeff: Callable[[np.ndarray], np.ndarray]

    n_elements: int = 8


@dataclass
class WingState:
    """Mutable pose of a wing relative to its hinge.

    Stored as a rotation matrix plus angular velocity (rather than an Euler
    triple) so the muscle layer can compose arbitrary rotations — in
    particular, stroke-plane tilt pre-multiplied onto the ZYX stroke Euler —
    without being constrained to a single Euler parameterization.
    """
    R_hinge_from_wing:   np.ndarray   # (3,3)
    omega_wing_in_hinge: np.ndarray   # (3,) wing-rel-hinge angular vel, in hinge frame


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wing_wrench(
    wing: Wing,
    state: WingState,
    v_body: np.ndarray,
    omega_body: np.ndarray,
    wind_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-steady blade-element aero wrench for one wing.

    All arguments in body frame, nondimensional.
    Returns (force, torque) in body frame; torque is about body COM.
    Wind is assumed uniform across the wing (single body-frame vector).
    """
    R_body_wing = wing.hinge_orientation @ state.R_hinge_from_wing
    omega_wing_rel_body = wing.hinge_orientation @ state.omega_wing_in_hinge

    N = wing.n_elements
    r = (np.arange(N) + 0.5) * (wing.span_ratio / N)   # (N,) radial midpoints

    wing_x_body = R_body_wing[:, 0]                                       # (3,)
    r_from_hinge_body = r[:, None] * wing_x_body[None, :]                 # (N,3)
    positions_body = wing.hinge_position[None, :] + r_from_hinge_body     # (N,3)

    v_elem_body = (
        v_body[None, :]
        + np.cross(omega_body, positions_body)
        + np.cross(omega_wing_rel_body, r_from_hinge_body)
    )
    v_rel_body = v_elem_body - wind_body[None, :]            # element vel rel to air
    v_rel_wing = v_rel_body @ R_body_wing                    # rotate body -> wing

    vy = v_rel_wing[:, 1]
    vz = v_rel_wing[:, 2]
    V2 = vy * vy + vz * vz
    # AOA: zero at head-on chordwise flow (+y wind direction), +pi/2 when wind
    # hits the dorsal (+z) side. v_wind = -v_rel, so alpha = atan2(-vz, vy).
    alpha = np.arctan2(-vz, vy)

    Cl = wing.lift_coeff(alpha)
    Cd = wing.drag_coeff(alpha)

    safeV = np.where(V2 > 1e-24, np.sqrt(V2), 1.0)
    d_y = -vy / safeV
    d_z = -vz / safeV
    # Lift direction in wing frame = cross(drag_dir_3d, x_hat) = (0, d_z, -d_y)
    l_y =  d_z
    l_z = -d_y

    scale = 0.5 * wing.aero_ratio / N * V2                    # (N,)
    Fy_wing = scale * (Cl * l_y + Cd * d_y)
    Fz_wing = scale * (Cl * l_z + Cd * d_z)
    F_elem_wing = np.stack([np.zeros(N), Fy_wing, Fz_wing], axis=1)

    F_elem_body = F_elem_wing @ R_body_wing.T                 # rotate wing -> body
    F_total = F_elem_body.sum(axis=0)
    T_total = np.cross(positions_body, F_elem_body).sum(axis=0)
    return F_total, T_total

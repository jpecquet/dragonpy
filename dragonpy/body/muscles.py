"""
Muscle / kinematics layer.

The brain writes `StrokePattern`s (one per wing) and the shared
`wing_frequency` scalar. On each fast tick the muscle layer integrates the
per-wing phase accumulator and expands (pattern, phase) into a `WingState`
(rotation matrix + angular velocity relative to the hinge).

Stroke parameterization (per wing):
  * sweep, elevation, feather are pure sinusoids of a single phase variable,
    with elevation allowed an integer harmonic multiplier (default 2) to
    produce the figure-8 stroke.
  * feather is locked to the fundamental; its offset sets the timing of the
    stroke-reversal wing flip.
  * stroke-plane tilt is applied as a rotation about the hinge spanwise axis
    (+x) _before_ the ZYX stroke rotation. The composed pose is therefore
    R_hinge_from_wing = Rx(tilt) @ Rz(sweep) @ Ry(elev) @ Rx(feather).

Chirality:
  Sweep, feather, and stroke-plane tilt get sign-flipped on left wings
  (chirality = -1) so that a single symmetric pattern produces mirror-
  symmetric physical motion on both sides. Elevation is symmetric and is
  not flipped.
"""

from dataclasses import dataclass

import numpy as np

from .wings import Wing, WingState, rot_x, rot_y, rot_z


@dataclass
class StrokePattern:
    """High-level command for one wing, written by the brain."""

    stroke_plane_tilt: float        # rotation of stroke plane about hinge +x

    sweep_amp:    float
    sweep_mean:   float
    sweep_phase:  float             # phase offset (sets fore/hind lag, L/R symmetry)

    elev_amp:       float
    elev_mean:      float
    elev_phase:     float           # relative to sweep
    elev_harmonic:  int = 2         # elevation oscillates at k * fundamental

    feather_amp:    float = 0.0
    feather_mean:   float = 0.0
    feather_phase:  float = 0.0     # ~±pi/2 places the flip at stroke reversal


def _zyx_rotation(sweep: float, elev: float, feather: float) -> np.ndarray:
    return rot_z(sweep) @ rot_y(elev) @ rot_x(feather)


def _zyx_omega(
    sweep: float, elev: float, feather: float,
    sweep_d: float, elev_d: float, feather_d: float,
) -> np.ndarray:
    """Angular velocity of a ZYX intrinsic Euler rotation, in the base frame."""
    cp, sp = np.cos(sweep), np.sin(sweep)
    ct, st = np.cos(elev),  np.sin(elev)
    return np.array([
        -sp * elev_d + cp * ct * feather_d,
         cp * elev_d + sp * ct * feather_d,
         sweep_d     - st * feather_d,
    ])


def expand_pattern(
    pattern: StrokePattern,
    phase: float,
    omega: float,
    chirality: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a stroke pattern at the given phase into a wing pose.

    `phase` is the integrated wing phase (radians). `omega` is dphase/dt
    (= 2*pi*wing_frequency). `chirality` is +1 (right) or -1 (left).

    Returns (R_hinge_from_wing, omega_wing_in_hinge).
    """
    # Raw sinusoidal angles (before chirality, before tilt).
    ps = phase + pattern.sweep_phase
    sweep_raw   = pattern.sweep_mean + pattern.sweep_amp * np.sin(ps)
    sweep_d_raw =                      pattern.sweep_amp * omega * np.cos(ps)

    k = pattern.elev_harmonic
    pe = k * phase + pattern.elev_phase
    elev_raw   = pattern.elev_mean + pattern.elev_amp * np.sin(pe)
    elev_d_raw =                     pattern.elev_amp * (k * omega) * np.cos(pe)

    pf = phase + pattern.feather_phase
    feather_raw   = pattern.feather_mean + pattern.feather_amp * np.sin(pf)
    feather_d_raw =                        pattern.feather_amp * omega * np.cos(pf)

    # Apply chirality: sweep, feather, and tilt flip on left wings.
    sweep      = chirality * sweep_raw
    sweep_d    = chirality * sweep_d_raw
    feather    = chirality * feather_raw
    feather_d  = chirality * feather_d_raw
    elev       = elev_raw
    elev_d     = elev_d_raw
    tilt       = chirality * pattern.stroke_plane_tilt

    # Compose: stroke-plane tilt about hinge +x, then ZYX stroke rotation.
    # Tilt is constant over a fast tick so it contributes no extra omega.
    R_zyx = _zyx_rotation(sweep, elev, feather)
    omega_zyx = _zyx_omega(sweep, elev, feather, sweep_d, elev_d, feather_d)

    R_tilt = rot_x(tilt)
    R_hinge_from_wing = R_tilt @ R_zyx
    omega_wing_in_hinge = R_tilt @ omega_zyx
    return R_hinge_from_wing, omega_wing_in_hinge


def expand_all(
    wings: list[Wing],
    patterns: list[StrokePattern],
    phases: list[float],
    wing_states: list[WingState],
    wing_frequency: float,
) -> None:
    """Write fresh poses into `wing_states` in place."""
    omega = 2.0 * np.pi * wing_frequency
    for wing, pat, phase, state in zip(wings, patterns, phases, wing_states):
        R, w = expand_pattern(pat, phase, omega, wing.chirality)
        state.R_hinge_from_wing = R
        state.omega_wing_in_hinge = w

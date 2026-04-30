"""
Muscle / kinematics layer.

The brain writes `StrokePattern`s (one per wing) and the shared
`wing_frequency` scalar. On each fast tick the muscle layer integrates the
per-wing phase accumulator and expands (pattern, phase) into a `WingState`
(rotation matrix + angular velocity relative to the hinge).

Stroke parameterization (per wing):
  * sweep, elevation, feather are pure sinusoids of a single phase variable,
    with elevation allowed an integer harmonic multiplier (default 2) to
    produce the figure-8 stroke. Positive elevation deflects the wing
    forward (toward the leading edge in the stroke plane); for a vertical
    stroke plane this means the wing tips point forward.
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

    # Optional multi-harmonic Fourier series.  When set, these replace the
    # single-sinusoid fields above.  Each entry is (amplitude, phase_offset)
    # for harmonic k = 1, 2, …  Convention: mean + Σ A_k cos(k φ + δ_k).
    sweep_harmonics:   list[tuple[float, float]] | None = None
    feather_harmonics: list[tuple[float, float]] | None = None
    elev_harmonics:    list[tuple[float, float]] | None = None


def _fourier_eval(mean: float, harmonics: list[tuple[float, float]], phase: float) -> float:
    val = mean
    for k, (amp, delta) in enumerate(harmonics, 1):
        val += amp * np.cos(k * phase + delta)
    return val


def _fourier_deriv(harmonics: list[tuple[float, float]], omega: float, phase: float) -> float:
    val = 0.0
    for k, (amp, delta) in enumerate(harmonics, 1):
        val += -k * omega * amp * np.sin(k * phase + delta)
    return val


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
    # Raw angles (before chirality, before tilt).
    if pattern.sweep_harmonics is not None:
        sweep_raw   = _fourier_eval(pattern.sweep_mean, pattern.sweep_harmonics, phase)
        sweep_d_raw = _fourier_deriv(pattern.sweep_harmonics, omega, phase)
    else:
        ps = phase + pattern.sweep_phase
        sweep_raw   = pattern.sweep_mean + pattern.sweep_amp * np.sin(ps)
        sweep_d_raw =                      pattern.sweep_amp * omega * np.cos(ps)

    if pattern.elev_harmonics is not None:
        elev_raw   = _fourier_eval(pattern.elev_mean, pattern.elev_harmonics, phase)
        elev_d_raw = _fourier_deriv(pattern.elev_harmonics, omega, phase)
    else:
        k = pattern.elev_harmonic
        pe = k * phase + pattern.elev_phase
        elev_raw   = pattern.elev_mean + pattern.elev_amp * np.sin(pe)
        elev_d_raw =                     pattern.elev_amp * (k * omega) * np.cos(pe)

    if pattern.feather_harmonics is not None:
        feather_raw   = _fourier_eval(pattern.feather_mean + np.pi / 2, pattern.feather_harmonics, phase)
        feather_d_raw = _fourier_deriv(pattern.feather_harmonics, omega, phase)
    else:
        pf = phase + pattern.feather_phase
        feather_raw   = pattern.feather_mean + np.pi / 2 + pattern.feather_amp * np.sin(pf)
        feather_d_raw =                        pattern.feather_amp * omega * np.cos(pf)

    # Apply chirality: sweep, feather, and tilt flip on left wings.
    # Elevation is negated so that positive elevation = forward (toward
    # the leading edge in the stroke plane).
    sweep      = chirality * -sweep_raw
    sweep_d    = chirality * -sweep_d_raw
    feather    = chirality * feather_raw
    feather_d  = chirality * feather_d_raw
    elev       = -elev_raw
    elev_d     = -elev_d_raw
    tilt       = chirality * -pattern.stroke_plane_tilt

    # Compose: stroke-plane tilt about hinge +x, then ZYX stroke rotation.
    # Tilt is constant over a fast tick so it contributes no extra omega.
    R_zyx = _zyx_rotation(sweep, elev, feather)
    omega_zyx = _zyx_omega(sweep, elev, feather, sweep_d, elev_d, feather_d)

    R_tilt = rot_x(tilt)
    R_hinge_from_wing = R_tilt @ R_zyx
    omega_wing_in_hinge = R_tilt @ omega_zyx
    return R_hinge_from_wing, omega_wing_in_hinge


def wing_station_kinematics(
    wing: Wing,
    pattern: StrokePattern,
    phases: np.ndarray,
    wing_frequency: float,
    s: float,
    v_body: np.ndarray | None = None,
    omega_body: np.ndarray | None = None,
    wind_body: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of one wing span station, in body frame.

    Sweeps the pattern over `phases` (radians) and reports the section at
    span fraction `s` (0 = hinge, 1 = tip). Returns (position, velocity),
    each shape (T, 3) where T = len(phases).

    With v_body, omega_body, wind_body all zero (default) the returned
    velocity is the section's velocity relative to the body, in body frame
    (i.e. pure wing motion). Supplying them gives the air-relative velocity
    of the section in body frame, matching the convention used in
    `wing_wrench`: v_section = v_body + omega_body x r + omega_wing x r_hinge − wind.

    `v_body`, `omega_body`, and `wind_body` may be (3,) for a constant value
    or (T, 3) to vary over the cycle.
    """
    phases = np.atleast_1d(np.asarray(phases, dtype=float))
    T = phases.shape[0]
    omega = 2.0 * np.pi * wing_frequency
    radius = s * wing.span_ratio

    def _broadcast(v):
        if v is None:
            return np.zeros((T, 3))
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            return np.broadcast_to(v, (T, 3))
        return v

    v_body_arr     = _broadcast(v_body)
    omega_body_arr = _broadcast(omega_body)
    wind_arr       = _broadcast(wind_body)

    pos = np.empty((T, 3))
    vel = np.empty((T, 3))
    for i, phase in enumerate(phases):
        R_hw, omega_wh = expand_pattern(pattern, float(phase), omega, wing.chirality)
        R_bw = wing.hinge_orientation @ R_hw
        omega_wing_rel_body = wing.hinge_orientation @ omega_wh
        wing_x_body = R_bw[:, 0]
        r_from_hinge_body = radius * wing_x_body
        position_body = wing.hinge_position + r_from_hinge_body
        v_section = (
            v_body_arr[i]
            + np.cross(omega_body_arr[i], position_body)
            + np.cross(omega_wing_rel_body, r_from_hinge_body)
            - wind_arr[i]
        )
        pos[i] = position_body
        vel[i] = v_section

    return pos, vel


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

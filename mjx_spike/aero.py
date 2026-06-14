"""
JAX port of the report's quasi-steady blade-element aero (single 2/3-span
element, four wings), faithful to `summer_paper/1_hover_feasibility/feasibility.py`
`_wing_force_vec` with `element_span_fracs=(2/3,)`, evaluated at ONE phase.

Everything here is `jnp`, pure, and differentiable. The instantaneous total
aerodynamic force is a function of the body-frame velocity, the wingbeat phase,
and the two control handles (phi1, psi0); the rest of the operating point is
carried in a static `Cfg`. This is the function the MJX step injects as an
external wrench, and the thing we differentiate through.

Frame note: the spike's body has translation-only DOF (attitude frozen, matching
the report's point-mass hover), so body frame == world frame and the returned
force is directly the world wrench on the thorax. Adding attitude later means
rotating `vel` into the body frame and returning the force in the body frame.

Units (repo convention): L0 = body length, T0 = sqrt(L0/g), body mass = 1, g = 1.
omega == omega_star (the stroke ODE is phase = omega_star * t).
"""

from typing import NamedTuple

import jax.numpy as jnp

# Aerodynamic coefficients, exactly report1 section 1.
CL0 = 1.5
CD0 = 0.1
CD90 = 2.0

# Study convention: one lumped blade element at the 2/3-span station.
FRAC = 2.0 / 3.0

# Hinge frames (+x spanwise outward, +z dorsal), per feasibility.py.
_R_HINGE_RIGHT = jnp.array([[0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
_R_HINGE_LEFT = jnp.array([[0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])

# Wing order fore_R, fore_L, hind_R, hind_L: (chirality, hinge, hind?).
_WINGS = (
    (+1.0, _R_HINGE_RIGHT, 0.0),
    (-1.0, _R_HINGE_LEFT, 0.0),
    (+1.0, _R_HINGE_RIGHT, 1.0),
    (-1.0, _R_HINGE_LEFT, 1.0),
)


class Cfg(NamedTuple):
    """Static operating point (everything except the two control handles)."""
    gamma0: float   # stroke-plane angle (rad)
    psi1: float     # feather amplitude (rad)
    delta0: float   # feather phase (rad)
    omega: float    # omega_star, nondim wingbeat angular rate
    Aw: float       # Aw/m_b, inverse wing loading per wing
    Lw: float       # wing length ratio
    sigma0: float   # fore/hind phase shift (rad)


def _Rx(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0],
                      [0.0, c, -s],
                      [0.0, s, c]])


def _Rz(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0.0],
                      [s, c, 0.0],
                      [0.0, 0.0, 1.0]])


def _wing_force(vel, theta, phi1, psi0, ch, H, cfg):
    """Body-frame force from one wing at phase `theta` (scalar). Mirrors the
    numpy `_wing_force_vec` single-element path line for line."""
    sweep = -ch * phi1 * jnp.sin(theta)
    sweep_d = -ch * phi1 * cfg.omega * jnp.cos(theta)
    feather = ch * (psi0 + jnp.pi / 2 + cfg.psi1 * jnp.sin(theta + cfg.delta0))
    feather_d = ch * (cfg.psi1 * cfg.omega * jnp.cos(theta + cfg.delta0))
    tilt = -ch * cfg.gamma0

    R_tilt = _Rx(tilt)
    R_hw = R_tilt @ _Rz(sweep) @ _Rx(feather)
    R_bw = H @ R_hw                                    # wing -> body

    cp, sp = jnp.cos(sweep), jnp.sin(sweep)
    omega_zyx = jnp.array([cp * feather_d, sp * feather_d, sweep_d])
    omega_rel_body = H @ (R_tilt @ omega_zyx)

    r = FRAC * cfg.Lw
    wing_x_body = R_bw[:, 0]
    r_from_hinge = r * wing_x_body
    v_elem = jnp.cross(omega_rel_body, r_from_hinge) + vel   # + body translation

    v_rel_wing = R_bw.T @ v_elem                       # body -> wing
    vy_le = ch * v_rel_wing[1]
    vz = v_rel_wing[2]
    V2 = vy_le * vy_le + vz * vz
    alpha = jnp.arctan2(-vz, vy_le)

    Cl = CL0 * jnp.sin(2.0 * alpha)
    Cd = CD0 * jnp.cos(alpha) ** 2 + CD90 * jnp.sin(alpha) ** 2

    safeV = jnp.sqrt(jnp.where(V2 > 1e-24, V2, 1.0))
    d_le, d_z = -vy_le / safeV, -vz / safeV
    l_le, l_z = d_z, -d_le

    scale = 0.5 * cfg.Aw * V2                          # N = 1 element
    F_le = scale * (Cl * l_le + Cd * d_le)
    Fz_w = scale * (Cl * l_z + Cd * d_z)
    F_wing = jnp.array([0.0, ch * F_le, Fz_w])
    return R_bw @ F_wing                               # wing -> body


def aero_force(vel, phase, phi1, psi0, cfg):
    """Total instantaneous aero force (body frame, (3,)) from all four wings."""
    F = jnp.zeros(3)
    for ch, H, hind in _WINGS:
        theta = phase + hind * cfg.sigma0
        F = F + _wing_force(vel, theta, phi1, psi0, ch, H, cfg)
    return F


def cycle_avg_force(phi1, psi0, cfg, n_phase=256):
    """Cycle-averaged force at rest (v_body = 0): the trim/feasibility quantity."""
    phases = jnp.linspace(0.0, 2.0 * jnp.pi, n_phase, endpoint=False)
    zero = jnp.zeros(3)

    def at(phase):
        return aero_force(zero, phase, phi1, psi0, cfg)

    import jax
    return jnp.mean(jax.vmap(at)(phases), axis=0)

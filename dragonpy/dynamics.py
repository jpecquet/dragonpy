"""
Dynamics driver: how the dragonfly body, mind, and the external world interact.

The integrated state is a flat vector:
    [ position(3) | attitude_quat(4) | velocity_body(3) | omega_body(3) | wing_phases(n) ]

A fast tick integrates this vector by RK4 under the rigid-body equations of
motion plus the wing-phase ODE. Aero forces are recomputed at every RK4
substep by re-expanding the stroke patterns at the substep's wing phases.

A slow tick wraps `fast_per_slow` fast ticks between one sensor sample and
one brain update.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .body.muscles import expand_pattern
from .body.wings import WingState, wing_force_point_mass, wing_wrench
from .dragonfly import Dragonfly
from .numerics.rotations import (
    quat_derivative,
    quat_normalize,
    quat_to_matrix,
)
from .world import Environment


@dataclass
class Simulation:
    dragonfly:     Dragonfly
    environment:   Environment
    t:             float = 0.0
    dt_fast:       float = 0.01
    fast_per_slow: int   = 15


# ---------------------------------------------------------------------------
# Pack / unpack between Dragonfly and a flat numpy state vector.


def _state_size(dfly: Dragonfly) -> int:
    if dfly.point_mass:
        return 6 + len(dfly.wings)
    return 13 + len(dfly.wings)


def pack(dfly: Dragonfly) -> np.ndarray:
    s = np.empty(_state_size(dfly))
    if dfly.point_mass:
        s[0:3] = dfly.position
        s[3:6] = dfly.velocity
        s[6:]  = dfly.wing_phases
        return s
    s[0:3]   = dfly.position
    s[3:7]   = dfly.attitude
    s[7:10]  = dfly.velocity
    s[10:13] = dfly.angular_velocity
    s[13:]   = dfly.wing_phases
    return s


def unpack(state: np.ndarray, dfly: Dragonfly) -> None:
    if dfly.point_mass:
        dfly.position    = state[0:3].copy()
        dfly.velocity    = state[3:6].copy()
        dfly.wing_phases = state[6:].copy()
        return
    dfly.position         = state[0:3].copy()
    dfly.attitude         = quat_normalize(state[3:7])
    dfly.velocity         = state[7:10].copy()
    dfly.angular_velocity = state[10:13].copy()
    dfly.wing_phases      = state[13:].copy()


# ---------------------------------------------------------------------------
# Derivative of the integrated state.


def _deriv_point_mass(state: np.ndarray, sim: Simulation) -> np.ndarray:
    dfly = sim.dragonfly
    env  = sim.environment

    position = state[0:3]
    velocity = state[3:6]
    phases   = state[6:]

    wind_body    = env.wind(position, sim.t)
    gravity_body = env.gravity_direction
    omega_phase  = 2.0 * np.pi * dfly.wing_frequency

    F_total = np.zeros(3)
    for wing, pat, phase in zip(dfly.wings, dfly.stroke_patterns, phases):
        R_hw, w_hinge = expand_pattern(pat, float(phase), omega_phase, wing.chirality)
        ws = WingState(R_hinge_from_wing=R_hw, omega_wing_in_hinge=w_hinge)
        F_total += wing_force_point_mass(wing, ws, velocity, wind_body)

    out = np.empty_like(state)
    if dfly.tethered:
        out[0:6] = 0.0
        out[6:]  = omega_phase
        return out
    out[0:3] = velocity
    out[3:6] = F_total + gravity_body
    out[6:]  = omega_phase
    return out


def deriv(state: np.ndarray, sim: Simulation) -> np.ndarray:
    dfly = sim.dragonfly
    if dfly.point_mass:
        return _deriv_point_mass(state, sim)

    env  = sim.environment

    position = state[0:3]
    q        = quat_normalize(state[3:7])
    velocity = state[7:10]
    omega    = state[10:13]
    phases   = state[13:]

    R_world_body = quat_to_matrix(q)
    R_body_world = R_world_body.T
    wind_body    = R_body_world @ env.wind(position, sim.t)
    gravity_body = R_body_world @ env.gravity_direction

    omega_phase = 2.0 * np.pi * dfly.wing_frequency

    F_total = np.zeros(3)
    T_total = np.zeros(3)
    for wing, pat, phase in zip(dfly.wings, dfly.stroke_patterns, phases):
        R_hw, w_hinge = expand_pattern(pat, float(phase), omega_phase, wing.chirality)
        ws = WingState(R_hinge_from_wing=R_hw, omega_wing_in_hinge=w_hinge)
        F_i, T_i = wing_wrench(wing, ws, velocity, omega, wind_body)
        F_total += F_i
        T_total += T_i

    dpos = R_world_body @ velocity
    dvel = F_total + gravity_body - np.cross(omega, velocity)

    dq = quat_derivative(q, omega)
    I = dfly.inertia_body
    I_omega = I * omega
    domega = (T_total - np.cross(omega, I_omega)) / I

    dphases = np.full(len(phases), omega_phase)

    out = np.zeros_like(state)
    if dfly.tethered:
        out[13:] = dphases
        return out
    out[0:3]   = dpos
    out[3:7]   = dq
    out[7:10]  = dvel
    out[10:13] = domega
    out[13:]   = dphases
    return out


# ---------------------------------------------------------------------------
# Integrator and tick loops.


def rk4_step(state: np.ndarray, dt: float, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def step_fast(sim: Simulation) -> None:
    state = pack(sim.dragonfly)
    state = rk4_step(state, sim.dt_fast, lambda s: deriv(s, sim))
    unpack(state, sim.dragonfly)

    # Refresh cached wing poses for observability (not part of integration).
    dfly = sim.dragonfly
    omega_phase = 2.0 * np.pi * dfly.wing_frequency
    for i, (wing, pat) in enumerate(zip(dfly.wings, dfly.stroke_patterns)):
        R_hw, w_hinge = expand_pattern(
            pat, float(dfly.wing_phases[i]), omega_phase, wing.chirality
        )
        dfly.wing_states[i].R_hinge_from_wing = R_hw
        dfly.wing_states[i].omega_wing_in_hinge = w_hinge

    sim.environment.step_prey(sim.t, sim.dt_fast)
    sim.t += sim.dt_fast


def step_slow(sim: Simulation) -> None:
    dt_slow = sim.dt_fast * sim.fast_per_slow
    sim.dragonfly.sensors.sample_all(sim)
    sim.dragonfly.brain.update(sim.dragonfly, dt_slow)
    for _ in range(sim.fast_per_slow):
        step_fast(sim)


def run(sim: Simulation, t_end: float) -> None:
    while sim.t < t_end:
        step_slow(sim)

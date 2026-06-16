"""
Articulated dragonfly in MuJoCo/MJX: a free body plus four wings as real
hinged bodies, in the project body-frame convention.

BODY FRAME (and the MuJoCo world frame, which coincides with the body's home
attitude):
    x  longitudinal, pointing FORWARD
    y  pointing to the insect's RIGHT
    z  pointing VENTRALLY (toward the belly) -- DOWN when upright
    (x, z) is the sagittal plane.  Right-handed: x_fwd x y_right = z_ventral.
Gravity therefore points along +z (ventral/down when upright), magnitude 1 in
the repo's nondimensionalization (L0 = body length, T0 = sqrt(L0/g), m_body = 1).

This frame is the old dragonpy body frame (x fwd, y LEFT, z UP) rotated 180 deg
about x, i.e. related by D = diag(1, -1, -1). The whole validated quasi-steady
aero of `feasibility.py` therefore carries over UNCHANGED except that each hinge
matrix H is replaced by D @ H (so spanwise-outward and dorsal flip with the
frame), and the hover force comes out as (0, 0, -1) -- dorsal, opposing gravity.

WINGS. Wing roots sit on the longitudinal axis at a signed x-offset from the COM
(fore > 0, hind < 0); y = z = 0 on the centerline. Each wing is two nested
single-joint bodies (verified to compose as parent * Rz(sweep) * Rx(feather)):
    sweep carrier  : hinge about local z (stroke), zero-pose = D@H @ Rx(tilt)
    wing plate     : hinge about local x (feather), carries the geom
so the wing-relative orientation is  D@H @ Rx(tilt) @ Rz(sweep) @ Rx(feather),
exactly the report kinematics. The plate's local frame is the aero "wing frame"
(x = span outward, z = dorsal, y = z x x), so the `feasibility` blade-element
formula applies verbatim with the same chirality factor.

Wings can be made massless (mass -> ~0) via `massless=True`; a small joint
`armature` then keeps the mass matrix positive-definite (a massless body with a
driven DOF is otherwise singular in MuJoCo). Massive wings use a realistic
mass_ratio and react inertially on the free body through the hinges.

Units & trim mirror the spike: gamma0 = 40 deg reference config, single 2/3-span
element, trim phi1 ~= 20.02 deg, psi0 ~= 26.67 deg.
"""

from dataclasses import dataclass, field

import numpy as np

# --- aero coefficients & study convention (identical to feasibility.py) ------
CL0, CD0, CD90 = 1.5, 0.1, 2.0
FRAC = 2.0 / 3.0

# Old-frame hinge matrices (feasibility.py), then reflected into the new frame.
_D = np.diag([1.0, -1.0, -1.0])
_H_OLD_RIGHT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_H_OLD_LEFT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
H_RIGHT = _D @ _H_OLD_RIGHT          # = [[0,1,0],[1,0,0],[0,0,-1]]
H_LEFT = _D @ _H_OLD_LEFT            # = [[0,-1,0],[-1,0,0],[0,0,-1]]


@dataclass(frozen=True)
class Cfg:
    gamma0: float = np.radians(40.0)   # stroke-plane angle
    psi1: float = np.radians(51.0)     # feather amplitude
    delta0: float = np.pi / 2.0        # feather phase
    omega: float = 14.0                # omega_star
    Aw: float = 0.15                   # inverse wing loading per wing
    Lw: float = 0.75                   # wing length ratio
    sigma0: float = np.pi              # fore/hind phase shift
    chord_frac: float = 0.25           # chord / span (geometry only)


@dataclass(frozen=True)
class WingSpec:
    name: str
    ch: float        # chirality: +1 right, -1 left
    hind: bool       # hind pair carries the sigma0 phase
    root_x: float    # longitudinal root offset from COM (fwd > 0)


def default_wings(root_fore=0.2, root_hind=-0.2):
    return [
        WingSpec("fore_R", +1.0, False, root_fore),
        WingSpec("fore_L", -1.0, False, root_fore),
        WingSpec("hind_R", +1.0, True, root_hind),
        WingSpec("hind_L", -1.0, True, root_hind),
    ]


# ---------------------------------------------------------------------------
# Kinematics (feasibility sign conventions; unchanged by the frame reflection).

def sweep_angle(theta, phi1, ch):
    return ch * phi1 * np.sin(theta)


def sweep_rate(theta, phi1, ch, omega):
    return ch * phi1 * omega * np.cos(theta)


def feather_angle(theta, psi0, cfg, ch):
    return ch * (psi0 + np.pi / 2.0 + cfg.psi1 * np.sin(theta - cfg.delta0))


def feather_rate(theta, cfg, ch, omega):
    return ch * cfg.psi1 * omega * np.cos(theta - cfg.delta0)


def wing_phase(theta, spec, cfg):
    return theta - (cfg.sigma0 if spec.hind else 0.0)


# ---------------------------------------------------------------------------
# Reference orientation: wing-relative rotation matrix (analytic oracle), new
# frame. This is exactly feasibility's R_bw with H -> D@H.

def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def analytic_R_bw(theta, phi1, psi0, spec, cfg):
    H = H_RIGHT if spec.ch > 0 else H_LEFT
    tilt = -spec.ch * cfg.gamma0
    s = sweep_angle(theta, phi1, spec.ch)
    f = feather_angle(theta, psi0, cfg, spec.ch)
    return H @ _Rx(tilt) @ _Rz(s) @ _Rx(f)


# ---------------------------------------------------------------------------
# Model construction (programmatic MJCF).

def _mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def build_xml(cfg: Cfg, wings, massless=True, mass_ratio=0.014, free=True,
              actuated=True, kp=200.0, kv=20.0, armature=None, thorax_inertia=None):
    """Programmatic MJCF for the articulated dragonfly in the new frame."""
    Lw, chord = cfg.Lw, cfg.chord_frac * cfg.Lw
    m_wing = 1e-6 if massless else mass_ratio
    # A driven DOF on a ~massless body is singular; a little rotor inertia
    # (armature) regularizes it. Massive wings need none.
    # Even massive wings have a tiny feather (pitch-about-span) inertia -- a thin
    # plate -- so a small armature is needed to keep the stiff tracking servo
    # well-conditioned at the integration timestep. Massless needs more.
    if armature is None:
        armature = 1e-2 if massless else 1e-4

    wing_blocks = []
    act_blocks = []
    for w in wings:
        H = H_RIGHT if w.ch > 0 else H_LEFT
        tilt = -w.ch * cfg.gamma0
        q0 = _mat2quat(H @ _Rx(tilt))
        qstr = " ".join(f"{v:.10f}" for v in q0)
        wing_blocks.append(f"""
      <body name="{w.name}_sweep" pos="{w.root_x:.6f} 0 0" quat="{qstr}">
        <joint name="{w.name}_sweep" type="hinge" axis="0 0 1" armature="{armature}"/>
        <geom type="sphere" size="0.01" mass="{m_wing/2:.3e}" contype="0" conaffinity="0"/>
        <body name="{w.name}" pos="0 0 0">
          <joint name="{w.name}_feather" type="hinge" axis="1 0 0" armature="{armature}"/>
          <geom name="{w.name}_plate" type="box" pos="{Lw/2:.6f} 0 0"
                size="{Lw/2:.6f} {chord/2:.6f} 0.004" mass="{m_wing/2:.3e}"
                contype="0" conaffinity="0" rgba="0.4 0.6 0.9 0.6"/>
        </body>
      </body>""")
        if actuated:
            # Trajectory tracking = position servo (kp, on angle error) PLUS a
            # velocity actuator fed the reference RATE (kv, on velocity error).
            # The feedforward rate avoids lag and damps the error without
            # fighting the prescribed motion -- pure-kp servos ring, pure
            # position+kv servos lag a fast reference.
            for jname in (f"{w.name}_sweep", f"{w.name}_feather"):
                act_blocks.append(
                    f'    <position name="{jname}_p" joint="{jname}" kp="{kp}"/>')
                act_blocks.append(
                    f'    <velocity name="{jname}_v" joint="{jname}" kv="{kv}"/>')

    if free == 'slide':       # translation only, attitude locked (point-mass)
        thorax_joint = ('<joint name="tx" type="slide" axis="1 0 0"/>'
                        '<joint name="ty" type="slide" axis="0 1 0"/>'
                        '<joint name="tz" type="slide" axis="0 0 1"/>')
    elif free:                # full 6-DOF
        thorax_joint = '<freejoint name="root"/>'
    else:                     # bolted down
        thorax_joint = ''

    # Body inertia: explicit diagonal (Ixx, Iyy, Izz) about the COM if given,
    # else auto-computed from the render ellipsoid (a slender unit-length body,
    # I_pitch ~ 0.05). Passive pitch stability is sensitive to this, so prefer
    # passing it explicitly.
    if thorax_inertia is None:
        thorax_inertial = ''
        thorax_geom_mass = '1'
    else:
        ixx, iyy, izz = thorax_inertia
        thorax_inertial = (f'<inertial pos="0 0 0" mass="1" '
                           f'diaginertia="{ixx} {iyy} {izz}"/>')
        thorax_geom_mass = '0'
    actuator_xml = ("  <actuator>\n" + "\n".join(act_blocks) + "\n  </actuator>"
                    if act_blocks else "")

    return f"""
<mujoco model="dragonfly_articulated">
  <option timestep="0.002" gravity="0 0 1" integrator="implicitfast"/>
  <compiler autolimits="true"/>
  <worldbody>
    <body name="thorax" pos="0 0 0">
      {thorax_joint}
      {thorax_inertial}
      <geom name="thorax" type="ellipsoid" size="0.5 0.1 0.1" mass="{thorax_geom_mass}"
            contype="0" conaffinity="0" rgba="0.8 0.5 0.2 1"/>{''.join(wing_blocks)}
    </body>
  </worldbody>
{actuator_xml}
</mujoco>"""

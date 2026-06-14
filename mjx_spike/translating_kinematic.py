"""
Kinematic wing driving (exact, no servo).

The thorax is the ONLY integrated body (free joint). The wing kinematics are
prescribed exactly from time -- s = s0 sin(wt), and its feather -- so there is no
servo, no tracking error, and the timestep is set only by body-integration
accuracy, not by tracking. The wing's aero force and its (acceleration-
independent) inertial reaction are applied to the thorax as a wrench:

    F_thorax(at wing) = F_aero + m_w g  -  m_w * a_known
    a_known = w x (w x r) + 2 w x v_rel + R (s_ddot u)     [centrifugal, Coriolis, stroke]

The acceleration-COUPLED reaction (-m_w[a_thorax + alpha x r]) cannot be applied as
a wrench (it depends on the unknown body acceleration). Its translational part is
a small added-mass (O(m_w/m_b)=3%), but its ROTATIONAL part is the wing's
parallel-axis pitch inertia m_w(s^2+h^2) -- NOT small when the stroke s0 is large
(here m_w*s0^2 ~ I_b). We restore it approximately by adding the cycle-averaged
value m_w(<s^2>+h^2) = m_w(s0^2/2 + h^2) to the body's pitch/yaw inertia. That is
the deliberate "different mass formulation": exact kinematics, wing weight + bob
exact, wing rotational inertia approximated by a constant. The EOM (exact,
time-varying inertia) stays the reference.

Wings are treated as one aggregate element (co-located, in-phase): set P.n_elem=1
so aero uses the full A_agg, and m_w = mw_agg.
"""
import sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import mujoco, mujoco.mjx as mjx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import translating as T


def build(p, dt, inertia):
    ixx, iyy, izz = inertia
    xml = f"""
<mujoco>
  <option timestep="{dt}" gravity="0 0 1" integrator="RK4"/>
  <worldbody>
    <body name="thorax" pos="0 0 0">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="1" diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="ellipsoid" size="0.5 0.08 0.08" mass="0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mjx.put_model(m), m.body("thorax").id


def body_inertia(p, I_b):
    """Body pitch/yaw inertia plus the cycle-averaged wing parallel-axis inertia."""
    I_wing = p.mw_agg * (p.s0 ** 2 / 2 + p.root_x ** 2)
    return (1e-3, I_b + I_wing, I_b + I_wing)


def make_rollout(mx, p, n_steps, thid, psi1, init_quat, wing_inertia=True):
    u = jnp.array([jnp.cos(p.gamma), 0.0, jnp.sin(p.gamma)])
    g = jnp.array([0.0, 0.0, 1.0])

    def step(dx, _):
        R = dx.xmat[thid].reshape(3, 3)
        t = dx.time
        s = p.s0 * jnp.sin(p.omega * t)
        sd = p.s0 * p.omega * jnp.cos(p.omega * t)
        sdd = -p.s0 * p.omega ** 2 * jnp.sin(p.omega * t)
        r_b = jnp.array([p.root_x, 0.0, 0.0]) + s * u
        p_wing = dx.xpos[thid] + R @ r_b
        jacp, _ = mjx.jac(mx, dx, p_wing, thid)
        v_wing = jacp.T @ dx.qvel + R @ (sd * u)          # exact: thorax point + stroke
        F_aero = T.aero_element(v_wing, t, 0.0, psi1, False, p, R)
        omega = dx.cvel[thid][:3]                         # world angular velocity
        r_w = p_wing - dx.xipos[thid]
        a_known = (jnp.cross(omega, jnp.cross(omega, r_w))
                   + 2 * jnp.cross(omega, R @ (sd * u)) + R @ (sdd * u))
        F = F_aero + p.mw_agg * g - (p.mw_agg * a_known if wing_inertia else 0.0)
        tau = jnp.cross(r_w, F)
        dx = dx.replace(xfrc_applied=dx.xfrc_applied.at[thid, :3].set(F).at[thid, 3:].set(tau))
        dx = mjx.step(mx, dx)
        return dx, (dx.xpos[thid], dx.xquat[thid])

    def run():
        dx = mjx.make_data(mx)
        dx = dx.replace(qpos=dx.qpos.at[3:7].set(jnp.asarray(init_quat)))
        _, (pos, quat) = jax.lax.scan(step, dx, None, length=n_steps)
        return pos, quat
    return run


if __name__ == "__main__":
    psi1 = np.radians(45.0)
    p = T.P(gamma=np.radians(90), omega=11.0, A_agg=0.024, s0=1.5, delta0=np.pi / 2,
            sigma0=0.0, n_elem=1, root_x=0.1, mw_agg=0.03, square=True)
    dt, NWB = 0.0003, 50
    spc = int((2 * np.pi / p.omega) / dt)
    inertia = body_inertia(p, 0.0915)
    print("EOM reference (square, 50 wb):  x=-225.93  altitude=+40.16  theta=+20.75 deg")
    print(f"kinematic model: exact stroke (no servo), body pitch inertia "
          f"{inertia[1]:.4f} (= I_b 0.0915 + cycle-avg wing {inertia[1]-0.0915:.4f})\n")
    m, mx, thid = build(p, dt, inertia)
    roll = make_rollout(mx, p, NWB * spc, thid, psi1, T.pitch_quat(90.0), wing_inertia=True)
    pos, quat = jax.jit(roll)()
    pos, quat = np.asarray(pos), np.asarray(quat)
    print(f"  endpoint: x={pos[-1,0]:+8.2f}  altitude={-pos[-1,2]:+8.2f}  "
          f"tilt={T.tilt_deg(quat[-1:])[0]:.2f} deg")

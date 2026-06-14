"""
MJX dynamics rollout of the articulated dragonfly (free 6-DOF thorax + four
hinged wings driven by position servos), in the new body frame.

Each step: set servo targets to the prescribed (sweep, feather); compute each
wing's quasi-steady aero from its MuJoCo pose and AC velocity (kinematic
Jacobian); apply it at the AC as an external wrench on the wing body; mjx.step.
MuJoCo composes body translation + attitude + wing reaction automatically -- the
whole reason to be on an articulated substrate.

Runs the trim open-loop and reports how well hover holds, for massive wings
(inertial reaction + attitude excited) and the near-massless limit.
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import articulated as A  # noqa: E402

PHI1 = np.radians(20.0152)
PSI0 = np.radians(26.6717)


def blade_element(Rw, v_ac, ch, Aw, Lw):
    """World-frame aero force from one wing (jnp), feasibility formula."""
    v_w = Rw.T @ v_ac
    vy_le = ch * v_w[1]
    vz = v_w[2]
    V2 = vy_le * vy_le + vz * vz
    alpha = jnp.arctan2(-vz, vy_le)
    Cl = A.CL0 * jnp.sin(2.0 * alpha)
    Cd = A.CD0 * jnp.cos(alpha) ** 2 + A.CD90 * jnp.sin(alpha) ** 2
    safeV = jnp.sqrt(jnp.where(V2 > 1e-24, V2, 1.0))
    d_le, d_z = -vy_le / safeV, -vz / safeV
    l_le, l_z = d_z, -d_le
    scale = 0.5 * Aw * V2
    F_wing = jnp.array([0.0, ch * scale * (Cl * l_le + Cd * d_le),
                        scale * (Cl * l_z + Cd * d_z)])
    return Rw @ F_wing


def build(cfg, wings, massless, dt, dof='free', kp=800.0, kv=8.0, mass_ratio=0.014,
          thorax_inertia=None):
    m = mujoco.MjModel.from_xml_string(
        A.build_xml(cfg, wings, massless=massless, free=dof, actuated=True,
                    kp=kp, kv=kv, mass_ratio=mass_ratio, thorax_inertia=thorax_inertia))
    m.opt.timestep = dt
    mx = mjx.put_model(m)
    # Host-side index maps (position + velocity actuator per joint).
    info = []
    for w in wings:
        info.append(dict(
            bid=m.body(w.name).id, ch=w.ch, hind=w.hind,
            sweep_p=m.actuator(f"{w.name}_sweep_p").id,
            sweep_v=m.actuator(f"{w.name}_sweep_v").id,
            feather_p=m.actuator(f"{w.name}_feather_p").id,
            feather_v=m.actuator(f"{w.name}_feather_v").id,
        ))
    return m, mx, info


def make_rollout(mx, info, cfg, n_steps, thid, dof,
                 phi1=PHI1, psi0=PSI0, init_quat=None):
    ac_local = jnp.array([A.FRAC * cfg.Lw, 0.0, 0.0])
    nu = len(info) * 4   # position + velocity actuator per sweep & feather

    def targets(t):
        ph = cfg.omega * t
        c = jnp.zeros(nu)
        for k in info:
            th = ph + (cfg.sigma0 if k["hind"] else 0.0)
            ch, om = k["ch"], cfg.omega
            sweep = -ch * phi1 * jnp.sin(th)
            sweep_d = -ch * phi1 * om * jnp.cos(th)
            feather = ch * (psi0 + jnp.pi / 2 + cfg.psi1 * jnp.sin(th + cfg.delta0))
            feather_d = ch * cfg.psi1 * om * jnp.cos(th + cfg.delta0)
            c = (c.at[k["sweep_p"]].set(sweep).at[k["sweep_v"]].set(sweep_d)
                  .at[k["feather_p"]].set(feather).at[k["feather_v"]].set(feather_d))
        return c

    def aero(dx):
        xfrc = jnp.zeros_like(dx.xfrc_applied)
        for k in info:
            bid = k["bid"]
            Rw = dx.xmat[bid].reshape(3, 3)
            p_ac = dx.xpos[bid] + Rw @ ac_local
            jacp, _ = mjx.jac(mx, dx, p_ac, bid)
            v_ac = jacp.T @ dx.qvel
            F = blade_element(Rw, v_ac, k["ch"], cfg.Aw, cfg.Lw)
            tau = jnp.cross(p_ac - dx.xipos[bid], F)
            xfrc = xfrc.at[bid, :3].add(F).at[bid, 3:].add(tau)
        return xfrc

    def step(dx, _):
        dx = dx.replace(ctrl=targets(dx.time), xfrc_applied=aero(dx))
        dx = mjx.step(mx, dx)
        return dx, (dx.xpos[thid], dx.xquat[thid])   # thorax world pose

    def run():
        dx = mjx.make_data(mx)
        if dof == 'free':
            q0 = jnp.array([1.0, 0, 0, 0]) if init_quat is None else jnp.asarray(init_quat)
            dx = dx.replace(qpos=dx.qpos.at[3:7].set(q0))
        _, (pos, quat) = jax.lax.scan(step, dx, None, length=n_steps)
        return pos, quat

    return run


def quat_tilt_deg(q):
    """Angle (deg) between body z (ventral) and world z, i.e. attitude excursion."""
    w, x, y, z = q.T
    # body z-axis expressed in world = third column of R(q)
    bz = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    return np.degrees(np.arccos(np.clip(bz[:, 2], -1, 1)))


def run_case(massless, dt, n_cycles, dof, label):
    cfg = A.Cfg()
    wings = A.default_wings()
    m, mx, info = build(cfg, wings, massless, dt, dof=dof)
    thid = m.body("thorax").id
    n_steps = int(n_cycles * (2 * np.pi / cfg.omega) / dt)
    pos, quat = jax.jit(make_rollout(mx, info, cfg, n_steps, thid, dof))()
    pos, quat = np.asarray(pos), np.asarray(quat)   # thorax starts at origin
    spc = int((2 * np.pi / cfg.omega) / dt)
    last = pos[-spc:]
    tilt = quat_tilt_deg(quat)
    print(f"  [{label}]  {n_cycles} cycles, dt={dt}")
    print(f"    net drift:  dx={pos[-1,0]:+.4f}  dy={pos[-1,1]:+.4f}  dz={pos[-1,2]:+.4f} L")
    print(f"    bob p2p:    x={np.ptp(last[:,0]):.2e}  z={np.ptp(last[:,2]):.2e} L")
    print(f"    attitude excursion (body-z vs vertical): max {tilt.max():.3f} deg")


def main():
    print("Articulated hover rollout (open-loop trim):\n")
    print(" translation-only thorax (attitude locked -> isolates aero/integration):")
    run_case(massless=True, dt=0.0002, n_cycles=10, dof='slide',
             label="massless, 3-slide")
    print("\n free 6-DOF thorax (attitude free -> real open-loop flapping flight):")
    run_case(massless=False, dt=0.0002, n_cycles=10, dof='free',
             label="massive  (m_w=0.014)")
    run_case(massless=True, dt=0.0002, n_cycles=10, dof='free',
             label="massless (m_w~0)")


if __name__ == "__main__":
    main()

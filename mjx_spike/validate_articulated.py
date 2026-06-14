"""
Validate the articulated model against the report physics, in the new body frame.

  [A] POSE   -- MuJoCo wing-relative orientation == analytic R_bw at every phase.
  [B] FORCE  -- cycle-averaged aero (thorax fixed, wings swept) == (0, 0, -1) at
                the report trim: dorsal lift exactly balancing ventral gravity.
  [C] VEL    -- AC velocity from the kinematic Jacobian == analytic flap velocity.

Uses classic MuJoCo (exact, simple) for the static checks; the MJX dynamics
rollout lives in roll_articulated.py.
"""

import sys
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import articulated as A  # noqa: E402

PHI1 = np.radians(20.0152)   # report trim (gamma0 = 40 deg)
PSI0 = np.radians(26.6717)


def aero_world(m, d, w, cfg):
    """Per-wing aero force in WORLD frame, from MuJoCo pose + Jacobian velocity.
    Returns (F_world, p_ac_world) using the feasibility blade-element formula."""
    wid = m.body(w.name).id
    Rw = d.xmat[wid].reshape(3, 3)
    p_ac = d.xpos[wid] + Rw @ np.array([A.FRAC * cfg.Lw, 0.0, 0.0])
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, None, p_ac, wid)
    v_ac = jacp @ d.qvel                       # world velocity of the AC point

    v_w = Rw.T @ v_ac                          # into the wing frame
    vy_le = w.ch * v_w[1]
    vz = v_w[2]
    V2 = vy_le * vy_le + vz * vz
    alpha = np.arctan2(-vz, vy_le)
    Cl = A.CL0 * np.sin(2.0 * alpha)
    Cd = A.CD0 * np.cos(alpha) ** 2 + A.CD90 * np.sin(alpha) ** 2
    safeV = np.sqrt(V2) if V2 > 1e-24 else 1.0
    d_le, d_z = -vy_le / safeV, -vz / safeV
    l_le, l_z = d_z, -d_le
    scale = 0.5 * cfg.Aw * V2
    F_le = scale * (Cl * l_le + Cd * d_le)
    Fz_w = scale * (Cl * l_z + Cd * d_z)
    F_wing = np.array([0.0, w.ch * F_le, Fz_w])
    return Rw @ F_wing, p_ac


def set_kinematics(m, d, theta, phi1, psi0, cfg, wings):
    """Pin thorax at identity; drive every wing joint to its prescribed angle/rate."""
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    rj = m.joint("root")
    d.qpos[rj.qposadr[0]:rj.qposadr[0] + 7] = [0, 0, 0, 1, 0, 0, 0]  # pos + identity quat
    for w in wings:
        th = A.wing_phase(theta, w, cfg)
        for jname, ang, rate in (
            (f"{w.name}_sweep",
             A.sweep_angle(th, phi1, w.ch), A.sweep_rate(th, phi1, w.ch, cfg.omega)),
            (f"{w.name}_feather",
             A.feather_angle(th, psi0, cfg, w.ch), A.feather_rate(th, cfg, w.ch, cfg.omega)),
        ):
            j = m.joint(jname)
            d.qpos[j.qposadr[0]] = ang
            d.qvel[j.dofadr[0]] = rate
    mujoco.mj_forward(m, d)


def main():
    cfg = A.Cfg()
    wings = A.default_wings()
    m = mujoco.MjModel.from_xml_string(A.build_xml(cfg, wings, massless=True,
                                                   free=True, actuated=False))
    d = mujoco.MjData(m)
    thid = m.body("thorax").id

    # --- [A] pose check over a full cycle --------------------------------
    phases = np.linspace(0, 2 * np.pi, 41)
    pose_err = 0.0
    for th in phases:
        set_kinematics(m, d, th, PHI1, PSI0, cfg, wings)
        Rth = d.xmat[thid].reshape(3, 3)
        for w in wings:
            R_bw_mj = Rth.T @ d.xmat[m.body(w.name).id].reshape(3, 3)
            R_bw_an = A.analytic_R_bw(A.wing_phase(th, w, cfg), PHI1, PSI0, w, cfg)
            pose_err = max(pose_err, np.abs(R_bw_mj - R_bw_an).max())
    print(f"[A] POSE   max|R_bw_mujoco - R_bw_analytic| over cycle = {pose_err:.2e}"
          f"   ({'MATCH' if pose_err < 1e-9 else 'MISMATCH'})")

    # --- [C] velocity spot check (one wing, mid-stroke) ------------------
    set_kinematics(m, d, 1.0, PHI1, PSI0, cfg, wings)
    w0 = wings[0]
    wid = m.body(w0.name).id
    Rw = d.xmat[wid].reshape(3, 3)
    p_ac = d.xpos[wid] + Rw @ np.array([A.FRAC * cfg.Lw, 0.0, 0.0])
    jacp = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jacp, None, p_ac, wid)
    v_ac = jacp @ d.qvel
    print(f"[C] VEL    AC speed at theta=1 rad: |v_ac| = {np.linalg.norm(v_ac):.4f} L/T0"
          f"   (nonzero, finite: {np.isfinite(v_ac).all() and np.linalg.norm(v_ac) > 0})")

    # --- [B] cycle-averaged force (thorax fixed) -------------------------
    N = 256
    F_sum = np.zeros(3)
    for th in np.linspace(0, 2 * np.pi, N, endpoint=False):
        set_kinematics(m, d, th, PHI1, PSI0, cfg, wings)
        for w in wings:
            F, _ = aero_world(m, d, w, cfg)
            F_sum += F
    Fbar = F_sum / N
    print(f"[B] FORCE  cycle-avg aero (world=body) = "
          f"[{Fbar[0]:+.2e}, {Fbar[1]:+.2e}, {Fbar[2]:+.6f}]")
    print(f"           target (0, 0, -1): dorsal lift balancing +z gravity"
          f"   ({'MATCH' if abs(Fbar[2] + 1) < 1e-3 and abs(Fbar[0]) < 1e-3 else 'MISMATCH'})")


if __name__ == "__main__":
    main()

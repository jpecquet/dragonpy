"""
Translating-element model (matches the report's quasi-steady model and the
user's known-stable reference).

Each wing is a POINT MASS m_w that translates along the stroke axis
    u_hat = (cos gamma, 0, sin gamma)      [body frame]
(the intersection of the stroke plane and the sagittal plane) with displacement
amplitude s0 -- there is NO wingspan, only s0. The feather angle psi(t) is
prescribed (not a DOF) and only sets the angle of attack. The aero is computed
from the element's TRANSLATIONAL velocity (slide + body motion), so there is no
spanwise gradient and no rotational element velocity -- the purely-translating
idealization. Contrast articulated.py, whose swept rigid wings add both (and the
huge Lw=6 rotational inertia), which destabilized this config.

Sagittal-plane geometry (orthonormal, body frame):
    u_hat = (cos g, 0, sin g)   stroke / translation axis
    n_hat = (sin g, 0, -cos g)  stroke normal = mean lift (hover) direction
Feather psi sets the chord c_hat = cos(psi-pi/2) u + sin(psi-pi/2) n (so psi=pi/2
is edge-on, alpha=0). Lift acts on the +n side; drag opposes velocity. The four
in-phase collinear elements emulate a two-wing pair (their lateral momenta cancel
in reality, absent here; their sagittal bob adds, captured here).

World frame = body home frame: x fwd, y right, z ventral(down), gravity +z.
"""

from dataclasses import dataclass
import numpy as np
import jax, jax.numpy as jnp
import mujoco, mujoco.mjx as mjx

CL0, CD0, CD90 = 1.5, 0.1, 2.0


@dataclass(frozen=True)
class P:
    gamma: float      # stroke-plane angle (body-relative); pi/2 = horizontal stroke, vertical body
    omega: float
    A_agg: float      # AGGREGATE A/m_b (summed over all wings); per-element = A_agg / n_elem
    s0: float
    delta0: float = np.pi / 2
    sigma0: float = 0.0
    n_elem: int = 4   # wing count -- a pure discretization axis when wings are co-located/in-phase
    root_x: float = 0.1
    mw_agg: float = 0.03   # AGGREGATE m_w/m_b (summed over all wings); per-element = mw_agg / n_elem
    square: bool = False     # True: bang-bang feather (+-psi1, flip at reversal)


def u_n(g):
    return (np.array([np.cos(g), 0, np.sin(g)]), np.array([np.sin(g), 0, -np.cos(g)]))


# ---------------------------------------------------------------------------
# Aero for one element. v_world: AC velocity; R_tw: world<-body (thorax xmat).

def aero_element(v_world, t, psi0, psi1, hind, p, R_tw):
    g = p.gamma
    u = jnp.array([jnp.cos(g), 0.0, jnp.sin(g)])
    n = jnp.array([jnp.sin(g), 0.0, -jnp.cos(g)])
    th = p.omega * t + jnp.where(hind, p.sigma0, 0.0)
    fp = jnp.sin(th + p.delta0)
    psi = psi0 + jnp.pi / 2 + psi1 * (jnp.sign(fp) if p.square else fp)

    vb = R_tw.T @ v_world
    a = vb @ u
    b = vb @ n
    speed = jnp.sqrt(a * a + b * b + 1e-12)
    c, s = jnp.cos(psi - jnp.pi / 2), jnp.sin(psi - jnp.pi / 2)
    vc = (a * c + b * s) / speed          # vhat . chord
    vn = (-a * s + b * c) / speed         # vhat . chord-normal
    alpha = jnp.arctan2(vn, vc)
    CL = CL0 * jnp.sin(2 * alpha)
    CD = CD0 * jnp.cos(alpha) ** 2 + CD90 * jnp.sin(alpha) ** 2
    lift = (b * u - a * n) / speed        # vhat rotated -90 in (u,n): +n side
    drag = -(a * u + b * n) / speed
    Fb = 0.5 * (p.A_agg / p.n_elem) * speed ** 2 * (CL * lift + CD * drag)
    return R_tw @ Fb


# ---------------------------------------------------------------------------
# Static cycle-averaged force (body fixed): for trim + direction check.

def cycle_avg_force_body(psi0, psi1, p, n=256):
    u, n_ = u_n(p.gamma)
    F = np.zeros(3)
    R = np.eye(3)
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        for k in range(p.n_elem):
            hind = False                       # all in-phase, body-fixed
            vw = p.s0 * p.omega * np.cos(th) * u   # stroke velocity only
            F += np.asarray(aero_element(jnp.asarray(vw), jnp.asarray(th / p.omega),
                                         psi0, psi1, hind, p, jnp.asarray(R)))
    return F / n


def solve_psi1(p, M, psi0=0.0):
    lo, hi = np.radians(0.5), np.radians(60.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = np.linalg.norm(cycle_avg_force_body(psi0, mid, p))
        lo, hi = (mid, hi) if f < M else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Model build.

def build_xml(p, dt, thorax_inertia=(1e-3, 1/12, 1/12), kp=500.0, kv=10.0):
    u, _ = u_n(p.gamma)
    ux, uy, uz = u
    elems, acts = [], []
    for i in range(p.n_elem):
        elems.append(f"""
      <body name="elem{i}" pos="{p.root_x} 0 0">
        <joint name="slide{i}" type="slide" axis="{ux:.6f} {uy:.6f} {uz:.6f}" range="-3 3"/>
        <geom type="sphere" size="0.03" mass="{p.mw_agg / p.n_elem}" contype="0" conaffinity="0" rgba="0.4 0.6 0.9 0.7"/>
      </body>""")
        acts.append(f'    <position name="slide{i}_p" joint="slide{i}" kp="{kp}"/>')
        acts.append(f'    <velocity name="slide{i}_v" joint="slide{i}" kv="{kv}"/>')
    ixx, iyy, izz = thorax_inertia
    return f"""
<mujoco model="translating">
  <option timestep="{dt}" gravity="0 0 1" integrator="implicitfast"/>
  <compiler autolimits="true"/>
  <worldbody>
    <body name="thorax" pos="0 0 0">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="1" diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="ellipsoid" size="0.5 0.08 0.08" mass="0" contype="0" conaffinity="0" rgba="0.8 0.5 0.2 1"/>{''.join(elems)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# Rollout.

def make_rollout(mx, p, n_steps, eids, aids, thid, psi0, psi1, init_quat):
    def targets(t):
        c = jnp.zeros(2 * p.n_elem)
        for i in range(p.n_elem):
            ph = p.omega * t + (p.sigma0 if False else 0.0)
            c = c.at[aids[i][0]].set(p.s0 * jnp.sin(ph)).at[aids[i][1]].set(p.s0 * p.omega * jnp.cos(ph))
        return c

    def aero(dx):
        xfrc = jnp.zeros_like(dx.xfrc_applied)
        R_tw = dx.xmat[thid].reshape(3, 3)
        for i in range(p.n_elem):
            bid = eids[i]
            jacp, _ = mjx.jac(mx, dx, dx.xpos[bid], bid)
            v = jacp.T @ dx.qvel
            F = aero_element(v, dx.time, psi0, psi1, False, p, R_tw)
            xfrc = xfrc.at[bid, :3].set(F)
        return xfrc

    def step(dx, _):
        dx = dx.replace(ctrl=targets(dx.time), xfrc_applied=aero(dx))
        dx = mjx.step(mx, dx)
        return dx, (dx.xpos[thid], dx.xquat[thid])

    def run():
        dx = mjx.make_data(mx).replace()
        dx = dx.replace(qpos=dx.qpos.at[3:7].set(jnp.asarray(init_quat)))
        _, (pos, quat) = jax.lax.scan(step, dx, None, length=n_steps)
        return pos, quat
    return run


def build(p, dt, thorax_inertia, kp=500.0, kv=10.0):
    m = mujoco.MjModel.from_xml_string(build_xml(p, dt, thorax_inertia, kp, kv))
    mx = mjx.put_model(m)
    eids = [m.body(f"elem{i}").id for i in range(p.n_elem)]
    aids = [(m.actuator(f"slide{i}_p").id, m.actuator(f"slide{i}_v").id) for i in range(p.n_elem)]
    return m, mx, eids, aids, m.body("thorax").id


def pitch_quat(deg):
    a = np.radians(deg) / 2
    return np.array([np.cos(a), 0.0, np.sin(a), 0.0])


def tilt_deg(quat):
    w, x, y, z = quat.T
    return np.degrees(np.arccos(np.clip(-(2 * (x * z - w * y)), -1, 1)))


def main():
    p = P(gamma=np.radians(90), omega=10.8, A_agg=0.138, s0=1.07,
          delta0=np.pi / 2, sigma0=0.0, n_elem=4, root_x=0.1, mw_agg=0.256)
    M = 1.0 + p.mw_agg
    psi1 = solve_psi1(p, M)
    F = cycle_avg_force_body(0.0, psi1, p)
    inertia = (1e-3, 1 / 12, 1 / 12)            # slender rod, body length 1
    dt, ncyc = 0.0003, 30
    spc = int((2 * np.pi / p.omega) / dt)
    print("Translating point-mass element model (report / user reference):")
    print(f"  s0={p.s0}, omega={p.omega}, A/m={p.A_agg}(agg), m_w/m={p.mw_agg}(agg), "
          f"n_elem={p.n_elem}, gamma0=90deg(body)=horizontal stroke, M={M}")
    print(f"  hover trim psi1={np.degrees(psi1):.2f}deg -> Fbar(body)={np.round(F,4)} "
          f"(+x = up when vertical)\n")
    for init in (90.0, 105.0):
        m, mx, eids, aids, thid = build(p, dt, inertia, kp=500, kv=10)
        roll = make_rollout(mx, p, ncyc * spc, eids, aids, thid, 0.0, psi1, pitch_quat(init))
        pos, quat = jax.jit(roll)()
        pos, tilt = np.asarray(pos), tilt_deg(np.asarray(quat))
        peak = np.array([tilt[i * spc:(i + 1) * spc].max() for i in range(ncyc)])
        env = "  ".join(f"c{i*5}:{peak[i*5]:.1f}" for i in range(ncyc // 5))
        decay = peak[-ncyc // 3:].mean() < peak[:ncyc // 3].mean()
        print(f"  start {init-90:+.0f}deg from vertical: peak tilt/cyc  {env}")
        print(f"    max {tilt.max():.1f}, final {peak[-1]:.1f} deg, alt {-pos[-1,2]:+.2f} L"
              f"   => {'STABLE (bounded, decaying)' if decay and peak.max()<90 else 'unstable'}")


if __name__ == "__main__":
    main()

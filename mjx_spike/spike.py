"""
MJX spike: inject the report's quasi-steady aero into a differentiable MuJoCo
(MJX) rigid-body rollout, and show it reproduces the published hover trim.

What this de-risks (the open question from the roadmap):
  1. PORT      -- the JAX aero matches the trusted numpy oracle to ~1e-12.
  2. TRIM      -- bisecting the JAX cycle-average rediscovers the report trim
                  (psi0 ~= 26.67 deg, phi1 ~= 20.02 deg) with Fbar = (0,0,1).
  3. INJECTION -- aero applied via data.xfrc_applied inside a jitted lax.scan of
                  mjx.step holds hover: the body barely drifts over 40 wingbeats.
  4. GRADIENTS -- jax.jacobian differentiates end-of-rollout position w.r.t. the
                  control handles; signs match the report's control Jacobian.
  5. BATCH     -- jax.vmap runs many rollouts at once (the PufferLib feed path).

Runs on CPU (M4): MJX's GPU backend (warp) is absent, which is fine -- the point
here is correctness + gradients, not throughput. Same code runs batched on a
CUDA GPU unchanged.

    python mjx_spike/spike.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

jax.config.update("jax_enable_x64", True)  # match the numpy oracle's precision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aero import Cfg, aero_force, cycle_avg_force  # noqa: E402

# Reference operating point (report Table: gamma0 = 40 deg config, single 2/3 elem).
CFG = Cfg(
    gamma0=np.radians(40.0),
    psi1=np.radians(51.0),
    delta0=np.pi / 2.0,
    omega=14.0,
    Aw=0.15,
    Lw=0.75,
    sigma0=np.pi,
)
STEPS_PER_CYCLE = 200
N_CYCLES = 40
PERIOD = 2.0 * np.pi / CFG.omega
DT = PERIOD / STEPS_PER_CYCLE


# ---------------------------------------------------------------------------
# Build the MJX model once.

def build_model():
    m = mujoco.MjModel.from_xml_path(str(HERE / "dragonfly_pointmass.xml"))
    m.opt.timestep = DT
    bid = m.body("thorax").id
    return mjx.put_model(m), bid


MX, BID = build_model()


# ---------------------------------------------------------------------------
# Differentiable rollout: aero injected as an external wrench each step.

def rollout(phi1, psi0, n_steps):
    dx0 = mjx.make_data(MX)

    def step(dx, _):
        F = aero_force(dx.qvel, CFG.omega * dx.time, phi1, psi0, CFG)
        dx = dx.replace(xfrc_applied=dx.xfrc_applied.at[BID, :3].set(F))
        dx = mjx.step(MX, dx)
        return dx, (dx.qpos, dx.qvel)

    _, (qpos, qvel) = jax.lax.scan(step, dx0, None, length=n_steps)
    return qpos, qvel


_rollout_jit = jax.jit(rollout, static_argnums=2)


# ---------------------------------------------------------------------------
# 1 + 2. Port cross-check and trim.

def _bisect(f, lo, hi, target, increasing, iters=60):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = float(f(mid)) < target
        if below == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_trim():
    """Alternate (phi1 for |F|=1, psi0 for Fx=0) on the JAX cycle-average."""
    cavg = jax.jit(lambda phi1, psi0: cycle_avg_force(phi1, psi0, CFG))
    phi1, psi0 = np.radians(30.0), 0.0
    for _ in range(8):
        phi1 = _bisect(lambda v: jnp.linalg.norm(cavg(v, psi0)),
                       0.05, 3.0, target=1.0, increasing=True)
        psi0 = _bisect(lambda v: cavg(phi1, v)[0],
                       np.radians(-10.0), np.radians(70.0), target=0.0,
                       increasing=False)
    return float(phi1), float(psi0)


def crosscheck_against_numpy(phi1, psi0):
    """JAX aero vs the trusted numpy feasibility oracle (single 2/3 element)."""
    try:
        sys.path.insert(0, str(HERE.parents[0] / "summer_paper" / "1_hover_feasibility"))
        sys.path.insert(0, str(HERE.parents[0]))
        from feasibility import Params, replace, cycle_averaged_force
    except Exception as e:  # pragma: no cover - oracle is optional
        return None, f"(numpy oracle unavailable: {e})"
    p = Params(Lw=CFG.Lw, Aw_over_mb=CFG.Aw, omega_star=CFG.omega,
               gamma0=CFG.gamma0, phi1=phi1, psi0=psi0, psi1=CFG.psi1,
               delta0=CFG.delta0, sigma0=CFG.sigma0,
               element_span_fracs=(2.0 / 3.0,))
    F_np = cycle_averaged_force(p, 256)
    F_jx = np.asarray(cycle_avg_force(phi1, psi0, CFG))
    return float(np.max(np.abs(F_np - F_jx))), F_np


# ---------------------------------------------------------------------------

def main():
    print(f"config: gamma0=40deg, omega*={CFG.omega}, Aw/m={CFG.Aw}, Lw={CFG.Lw}, "
          f"sigma0=180deg | dt={DT:.5f}, {N_CYCLES} cycles\n")

    # --- 2. Trim ---------------------------------------------------------
    phi1, psi0 = solve_trim()
    Fbar = np.asarray(cycle_avg_force(phi1, psi0, CFG))
    print("[2] TRIM (bisected on JAX cycle-average)")
    print(f"    phi1 = {np.degrees(phi1):7.4f} deg   (report 20.02)")
    print(f"    psi0 = {np.degrees(psi0):7.4f} deg   (report 26.67)")
    print(f"    Fbar = [{Fbar[0]:+.2e}, {Fbar[1]:+.2e}, {Fbar[2]:.6f}]  (target [0,0,1])\n")

    # --- 1. Port cross-check --------------------------------------------
    err, F_np = crosscheck_against_numpy(phi1, psi0)
    print("[1] PORT cross-check  (JAX aero vs numpy feasibility oracle)")
    if err is None:
        print(f"    {F_np}")
    else:
        print(f"    max|F_numpy - F_jax| = {err:.2e}   ({'MATCH' if err < 1e-9 else 'MISMATCH'})\n")

    # --- 3. Hover rollout ------------------------------------------------
    n_steps = N_CYCLES * STEPS_PER_CYCLE
    qpos, qvel = _rollout_jit(phi1, psi0, n_steps)
    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    drift = qpos[-1] - np.zeros(3)
    last = qpos[-STEPS_PER_CYCLE:]
    p2p = last.max(0) - last.min(0)
    print("[3] HOVER rollout in MJX (aero via xfrc_applied inside jitted scan)")
    print(f"    net drift over {N_CYCLES} cycles: "
          f"dx={drift[0]:+.4f}, dy={drift[1]:+.4f}, dz={drift[2]:+.4f} body lengths")
    print(f"    within-cycle bob (last cycle p2p): "
          f"x={p2p[0]:.2e}, z={p2p[2]:.2e}")
    print(f"    mean speed: {np.linalg.norm(qvel, axis=1).mean():.2e} L/T0\n")

    # --- 4. Gradients ----------------------------------------------------
    # End-of-rollout (x, z) as a function of the two control handles.
    def end_xz(controls):
        qp, _ = rollout(controls[0], controls[1], n_steps)
        return jnp.array([qp[-1, 0], qp[-1, 2]])

    jac = jax.jit(jax.jacobian(end_xz))
    J = np.asarray(jac(jnp.array([phi1, psi0])))
    print("[4] GRADIENTS  d(end x,z)/d(phi1, psi0) through the rollout")
    print(f"    d x_end/d phi1 = {J[0,0]:+.3f}   d x_end/d psi0 = {J[0,1]:+.3f}")
    print(f"    d z_end/d phi1 = {J[1,0]:+.3f}   d z_end/d psi0 = {J[1,1]:+.3f}")
    print("    expected signs: more phi1 -> climbs (dz/dphi1>0); "
          "more psi0 -> drifts -x (dx/dpsi0<0)\n")

    # The differentiable payoff: a Gauss-Newton step on the rollout Jacobian
    # finds a *dynamic* trim that returns the body to the origin after 40 cycles,
    # beating the static cycle-average trim. (Plain fixed-step GD diverges here:
    # the 40-cycle landscape is extremely stiff, |dz/dphi1| ~ 186.)
    end0 = np.asarray(end_xz(jnp.array([phi1, psi0])))
    c = jnp.array([phi1, psi0])
    for _ in range(2):
        c = c - jnp.linalg.solve(jac(c), end_xz(c))
    end1 = np.asarray(end_xz(c))
    print("[4b] Gauss-Newton dynamic trim (2 steps via the rollout Jacobian)")
    print(f"    end (x,z): ({end0[0]:+.2e}, {end0[1]:+.2e}) -> "
          f"({end1[0]:+.2e}, {end1[1]:+.2e})")
    print(f"    phi1: {np.degrees(phi1):.4f} -> {np.degrees(float(c[0])):.4f} deg | "
          f"psi0: {np.degrees(psi0):.4f} -> {np.degrees(float(c[1])):.4f} deg\n")

    # --- 5. Batch --------------------------------------------------------
    phi1_batch = jnp.linspace(0.9 * phi1, 1.1 * phi1, 16)
    batched = jax.jit(jax.vmap(lambda a: rollout(a, psi0, n_steps)[0][-1, 2]))
    z_ends = np.asarray(batched(phi1_batch))
    print("[5] BATCH via vmap (16 rollouts, the PufferLib feed path)")
    print(f"    phi1 in [{np.degrees(phi1_batch[0]):.1f}, {np.degrees(phi1_batch[-1]):.1f}] deg "
          f"-> z_end in [{z_ends.min():+.3f}, {z_ends.max():+.3f}] "
          f"(monotone climb with amplitude: {np.all(np.diff(z_ends) > 0)})")


if __name__ == "__main__":
    main()

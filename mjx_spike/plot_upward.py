"""
(x, z) trajectory of the translating-element model over 100 wingbeats, for the
stable upward-flight case: Aw/m=0.01725, psi1=45deg (fixed), F/W=2.58, body
starting vertical. Body attitude (longitudinal axis) drawn along the path.

World frame: x forward (horizontal), z ventral (down) so altitude = -z.
"""
import sys
from pathlib import Path
import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import translating as T

p = T.P(gamma=np.radians(90), omega=11.0, A_agg=0.024, s0=1.5,
        delta0=np.pi / 2, sigma0=0.0, n_elem=4, root_x=0.1, mw_agg=0.03, square=True)
psi1 = np.radians(45.0)
DT, NC = 0.0003, 50
spc = int((2 * np.pi / p.omega) / DT)
inertia = (1e-3, 0.0915, 0.0915)        # paper's dimensionless pitch inertia

m, mx, eids, aids, thid = T.build(p, DT, inertia, kp=500, kv=10)
roll = T.make_rollout(mx, p, NC * spc, eids, aids, thid, 0.0, psi1, T.pitch_quat(90.0))
pos, quat = jax.jit(roll)()
pos, quat = np.asarray(pos), np.asarray(quat)

x = pos[:, 0]
h = -pos[:, 2]                      # altitude (up)
# body longitudinal (+x) axis in world, from quat (w,x,y,z)
w, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
bx_x = 1 - 2 * (qy * qy + qz * qz)
bx_z = 2 * (qx * qz - w * qy)       # world-z component (down)

fig, ax = plt.subplots(figsize=(6, 7))
ax.plot(x, h, color="0.5", lw=1.0, zorder=1)
sc = ax.scatter(x, h, c=np.arange(len(x)) / spc, cmap="viridis", s=2, zorder=2)
cb = fig.colorbar(sc, ax=ax, label="wingbeats", shrink=0.8)
# attitude arrows every 5 wingbeats: body +x (points "up" when vertical)
L = 0.6
for i in range(0, len(x), 5 * spc):
    ax.plot([x[i], x[i] + L * bx_x[i]], [h[i], h[i] - L * bx_z[i]],
            color="crimson", lw=1.6, zorder=3)
ax.plot(x[0], h[0], "o", color="black", ms=6, zorder=4, label="start (vertical)")
ax.set_xlabel("x  (forward, body lengths)")
ax.set_ylabel("altitude  −z  (up, body lengths)")
ax.plot(x[-1], h[-1], "r*", ms=14, zorder=5)
ax.set_title(f"MuJoCo translating model (square pitch): A/m=0.024, m_w/m=0.03 (agg)\n"
             f"{NC} wingbeats, ω=11, s₀=1.5, ψ₁=45°, I=0.0915, n_elem={p.n_elem}")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(loc="best", fontsize=9)
out = HERE / "trajectory_ref_square.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
M = 1 + p.mw_agg
Fb = T.cycle_avg_force_body(0.0, psi1, p)
print(f"wrote {out}")
print(f"F/W = {np.linalg.norm(Fb)/M:.3f}  (M={M})")
print(f"MuJoCo endpoint @ {NC} wb:  x={x[-1]:+.3f}  z={pos[-1,2]:+.3f} "
      f"(altitude -z = {h[-1]:+.3f})  tilt={T.tilt_deg(quat[-1:])[0]:.2f} deg")

"""
Wing kinematics comparison: hover vs pursuit at one span station.

Top row:
  Position trace of the section in the body x-z plane (side view).

Bottom row:
  Velocity hodograph (vx, vz). For pursuit we plot two curves:
    - dashed: section velocity relative to the body (pure wing motion).
    - solid:  section velocity relative to still air, while the body
              translates forward at V_PURSUIT in body frame.
  The gap between them is the contribution of forward flight to the
  apparent wind seen by the wing.

The right fore wing is used; other wings are mirrored / phase-shifted
versions of the same trace.
"""

import numpy as np
import matplotlib.pyplot as plt

from dragonpy.body.muscles import StrokePattern, wing_station_kinematics
from dragonpy.body.wings import Wing

from examples.parametric import BASELINE, R_HINGE_RIGHT, wang_cl, wang_cd

# ── Configuration ──────────────────────────────────────────────────────────

S = 0.75                                  # span fraction (0 = hinge, 1 = tip)
V_PURSUIT = np.array([1.5, 0.0, 0.0])     # representative body-frame velocity, T0 units
N_PHASES = 200

# ── Wing and patterns ──────────────────────────────────────────────────────

wing_right = Wing(
    hinge_position=np.zeros(3),
    hinge_orientation=R_HINGE_RIGHT,
    chirality=+1,
    span_ratio=BASELINE["span_ratio"],
    mass_ratio=BASELINE["mass_ratio"],
    aero_ratio=BASELINE["aero_ratio"],
    lift_coeff=wang_cl,
    drag_coeff=wang_cd,
    n_elements=BASELINE["n_elements"],
)

hover = StrokePattern(
    stroke_plane_tilt=BASELINE["hover_stroke_plane_tilt"],
    sweep_amp=BASELINE["hover_sweep_amp"],
    sweep_mean=0.0, sweep_phase=0.0,
    elev_amp=0.0, elev_mean=0.0, elev_phase=0.0, elev_harmonic=2,
    feather_amp=BASELINE["hover_feather_amp"],
    feather_mean=0.0,
    feather_phase=BASELINE["hover_feather_phase"],
)

# Snapshot of the pursuit pattern: same stroke-plane tilt as hover (the
# brain ramps tilt up dynamically; pick a representative value here).
pursuit = StrokePattern(
    stroke_plane_tilt=BASELINE["hover_stroke_plane_tilt"],
    sweep_amp=BASELINE["intercept_sweep_amp"],
    sweep_mean=0.0, sweep_phase=0.0,
    elev_amp=0.0, elev_mean=0.0, elev_phase=0.0, elev_harmonic=2,
    feather_amp=BASELINE["intercept_feather_amp"],
    feather_mean=0.0,
    feather_phase=BASELINE["intercept_feather_phase"],
)

# ── Kinematics ─────────────────────────────────────────────────────────────

phases = np.linspace(0.0, 2.0 * np.pi, N_PHASES, endpoint=False)
wf = BASELINE["wing_frequency"]

pos_hover, vel_hover = wing_station_kinematics(wing_right, hover, phases, wf, S)
pos_purs,  vel_purs  = wing_station_kinematics(wing_right, pursuit, phases, wf, S)
_, vel_purs_air = wing_station_kinematics(
    wing_right, pursuit, phases, wf, S, v_body=V_PURSUIT,
)

# ── Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

ax = axes[0, 0]
ax.plot(pos_hover[:, 0], pos_hover[:, 2], color="C0", lw=1.5)
ax.set_title("Hover — position (body)")
ax.set_xlabel("x (forward)"); ax.set_ylabel("z (up)")
ax.set_aspect("equal"); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(pos_purs[:, 0], pos_purs[:, 2], color="C1", lw=1.5)
ax.set_title("Pursuit — position (body)")
ax.set_xlabel("x (forward)"); ax.set_ylabel("z (up)")
ax.set_aspect("equal"); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(vel_hover[:, 0], vel_hover[:, 2], color="C0", lw=1.5)
ax.set_title("Hover — velocity hodograph")
ax.set_xlabel(r"$v_x$"); ax.set_ylabel(r"$v_z$")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

ax = axes[1, 1]
ax.plot(vel_purs[:, 0], vel_purs[:, 2], "--", color="C1", lw=1.2,
        label="rel. body (body stationary)")
ax.plot(vel_purs_air[:, 0], vel_purs_air[:, 2], "-", color="C1", lw=1.5,
        label=f"rel. air (v_body = {V_PURSUIT.tolist()})")
ax.set_title("Pursuit — velocity hodograph")
ax.set_xlabel(r"$v_x$"); ax.set_ylabel(r"$v_z$")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.legend(fontsize=8, loc="best")

fig.suptitle(f"Right fore wing, span fraction s = {S}")
fig.tight_layout()
plt.show()

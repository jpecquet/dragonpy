"""
Wing-tip-path and AoA analysis applied to the turn-pursuit run.

Loads `data/wing_delta/turn_dx50_dz20_K3.npz`, picks four wingbeat cycles
labeled by their offset from the target-jump moment (one pre-jump cycle as
the steady-state forward-cruise reference, plus three post-jump cycles
spanning the elbow turn), and renders the same two figures produced by
`wing_delta_plot.py` for the straight pursuit:

  - `turn_wing_tip_loops.dark.png` — air-frame and drift-subtracted tip
    paths against the converged hover loop.
  - `turn_wing_aoa.dark.png` — alpha and |V_air| at r/R = 0.7 over each cycle.

The renderers and the hover reference are imported from `wing_delta_plot`
so the comparison is consistent between the straight and turning runs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.capture_study import WING_FREQUENCY
from examples.wing_delta_plot import (
    OUT_DIR,
    REFERENCE_WING,
    converge_hover_pattern,
    cycle_bounds,
    hover_tip_loop,
    render_aoa,
    render_tip_loops,
)
from post.style import apply_matplotlib_style, resolve_style


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data" / "wing_delta"
T_WB      = 1.0 / WING_FREQUENCY

DATA_FILE     = "turn_dx50_dz20_K3.npz"
OUT_TIP_FILE  = "turn_wing_tip_loops.dark.png"
OUT_AOA_FILE  = "turn_wing_aoa.dark.png"

# Wingbeat offsets from jump for the four sampled cycles. -1 = the last full
# cycle before the jump (steady forward cruise); the rest are post-jump.
JUMP_OFFSETS = (-1, +1, +3, +10)


def main() -> None:
    data = np.load(DATA_DIR / DATA_FILE)
    t              = data["t"]
    pos            = data["pos"]
    vel            = data["vel"]
    gamma          = data["gamma"]
    R_hw_all       = data["R_hinge_from_wing"]
    wing_phases    = data["wing_phases"]
    hinge_orient   = data["hinge_orientations"]
    span_ratios    = data["span_ratios"]
    chiralities    = data["chiralities"]
    j              = int(data["jump_idx"])
    if j < 0:
        raise RuntimeError("turn data has no jump (jump_idx < 0)")

    w = REFERENCE_WING
    span_ratio = float(span_ratios[w])
    R_bw_w = hinge_orient[w] @ R_hw_all[:, w]
    tip_body  = R_bw_w @ np.array([span_ratio, 0.0, 0.0])
    tip_world = pos + tip_body

    bounds = cycle_bounds(wing_phases[:, w])
    n_cycles = len(bounds)

    # Find the cycle containing the jump tick.
    jump_cycle = next(
        k for k, (i0, i1) in enumerate(bounds) if i0 <= j < i1
    )

    pick = []
    for off in JUMP_OFFSETS:
        k = jump_cycle + off
        k = max(0, min(n_cycles - 2, k))
        pick.append(k)

    labels = []
    for off in JUMP_OFFSETS:
        if off < 0:
            labels.append("pre-jump")
        else:
            labels.append(rf"$+{off}\,T_\mathrm{{wb}}$")

    print(f"cycles: {n_cycles}, jump in cycle {jump_cycle}")
    for k, off, lab in zip(pick, JUMP_OFFSETS, labels):
        i0, i1 = bounds[k]
        v_cycle = vel[i0:i1].mean(axis=0)
        gamma_cycle = float(np.degrees(np.mean(gamma[i0:i1])))
        speed = float(np.linalg.norm(v_cycle[[0, 2]]))
        print(f"  cycle {k} ({lab:>16s}, off={off:+d}): "
              f"t={t[i0]:.2f}..{t[i1-1]:.2f} s, "
              f"|V|={speed:.2f}, ⟨γ⟩={gamma_cycle:+.1f}°")

    hover_pattern = converge_hover_pattern()
    hover_tip = hover_tip_loop(
        hover_pattern, hinge_orient[w], span_ratio, int(chiralities[w]),
    )

    style = resolve_style(theme="dark")
    apply_matplotlib_style(style)

    n_pick = len(pick)
    cmap = plt.cm.plasma
    colors = [cmap(0.15 + 0.7 * i / max(1, n_pick - 1)) for i in range(n_pick)]
    hover_color = "#4aa3ff"

    render_tip_loops(
        bounds, pick, labels, colors, hover_color, hover_tip,
        tip_world, pos, t, n_cycles, style,
        out_path=OUT_DIR / OUT_TIP_FILE,
    )
    render_aoa(
        bounds, pick, labels, colors, hover_color,
        R_hw_all, wing_phases, vel, gamma,
        hinge_orient, span_ratios, chiralities,
        hover_pattern, w, style,
        out_path=OUT_DIR / OUT_AOA_FILE,
    )


if __name__ == "__main__":
    main()

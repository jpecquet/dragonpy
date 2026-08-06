# report1 backend

Streamlined, self-contained (numpy-only) core of `report1`, replacing the
sprawl of `summer_paper/scripts/` for the report's quantitative results, and
intended as the physics/controller backend for the GUI.

| file | role |
|------|------|
| `sim.py` | model + controller: single-element wing force (instantaneous and cycle-averaged), analytic normal-inflow mean-force coefficients, γ-schedule, ψ₁-slave, analytic and Newton trims, once-per-beat closed-loop RK4 simulator. No plotting, no paths. |
| `trajectory.py` | reference trajectories: freehand smoothing, the primitive test course, the saved GUI drawing, the trapezoidal-speed `Reference`, hover/pursuit case definitions. |
| `figures.py` | regenerates the report's quantitative figures and metrics from the two modules above. |

## Reproduce the report

```sh
PYENV_VERSION=dragonpy pyenv exec python summer_paper/backend/figures.py [out_dir] [names...]
```

Default `out_dir` is `summer_paper/figures/` (the report's `\graphicspath`).
Names (default all): `cf_components pitch_efficiency force_direction maps_J
optimum_J beta_J hover trajectory pursuit`. Covers every computed figure of
the report — `cf_components`, `pitch_efficiency`, `force_direction_test`,
`pitch_efficiency_J`, `hover_optimum_J`, `force_direction_J`,
`analytic_trim_hover`, `trajectory_gains[_drawn]`, `analytic_trim_pursuit` —
the remaining figures are illustrative diagrams (still in
`summer_paper/scripts/`).

Verified against the originals: every core function (`cycle_force`,
`instant_force`, `averages`, `psi1_opt`, both trims, `gamma_schedule`, the
allocators) matches `summer_paper/scripts/` to machine precision, and the
hover run is bit-identical. The one deliberate change: a single ψ₁-slave
(the analytic argmax of eq:psi1-slave on a 0.25° grid) everywhere, where the
old trajectory scripts used a coarser 2°-grid numerical slave — sub-degree,
sub-linewidth effect on the trajectory figures.

## GUI use

`sim.py` is morphology-parametrized: pass `Morphology(Lw, A_over_m,
omega_star)` to `allocate`/`simulate`/`instant_force` (defaults to the
Table-1 reference). The schedules keep the fixed reference advance-ratio
scale `SCALE`, as in the report. `gui/dragonfly_sim.py` is a thin real-time
layer over this package: `core.instant_force`/`newton_trim`/schedules plus
`trajectory.smooth_path`/`Reference`, with only the GUI-specific hold-pin and
heading-latch logic kept locally.

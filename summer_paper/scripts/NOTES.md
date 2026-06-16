# Hover feasibility (report1, section 2)

Working notes for the feasibility envelope: the set of parameters whose
cycle-averaged aerodynamic force reaches the body weight. Outside it, hover and
any maneuvering anchored on hover are impossible.

## Method

The cycle-averaged force is a **1-D quadrature, not a dynamics simulation**. At
hover the body is stationary (`v_body = 0`, `omega_body = 0`) in still air, so
every blade element's air-relative velocity comes purely from wing motion:

```
Fbar = (1/2pi) integral_0^2pi  sum_wings  F_wing(state(phi)) dphi
```

Both pieces already exist in `dragonpy`: `expand_pattern` maps (pattern, phase)
to a wing pose, and `wing_force_point_mass` maps a pose to a force with
`v_body = wind = 0`. We sample the phase and average. Units: body length,

**Span model:** the whole study lumps each wing into a *single* blade element
at 2/3 span (`element_span_fracs=STUDY_ELEMENT` in `feasibility.py`). This is the
representative-station idealization; it deliberately **overestimates** the true
span-integrated force, since the force ~ `<V^2>` whose r^2-weighted span mean is
1/3 while a single 2/3 station gives `(2/3)^2 = 4/9` (~4/3 high; see
`check_translating.py`). We accept the bias because it makes
`q* = (Aw/mb) s0^2 omega^2` an *exact* velocity scale -- `s0 = (2/3) Lw phi1` is
precisely this station's arc-length excursion -- which is what the
`F_a* = q* C_psi` separation rests on. The 8-element path is retained only as the
correctness oracle for the vectorized code (`test_feasibility.py`).

`T0 = sqrt(L0/g)`, body mass = 1, so the weight to clear is exactly `W = 1`.
Aerodynamic coefficients are taken verbatim from section 1
(`C_L = 1.5 sin 2a`, `C_D = 0.1 cos^2 a + 2 sin^2 a`).

Files:
- `feasibility.py`  — `Params`, `cycle_averaged_force` (vectorized over phase),
  `cycle_averaged_force_scalar` (reference oracle), `avg_force_magnitude`,
  `is_feasible`. Parameter names mirror Table 1.
- `test_feasibility.py` — checks the vectorized path against the scalar
  reference (matches to ~2e-15) and benchmarks it (~25x faster).
- `invariances.py`  — numerical tests of the three reduction claims.
- `feasibility_sweep.py` — 2-D slices of `|Fbar|` with the `|Fbar| = W` boundary;
  writes `figures/feasibility_envelope.light.png` (light mode, for the report).

## Reduction claims, verified numerically (`invariances.py`)

| claim | result |
|---|---|
| (1) `|Fbar|` independent of stroke-plane angle `gamma0` | spread `1.9e-15` of the mean (machine zero); force *direction* swings the full +-180 deg |
| (2) `|Fbar|` independent of fore/hind phase `sigma0` | spread `4.8e-9` |
| (3) `Lw`, `phi1` collapse onto `s0 = (2/3) Lw phi1` | within-`s0`-bin scatter = 1.45% of total range |

So the 9 parameters of Table 1 reduce to **6** that set the magnitude:
`s0`, `A_over_m`, `omega_star`, `psi0`, `psi1`, `delta0`.

(`Params.A_over_m` is the *total* inverse wing loading `A*/m* = N_w Aw rho L/m`
of Table 1; the per-wing `Aw/m = A_over_m / N_w` that the force law and the `q*`
above use is the `Params.aw_over_m` property. `N_w = 4`.)

Claim (1) is the key one for the paper's theme: `gamma0` rotates the whole
averaged force vector without changing its magnitude, so it is exactly the
**steering knob** for maneuvering. Feasibility is therefore a pure magnitude
test, `|Fbar| >= W`; `gamma0` then aims the surplus.

## Envelope (`feasibility_envelope.dark.png`)

- **(a) excursion vs frequency** and **(b) wing loading vs frequency**:
  `|Fbar|` scales like `(omega* s0)^2` and linearly in wing loading, so both
  boundaries are hyperbolas; feasibility lives in the upper-right.
- **(c) pitch geometry**: lift peaks near `psi0 = 0` and dies at large mean
  pitch (the section stalls / feathers out of the lifting regime).
- **(d) pitch timing**: two feasible lobes around `delta0 = +-90 deg` (wing flip
  at stroke reversal); near `delta0 = 0, +-180 deg` the pitch is mistimed and the
  cycle-averaged force collapses below weight.

## Caveats / next steps

- The force is vectorized over phase, so a 60x60x128-phase sweep is ~12 s
  (was ~4.5 min with the scalar loop). The scalar path is retained as a
  correctness oracle (`test_feasibility.py`).
- Feasibility here is a *necessary* condition (enough average force, orientable
  by `gamma0`). It does not yet check that the surplus can be modulated fast
  enough for stable proportional control — that is the controller question the
  rest of the report addresses.
- Elevation (figure-8) is held at zero, matching Table 1; adding it would be a
  natural sensitivity check.

## Maneuvering: C_psi over the velocity plane (report1, section 3.1)

Body velocity parameterized in the stroke-plane frame: advance ratio
`J = U*/(s0* omega*)` (radius) and angle `chi` from the stroke-plane normal.
This keeps the gamma0-invariance (rotate stroke plane + velocity together) and
preserves the q* scaling, so velocity enters only through `C_psi(pitch; J, chi)`.

Files:
- `feasibility_velocity.py` — does the `F = q* C_psi` separation survive
  velocity? Residual is *exactly zero* for axial inflow (chi = 0/180) and grows
  with the in-plane component only (~7% CV at J=0.5, ~11% at J=1, chi=90).
  Geometric reason: axial inflow meets the wing identically along the stroke
  arc; in-plane wind makes a phi-varying angle with the wing velocity,
  re-introducing a phi1 dependence.
- `recon_velocity_cpsi.py` — structure checks that license the disk-map
  visualization (not a report figure).
- `velocity_cpsi_maps.py` — the report figure: three polar maps over the
  velocity disk (`figures/velocity_cpsi_maps.light.png`).

Structure found (recon, all numerical, reference config, psi0 = 0 unless noted):

| claim | result |
|---|---|
| mirror symmetry `C(J, chi, psi0) = C(J, -chi, -psi0)` | machine zero |
| optimal `delta0` over the whole disk | pinned at 90 deg |
| optimal `psi1` | retunes smoothly: ~20 deg head-on, 60-deg bound tail-on |
| `C(J, chi)` polynomial order in `(u, w)/(s0 omega)` | deg 2 fits to 3.8%, deg 4 to 1.0% |

Regimes: head-on inflow washes out AoA (hover wingbeat C: 1.43 -> 0.04 by
J=0.8; retuning psi1 down recovers ~1.0); in-plane wind raises mean-square
relative velocity (C -> 3.7 at J=0.8, hover pitch already near-optimal);
tail-on inflow steepens AoA (optimum saturates at psi1 = 60 deg). Sharp branch
switch in psi1* near chi ~ +-110 deg at high J (fixed-pitch-like vs
large-amplitude strategy).

Control-relevant warning for section 3.2: at velocity, beta is no longer a
function of psi0 alone — e.g. at chi=90, J=0.3 the mean force tilts -31 deg at
psi0 = 0, and the beta(psi0) slope roughly doubles by J=0.6 head-on.

## Power efficiency (report1, stationary-flight section)

`power_efficiency.py` adds an energetic measure alongside `C_psi`, following the
two Wang papers in the repo root (`2004_JEB_Wang.pdf`, `jeb013797p234.pdf`).

Aerodynamic power, massless wings: only drag does work (lift is perpendicular to
the air-relative velocity; pitch-axis torque dropped with the wing inertia). Per
element `P = (1/2)(A_w*/m*) V^3 C_D`, with `V` the same in-plane relative speed
the force law uses. Cycle-mean `Pbar*` is in units of weight*sqrt(gL).

Cost of endurance (Wang 2008, eq. 1): `P* = Pbar* / U_ref*`, with
`U_ref* = sqrt(2/(A*/m*))`. For a steady translating wing it reduces to
`C_D/C_L^(3/2)`. Imposing hover (`q* C_psi = 1`) kills the amplitude scale, giving
the closed form

```
P* = 2 sqrt(2) <|cos(omega t)|^3 C_D(alpha)> / C_psi^(3/2)
```

the flapping analog of `C_D/C_L^(3/2)` (C_psi plays the role of C_L). This closed
form matches the direct power integral to machine precision (checked in main()).

Findings (reference config, single 2/3 element):
- **Max force != min power** (Wang 2008's thesis): the C_psi-optimal pitch
  (psi1~51 deg) is NOT the power optimum. Holding the force vertical, P* keeps
  dropping as psi1 rises past the C_psi peak; the power optimum sits at the
  Table-1 bound (psi1=60 deg, ~77 deg unconstrained), ~20% cheaper than the
  C_psi-optimal pitch at gamma0=0. Extra AoA is added near reversal where V is
  small, so it sheds drag power (~V^3) faster than force.
- **Inclined stroke costs power.** P* RISES with gamma0 for both the
  force-optimal and power-optimal pitch (~+50-70% from 0 to 40 deg). So power
  efficiency degrades with inclination, like C_psi -- the OPPOSITE of the
  hoped-for result.
- **Drag-supports-weight mechanism IS captured** (Wang 2004): drag share of the
  vertical force goes 0% (gamma0=0, half-strokes cancel) -> 75% (gamma0=60 deg),
  vs Wang's 76% at 63 deg. But that drag is expensive force (high AoA, high C_D),
  so C_psi drops and via C_psi^(-3/2) the cost rises.
- Cheapest hover here is ~1.7x the steady-optimal C_D/C_L^(3/2): a symmetric
  back-and-forth stroke is an inefficient member of Wang 2008's two-stroke family.

Why we differ from Wang 2004's flat specific power: Wang's efficient inclined
strokes recover cost via a near-free feathered upstroke (Fig 6 / the two-stroke)
and unsteady mechanisms, neither of which a symmetric quasi-steady stroke has.
So this model captures the force side of the drag story but not the power saving;
within the model the inclined stroke plane is a force-direction control choice,
not an efficiency one.

Figure: `figures/power_efficiency.light.png` -- (a) P* vs gamma0 for force- and
power-optimal pitch with the steady C_D/C_L^(3/2) reference; (b) drag share of
the vertical force vs gamma0 with Wang 2004's point marked.

### Square-wave pitch (does flattening the feather unlock the cheap return?)

`power_efficiency.py` also compares a **square-wave** feather (`pitch_profile=
"square"`, constant pitch each half-stroke, instant flip at reversal). Motivation:
the cheap return stroke of an efficient hoverer is a *feathered* (near-constant,
small-AoA) stroke, which a sinusoid coupled to the sweep cannot make.

- The closed form `P* = 2 sqrt(2) <|cos|^3 C_D>/C_psi^(3/2)` holds for the square
  wave too (feather rate adds no span-station velocity, so only the chord angle
  changes); `closed_form_cost`/`_reference_alpha` honor `pitch_profile`, checked
  vs the direct integral.
- Square is **cheaper at every gamma0** (each half sits near its best AoA instead
  of sweeping through high-alpha at mid-stroke): P*~0.47 vs sinusoid 0.59 at
  gamma0=0 (~1.4x the steady ideal vs 1.7x).
- Square **does feather the return**: at gamma0=40 the power-optimal square holds
  one half-stroke at <|aoa|>~61 deg (92% of cycle power) and the other at ~7 deg
  (8%). The sinusoid can't (its lighter half ~23 deg). `halfstroke_split()`
  reports this.
- **But P* still rises with gamma0** (0.47 -> 0.91 over 0->40 deg). The powered
  half must over-pitch to ~60 deg (past best L/D) to aim the resultant vertical;
  that penalty outweighs the free-return saving. Horizontal wins because the two
  halves' drag cancels, so both can run near the optimal AoA.

Figure now has three P* curves in panel (a): force-optimal (dotted),
power-optimal sinusoid (solid), power-optimal square (dash-dot). Note psi1 is
capped at the Table-1 bound 60 deg; the unconstrained optimum is higher (~77 deg
sinusoid, >60 square), so the constrained optimum rides the bound.

Bottom line unchanged: flattening the pitch improves efficiency and unlocks the
feathered return, but does NOT make inclined hover competitive with horizontal in
this quasi-steady model. Residual gap from Wang 2004's flat specific power is
unsteady aero (her flat result is Navier-Stokes; she flags quasi-steady misses it).

## Maneuvering flight control (report1, section 3.2)

`maneuver_control.py`. Two structural facts drive the design:
1. **No body drag => cruise = re-trim.** Steady straight flight at any velocity
   needs the same force as hover (weight, up). Maneuvering = transient force
   vectoring. There is no speed-dependent trim curve.
2. The cycle-averaged force is a **velocity-scheduled map** F(u; v_body); the
   hover controller is its v=0 slice. Both `cycle_averaged_force` and
   `cycle_averaged_power` already accept `v_body`, so the plant is free.

Controls u = (phi1, psi0, gamma0, psi1), delta0=90 fixed. Redundancy (4 controls,
2 force DOF) resolved by minimizing cycle-averaged power (= cost of endurance up
to the constant U_ref normalization).

**Part A - power-optimal trim over velocity** (`power_optimal_trim`,
`trim_line`, fig `maneuver_trim.light.png`):
- `trim_fast`: 2x2 damped Newton on (phi1,psi0) to hit F_des at given (gamma0,
  psi1, v) -- the velocity-aware generalization of the hover trim. Warm-started
  by continuation along velocity.
- power-optimal allocation: sweep (gamma0,psi1), trim (phi1,psi0), pick min power.
- **Findings:** power-optimal gamma0 = 0 (horizontal) for hover/forward/climb;
  only descent tilts. Forward flight: psi0 0->-54 (saturates ~J=0.9), psi1 retunes
  60->39, phi1 grows; aero power rises ~9x by J=0.9 (monotonic, no induced-power
  benefit modeled -> no efficient cruise speed). Climb feasible via psi1 retune;
  metabolic cost = aero power + climb work Vz*W. Descent -> autorotation (phi1
  small, gravity carries load).
- Muscle vs dissipated power identity: <P_muscle> = <P_dissip> + v_body.<F> =
  cycle_averaged_power + Vz*W at the weight-up trim. The allocation argmin is the
  same either way (differ by control-independent constant), so minimizing
  dissipation is fine.

**Part B - directional authority psi0 vs gamma0** (the comparison promised in the
hover-control section): at hover dbeta/dpsi0=-1.25, dbeta/dgamma0=-1.00 deg/deg
(psi0 wins, full range). By J~0.8 they cross: psi0 -> -0.80 (and trim saturates
near -55, 5 deg from bound) while gamma0 -> -1.26. => hold gamma0 horizontal at
low speed (psi0 effective + unsaturated), reserve gamma0 for fast forward flight.

**Part C - closed-loop maneuvers** (`simulate_maneuver`, fig
`maneuver_control.light.png`): first-order velocity tracker a_des = a_ref +
Kv(v_ref - v), Kv=2; F_des = a_des - g; dynamic inversion (gamma0=0, psi1 power-
opt, trim phi1,psi0). Control updated once per wingbeat, held, UNSTEADY force
integrated (reuses hover_drift.instant_force) -- same design/verify split as
hover. Forward dash (accel-cruise-decel to J~0.5): peak vx err 0.19 (accel
phase), cruise ripple std 0.018. Pull-up (velocity rotates fwd->vertical): clean
quarter-turn, peak speed err 0.04, psi0 -42->2.

Bottom line: maneuvering control = velocity-scheduled hover controller + psi1
retuning + power-optimal (=horizontal) stroke plane; gamma0 is a high-speed
directional backup, not a primary control. Runtime ~70s for both figures.

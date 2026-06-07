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
`s0`, `Aw_over_mb`, `omega_star`, `psi0`, `psi1`, `delta0`.

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

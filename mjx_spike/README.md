# MJX spike — differentiable MuJoCo substrate for flapping flight

A throwaway-friendly proof that the proposed simulator substrate works:
**inject the report's quasi-steady blade-element aero into a differentiable
MuJoCo (MJX) rigid-body rollout, and reproduce the published hover trim.**

This de-risks the one open question from the roadmap before any real
investment: *can we get MuJoCo's rigid-body dynamics + rendering + GPU batch +
differentiability while keeping our own flapping-wing aerodynamics?*

## Run

```bash
python mjx_spike/spike.py
```

Needs `jax`, `mujoco`, `mujoco-mjx` (already installed in the `dragonpy` env).
Runs on **CPU** — the M4 has no CUDA, and MJX's GPU backend (`warp`) is absent.
That is fine: this validates *correctness and gradients*, not throughput. The
exact same code runs batched on a CUDA GPU unchanged.

## What it shows (all five pass)

| # | Claim | Result |
|---|-------|--------|
| 1 | **Port is exact** — JAX aero == trusted numpy oracle (`feasibility.py`) | `max|ΔF| = 4e-16` |
| 2 | **Trim recovered** — bisecting the JAX cycle-average finds the report trim | `phi1=20.015°, psi0=26.672°`, `Fbar=(0,0,1.000000)` |
| 3 | **Aero injection holds hover** — applied via `xfrc_applied` inside a jitted `lax.scan` of `mjx.step` | drift `dz=+0.005` over 40 wingbeats; bob `z_p2p=3e-3` |
| 4 | **Gradients flow through the rollout** — `jax.jacobian` of end position w.r.t. controls, signs match the report's control Jacobian | `dz/dphi1=+186>0`, `dx/dpsi0=-59<0` |
| 4b| **Differentiable trim** — 2 Gauss–Newton steps on the rollout Jacobian zero the 40-cycle drift | end `(x,z)` → `(3e-14, 3e-15)` |
| 5 | **Batches** — `jax.vmap` runs 16 rollouts at once (the PufferLib feed path) | monotone climb with amplitude |

## Architecture (the pattern that scales)

```
aero.py                 pure JAX blade-element aero (single 2/3 element, 4 wings)
                        — a line-for-line port of feasibility._wing_force_vec,
                          differentiable, the ONLY physics we own.
dragonfly_pointmass.xml  MuJoCo model: one body, 3 slide joints (attitude frozen),
                          mass=1, gravity=-1. MuJoCo owns the rigid body.
spike.py                rollout = lax.scan over [compute aero -> set xfrc_applied
                          -> mjx.step]; jit/grad/vmap on top.
```

The division of labor is the whole point: **MuJoCo integrates the body; we supply
the aero as an external wrench.** MuJoCo never needs to know about flapping wings.

## Why this validates the substrate choice

- **Differentiability is real and useful**, not just present: [4b] gets a better
  trim than the static cycle-average by differentiating through 8000 `mjx.step`s.
  This is the lever for trajectory optimization and analytic policy gradients.
- **The numpy oracle stays the source of truth** — [1] pins the port to machine
  precision, so any future fidelity change is checked against trusted physics.
- **The batch path for RL already works** — [5] is exactly what PufferLib feeds on.

## Articulated framework (the real model — `articulated.py`)

The spike above is point-mass / attitude-frozen. `articulated.py` is the actual
framework: a free 6-DOF thorax plus **four wings as real hinged MuJoCo bodies**,
in the project body-frame convention.

**Body frame (= MuJoCo world frame at home attitude):** x forward (longitudinal),
y to the right, **z ventral (down when upright)**; (x,z) is the sagittal plane.
Right-handed. Gravity is +z (ventral). This is the old dragonpy frame (x fwd, y
left, z up) reflected by D = diag(1,−1,−1), so the validated `feasibility` aero
carries over by replacing each hinge matrix H with D·H; hover lift comes out as
(0,0,−1) (dorsal), opposing gravity.

**Wings.** Roots sit on the longitudinal axis at a signed x-offset from the COM
(fore > 0, hind < 0). Each wing is two nested single-joint bodies (sweep about
local z, feather about local x) — verified to compose as the report kinematics
`D·H · Rx(tilt) · Rz(sweep) · Rx(feather)`. `massless=True` shrinks wing mass to
~0 (a small joint `armature` regularizes the otherwise-singular driven DOF);
massive wings react inertially on the thorax through the hinges.

**Driving.** Wing joints track the prescribed sinusoid via a **position servo +
velocity-feedforward actuator** (a pure-kp servo rings; position+kv lags a fast
reference). `implicitfast` integration keeps the stiff thin-plate feather servo
stable.

### Validation (`validate_articulated.py`, `roll_articulated.py`)

| Check | Result |
|---|---|
| **[A] Pose** — MuJoCo wing-relative orientation vs analytic R_bw, over a cycle | `max|ΔR| = 1.3e-10` |
| **[B] Force** — cycle-averaged aero at the trim (thorax fixed) | `(1e-6, 0, −1.000002)` — dorsal lift balancing +z gravity |
| **[C] Velocity** — AC velocity via kinematic Jacobian | nonzero, finite |
| **Dynamics, translation-only massless** — reproduces the point-mass spike | `dz=+0.0003`, bob `5e-3` (spike: `+0.005`) |
| **Dynamics, 6-DOF massless / massive** — open-loop trim, attitude free | drifts + 5–7° attitude wobble — the **expected open-loop pitch instability** of flapping hover (massive wings *more* damped) |

The 6-DOF drift is physics, not a bug: locking attitude (3-slide thorax) recovers
the spike's tight hover; freeing it surfaces the instability that motivates the
control work. This is the flight-stability regime the substrate now exposes.

Run: `python mjx_spike/validate_articulated.py` and `python mjx_spike/roll_articulated.py`.

## Honest limitations (deliberately out of scope for a spike)

- **Translation only.** Attitude is frozen (3 slide joints) to match the report's
  point-mass hover. The next step is a **free joint + aero moments** — the whole
  reason to be on MuJoCo. The aero already computes per-wing wrenches; wiring the
  moment and a 6-DOF body is the first real extension.
- **Wings are massless and kinematically prescribed**, not MuJoCo bodies. Adding
  wing inertia (the `m_w` bob effect, body-pitch coupling) means modelling the
  hinges as real joints — a bigger change, worth it for the stability work.
- **Force is piecewise-constant over each `mjx.step`** (MuJoCo holds `xfrc_applied`
  fixed across RK4 substages, unlike the report's own integrator). Converges with
  `dt`; here `dt = period/200` and the hover drift confirms it is fine.
- **CPU, single rollout ≈ a few seconds.** Throughput is not the point yet; the
  GPU batch story is [5] run on CUDA.

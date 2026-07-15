# Dragonfly trajectory-control GUI

An interactive, browser-based front end for the maneuvering controller of
`report1`. A dragonfly is spawned hovering at the origin in the sagittal
\((x, z)\) plane; you **draw a path with the mouse** and it follows the path at a
chosen speed, then hovers at the end until you draw the next one (which starts
from the body's actual position at the moment you release the mouse).

The physics and controller are the *same* validated code as the report
(`summer_paper/scripts/feasibility.py` force model + the generalized controller
of `generalized_control.py`), re-expressed for live, mutable morphology. Nothing
is reimplemented in the browser — the backend streams the real simulation.

## Run

```sh
PYENV_VERSION=dragonpy pyenv exec python -m summer_paper.gui.server
```

Then open <http://127.0.0.1:8765/>. `Ctrl-C` to stop.

Requires only the project `dragonpy` env (numpy); no web framework or other
third-party packages.

## Use

- **Draw**: press and drag anywhere on the canvas. On release the path is
  smoothed and the dragonfly tracks it, then holds at the end. The next path
  starts from wherever the body actually is when you release.
- **Orange trace**: the actual flown trajectory, drawn beneath the prescribed
  (dashed blue) path for comparison. It persists across successive paths and
  clears only on *Reset*.
- **Left panel**:
  - *Morphology* — wing length \(L_w^*\), inverse wing loading \(A^*/m^*\),
    wingbeat frequency \(\omega^*\) (Table 1 ranges).
  - *Control* — the single outer-loop velocity gain \(K_d\) of the report's
    velocity tracker, \(\vec a^*_\text{des} = K_d(\vec u^*_r - \vec u^*)\).
  - *Trajectory* — follow speed (body lengths per \(\sqrt{L/g}\)), a velocity
    taper (the accel/decel distance over which the follow speed ramps linearly up
    at the path start and back down at the end), and a playback multiplier
    (simulated time per wall-clock second).
  - *Reset* re-spawns hovering at the origin; *Clear path* drops the current
    path.
- **Readouts**: position, speed, advance ratio \(J\), the direction controls
  \((\gamma, \psi_0)\), the magnitude controls \((\phi_1, \psi_1)\), and a status
  chip (hover / following / saturated).

The wing is drawn as a **side-view schematic** on top of the body: a single chord
that sweeps back and forth along the (tilted) stroke-plane line with amplitude
\(\propto \phi_1\) and pitches by \(\psi(t) = \psi_0 + \psi_1\sin(\omega t -
\delta_0)\); a small open circle marks its leading edge. The hollow circle is the
body (point mass).

## Architecture

| file | role |
|------|------|
| `dragonfly_sim.py` | `Config`, the morphology-parametrized controller (γ-schedule, ψ₁-slave, φ₁/ψ₀ trim), path smoothing, and the real-time RK4 `Simulator`. |
| `server.py` | stdlib HTTP server: serves the page, streams render state over Server-Sent Events (`/stream`), accepts `/path`, `/params`, `/reset` POSTs; runs the sim on a background thread paced to the wall clock. |
| `index.html` | single-file front end (canvas + control panel), styled after `reference.html`. |

The controller's outer loop is the report's **pure velocity tracker**: a single
velocity gain, no position feedback, no acceleration feedforward. A finished
path ends with \(v_\text{ref} = 0\), so the body damps to a hover *near* the
endpoint (within its tracking lag) rather than being position-held on it.
The inner allocation is unchanged except for two GUI tweaks: the
inclined-hover stroke-plane *sign* (which way the plane leans) follows the
**commanded** heading rather than the actual velocity, so it does not flip on the
small hover velocity oscillations (the alignment angle and inflow still track the
actual velocity); and once **settled at a hold point** (holding, within
`HOLD_PIN_RADIUS` of the target) γ and ψ₁ are pinned to their fixed hover values
rather than scheduled, so they do not jitter -- a body still moving keeps
scheduling until it settles. The γ-schedule and ψ₁-slave use the
fixed reference advance-ratio scale (`REF_S0 * REF_OMEGA_STAR`), exactly as in
the report scripts, while the inner trim and the integrated dynamics use the live
morphology.

## Runtime behavior

Safe to leave running indefinitely:

- **Idles when unwatched.** The background sim only integrates while at least one
  browser is connected to `/stream`; with no clients it pauses (no CPU/battery
  cost) and freezes its state, so reconnecting resumes seamlessly — an
  in-progress path picks up where it left off (`self.t` doesn't advance while
  paused).
- **Self-healing loop.** The sim step is wrapped so an unhandled error can't
  silently kill the thread: it logs to stderr and auto-resets. If the state goes
  non-finite (e.g. very aggressive gains drive an instability), it is detected
  and auto-reset to a clean hover.
- No history is accumulated server-side (fixed-size state, transient per-frame
  snapshots), so memory stays flat; the browser's trajectory trace is capped and
  decimated client-side. A long sleep/suspend is absorbed by a capped catch-up
  that drops backlog rather than running away.

Localhost only, no auth — don't expose the port publicly.

## Known limits

- Sustained **descent** is force-limited (the report's noted weakness): steep
  downward paths saturate \(\psi_0\) and the body sags off the path. The status
  chip turns red when the trim saturates.
- A reloaded browser tab does not re-fetch an in-progress path (the committed
  path line is client-side); the dragonfly keeps following it regardless. The
  orange trajectory trace is likewise client-side and lost on reload.
- Targets to *pursue* (pure-pursuit prey) are not wired in yet — a natural next
  addition, reusing `simulate_pursuit` from `generalized_control.py`.

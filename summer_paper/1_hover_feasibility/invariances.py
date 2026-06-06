"""
Numerical checks of the dimensional-reduction claims in report1 section 2.

Three claims, three tests against the cycle-averaged-force evaluator:

  (1) |Fbar| is invariant to the stroke-plane angle gamma0.  gamma0 rotates the
      whole averaged vector, so its magnitude is unchanged while its direction
      sweeps -- this is what makes gamma0 the steering knob for maneuvering.

  (2) |Fbar| is invariant to the fore/hind phase shift sigma0.  A constant
      offset is a time shift, and a full-cycle average is shift-invariant.

  (3) The wing length Lw and flapping amplitude phi1 enter |Fbar| only through
      a single excursion combination s0 = (2/3) Lw phi1, so the (Lw, phi1)
      magnitude contours collapse onto level curves of one parameter s0.

(1) and (2) drop two parameters (9 -> 7); (3) drops one more (7 -> 6).

Run:  python summer_paper/1_hover_feasibility/invariances.py
"""

import numpy as np

from feasibility import (
    Params, avg_force_magnitude, cycle_averaged_force, replace,
)


def check_invariance(name, attr, values, base=Params()):
    """Vary one attribute; report the spread of |Fbar| and the direction swing."""
    mags, angles = [], []
    for v in values:
        p = replace(base, **{attr: v})
        F = cycle_averaged_force(p)
        mags.append(float(np.linalg.norm(F)))
        # Tilt of the averaged force from vertical, in the body x-z plane.
        angles.append(float(np.degrees(np.arctan2(F[0], F[2]))))
    mags = np.array(mags)
    spread = mags.max() - mags.min()
    rel = spread / mags.mean() if mags.mean() else 0.0
    print(f"\n{name}: vary {attr} over {len(values)} values")
    print(f"  |Fbar|  mean={mags.mean():.6f}  spread={spread:.2e}  "
          f"(relative {rel:.2e})")
    print(f"  force-vector tilt ranges over "
          f"[{min(angles):+.1f}, {max(angles):+.1f}] deg")
    verdict = "INVARIANT" if rel < 1e-3 else "NOT invariant"
    print(f"  -> magnitude {verdict}")
    return rel


def check_collapse(base=Params(), n=25):
    """Map |Fbar| over an (Lw, phi1) grid and test a one-parameter collapse.

    Hypothesis: |Fbar| depends on (Lw, phi1) only through an excursion amplitude
    s0 = (2/3) Lw phi1 (the arc-length sweep of the 2/3-span station; report
    definition). If so, the scattered (s0, |Fbar|) points fall on a single curve.
    We quantify that by binning in s0 and comparing the within-bin scatter to the
    overall scatter.
    """
    Lw_vals = np.linspace(0.65, 0.85, n)
    phi1_vals = np.radians(np.linspace(5.0, 60.0, n))

    s0, mag = [], []
    for Lw in Lw_vals:
        for phi1 in phi1_vals:
            p = replace(base, Lw=Lw, phi1=phi1)
            s0.append((2.0 / 3.0) * Lw * phi1)
            mag.append(avg_force_magnitude(p))
    s0 = np.array(s0)
    mag = np.array(mag)

    # Sort by s0, bin into narrow s0 slices, measure residual scatter of |Fbar|
    # within each slice relative to the total range.
    order = np.argsort(s0)
    s0_s, mag_s = s0[order], mag[order]
    n_bins = 20
    edges = np.linspace(s0_s.min(), s0_s.max(), n_bins + 1)
    within = []
    for i in range(n_bins):
        m = (s0_s >= edges[i]) & (s0_s <= edges[i + 1])
        if m.sum() >= 2:
            within.append(mag_s[m].std())
    within_scatter = float(np.mean(within))
    total_range = float(mag.max() - mag.min())
    ratio = within_scatter / total_range

    print(f"\nCollapse of (Lw, phi1) onto s0 = (2/3) Lw phi1:")
    print(f"  grid {n}x{n}, |Fbar| range [{mag.min():.4f}, {mag.max():.4f}]")
    print(f"  mean within-s0-bin scatter = {within_scatter:.4e}")
    print(f"  fraction of total range    = {ratio:.2e}")
    verdict = "collapses well" if ratio < 0.05 else "does NOT collapse cleanly"
    print(f"  -> (Lw, phi1) {verdict} onto a single s0")
    return ratio


def main():
    print("=" * 70)
    print("Dimensional-reduction checks for hover feasibility (report1 sec. 2)")
    print("=" * 70)
    base = Params()
    print(f"\nbaseline |Fbar| = {avg_force_magnitude(base):.4f} "
          f"(weight = 1.0)")

    check_invariance("(1) stroke-plane angle gamma0", "gamma0",
                     np.radians(np.linspace(-180, 180, 13)), base)
    check_invariance("(2) fore/hind phase sigma0", "sigma0",
                     np.radians(np.linspace(-180, 180, 13)), base)
    check_collapse(base)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

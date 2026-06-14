"""
Reference planar equations of motion (user-provided), integrated directly.

State: x, z (world, z up), theta (pitch from vertical), and rates.
Wing: point mass m_w on the pitch axis, translating along the stroke axis with
s = s0 sin(wt); h = root COM offset along the body x-axis. The stroke axis in
world is (cos th, sin th); the body longitudinal (h) axis is (-sin th, cos th).

Aero: C_L = C_L0 sin 2a, C_D = C_D0 cos^2 a + C_D90 sin^2 a, with angle of attack
a from the feather (square or sinusoidal) and the wing-relative wind v_w. This is
the one piece not in the reference image -- flagged for confirmation.

Params are the AGGREGATE single wing for the 4-wing emulation:
    A/m_b = 4*0.006 = 0.024,  m_w/m_b = 4*0.03 = 0.12,  I_b/m_b = 0.0915,
    h = 0.1, s0 = 1.5, omega = 11, psi1 = 45 deg, delta0 = pi/2, g = 1.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CL0, CD0, CD90 = 1.5, 0.1, 2.0
AoverM = 4 * 0.006
MWB = 0.03                # aggregate m_w/m_b (corrected)
IBMB = 0.0915
H, S0, OMEGA = 0.1, 1.5, 11.0
PSI1, DELTA0, G = np.radians(45.0), np.pi / 2, 1.0


def deriv(t, y, square):
    x, z, th, xd, zd, thd = y
    s = S0 * np.sin(OMEGA * t)
    sd = S0 * OMEGA * np.cos(OMEGA * t)
    sdd = -S0 * OMEGA ** 2 * np.sin(OMEGA * t)
    ct, st = np.cos(th), np.sin(th)

    xwd = xd + sd * ct - s * thd * st - H * thd * ct
    zwd = zd + sd * st + s * thd * ct - H * thd * st
    vw = np.hypot(xwd, zwd) + 1e-12

    # angle of attack from feather (stroke axis u=(ct,st), normal n=(-st,ct))
    a = xwd * ct + zwd * st
    b = -xwd * st + zwd * ct
    fp = np.sin(OMEGA * t + DELTA0)
    f = PSI1 * (np.sign(fp) if square else fp)
    alpha = np.arctan2(a * np.sin(f) - b * np.cos(f), a * np.cos(f) + b * np.sin(f))
    CL = CL0 * np.sin(2 * alpha)
    CD = CD0 * np.cos(alpha) ** 2 + CD90 * np.sin(alpha) ** 2

    aero1 = 0.5 * AoverM * vw * (-CL * zwd - CD * xwd)
    aero2 = 0.5 * AoverM * vw * (CL * xwd - CD * zwd)
    K1 = aero1 - MWB * (sdd * ct - 2 * sd * thd * st - s * thd ** 2 * ct + H * thd ** 2 * st)
    K2 = aero2 - MWB * (sdd * st + 2 * sd * thd * ct - s * thd ** 2 * st - H * thd ** 2 * ct)

    den = IBMB * (1 + MWB) + MWB * (s ** 2 + H ** 2)
    thdd = ((s * ct - H * st) * K2 - (s * st + H * ct) * K1) / den
    xdd = (K1 + MWB * (s * st + H * ct) * thdd) / (1 + MWB)
    zdd = (K2 - MWB * (s * ct - H * st) * thdd) / (1 + MWB) - G
    return np.array([xd, zd, thd, xdd, zdd, thdd])


def integrate(square, n_wb=100, spc=400):
    dt = (2 * np.pi / OMEGA) / spc
    n = int(n_wb * spc)
    y = np.zeros(6)                      # start vertical at rest
    Y = np.empty((n + 1, 6)); Y[0] = y
    t = 0.0
    for i in range(n):
        k1 = deriv(t, y, square)
        k2 = deriv(t + dt / 2, y + dt / 2 * k1, square)
        k3 = deriv(t + dt / 2, y + dt / 2 * k2, square)
        k4 = deriv(t + dt, y + dt * k3, square)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        Y[i + 1] = y
    return Y, spc


def cycavg_Fz(square, spc=400):
    """cycle-avg aero z-force at rest, body vertical -> F/W check."""
    dt = (2 * np.pi / OMEGA) / spc
    Fz = 0.0
    for i in range(spc):
        t = i * dt
        s = S0 * np.sin(OMEGA * t); sd = S0 * OMEGA * np.cos(OMEGA * t)
        xwd, zwd = sd, 0.0; vw = abs(sd) + 1e-12
        a, b = xwd, 0.0
        fp = np.sin(OMEGA * t + DELTA0); f = PSI1 * (np.sign(fp) if square else fp)
        al = np.arctan2(a * np.sin(f) - b * np.cos(f), a * np.cos(f) + b * np.sin(f))
        CL = CL0 * np.sin(2 * al)
        Fz += 0.5 * AoverM * vw * (CL * xwd)
    return Fz / spc


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    NWB = 50
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax, square in zip(axes, [False, True]):
        Y, spc = integrate(square, n_wb=NWB)
        x, z, th = Y[:, 0], Y[:, 1], np.degrees(Y[:, 2])
        W = G * (1 + MWB)
        ax.plot(x, z, lw=0.8, color="0.5")
        sc = ax.scatter(x, z, c=np.arange(len(x)) / spc, cmap="viridis", s=2)
        ax.plot(0, 0, "ko", ms=5)
        ax.plot(x[-1], z[-1], "r*", ms=12)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x"); ax.set_ylabel("z (up)")
        lab = "square" if square else "sinusoidal"
        ax.set_title(f"{lab} pitch, {NWB} wingbeats\n"
                     f"end (x,z,theta) = ({x[-1]:.2f}, {z[-1]:.2f}, {th[-1]:.1f} deg)")
        print(f"{lab:10s}: end x={x[-1]:+8.3f}  z={z[-1]:+8.3f}  theta={th[-1]:+7.2f} deg  "
              f"| F/W={cycavg_Fz(square)/W:.3f}  max|tilt|={np.abs(th).max():.1f} deg")
    fig.colorbar(sc, ax=axes, label="wingbeats", shrink=0.7)
    out = HERE / "reference_eom_trajectory.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

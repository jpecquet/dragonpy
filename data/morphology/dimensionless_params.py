"""
Dimensionless morphology parameters for Anisoptera (dragonflies).

Combines two digitized datasets to extract three dimensionless groups that
characterize dragonfly flapping flight:

  1. lambda = L_wing / L_body          (slenderness; forewing + hindwing)
       wing length R over body length L.

  2. mu = A_wing * rho_air * L_body / m_body   (added-air-mass ratio; fore + hind)
       the mass of a column of air of cross-section A_wing and height L_body,
       relative to body mass. Note this uses BODY length L_body, not wing
       length -- so mu = lambda * (rho_air * A_wing * R / m), differing from the
       added-mass group built on wing length used elsewhere in the repo.

  3. Omega = omega_wingbeat * sqrt(L_body / g)   (gravity-normalized wingbeat)
       angular wingbeat frequency omega = 2*pi*f non-dimensionalized by the
       gravitational time sqrt(L_body / g). Under a resonance hypothesis
       (omega ~ 1/sqrt(L)) this group is constant across body size.

Sources
-------
lambda, mu : Wakeling 1997, per-specimen body + fore/hind wing morphology.
Omega      : Aracheloff 2024, digitized (body length, flapping frequency) cloud.

The two datasets are not specimen-matched, so Omega is reported as its own
population; lambda and mu are reported per specimen.

Outputs dark-theme figures to docs/_static/media/morphology/ and prints a
summary table of the dimensionless values.

Runs on the project env (numpy + matplotlib only; no pandas/scipy).
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from post.style import apply_matplotlib_style, resolve_style  # noqa: E402

# Physical constants (SI).
RHO_AIR = 1.205      # kg / m^3, sea-level air density (matches read_data.py)
G = 9.80665          # m / s^2

WAKELING_DIR = HERE / "wakeling1997"
ARACHELOFF_CSV = HERE / "aracheloff2024" / "flapping_freq_vs_body_length.csv"
OUT_DIR = REPO_ROOT / "docs" / "_static" / "media" / "morphology"

# Calopteryx splendens (CS*) is a damselfly (Zygoptera), not a dragonfly
# (Anisoptera); excluded to keep the population to true dragonflies, matching
# the existing wakeling1997/read_data.py loader.
EXCLUDE_PREFIX = "CS"


# ---------------------------------------------------------------------------
# Data loading (csv module; missing cells -> NaN).

def _to_float(token):
    token = token.strip()
    if token == "":
        return np.nan
    return float(token)


def _read_csv(path):
    """Return {column_name: np.ndarray} with the ID column kept as a list."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    cols = {h: [] for h in header}
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue  # skip blank trailing lines
        for h, cell in zip(header, row):
            cols[h].append(cell)
    out = {}
    for h, values in cols.items():
        if h in ("species", "sex", "ID"):
            out[h] = values
        else:
            out[h] = np.array([_to_float(v) for v in values])
    return out


def load_specimens():
    """
    Merge body + fore + hind tables on specimen ID, in SI units.

    Returns a dict of arrays (one entry per specimen present in all three
    tables, excluding damselflies). Wing fields may be NaN where the source
    cell was blank; the caller masks per parameter.
    """
    body = _read_csv(WAKELING_DIR / "body_data.csv")
    fore = _read_csv(WAKELING_DIR / "forewing_data.csv")
    hind = _read_csv(WAKELING_DIR / "hindwing_data.csv")

    # Index fore/hind by ID (first occurrence wins; a duplicate CS row exists).
    def index_by_id(table):
        idx = {}
        for i, sid in enumerate(table["ID"]):
            idx.setdefault(sid, i)
        return idx

    fore_idx = index_by_id(fore)
    hind_idx = index_by_id(hind)

    rec = {k: [] for k in
           ("ID", "species", "L", "m", "R_f", "S_f", "R_h", "S_h")}

    for i, sid in enumerate(body["ID"]):
        if sid.startswith(EXCLUDE_PREFIX):
            continue
        if sid not in fore_idx or sid not in hind_idx:
            continue
        fi, hi = fore_idx[sid], hind_idx[sid]
        rec["ID"].append(sid)
        rec["species"].append(body["species"][i])
        rec["L"].append(body["L"][i] * 1e-3)        # mm -> m
        rec["m"].append(body["m"][i] * 1e-6)        # mg -> kg
        rec["R_f"].append(fore["R"][fi] * 1e-3)     # mm -> m
        rec["S_f"].append(fore["S"][fi] * 1e-6)     # mm^2 -> m^2
        rec["R_h"].append(hind["R"][hi] * 1e-3)
        rec["S_h"].append(hind["S"][hi] * 1e-6)

    rec["species"] = np.array(rec["species"], dtype=object)
    rec["ID"] = np.array(rec["ID"], dtype=object)
    for k in ("L", "m", "R_f", "S_f", "R_h", "S_h"):
        rec[k] = np.array(rec[k], dtype=float)
    return rec


def load_frequency():
    """Aracheloff cloud: (L_body [m], f [Hz])."""
    data = np.genfromtxt(ARACHELOFF_CSV, delimiter=",")
    data = data[np.isfinite(data).all(axis=1)]
    L = data[:, 0] * 1e-3     # mm -> m
    f = data[:, 1]            # Hz
    return L, f


# ---------------------------------------------------------------------------
# Dimensionless parameters.

def slope_through_origin(x, y):
    """Least-squares slope of y = a*x (no intercept), ignoring NaNs."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    return float(np.dot(x, y) / np.dot(x, x))


def compute_parameters(rec, L_freq, f_freq):
    lam_f = rec["R_f"] / rec["L"]
    lam_h = rec["R_h"] / rec["L"]
    mu_f = RHO_AIR * rec["S_f"] * rec["L"] / rec["m"]
    mu_h = RHO_AIR * rec["S_h"] * rec["L"] / rec["m"]

    omega = 2.0 * np.pi * f_freq
    Omega = omega * np.sqrt(L_freq / G)

    return {
        "lam_f": lam_f, "lam_h": lam_h,
        "mu_f": mu_f, "mu_h": mu_h,
        "Omega": Omega,
        # population-level regression slopes (alternative point estimates)
        "lam_f_fit": slope_through_origin(rec["L"], rec["R_f"]),
        "lam_h_fit": slope_through_origin(rec["L"], rec["R_h"]),
        "mu_f_fit": slope_through_origin(rec["m"], RHO_AIR * rec["S_f"] * rec["L"]),
        "mu_h_fit": slope_through_origin(rec["m"], RHO_AIR * rec["S_h"] * rec["L"]),
    }


def print_summary(rec, params, L_freq, f_freq):
    def stat(name, arr):
        arr = arr[np.isfinite(arr)]
        print(f"  {name:24s} n={arr.size:2d}  "
              f"mean={arr.mean():7.3f}  std={arr.std(ddof=1):6.3f}  "
              f"[{arr.min():.3f}, {arr.max():.3f}]")

    print("=" * 70)
    print(f"Dragonfly dimensionless morphology  (Anisoptera, n={rec['L'].size} specimens)")
    print("=" * 70)
    print("\n1) lambda = L_wing / L_body")
    stat("lambda forewing", params["lam_f"])
    stat("lambda hindwing", params["lam_h"])
    print(f"     regression slopes (R vs L, through origin): "
          f"fore={params['lam_f_fit']:.3f}  hind={params['lam_h_fit']:.3f}")

    print("\n2) mu = rho_air * A_wing * L_body / m_body")
    stat("mu forewing", params["mu_f"])
    stat("mu hindwing", params["mu_h"])
    print(f"     regression slopes (rho*S*L vs m, through origin): "
          f"fore={params['mu_f_fit']:.3f}  hind={params['mu_h_fit']:.3f}")

    print("\n3) Omega = omega_wingbeat * sqrt(L_body / g)   [omega = 2*pi*f]")
    stat("Omega", params["Omega"])
    print(f"     (Aracheloff cloud, n={f_freq.size}: "
          f"f in [{f_freq.min():.0f}, {f_freq.max():.0f}] Hz, "
          f"L in [{L_freq.min()*1e3:.0f}, {L_freq.max()*1e3:.0f}] mm)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Correlation between lambda and mu (per specimen, fore and hind separately).

def _fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _ranks(a):
    """Ordinal ranks (ties broken arbitrarily; data are continuous floats)."""
    return a.argsort().argsort().astype(float)


def correlate(x, y, method="pearson"):
    """
    Correlation of x and y over jointly finite samples.

    Returns dict with r, n, 95% CI and a two-sided p-value, both from the
    Fisher z-transform (normal approximation -- exact enough for n ~ 30 and
    keeps the script free of scipy). Spearman runs Pearson on ranks.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    if method == "spearman":
        x, y = _ranks(x), _ranks(y)
    r = float(np.corrcoef(x, y)[0, 1])

    # Fisher z-transform: z ~ Normal(arctanh(rho), 1/(n-3)).
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    ci = (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))
    p = math.erfc(abs(z) * math.sqrt(n - 3) / math.sqrt(2.0))
    return {"r": r, "n": n, "ci": ci, "p": p, "method": method}


def correlation_stats(params):
    return {
        "fore_pearson": correlate(params["lam_f"], params["mu_f"], "pearson"),
        "fore_spearman": correlate(params["lam_f"], params["mu_f"], "spearman"),
        "hind_pearson": correlate(params["lam_h"], params["mu_h"], "pearson"),
        "hind_spearman": correlate(params["lam_h"], params["mu_h"], "spearman"),
    }


def print_correlation(stats):
    print("\nCorrelation  lambda vs mu  (per specimen)")
    print("-" * 70)

    def line(name, s):
        lo, hi = s["ci"]
        print(f"  {name:22s} r={s['r']:+.3f}  95%CI=[{lo:+.2f}, {hi:+.2f}]  "
              f"{_fmt_p(s['p'])}  n={s['n']}  ({s['method']})")

    line("forewing (Pearson)", stats["fore_pearson"])
    line("forewing (Spearman)", stats["fore_spearman"])
    line("hindwing (Pearson)", stats["hind_pearson"])
    line("hindwing (Spearman)", stats["hind_spearman"])
    print("-" * 70)


# ---------------------------------------------------------------------------
# Visualization.

def make_correlation_figure(params, stats, style, out_path):
    fore_c = style.trajectory_color
    hind_c = style.error_color

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)

    def panel(a, lam, mu, color, label, s):
        mask = np.isfinite(lam) & np.isfinite(mu)
        x, y = lam[mask], mu[mask]
        a.scatter(x, y, s=30, facecolors="none", edgecolors=color, linewidths=1.4)
        # OLS fit mu = b0 + b1*lam, drawn across the observed lambda range.
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        a.plot(xs, b0 + b1 * xs, "-", color=color, lw=1.5)
        a.set_title(label)
        a.set_xlabel(r"$\lambda = L_\mathrm{wing}/L_\mathrm{body}$")
        lo, hi = s["ci"]
        a.text(0.04, 0.93,
               rf"$r={s['r']:+.2f}$" + "\n"
               rf"95%CI $[{lo:+.2f},{hi:+.2f}]$" + "\n"
               + _fmt_p(s["p"]) + rf", $n={s['n']}$",
               transform=a.transAxes, va="top", ha="left",
               color=color, fontsize=10)

    panel(ax[0], params["lam_f"], params["mu_f"], fore_c, "forewing",
          stats["fore_pearson"])
    panel(ax[1], params["lam_h"], params["mu_h"], hind_c, "hindwing",
          stats["hind_pearson"])
    ax[0].set_ylabel(r"$\mu = \rho_\mathrm{air} A_\mathrm{wing} L_\mathrm{body}/m_\mathrm{body}$")

    fig.suptitle(r"Correlation of $\lambda$ and $\mu$ across specimens", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return fig


def make_figure(rec, params, L_freq, f_freq, style, out_path):
    fore_c = style.trajectory_color
    hind_c = style.error_color
    omg_c = style.target_color
    muted = style.muted_text_color

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.6))

    # ----- Column 1: lambda = R / L --------------------------------------
    L_mm = rec["L"] * 1e3
    a = ax[0, 0]
    a.scatter(L_mm, rec["R_f"] * 1e3, s=28, facecolors="none",
              edgecolors=fore_c, linewidths=1.4, label="forewing")
    a.scatter(L_mm, rec["R_h"] * 1e3, s=28, marker="s", facecolors="none",
              edgecolors=hind_c, linewidths=1.4, label="hindwing")
    lam_f_m, lam_f_s = np.nanmean(params["lam_f"]), np.nanstd(params["lam_f"], ddof=1)
    lam_h_m, lam_h_s = np.nanmean(params["lam_h"]), np.nanstd(params["lam_h"], ddof=1)
    L_hi = np.nanmax(L_mm) * 1.05
    xline = np.array([0.0, L_hi])
    a.plot(xline, lam_f_m * xline, "-", color=fore_c, lw=1.5)
    a.plot(xline, lam_h_m * xline, "--", color=hind_c, lw=1.5)
    a.set_xlim(0, L_hi)
    a.set_ylim(0, max(np.nanmax(rec["R_f"]), np.nanmax(rec["R_h"])) * 1e3 * 1.1)
    a.set_xlabel(r"$L_\mathrm{body}$  (mm)")
    a.set_ylabel(r"$L_\mathrm{wing}=R$  (mm)")
    a.set_title(r"(1)  $\lambda = L_\mathrm{wing}/L_\mathrm{body}$")
    a.legend(loc="upper left", frameon=False, fontsize=10)
    a.text(0.97, 0.10, rf"$\lambda_\mathrm{{fore}}={lam_f_m:.2f}\pm{lam_f_s:.2f}$",
           transform=a.transAxes, ha="right", color=fore_c, fontsize=11)
    a.text(0.97, 0.02, rf"$\lambda_\mathrm{{hind}}={lam_h_m:.2f}\pm{lam_h_s:.2f}$",
           transform=a.transAxes, ha="right", color=hind_c, fontsize=11)

    # ----- Column 2: mu = rho_air * S * L / m ----------------------------
    m_g = rec["m"] * 1e3
    air_f_g = RHO_AIR * rec["S_f"] * rec["L"] * 1e3
    air_h_g = RHO_AIR * rec["S_h"] * rec["L"] * 1e3
    a = ax[0, 1]
    a.scatter(m_g, air_f_g, s=28, facecolors="none",
              edgecolors=fore_c, linewidths=1.4, label="forewing")
    a.scatter(m_g, air_h_g, s=28, marker="s", facecolors="none",
              edgecolors=hind_c, linewidths=1.4, label="hindwing")
    mu_f_m, mu_f_s = np.nanmean(params["mu_f"]), np.nanstd(params["mu_f"], ddof=1)
    mu_h_m, mu_h_s = np.nanmean(params["mu_h"]), np.nanstd(params["mu_h"], ddof=1)
    m_hi = np.nanmax(m_g) * 1.05
    xline = np.array([0.0, m_hi])
    a.plot(xline, mu_f_m * xline, "-", color=fore_c, lw=1.5)
    a.plot(xline, mu_h_m * xline, "--", color=hind_c, lw=1.5)
    a.set_xlim(0, m_hi)
    a.set_ylim(0, max(np.nanmax(air_f_g), np.nanmax(air_h_g)) * 1.15)
    a.set_xlabel(r"$m_\mathrm{body}$  (g)")
    a.set_ylabel(r"$\rho_\mathrm{air}\,A_\mathrm{wing}\,L_\mathrm{body}$  (g)")
    a.set_title(r"(2)  $\mu = \rho_\mathrm{air} A_\mathrm{wing} L_\mathrm{body}/m_\mathrm{body}$")
    a.legend(loc="upper left", frameon=False, fontsize=10)
    a.text(0.97, 0.10, rf"$\mu_\mathrm{{fore}}={mu_f_m:.2f}\pm{mu_f_s:.2f}$",
           transform=a.transAxes, ha="right", color=fore_c, fontsize=11)
    a.text(0.97, 0.02, rf"$\mu_\mathrm{{hind}}={mu_h_m:.2f}\pm{mu_h_s:.2f}$",
           transform=a.transAxes, ha="right", color=hind_c, fontsize=11)

    # ----- Column 3: Omega = 2*pi*f * sqrt(L/g) --------------------------
    a = ax[0, 2]
    L_freq_mm = L_freq * 1e3
    Omega = params["Omega"]
    Om_mean = Omega.mean()
    Om_std = Omega.std(ddof=1)
    a.scatter(L_freq_mm, Omega, s=28, facecolors="none",
              edgecolors=omg_c, linewidths=1.4)
    a.axhline(Om_mean, color=omg_c, lw=1.5)
    a.axhspan(Om_mean - Om_std, Om_mean + Om_std, color=omg_c, alpha=0.15)
    a.set_xlim(L_freq_mm.min() - 2, L_freq_mm.max() + 2)
    a.set_ylim(0, Omega.max() * 1.15)
    a.set_xlabel(r"$L_\mathrm{body}$  (mm)")
    a.set_ylabel(r"$\Omega = \omega_\mathrm{wb}\sqrt{L_\mathrm{body}/g}$")
    a.set_title(r"(3)  $\Omega = \omega_\mathrm{wb}\sqrt{L_\mathrm{body}/g}$")
    a.text(0.97, 0.06,
           rf"$\overline{{\Omega}}={Om_mean:.1f}\pm{Om_std:.1f}$",
           transform=a.transAxes, ha="right", color=omg_c, fontsize=11)

    # ----- Bottom row: distributions of the dimensionless values ---------
    def strip(a, arrays, colors, labels):
        clean = [arr[np.isfinite(arr)] for arr in arrays]
        positions = np.arange(1, len(clean) + 1)
        bp = a.boxplot(clean, positions=positions, widths=0.5,
                       patch_artist=True, showfliers=False)
        for i, (arr, c) in enumerate(zip(clean, colors)):
            _style_box_single(bp, i, c)
            jitter = (rng.random(arr.size) - 0.5) * 0.18
            a.scatter(np.full(arr.size, positions[i]) + jitter, arr,
                      s=16, facecolors="none", edgecolors=c, linewidths=1.0,
                      alpha=0.8, zorder=3)
        a.set_xticks(positions)
        a.set_xticklabels(labels)

    strip(ax[1, 0], [params["lam_f"], params["lam_h"]],
          [fore_c, hind_c], ["fore", "hind"])
    ax[1, 0].set_ylabel(r"$\lambda$")
    ax[1, 0].set_title(r"$\lambda$  per specimen", color=muted, fontsize=11)

    strip(ax[1, 1], [params["mu_f"], params["mu_h"]],
          [fore_c, hind_c], ["fore", "hind"])
    ax[1, 1].set_ylabel(r"$\mu$")
    ax[1, 1].set_title(r"$\mu$  per specimen", color=muted, fontsize=11)

    strip(ax[1, 2], [params["Omega"]], [omg_c], [r"all"])
    ax[1, 2].set_ylabel(r"$\Omega$")
    ax[1, 2].set_title(r"$\Omega$  population", color=muted, fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return fig


def _style_box_single(bp, i, color):
    bp["boxes"][i].set_facecolor(color)
    bp["boxes"][i].set_alpha(0.25)
    bp["boxes"][i].set_edgecolor(color)
    bp["medians"][i].set_color(color)
    bp["medians"][i].set_linewidth(2.0)
    for j in (2 * i, 2 * i + 1):
        bp["whiskers"][j].set_color(color)
        bp["caps"][j].set_color(color)


def main():
    style = resolve_style(theme="dark")
    apply_matplotlib_style(style)

    rec = load_specimens()
    L_freq, f_freq = load_frequency()
    params = compute_parameters(rec, L_freq, f_freq)

    print_summary(rec, params, L_freq, f_freq)
    stats = correlation_stats(params)
    print_correlation(stats)

    make_figure(rec, params, L_freq, f_freq, style,
                OUT_DIR / "dimensionless_params.dark.png")
    make_correlation_figure(params, stats, style,
                            OUT_DIR / "lambda_mu_correlation.dark.png")


if __name__ == "__main__":
    main()

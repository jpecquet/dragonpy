"""
Generate `docs/research/capture_stationary.md` and the single-wedge PNGs
that back it.

The page has four sweep sections (Gain, Feather Amplitude, Aero Ratio,
Span Ratio). Each section uses the same nested-tab layout:

  outer tab-set : the swept parameter
    inner tab-set : initial stroke-plane tilt (12 values, signed degrees)
      grid 2 : [avg-over-tilts wedge] | [per-tilt wedge with normal arrow]

PNGs are emitted dark-only at the layout's compact size (3" wide, H/W=1.1).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.capture_plot import load_result, plot_half_disk
from examples.capture_study import WING_FREQUENCY
from examples.capture_trajectory_plot import render_sweep_trajectory
from dragonpy.studies import (
    capture_rate, captured_mask, time_to_intercept,
)

T_WB = 1.0 / WING_FREQUENCY


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "capture"
OUT_DIR   = REPO_ROOT / "docs" / "_static" / "media" / "capture"
DOCS_PAGE = REPO_ROOT / "docs" / "research" / "capture_stationary.md"

CAPTURE_RADIUS = 0.5
PLOT_WIDTH_IN  = 3.0
PLOT_RATIO     = 1.1

# Baseline study: k=3, feather=30 deg, aero=0.025, span=0.75. Reused as the
# "1x" reference value in every sweep that includes it.
BASELINE_DIR = "k3_tau1Twb_hind0.5pi_fov360"


@dataclass
class SweepValue:
    label:    str   # tab label (markdown)
    file_tag: str   # filename infix
    data_dir: str   # data/capture/<this>/result.npz
    overrides: dict = None  # kwargs that override BASELINE_KWARGS in make_setup

    def __post_init__(self):
        if self.overrides is None:
            self.overrides = {}


@dataclass
class Sweep:
    name:          str   # filename prefix
    section_title: str   # markdown header
    param_header:  str   # column header for the swept parameter in the table
    values:        list[SweepValue]


SWEEPS: list[Sweep] = [
    Sweep(
        name="gain_sweep",
        section_title=r"$K$ Sweep",
        param_header=r"$K$",
        values=[
            SweepValue(f"{k}", f"K{k}",
                       f"k{k}_tau1Twb_hind0.5pi_fov360",
                       {"k_tilt": float(k)})
            for k in (1, 2, 3, 4)
        ],
    ),
    Sweep(
        name="feather_sweep",
        section_title=r"$\Delta \psi$ Sweep",
        param_header=r"$\Delta \psi$",
        values=[
            SweepValue("15°", "F15",
                       "k3_tau1Twb_hind0.5pi_fov360_feath15deg",
                       {"intercept_feather_amp": np.radians(15.0)}),
            SweepValue("30°", "F30", BASELINE_DIR, {}),
            SweepValue("45°", "F45",
                       "k3_tau1Twb_hind0.5pi_fov360_feath45deg",
                       {"intercept_feather_amp": np.radians(45.0)}),
            SweepValue("60°", "F60",
                       "k3_tau1Twb_hind0.5pi_fov360_feath60deg",
                       {"intercept_feather_amp": np.radians(60.0)}),
        ],
    ),
    Sweep(
        name="aero_sweep",
        section_title=r"$\mu_0 = \rho L_w A_w / m$ Sweep",
        param_header=r"$\mu_0$",
        values=[
            SweepValue("0.0125", "A0125",
                       "k3_tau1Twb_hind0.5pi_fov360_feath30deg_aero0125",
                       {"aero_ratio": 0.0125}),
            SweepValue("0.025",  "A0250", BASELINE_DIR, {}),
            SweepValue("0.05",   "A0500",
                       "k3_tau1Twb_hind0.5pi_fov360_feath30deg_aero0500",
                       {"aero_ratio": 0.05}),
            SweepValue("0.10",   "A1000",
                       "k3_tau1Twb_hind0.5pi_fov360_feath30deg_aero1000",
                       {"aero_ratio": 0.10}),
        ],
    ),
    Sweep(
        name="span_sweep",
        section_title=r"$\lambda_0 = L_w / L$ Sweep",
        param_header=r"$\lambda_0$",
        values=[
            SweepValue("0.5",  "S0500",
                       "k3_tau1Twb_hind0.5pi_fov360_feath30deg_span0500",
                       {"span_ratio": 0.5}),
            SweepValue("0.75", "S0750", BASELINE_DIR, {}),
            SweepValue("1.0",  "S1000",
                       "k3_tau1Twb_hind0.5pi_fov360_feath30deg_span1000",
                       {"span_ratio": 1.0}),
        ],
    ),
]


CAPTURE_RADII_REPORTED = (0.2, 0.5, 1.0)


def _signed_tilts(tilts: np.ndarray) -> list[tuple[int, float]]:
    """Convert a (0, 2pi) tilt grid into [(signed_deg, t_rad), ...] sorted by
    signed degree ascending. Maps angles > pi to deg - 360 so the range is
    (-180, 180]."""
    pairs: list[tuple[int, float]] = []
    for t in tilts:
        deg = int(round(np.degrees(float(t))))
        signed = deg if deg <= 180 else deg - 360
        pairs.append((signed, float(t)))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _avg_name(sweep: Sweep, value: SweepValue) -> str:
    return f"{sweep.name}_{value.file_tag}_avg_rcap{CAPTURE_RADIUS:g}.dark.png"


def _tilt_name(sweep: Sweep, value: SweepValue, signed_deg: int) -> str:
    return (f"{sweep.name}_{value.file_tag}_tilt{signed_deg:+04d}"
            f"_rcap{CAPTURE_RADIUS:g}.dark.png")


def _result_path(value: SweepValue) -> Path:
    return DATA_ROOT / value.data_dir / "result.npz"


def render_sweep_pngs(sweep: Sweep) -> None:
    for v in sweep.values:
        result = load_result(_result_path(v))

        out = OUT_DIR / _avg_name(sweep, v)
        plot_half_disk(
            result, capture_radius=CAPTURE_RADIUS, tilt=None,
            theme="dark", title="", savepath=out,
            show_colorbar=True,
            width_in=PLOT_WIDTH_IN, height_over_width=PLOT_RATIO,
        )
        print(f"saved {out.name}")

        for signed_deg, t_rad in _signed_tilts(result.tilts):
            out = OUT_DIR / _tilt_name(sweep, v, signed_deg)
            plot_half_disk(
                result, capture_radius=CAPTURE_RADIUS, tilt=t_rad,
                theme="dark", title="", savepath=out,
                show_colorbar=False, show_tilt_arrow=True,
                width_in=PLOT_WIDTH_IN, height_over_width=PLOT_RATIO,
            )
            print(f"saved {out.name}")


def _img_block(rel_src: str, alt: str) -> str:
    return (
        '```{raw} html\n'
        '<img\n'
        '  class="case-study-image"\n'
        f'  src="{rel_src}"\n'
        f'  alt="{alt}"\n'
        '/>\n'
        '```\n'
    )


def render_table(sweep: Sweep) -> str:
    """Markdown summary table: capture rate at three rcap thresholds plus
    mean intercept time at rcap=0.5, one row per swept value."""
    rcaps = CAPTURE_RADII_REPORTED
    header = (
        f"| {sweep.param_header} "
        + "".join(fr"| $r_\mathrm{{cap}} = {r:.1f}$ " for r in rcaps)
        + r"| $\langle t_\mathrm{int} \rangle / T_\mathrm{wb}$ |"
    )
    sep = "|---" + "|---:" * (len(rcaps) + 1) + "|"

    rows: list[str] = []
    for v in sweep.values:
        result = load_result(_result_path(v))
        rate_cells = []
        for r in rcaps:
            rate = capture_rate(result, r)
            rate_cells.append(f"{rate * 100.0:.0f}%")
        ttis = time_to_intercept(result, 0.5)
        n_cap = int(captured_mask(result, 0.5).sum())
        mean_t = float(np.nanmean(ttis)) if n_cap else float("nan")
        t_str = f"{mean_t / T_WB:.1f}" if n_cap else "—"
        rows.append(
            f"| {v.label} | "
            + " | ".join(rate_cells)
            + f" | {t_str} |"
        )

    return "\n".join([header, sep, *rows]) + "\n\n"


def _trajectory_img_block(sweep: Sweep) -> str:
    rel_src = f"../_static/media/capture/{sweep.name}_traj.dark.png"
    return _img_block(
        rel_src,
        f"Representative trajectories: {sweep.section_title}",
    )


def render_section(sweep: Sweep, signed_degs: list[int]) -> str:
    lines: list[str] = []
    lines.append(f"## {sweep.section_title}\n")
    lines.append("\n")
    lines.append(render_table(sweep))

    # Gain (8 colons) -> tilt (6) -> grid (4) -> grid-item (3).
    lines.append("::::::::{tab-set}\n")
    for v in sweep.values:
        lines.append(f":::::::{{tab-item}} {v.label}\n")
        lines.append("::::::{tab-set}\n")
        avg_src = f"../_static/media/capture/{_avg_name(sweep, v)}"
        for deg in signed_degs:
            lines.append(f":::::{{tab-item}} {deg:+d}°\n")
            # :sync: key shared across every inner tab-set on the page so
            # the tilt selection persists when the outer (parameter) tab is
            # switched. Blank line separates the directive options from the
            # body content.
            lines.append(f":sync: tilt{deg:+04d}\n")
            lines.append("\n")
            lines.append("::::{grid} 2\n")
            lines.append(":gutter: 3\n")
            lines.append("\n")

            lines.append(":::{grid-item}\n")
            lines.append(_img_block(
                avg_src,
                f"Capture map averaged over initial stroke-plane tilts "
                f"({sweep.section_title}, {v.label})",
            ))
            lines.append(":::\n")
            lines.append("\n")

            tilt_src = f"../_static/media/capture/{_tilt_name(sweep, v, deg)}"
            lines.append(":::{grid-item}\n")
            lines.append(_img_block(
                tilt_src,
                f"Capture map at initial stroke-plane tilt = {deg} deg "
                f"({sweep.section_title}, {v.label})",
            ))
            lines.append(":::\n")

            lines.append("::::\n")    # close grid
            lines.append(":::::\n")   # close tilt tab-item
        lines.append("::::::\n")      # close tilt tab-set
        lines.append(":::::::\n")     # close outer tab-item
    lines.append("::::::::\n")        # close outer tab-set
    lines.append("\n")

    lines.append("### Representative Trajectories\n")
    lines.append("\n")
    lines.append(_trajectory_img_block(sweep))
    lines.append("\n")
    return "".join(lines)


def render_markdown(sweeps: list[Sweep]) -> None:
    sample = load_result(_result_path(sweeps[0].values[0]))
    signed_degs = [d for d, _ in _signed_tilts(sample.tilts)]

    body = "# Stationary Prey Capture\n\n"
    for sweep in sweeps:
        body += render_section(sweep, signed_degs)
    DOCS_PAGE.write_text(body)
    print(f"wrote {DOCS_PAGE}")


def _delete_obsolete() -> None:
    """Remove PNGs from earlier filename schemes."""
    for pat in (
        # very old composites
        f"gain_sweep_avg_rcap{CAPTURE_RADIUS:g}.dark.png",
        f"gain_sweep_tilt[0-9][0-9][0-9]_rcap{CAPTURE_RADIUS:g}.dark.png",
        # unsigned-tilt single-wedge variant of the gain sweep
        f"gain_sweep_K*_tilt[0-9][0-9][0-9]_rcap{CAPTURE_RADIUS:g}.dark.png",
    ):
        for p in OUT_DIR.glob(pat):
            p.unlink()
            print(f"removed {p.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _delete_obsolete()
    for sweep in SWEEPS:
        render_sweep_pngs(sweep)
    for sweep in SWEEPS:
        render_sweep_trajectory(
            sweep_name=sweep.name,
            section_title=sweep.section_title,
            value_labels=[v.label for v in sweep.values],
            value_overrides=[v.overrides for v in sweep.values],
            value_data_dirs=[v.data_dir for v in sweep.values],
        )
    render_markdown(SWEEPS)


if __name__ == "__main__":
    main()

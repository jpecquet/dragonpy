"""
Shared plotting style helpers.

This module centralizes matplotlib rcParams and theme resolution so that all
plotting scripts share the same fonts and light/dark behavior.
"""

from typing import Optional

from .hybrid_config import StyleConfig

DEFAULT_FIGURE_WIDTH_IN = 6.5


def figure_size(height_over_width: float = 1.0, width_in: float = DEFAULT_FIGURE_WIDTH_IN) -> tuple[float, float]:
    """Return a `(width, height)` tuple with a standardized default width."""
    width = float(width_in)
    ratio = float(height_over_width)
    if width <= 0.0:
        raise ValueError("width_in must be > 0")
    if ratio <= 0.0:
        raise ValueError("height_over_width must be > 0")
    return (width, width * ratio)


def resolve_style(style: Optional[StyleConfig] = None, theme: Optional[str] = None) -> StyleConfig:
    """
    Resolve a style object, optionally overriding with a named theme.

    When a theme override is requested, color tokens are replaced with the
    selected theme while font/sizing knobs from the original style are kept.
    """
    if style is None:
        base_theme = StyleConfig.normalize_theme(theme or 'light')
        return StyleConfig.themed(base_theme)

    if theme is None:
        return style

    resolved_theme = StyleConfig.normalize_theme(theme)
    if resolved_theme == StyleConfig.normalize_theme(style.theme):
        style.theme = resolved_theme
        return style

    themed = StyleConfig.themed(resolved_theme)
    themed.font_family = style.font_family
    themed.font_serif = style.font_serif
    themed.mathtext_fontset = style.mathtext_fontset
    themed.font_size = style.font_size
    themed.trajectory_linewidth = style.trajectory_linewidth
    themed.force_linewidth = style.force_linewidth
    themed.marker_size = style.marker_size
    return themed


def apply_matplotlib_style(style: Optional[StyleConfig] = None) -> None:
    """Apply style tokens to matplotlib global rcParams."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    resolved = resolve_style(style)

    plt.rcParams["font.family"] = resolved.font_family
    plt.rcParams["font.serif"] = [resolved.font_serif]
    plt.rcParams["mathtext.fontset"] = resolved.mathtext_fontset
    plt.rcParams["font.size"] = resolved.font_size

    plt.rcParams["figure.facecolor"] = resolved.figure_facecolor
    plt.rcParams["axes.facecolor"] = resolved.axes_facecolor
    plt.rcParams["savefig.facecolor"] = resolved.figure_facecolor
    plt.rcParams["savefig.edgecolor"] = resolved.figure_facecolor

    plt.rcParams["text.color"] = resolved.text_color
    plt.rcParams["axes.labelcolor"] = resolved.text_color
    plt.rcParams["axes.edgecolor"] = resolved.axes_edge_color
    plt.rcParams["xtick.color"] = resolved.text_color
    plt.rcParams["ytick.color"] = resolved.text_color
    plt.rcParams["grid.color"] = resolved.grid_color

    plt.rcParams["legend.facecolor"] = resolved.legend_facecolor
    plt.rcParams["legend.edgecolor"] = resolved.legend_edge_color

    try:
        from cycler import cycler

        plt.rcParams["axes.prop_cycle"] = cycler(color=[
            resolved.trajectory_color,
            resolved.error_color,
            resolved.target_color,
            resolved.lift_color,
            resolved.drag_color,
        ])
    except Exception:
        pass

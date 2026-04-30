"""
Shared style configuration for postprocessing visualization.
"""

from dataclasses import asdict, dataclass


@dataclass
class StyleConfig:
    """Visual style configuration for matplotlib overlay."""
    theme: str = 'light'

    font_family: str = 'serif'
    font_serif: str = 'Times New Roman'
    mathtext_fontset: str = 'stix'
    font_size: int = 12

    # Figure/axes colors
    figure_facecolor: str = '#ffffff'
    axes_facecolor: str = '#ffffff'
    axes_edge_color: str = '#303030'
    text_color: str = '#101010'
    muted_text_color: str = '#666666'
    grid_color: str = '#b0b0b0'

    # Colors
    trajectory_color: str = '#1f77b4'
    target_color: str = '#2ca02c'
    error_color: str = '#d62728'
    lift_color: str = '#1f77b4'
    drag_color: str = '#d62728'
    body_color: str = '#111111'
    wing_color: str = '#d3d3d3'
    wing_edge_color: str = '#2e2e2e'
    legend_facecolor: str = '#ffffff'
    legend_edge_color: str = '#cccccc'
    landscape_colormap: str = 'viridis'
    landscape_contour_color: str = '#ffffff'
    landscape_min_marker_color: str = '#d62728'

    # Line widths
    trajectory_linewidth: float = 1.5
    force_linewidth: float = 2.0

    # Sizes
    marker_size: float = 50

    @staticmethod
    def normalize_theme(theme: str) -> str:
        """Normalize theme name and fall back to light for unknown themes."""
        name = str(theme).strip().lower()
        if name in ('light', 'dark'):
            return name
        return 'light'

    @classmethod
    def themed(cls, theme: str = 'light') -> 'StyleConfig':
        """Create style defaults for a named theme."""
        name = cls.normalize_theme(theme)
        if name == 'dark':
            return cls(
                theme='dark',
                figure_facecolor='#131416',
                axes_facecolor='#131416',
                axes_edge_color='#c7d1db',
                text_color='#f1f5f9',
                muted_text_color='#9fb0c0',
                grid_color='#4a5563',
                trajectory_color='#4aa3ff',
                target_color='#56d68b',
                error_color='#ff6b6b',
                lift_color='#4aa3ff',
                drag_color='#ff8f6b',
                body_color='#f2f5f7',
                wing_color='#8f9aa6',
                wing_edge_color='#d9dee4',
                legend_facecolor='#1a222b',
                legend_edge_color='#4a5563',
                landscape_colormap='cividis',
                landscape_contour_color='#d9dee4',
                landscape_min_marker_color='#ff6b6b',
            )
        return cls(theme='light')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'StyleConfig':
        if not d:
            return cls.themed('light')
        theme = cls.normalize_theme(d.get('theme', 'light'))
        style = cls.themed(theme)
        for key, value in d.items():
            if key in cls.__dataclass_fields__:
                setattr(style, key, value)
        style.theme = cls.normalize_theme(style.theme)
        return style

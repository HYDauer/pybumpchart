"""Color palette handling for bump charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    pass


def get_color_palette(
    palette: str | Sequence[Any] | None,
    n_colors: int,
) -> list[Any]:
    """Get a list of colors from a palette specification.

    Parameters
    ----------
    palette : str, Sequence, or None
        Color palette specification:
        - None: Use matplotlib's default color cycle
        - str: Name of a matplotlib colormap (e.g., 'viridis', 'tab10')
               or seaborn palette name if seaborn is installed
        - Sequence: List of colors to use directly
    n_colors : int
        Number of colors needed.

    Returns
    -------
    list
        List of colors, cycling if necessary.

    Examples
    --------
    >>> colors = get_color_palette('tab10', 3)
    >>> len(colors)
    3
    >>> colors = get_color_palette(['red', 'blue'], 4)
    >>> colors
    ['red', 'blue', 'red', 'blue']
    """
    colors: list[Any]
    if palette is None:
        # Use matplotlib default color cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = list(prop_cycle.by_key().get("color", ["#1f77b4"]))
    elif isinstance(palette, str):
        colors = _get_colors_from_name(palette, n_colors)
    else:
        # Assume it's a sequence of colors
        colors = list(palette)

    # Cycle colors if we need more than provided
    if len(colors) < n_colors:
        colors = (colors * ((n_colors // len(colors)) + 1))[:n_colors]

    return colors[:n_colors]


def _get_colors_from_name(name: str, n_colors: int) -> list[Any]:
    """Get colors from a palette name.

    Parameters
    ----------
    name : str
        Name of the colormap or palette.
    n_colors : int
        Number of colors needed.

    Returns
    -------
    list
        List of colors.
    """
    # Try matplotlib colormap first
    try:
        cmap = colormaps.get_cmap(name)
        # For qualitative colormaps like tab10, use discrete colors
        if hasattr(cmap, "colors") and cmap.colors is not None:
            return list(cmap.colors)
        # For continuous colormaps, sample evenly
        return [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
    except (ValueError, KeyError):
        pass

    # Try seaborn if available
    try:
        import seaborn as sns

        return list(sns.color_palette(name, n_colors))
    except ImportError:
        pass
    except ValueError:
        pass

    # If nothing works, return default colors
    return get_color_palette(None, n_colors)


def get_highlight_colors(
    entities: Sequence[str],
    highlight: Sequence[str] | None,
    base_colors: Sequence[Any],
    highlight_color: Any | None = None,
    dim_alpha: float = 0.3,
    dim_color: str = "#999999",
) -> tuple[list[Any], list[float]]:
    """Get colors and alphas for highlighted/dimmed entities.

    Parameters
    ----------
    entities : Sequence[str]
        List of all entity names in order.
    highlight : Sequence[str] or None
        List of entities to highlight. If None, all entities are shown normally.
    base_colors : Sequence
        Base colors for each entity (used when not highlighting).
    highlight_color : Any, optional
        Custom color for highlighted entities. If None, uses base colors.
    dim_alpha : float, default 0.3
        Alpha value for dimmed (non-highlighted) entities.
    dim_color : str, default '#999999'
        Color for dimmed entities.

    Returns
    -------
    tuple[list, list[float]]
        Tuple of (colors, alphas) lists for each entity.
    """
    if highlight is None:
        # No highlighting - use base colors at full alpha
        return list(base_colors), [1.0] * len(entities)

    highlight_set = set(highlight)
    colors: list[Any] = []
    alphas: list[float] = []

    for i, entity in enumerate(entities):
        if entity in highlight_set:
            # Highlighted entity
            if highlight_color is not None:
                colors.append(highlight_color)
            else:
                colors.append(base_colors[i])
            alphas.append(1.0)
        else:
            # Dimmed entity
            colors.append(dim_color)
            alphas.append(dim_alpha)

    return colors, alphas


def adjust_color_alpha(color: Any, alpha: float) -> tuple[float, float, float, float]:
    """Adjust the alpha of a color.

    Parameters
    ----------
    color : Any
        Input color in any matplotlib-compatible format.
    alpha : float
        New alpha value (0-1).

    Returns
    -------
    tuple[float, float, float, float]
        RGBA tuple with adjusted alpha.
    """
    rgba = to_rgba(color)
    return (rgba[0], rgba[1], rgba[2], alpha)

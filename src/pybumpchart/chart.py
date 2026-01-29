"""Core chart rendering functions for bump charts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


def _draw_lines(
    ax: Axes,
    times: NDArray[Any],
    ranks: NDArray[Any],
    color: Any = None,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    zorder: int = 1,
    **kwargs: Any,
) -> Line2D:
    """Draw connecting lines between rank positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    times : np.ndarray
        Array of time values (x-axis).
    ranks : np.ndarray
        Array of rank values (y-axis).
    color : Any, optional
        Line color. If None, uses matplotlib default cycle.
    linewidth : float, default 2.0
        Width of the line.
    alpha : float, default 1.0
        Transparency of the line (0-1).
    zorder : int, default 1
        Drawing order (higher = on top).
    **kwargs
        Additional arguments passed to ax.plot().

    Returns
    -------
    Line2D
        The matplotlib Line2D object that was drawn.
    """
    # Filter out NaN values for continuous line segments
    mask = ~np.isnan(ranks)
    if not mask.any():
        # Return empty line if all NaN
        (line,) = ax.plot(
            [], [], color=color, linewidth=linewidth, alpha=alpha, **kwargs
        )
        return line

    # Draw the line
    (line,) = ax.plot(
        times[mask],
        ranks[mask],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        **kwargs,
    )
    return line


def _draw_points(
    ax: Axes,
    times: NDArray[Any],
    ranks: NDArray[Any],
    color: Any = None,
    marker: str = "o",
    markersize: float = 8.0,
    alpha: float = 1.0,
    zorder: int = 2,
    **kwargs: Any,
) -> Any:
    """Draw markers at each time-rank position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    times : np.ndarray
        Array of time values (x-axis).
    ranks : np.ndarray
        Array of rank values (y-axis).
    color : Any, optional
        Marker color. If None, uses matplotlib default cycle.
    marker : str, default 'o'
        Marker style.
    markersize : float, default 8.0
        Size of the markers.
    alpha : float, default 1.0
        Transparency of the markers (0-1).
    zorder : int, default 2
        Drawing order (higher = on top).
    **kwargs
        Additional arguments passed to ax.scatter().

    Returns
    -------
    PathCollection
        The matplotlib PathCollection object that was drawn.
    """
    # Filter out NaN values
    mask = ~np.isnan(ranks)
    if not mask.any():
        return ax.scatter([], [], color=color, marker=marker, s=markersize**2, **kwargs)

    return ax.scatter(
        times[mask],
        ranks[mask],
        color=color,
        marker=marker,
        s=markersize**2,  # scatter uses area, not diameter
        alpha=alpha,
        zorder=zorder,
        edgecolors="white",
        linewidths=1,
        **kwargs,
    )


def format_axes(
    ax: Axes,
    time_values: NDArray[Any],
    max_rank: int,
    show_grid: bool = True,
) -> None:
    """Format the axes for a bump chart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format.
    time_values : np.ndarray
        Array of unique time values for x-axis ticks.
    max_rank : int
        Maximum rank value for y-axis configuration.
    show_grid : bool, default True
        Whether to show grid lines.
    """
    # Invert y-axis so rank 1 is at top
    ax.invert_yaxis()

    # Set x-axis ticks to time values
    ax.set_xticks(time_values)
    ax.set_xticklabels([str(t) for t in time_values])

    # Set y-axis ticks to rank numbers
    ax.set_yticks(range(1, max_rank + 1))
    ax.set_yticklabels([str(i) for i in range(1, max_rank + 1)])

    # Set axis limits with padding
    ax.set_xlim(time_values.min() - 0.5, time_values.max() + 0.5)
    ax.set_ylim(max_rank + 0.5, 0.5)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Style remaining spines
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")

    # Grid styling
    if show_grid:
        ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="#666666")
        ax.xaxis.grid(False)
    else:
        ax.grid(False)

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=10, colors="#333333")


def create_figure(
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Create a new figure and axes for a bump chart.

    Parameters
    ----------
    figsize : tuple[float, float], optional
        Figure size as (width, height) in inches.
        Defaults to (10, 6).

    Returns
    -------
    tuple[Figure, Axes]
        The created figure and axes objects.
    """
    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

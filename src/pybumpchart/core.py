"""Core bumpchart function - main public API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pybumpchart.chart import _draw_lines, _draw_points, create_figure, format_axes
from pybumpchart.colors import get_color_palette, get_highlight_colors
from pybumpchart.data import TieMethod, get_entity_data, prepare_data
from pybumpchart.labels import LabelPosition, add_labels_with_collision_avoidance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def bumpchart(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    rank_col: str | None = None,
    value_col: str | None = None,
    *,
    ascending: bool = True,
    tie_method: TieMethod = "average",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    palette: str | Sequence[Any] | None = None,
    highlight: Sequence[str] | None = None,
    highlight_color: Any | None = None,
    show_labels: LabelPosition = True,
    label_padding: float = 0.3,
    min_label_distance: float = 0.5,
    label_fontsize: float = 10,
    linewidth: float = 2.5,
    markersize: float = 10.0,
    marker: str = "o",
    show_grid: bool = True,
    show_points: bool = True,
) -> Axes:
    """Create a bump chart visualization.

    A bump chart shows how rankings change over time, with each entity
    represented as a line connecting its rank at each time point.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing time, entity, and rank/value information.
    time_col : str
        Name of the column containing time values (e.g., years, dates).
    entity_col : str
        Name of the column containing entity identifiers (e.g., team names).
    rank_col : str, optional
        Name of the column containing pre-calculated rank values.
        If provided, these ranks are used directly.
    value_col : str, optional
        Name of the column containing values to rank by.
        Used to calculate ranks if rank_col is not provided.

    ascending : bool, default True
        When calculating ranks from value_col:
        - True: Lower values get lower (better) ranks
        - False: Higher values get lower (better) ranks
    tie_method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        Method for handling ties when calculating ranks from values.

    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, creates new figure and axes.
    figsize : tuple[float, float], optional
        Figure size as (width, height) in inches. Default (10, 6).
        Only used when ax is None.
    palette : str, Sequence, or None, optional
        Color palette specification:
        - None: Use matplotlib's default color cycle
        - str: Name of a matplotlib colormap (e.g., 'viridis', 'tab10')
        - Sequence: List of colors to use
    highlight : Sequence[str], optional
        List of entity names to highlight. Non-highlighted entities
        are dimmed.
    highlight_color : Any, optional
        Custom color for highlighted entities. If None, uses palette colors.
    show_labels : bool or str, default True
        Where to show entity labels:
        - True or 'both': Show on both sides
        - False: Don't show labels
        - 'left': Show only at first time point
        - 'right': Show only at last time point
    label_padding : float, default 0.3
        Horizontal distance between line endpoint and label.
    min_label_distance : float, default 0.5
        Minimum vertical distance between labels (for collision avoidance).
    label_fontsize : float, default 10
        Font size for entity labels.
    linewidth : float, default 2.5
        Width of the ranking lines.
    markersize : float, default 10.0
        Size of the markers at each time point.
    marker : str, default 'o'
        Marker style for time points.
    show_grid : bool, default True
        Whether to show horizontal grid lines.
    show_points : bool, default True
        Whether to show markers at each time point.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the bump chart.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If required columns are missing or data is invalid.

    Examples
    --------
    Using pre-calculated ranks:

    >>> import pandas as pd
    >>> import pybumpchart as bc
    >>> df = pd.DataFrame({
    ...     'year': [2020, 2020, 2021, 2021],
    ...     'team': ['A', 'B', 'A', 'B'],
    ...     'rank': [1, 2, 2, 1]
    ... })
    >>> ax = bc.bumpchart(df, time_col='year', entity_col='team', rank_col='rank')

    Calculating ranks from values (higher is better):

    >>> df = pd.DataFrame({
    ...     'year': [2020, 2020, 2021, 2021],
    ...     'team': ['A', 'B', 'A', 'B'],
    ...     'score': [100, 80, 85, 95]
    ... })
    >>> ax = bc.bumpchart(
    ...     df, time_col='year', entity_col='team',
    ...     value_col='score', ascending=False
    ... )

    Highlighting specific entities:

    >>> ax = bc.bumpchart(
    ...     df, time_col='year', entity_col='team', rank_col='rank',
    ...     highlight=['A'], palette='tab10'
    ... )
    """
    # Prepare data
    prepared_df = prepare_data(
        df,
        time_col=time_col,
        entity_col=entity_col,
        rank_col=rank_col,
        value_col=value_col,
        ascending=ascending,
        tie_method=tie_method,
    )

    # Get unique values
    entities = prepared_df[entity_col].unique().tolist()
    time_values = np.sort(prepared_df[time_col].unique())
    max_rank = int(prepared_df["_rank"].max())

    # Create or use axes
    if ax is None:
        _, ax = create_figure(figsize)

    # Get colors
    n_entities = len(entities)
    base_colors = get_color_palette(palette, n_entities)
    colors, alphas = get_highlight_colors(
        entities, highlight, base_colors, highlight_color
    )

    # Collect entity data for labels
    entity_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # type: ignore[type-arg]

    # Draw each entity
    for i, entity in enumerate(entities):
        times, ranks = get_entity_data(prepared_df, entity_col, entity, time_col)
        entity_data[entity] = (times, ranks)

        # Adjust linewidth for highlighted entities
        entity_linewidth = linewidth
        if highlight is not None and entity in highlight:
            entity_linewidth = linewidth * 1.5

        # Draw line
        _draw_lines(
            ax,
            times,
            ranks,
            color=colors[i],
            linewidth=entity_linewidth,
            alpha=alphas[i],
        )

        # Draw points
        if show_points:
            _draw_points(
                ax,
                times,
                ranks,
                color=colors[i],
                marker=marker,
                markersize=markersize,
                alpha=alphas[i],
            )

    # Format axes
    format_axes(ax, time_values, max_rank, show_grid=show_grid)

    # Add labels with collision avoidance
    add_labels_with_collision_avoidance(
        ax,
        entities,
        entity_data,
        colors,
        show_labels=show_labels,
        label_padding=label_padding,
        min_label_distance=min_label_distance,
        fontsize=label_fontsize,
    )

    return ax

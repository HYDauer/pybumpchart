"""Label positioning for bump charts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

LabelPosition = Literal[True, False, "left", "right", "both"]


def add_labels(
    ax: Axes,
    entity: str,
    times: NDArray[Any],
    ranks: NDArray[Any],
    color: Any,
    show_labels: LabelPosition = True,
    label_padding: float = 0.3,
    fontsize: float = 10,
    **kwargs: Any,
) -> None:
    """Add entity labels at line endpoints.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    entity : str
        The entity name to display.
    times : np.ndarray
        Array of time values.
    ranks : np.ndarray
        Array of rank values.
    color : Any
        Color for the label text.
    show_labels : bool or str, default True
        Where to show labels:
        - True or 'both': Show on both left and right
        - False: Don't show labels
        - 'left': Show only on left (first time point)
        - 'right': Show only on right (last time point)
    label_padding : float, default 0.3
        Horizontal padding between line endpoint and label.
    fontsize : float, default 10
        Font size for labels.
    **kwargs
        Additional arguments passed to ax.text().
    """
    if show_labels is False:
        return

    # Filter out NaN values
    mask = ~np.isnan(ranks)
    if not mask.any():
        return

    valid_times = times[mask]
    valid_ranks = ranks[mask]

    # Determine which sides to show labels
    show_left = show_labels in (True, "left", "both")
    show_right = show_labels in (True, "right", "both")

    if show_left and len(valid_times) > 0:
        _add_single_label(
            ax,
            entity,
            valid_times[0],
            valid_ranks[0],
            color,
            side="left",
            padding=label_padding,
            fontsize=fontsize,
            **kwargs,
        )

    if show_right and len(valid_times) > 0:
        _add_single_label(
            ax,
            entity,
            valid_times[-1],
            valid_ranks[-1],
            color,
            side="right",
            padding=label_padding,
            fontsize=fontsize,
            **kwargs,
        )


def _add_single_label(
    ax: Axes,
    text: str,
    x: Any,
    y: Any,
    color: Any,
    side: Literal["left", "right"],
    padding: float,
    fontsize: float,
    **kwargs: Any,
) -> None:
    """Add a single label at a specified position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    text : str
        Label text.
    x : Any
        X position (time value).
    y : Any
        Y position (rank value).
    color : Any
        Text color.
    side : {'left', 'right'}
        Which side the label is on (affects alignment and offset).
    padding : float
        Horizontal offset from the point.
    fontsize : float
        Font size.
    **kwargs
        Additional arguments passed to ax.text().
    """
    if side == "left":
        x_offset = -padding
        ha = "right"
    else:
        x_offset = padding
        ha = "left"

    ax.text(
        float(x) + x_offset,
        float(y),
        text,
        color=color,
        fontsize=fontsize,
        ha=ha,
        va="center",
        fontweight="medium",
        **kwargs,
    )


def detect_label_collisions(
    labels: list[tuple[str, float]],
    min_distance: float = 0.5,
) -> list[tuple[str, float]]:
    """Detect and adjust overlapping labels.

    Parameters
    ----------
    labels : list[tuple[str, float]]
        List of (label_text, y_position) tuples.
    min_distance : float, default 0.5
        Minimum vertical distance between labels.

    Returns
    -------
    list[tuple[str, float]]
        List of (label_text, adjusted_y_position) tuples.
    """
    if len(labels) <= 1:
        return labels

    # Sort by y position
    sorted_labels = sorted(labels, key=lambda x: x[1])

    # Adjust positions to avoid collisions
    adjusted: list[tuple[str, float]] = []
    for i, (text, y) in enumerate(sorted_labels):
        if i == 0:
            adjusted.append((text, y))
        else:
            _prev_text, prev_y = adjusted[-1]
            if y - prev_y < min_distance:
                # Push this label down
                new_y = prev_y + min_distance
                adjusted.append((text, new_y))
            else:
                adjusted.append((text, y))

    return adjusted


def get_endpoint_positions(
    entities: list[str],
    entity_data: dict[str, tuple[NDArray[Any], NDArray[Any]]],
    side: Literal["left", "right"],
) -> list[tuple[str, float]]:
    """Get label positions for all entities on one side.

    Parameters
    ----------
    entities : list[str]
        List of entity names.
    entity_data : dict
        Dictionary mapping entity names to (times, ranks) tuples.
    side : {'left', 'right'}
        Which endpoint to get positions for.

    Returns
    -------
    list[tuple[str, float]]
        List of (entity_name, y_position) tuples.
    """
    positions: list[tuple[str, float]] = []

    for entity in entities:
        times, ranks = entity_data[entity]
        mask = ~np.isnan(ranks)
        if not mask.any():
            continue

        valid_ranks = ranks[mask]
        y = valid_ranks[0] if side == "left" else valid_ranks[-1]

        positions.append((entity, float(y)))

    return positions


def add_labels_with_collision_avoidance(
    ax: Axes,
    entities: list[str],
    entity_data: dict[str, tuple[NDArray[Any], NDArray[Any]]],
    colors: list[Any],
    show_labels: LabelPosition = True,
    label_padding: float = 0.3,
    min_label_distance: float = 0.5,
    fontsize: float = 10,
) -> None:
    """Add labels with collision avoidance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    entities : list[str]
        List of entity names.
    entity_data : dict
        Dictionary mapping entity names to (times, ranks) tuples.
    colors : list
        List of colors for each entity.
    show_labels : bool or str, default True
        Where to show labels.
    label_padding : float, default 0.3
        Horizontal padding between line endpoint and label.
    min_label_distance : float, default 0.5
        Minimum vertical distance between labels.
    fontsize : float, default 10
        Font size for labels.
    """
    if show_labels is False:
        return

    entity_colors = dict(zip(entities, colors))

    # Handle left labels
    if show_labels in (True, "left", "both"):
        left_positions = get_endpoint_positions(entities, entity_data, "left")
        adjusted_left = detect_label_collisions(left_positions, min_label_distance)

        for entity, y in adjusted_left:
            times, ranks = entity_data[entity]
            mask = ~np.isnan(ranks)
            if mask.any():
                _add_single_label(
                    ax,
                    entity,
                    times[mask][0],
                    y,
                    entity_colors[entity],
                    side="left",
                    padding=label_padding,
                    fontsize=fontsize,
                )

    # Handle right labels
    if show_labels in (True, "right", "both"):
        right_positions = get_endpoint_positions(entities, entity_data, "right")
        adjusted_right = detect_label_collisions(right_positions, min_label_distance)

        for entity, y in adjusted_right:
            times, ranks = entity_data[entity]
            mask = ~np.isnan(ranks)
            if mask.any():
                _add_single_label(
                    ax,
                    entity,
                    times[mask][-1],
                    y,
                    entity_colors[entity],
                    side="right",
                    padding=label_padding,
                    fontsize=fontsize,
                )

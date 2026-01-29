"""Tests for chart rendering functions."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt

from pybumpchart.chart import (
    _draw_lines,
    _draw_points,
    create_figure,
    format_axes,
)


class TestDrawLines:
    """Tests for _draw_lines function."""

    def test_simple_line(self) -> None:
        fig, ax = plt.subplots()
        times = np.array([2020, 2021, 2022])
        ranks = np.array([1.0, 2.0, 3.0])

        line = _draw_lines(ax, times, ranks, color="blue")

        assert line is not None
        assert line.get_color() == "blue"
        plt.close(fig)

    def test_line_with_nan(self) -> None:
        fig, ax = plt.subplots()
        times = np.array([2020, 2021, 2022])
        ranks = np.array([1.0, np.nan, 3.0])

        line = _draw_lines(ax, times, ranks, color="red")

        # Should only have 2 points (NaN filtered out)
        xdata, ydata = line.get_data()
        assert len(xdata) == 2
        assert len(ydata) == 2
        plt.close(fig)

    def test_all_nan_line(self) -> None:
        fig, ax = plt.subplots()
        times = np.array([2020, 2021, 2022])
        ranks = np.array([np.nan, np.nan, np.nan])

        line = _draw_lines(ax, times, ranks, color="green")

        # Should return empty line
        xdata, ydata = line.get_data()
        assert len(xdata) == 0
        plt.close(fig)

    def test_line_styling(self) -> None:
        fig, ax = plt.subplots()
        times = np.array([2020, 2021])
        ranks = np.array([1.0, 2.0])

        line = _draw_lines(ax, times, ranks, color="purple", linewidth=3.0, alpha=0.5)

        assert line.get_linewidth() == 3.0
        assert line.get_alpha() == 0.5
        plt.close(fig)


class TestDrawPoints:
    """Tests for _draw_points function."""

    def test_simple_points(self) -> None:
        fig, ax = plt.subplots()
        times = np.array([2020, 2021, 2022])
        ranks = np.array([1.0, 2.0, 3.0])

        scatter = _draw_points(ax, times, ranks, color="blue")

        assert scatter is not None
        offsets = scatter.get_offsets()
        assert len(offsets) == 3
        plt.close(fig)

    def test_points_with_nan(self) -> None:
        fig, ax = plt.subplots()
        times = np.array([2020, 2021, 2022])
        ranks = np.array([1.0, np.nan, 3.0])

        scatter = _draw_points(ax, times, ranks, color="red")

        offsets = scatter.get_offsets()
        assert len(offsets) == 2  # NaN filtered out
        plt.close(fig)

    def test_point_styling(self) -> None:
        fig, ax = plt.subplots()
        times = np.array([2020, 2021])
        ranks = np.array([1.0, 2.0])

        scatter = _draw_points(
            ax, times, ranks, color="green", marker="s", markersize=12.0
        )

        # markersize is converted to area (squared)
        sizes = scatter.get_sizes()
        assert sizes[0] == 144.0  # 12^2
        plt.close(fig)


class TestFormatAxes:
    """Tests for format_axes function."""

    def test_y_axis_inverted(self) -> None:
        fig, ax = plt.subplots()
        time_values = np.array([2020, 2021, 2022])

        format_axes(ax, time_values, max_rank=5)

        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # Inverted: higher value at bottom
        plt.close(fig)

    def test_x_ticks_set(self) -> None:
        fig, ax = plt.subplots()
        time_values = np.array([2020, 2021, 2022])

        format_axes(ax, time_values, max_rank=5)

        xticks = ax.get_xticks()
        np.testing.assert_array_equal(xticks, [2020, 2021, 2022])
        plt.close(fig)

    def test_y_ticks_set(self) -> None:
        fig, ax = plt.subplots()
        time_values = np.array([2020, 2021, 2022])

        format_axes(ax, time_values, max_rank=5)

        yticks = ax.get_yticks()
        np.testing.assert_array_equal(yticks, [1, 2, 3, 4, 5])
        plt.close(fig)

    def test_spines_removed(self) -> None:
        fig, ax = plt.subplots()
        time_values = np.array([2020, 2021, 2022])

        format_axes(ax, time_values, max_rank=5)

        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()
        assert ax.spines["bottom"].get_visible()
        plt.close(fig)

    def test_grid_shown_by_default(self) -> None:
        fig, ax = plt.subplots()
        time_values = np.array([2020, 2021, 2022])

        format_axes(ax, time_values, max_rank=5, show_grid=True)

        # Check that y-axis grid is enabled
        assert ax.yaxis.get_gridlines()[0].get_visible()
        plt.close(fig)

    def test_grid_can_be_hidden(self) -> None:
        fig, ax = plt.subplots()
        time_values = np.array([2020, 2021, 2022])

        format_axes(ax, time_values, max_rank=5, show_grid=False)

        # Grid should be off
        plt.close(fig)


class TestCreateFigure:
    """Tests for create_figure function."""

    def test_default_figsize(self) -> None:
        fig, ax = create_figure()

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)

    def test_custom_figsize(self) -> None:
        fig, ax = create_figure(figsize=(12, 8))

        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        plt.close(fig)

    def test_returns_axes(self) -> None:
        fig, ax = create_figure()

        assert ax is not None
        assert hasattr(ax, "plot")
        plt.close(fig)

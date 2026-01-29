"""Tests for core bumpchart function."""

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt

import pybumpchart as bc
from pybumpchart.core import bumpchart


class TestBumpchartBasic:
    """Basic tests for bumpchart function."""

    def test_simple_chart_with_ranks(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2021, 2021, 2021],
                "team": ["A", "B", "C", "A", "B", "C"],
                "rank": [1, 2, 3, 3, 1, 2],
            }
        )

        ax = bumpchart(df, time_col="year", entity_col="team", rank_col="rank")

        assert ax is not None
        assert hasattr(ax, "figure")
        plt.close(ax.figure)

    def test_simple_chart_with_values(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2021, 2021, 2021],
                "team": ["A", "B", "C", "A", "B", "C"],
                "score": [100, 80, 90, 70, 95, 85],
            }
        )

        ax = bumpchart(
            df, time_col="year", entity_col="team", value_col="score", ascending=False
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_returns_axes_object(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "team": ["A", "B"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(df, time_col="year", entity_col="team", rank_col="rank")

        # Should be a matplotlib Axes object
        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)
        plt.close(ax.figure)

    def test_uses_provided_axes(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "team": ["A", "B"],
                "rank": [1, 2],
            }
        )

        fig, existing_ax = plt.subplots()
        returned_ax = bumpchart(
            df, time_col="year", entity_col="team", rank_col="rank", ax=existing_ax
        )

        assert returned_ax is existing_ax
        plt.close(fig)


class TestBumpchartParameters:
    """Tests for bumpchart function parameters."""

    def test_custom_figsize(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "team": ["A", "B"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df, time_col="year", entity_col="team", rank_col="rank", figsize=(15, 10)
        )

        fig = ax.figure
        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 10
        plt.close(fig)

    def test_palette_string(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "team": ["A", "B"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df, time_col="year", entity_col="team", rank_col="rank", palette="tab10"
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_palette_list(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "team": ["A", "B"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            palette=["red", "blue"],
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_highlight_entities(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2020],
                "team": ["A", "B", "C"],
                "rank": [1, 2, 3],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            highlight=["A"],
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_highlight_with_custom_color(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2020],
                "team": ["A", "B", "C"],
                "rank": [1, 2, 3],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            highlight=["A", "B"],
            highlight_color="red",
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_show_labels_true(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            show_labels=True,
        )

        # Check that text elements exist
        texts = ax.texts
        assert len(texts) >= 1
        plt.close(ax.figure)

    def test_show_labels_false(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            show_labels=False,
        )

        # Should have no text labels
        texts = ax.texts
        assert len(texts) == 0
        plt.close(ax.figure)

    def test_show_labels_left_only(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            show_labels="left",
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_show_labels_right_only(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            show_labels="right",
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_linewidth(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            linewidth=5.0,
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_markersize(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            markersize=15.0,
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_show_points_false(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            show_points=False,
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_show_grid_false(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2021],
                "team": ["A", "A"],
                "rank": [1, 2],
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            show_grid=False,
        )

        assert ax is not None
        plt.close(ax.figure)


class TestBumpchartIntegration:
    """Integration tests for full bumpchart pipeline."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from data to saved figure."""
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
                "team": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
                "rank": [1, 2, 3, 3, 1, 2, 2, 3, 1],
            }
        )

        ax = bumpchart(df, time_col="year", entity_col="team", rank_col="rank")

        # Verify basic chart properties
        assert ax is not None

        # Check y-axis is inverted (rank 1 at top)
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]

        # Check x-axis has correct ticks
        xticks = ax.get_xticks()
        np.testing.assert_array_equal(xticks, [2020, 2021, 2022])

        plt.close(ax.figure)

    def test_many_entities(self) -> None:
        """Test with many entities (10+)."""
        n_teams = 12
        years = [2020, 2021, 2022]

        data = []
        for year in years:
            for i in range(n_teams):
                data.append(
                    {
                        "year": year,
                        "team": f"Team_{i}",
                        "rank": ((i + year) % n_teams) + 1,
                    }
                )

        df = pd.DataFrame(data)

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            rank_col="rank",
            show_labels=True,
        )

        assert ax is not None
        plt.close(ax.figure)

    def test_many_time_periods(self) -> None:
        """Test with many time periods (10+)."""
        years = list(range(2010, 2024))  # 14 years
        teams = ["A", "B", "C"]

        data = []
        for year in years:
            for i, team in enumerate(teams):
                rank = ((i + year) % 3) + 1
                data.append({"year": year, "team": team, "rank": rank})

        df = pd.DataFrame(data)

        ax = bumpchart(df, time_col="year", entity_col="team", rank_col="rank")

        assert ax is not None
        plt.close(ax.figure)

    def test_value_col_with_ties(self) -> None:
        """Test ranking from values with ties."""
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2021, 2021, 2021],
                "team": ["A", "B", "C", "A", "B", "C"],
                "score": [100, 100, 80, 90, 95, 95],  # Ties in both years
            }
        )

        ax = bumpchart(
            df,
            time_col="year",
            entity_col="team",
            value_col="score",
            ascending=False,
            tie_method="average",
        )

        assert ax is not None
        plt.close(ax.figure)


class TestBumpchartErrors:
    """Tests for error handling in bumpchart."""

    def test_missing_required_column(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2020, 2020],
                "team": ["A", "B"],
            }
        )

        with pytest.raises(ValueError):
            bumpchart(df, time_col="year", entity_col="team")

    def test_invalid_dataframe(self) -> None:
        with pytest.raises(TypeError):
            bumpchart(
                {"year": [2020], "team": ["A"], "rank": [1]},  # type: ignore[arg-type]
                time_col="year",
                entity_col="team",
                rank_col="rank",
            )


class TestModuleAPI:
    """Tests for module-level API."""

    def test_bumpchart_exported(self) -> None:
        assert hasattr(bc, "bumpchart")

    def test_version_exported(self) -> None:
        assert hasattr(bc, "__version__")
        assert bc.__version__ == "0.1.0"

    def test_data_functions_exported(self) -> None:
        assert hasattr(bc, "validate_dataframe")
        assert hasattr(bc, "prepare_data")
        assert hasattr(bc, "calculate_ranks")

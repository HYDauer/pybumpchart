"""Data validation and rank calculation for bump charts."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from numpy.typing import NDArray

TieMethod = Literal["average", "min", "max", "first", "dense"]


def validate_dataframe(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    rank_col: str | None = None,
    value_col: str | None = None,
) -> None:
    """Validate that the DataFrame has required columns and structure.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    time_col : str
        Name of the column containing time values.
    entity_col : str
        Name of the column containing entity identifiers.
    rank_col : str, optional
        Name of the column containing rank values.
    value_col : str, optional
        Name of the column containing values to rank by.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If required columns are missing, or if neither rank_col nor value_col
        is provided.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    required_cols = [time_col, entity_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if rank_col is None and value_col is None:
        raise ValueError("Either 'rank_col' or 'value_col' must be provided")

    if rank_col is not None and rank_col not in df.columns:
        raise ValueError(f"Rank column '{rank_col}' not found in DataFrame")

    if value_col is not None and value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in DataFrame")


def prepare_data(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    rank_col: str | None = None,
    value_col: str | None = None,
    ascending: bool = True,
    tie_method: TieMethod = "average",
) -> pd.DataFrame:
    """Prepare and validate data for bump chart rendering.

    This function validates the input data, calculates ranks if needed,
    and returns a sorted DataFrame ready for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    time_col : str
        Name of the column containing time values.
    entity_col : str
        Name of the column containing entity identifiers.
    rank_col : str, optional
        Name of the column containing rank values. If provided, these ranks
        are used directly.
    value_col : str, optional
        Name of the column containing values to rank by. Used to calculate
        ranks if rank_col is not provided.
    ascending : bool, default True
        If True, lower values get lower (better) ranks.
        If False, higher values get lower (better) ranks.
    tie_method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        Method for handling ties when calculating ranks from values.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with columns: time_col, entity_col, and '_rank'.
        Sorted by time_col and entity_col.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If required columns are missing, data is invalid, or both/neither
        of rank_col and value_col are effectively provided.
    """
    validate_dataframe(df, time_col, entity_col, rank_col, value_col)

    # Work with a copy to avoid modifying original
    result = df.copy()

    # Calculate ranks if value_col is provided
    if value_col is not None:
        result["_rank"] = calculate_ranks(
            result, time_col, value_col, ascending=ascending, method=tie_method
        )
    elif rank_col is not None:
        result["_rank"] = result[rank_col].astype(float)
    else:
        raise ValueError("Either 'rank_col' or 'value_col' must be provided")

    # Validate ranks
    _validate_ranks(result, time_col, entity_col)

    # Sort by time and entity for consistent ordering
    result = result.sort_values([time_col, entity_col]).reset_index(drop=True)

    return result


def calculate_ranks(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    ascending: bool = True,
    method: TieMethod = "average",
) -> pd.Series:
    """Calculate ranks from values within each time period.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    time_col : str
        Name of the column containing time values.
    value_col : str
        Name of the column containing values to rank by.
    ascending : bool, default True
        If True, lower values get lower (better) ranks.
        If False, higher values get lower (better) ranks.
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        Method for handling ties:
        - 'average': Average rank of tied values
        - 'min': Lowest rank among ties
        - 'max': Highest rank among ties
        - 'first': Ranks assigned by order of appearance
        - 'dense': Like 'min', but ranks always increase by 1

    Returns
    -------
    pd.Series
        Series of rank values corresponding to input rows.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'year': [2020, 2020, 2020],
    ...     'team': ['A', 'B', 'C'],
    ...     'score': [100, 80, 90]
    ... })
    >>> calculate_ranks(df, 'year', 'score', ascending=False)
    0    1.0
    1    3.0
    2    2.0
    dtype: float64
    """
    return df.groupby(time_col)[value_col].rank(
        method=method, ascending=ascending, na_option="bottom"
    )


def _validate_ranks(df: pd.DataFrame, time_col: str, entity_col: str) -> None:
    """Validate rank values in the prepared DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with '_rank' column to validate.
    time_col : str
        Name of the time column.
    entity_col : str
        Name of the entity column.

    Raises
    ------
    ValueError
        If ranks contain invalid values or duplicates within time periods.
    """
    # Check for NaN ranks (excluding intentional gaps)
    if df["_rank"].isna().all():
        raise ValueError("All rank values are NaN")

    # Check for negative ranks
    valid_ranks = df["_rank"].dropna()
    if (valid_ranks < 0).any():
        raise ValueError("Rank values must be non-negative")

    # Check for duplicate entity-time combinations
    duplicates = df.duplicated(subset=[time_col, entity_col], keep=False)
    if duplicates.any():
        dup_rows = df[duplicates][[time_col, entity_col]].drop_duplicates()
        raise ValueError(
            f"Duplicate entity-time combinations found: {dup_rows.to_dict('records')}"
        )


def handle_missing_periods(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    fill_method: Literal["gap", "interpolate"] = "gap",
) -> pd.DataFrame:
    """Handle missing time periods for entities.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with '_rank' column.
    time_col : str
        Name of the time column.
    entity_col : str
        Name of the entity column.
    fill_method : {'gap', 'interpolate'}, default 'gap'
        How to handle missing periods:
        - 'gap': Leave as NaN (will create gaps in lines)
        - 'interpolate': Linear interpolation between known ranks

    Returns
    -------
    pd.DataFrame
        DataFrame with all entity-time combinations, missing values handled
        according to fill_method.
    """
    # Get all unique times and entities
    all_times = df[time_col].unique()
    all_entities = df[entity_col].unique()

    # Create complete index
    complete_index = pd.MultiIndex.from_product(
        [all_times, all_entities], names=[time_col, entity_col]
    )

    # Reindex to include all combinations
    result = df.set_index([time_col, entity_col]).reindex(complete_index).reset_index()

    if fill_method == "interpolate":
        # Interpolate ranks within each entity
        result["_rank"] = result.groupby(entity_col)["_rank"].transform(
            lambda x: x.interpolate(method="linear")
        )

    return result


def get_entity_data(
    df: pd.DataFrame, entity_col: str, entity: str, time_col: str
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Extract time and rank arrays for a single entity.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame with '_rank' column.
    entity_col : str
        Name of the entity column.
    entity : str
        The entity to extract data for.
    time_col : str
        Name of the time column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (time_values, rank_values) as numpy arrays.
    """
    entity_df = df[df[entity_col] == entity].sort_values(by=time_col)  # type: ignore[call-overload]
    times: NDArray[Any] = entity_df[time_col].to_numpy()
    ranks: NDArray[Any] = entity_df["_rank"].to_numpy()
    return times, ranks

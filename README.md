# pybumpchart

A Python library for creating bump chart visualizations showing how rankings change over time.

[![PyPI version](https://badge.fury.io/py/pybumpchart.svg)](https://badge.fury.io/py/pybumpchart)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install pybumpchart
```

For seaborn color palette support:

```bash
pip install pybumpchart[seaborn]
```

## Quick Start

```python
import pandas as pd
import pybumpchart as bc

# Create sample data
df = pd.DataFrame({
    'year': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
    'team': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'rank': [1, 2, 3, 3, 1, 2, 2, 3, 1]
})

# Create a bump chart
ax = bc.bumpchart(df, time_col='year', entity_col='team', rank_col='rank')
ax.figure.savefig('bumpchart.png')
```

## Features

- **Automatic rank calculation**: Provide values and let pybumpchart calculate ranks
- **Multiple tie-breaking methods**: average, min, max, first, dense
- **Customizable appearance**: colors, line widths, markers, labels
- **Entity highlighting**: Emphasize specific entities while dimming others
- **Label collision avoidance**: Automatic adjustment of overlapping labels
- **Matplotlib integration**: Returns standard Axes objects for further customization

## API Reference

### `bumpchart()`

The main function for creating bump charts.

```python
bc.bumpchart(
    df,                          # DataFrame with your data
    time_col,                    # Column name for time values (x-axis)
    entity_col,                  # Column name for entity identifiers
    rank_col=None,               # Column with pre-calculated ranks
    value_col=None,              # Column with values to rank by
    ascending=True,              # Lower values = better rank (when using value_col)
    tie_method='average',        # How to handle ties: 'average', 'min', 'max', 'first', 'dense'
    ax=None,                     # Existing matplotlib Axes to draw on
    figsize=(10, 6),             # Figure size if creating new figure
    palette=None,                # Color palette: colormap name, list, or None for default
    highlight=None,              # List of entities to highlight
    highlight_color=None,        # Custom color for highlighted entities
    show_labels=True,            # True, False, 'left', 'right', or 'both'
    label_padding=0.3,           # Distance between line and label
    min_label_distance=0.5,      # Minimum vertical space between labels
    label_fontsize=10,           # Font size for labels
    linewidth=2.5,               # Line thickness
    markersize=10.0,             # Marker size
    marker='o',                  # Marker style
    show_grid=True,              # Show horizontal grid lines
    show_points=True,            # Show markers at each time point
)
```

**Returns:** `matplotlib.axes.Axes` - The axes containing the bump chart.

## Examples

### Using Pre-calculated Ranks

```python
import pandas as pd
import pybumpchart as bc

df = pd.DataFrame({
    'year': [2020, 2020, 2020, 2021, 2021, 2021],
    'team': ['Warriors', 'Lakers', 'Celtics', 'Warriors', 'Lakers', 'Celtics'],
    'rank': [1, 2, 3, 2, 1, 3]
})

ax = bc.bumpchart(df, time_col='year', entity_col='team', rank_col='rank')
```

### Calculating Ranks from Values

```python
df = pd.DataFrame({
    'quarter': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2'],
    'product': ['Widget', 'Gadget', 'Gizmo', 'Widget', 'Gadget', 'Gizmo'],
    'sales': [1000, 800, 900, 850, 950, 920]
})

# Higher sales = better rank (rank 1)
ax = bc.bumpchart(
    df,
    time_col='quarter',
    entity_col='product',
    value_col='sales',
    ascending=False  # Higher values get lower (better) ranks
)
```

### Highlighting Specific Entities

```python
ax = bc.bumpchart(
    df,
    time_col='year',
    entity_col='team',
    rank_col='rank',
    highlight=['Warriors'],
    palette='tab10'
)
```

### Custom Styling

```python
ax = bc.bumpchart(
    df,
    time_col='year',
    entity_col='team',
    rank_col='rank',
    palette=['#e41a1c', '#377eb8', '#4daf4a'],  # Custom colors
    linewidth=4,
    markersize=12,
    show_labels='right',  # Labels only on the right
    figsize=(12, 8)
)

# Further customize with matplotlib
ax.set_title('Team Rankings Over Time', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Rank')
```

### Using with Existing Axes

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

bc.bumpchart(df1, time_col='year', entity_col='team', rank_col='rank', ax=ax1)
ax1.set_title('Sports Rankings')

bc.bumpchart(df2, time_col='quarter', entity_col='product', value_col='sales', ax=ax2)
ax2.set_title('Product Sales Rankings')

plt.tight_layout()
plt.savefig('comparison.png')
```

## Data Format

Your DataFrame should have:

1. **Time column**: Values representing time periods (years, quarters, dates, etc.)
2. **Entity column**: Identifiers for the items being ranked (team names, product names, etc.)
3. **Rank or value column**: Either pre-calculated ranks OR values to rank by

Each entity should appear once per time period.

## Dependencies

- Python >= 3.9
- matplotlib >= 3.5
- pandas >= 1.4
- numpy >= 1.21
- seaborn >= 0.12 (optional, for extended palette support)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

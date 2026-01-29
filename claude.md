# pybumpchart - Bump Chart Visualization Library

## Project Overview

A Python library for creating bump charts (rank change visualizations) that show how rankings change over time. This fills a documented gap in the Python visualization ecosystem.

**Unique advantage**: Seaborn maintainer explicitly blessed this package in GitHub issue #1797: "Interesting, but I think out of scope for seaborn. **I'd encourage you to make a library and put it on pypi.**"

**Target users**: Data journalists, sports analysts, business intelligence analysts, anyone visualizing competitive rankings

## Core Functionality (MVP)

```python
import pybumpchart as bc

# Basic usage - from ranking data
bc.bumpchart(df, time_col='year', entity_col='team', rank_col='position')

# From raw values (auto-calculate ranks)
bc.bumpchart(df, time_col='quarter', entity_col='company', value_col='revenue')

# With customization
bc.bumpchart(
    df,
    time_col='year',
    entity_col='team',
    rank_col='position',
    highlight=['Team A', 'Team B'],  # Highlight specific entities
    palette='viridis',               # Color palette
    show_labels=True,                # Entity labels at endpoints
    smooth=True                      # Smooth line interpolation
)
```

## Technical Requirements

### Input Data Format

```
| year | team   | rank |
|------|--------|------|
| 2020 | Team A | 1    |
| 2020 | Team B | 2    |
| 2021 | Team A | 3    |
| 2021 | Team B | 1    |
```

### Features

1. **Automatic rank calculation** - Can compute ranks from raw values
2. **Smooth line interpolation** - Bezier curves between time points
3. **Smart label positioning** - Avoid overlapping labels at endpoints
4. **Highlighting** - Emphasize specific entities, dim others
5. **Color palette integration** - Works with matplotlib/seaborn palettes
6. **Tied ranks handling** - Configurable behavior for ties

### Output

- Returns matplotlib Figure/Axes for further customization
- `savefig()` wrapper for easy export
- Style presets for common use cases (sports, business, academic)

## Technical Stack

- **matplotlib** - Core rendering
- **pandas** - Data handling
- **numpy** - Interpolation calculations
- **Optional**: seaborn for color palette integration

## Development Priorities

### MVP (v0.1.0) - 2-3 weeks
- Basic `bumpchart()` function
- Rank calculation from values
- Simple label positioning
- Basic color support
- README with examples

### v0.2.0
- Smooth line interpolation (Bezier)
- Entity highlighting
- Improved label collision avoidance
- Multiple style presets

### v1.0.0
- Faceted bump charts
- Animation support (for presentations)
- Interactive version (plotly backend option)
- Comprehensive documentation

## API Design Principles

1. **Simple default, powerful options** - `bc.bumpchart(df, 'year', 'team', 'rank')` should just work
2. **Matplotlib-compatible** - Return Axes object for customization
3. **Pandas-native** - Accept DataFrames directly
4. **Sensible defaults** - Auto-detect numeric vs categorical, handle NaN gracefully

## Example Use Cases

- **Sports**: NFL/NBA team rankings over seasons
- **Business**: Company market share rankings over quarters
- **Education**: University rankings over years
- **Politics**: Polling position changes during campaigns
- **Music**: Billboard chart position tracking

## Resources

- [Seaborn GitHub Issue #1797](https://github.com/mwaskom/seaborn/issues/1797) - Maintainer blessing
- [Bump chart examples](https://datavizproject.com/data-type/bump-chart/) - Design inspiration
- [R ggbump package](https://github.com/davidsjoberg/ggbump) - Reference implementation

## Testing Strategy

1. Unit tests for rank calculation
2. Visual regression tests for chart output
3. Test with real-world datasets (sports rankings, company data)
4. Edge cases: ties, missing time periods, single entity

## Distribution

- PyPI package: `pip install pybumpchart`
- Conda-forge (after initial adoption)
- Documentation on Read the Docs or GitHub Pages

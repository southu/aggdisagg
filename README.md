# aggdisagg

> **Temporal Aggregation & Disaggregation for Modern Python**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Polars](https://img.shields.io/badge/Polars-first-orange)](https://pola.rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyPI](https://img.shields.io/pypi/v/aggdisagg)](https://pypi.org/project/aggdisagg/)
[![GitHub](https://img.shields.io/badge/GitHub-aggdisagg-black?logo=github)](https://github.com/aggdisagg/aggdisagg)

<!-- GitHub social preview: the image below is auto-generated for social sharing -->
![aggdisagg social preview](https://opengraph.githubassets.com/1/aggdisagg/aggdisagg)

**Install & try in 10 seconds:**

```bash
pip install "aggdisagg[all]"
```

**Note:** Requires Python ≥ 3.10. For Excel support (`read_excel` etc.) use the `excel` extra or install `fastexcel` / `openpyxl` separately (or read with pandas + `pl.from_pandas`).

```python
import polars as pl
from aggdisagg import TemporalAligner
df = pl.DataFrame({"date": ["2020", "2021"], "y": [100.0, 120.0]})
print(TemporalAligner().fit_transform(df, datetime_col="date", target_col="y"))
```

**aggdisagg** is a clean, **Polars-first** Python library for converting time series between frequencies with **perfect aggregation consistency**.

- Disaggregate low → high frequency (with indicators)
- Aggregate high → low frequency (symmetric)
- Works with **Polars** (primary), **pandas**, and **xarray**

## Installation + Try in 10 Seconds

```bash
pip install "aggdisagg[all]" && python -c "
import polars as pl
from datetime import date
from aggdisagg import TemporalAligner
df = pl.DataFrame({'date':[date(2020,1,1),date(2021,1,1)], 'y':[1000.,1200.]})
print(TemporalAligner(method='uniform').fit_transform(df, datetime_col='date', target_col='y'))
"
```

## Quickstart with `TemporalAligner`

```python
import polars as pl
from datetime import date
from aggdisagg import TemporalAligner

df = pl.DataFrame({
    "date": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
    "y": [1200.0, 1500.0, 1350.0],      # low-frequency target
    "indicator": [100.0, 125.0, 110.0], # high-frequency indicator
})

aligner = TemporalAligner(
    method="chow-lin-opt",
    target_freq="1mo",
    agg="sum",
    indicator_cols=["indicator"],
)

monthly = aligner.fit_transform(df, datetime_col="date", target_col="y")
print(monthly.head())

# Perfect symmetric aggregation
yearly_back = aligner.aggregate(monthly, freq="1y")
print("Roundtrip OK:", (yearly_back["y_1y"] - df["y"]).abs().sum() < 1e-8)

# Plot (requires plotly)
monthly.plot()  # or use .plot() on the result if extended
```

## Supported Methods

- `uniform`
- `linear`
- `denton` / `denton-cholette`
- `chow-lin`, `chow-lin-opt` (auto ρ via maxlog/minrss)
- `litterman`, `fernandez`

All methods guarantee `C @ y_high ≈ y_low` exactly.

## Real-World Example: Disaggregating Multiple Quarterly Series to Monthly

When you have a DataFrame with several low-frequency series (e.g. quarterly revenue for multiple companies) and want to convert them all to monthly while preserving the aggregation constraint, use the `disaggregate_columns` helper:

```python
import polars as pl
from datetime import date
from aggdisagg import TemporalAligner

# Synthetic quarterly data (mimics real company revenue files)
df_q = pl.DataFrame({
    "date": [date(2018, 3, 1), date(2018, 6, 1), date(2018, 9, 1), date(2018, 12, 1)],
    "Krones": [1_020_000_000, 1_028_000_000, 1_032_000_000, 1_328_000_000],
    "JBT":     [409_200_000,   491_300_000,   481_900_000,   537_300_000],
    "GEA":     [1_189_000_000, 1_403_000_000, 1_360_000_000, 1_570_000_000],
})

aligner = TemporalAligner(method="linear", target_freq="1mo", agg="sum")

monthly = aligner.disaggregate_columns(
    df_q,
    datetime_col="date",
    include_dates=True,   # automatically generates proper monthly dates
)

print(monthly.head(6))
# date        Krones        JBT           GEA
# 2018-01-01  ~339.1m      ~131.1m      ~374.3m
# ...
# Round-trip check
reagg = aligner.aggregate(monthly.drop("date"), freq="1q")
print("Sums match original quarters:", 
      (reagg["y_1q"] - df_q["Krones"]).abs().sum() < 1e-6)
```

**Notes**
- The helper automatically detects numeric columns as targets (or pass `target_cols=[...]`).
- Date inference now correctly chooses a ratio of 3 for quarterly → monthly (instead of assuming annual).
- Use `include_dates=True` for a ready-to-use monthly date column, or call `expand_high_freq_dates` yourself for custom alignment.
- All series are disaggregated independently but share the same frequency mapping.
- New in 1.4.1: `extrapolate` ("nan" default) controls NaN low-freq input handling. "nan" (default) and "drop" never fabricate values from missing inputs; "hold"/"linear" fill using last anchor when requested. Pass on `fit_transform(..., extrapolate=...)` or `disaggregate_columns(...)`.

See `examples/quickstart.py` for more patterns.

## Why aggdisagg?

- **Polars-native** core (lazy-friendly)
- **Perfect consistency** by construction (C/D matrices)
- **Sklearn-style** + fluent API
- Real econometric methods (Denton quadratic, Chow-Lin GLS)
- Excellent pandas / xarray interop
- Production quality (typed, tested, documented)

## First-User Tips & Current Limitations

**Recommended starting point**
```python
aligner = TemporalAligner(method="chow-lin-opt", target_freq="1mo", agg="sum", indicator_cols=[...])
high = aligner.fit_transform(low_df, datetime_col="date", target_col="y")
back = aligner.aggregate(high, freq="1y")   # should match original low almost exactly
```

**Output shape**
The returned DataFrame contains `y_disaggregated` (and `y_std` when uncertainty was computed). Original context columns are **not** automatically repeated (this was changed for robustness across Polars/pandas/object dates). You can expand dates yourself:

```python
# Example: attach proper high-freq dates (fit_transform itself returns only values)
low_dates = low_df["date"]
high = aligner.fit_transform(low_df, datetime_col="date", target_col="y")
high = high.with_columns(aligner.expand_high_freq_dates(low_dates).alias("date"))
```

**Limitations (as of 1.1.0)**
- Date expansion in the output is basic (low-freq dates are not auto-expanded).
- Uncertainty is a simple bootstrap and can be noisy or near-zero.
- All listed methods are fully implemented, produce distinct results, respond to indicators/ρ where applicable, and satisfy the aggregation constraint. (Denton uses quadratic penalty on first/second differences; Chow-Lin family uses GLS with AR(1) or IAR(1) errors.)
- Only regular frequency ratios are supported.

See the CHANGELOG for the full list of recent robustness and correctness fixes.
- Negative post-correction + NNLS ensemble
- sktime / statsforecast compatible wrapper

```python
# Hierarchical
rec = aligner.reconcile_hierarchical([nat_df, reg_df])

# Uncertainty
mean, std = aligner.predict_with_uncertainty()

# Lazy + xarray
lazy_high = aligner.fit_transform(lazy_df)
xa = aligner.to_xarray(high_df)
```

See `examples/quickstart.py` for complete gallery.

## Development & Publishing

```bash
uv sync --all-extras
uv run pytest
uv run python examples/quickstart.py
uv build
# twine or uv publish
```

## License

MIT

---

**Built for data scientists who want temporal frequency conversion that just works.**

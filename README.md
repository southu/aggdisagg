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

aggdisagg focuses on **correctness and calendar fidelity** first:

- Calendar-aware disaggregation across **all frequency pairs** (including irregular ratios, leap years, week boundaries)
- Symmetric **calendar-aware aggregation**
- 8 real methods: uniform, linear, denton, denton-cholette, chow-lin, chow-lin-opt, litterman, fernandez
- Automatic stock/flow semantics detection
- `week_start` + `partial_weeks` control
- Honest NaN handling (`nan` default never fabricates values)
- Calibrated opt-in uncertainty bands (`with_uncertainty=True`)

It is **Polars-native** (core since 1.6.1; lazy-friendly, no pandas routing in the main paths) with excellent pandas/xarray interop, a sklearn-style API, and production quality (typed, tested, documented).

The recommended multi-series API (`disaggregate_columns`) has negligible overhead vs. calling `fit_transform` in a loop (see Benchmarks below).

## First-User Tips & Current Limitations

**Recommended starting point**
```python
aligner = TemporalAligner(method="chow-lin-opt", target_freq="1mo", agg="sum", indicator_cols=[...])
high = aligner.fit_transform(low_df, datetime_col="date", target_col="y")
back = aligner.aggregate(high, freq="1y")   # should match original low almost exactly
```

**Output shape**
By default (`return_dataframe=True, include_dates=True`), `fit_transform` returns a ready-to-use Polars DataFrame with a leading `date` (pl.Date) column plus `y_disaggregated` (and bands when `with_uncertainty=True`). This matches the behavior of `disaggregate_columns(..., include_dates=True)`.

```python
high = aligner.fit_transform(low_df, datetime_col="date", target_col="y")  # has "date" + "y_disaggregated"
```

To get the prior values-only DataFrame (no date column), use `return_dataframe=False`. The manual date expansion is still available as an advanced/optional path:

```python
# Advanced: manual date attachment (or for custom week_start etc.)
low_dates = low_df["date"]
high = aligner.fit_transform(low_df, datetime_col="date", target_col="y", return_dataframe=False)
high = high.with_columns(aligner.expand_high_freq_dates(low_dates).alias("date"))
```

**Limitations (as of 1.10.0)**
- Date expansion uses the same calendar-aware logic as `expand_high_freq_dates` (now attached by default on `fit_transform`).
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
mean, std = aligner.predict_with_uncertainty()  # real std when uncertainty was requested in disagg (otherwise near-zero)

# Lazy + xarray
lazy_high = aligner.fit_transform(lazy_df)
xa = aligner.to_xarray(high_df)
```

See `examples/quickstart.py` for complete gallery.

## Benchmarks

`benchmarks/bench_disagg.py` runs a deterministic, closed-form benchmark comparing `disaggregate_columns` (the convenient multi-series wrapper) against an equivalent hand-written per-series loop using `fit_transform`.

Because `disaggregate_columns` internally loops over columns (calling the same core per series), the overhead is negligible — the wrapper costs you essentially nothing while providing shared configuration, semantics handling, etc.

Example run (this hardware):

```
aggdisagg 1.10 multi-series benchmark (disaggregate_columns vs per-series naive)
Python 3.12.13, polars 1.42.1
Platform: macOS-26.3.1-arm64-arm-64bit

| N | n_low | method | vec_ms | naive_ms | ratio | note |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 12 | uniform | 6.3 | 5.5 | 0.9x |  |
| 20 | 12 | linear | 6.2 | 5.8 | 0.9x |  |
| 20 | 12 | denton | 16.5 | 15.3 | 0.9x | quad |
| 20 | 12 | chow-lin-opt | 317.3 | 301.3 | 0.9x | rho opt |
| 100 | 12 | uniform | 28.9 | 27.7 | 1.0x |  |
| 100 | 12 | linear | 30.1 | 29.0 | 1.0x |  |
| 100 | 12 | denton | 79.0 | 76.7 | 1.0x | quad |
| 100 | 12 | chow-lin-opt | 1526.6 | 1510.8 | 1.0x | rho opt |
```

Re-run `python benchmarks/bench_disagg.py` to refresh on your hardware. The numbers show the expected ~1.0× (the convenience wrapper internally loops per column, so ratios near 1.0× are expected; the layer adds negligible overhead). Note that `chow-lin-opt` is slower by design (ρ optimization) and `denton` due to the quadratic solve.

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

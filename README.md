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
pip install aggdisagg
```

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

## Installation

```bash
uv pip install aggdisagg
# or with extras
uv pip install "aggdisagg[all]"
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

## Why aggdisagg?

- **Polars-native** core (lazy-friendly)
- **Perfect consistency** by construction (C/D matrices)
- **Sklearn-style** + fluent API
- Real econometric methods (Denton quadratic, Chow-Lin GLS)
- Excellent pandas / xarray interop
- Production quality (typed, tested, documented)

## v0.2 Highlights

- Hierarchical reconciliation (national → regional)
- Uncertainty (bootstrap + analytic std errors)
- Full Polars lazy + xarray DataArray I/O
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

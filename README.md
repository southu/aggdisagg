# aggdisagg

> **Temporal Aggregation & Disaggregation for Modern Python**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Polars](https://img.shields.io/badge/Polars-first-orange)](https://pola.rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**aggdisagg** is a clean, Polars-first Python library for converting time series between frequencies while guaranteeing **perfect aggregation consistency**.

- Disaggregate (low → high frequency)
- Aggregate (high → low frequency)
- Works with **Polars**, **pandas**, and **xarray**

## Installation

```bash
# Core (Polars + NumPy + SciPy)
uv pip install aggdisagg

# With pandas + xarray + plotting support
uv pip install "aggdisagg[all]"
```

## Quickstart (Polars Native)

```python
import polars as pl
from aggdisagg import disaggregate, aggregate, AggDisaggModel

# Yearly sales data
y_yearly = pl.Series("sales", [1200.0, 1500.0, 1350.0])

# Disaggregate to monthly (uniform distribution)
y_monthly = disaggregate(
    y_yearly, 
    n_high=36, 
    method="uniform", 
    conversion="sum"
)

print(y_monthly.head(6))
# shape: (6,)
# Series: 'y_high' [f64]
# [
#     100.0
#     100.0
#     ...
# ]

# Perfect round-trip
y_back = aggregate(y_monthly, n_low=3, conversion="sum")
assert (y_back - y_yearly).abs().sum() < 1e-10
```

### Using the Model API (scikit-learn style)

```python
df = pl.DataFrame({"y": [1200.0, 1500.0, 1350.0]})

model = AggDisaggModel(method="linear", conversion="sum")
model.fit(df, n_high=36)

y_high = model.predict()
print(model.check_consistency())   # True

# Re-aggregate anytime
y_reagg = model.aggregate(y_high)
```

## Why aggdisagg?

- **Polars first** — lazy, fast, modern
- **Guaranteed consistency** — `C @ y_high == y_low` (within floating point)
- **Multiple methods** — `uniform`, `linear`, and extensible (Denton, Chow-Lin coming)
- **Works everywhere** — Polars ↔ pandas ↔ xarray interop
- **Production ready** — typed, tested, documented

## Supported Conversions

| Conversion | Meaning                     | Example use case          |
|------------|-----------------------------|---------------------------|
| `sum`      | High-freq values sum to low | Sales, production         |
| `mean`     | Average of high-freq        | Prices, rates             |
| `first`    | First high-freq value       | Stock levels (beginning)  |
| `last`     | Last high-freq value        | Stock levels (end)        |

## Roadmap

- Full Denton & Denton-Cholette
- Chow-Lin with indicator series + automatic ρ
- Proper uncertainty quantification
- Hierarchical (multi-level) reconciliation
- Daily / weekly / irregular calendar support

## Development

This project uses the modern Python stack:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright src
uv run mkdocs serve
```

## License

MIT

---

**Built for data scientists who want temporal frequency conversion that just works.**

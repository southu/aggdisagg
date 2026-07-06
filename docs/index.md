# aggdisagg

Welcome to the documentation for **aggdisagg** — Temporal Aggregation & Disaggregation for Modern Python.

## Installation

See the [README](../README.md#installation).

## Quickstart

```python
from aggdisagg import disaggregate
import polars as pl

y = pl.Series([100, 120, 110])
y_high = disaggregate(y, n_high=12, method="uniform", conversion="sum")
print(y_high)
```

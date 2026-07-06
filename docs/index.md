# aggdisagg

**Temporal Aggregation & Disaggregation for Modern Python** (v0.2)

Polars-first library with hierarchical reconciliation, uncertainty, ensemble, negative correction, and sktime compatibility.

## Installation

```bash
pip install aggdisagg
# or
uv pip install "aggdisagg[all]"
```

## Quickstart

```python
import polars as pl
from datetime import date
from aggdisagg import TemporalAligner

df = pl.DataFrame({
    "date": [date(2020,1,1), date(2021,1,1)],
    "y": [1000., 1200.],
    "ind": [80., 95.],
})

aligner = TemporalAligner(method="chow-lin-opt", target_freq="1mo", agg="sum", indicator_cols=["ind"])
monthly = aligner.fit_transform(df)
print(monthly.head())

# Hierarchical example
rec = aligner.reconcile_hierarchical([df, df])  # toy

# Uncertainty
m, s = aligner.predict_with_uncertainty()
print("std sample:", float(s.mean()) if s is not None else 0)
```

See README for full API, hierarchical, xarray, lazy, ensemble, sktime wrapper.

## API Reference

Main class: `TemporalAligner`

Methods: `fit`, `fit_transform`, `aggregate`, `predict_with_uncertainty`, `reconcile_hierarchical`, `to_xarray`, `get_sktime_transformer`

## Features (v0.2)

- Multiple econometric methods (Denton, Chow-Lin with auto-ρ, etc.)
- Perfect aggregation symmetry
- Hierarchical reconciliation
- Bootstrap + analytic uncertainty
- Polars Lazy + xarray full support
- Negative correction + NNLS ensemble
- sktime/statsforecast transformers

## Development

See repository README.

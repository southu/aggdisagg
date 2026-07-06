# Example Gallery

## Basic Disaggregation

```python
import polars as pl
from aggdisagg import TemporalAligner

df = pl.DataFrame({"date": ["2020", "2021"], "y": [100, 120]})
aligner = TemporalAligner(method="chow-lin-opt")
result = aligner.fit_transform(df)
```

## Hierarchical

See README.

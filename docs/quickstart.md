# Quickstart

Install:

```bash
pip install aggdisagg[all]
```

Basic usage with TemporalAligner:

See main README for examples.

## Pandas DatetimeIndex support

```python
import pandas as pd
from aggdisagg import TemporalAligner

df = pd.DataFrame(
    {"y": [100, 120]},
    index=pd.date_range("2020", periods=2, freq="YS")
)
# Will auto-detect
aligner = TemporalAligner(method="uniform", target_freq="M", agg="sum")
high = aligner.fit_transform(df)
print(high)
```

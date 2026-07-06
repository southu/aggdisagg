import polars as pl
import numpy as np
from aggdisagg import disaggregate, aggregate, AggDisaggModel

def test_roundtrip_uniform_sum():
    y_low = pl.Series("y", [100.0, 120.0, 110.0])
    y_high = disaggregate(y_low, n_high=12, method="uniform", conversion="sum")
    y_back = aggregate(y_high, n_low=3, method="uniform", conversion="sum")
    assert np.allclose(y_back.to_numpy(), y_low.to_numpy(), atol=1e-10)

def test_model_api():
    df = pl.DataFrame({"y": [100.0, 120.0]})
    model = AggDisaggModel(method="linear", conversion="sum")
    model.fit(df, n_high=8)
    y_high = model.predict()
    assert len(y_high) == 8
    assert model.check_consistency()

def test_polars_native():
    y = [10.0, 20.0, 15.0]
    high = disaggregate(y, n_high=6, method="uniform", conversion="mean")
    assert isinstance(high, pl.Series)
    assert len(high) == 6

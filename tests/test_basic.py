from datetime import date

import numpy as np
import polars as pl

from aggdisagg import AggDisaggModel, TemporalAligner, aggregate, disaggregate


def test_temporal_aligner_basic():
    df = pl.DataFrame({
        "date": [date(2020,1,1), date(2021,1,1)],
        "y": [100.0, 120.0],
    })
    aligner = TemporalAligner(method="uniform", target_freq="1mo", agg="sum")
    high = aligner.fit_transform(df, datetime_col="date", target_col="y")
    assert len(high) == 24
    back = aligner.aggregate(high, freq="1y")
    assert np.allclose(back["y_1y"].to_numpy(), [100.0, 120.0], atol=1e-8)

def test_chow_lin_runs():
    df = pl.DataFrame({
        "date": [date(2020,1,1), date(2021,1,1)],
        "y": [100.0, 120.0],
        "ind": [10.0, 12.0],
    })
    aligner = TemporalAligner(method="chow-lin-opt", target_freq="1mo", agg="sum", indicator_cols=["ind"])
    high = aligner.fit_transform(df, datetime_col="date", target_col="y")
    assert len(high) == 24
    assert aligner.rho_ is not None

def test_roundtrip_uniform_sum_legacy():
    # legacy functions still work
    y_low = pl.Series("y", [100.0, 120.0, 110.0])
    y_high = disaggregate(y_low, n_high=12, method="uniform", conversion="sum")
    y_back = aggregate(y_high, n_low=3, method="uniform", conversion="sum")
    assert np.allclose(y_back.to_numpy(), y_low.to_numpy(), atol=1e-10)

def test_model_api_legacy():
    df = pl.DataFrame({"y": [100.0, 120.0]})
    model = AggDisaggModel(method="linear", conversion="sum")
    model.fit(df, n_high=8)
    y_high = model.predict()
    assert len(y_high) == 8
    # legacy AggDisaggModel does not implement check_consistency on the model itself
    assert len(y_high) == 8

"""Quickstart example using Polars (native feel)."""

import polars as pl
from aggdisagg import disaggregate, aggregate, AggDisaggModel

print("=== aggdisagg Quickstart ===\n")

# Low frequency (e.g. yearly) data
y_low = pl.Series("sales", [1200.0, 1500.0, 1350.0, 1600.0])

# 1. Simple one-liner disaggregation (uniform distribution)
y_monthly = disaggregate(y_low, n_high=48, method="uniform", conversion="sum")
print("Disaggregated (first 6 months of first year):")
print(y_monthly.head(6))
print(f"Original sum: {y_low.sum()}, Re-aggregated sum: {aggregate(y_monthly, n_low=4, conversion='sum').sum()}\n")

# 2. Using the model class (sklearn style)
df = pl.DataFrame({"y": y_low})
model = AggDisaggModel(method="linear", conversion="sum")
model.fit(df, n_high=48)
y_lin = model.predict()
print("Linear interpolation disagg (first 4):", y_lin.head(4).to_list())

# Verify perfect consistency
print("Consistency check passed:", model.check_consistency())
print("\nRe-aggregated values:", model.aggregate(y_lin).to_list())

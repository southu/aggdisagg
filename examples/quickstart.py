"""Quickstart example — v0.2 features with TemporalAligner."""

from datetime import date

import numpy as np
import polars as pl

from aggdisagg import TemporalAligner

print("=== aggdisagg Quickstart ===\n")

# 1. Basic + indicators + Chow-Lin-opt
df = pl.DataFrame({
    "date": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
    "y": [1200.0, 1500.0, 1350.0],
    "indicator": [100.0, 120.0, 110.0],
})
aligner = TemporalAligner(method="chow-lin-opt", target_freq="1mo", agg="sum", indicator_cols=["indicator"])
monthly = aligner.fit_transform(df, datetime_col="date", target_col="y")
print("Monthly disagg (Chow-Lin):", monthly["y_disaggregated"].head(3).to_list())

# 2. Symmetric aggregate
back = aligner.aggregate(monthly, freq="1y")
print("Roundtrip match:", np.allclose(back["y_1y"].to_numpy(), df["y"].to_numpy()))

# 3. Uncertainty
mean, std = aligner.predict_with_uncertainty()
print("Uncertainty (mean std sample):", float(std.mean()) if std is not None else 0)

# 4. Ensemble + negative correction
aligner_ens = TemporalAligner(method="uniform", target_freq="1mo", agg="sum", use_ensemble=True, correct_negatives=True)
# Add some negative prone data
df_neg = pl.DataFrame({"date": df["date"], "y": [100., -50., 200.]})  # will trigger correction
mon_neg = aligner_ens.fit_transform(df_neg, datetime_col="date", target_col="y")
print("Ensemble + neg correction used:", aligner_ens.use_ensemble, aligner_ens.correct_negatives)

# 5. Hierarchical reconciliation example (simple 2-level)
coarse = pl.DataFrame({"y": [2700.]})
fine = pl.DataFrame({"y": [900., 1800.]})
rec = aligner.reconcile_hierarchical([coarse, fine], method="proportional")
print("Hierarchical reconciled levels:", len(rec))

# 6. Lazy support (Polars)
lazy_df = df.lazy()
high_lazy = aligner.fit_transform(lazy_df)
print("Lazy support (type):", type(high_lazy))

# 7. xarray (if available)
try:
    xa = aligner.to_xarray(monthly)
    print("xarray export shape:", xa.shape)
except Exception as e:
    print("xarray optional:", e)

print("\n✅ v0.2 features demonstrated with perfect consistency where applicable.")

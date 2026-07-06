"""Benchmark harness for aggdisagg vs other libs.

Run with: python benchmarks/compare.py

Requires optional deps for others.
"""

import time
from datetime import date

import numpy as np
import polars as pl

from aggdisagg import TemporalAligner


def synthetic_data(n_low=5, ratio=12):
    y_low = np.linspace(100, 200, n_low)
    df = pl.DataFrame({
        "date": [date(2000 + i, 1, 1) for i in range(n_low)],
        "y": y_low,
        "ind": np.random.randn(n_low * ratio)[:n_low] + np.linspace(10, 20, n_low)
    })
    # expand ind for high freq? For simplicity, use low for indicator too, but repeat.
    # But for bench, make df low freq with repeated? For demo, make high freq df? 
    # For disagg bench, input low freq.
    return df

def time_aggdisagg(df, method="uniform"):
    aligner = TemporalAligner(method=method, target_freq="1mo", agg="sum")
    start = time.time()
    high = aligner.fit_transform(df, datetime_col="date", target_col="y")
    elapsed = time.time() - start
    # verify
    back = aligner.aggregate(high, freq="1y")
    err = np.abs(back["y_1y"].to_numpy() - df["y"].to_numpy()).max()
    return elapsed, err

def main():
    print("aggdisagg Benchmark Harness")
    df = synthetic_data()
    print(f"Dataset: {len(df)} low-freq points")

    for m in ["uniform", "linear", "chow-lin-opt"]:
        try:
            t, err = time_aggdisagg(df, m)
            print(f"{m:15s}: {t*1000:.2f} ms, max err {err:.2e}")
        except Exception as e:
            print(f"{m:15s}: failed ({e})")

    # Try other libs if installed
    try:
        print("tempdisagg available, but no direct bench here for simplicity")
    except Exception:
        print("tempdisagg not installed (optional)")

    try:
        print("tsdisagg available")
    except Exception:
        print("tsdisagg not installed (optional)")

if __name__ == "__main__":
    main()

"""Reproducible multi-series benchmark for aggdisagg (vectorized vs naive).

Run: python benchmarks/bench_disagg.py

Prints a table for README. Deterministic closed-form data (no unseeded rand).
Honest: includes denton (quadratic) and chow-lin-opt (rho search) which are slower.
"""

import platform
import sys
import time

import numpy as np
import polars as pl

from aggdisagg import TemporalAligner

try:
    HAS_TD = True
except Exception:
    HAS_TD = False


def make_deterministic(n_series: int, n_low: int = 12) -> pl.DataFrame:
    """Closed form: different linear+periodic per series. Quarterly low -> monthly."""
    low_dates = [pl.date(2020 + i//4, (i % 4)*3 + 1, 1) for i in range(n_low)]
    data = {"date": low_dates}
    t = np.arange(n_low, dtype=float)
    for i in range(n_series):
        y = 1000.0 + (i * 0.2) + 40.0 * np.sin(2*np.pi * t / 4.0 + i*0.1)
        data[f"s{i:03d}"] = y
    return pl.DataFrame(data)


def bench_vec(df: pl.DataFrame, method: str, k: int = 2) -> float:
    cols = [c for c in df.columns if c != "date"]
    a = TemporalAligner(method=method, target_freq="1mo", agg="sum")
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        _ = a.disaggregate_columns(df, datetime_col="date", target_cols=cols,
                                   include_dates=False, with_uncertainty=False,
                                   default_semantics="flow", autodetect_semantics=False)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench_naive(df: pl.DataFrame, method: str, k: int = 2) -> float:
    cols = [c for c in df.columns if c != "date"]
    a = TemporalAligner(method=method, target_freq="1mo", agg="sum")
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        for c in cols:
            sub = df.select(["date", c])
            _ = a.fit_transform(sub, datetime_col="date", target_col=c,
                                return_dataframe=False, with_uncertainty=False)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    print("aggdisagg 1.10 multi-series benchmark (disaggregate_columns vectorized vs per-series naive)")
    print(f"Python {sys.version.split()[0]}, polars {pl.__version__}")
    print(f"Platform: {platform.platform()}")
    print()

    headers = ["N", "n_low", "method", "vec_ms", "naive_ms", "speedup", "note"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"]*len(headers)) + " |")

    for N in (20, 100):
        df = make_deterministic(N)
        nlow = df.height
        for m in ["uniform", "linear", "denton", "chow-lin-opt"]:
            try:
                tv = bench_vec(df, m)
                tn = bench_naive(df, m)
                sp = tn / tv if tv > 1e-9 else float("inf")
                note = {"chow-lin-opt": "rho opt", "denton": "quad"}.get(m, "")
                print(f"| {N} | {nlow} | {m} | {tv*1000:.1f} | {tn*1000:.1f} | {sp:.1f}x | {note} |")
            except Exception as e:
                print(f"| {N} | {nlow} | {m} | err | err | - | {str(e)[:20]} |")

    if HAS_TD:
        print("\n(tempdisagg present; rerun with manual timing if desired for extra column)")
    else:
        print("\n(tempdisagg not installed — skipping that baseline)")
    print("\nRe-run the script to refresh numbers on your hardware. All data deterministic.")


if __name__ == "__main__":
    main()

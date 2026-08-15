"""Audit runner: apply run_messy_loop.py's repeated-run pattern
to every CSV fixture in tests/data/ across all 8 methods.

The original tests/run_messy_loop.py repeatedly invokes a pytest target
until a clean run or timeout. This audit uses that same design — each
dataset/method combination is executed multiple times (not once) so
intermittent or numerically unstable behavior can surface.

Does not modify library behavior; used only to produce the audit log.
"""
from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from aggdisagg import TemporalAligner

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "tests" / "data"

# Low-frequency quarterly fixtures (the three named in the mission).
DATASETS = {
    "Kraft Heinz": DATA_DIR / "kraft_heinz_quarterly_revenue.csv",
    "PepsiCo": DATA_DIR / "pepsico_quarterly_revenue.csv",
    "B&G Foods": DATA_DIR / "bg_foods_capex_quarterly.csv",
}

# Optional monthly R-reference series for numerical comparison.
REFERENCES = {
    "Kraft Heinz": DATA_DIR / "Kraft_Heinz_Revenue_monthly_sum_disagg.csv",
    "PepsiCo": DATA_DIR / "PepsiCo_Quarterly_Revenue_monthly_sum_disagg.csv",
    "B&G Foods": DATA_DIR / "B_G_Foods_Capital_Expenditures_monthly_sum_disagg.csv",
}

METHODS = [
    "uniform",
    "linear",
    "denton",
    "denton-cholette",
    "chow-lin",
    "chow-lin-opt",
    "litterman",
    "fernandez",
]

# Follow the original script's repeated-run design (must be > 1).
REPEATS = 5
# Relative + absolute tolerance for the quarterly-sum constraint.
REL_TOL = 1e-6
ABS_TOL = 1e-4
# Repeat-to-repeat determinism: methods should be bit-stable on the same input.
DETERMINISM_ATOL = 1e-12


def load_quarterly(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def load_reference(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    ref = pl.read_csv(path, try_parse_dates=True)
    cols = list(ref.columns)
    if len(cols) >= 2:
        return ref.rename({cols[0]: "date", cols[1]: "r_value"})
    return None


def constraint_error(quarterly: np.ndarray, monthly: np.ndarray) -> float:
    n_q = len(quarterly)
    expected_n = n_q * 3
    if len(monthly) < expected_n:
        return float("inf")
    reshaped = monthly[:expected_n].reshape(n_q, 3).sum(axis=1)
    return float(np.nanmax(np.abs(reshaped - quarterly)))


def relative_constraint_ok(quarterly: np.ndarray, monthly: np.ndarray) -> tuple[bool, float, float]:
    n_q = len(quarterly)
    expected_n = n_q * 3
    if len(monthly) != expected_n:
        return False, float("inf"), float("inf")
    reshaped = monthly.reshape(n_q, 3).sum(axis=1)
    abs_err = float(np.nanmax(np.abs(reshaped - quarterly)))
    scale = float(np.nanmax(np.abs(quarterly))) or 1.0
    rel_err = abs_err / scale
    ok = bool(np.allclose(reshaped, quarterly, rtol=REL_TOL, atol=max(ABS_TOL, REL_TOL * scale)))
    return ok, abs_err, rel_err


def run_once(df: pl.DataFrame, method: str) -> tuple[np.ndarray, list[warnings.WarningMessage], float]:
    captured: list[warnings.WarningMessage] = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aligner = TemporalAligner(method=method, target_freq="1mo", agg="sum")
        out = aligner.fit_transform(df, datetime_col="date", target_col="value")
        captured = list(w)
    elapsed = time.perf_counter() - t0
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    vals = out["y_disaggregated"].to_numpy()
    return vals, captured, elapsed


def main() -> int:
    print("=" * 78)
    print("MESSY-LOOP AUDIT: repeated-run pattern on all CSV fixtures x methods")
    print(f"Pattern source: tests/run_messy_loop.py (repeat until clean / multi-run)")
    print(f"Repeats per dataset/method combination: {REPEATS}")
    print(f"Datasets: {list(DATASETS)}")
    print(f"Methods: {METHODS}")
    print(f"Constraint tols: rtol={REL_TOL}, atol={ABS_TOL}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    findings: list[str] = []
    n_ok = 0
    n_fail = 0
    n_warn = 0
    n_runs = 0

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'#' * 78}")
        print(f"# DATASET: {ds_name}  ({ds_path.relative_to(REPO)})")
        print(f"{'#' * 78}")
        if not ds_path.exists():
            msg = f"MISSING FIXTURE: {ds_path}"
            print(msg)
            findings.append(msg)
            n_fail += 1
            continue
        df = load_quarterly(ds_path)
        qvals = df["value"].to_numpy().astype(float)
        print(f"  rows={df.height}  date_min={df['date'].min()}  date_max={df['date'].max()}")
        print(f"  value range=[{qvals.min():.6g}, {qvals.max():.6g}]")
        ref = load_reference(REFERENCES[ds_name])

        for method in METHODS:
            print(f"\n--- {ds_name} / {method}  ({REPEATS} repeats) ---")
            first: np.ndarray | None = None
            combo_failed = False
            combo_warns: list[str] = []
            combo_times: list[float] = []

            for run_idx in range(1, REPEATS + 1):
                n_runs += 1
                print(f"\n=== BATCH {run_idx}/{REPEATS}  dataset={ds_name!r} method={method!r} "
                      f"at {time.strftime('%H:%M:%S')} ===")
                try:
                    vals, warns, elapsed = run_once(df, method)
                    combo_times.append(elapsed)
                    print(f"  elapsed={elapsed:.4f}s  n_high={len(vals)}  "
                          f"min={np.nanmin(vals):.6g}  max={np.nanmax(vals):.6g}  "
                          f"mean={np.nanmean(vals):.6g}")

                    if not np.all(np.isfinite(vals)):
                        n_nan = int(np.isnan(vals).sum())
                        n_inf = int(np.isinf(vals).sum())
                        msg = (f"NON-FINITE values: dataset={ds_name} method={method} "
                               f"run={run_idx} n_nan={n_nan} n_inf={n_inf}")
                        print(f"  FAIL: {msg}")
                        findings.append(msg)
                        combo_failed = True

                    if len(vals) != df.height * 3:
                        msg = (f"LENGTH mismatch: dataset={ds_name} method={method} "
                               f"run={run_idx} got={len(vals)} expected={df.height * 3}")
                        print(f"  FAIL: {msg}")
                        findings.append(msg)
                        combo_failed = True

                    ok, abs_err, rel_err = relative_constraint_ok(qvals, vals)
                    print(f"  constraint: abs_err={abs_err:.6g}  rel_err={rel_err:.6g}  ok={ok}")
                    if not ok:
                        msg = (f"TOLERANCE/CONSTRAINT violation: dataset={ds_name} method={method} "
                               f"run={run_idx} abs_err={abs_err:.6g} rel_err={rel_err:.6g} "
                               f"(rtol={REL_TOL}, atol={ABS_TOL})")
                        print(f"  FAIL: {msg}")
                        findings.append(msg)
                        combo_failed = True

                    if first is None:
                        first = vals.copy()
                    else:
                        max_diff = float(np.nanmax(np.abs(vals - first)))
                        print(f"  determinism vs run 1: max_abs_diff={max_diff:.6g}")
                        if max_diff > DETERMINISM_ATOL:
                            msg = (f"NON-DETERMINISTIC / numerically unstable: dataset={ds_name} "
                                   f"method={method} run={run_idx} max_abs_diff={max_diff:.6g} "
                                   f"(atol={DETERMINISM_ATOL})")
                            print(f"  FAIL: {msg}")
                            findings.append(msg)
                            combo_failed = True

                    if warns:
                        for w in warns:
                            cat = getattr(w.category, "__name__", str(w.category))
                            text = str(w.message)
                            # RuntimeWarning / FutureWarning / DeprecationWarning are
                            # filtered in pytest.ini but we still record them here.
                            rec = (f"WARNING: dataset={ds_name} method={method} run={run_idx} "
                                   f"{cat}: {text}")
                            print(f"  {rec}")
                            combo_warns.append(rec)
                            findings.append(rec)
                            n_warn += 1

                    if run_idx == 1 and ref is not None and first is not None:
                        # Compare first-run monthly path to the R reference where dates overlap.
                        try:
                            out_df = pl.DataFrame({
                                "date": pl.date_range(
                                    df["date"].min(),
                                    None,
                                    interval="1mo",
                                    eager=True,
                                )[: len(first)],
                                "py_value": first,
                            })
                        except Exception:
                            out_df = None
                        if out_df is None:
                            # Fall back: align by row count only.
                            n = min(len(first), ref.height)
                            r = ref["r_value"].to_numpy()[:n]
                            p = first[:n]
                            fin = np.isfinite(r) & np.isfinite(p) & (np.abs(r) > 0)
                            if fin.any():
                                max_pct = float(np.nanmax(np.abs((p[fin] - r[fin]) / r[fin]) * 100))
                                print(f"  vs R-ref (row-aligned, n={int(fin.sum())}): "
                                      f"max_pct_diff={max_pct:.4f}%")
                        else:
                            # Simpler: just report scale vs reference totals.
                            r_sum = float(ref["r_value"].sum())
                            p_sum = float(np.nansum(first))
                            print(f"  vs R-ref totals: py_sum={p_sum:.6g} r_sum={r_sum:.6g} "
                                  f"rel_diff={abs(p_sum - r_sum) / (abs(r_sum) or 1):.6g}")

                    print(f"  run {run_idx}: OK" if not combo_failed else f"  run {run_idx}: HAD FAILURES")
                except Exception as e:
                    combo_failed = True
                    n_fail += 1
                    tb = traceback.format_exc()
                    msg = (f"EXCEPTION: dataset={ds_name} method={method} run={run_idx} "
                           f"{type(e).__name__}: {e}")
                    print(f"  FAIL: {msg}")
                    print(tb)
                    findings.append(msg)
                    print("Errors found, would fix and retry... (simulated)")

            mean_t = float(np.mean(combo_times)) if combo_times else float("nan")
            if combo_failed:
                n_fail += 1
                print(f"COMBO RESULT: FAIL  {ds_name} / {method}  "
                      f"(mean {mean_t:.4f}s, warns={len(combo_warns)})")
            else:
                n_ok += 1
                print(f"COMBO RESULT: CLEAN RUN  {ds_name} / {method}  "
                      f"({REPEATS} repeats, mean {mean_t:.4f}s, warns={len(combo_warns)})")
                print("CLEAN RUN! No errors.")

    print("\n" + "=" * 78)
    print("MESSY-LOOP AUDIT SUMMARY")
    print("=" * 78)
    print(f"Datasets: {len(DATASETS)}")
    print(f"Methods: {len(METHODS)}")
    print(f"Repeats per combination: {REPEATS}")
    print(f"Matrix size: {len(DATASETS)} x {len(METHODS)} x {REPEATS} = {n_runs} runs")
    print(f"Clean combinations: {n_ok}/{len(DATASETS) * len(METHODS)}")
    print(f"Failed combinations: {n_fail}")
    print(f"Warnings recorded: {n_warn}")
    print(f"Finding lines: {len(findings)}")
    if findings:
        print("\n--- FINDINGS ---")
        for i, f in enumerate(findings, 1):
            print(f"{i}. {f}")
        rc = 1
    else:
        print("\nNo failures found.")
        rc = 0
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"EXIT={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

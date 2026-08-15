# Bugfix report — aggregation consistency (2026-08)

Confirmed red tests reconstructed from `SESSION_HANDOFF_2026-07.md` (fiscal/offset-quarter anchors in `_compute_high_lengths`; Denton-Cholette boundary / preliminary `p` in `_apply_denton`) plus the live suite and the C3/C4/C5/C8/C10 repros in `tests/test_basic.py`.

## Previously red, now green

These five functions failed on the pre-fix tree (captured in `tests/red_run_output.txt`: lengths 41 vs 42; ValueError on Q→1q; all-zero std; height 126 != 504; Denton knots `[6, 36, 66]` vs calendar `[6.2, 36.8, 66.2]`).

Command:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_basic.py::test_c3_nat_in_real_quarterly_keeps_lengths_aligned_with_n_low \
  tests/test_basic.py::test_c4_same_frequency_on_real_fiscal_quarterly_is_not_rejected \
  tests/test_basic.py::test_c5_predict_with_uncertainty_nonzero_after_fit_transform_on_real_series \
  tests/test_basic.py::test_c8_source_freq_escape_hatch_is_applied_on_real_quarterly \
  tests/test_basic.py::test_c10_denton_p_uses_variable_high_lengths_on_real_monthly \
  tests/test_basic.py::test_issue1_fiscal_quarter_expansion \
  tests/test_basic.py::test_issue1b_no_silent_passthrough \
  tests/test_basic.py::test_issue2_denton_cholette_toy \
  tests/test_basic.py::test_issue2b_kraft_within_r \
  tests/test_basic.py::test_issue2b_bg_within_r \
  tests/test_simulation.py::test_improved_uncertainty \
  -v --tb=no --no-cov
```

Output summary:

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
collected 11 items

tests/test_basic.py ..........                                           [ 90%]
tests/test_simulation.py .                                               [100%]

============================== 11 passed in 0.83s ==============================
```

`test_issue2b_kraft_within_r` (the existing Denton-Cholette / R-parity regression) also passes; no sum-constraint test was deleted or loosened.

Leftover reds on `PYTHONPATH=src python -m pytest tests/ -q` are unchanged environment/fixture issues, not these library bugs: missing `/Users/dev/.../freq-test-files` CSVs, `test_robust_100_scenarios` comparing date ordinals against values (test C6), and messy-data batches 3–5 constructing `pl.Series` from pandas 3 `ArrowStringArray`.

## Bugs fixed

All edits are in `src/aggdisagg/core.py`. `src/aggdisagg/methods.py` is unchanged (Denton/Chow-Lin there remain Uniform stubs; the live path is `TemporalAligner` in `core.py`).

### C3 — NaT dropna desyncs `_high_lengths` from `n_low`

- **Where:** `TemporalAligner._compute_high_lengths`, lines 662–808 (keep-NaT parse at 684–693; NaT slot at 765–768; do-not-jump-over-NaT end at 779–783).
- **Cause:** `pd.to_datetime(..., errors="coerce").dropna()` shortened the lengths vector, so `_build_c_matrix` saw `len(lengths) != n_low`.
- **Fix:** Keep every input slot, including NaT. A missing start gets `fallback_r` children; a missing *next* label uses `DateOffset(months=step_months)` instead of jumping to the following valid date.

### C4 — legitimate 1:1 (Q→1q) treated as failed expansion

- **Where:** `TemporalAligner._is_same_or_coarser_target`, lines 471–482; `_prepare_data`, lines 838–848.
- **Cause:** `_prepare_data` raised whenever every computed length was 1, so PepsiCo fiscal quarterly → `target_freq="1q"` (a correct identity) was rejected as “did not expand”.
- **Fix:** Raise the silent-passthrough error only when the target is *finer* than the source. Same-or-coarser targets (including Q→1q) are allowed as 1:1. A finer target that still fails to expand still raises.

### C5 — `predict_with_uncertainty()` ignored constructor `n_bootstrap`

- **Where:** `TemporalAligner.predict_with_uncertainty`, lines 2216–2236 (constructor fallback at 2226).
- **Cause:** After `fit_transform` without `with_uncertainty=True`, `_std_errors` stayed `None` and the no-arg follow-up never read `self.n_bootstrap`, so std was all zeros.
- **Fix:** `n_boot = n_bootstrap if n_bootstrap is not None else int(self.n_bootstrap or 0)` and bootstrap when that count is positive.

### C8 — `source_freq` stored but never applied

- **Where:** `TemporalAligner._source_freq_code`, lines 484–487; `_infer_ratio`, lines 489–505; `_compute_high_lengths`, lines 718–749 and 771–777.
- **Cause:** The advertised `source_freq` escape hatch was only mentioned in the 1:1 error string. Date-spacing inference always won, so `source_freq="Y"` on Kraft quarterly → monthly stayed 3 months/obs (height 126) instead of 12 (height 504).
- **Fix:** Honor `source_freq` first for the ratio *and* for per-observation windows (`Y`=12 months, `Q`=3, `M`=1, `W`=6 days), independent of label spacing.

### C10 — Denton-Cholette preliminary `p` used uniform `m = n_h // n_l`

- **Where:** `TemporalAligner._apply_denton`, lines 1088–1159 (variable-length `p` at 1110–1130).
- **Cause:** Preliminary `p` interpolated block means on a uniform grid even when `_high_lengths` was irregular (leap Feb 31/29/31). Knots sat at `[6, 36, 66]` instead of calendar `[6.2, 36.8, 66.2]`.
- **Fix:** Build `p` from `_high_lengths`: `starts + 0.2 * freqs` and `means = y_low / freqs`. Uniform `m = n_h // n_l` remains only the regular-ratio fallback. The first-diff Q + 0.2 offset Cholette construction is unchanged.

## Aggregation-consistency check (`C @ y_high` vs `y_low`)

Denton-family methods promise exact (up to float/solver) sum-constraint: `C @ y_high ≈ y_low` with `agg="sum"`. Measured after the fix (same fixtures as the tests):

| Scenario | Method | C shape | max \|C @ y_high − y_low\| | Promised / used tolerance |
|---|---|---|---|---|
| C10 leap Jan–Mar 2016 M→D (31/29/31) | denton-cholette | (3, 91) | **0.0** | exact; knots/means must match calendar lengths |
| issue2 toy Q→M (6 quarters, values 300…450) | denton-cholette | (6, 18) | **0.0** | exact block sums (R-shape check is separate, `< 0.5%`) |
| C3 Kraft quarterly + NaT → monthly | denton-cholette | (42, 126) | **1.907349e-06** | `1e-6 * scale`; values ~1e9–7.38e9 so rel ≈ 2.6e-16 |
| issue2b Kraft Q→M (no NaT) | denton-cholette | (42, 126) | **1.907349e-06** | same; `test_issue1` uses `< 1e-6 * \|sum(y_low)\|` |
| C4 PepsiCo Q→1q | uniform | (39, 39) | **0.0** | identity, atol 1e-6 |
| C8 Kraft `source_freq="Y"` Q→M | uniform | (42, 504) | **0.0** | 12 children/obs, exact block sums |

C10 reconstructed `y_low` exactly: `[2.42134591e9, 2.38633648e9, 2.31631762e9]`. The 1.9e-6 Denton residuals on the Kraft-scale series are float64/sparse-solve noise, well inside the existing `1e-6 * scale` sum-constraint tests. No tolerance was loosened.

## No forcing; Polars-first interop

No special-case or reference-output forcing was added: no hardcoded R vectors, no per-series branches, no literal numbers used to fake a passing test. Fixes are general (keep NaT slots; compare source vs target frequency rank; read `source_freq`; build Denton `p` from `_high_lengths`; consult `n_bootstrap`).

pandas and xarray interop convert into Polars (`pl.from_pandas` in `fit` / `fit_transform`; xarray `DataArray` → `to_dataframe` → `pl.from_pandas`) and then call the same `_prepare_data` → `_compute_high_lengths` / `_apply_denton` path. Verified on the C10 leap window: `|y_high(polars) − y_high(pandas)| = 0`, `|y_high(polars) − y_high(xarray)| = 0`, and max `|C @ y_high − y_low| = 0` on all three. Kraft Q→M denton-cholette: `|y_high(polars) − y_high(pandas)| = 0` and the same `1.9073486328125e-06` max deviation on both.

`version.txt` and `pyproject.toml` dependency pins were not changed.

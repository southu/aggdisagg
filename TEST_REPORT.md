# aggdisagg test-suite audit

- **Date:** 2026-08-15
- **Repo:** https://github.com/southu/aggdisagg
- **Commit under test:** `efe6acc` (`docs: put unambiguous SSE3 (a) classification at the top`)
- **Environment:** Linux, CPython 3.12.3, `uv 0.12.5`, pytest 9.1.1, pytest-cov 7.1.0
- **Library version:** 1.11.0 (`pyproject.toml` / `src/aggdisagg/__init__.py`)
- **`version.txt`:** not present in the repository; it was not created or modified.

This is an audit-only report. Library behavior was not changed.

Full captured stdout/stderr lives in:

- [`artifacts/pytest_output.log`](artifacts/pytest_output.log)
- [`artifacts/messy_loop_output.log`](artifacts/messy_loop_output.log)

The messy-loop runner used to exercise the fixture × method matrix is
[`artifacts/run_messy_loop_audit.py`](artifacts/run_messy_loop_audit.py).

---

## Pytest

### Command

```bash
uv run pytest --cov=aggdisagg
```

`pyproject.toml` `[tool.pytest.ini_options].addopts` already injects
`-ra -q --cov=aggdisagg --cov-report=term-missing`. The capture also passed
`-vv --tb=short` so every collected test name, skip reason, failure, warning,
and the coverage table appear in `artifacts/pytest_output.log`. That file is
the raw pytest stdout/stderr; the summary banner was not rewritten after
capture.

Collected: **122** items (`tests/test_basic.py` + `tests/test_simulation.py`).

### Totals

| Result | Count |
|---|---|
| Passed | **59** |
| Failed | **61** |
| Skipped | **2** |
| Errors | **0** |
| Warnings | **71** |
| Duration | 78.65s |

Final pytest line (verbatim from `artifacts/pytest_output.log`):

```
=========================== 61 failed, 59 passed, 2 skipped, 71 warnings in 78.65s (0:01:18) ===========================
```

### Coverage

**75%** total (`1401` statements, `346` missed).

| Module | Stmts | Miss | Cover |
|---|---:|---:|---:|
| `src/aggdisagg/__init__.py` | 10 | 0 | 100% |
| `src/aggdisagg/api.py` | 74 | 0 | 100% |
| `src/aggdisagg/conversion.py` | 37 | 0 | 100% |
| `src/aggdisagg/core.py` | 1224 | 345 | 72% |
| `src/aggdisagg/methods.py` | 56 | 1 | 98% |
| **TOTAL** | **1401** | **346** | **75%** |

---

## Messy-loop matrix

`tests/run_messy_loop.py` repeatedly invokes a single pytest target until a
clean run or timeout. That same **repeat-until-stable / multi-run** design was
applied to every quarterly CSV fixture in `tests/data/` across all 8 methods.

| Axis | Values |
|---|---|
| Datasets (3) | Kraft Heinz (`tests/data/kraft_heinz_quarterly_revenue.csv`), PepsiCo (`tests/data/pepsico_quarterly_revenue.csv`), B&G Foods (`tests/data/bg_foods_capex_quarterly.csv`) |
| Methods (8) | `uniform`, `linear`, `denton`, `denton-cholette`, `chow-lin`, `chow-lin-opt`, `litterman`, `fernandez` |
| Repeat count | **5** per dataset/method combination (must be > 1 to surface intermittent failures) |
| Matrix size | **3 × 8 × 5 = 120 runs** |
| Command | `uv run python artifacts/run_messy_loop_audit.py` |

Each run disaggregated quarterly `date`/`value` to monthly (`target_freq="1mo"`,
`agg="sum"`) and checked:

- no exception
- finite values
- output length = `3 * n_quarters`
- quarterly-sum constraint (`rtol=1e-6`, `atol=1e-4`)
- bit-stability vs run 1 (`max_abs_diff` ≤ `1e-12`)
- captured `warnings`

Result: **24/24 combinations clean**, **0 exceptions**, **0 constraint
violations**, **0 non-deterministic repeats**, **0 warnings**. Every combo
printed `CLEAN RUN! No errors.`

The three monthly CSVs in `tests/data/`
(`Kraft_Heinz_Revenue_monthly_sum_disagg.csv`,
`PepsiCo_Quarterly_Revenue_monthly_sum_disagg.csv`,
`B_G_Foods_Capital_Expenditures_monthly_sum_disagg.csv`) are R-reference
outputs, not additional low-frequency inputs. They were used only for an
informational first-run percent-diff printout (not as pass/fail).

---

## Findings

The pytest suite is **not** clean. The fixture × method messy-loop matrix **is**
clean. Itemized observations follow.

### A. Pytest skips (2)

1. `tests/test_basic.py::test_gls_analytical_uncertainty_coverage_regression` —
   `pytest.skip`: test signal files not available at
   `/Users/dev/Documents/GitHub/scrap-testing-delme/freq-test-files/`.
2. `tests/test_basic.py::test_fit_transform_return_dataframe_and_include_dates` —
   `pytest.skip`: test data not available
   (`/Users/dev/Documents/GitHub/scrap-testing-delme/freq-test-files/signal-quarterly.csv`).

### B. Pytest failures — missing developer-machine paths (18)

These tests hard-code absolute paths that do not exist in this checkout.
Reproduce with `uv run pytest <nodeid>`.

| # | Node id | Error |
|---|---|---|
| 1 | `tests/test_basic.py::test_general_calendar_variable_ratios[yearly-1d-2026-12-31-6900-None]` | `FileNotFoundError: .../signal-yearly.csv` |
| 2 | `tests/test_basic.py::test_general_calendar_variable_ratios[quarterly-1d-2026-06-30-4500-None]` | `FileNotFoundError: .../signal-quarterly.csv` |
| 3 | `tests/test_basic.py::test_general_calendar_variable_ratios[weekly-1d-None-2240-max_diff_1day]` | `FileNotFoundError: .../signal-weekly.csv` |
| 4 | `tests/test_basic.py::test_general_calendar_variable_ratios[yearly-1q-None-70-quarterly_step]` | `FileNotFoundError: .../signal-yearly.csv` |
| 5 | `tests/test_basic.py::test_no_crash_on_negatives_with_irregular_M_to_D` | `FileNotFoundError: .../signal-monthly.csv` |
| 6 | `tests/test_basic.py::test_standalone_aggregate_calendar_aware` | `FileNotFoundError: .../signal-daily.csv` |
| 7 | `tests/test_basic.py::test_install_note_and_excel_extra` | `FileNotFoundError: /Users/dev/Documents/GitHub/aggdisagg/pyproject.toml` |
| 8 | `tests/test_basic.py::test_161_week_to_coarser_flow_conserves_mass_and_calendar_counts` | `FileNotFoundError: .../signal-weekly.csv` |
| 9 | `tests/test_basic.py::test_161_default_aggregate_is_flow_not_auto_stock` | `FileNotFoundError: .../signal-daily.csv` |
| 10 | `tests/test_basic.py::test_161_aggregate_sets_detected_semantics` | `FileNotFoundError: .../signal-daily.csv` |
| 11 | `tests/test_basic.py::test_161_aggregate_no_pandas_required_for_calendar` | `FileNotFoundError: .../signal-daily.csv` |
| 12 | `tests/test_basic.py::test_161_nesting_aggregations_match_groupby_and_preserve_semantics` | `FileNotFoundError: .../signal-daily.csv` |
| 13 | `tests/test_basic.py::test_162_auto_detection_restored_and_symmetric` | `FileNotFoundError: .../signal-quarterly.csv` |
| 14 | `tests/test_basic.py::test_170_methods_are_distinct_and_machinery_works` | `FileNotFoundError: .../signal-quarterly.csv` |
| 15 | `tests/test_basic.py::test_180_week_start_and_partial_weeks` | `FileNotFoundError: .../signal-daily.csv` |
| 16 | `tests/test_basic.py::test_180_denton_cholette_and_perf` | `FileNotFoundError: .../signal-quarterly.csv` |
| 17 | `tests/test_basic.py::test_162_ambiguous_trending_flow_emits_warning_and_records_actual` | `FileNotFoundError: .../signal-daily.csv` |
| 18 | `tests/test_basic.py::test_181_weekly_source_freq_detection_any_anchor` | `FileNotFoundError: .../signal-weekly.csv` |

Base path for the signal files:
`/Users/dev/Documents/GitHub/scrap-testing-delme/freq-test-files/`.

### C. Pytest failures — numerical / assertion (2)

19. **`tests/test_simulation.py::test_improved_uncertainty`**
    - Dataset: synthetic 4 annual points `[100, 110, 105, 130]`,
      2020–2023 (`freq="YE"`).
    - Method: `uniform`, `target_freq="1mo"`, `n_bootstrap=20`.
    - Error: `assert (np.False_ or np.float64(0.0) > 0)` —
      `predict_with_uncertainty()` returned an all-zero `std` array of length 48.
    - Reproduce: `uv run pytest tests/test_simulation.py::test_improved_uncertainty -vv`.

20. **`tests/test_simulation.py::test_robust_100_scenarios`**
    - Deterministic combo sample (`np.random.default_rng(42)`, 100 of the
      product of methods × conversions × freqs × sizes × flags).
    - Seed for scenario `idx` is `1000 + idx`.
    - Assertion: `assert passed >= 95` failed with **0/100** scenarios passing.
    - First five scenario failures (all `np.allclose` on the aggregate
      roundtrip). The left-hand array starts with `17897`, which is a year
      ordinal leaking in via `back.to_numpy().ravel()[:n_low]` (date column
      mixed with values), compared against the generated `y`:
      - idx 0, method=`denton`, seed=1000
      - idx 1, method=`fernandez`, seed=1001
      - idx 2, method=`litterman`, seed=1002
      - idx 3, method=`denton`, seed=1003
      - idx 4, method=`denton`, seed=1004
    - Reproduce: `uv run pytest tests/test_simulation.py::test_robust_100_scenarios -vv`.

### D. Pytest failures — messy-data batches, `lengths length must match n_low` (39)

Raised from `src/aggdisagg/core.py` (`_build_c_matrix`) during
`TemporalAligner.fit_transform`. Batches 1, 2, 4, 5 passed; 3 and 6–43 failed
with the same message.

21–59. `tests/test_simulation.py::test_messy_incomplete_data_batch_{3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43}`
    - Error: `ValueError: lengths length must match n_low`
    - These batches construct messy/incomplete inputs (NaNs, NaTs, timezone
      dates, object dates, gaps, duplicates) and then call `fit_transform`.
    - Reproduce one example:
      `uv run pytest tests/test_simulation.py::test_messy_incomplete_data_batch_3 -vv --tb=short`.

### E. Pytest failures — calendar expansion (2)

60. `tests/test_simulation.py::test_messy_incomplete_data_batch_46` —
    `ValueError: Disaggregation did not expand the series (output length 4 == input 4). The low-frequency dates could not be expanded calendar-correctly to the target frequency. Use standard calendar-aligned dates or the source_freq=... parameter for explicit control.`
61. `tests/test_simulation.py::test_messy_incomplete_data_batch_47` —
    same `ValueError` (output length 4 == input 4).

Reproduce:
`uv run pytest tests/test_simulation.py::test_messy_incomplete_data_batch_46 tests/test_simulation.py::test_messy_incomplete_data_batch_47 -vv`.

### F. Pytest warnings (71)

All 71 are the same `UserWarning` from `src/aggdisagg/core.py` (emitted at
`fit_transform` call sites around lines 1564 and 1580), text:

> NaN values present in disaggregated series for one or more high-frequency
> periods (caused by NaN in corresponding low-frequency input values or
> end-of-range). Current extrapolate='…'. Default 'nan' leaves NaN-input
> periods honest (no fabrication); use 'hold'/'linear' to fill or 'drop' to
> shorten.

| Test | Count | `extrapolate` |
|---|---:|---|
| `test_extrapolate_hold_fills_nan_input_blocks` | 2 | `hold` |
| `test_extrapolate_drop_shortens_output` | 2 | `drop` |
| `test_aggregate_preserves_per_group_nan_from_incomplete_tail` | 2 | `nan` |
| `test_messy_incomplete_data_batch_1` (module-level group) | 11 | `nan` |
| `test_messy_incomplete_data_batch_2` | 9 | `nan` |
| `test_messy_incomplete_data_batch_3` | 1 | `nan` |
| `test_messy_incomplete_data_batch_4` (module-level group) | 10 | `nan` |
| `test_messy_incomplete_data_batch_5` | 7 | `nan` |
| `test_messy_incomplete_data_batch_6` | 2 | `nan` |
| `test_messy_incomplete_data_batch_7` | 4 | `nan` |
| `test_messy_incomplete_data_batch_8` | 2 | `nan` |
| `test_messy_incomplete_data_batch_9` | 4 | `nan` |
| `test_messy_incomplete_data_batch_11` | 1 | `nan` |
| `test_messy_incomplete_data_batch_12` | 1 | `nan` |
| `test_messy_incomplete_data_batch_13` | 2 | `nan` |
| `test_messy_incomplete_data_batch_14` | 2 | `nan` |
| `test_messy_incomplete_data_batch_38` | 4 | `nan` |
| `test_messy_incomplete_data_batch_41` | 1 | `nan` |
| `test_messy_incomplete_data_batch_42` | 2 | `nan` |
| `test_messy_incomplete_data_batch_43` | 2 | `nan` |

Full warning text and file:line are in `artifacts/pytest_output.log` under
`warnings summary`. `filterwarnings` in `pyproject.toml` ignores
`FutureWarning` / `DeprecationWarning` / `RuntimeWarning`; these `UserWarning`s
are not filtered.

### G. Messy-loop fixture × method matrix

Every combination below was run **5 times**. Constraint, finiteness, length,
and determinism all held on every repeat. No warnings were raised.

| Dataset | Method | Repeats | Constraint | Determinism | Result |
|---|---|---:|---|---|---|
| Kraft Heinz | uniform | 5 | ok (abs_err=0) | max_abs_diff=0 | CLEAN |
| Kraft Heinz | linear | 5 | ok (abs_err=1.91e-06, rel=2.58e-16) | 0 | CLEAN |
| Kraft Heinz | denton | 5 | ok (abs_err=1.91e-06) | 0 | CLEAN |
| Kraft Heinz | denton-cholette | 5 | ok (abs_err=9.54e-07) | 0 | CLEAN |
| Kraft Heinz | chow-lin | 5 | ok (abs_err=1.91e-06) | 0 | CLEAN |
| Kraft Heinz | chow-lin-opt | 5 | ok (abs_err=1.91e-06) | 0 | CLEAN |
| Kraft Heinz | litterman | 5 | ok | 0 | CLEAN |
| Kraft Heinz | fernandez | 5 | ok | 0 | CLEAN |
| PepsiCo | uniform | 5 | ok | 0 | CLEAN |
| PepsiCo | linear | 5 | ok | 0 | CLEAN |
| PepsiCo | denton | 5 | ok | 0 | CLEAN |
| PepsiCo | denton-cholette | 5 | ok | 0 | CLEAN |
| PepsiCo | chow-lin | 5 | ok | 0 | CLEAN |
| PepsiCo | chow-lin-opt | 5 | ok | 0 | CLEAN |
| PepsiCo | litterman | 5 | ok (abs_err=3.81e-06) | 0 | CLEAN |
| PepsiCo | fernandez | 5 | ok (abs_err=7.63e-06) | 0 | CLEAN |
| B&G Foods | uniform | 5 | ok (abs_err=1.91e-06) | 0 | CLEAN |
| B&G Foods | linear | 5 | ok | 0 | CLEAN |
| B&G Foods | denton | 5 | ok (abs_err=9.54e-07) | 0 | CLEAN |
| B&G Foods | denton-cholette | 5 | ok (abs_err=1.91e-06) | 0 | CLEAN |
| B&G Foods | chow-lin | 5 | ok (abs_err=3.81e-06) | 0 | CLEAN |
| B&G Foods | chow-lin-opt | 5 | ok (abs_err=3.81e-06) | 0 | CLEAN |
| B&G Foods | litterman | 5 | ok | 0 | CLEAN |
| B&G Foods | fernandez | 5 | ok | 0 | CLEAN |

Informational only (not scored as a failure): first-run max percent difference
vs the R monthly reference, which is a **denton-cholette** path. Comparing
other methods to that reference is expected to be large, especially on the
volatile B&G series.

| Dataset | Method | max % vs R-ref (row-aligned) |
|---|---|---:|
| Kraft Heinz | denton-cholette | 1.7128% (inside existing `test_issue2b_kraft_within_r` bound of 2%) |
| B&G Foods | denton-cholette | 26.7176% (inside existing `test_issue2b_bg_within_r` bound of 30%) |
| PepsiCo | fernandez | 0.0572% |
| B&G Foods | fernandez | 0.1139% |
| B&G Foods | uniform | 51.1681% (expected: reference is not uniform) |

No intermittent / flaky behavior was observed across the 5 repeats of any
combo.

---

## Notes

- In-repo regression tests that use the bundled fixtures
  (`test_issue1_fiscal_quarter_expansion`, `test_issue1b_no_silent_passthrough`,
  `test_issue2_denton_cholette_toy`, `test_issue2b_kraft_within_r`,
  `test_issue2b_bg_within_r`) all **passed**.
- `tests/run_messy_loop.py` itself targets only
  `tests/test_simulation.py::test_messy_incomplete_data_batch_1`, which
  **passed** in the suite (so a single invocation of that script would exit 0
  on the first batch). The audit therefore reused its repeat design against
  the full 3 × 8 fixture/method matrix instead of only that one node id.
- No library source, `version.txt`, or test behavior was modified for this
  audit.

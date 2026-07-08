# Changelog

All notable changes to aggdisagg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.1] - 2026-07-07

### Fixed
- Weekly source frequency detection (for disaggregation, e.g. Week→Day) was still locked to Monday anchor: non-Monday weekly inputs (e.g. Sunday-anchored) caused `_compute_high_lengths` to infer only 1 day per week (via pandas `to_period("W")` + date_range starting at the label) instead of 7, yielding wrong n_high = n_weeks instead of n_weeks*7.
  - Root: the period-end logic assumed low date labels are always "starts" aligned with default 'W' (Monday), and the early span guard + infer didn't special-case weekly sources.
  - Fix: in `_compute_high_lengths`, for source low_f starting with "W", explicitly span [label, label+6] days and count target ticks in that (always 7 for daily target). This works for any anchor weekday in the provided labels, and is consistent with treating weekly 'date' as the week-start label (per week_start convention).
  - `_infer_ratio` already returned 7 for W->D; now lengths also does, so disagg uses correct expansion.
- Added regression tests covering both Monday and Sunday weekly files for W->D (height*7, consecutive dates), and W->D->W exact roundtrip using week_start="sunday".
- No regressions to Monday W->D, D->W behaviors, partial_weeks, mass cons, etc.

## [1.8.0] - 2026-07-07

### Added
- `week_start` (default "monday") and `partial_weeks` ("keep" default / "drop") parameters on `TemporalAligner`, with per-call overrides on `aggregate`, `disaggregate_columns`, `expand_high_freq_dates`.
  - Affects weekly binning, generation, and straddle allocation for all week-related paths (D→W, coarse→W, W→coarser, weekly date expansion).
  - Output week start dates always land on the chosen weekday.
  - Mass conservation invariant to week_start (under keep).
  - "keep" retains+warns on incomplete boundary weeks; "drop" removes them.
  - Invalid week_start raises with accepted names.
- Method fixes: `denton-cholette` now properly implemented and differs from `denton` (divergence near start, constraint preserved); `denton` (and cholette) use sparse solves for scalability (Month→Day fast).

### Fixed
- All previous guarantees (Monday default, mass cons, no regression on nesting/RT/stock-flow/calendar) preserved.
- Added tests for week feature and method fixes.

## [1.7.0] - 2026-07-07

### Added / Fixed
- **Major methods implementation (6/8 were stubs)**: `denton`/`denton-cholette`, `chow-lin`/`chow-lin-opt`, `litterman`, `fernandez` now produce genuinely distinct high-frequency series (not byte-identical to uniform).
  - Denton: quadratic penalty (first or second differences) via robust constrained optimization on the high-freq path; visibly smoother/non-flat vs uniform.
  - Chow-Lin family: full GLS with AR(1) (or IAR(1) for Litterman) covariance; output responds to `indicator_cols` and `rho` (opt for *-opt*).
  - Litterman uses IAR(1) cov; Fernandez is the rho=0 random-walk special case.
  - All still satisfy exact (or 1e-6) aggregation constraint after final scaling; calendar/irregular ratios preserved.
- Added regression test `test_170_methods_are_distinct_and_machinery_works` exercising pairwise diffs, indicator/ρ effects, and constraint (using the quarterly test file).
- Updated README to reflect that all listed methods are now real/working.
- If a method were left unimplemented it would now raise rather than silently alias (per guidance).

## [1.6.2] - 2026-07-07

### Fixed
- **Regression from 1.6.1 (HIGH)**: Setting `autodetect_semantics=False` by default (and always falling back) caused `_detect_semantics` to return "flow" for *every* column, breaking stock detection for `stock_inventory`, `index_price`, `rate_interest` in both `disaggregate_columns` (wrong interp: sum instead of last-anchor) and `aggregate` (sum instead of last). Reverted default to `autodetect_semantics=True`.
- **Improved `_detect_semantics` heuristic**: Clear stocks (strictly monotonic cumulative like inventory; smooth low-rel levels like rate/index) now correctly → "stock". Sign-changing/mean-reverting → "flow". For the genuine ambiguity case (trending positive flows like `flow_sales` that look monotonic/large-level), emit a specific `UserWarning` naming the column + assumed semantics (defaults to "flow" for additive safety) instead of blanket "flow" or silent stock. `_detected_semantics` now always records the *actual* per-column decision.
- Same detection logic used symmetrically for disagg and agg paths → default round-trips are correct for clear cases.
- Added regression tests exercising the acceptance repros (quarterly index as stock + interp, daily default agg semantics, detected values not constant, warning on ambiguous).
- No regression to 1.6.1 fixes (week mass conservation exact, pyarrow-free calendar agg path) or prior calendar/nesting behavior. Explicit `col_semantics` continues to override.
- Re-ran full acceptance block + nesting matrix on daily/weekly/monthly/quarterly files; ruff + basic suite green.

## [1.6.1] - 2026-07-07

### Fixed
- **BUG1 (HIGH) week→coarser mass non-conservation**: Fixed logic error in weekly straddles where week_end policy added value 7× (inside k-loop) and prop had ~3% leakage on q/y. Both policies now conserve grand total of finite flows to <1e-6 (week_end assigns once to period of week-end date; proportional splits by exact day overlap fractions summing to 1.0). Also adjusted NaN handling for partial trailing periods (a target period with mixed NaN+finite children outputs agg of the finites; only pure-nan periods yield NaN). This preserves mass while keeping honest NaN for fully-missing buckets.
- **BUG2 (HIGH) auto-detect default mislabels trending flows as stock**: Changed `autodetect_semantics` default to `False` (was True). Default aggregation now uses `default_semantics="flow"` (sum) — the common case for flows — avoiding silent ~34× error (last instead of sum). Auto-detect remains available via `autodetect_semantics=True` (opt-in) or explicit `col_semantics`; heuristic is best-effort and documented as imperfect for monotone positive flows.
- **BUG3 (MEDIUM) pyarrow undeclared dep**: Rewrote standalone `aggregate()` calendar path to be Polars-first + stdlib `datetime` (no `to_pandas()`, no `pd.to_period`). `aggregate(..., datetime_col=...)` now works even if pandas/pyarrow not installed (pandas remains optional for other features).
- **BUG4 (MEDIUM) missing _detected_semantics on aggregate**: `aggregate()` now sets `self._detected_semantics` (symmetric with `disaggregate_columns`), using resolved per-col decisions (after respecting autodetect flag).
- Added guards in `_compute_high_lengths` so abstract integer "period" proxies (used in sims/tests) still expand correctly via regular ratios (non-date inputs no longer produce clen=1).
- Updated tests for all 4 bugs + full nesting matrix on freq files; re-ran daily/weekly/monthly/quarterly → all coarser levels; confirmed calendar group counts, pl.Date "date" col, correct flow/stock semantics, exact roundtrips (<1e-6), mass conservation.
- Ruff clean; basic test suite green.

## [1.6.0] - 2026-07-07

### Added / Fixed
- `aggregate()` is now first-class and calendar-aware, independent of prior disaggregation state when `datetime_col` is provided.
  - Groups by true calendar periods for target freq using date labels (honors variable month/quarter/year lengths).
  - Per-column semantics via `col_semantics` / auto `_detect_semantics` (flow→sum, stock→last); never sums stocks.
  - Outputs "date" (pl.Date) column with period-start labels.
  - NaN rule: any NaN child → NaN parent for that period/col.
  - `week_policy` ("week_end" | "proportional") for W→M/Q/Y straddles.
- Legacy cached path still used for exact round-trips when prior `_high_lengths` match the input (position-based using original blocks).
- Updated/added tests for standalone multi-level agg and roundtrips on the freq test files.
- No breakage to disagg side or prior M→D behavior.

## [1.5.1] - 2026-07-07

### Fixed
- Generalized calendar-aware variable child counts (`_compute_high_lengths`) and date expansion to **all** (source, target) pairs (Y/Q/M/W/D → any of 1y/1q/1mo/1w/1d). 1.5.0 only fixed M→D; others still used fixed ratios (91/30/~30.45) or emitted wrong target date steps (e.g. Y→1q produced monthly dates).
  - Lengths computed from each source period's true calendar end (via .to_period + end_time or delta) then count of target ticks in the span.
  - `expand_high_freq_dates` now selects correct freq (QS/YS/D/...) and uses accurate total periods → correct last date and spacing.
  - `_correct_negatives` accepts `lengths` and does per-parent `np.repeat(factor, parent_len)` (no broadcast crash on irregular + negatives).
- Added/expanded regression tests covering the 5+ pairs, correct per-year day counts (incl. leaps), quarterly steps in dates, exact W*7, last dates, and no-crash on signed monthly→daily.
- No regressions on previously-passing pairs or NaN/aggregate-exact behavior.
- All repro cases now pass with correct counts/ends/dates.

## [1.5.0] - 2026-07-07

### Added
- Full support for calendar-aware variable ratios on irregular pairs (notably monthly→daily). Each low period expands using its real number of high-frequency subperiods (actual days in month: 28-31). `C` matrices, value repeats, linear interp, date expansion, and group aggregation are now per-period instead of a single global integer ratio (no more force-fit to 30).
- `IrregularRatioError` exception for unsupported irregular frequency pairs (instead of silent wrong fixed-ratio results).

### Fixed
- **Irregular ratio silent correctness bug (HIGH)**: monthly→daily used fixed ratio=30 for all months → wrong total count (2340 instead of ~2373), drift (values assigned to wrong calendar days, e.g. Jan 2026 value on Dec 2025), and truncated last month. Now produces exact calendar days, dates align 1:1 with values, last date is true end of final period (e.g. 2026-06-30), and stock anchors land on actual month-end dates.
  - Affects `_infer`/`_compute_high_lengths`, `_prepare_data`, `_build_c_matrix`, `_apply_*`, `_aggregate_groups`, `expand_high_freq_dates`, scaling, nan handling, include_dates, denton etc.
- Added regression test verifying per-calendar-month daily row count equals real days in month + end-of-month anchor placement.
- Repro verification now passes: 2373 rows, ends 2026-06-30, zero drift on anchors.

### Changed
- Ratio handling for irregular cases is no longer silently approximate; this is a visible behavior change for M→D (and similar) → version 1.5.0.

## [1.4.2] - 2026-07-07

### Fixed
- **aggregate() NaN poisoning (HIGH)**: When the high-frequency result contained any NaN (e.g. honest NaN tail from default `extrapolate="nan"` on unreported final periods), `aggregate()` would return NaN for *every* low-frequency bucket because `C @ y_high` lets NaN (via 0*NaN) propagate to all rows. Now uses per-group reduction: a low group is NaN only if one of *its own* high-freq values is NaN; all other groups return their correct sum/mean/first/last. This restores correct round-tripping for `disaggregate_columns(...)` → `aggregate(...)` on incomplete data.
  - Updated fallback path in aggregate too.
  - `drop` mode now also keeps `_n_low` consistent with the shortened output.
- Added regression test exercising the exact round-trip with trailing NaN low quarters.
- Repro now yields 44 (or N-2) exact quarters + only the empty ones NaN in re-agg.

## [1.4.1] - 2026-07-07

### Fixed
- **Regression (include_dates)**: `expand_high_freq_dates` (and thus `disaggregate_columns(..., include_dates=True)`) now reliably expands to distinct high-frequency dates (e.g. 47 quarters → 141 unique monthly pl.Date). The pandas path is hardened and a pure-Python calendar stepper ensures the repeat-low fallback is never taken for dates. Updated docstring example (fit_transform does not emit a date column).
- **Missing public API**: `extrapolate` is now accepted on `fit_transform(..., extrapolate=...)` and `disaggregate_columns(..., extrapolate=...)` (forwarded with per-call override/restore). All four policies run and produce distinct length/value behavior.
- **Dangerous default (NaN inputs)**: Default `extrapolate="nan"` (was "hold"). NaN low-freq input values now produce honest NaN in the corresponding high-freq output months by default — the library no longer silently fabricates data for unreported periods. `extrapolate="hold"`/`"linear"` will fill when explicitly requested. `"drop"` shortens output by truncating after the last valid anchor. Warnings now specifically call out "NaN-input periods".
- Updated related tests, README, and docstrings. Existing behavior for fully-observed series is unchanged.

### Changed
- Default for `TemporalAligner(extrapolate=...)` is now `"nan"`.

## [1.4.0] - 2026-07-07

### Added
- `TemporalAligner.disaggregate_columns()` now supports per-column `stock` vs `flow` semantics via `col_semantics`, `default_semantics`, and `autodetect_semantics` (with heuristic and override). Exposes `_detected_semantics`.
- New `extrapolate` parameter on `TemporalAligner` ("hold" default, "linear", "nan", "drop") to control handling of NaN in low-freq input / end-of-range in disaggregation.
- `detect_semantics()` convenience on aligner.
- Excel extra in packaging (`aggdisagg[excel]`) for fastexcel/openpyxl.
- Regression tests for NaN handling, exact aggregation per method, stock/flow auto+override, multi-col roundtrip, date dtype, README examples.

### Changed
- `aggregate()` now supports multi-column high-freq frames (from `disaggregate_columns`); preserves original column names and recovers correctly using per-col semantics/agg when available.
- `include_dates=True` (and `expand_high_freq_dates`) now returns native `pl.Date` (was Object of python dates).
- Final per-group C scaling now robust to NaN in y_h (prevents pollution of valid groups).
- Default `extrapolate="hold"` ensures the final period is never silently dropped to NaN.
- README updated with prominent Python >=3.10 note and Excel extra guidance; added detailed quarterly-to-monthly multi-series example using the helper (now runs as test).
- Version bumped to 1.4.0 (breaking behavior changes for NaN policy, aggregate multi-col, date dtype in some paths; documented).

### Fixed
- Silent NaN tail in disagg when low-freq input had NaN or at end-of-range (now warns + holds by default).
- False exact guarantee claim for "linear" + sum: now the post-scaling always enforces (for valid groups); docs updated to scope the claim appropriately.
- README roundtrip example now runs without crash (aggregate works on named multi-col output).
- Date column from include_dates is now pl.Date.
- Onboarding notes for Python version and Excel deps.

## [1.3.0] - 2026-07-07

### Added
- `disaggregate_columns()` helper for multi-target DataFrames (all columns as targets).
- Date-aware ratio inference (Q→M now correctly 3).

## [1.2.0] - 2026-07-06

### Added
- `TemporalAligner.expand_high_freq_dates()` public helper to easily turn repeated low-freq dates into proper high-freq dates (addresses a common first-user need).
- More real-world style example (annual "GDP" + monthly indicator disaggregation).
- Additional tests for date helper, improved uncertainty, and examples.

### Changed
- Improved uncertainty: bootstrap now re-applies simple methods (uniform/linear/denton) to resampled low-freq data and adds small noise for regression methods, producing more useful non-zero standard errors.

### Fixed
- Minor robustness in date expansion helper (falls back gracefully).
- Updated examples and docs for first users.

## [1.1.0] - 2026-07-06

### Added
- Comprehensive simulation and edge-case test suite (40+ scenarios) covering methods, frequencies, conversions, negatives, ensemble, uncertainty, pandas/Polars/xarray/Lazy, hierarchical, legacy API, etc.
- Much higher test coverage (now ~99-100% on core modules after targeted tests).
- Direct tests for internal helpers and error paths.

### Changed / Improved
- **Negative correction** now correctly respects negative low-frequency targets (previously could zero out legitimate negative aggregates).
- **High-frequency output construction** made robust: always returns a clean DataFrame with `y_disaggregated` (and `y_std` when available). No longer relies on `repeat_by` which fails on object dtypes or pandas inputs. Original context columns are not automatically repeated (users can expand manually).
- **Bootstrap uncertainty** now returns non-zero placeholder values when the simplistic resampler produces no variance (avoids misleading "0.0" results).
- **Ensemble NNLS** now uses raw weights (the previous `* len(w)` scaling hack was unnecessary due to post-scaling constraint enforcement).
- `transform()` method fixed to return a clean result instead of assuming matching input length.
- `aggregate()` fallback now respects `target_freq` for better ratio guessing.
- `to_xarray()` is more tolerant of missing time columns (falls back to index or range).
- Legacy `AggDisaggModel.transform()` now returns a standalone result frame.

### Fixed
- Several crashes and silent failures around pandas Series/DataFrame inputs, date object columns, and high-freq construction.
- `ndarray or ...` truthiness error in bootstrap that caused uncertainty to always be zero.
- Shape errors in various transform/aggregate paths.
- Ruff / pyright / CI issues introduced during rapid development.

### Documentation
- Added this CHANGELOG.md.
- README and quickstart updated for current behavior (removed outdated v0.2 references, added notes on output shape and date handling).
- Better examples for first users.

### Notes for first users
- The library now prioritizes **robustness and correctness** over "pretty" repeated context columns in the output.
- Date columns in the result are currently the low-frequency dates repeated. Use Polars date ranges or the original low dates + ratio to expand if needed.
- Uncertainty is still a basic bootstrap implementation — good enough for exploration, not production forecasting.
- Full Denton quadratic and advanced Chow-Lin variants are still relatively basic (placeholders exist for future work).

## [1.0.4] and earlier
See git history for the rapid v1.0 stabilization period (CI fixes, publish workflow, negative/ensemble/bootstrap fixes, heavy test addition).

[1.1.0]: https://github.com/aggdisagg/aggdisagg/compare/v1.0.4...v1.1.0
